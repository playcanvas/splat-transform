import { join } from 'pathe';
import { type GraphicsDevice } from 'playcanvas';

import {
    allocatePlanPrefixes,
    storeBlockPlan,
    type StoredBlockPlan
} from './block-allocation';
import { blockPlanMergeStream } from './block-merge-stream';
import { planBlockMerges } from './block-plan';
import { prepareGpuBlock, type PreparedBlock } from './block-prepare';
import { createBlockProducerSource, type DestBuffers } from './block-producer';
import { mergeStream } from './merge-stream';
import { buildBlockHalo, kdPartition, coherenceRuns, type ResidentPositions } from './partition';
import { runPriorityPass, type CandidateArrays } from './priority';
import { selectMerges, type SelectionResult } from './select';
import { selectMergesRecosted, CACHE_STRIDE } from './select-recost';
import {
    compact,
    type ChunkDataPool,
    type ChunkSource,
    type ChunkSourceMetadata
} from '../chunk';
import { type ReadFileSystem } from '../io/read';
import { type FileSystem } from '../io/write';
import { bakeTransform, permuteSource } from '../ops';
import { readPly } from '../readers/read-ply';
import { type DeviceCreator } from '../types';
import { fmtBytes, fmtCount, logger, Transform } from '../utils';
import { writePlyStreaming } from '../writers/write-ply-streaming';

/** Neighbours per query — unchanged from legacy. */
const KNN_K = 16;

/** Default owned gaussians per KD block. */
const BLOCK_SIZE = 1 << 21;

/** Same no-grind stall semantics as legacy: a shortfall generation must remove at least this fraction. */
const MIN_ITERATION_PROGRESS = 0.05;

/** Default resident-memory budget steering the candidate-K and re-costed-selection policies. */
const DEFAULT_MEMORY_BUDGET = 48 * 2 ** 30;

/** Conservative host bytes per core row for two overlapping core+halo views. */
const MULTI_BLOCK_BYTES_PER_CORE = 1024;

// Per-gaussian residency of re-costed selection beyond the base state: splat
// cache (16 f32) + neighbour ids (k u32) + integer structure (union-find,
// chains, seq/round bookkeeping: 8×u32) + heap (5 arrays × 1.25N ≈ 25 B).
// Conservative round-up; used by the per-generation gate that falls back to
// one-shot selectMerges when over budget.
const RECOST_BYTES_PER_GAUSSIAN = (k: number) => CACHE_STRIDE * 4 + k * 4 + 80;

/** Coherence heuristic: gap (rows) merged into one run / runs-per-block considered scattered. */
const COHERENCE_GAP_ROWS = 64;
const INCOHERENT_RUNS_PER_BLOCK = 64;

/** Warn about incoherent input only for scenes big enough for it to matter. */
const COHERENCE_MIN_N = 1 << 22;

/**
 * Where intermediate generations spill when they exceed the in-memory
 * budget. `remove` deletes a spill file once its generation is consumed
 * (optional; without it temp files are left behind).
 */
type DecimateSpill = {
    writeFs: FileSystem;
    readFs: ReadFileSystem;
    scratchDir: string;
    remove?: (path: string) => Promise<void>;
};

type DecimateOptions = {
    /** Exact number of gaussians to keep (≥ 1). */
    targetCount: number;
    /** Optional GPU device factory; CPU fallback without it. */
    createDevice?: DeviceCreator;
    /** Spill destination for over-budget intermediate generations. */
    spill?: DecimateSpill;
    /** Resident-memory budget driving the candidate-K and re-costed-selection policies (default 48 GiB). */
    memoryBudgetBytes?: number;
};

// Candidate-K policy: keep 4 when the resident estimate fits the budget,
// else 2. Estimate: positions (12) + candidates (K*8) + selection counting
// sort (K*4) + memberGroup (4) per gaussian, plus a flat block-working fudge.
const chooseK = (n: number, budget: number): number => {
    const estimate = (K: number) => n * (12 + K * 8 + K * 4 + 4) + 3 * 2 ** 30;
    return estimate(4) <= budget ? 4 : 2;
};

const chooseBlockSize = (
    n: number,
    budget: number,
    residentInputBytes: number,
    outputPositionBytes: number,
    device?: GraphicsDevice
): number => {
    const residentIndex = n * 16;
    const available = Math.max(0, budget - residentInputBytes - residentIndex - outputPositionBytes);
    let blockSize = Math.min(BLOCK_SIZE, Math.max(1 << 16, Math.floor(available / MULTI_BLOCK_BYTES_PER_CORE)));
    const limits = (device as unknown as {
        limits?: {
            maxStorageBufferBindingSize?: number;
            maxBufferSize?: number;
        };
    } | undefined)?.limits;
    const bindingLimit = Math.min(
        limits?.maxStorageBufferBindingSize ?? Infinity,
        limits?.maxBufferSize ?? Infinity
    );
    if (Number.isFinite(bindingLimit)) {
        // The largest local binding is one half of the core+halo cache:
        // approximately coreCount × CACHE_STRIDE × sizeof(f32).
        blockSize = Math.min(blockSize, Math.floor(bindingLimit / (CACHE_STRIDE * 4)));
    }
    return Math.max(1, Math.min(blockSize, n));
};

// Read the position layer sequentially into resident columns (generation 1
// only; later generations carry positions forward from the merge stream).
const extractPositions = async (source: ChunkSource, pool: ChunkDataPool): Promise<ResidentPositions> => {
    const { meta } = source;
    const n = meta.numGaussians;
    const out: ResidentPositions = {
        x: new Float32Array(n),
        y: new Float32Array(n),
        z: new Float32Array(n)
    };
    const bar = logger.bar('reading positions', meta.numChunks[0] ?? 0);
    let base = 0;
    for (let c = 0; c < (meta.numChunks[0] ?? 0); c++) {
        const count = Math.min(meta.chunkSize, n - c * meta.chunkSize);
        const cd = pool.acquire('position', meta.layouts.position!, count);
        await source.read({ chunkIndex: c, position: cd });
        const p = new Float32Array(cd.data, 0, count * 3);
        for (let i = 0; i < count; i++) {
            out.x[base + i] = p[i * 3];
            out.y[base + i] = p[i * 3 + 1];
            out.z[base + i] = p[i * 3 + 2];
        }
        base += count;
        cd.release();
        bar.tick();
    }
    bar.end();
    return out;
};

/**
 * Chunk-native, memory-bounded decimation to an exact target count.
 *
 * Design: positions resident; KD blocks as an IO pattern only; per-block exact global 16-NN +
 * edge costs (GPU when a device is supplied) reduced to K resident
 * candidates; global bucketed greedy matching with chain closure; a second
 * heavy pass moment-matches groups and streams the output.
 *
 * The returned source supports a single sequential pass (it computes the
 * merge stream on demand) — the PLY-terminal consumption model. Its `close`
 * releases the input source and any intermediate spill files. Deep targets
 * run multiple generations; intermediates land in RAM when small enough,
 * else in temp PLY spills under `opts.spill.scratchDir`.
 *
 * @param source - Input (consumed: the returned source owns it). Single LOD, gaussian layers required.
 * @param pool - Chunk-data pool; its chunk size must match the source's.
 * @param opts - Options.
 * @returns The decimated stream-once source with exact metadata.
 */
const decimateSource = async (
    source: ChunkSource,
    pool: ChunkDataPool,
    opts: DecimateOptions
): Promise<ChunkSource> => {
    const { targetCount } = opts;
    const inputMeta = source.meta;

    if (inputMeta.numLods > 1) {
        throw new Error(
            `decimate requires a single-LOD source (got ${inputMeta.numLods} LODs); select a level first (--select-lod / selectLod)`
        );
    }
    for (const layer of ['position', 'geometric', 'color'] as const) {
        if (!inputMeta.availableLayers.has(layer)) {
            throw new Error(`decimate requires gaussian splat data (missing '${layer}' layer)`);
        }
    }
    if (targetCount < 1) {
        throw new Error(`decimate target must be at least 1 (got ${targetCount})`);
    }
    if (targetCount >= inputMeta.numGaussians) {
        return source;
    }

    const device: GraphicsDevice | undefined = opts.createDevice ? await opts.createDevice() : undefined;
    const budget = opts.memoryBudgetBytes ?? DEFAULT_MEMORY_BUDGET;
    const colorDim = inputMeta.layouts.color!.stride >> 2;
    const otherStride = inputMeta.layouts.other?.stride ?? 0;

    // Bake the pending transform to PLY space up front (identity fast-path):
    // intermediate spills go through writePlyStreaming (which bakes anyway)
    // and carried-forward resident positions must match spilled values.
    // Decimation is TRS-covariant, so bake timing cannot change the result.
    let src: ChunkSource = bakeTransform(source, Transform.PLY);
    let positions: ResidentPositions | null = null;
    // Cleanup for the CURRENT generation's input (previous spill / RAM source).
    let disposeCurrentInput: (() => Promise<void>) | null = null;
    // Resident footprint of the current generation's input (non-zero when the
    // previous generation was materialized in RAM) — counted by the re-costed
    // selection gate so the budget covers everything actually resident.
    let residentInputBytes = 0;
    let boundaryRetries = 0;

    const totalGenerations = Math.max(1, Math.ceil(Math.log2(inputMeta.numGaussians / targetCount)));

    try {
        for (let generation = 1; ; generation++) {
            const N = src.meta.numGaussians;
            const gen = logger.group('Decimate generation', {
                index: Math.min(generation, totalGenerations),
                total: totalGenerations
            });

            positions ??= await extractPositions(src, pool);

            // Multi-block working sets are sized from the actual host budget
            // and adapter binding limits.
            const nextCount = Math.max(targetCount, N - Math.floor(N / 2));
            const blockSize = chooseBlockSize(N, budget, residentInputBytes, nextCount * 12, device);
            if (blockSize !== BLOCK_SIZE && N > blockSize) {
                logger.info(`decimate core size ${fmtCount(blockSize)} (memory/device working-set limit)`);
            }

            const partSub = logger.group('Partitioning');
            let partition = kdPartition(positions, blockSize, generation);
            let { order, blocks } = partition;
            partSub.end();

            if (generation === 1 && N >= COHERENCE_MIN_N) {
                const runs = blocks.map(b => coherenceRuns(order, b.start, b.end, COHERENCE_GAP_ROWS)).sort((a, b) => a - b);
                const median = runs[runs.length >> 1] ?? 0;
                if (median > INCOHERENT_RUNS_PER_BLOCK) {
                    if (blocks.length > 1) {
                        if (!opts.spill) {
                            throw new Error(
                                'multi-block adaptive decimation needs scratch storage to stage spatially incoherent input; ' +
                            'provide opts.spill / --scratch-dir'
                            );
                        }
                        const rowBytes = 12 + 32 + colorDim * 4 + otherStride;
                        logger.info(
                            'spatially incoherent input: staging one KD-ordered PLY ' +
                        `(estimated ${fmtBytes(N * rowBytes)})`
                        );
                        const spill = opts.spill;
                        const filename = join(
                            spill.scratchDir,
                            `.decimate-stage-${Date.now().toString(36)}.tmp.ply`
                        );
                        const stagedView = permuteSource(src, order);
                        let plySrc: ChunkSource;
                        try {
                            await writePlyStreaming(stagedView, pool, { filename }, spill.writeFs);
                            const readSource = await spill.readFs.createSource(filename);
                            plySrc = await readPly(readSource, pool);
                        } catch (err) {
                            try {
                                await spill.remove?.(filename);
                            } catch {
                            // Preserve the staging failure.
                            }
                            throw err;
                        }

                        const reordered: ResidentPositions = {
                            x: new Float32Array(N),
                            y: new Float32Array(N),
                            z: new Float32Array(N)
                        };
                        for (let i = 0; i < N; i++) {
                            const g = order[i];
                            reordered.x[i] = positions.x[g];
                            reordered.y[i] = positions.y[g];
                            reordered.z[i] = positions.z[g];
                        }

                        await src.close();
                        await disposeCurrentInput?.();
                        src = plySrc;
                        positions = reordered;
                        disposeCurrentInput = async () => {
                            await spill.remove?.(filename);
                        };
                        partition = kdPartition(positions, blockSize, generation);
                        ({ order, blocks } = partition);
                    } else {
                        logger.warn(
                            'input is spatially incoherent (scattered gathers expected); run a one-time --morton-order prepass for much faster IO'
                        );
                    }
                }
            }

            // Re-costed selection (exact within-generation greedy) when its
            // resident state fits the budget alongside the base state; one-shot
            // selection otherwise. Gated per generation, so large scenes regain
            // re-costing as soon as the cascade shrinks under the budget.
            const K = chooseK(N, budget);
            const k = Math.min(KNN_K, Math.max(1, N - 1));
            const generationTarget = Math.max(targetCount, N - Math.floor(N / 2));
            const needed = N - generationTarget;
            const multiBlock = blocks.length > 1;
            let selection: SelectionResult | undefined;
            let storedPlans: StoredBlockPlan[] | undefined;
            let planPrefixes: Uint32Array | undefined;
            let removed: number;

            if (multiBlock) {
                if (!device) {
                    throw new Error(
                        `multi-block adaptive decimation requires WebGPU (${fmtCount(N)} splats, ` +
                    `${fmtCount(blockSize)}-splat cores); provide a device, or use --decimate`
                    );
                }
                if (!opts.spill) {
                    throw new Error(
                        'multi-block adaptive decimation needs scratch storage for merge plans ' +
                    `(approximately ${fmtBytes(N * 12)} this generation); provide opts.spill / --scratch-dir`
                    );
                }
                if (generation === 1) {
                    const rowBytes = 12 + 32 + colorDim * 4 + otherStride;
                    logger.info(
                        `decimate scratch estimate: staging up to ${fmtBytes(N * rowBytes)}; ` +
                    `merge plans up to ${fmtBytes(N * 12)} per generation`
                    );
                }

                const planBar = logger.bar('planning local merges', N);
                storedPlans = new Array(blocks.length);
                let cappedHalos = 0;
                let frozen = 0;
                let unfrozen = 0;
                let planned = 0;
                let waves = 0;
                let refreshes = 0;
                let reverseInvalidations = 0;
                let heapPops = 0;
                let staleHeapPops = 0;
                let knnMs = 0;
                let refreshMs = 0;
                let allocationMs = 0;
                let preparedNext: Promise<PreparedBlock> | null = null;

                // eslint-disable-next-line no-loop-func
                const prepare = (bi: number): Promise<PreparedBlock> => {
                    const coreCount = blocks[bi].end - blocks[bi].start;
                    const halo = buildBlockHalo(positions!, partition, bi, coreCount);
                    if (halo.capped) cappedHalos++;
                    const started = Date.now();
                    return prepareGpuBlock(
                        { source: src, pool, pos: positions!, order, blocks },
                        bi,
                        halo.rows,
                        device,
                        KNN_K
                    ).finally(() => {
                        knnMs += Date.now() - started;
                    });
                };

                try {
                    preparedNext = prepare(0);
                    for (let bi = 0; bi < blocks.length; bi++) {
                        const prepared = await preparedNext;
                        preparedNext = bi + 1 < blocks.length ? prepare(bi + 1) : null;
                        const refreshStarted = Date.now();
                        const plan = await planBlockMerges({
                            splatCache: prepared.cache,
                            neighbors: prepared.neighbors,
                            D: KNN_K,
                            coreCount: prepared.ownedCount,
                            totalCount: prepared.view.pos.length / 3,
                            device,
                            requireGpu: true
                        });
                        refreshMs += Date.now() - refreshStarted;
                        frozen += plan.frozen;
                        unfrozen += plan.unfrozen;
                        planned += plan.costs.length;
                        waves += plan.diagnostics!.waves;
                        refreshes += plan.diagnostics!.refreshes;
                        reverseInvalidations += plan.diagnostics!.reverseInvalidations;
                        heapPops += plan.diagnostics!.heapPops;
                        staleHeapPops += plan.diagnostics!.staleHeapPops;
                        storedPlans[bi] = await storeBlockPlan(opts.spill, generation, bi, plan);
                        planBar.tick(prepared.ownedCount);
                    }
                    const allocationStarted = Date.now();
                    const allocation = await allocatePlanPrefixes(storedPlans, opts.spill, needed);
                    allocationMs = Date.now() - allocationStarted;
                    planPrefixes = allocation.prefixes;
                    removed = allocation.removed;
                } catch (err) {
                    if (preparedNext) {
                        try {
                            await preparedNext;
                        } catch {
                        // Preserve the active planning failure.
                        }
                    }
                    try {
                        await Promise.all(storedPlans.filter(Boolean).map(plan => opts.spill!.remove?.(plan.path)));
                    } catch {
                        // Preserve the planning failure.
                    }
                    throw err;
                } finally {
                    planBar.end();
                }
                logger.info(
                    `local merge stats: ${fmtCount(cappedHalos)} capped halo${cappedHalos === 1 ? '' : 's'}, ` +
                `${fmtCount(frozen)} freezes, ${fmtCount(unfrozen)} unfreezes, ${fmtCount(removed!)} removals`
                );
                logger.info(
                    `local planner work: ${fmtCount(planned)} planned, ${fmtCount(waves)} waves, ` +
                `${fmtCount(refreshes)} refreshes, ${fmtCount(reverseInvalidations)} reverse invalidations, ` +
                `${fmtCount(heapPops)} heap pops (${fmtCount(staleHeapPops)} stale)`
                );
                logger.info(
                    `local timings: KNN/gather ${(knnMs / 1000).toFixed(2)}s, ` +
                `refresh/plan ${(refreshMs / 1000).toFixed(2)}s, allocation ${(allocationMs / 1000).toFixed(2)}s`
                );
            } else {
                const baseBytes = residentInputBytes + N * (12 + K * 8 + K * 4 + 4) + 3 * 2 ** 30;
                const recost = baseBytes + N * RECOST_BYTES_PER_GAUSSIAN(k) <= budget;

                // One block deliberately follows the pre-existing path with no
                // staging, halos, plan files, or k-way coordination.
                const cand: CandidateArrays | undefined = recost ?
                    undefined :
                    {
                        idx: new Uint32Array(N * K).fill(0xFFFFFFFF),
                        cost: new Float32Array(N * K).fill(Infinity)
                    };
                const cacheOut = recost ? new Float32Array(N * CACHE_STRIDE) : undefined;
                const neighborsOut = recost ? new Uint32Array(N * k) : undefined;
                const priorityBar = logger.bar('computing merge priorities', N);
                await runPriorityPass(
                    { source: src, pool, pos: positions, order, blocks, device, K, k, cacheOut, neighborsOut },
                    cand,
                    n => priorityBar.tick(n)
                );
                priorityBar.end();

                const selectSub = logger.group(recost ? 'Selecting merges (re-costed)' : 'Selecting merges');
                selection = cacheOut ?
                    await selectMergesRecosted({ splatCache: cacheOut, neighbors: neighborsOut!, D: k, N, mergesNeeded: needed, device }) :
                    selectMerges(cand!, N, K, needed);
                selectSub.end();
                removed = selection.removed;
            }

            let plansDisposed = false;
            const disposePlans = async (): Promise<void> => {
                if (plansDisposed || !storedPlans) return;
                plansDisposed = true;
                await Promise.all(storedPlans.map(plan => opts.spill?.remove?.(plan.path)));
            };

            if (removed === 0) {
                await disposePlans();
                if (multiBlock && boundaryRetries < 8) {
                    boundaryRetries++;
                    logger.warn(
                        `no productive local cores at jitter ${generation}; repartitioning unchanged input ` +
                    `(${boundaryRetries}/8 boundary retries)`
                    );
                    gen.end();
                    continue;
                }
                gen.end();
                const cause = device ?
                    'the GPU step likely failed (e.g. out-of-memory) or produced non-finite costs' :
                    'cost computation produced no finite merge candidates (e.g. non-finite inputs)';
                throw new Error(
                    `decimation found no valid merges at ${N} splats (target ${targetCount}) — ${cause}. ` +
                'Refusing to return an incompletely-decimated scene.'
                );
            }
            boundaryRetries = 0;
            const removedFraction = removed / N;
            if (removed < needed && removedFraction < MIN_ITERATION_PROGRESS) {
                await disposePlans();
                gen.end();
                throw new Error(
                    `decimation stalled at ${N} splats (target ${targetCount}): a generation removed only ` +
                `${removed} splat${removed === 1 ? '' : 's'} (${(removedFraction * 100).toFixed(3)}% of ${N}) — ` +
                'the nearest-neighbour graph is too degenerate to merge further (e.g. many coincident splats). ' +
                'Refusing to grind toward the target.'
                );
            }

            const outCount = N - removed;
            const outMeta: ChunkSourceMetadata = {
                numGaussians: outCount,
                numLods: 1,
                lodCounts: [outCount],
                chunkSize: src.meta.chunkSize,
                numChunks: [Math.ceil(outCount / src.meta.chunkSize)],
                shBands: src.meta.shBands,
                model: src.meta.model,
                extraColumns: src.meta.extraColumns,
                transform: src.meta.transform,
                availableLayers: src.meta.availableLayers,
                layouts: src.meta.layouts
            };

            const isFinal = outCount <= targetCount;
            const nextPositions: ResidentPositions | undefined = isFinal ? undefined : {
                x: new Float32Array(outCount),
                y: new Float32Array(outCount),
                z: new Float32Array(outCount)
            };

            // `src` is reassigned each generation; capture this generation's
            // values for the deferred producer closures.
            const genSrc = src;
            const genChunkSize = genSrc.meta.chunkSize;
            const genPositions = positions;
            const createStream = (
                tick: (n: number) => void
            ): AsyncGenerator<number, void, DestBuffers> => {
                if (storedPlans) {
                    return blockPlanMergeStream({
                        source: genSrc,
                        pool,
                        pos: genPositions,
                        order,
                        blocks,
                        plans: storedPlans,
                        prefixes: planPrefixes!,
                        scratch: opts.spill!,
                        nextPositions
                    }, genChunkSize, tick);
                }
                return mergeStream({
                    source: genSrc,
                    pool,
                    pos: genPositions,
                    order,
                    blocks,
                    selection: selection!,
                    nextPositions
                }, genChunkSize, tick);
            };

            if (isFinal) {
            // The producer reads the input lazily while the consumer pulls
            // chunks: the input chain (and any pending spill) is released on
            // close. The merge bar lives outside the generation group since
            // streaming happens after this function returns.
                gen.end();
                const mergeBar = logger.bar('merging', N);
                const producer = createBlockProducerSource(outMeta, () => createStream(n => mergeBar.tick(n)));
                const disposeSpill = disposeCurrentInput;
                let closed = false;
                return {
                    meta: producer.meta,
                    read: request => producer.read(request),
                    close: async () => {
                        if (closed) return;
                        closed = true;
                        mergeBar.end();
                        try {
                            await producer.close();
                        } finally {
                            try {
                                await genSrc.close();
                            } finally {
                                try {
                                    await disposePlans();
                                } finally {
                                    await disposeSpill?.();
                                }
                            }
                        }
                    }
                };
            }

            const mergeBar = logger.bar('merging', N);
            const producer = createBlockProducerSource(outMeta, () => createStream(n => mergeBar.tick(n)));

            // Intermediate generation: materialize (RAM when comfortably within
            // budget, else temp PLY spill), then advance the loop.
            const estBytes = outCount * (12 + 32 + colorDim * 4 + otherStride);
            let nextSrc: ChunkSource;
            let disposeNext: (() => Promise<void>) | null = null;

            try {
                if (estBytes <= budget / 4) {
                    nextSrc = await compact(producer, pool);
                    residentInputBytes = estBytes;
                } else {
                    residentInputBytes = 0;
                    if (!opts.spill) {
                        throw new Error(
                            `decimation intermediate generation needs ${fmtBytes(estBytes)}, over the in-memory budget — ` +
                        'a spill location is required (opts.spill / --scratch-dir)'
                        );
                    }
                    const spill = opts.spill;
                    const filename = join(spill.scratchDir, `.decimate-gen${generation}.${Date.now().toString(36)}.tmp.ply`);
                    let plySrc: ChunkSource;
                    try {
                        await writePlyStreaming(producer, pool, { filename }, spill.writeFs);
                        const readSource = await spill.readFs.createSource(filename);
                        plySrc = await readPly(readSource, pool);
                    } catch (err) {
                        try {
                            await spill.remove?.(filename);
                        } catch {
                            // Preserve the generation failure.
                        }
                        throw err;
                    }
                    nextSrc = plySrc;
                    disposeNext = async () => {
                        await spill.remove?.(filename);
                    };
                }
            } finally {
                await disposePlans();
            }
            try {
                mergeBar.end();
                await producer.close();

                // The consumed input of THIS generation can now be released:
                // for generation 1 that is the caller's source (we own it),
                // for later generations the previous spill / RAM intermediate.
                await src.close();
                await disposeCurrentInput?.();
            } catch (err) {
                try {
                    await nextSrc.close();
                } catch {
                    // Preserve the transition failure.
                }
                try {
                    await disposeNext?.();
                } catch {
                    // Preserve the transition failure.
                }
                throw err;
            }
            disposeCurrentInput = disposeNext;

            positions = nextPositions!;
            src = nextSrc;
            gen.end();
        }
    } catch (err) {
        // Construction failures own the current input just like a returned
        // decimation source does. Preserve the active error while making a
        // best effort to remove any staged/intermediate generation.
        try {
            await src.close();
        } catch {
            // Preserve the construction failure.
        }
        try {
            await disposeCurrentInput?.();
        } catch {
            // Preserve the construction failure.
        }
        throw err;
    }
};

export { decimateSource, type DecimateOptions, type DecimateSpill };
