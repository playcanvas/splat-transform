import { join } from 'pathe';
import { type GraphicsDevice } from 'playcanvas';

import { createBlockProducerSource } from './block-producer';
import { mergeStream } from './merge-stream';
import { kdPartition, coherenceRuns, type ResidentPositions } from './partition';
import { runPriorityPass, HALO_CAP, type CandidateArrays } from './priority';
import { selectMerges } from './select';
import {
    compact,
    type ChunkDataPool,
    type ChunkSource,
    type ChunkSourceMetadata
} from '../chunk';
import { APP_CHUNK } from '../gpu/gpu-edge-cost';
import { type ReadFileSystem } from '../io/read';
import { type FileSystem } from '../io/write';
import { bakeTransform } from '../ops';
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

/** Default resident-memory budget steering the candidate-K policy. */
const DEFAULT_MEMORY_BUDGET = 24 * 2 ** 30;

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
    /** Resident-memory budget driving the candidate-K policy (default 24 GiB). */
    memoryBudgetBytes?: number;
};

// Candidate-K policy: keep 4 when the resident estimate fits the budget,
// else 2. Estimate: positions (12) + candidates (K*8) + selection counting
// sort (K*4) + memberGroup (4) per gaussian, plus a flat block-working fudge.
const chooseK = (n: number, budget: number): number => {
    const estimate = (K: number) => n * (12 + K * 8 + K * 4 + 4) + 3 * 2 ** 30;
    return estimate(4) <= budget ? 4 : 2;
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
 * Design: `docs/specs/2026-07-03-chunk-decimation-design.md`. Positions
 * resident; KD blocks as an IO pattern only; per-block exact global 16-NN +
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
            `decimate requires a single-LOD source (got ${inputMeta.numLods} LODs); select a level first (--lod-select / selectLod)`
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

    const totalGenerations = Math.max(1, Math.ceil(Math.log2(inputMeta.numGaussians / targetCount)));

    for (let generation = 1; ; generation++) {
        const N = src.meta.numGaussians;
        const gen = logger.group('Decimate generation', {
            index: Math.min(generation, totalGenerations),
            total: totalGenerations
        });

        positions ??= await extractPositions(src, pool);

        // Device binding-limit clamp: the largest per-binding buffer scales
        // with block size, never scene size; halve the block size until it
        // fits the adapter's storage-binding limit.
        let blockSize = BLOCK_SIZE;
        const bindingLimit = (device as unknown as { limits?: { maxStorageBufferBindingSize?: number } } | undefined)
        ?.limits?.maxStorageBufferBindingSize;
        if (typeof bindingLimit === 'number') {
            const largestBinding = (bs: number) => bs * (1 + HALO_CAP) * Math.max(Math.min(APP_CHUNK, colorDim) * 4, 36);
            while (blockSize > (1 << 16) && largestBinding(blockSize) > bindingLimit) {
                blockSize >>= 1;
            }
            if (blockSize !== BLOCK_SIZE) {
                logger.warn(`reducing decimate block size to ${fmtCount(blockSize)} to fit GPU binding limit ${fmtBytes(bindingLimit)}`);
            }
        }

        const partSub = logger.group('Partitioning');
        const { order, blocks } = kdPartition(positions, blockSize);
        partSub.end();

        if (generation === 1 && N >= COHERENCE_MIN_N) {
            const runs = blocks.map(b => coherenceRuns(order, b.start, b.end, COHERENCE_GAP_ROWS)).sort((a, b) => a - b);
            const median = runs[runs.length >> 1] ?? 0;
            if (median > INCOHERENT_RUNS_PER_BLOCK) {
                logger.warn(
                    'input is spatially incoherent (scattered gathers expected); run a one-time --morton-order prepass for much faster IO'
                );
            }
        }

        const K = chooseK(N, budget);
        const cand: CandidateArrays = {
            idx: new Uint32Array(N * K).fill(0xFFFFFFFF),
            cost: new Float32Array(N * K).fill(Infinity)
        };

        const priorityBar = logger.bar('computing merge priorities', N);
        await runPriorityPass(
            { source: src, pool, pos: positions, order, blocks, device, K, k: Math.min(KNN_K, Math.max(1, N - 1)) },
            cand,
            n => priorityBar.tick(n)
        );
        priorityBar.end();

        const generationTarget = Math.max(targetCount, N - Math.floor(N / 2));
        const needed = N - generationTarget;
        const selectSub = logger.group('Selecting merges');
        const selection = selectMerges(cand, N, K, needed);
        selectSub.end();

        if (selection.removed === 0) {
            gen.end();
            const cause = device ?
                'the GPU step likely failed (e.g. out-of-memory) or produced non-finite costs' :
                'cost computation produced no finite merge candidates (e.g. non-finite inputs)';
            throw new Error(
                `decimation found no valid merges at ${N} splats (target ${targetCount}) — ${cause}. ` +
                'Refusing to return an incompletely-decimated scene.'
            );
        }
        const removedFraction = selection.removed / N;
        if (selection.removed < needed && removedFraction < MIN_ITERATION_PROGRESS) {
            gen.end();
            throw new Error(
                `decimation stalled at ${N} splats (target ${targetCount}): a generation removed only ` +
                `${selection.removed} splat${selection.removed === 1 ? '' : 's'} (${(removedFraction * 100).toFixed(3)}% of ${N}) — ` +
                'the nearest-neighbour graph is too degenerate to merge further (e.g. many coincident splats). ' +
                'Refusing to grind toward the target.'
            );
        }

        const outCount = N - selection.removed;
        const outMeta: ChunkSourceMetadata = {
            numGaussians: outCount,
            numLods: 1,
            lodCounts: [outCount],
            chunkSize: src.meta.chunkSize,
            numChunks: [Math.ceil(outCount / src.meta.chunkSize)],
            shBands: src.meta.shBands,
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
        const streamCtx = { source: genSrc, pool, pos: positions, order, blocks, selection, nextPositions };

        if (isFinal) {
            // The producer reads the input lazily while the consumer pulls
            // chunks: the input chain (and any pending spill) is released on
            // close. The merge bar lives outside the generation group since
            // streaming happens after this function returns.
            gen.end();
            const mergeBar = logger.bar('merging', N);
            const producer = createBlockProducerSource(outMeta, () => mergeStream(streamCtx, genChunkSize, n => mergeBar.tick(n)));
            const disposeSpill = disposeCurrentInput;
            let closed = false;
            return {
                meta: producer.meta,
                read: request => producer.read(request),
                close: async () => {
                    if (closed) return;
                    closed = true;
                    mergeBar.end();
                    await producer.close();
                    await genSrc.close();
                    await disposeSpill?.();
                }
            };
        }

        const mergeBar = logger.bar('merging', N);
        const producer = createBlockProducerSource(outMeta, () => mergeStream(streamCtx, genChunkSize, n => mergeBar.tick(n)));

        // Intermediate generation: materialize (RAM when comfortably within
        // budget, else temp PLY spill), then advance the loop.
        const estBytes = outCount * (12 + 32 + colorDim * 4 + otherStride);
        let nextSrc: ChunkSource;
        let disposeNext: (() => Promise<void>) | null = null;

        if (estBytes <= budget / 4) {
            nextSrc = await compact(producer, pool);
        } else {
            if (!opts.spill) {
                throw new Error(
                    `decimation intermediate generation needs ${fmtBytes(estBytes)}, over the in-memory budget — ` +
                    'a spill location is required (opts.spill / --scratch-dir)'
                );
            }
            const spill = opts.spill;
            const filename = join(spill.scratchDir, `.decimate-gen${generation}.${Date.now().toString(36)}.tmp.ply`);
            await writePlyStreaming(producer, pool, { filename }, spill.writeFs);
            const readSource = await spill.readFs.createSource(filename);
            const plySrc = await readPly(readSource, pool);
            nextSrc = plySrc;
            disposeNext = async () => {
                await plySrc.close();
                await spill.remove?.(filename);
            };
        }
        mergeBar.end();
        await producer.close();

        // The consumed input of THIS generation can now be released: for
        // generation 1 that is the caller's source (we own it), for later
        // generations the previous spill / RAM intermediate.
        await src.close();
        await disposeCurrentInput?.();
        disposeCurrentInput = disposeNext;

        positions = nextPositions!;
        src = nextSrc;
        gen.end();
    }
};

export { decimateSource, type DecimateOptions, type DecimateSpill };
