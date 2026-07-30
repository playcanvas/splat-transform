import { type GraphicsDevice } from 'playcanvas';

import { buildSplatCache, computeEdgeCost, CACHE_STRIDE } from './edge-cost-cpu';
import { KNN_SENTINEL, type ForestPart } from './knn-core';
import { type SplatView } from './moment-match';
import { type BlockRange, type ResidentPositions } from './partition';
import { type ChunkData, type ChunkDataPool, type ChunkSource } from '../chunk';
import { GpuEdgeCost } from '../gpu/gpu-edge-cost';
import { GpuKnn } from '../gpu/gpu-knn';
import { WorkerQueue } from '../workers';

/**
 * Max points per forest part. Parts build in parallel on the worker pool
 * (the build is each generation's serial prefix — smaller parts spread it
 * across the pool) and their trees must fit the GPU binding limits; query
 * cost is ~part-count-independent (the carried bound prunes distant parts
 * at their root).
 */
const PART_SIZE_MAX = 1 << 22;

/** Initial per-block view capacity multiplier (externals beyond it grow the cost buffers on demand). */
const VIEW_GROW = 1.25;

/**
 * Per-gaussian best-K merge candidates, the resident output of the priority
 * pass. `idx[g * K + s]` is the global index of gaussian g's s-th cheapest
 * candidate (0xFFFFFFFF when absent); `cost[g * K + s]` its cost (+Inf when
 * absent).
 */
type CandidateArrays = {
    idx: Uint32Array;
    cost: Float32Array;
};

/**
 * Colour components the pass gathers per row. The field-L2 cost reads only
 * DC, so at SH band 3 this is 16× less colour RAM and copy traffic per block
 * than gathering every band. (`--decimate-uniform` scores full SH and has its
 * own pass — see lib/decimate-uniform/priority.ts.)
 */
const COLOR_COMPONENTS = 3;

/** Everything the block passes need: baked single-LOD source + resident state. */
type PriorityContext = {
    source: ChunkSource;
    pool: ChunkDataPool;
    pos: ResidentPositions;
    order: Uint32Array;
    blocks: BlockRange[];
    device?: GraphicsDevice;
    /** Candidates kept per gaussian (K). */
    K: number;
    /** Neighbours per query (16). */
    k: number;
    /**
     * Optional resident splat-cache output for re-costed selection
     * (CACHE_STRIDE floats per gaussian, buildSplatCache row layout), filled
     * for every owned gaussian.
     */
    cacheOut?: Float32Array;
    /** Optional resident neighbour-id output (k per gaussian, KNN_SENTINEL padded). */
    neighborsOut?: Uint32Array;
};

/**
 * A block's gathered splat columns: owned rows first (block order), then the
 * requested extra globals. Positions come from the resident arrays, never
 * from the source.
 */
type BlockView = {
    view: SplatView;
    /** u32 `other` columns (extraDim per row), when requested and present. */
    other?: Uint32Array;
    otherDim: number;
    ownedCount: number;
};

/**
 * Gather geometric + color (and optionally `other`) for a block's owned rows
 * plus `extraGlobals`, into tight column arrays. Reads are batched at the
 * pool's chunk size; owned and extra index lists must be sorted ascending
 * for gather coalescing.
 *
 * @param ctx - The pass context.
 * @param blockIdx - Which block.
 * @param extraGlobals - Sorted out-of-block rows to append after the owned rows.
 * @param includeOther - Also gather the `other` layer (merge pass only).
 * @param colorComponents - Colour components to copy per row (default: all).
 * The adaptive cost reads only DC, so its pass gathers 3 — at SH band 3 that
 * is 16× less colour RAM and copy traffic per block. The read itself still
 * decodes whole rows (row-interleaved sources); only the view copy narrows.
 * @returns The gathered block view.
 */
const gatherBlockView = async (
    ctx: Pick<PriorityContext, 'source' | 'pool' | 'pos' | 'order' | 'blocks'>,
    blockIdx: number,
    extraGlobals: Uint32Array,
    includeOther = false,
    colorComponents?: number
): Promise<BlockView> => {
    const { source, pool, pos, order, blocks } = ctx;
    const block = blocks[blockIdx];
    const owned = order.subarray(block.start, block.end);
    const nOwned = owned.length;
    const n = nOwned + extraGlobals.length;
    const { layouts, availableLayers } = source.meta;

    const colorDim = layouts.color!.stride >> 2;
    const viewColorDim = Math.min(colorComponents ?? colorDim, colorDim);
    const wantOther = includeOther && availableLayers.has('other') && (layouts.other?.stride ?? 0) > 0;
    const otherDim = wantOther ? layouts.other!.stride >> 2 : 0;

    const view: SplatView = {
        pos: new Float32Array(n * 3),
        geo: new Float32Array(n * 8),
        color: new Float32Array(n * viewColorDim),
        colorDim: viewColorDim
    };
    const other = wantOther ? new Uint32Array(n * otherDim) : undefined;

    const readInto = async (indices: Uint32Array, rowBase: number): Promise<void> => {
        const batch = pool.chunkSize;
        for (let off = 0; off < indices.length; off += batch) {
            const count = Math.min(batch, indices.length - off);
            const geoCd = pool.acquire('geometric', layouts.geometric!, count);
            const colCd = pool.acquire('color', layouts.color!, count);
            const othCd: ChunkData | undefined = wantOther ? pool.acquire('other', layouts.other!, count) : undefined;
            await source.read({
                indices,
                indexOffset: off,
                count,
                geometric: geoCd,
                color: colCd,
                other: othCd
            });
            view.geo.set(new Float32Array(geoCd.data, 0, count * 8), (rowBase + off) * 8);
            const colSrc = new Float32Array(colCd.data, 0, count * colorDim);
            if (viewColorDim === colorDim) {
                view.color.set(colSrc, (rowBase + off) * colorDim);
            } else {
                for (let r = 0; r < count; r++) {
                    const src = r * colorDim, dst = (rowBase + off + r) * viewColorDim;
                    for (let c = 0; c < viewColorDim; c++) view.color[dst + c] = colSrc[src + c];
                }
            }
            if (othCd) other!.set(new Uint32Array(othCd.data, 0, count * otherDim), (rowBase + off) * otherDim);
            geoCd.release();
            colCd.release();
            othCd?.release();
        }
    };

    await readInto(owned, 0);
    await readInto(extraGlobals, nOwned);

    for (let i = 0; i < nOwned; i++) {
        const g = owned[i];
        view.pos[i * 3] = pos.x[g];
        view.pos[i * 3 + 1] = pos.y[g];
        view.pos[i * 3 + 2] = pos.z[g];
    }
    for (let i = 0; i < extraGlobals.length; i++) {
        const g = extraGlobals[i];
        const r = nOwned + i;
        view.pos[r * 3] = pos.x[g];
        view.pos[r * 3 + 1] = pos.y[g];
        view.pos[r * 3 + 2] = pos.z[g];
    }

    return { view, other, otherDim, ownedCount: nOwned };
};

// Binary search `g` in the sorted array; -1 when absent.
const indexOfSorted = (sorted: Uint32Array, g: number): number => {
    let lo = 0, hi = sorted.length - 1;
    while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        const v = sorted[mid];
        if (v === g) return mid;
        if (v < g) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
};

/**
 * Build the generation's KNN forest: consecutive kdPartition blocks grouped
 * to {@link PART_SIZE_MAX} points per part, each part's flat tree built on
 * the worker pool with node splat ids remapped to global. With `shared` the
 * arrays land on SharedArrayBuffers so the CPU query tasks read them without
 * copies.
 *
 * @param pos - Resident position columns.
 * @param order - Partition order (block-contiguous global ids).
 * @param blocks - Partition blocks.
 * @param shared - Allocate the flat arrays on shared memory.
 * @returns The forest parts and each block's home part index (queries
 * traverse their home part first so every other part culls on its AABB).
 */
const buildForest = async (
    pos: ResidentPositions,
    order: Uint32Array,
    blocks: BlockRange[],
    shared: boolean
): Promise<{ parts: ForestPart[], blockPart: Uint32Array }> => {
    const jobs: Promise<ForestPart>[] = [];
    const blockPart = new Uint32Array(blocks.length);
    let bi = 0;
    while (bi < blocks.length) {
        const startRow = blocks[bi].start;
        let endRow = blocks[bi].end;
        blockPart[bi] = jobs.length;
        bi++;
        while (bi < blocks.length && blocks[bi].end - startRow <= PART_SIZE_MAX) {
            endRow = blocks[bi].end;
            blockPart[bi] = jobs.length;
            bi++;
        }
        const cnt = endRow - startRow;
        const x = new Float32Array(cnt);
        const y = new Float32Array(cnt);
        const z = new Float32Array(cnt);
        const ids = new Uint32Array(cnt);
        for (let i = 0; i < cnt; i++) {
            const g = order[startRow + i];
            ids[i] = g;
            x[i] = pos.x[g];
            y[i] = pos.y[g];
            z[i] = pos.z[g];
        }
        jobs.push(WorkerQueue.run('buildKdForestPart', { x, y, z, ids, shared }, [
            x.buffer as ArrayBuffer, y.buffer as ArrayBuffer, z.buffer as ArrayBuffer, ids.buffer as ArrayBuffer
        ]));
    }
    return { parts: await Promise.all(jobs), blockPart };
};

/**
 * Canonically order each neighbour row by (distance², id) ascending with
 * sentinels last, so all downstream tie-breaking is deterministic and
 * identical across the GPU and CPU KNN paths.
 *
 * @param pos - Resident position columns.
 * @param owned - The block's owned global ids (row order).
 * @param nb - Neighbour rows (`owned.length * k` global ids), sorted in place.
 * @param k - Neighbours per row.
 */
const sortNeighborRows = (
    pos: ResidentPositions,
    owned: Uint32Array,
    nb: Uint32Array,
    k: number
): void => {
    const d = new Float64Array(k);
    const id = new Uint32Array(k);
    for (let q = 0; q < owned.length; q++) {
        const g = owned[q];
        const qx = pos.x[g], qy = pos.y[g], qz = pos.z[g];
        const base = q * k;
        let m = 0;
        for (let s = 0; s < k; s++) {
            const j = nb[base + s];
            if (j === KNN_SENTINEL) continue;
            const dx = pos.x[j] - qx, dy = pos.y[j] - qy, dz = pos.z[j] - qz;
            const dist = dx * dx + dy * dy + dz * dz;
            let at = m;
            while (at > 0 && (d[at - 1] > dist || (d[at - 1] === dist && id[at - 1] > j))) {
                d[at] = d[at - 1];
                id[at] = id[at - 1];
                at--;
            }
            d[at] = dist;
            id[at] = j;
            m++;
        }
        for (let s = 0; s < k; s++) nb[base + s] = s < m ? id[s] : KNN_SENTINEL;
    }
};

/**
 * The priority pass (heavy read 1): per block — exact global KNN, edge costs
 * for each owned gaussian's k neighbours, reduction to the best K candidates
 * — written into the resident candidate arrays.
 *
 * Without `cand` (re-costed selection seeds itself from the neighbour graph)
 * the pass only persists the splat cache + neighbour ids: no externals, no
 * edge costs, no top-K — the gather narrows to owned rows and the GPU cost
 * kernel is never constructed.
 *
 * @param ctx - The pass context.
 * @param cand - Candidate arrays (`N*K`) to fill, or undefined to skip all
 * cost work (cacheOut/neighborsOut persistence only).
 * @param tick - Optional progress callback (owned gaussians completed).
 */
const runPriorityPass = async (
    ctx: PriorityContext,
    cand: CandidateArrays | undefined,
    tick?: (n: number) => void
): Promise<void> => {
    const { pos, order, blocks, device, K, k } = ctx;

    let maxOwned = 0;
    for (const b of blocks) maxOwned = Math.max(maxOwned, b.end - b.start);

    let gpuKnn: GpuKnn | undefined;
    let gpuCost: GpuEdgeCost | undefined;
    let gpuCostCapacity = Math.ceil(maxOwned * VIEW_GROW);

    // The forest is built once per generation (its trees are exact and
    // global, so block boundaries are invisible to the results); blocks stay
    // the query/IO batching unit. Shared arrays feed the CPU query tasks.
    const { parts: forest, blockPart } = await buildForest(pos, order, blocks, !device && !WorkerQueue.isInline);

    // 1-deep prefetch: the next block's KNN runs while the current block
    // gathers and costs. GpuKnn executions share one set of buffers, so they
    // are serialized through `gpuKnnQueue` — the prefetched block's KNN
    // starts only after the current block's has finished.
    let gpuKnnQueue: Promise<unknown> = Promise.resolve();
    const prepare = (bi: number): Promise<Uint32Array> => {
        const owned = order.subarray(blocks[bi].start, blocks[bi].end);
        const nOwned = owned.length;
        const home = blockPart[bi];
        const queryPos = new Float32Array(nOwned * 3);
        for (let i = 0; i < nOwned; i++) {
            const g = owned[i];
            queryPos[i * 3] = pos.x[g];
            queryPos[i * 3 + 1] = pos.y[g];
            queryPos[i * 3 + 2] = pos.z[g];
        }
        if (device) {
            const out = new Uint32Array(nOwned * k);
            const run = gpuKnnQueue.then(() => gpuKnn!.execute(queryPos, owned, nOwned, out, home));
            gpuKnnQueue = run.catch(() => { /* surfaced by the awaiting block */ });
            return run.then(() => out);
        }
        // CPU path: split the block's queries across the worker pool (extra
        // slices just queue when the pool is smaller). Home part first, so
        // the other parts cull on their AABBs.
        const ordered = [forest[home], ...forest.filter((_, i) => i !== home)];
        const per = Math.ceil(nOwned / 4);
        const jobs: Promise<Uint32Array>[] = [];
        for (let off = 0; off < nOwned; off += per) {
            const cnt = Math.min(per, nOwned - off);
            const qp = queryPos.slice(off * 3, (off + cnt) * 3);
            const qi = owned.slice(off, off + cnt);
            jobs.push(WorkerQueue.run('knnForest', { parts: ordered, queryPos: qp, queryIds: qi, k }, [
                qp.buffer as ArrayBuffer, qi.buffer as ArrayBuffer
            ]));
        }
        return Promise.all(jobs).then((outs) => {
            const out = new Uint32Array(nOwned * k);
            let at = 0;
            for (const o of outs) {
                out.set(o, at);
                at += o.length;
            }
            return out;
        });
    };

    try {
        if (device) {
            gpuKnn = new GpuKnn(device, forest, k);
            if (cand) gpuCost = new GpuEdgeCost(device, gpuCostCapacity, k);
        }

        let next: Promise<Uint32Array> | null = blocks.length > 0 ? prepare(0) : null;

        for (let bi = 0; bi < blocks.length; bi++) {
            const nbPromise = next!;
            next = bi + 1 < blocks.length ? prepare(bi + 1) : null;

            const owned = order.subarray(blocks[bi].start, blocks[bi].end);
            const nOwned = owned.length;
            // Global neighbour ids, canonically (d², id)-ordered per row so
            // downstream tie-breaking is deterministic across KNN paths.
            const nb = await nbPromise;
            sortNeighborRows(pos, owned, nb, k);

            const slots = nOwned * k;

            // Externals: referenced ids outside the owned range — count,
            // collect, sort, dedup. Only cost evaluation needs them; the
            // persisted cache covers owned rows only (every gaussian is owned
            // by exactly one block).
            let extraGlobals = new Uint32Array(0);
            if (cand) {
                let extCount = 0;
                for (let s = 0; s < slots; s++) {
                    const g = nb[s];
                    if (g === KNN_SENTINEL || indexOfSorted(owned, g) >= 0) continue;
                    extCount++;
                }
                const extSorted = new Uint32Array(extCount);
                extCount = 0;
                for (let s = 0; s < slots; s++) {
                    const g = nb[s];
                    if (g === KNN_SENTINEL || indexOfSorted(owned, g) >= 0) continue;
                    extSorted[extCount++] = g;
                }
                extSorted.sort();
                let uniq = 0;
                for (let i = 0; i < extCount; i++) {
                    if (i === 0 || extSorted[i] !== extSorted[i - 1]) extSorted[uniq++] = extSorted[i];
                }
                extraGlobals = extSorted.subarray(0, uniq);
            }

            // Cross-block neighbours aren't bounded by the initial capacity
            // estimate, so a pathological block's view can exceed the
            // preallocated cost buffers — grow them to the actual view size
            // when that happens (rare; costs one reallocation).
            const viewN = nOwned + extraGlobals.length;
            if (gpuCost && viewN > gpuCostCapacity) {
                gpuCost.destroy();
                gpuCostCapacity = Math.ceil(viewN * 1.1);
                gpuCost = new GpuEdgeCost(device!, gpuCostCapacity, k);
            }

            const { view } = await gatherBlockView(ctx, bi, extraGlobals, false, COLOR_COMPONENTS);

            // Per-view cost cache: the 16-f32 rows double as the GPU upload
            // and the re-costed selection's resident copy.
            const cache = new Float32Array((view.geo.length / 8) * CACHE_STRIDE);
            buildSplatCache(view, cache);

            if (cand) {
                // Translate neighbour slots: global ids → view rows
                // (dense-slot edge model; sentinel slots stay sentinel). `nb`
                // keeps the global ids for the candidate output.
                const nbRow = new Uint32Array(slots);
                for (let s = 0; s < slots; s++) {
                    const g = nb[s];
                    if (g === KNN_SENTINEL) {
                        nbRow[s] = KNN_SENTINEL;
                        continue;
                    }
                    const oi = indexOfSorted(owned, g);
                    nbRow[s] = oi >= 0 ? oi : nOwned + indexOfSorted(extraGlobals, g);
                }

                const blockCosts = new Float32Array(slots);
                if (gpuCost) {
                    await gpuCost.execute(cache, view.geo.length / 8, nbRow, blockCosts);
                } else {
                    for (let s = 0; s < slots; s++) {
                        const row = nbRow[s];
                        blockCosts[s] = row === KNN_SENTINEL ?
                            0 :
                            computeEdgeCost(cache, (s / k) | 0, row);
                    }
                }

                // Reduce to best K candidates per owned gaussian (ascending by
                // cost; sentinel slots are skipped by id — their cost values
                // are never read).
                const bestIdx = new Uint32Array(K);
                const bestCost = new Float64Array(K);
                for (let qi = 0; qi < nOwned; qi++) {
                    let size = 0;
                    const base = qi * k;
                    for (let s = 0; s < k; s++) {
                        if (nb[base + s] === KNN_SENTINEL) continue;
                        const c = blockCosts[base + s];
                        if (!Number.isFinite(c)) continue;
                        if (size === K && c >= bestCost[K - 1]) continue;
                        let at = size < K ? size : K - 1;
                        while (at > 0 && bestCost[at - 1] > c) {
                            bestCost[at] = bestCost[at - 1];
                            bestIdx[at] = bestIdx[at - 1];
                            at--;
                        }
                        bestCost[at] = c;
                        bestIdx[at] = nb[base + s];
                        size = Math.min(size + 1, K);
                    }
                    const g = owned[qi];
                    for (let s = 0; s < K; s++) {
                        cand.idx[g * K + s] = s < size ? bestIdx[s] : 0xFFFFFFFF;
                        cand.cost[g * K + s] = s < size ? bestCost[s] : Infinity;
                    }
                }
            }

            // Persist owned rows for re-costed selection: the splat cache and
            // the global neighbour ids (sentinel-padded).
            if (ctx.cacheOut) {
                const CO = ctx.cacheOut;
                for (let qi = 0; qi < nOwned; qi++) {
                    CO.set(cache.subarray(qi * CACHE_STRIDE, (qi + 1) * CACHE_STRIDE), owned[qi] * CACHE_STRIDE);
                }
            }
            if (ctx.neighborsOut) {
                const NO = ctx.neighborsOut;
                for (let qi = 0; qi < nOwned; qi++) {
                    NO.set(nb.subarray(qi * k, (qi + 1) * k), owned[qi] * k);
                }
            }

            tick?.(nOwned);
        }
    } finally {
        gpuKnn?.destroy();
        gpuCost?.destroy();
    }
};

export {
    runPriorityPass,
    buildForest,
    gatherBlockView,
    sortNeighborRows,
    indexOfSorted,
    VIEW_GROW,
    type CandidateArrays,
    type PriorityContext,
    type BlockView
};
