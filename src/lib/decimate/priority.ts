import { type GraphicsDevice } from 'playcanvas';

import { buildSplatCache, computeEdgeCost, CACHE_STRIDE } from './edge-cost-cpu';
import { collectBlock, verifyAndFixKnn, toGlobalNeighbors, KNN_FIXED, type BlockLocals } from './knn-blocks';
import { KNN_SENTINEL } from './knn-core';
import { type SplatView } from './moment-match';
import { type BlockRange, type ResidentPositions } from './partition';
import { type ChunkData, type ChunkDataPool, type ChunkSource } from '../chunk';
import { GpuEdgeCost } from '../gpu/gpu-edge-cost';
import { GpuKnn } from '../gpu/gpu-knn';
import { WorkerQueue } from '../workers';

/** Halo radius multiplier on the density-estimated k-NN radius. */
const HALO_FACTOR = 2.5;

/** Halo size cap as a multiple of a block's owned count (buffer-sizing bound). */
const HALO_CAP = 1;

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
 * The quality cost reads only DC, so its pass gathers 3 — at SH band 3 that
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
    const maxLocalN = maxOwned * (1 + HALO_CAP);

    let gpuKnn: GpuKnn | undefined;
    let gpuCost: GpuEdgeCost | undefined;
    let gpuCostCapacity = maxLocalN;

    // 1-deep prefetch: the next block's halo collection + tree build runs
    // while the current block computes. GpuKnn executions share one set of
    // buffers, so they are serialized through `gpuKnnQueue` — the prefetched
    // block's KNN starts only after the current block's has finished.
    let gpuKnnQueue: Promise<unknown> = Promise.resolve();
    type Prepared = { locals: BlockLocals; nb: Promise<Uint32Array> };
    const prepare = (bi: number): Prepared => {
        const locals = collectBlock(pos, order, blocks, bi, k, HALO_FACTOR, HALO_CAP);
        const copy = locals.positions.slice();
        if (device) {
            const treePromise = WorkerQueue.run('buildFlatKdTree', { positions: copy }, [copy.buffer as ArrayBuffer]);
            const out = new Uint32Array(locals.ownedCount * k);
            const run = Promise.all([treePromise, gpuKnnQueue]).then(([flat]) => {
                return gpuKnn!.execute(flat, locals.positions, locals.ids.length, locals.ownedCount, out);
            });
            gpuKnnQueue = run.catch(() => { /* surfaced by the awaiting block */ });
            return { locals, nb: run.then(() => out) };
        }
        const nb = WorkerQueue.run('knnBlock', { positions: copy, ownedCount: locals.ownedCount, k }, [copy.buffer as ArrayBuffer]);
        return { locals, nb };
    };

    try {
        if (device) {
            gpuKnn = new GpuKnn(device, maxLocalN, k);
            if (cand) gpuCost = new GpuEdgeCost(device, maxLocalN, k);
        }

        let next: Prepared | null = blocks.length > 0 ? prepare(0) : null;

        for (let bi = 0; bi < blocks.length; bi++) {
            const { locals, nb: nbPromise } = next!;
            next = bi + 1 < blocks.length ? prepare(bi + 1) : null;

            const nOwned = locals.ownedCount;
            const owned = order.subarray(blocks[bi].start, blocks[bi].end);
            const nbLocal = await nbPromise;
            const nbGlobal = toGlobalNeighbors(locals, nbLocal);
            verifyAndFixKnn(pos, order, blocks, bi, locals, k, nbGlobal, nbLocal);

            const slots = nOwned * k;

            // Externals: referenced rows outside the owned range (halo members
            // and verification-fixed neighbours) — count, collect, sort, dedup.
            // Only cost evaluation needs them; the persisted cache covers owned
            // rows only (every gaussian is owned by exactly one block).
            let extraGlobals = new Uint32Array(0);
            if (cand) {
                let extCount = 0;
                for (let s = 0; s < slots; s++) {
                    const l = nbLocal[s];
                    if (l === KNN_SENTINEL || l < nOwned) continue;
                    if (l === KNN_FIXED && indexOfSorted(owned, nbGlobal[s]) >= 0) continue;
                    extCount++;
                }
                const extSorted = new Uint32Array(extCount);
                extCount = 0;
                for (let s = 0; s < slots; s++) {
                    const l = nbLocal[s];
                    if (l === KNN_SENTINEL || l < nOwned) continue;
                    const g = nbGlobal[s];
                    if (l === KNN_FIXED && indexOfSorted(owned, g) >= 0) continue;
                    extSorted[extCount++] = g;
                }
                extSorted.sort();
                let uniq = 0;
                for (let i = 0; i < extCount; i++) {
                    if (i === 0 || extSorted[i] !== extSorted[i - 1]) extSorted[uniq++] = extSorted[i];
                }
                extraGlobals = extSorted.subarray(0, uniq);
            }

            // Verification-fixed externals are not bounded by the halo cap, so
            // a pathological block's view can exceed the preallocated cost
            // buffers — grow them to the actual view size when that happens
            // (rare; costs one reallocation).
            const viewN = nOwned + extraGlobals.length;
            if (gpuCost && viewN > gpuCostCapacity) {
                gpuCost.destroy();
                gpuCostCapacity = Math.ceil(viewN * 1.1);
                gpuCost = new GpuEdgeCost(device!, gpuCostCapacity, k);
            }

            const { view } = await gatherBlockView(ctx, bi, extraGlobals, false, 3);

            // Per-splat cost cache, built once for the GPU upload, the CPU
            // path, and the re-costed selection's resident copy alike.
            const cache = new Float32Array(viewN * CACHE_STRIDE);
            buildSplatCache(view, cache);

            if (cand) {
                // Translate neighbour slots in place: local/global → view rows
                // (dense-slot edge model; sentinel slots stay sentinel).
                for (let s = 0; s < slots; s++) {
                    const l = nbLocal[s];
                    if (l === KNN_SENTINEL || l < nOwned) continue;
                    const g = nbGlobal[s];
                    if (l === KNN_FIXED) {
                        const oi = indexOfSorted(owned, g);
                        nbLocal[s] = oi >= 0 ? oi : nOwned + indexOfSorted(extraGlobals, g);
                    } else {
                        nbLocal[s] = nOwned + indexOfSorted(extraGlobals, g);
                    }
                }

                const blockCosts = new Float32Array(slots);
                if (device) {
                    await gpuCost!.execute(cache, viewN, nbLocal, blockCosts);
                } else {
                    for (let s = 0; s < slots; s++) {
                        const row = nbLocal[s];
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
                        if (nbLocal[base + s] === KNN_SENTINEL) continue;
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
                        bestIdx[at] = nbGlobal[base + s];
                        size = Math.min(size + 1, K);
                    }
                    const g = owned[qi];
                    for (let s = 0; s < K; s++) {
                        cand.idx[g * K + s] = s < size ? bestIdx[s] : 0xFFFFFFFF;
                        cand.cost[g * K + s] = s < size ? bestCost[s] : Infinity;
                    }
                }
            }

            // Persist owned rows for re-costed selection: the splat cache
            // (identical layout on both paths) and the global neighbour ids
            // (sentinel-padded).
            if (ctx.cacheOut) {
                const CO = ctx.cacheOut;
                for (let qi = 0; qi < nOwned; qi++) {
                    CO.set(cache.subarray(qi * CACHE_STRIDE, (qi + 1) * CACHE_STRIDE), owned[qi] * CACHE_STRIDE);
                }
            }
            if (ctx.neighborsOut) {
                const NO = ctx.neighborsOut;
                for (let qi = 0; qi < nOwned; qi++) {
                    NO.set(nbGlobal.subarray(qi * k, (qi + 1) * k), owned[qi] * k);
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
    gatherBlockView,
    indexOfSorted,
    HALO_FACTOR,
    HALO_CAP,
    type CandidateArrays,
    type PriorityContext,
    type BlockView
};
