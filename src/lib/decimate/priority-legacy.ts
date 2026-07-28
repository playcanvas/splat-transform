/**
 * LEGACY priority pass — the pre-quality-study pipeline (KL-style cost with
 * full-SH colour term, appearance-chunk GPU kernel), preserved verbatim from
 * main for `--decimate-mode legacy`. Self-contained on purpose: shares only
 * the KNN/partition machinery with the current pass.
 */
import { type GraphicsDevice } from 'playcanvas';

import { buildCostCacheLegacy, computeEdgeCostViewLegacy } from './edge-cost-legacy';
import { KNN_SENTINEL } from './knn-core';
import { createMergeScratch, makeGaussianSamples, sigmoid, ellipsoidArea, type SplatView } from './moment-match';
import { type BlockRange, type ResidentPositions } from './partition';
import { buildForest, sortNeighborRows, VIEW_GROW } from './priority';
import { type ChunkData, type ChunkDataPool, type ChunkSource } from '../chunk';
import { APP_CHUNK, GpuEdgeCostLegacy, type EdgeCostCacheLegacy } from '../gpu/gpu-edge-cost-legacy';
import { GpuKnn } from '../gpu/gpu-knn';
import { WorkerQueue } from '../workers';

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
 * @returns The gathered block view.
 */
const gatherBlockView = async (
    ctx: Pick<PriorityContext, 'source' | 'pool' | 'pos' | 'order' | 'blocks'>,
    blockIdx: number,
    extraGlobals: Uint32Array,
    includeOther = false
): Promise<BlockView> => {
    const { source, pool, pos, order, blocks } = ctx;
    const block = blocks[blockIdx];
    const owned = order.subarray(block.start, block.end);
    const nOwned = owned.length;
    const n = nOwned + extraGlobals.length;
    const { layouts, availableLayers } = source.meta;

    const colorDim = layouts.color!.stride >> 2;
    const wantOther = includeOther && availableLayers.has('other') && (layouts.other?.stride ?? 0) > 0;
    const otherDim = wantOther ? layouts.other!.stride >> 2 : 0;

    const view: SplatView = {
        pos: new Float32Array(n * 3),
        geo: new Float32Array(n * 8),
        color: new Float32Array(n * colorDim),
        colorDim
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
            view.color.set(new Float32Array(colCd.data, 0, count * colorDim), (rowBase + off) * colorDim);
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

// Pack the block view into the GpuEdgeCostLegacy cache layout (legacy packing:
// posScalars 8-wide, rotR from normalized quats, appearance in ≤APP_CHUNK
// column chunks with live-width strides).
const packGpuCacheLegacy = (view: SplatView): EdgeCostCacheLegacy => {
    const { pos, geo, color, colorDim } = view;
    const n = geo.length / 8;
    const posScalars = new Float32Array(n * 8);
    const rotR = new Float32Array(n * 9);
    const rot = new Float32Array(9);

    for (let i = 0; i < n; i++) {
        const i8 = i * 8;
        const o = i * 8;
        const linAlpha = sigmoid(geo[i8 + 7]);
        const sx = Math.max(Math.exp(geo[i8 + 4]), 1e-12);
        const sy = Math.max(Math.exp(geo[i8 + 5]), 1e-12);
        const sz = Math.max(Math.exp(geo[i8 + 6]), 1e-12);
        const vx = sx * sx + 1e-8;
        const vy = sy * sy + 1e-8;
        const vz = sz * sz + 1e-8;
        posScalars[o] = pos[i * 3];
        posScalars[o + 1] = pos[i * 3 + 1];
        posScalars[o + 2] = pos[i * 3 + 2];
        posScalars[o + 3] = linAlpha * ellipsoidArea(sx, sy, sz) + 1e-12;
        posScalars[o + 4] = Math.log(Math.max(vx, 1e-30)) + Math.log(Math.max(vy, 1e-30)) + Math.log(Math.max(vz, 1e-30));
        posScalars[o + 5] = vx;
        posScalars[o + 6] = vy;
        posScalars[o + 7] = vz;

        let qw = geo[i8], qx = geo[i8 + 1], qy = geo[i8 + 2], qz = geo[i8 + 3];
        const invq = 1 / Math.max(Math.hypot(qw, qx, qy, qz), 1e-12);
        qw *= invq; qx *= invq; qy *= invq; qz *= invq;
        const xx = qx * qx, yy = qy * qy, zz = qz * qz;
        const wx = qw * qx, wy = qw * qy, wz = qw * qz;
        const xy = qx * qy, xz = qx * qz, yz = qy * qz;
        rot[0] = 1 - 2 * (yy + zz); rot[1] = 2 * (xy - wz); rot[2] = 2 * (xz + wy);
        rot[3] = 2 * (xy + wz); rot[4] = 1 - 2 * (xx + zz); rot[5] = 2 * (yz - wx);
        rot[6] = 2 * (xz - wy); rot[7] = 2 * (yz + wx); rot[8] = 1 - 2 * (xx + yy);
        rotR.set(rot, i * 9);
    }

    const numChunks = Math.ceil(colorDim / APP_CHUNK);
    const appChunks: Float32Array[] = [];
    for (let ch = 0; ch < numChunks; ch++) {
        const kStart = ch * APP_CHUNK;
        const width = Math.min(APP_CHUNK, colorDim - kStart);
        const chunk = new Float32Array(n * width);
        for (let s = 0; s < n; s++) {
            const dst = s * width;
            const src = s * colorDim + kStart;
            for (let kk = 0; kk < width; kk++) chunk[dst + kk] = color[src + kk];
        }
        appChunks.push(chunk);
    }

    return { posScalars, rotR, appChunks, numAppCols: colorDim, numSplats: n };
};

/**
 * The priority pass (heavy read 1): per block — exact global KNN, edge costs
 * for each owned gaussian's k neighbours, reduction to the best K candidates
 * — written into the resident candidate arrays.
 *
 * @param ctx - The pass context.
 * @param cand - Preallocated candidate arrays (`N*K`), filled per block.
 * @param tick - Optional progress callback (owned gaussians completed).
 */
const runPriorityPassLegacy = async (
    ctx: PriorityContext,
    cand: CandidateArrays,
    tick?: (n: number) => void
): Promise<void> => {
    const { pos, order, blocks, device, K, k } = ctx;
    const Z = makeGaussianSamples(1, 0);
    const z = new Float32Array([Z[0][0], Z[0][1], Z[0][2]]);
    const colorDim = ctx.source.meta.layouts.color!.stride >> 2;

    let maxOwned = 0;
    for (const b of blocks) maxOwned = Math.max(maxOwned, b.end - b.start);

    let gpuKnn: GpuKnn | undefined;
    let gpuCost: GpuEdgeCostLegacy | undefined;
    let gpuCostCapacity = Math.ceil(maxOwned * VIEW_GROW);

    // The forest is built once per generation; blocks stay the query/IO
    // batching unit (see the quality pass for the shared machinery).
    const { parts: forest, blockPart } = await buildForest(pos, order, blocks, !device && !WorkerQueue.isInline);

    // 1-deep prefetch: the next block's KNN runs while the current block
    // gathers and costs. GpuKnn executions share one set of buffers, so they
    // are serialized through `gpuKnnQueue`.
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
            gpuCost = new GpuEdgeCostLegacy(device, gpuCostCapacity, maxOwned * k, colorDim);
        }

        let next: Promise<Uint32Array> | null = blocks.length > 0 ? prepare(0) : null;

        for (let bi = 0; bi < blocks.length; bi++) {
            const nbPromise = next!;
            next = bi + 1 < blocks.length ? prepare(bi + 1) : null;

            const owned = order.subarray(blocks[bi].start, blocks[bi].end);
            const nOwned = owned.length;
            const nb = await nbPromise;
            sortNeighborRows(pos, owned, nb, k);

            // Externals: referenced ids outside the owned range, sorted for
            // the gather.
            const extRow = new Map<number, number>();
            for (let s = 0; s < nOwned * k; s++) {
                const g = nb[s];
                if (g === KNN_SENTINEL) continue;
                if (indexOfSorted(owned, g) < 0 && !extRow.has(g)) extRow.set(g, 0);
            }
            const extraGlobals = Uint32Array.from(extRow.keys()).sort();
            for (let i = 0; i < extraGlobals.length; i++) extRow.set(extraGlobals[i], nOwned + i);

            // Cross-block neighbours aren't bounded by the initial capacity
            // estimate, so a pathological block's view can exceed the
            // preallocated cost buffers — grow them to the actual view size
            // when that happens (rare; costs one reallocation).
            const viewN = nOwned + extraGlobals.length;
            if (gpuCost && viewN > gpuCostCapacity) {
                gpuCost.destroy();
                gpuCostCapacity = Math.ceil(viewN * 1.1);
                gpuCost = new GpuEdgeCostLegacy(device!, gpuCostCapacity, maxOwned * k, colorDim);
            }

            const { view } = await gatherBlockView(ctx, bi, extraGlobals);

            // Edge lists in owned-major order (view-local endpoints).
            const edgeI = new Uint32Array(nOwned * k);
            const edgeJ = new Uint32Array(nOwned * k);
            const edgeNb = new Uint32Array(nOwned * k);   // global neighbour per edge
            const edgeOf = new Uint32Array(nOwned + 1);   // CSR into the edge list per owned row
            let e = 0;
            for (let qi = 0; qi < nOwned; qi++) {
                edgeOf[qi] = e;
                for (let s = 0; s < k; s++) {
                    const g = nb[qi * k + s];
                    if (g === KNN_SENTINEL) continue;
                    const oi = indexOfSorted(owned, g);
                    edgeI[e] = qi;
                    edgeJ[e] = oi >= 0 ? oi : extRow.get(g)!;
                    edgeNb[e] = g;
                    e++;
                }
            }
            edgeOf[nOwned] = e;

            const costs = new Float32Array(e);
            if (device) {
                await gpuCost!.execute(packGpuCacheLegacy(view), edgeI.subarray(0, e), edgeJ.subarray(0, e), z, costs);
            } else {
                const cache = buildCostCacheLegacy(view);
                const scratch = createMergeScratch();
                for (let i = 0; i < e; i++) {
                    costs[i] = computeEdgeCostViewLegacy(view, cache, edgeI[i], edgeJ[i], Z, scratch);
                }
            }

            // Reduce to best K candidates per owned gaussian (ascending by cost).
            const bestIdx = new Uint32Array(K);
            const bestCost = new Float64Array(K);
            for (let qi = 0; qi < nOwned; qi++) {
                let size = 0;
                for (let s = edgeOf[qi]; s < edgeOf[qi + 1]; s++) {
                    const c = costs[s];
                    if (!Number.isFinite(c)) continue;
                    if (size === K && c >= bestCost[K - 1]) continue;
                    let at = size < K ? size : K - 1;
                    while (at > 0 && bestCost[at - 1] > c) {
                        bestCost[at] = bestCost[at - 1];
                        bestIdx[at] = bestIdx[at - 1];
                        at--;
                    }
                    bestCost[at] = c;
                    bestIdx[at] = edgeNb[s];
                    size = Math.min(size + 1, K);
                }
                const g = owned[qi];
                for (let s = 0; s < K; s++) {
                    cand.idx[g * K + s] = s < size ? bestIdx[s] : 0xFFFFFFFF;
                    cand.cost[g * K + s] = s < size ? bestCost[s] : Infinity;
                }
            }

            tick?.(nOwned);
        }
    } finally {
        gpuKnn?.destroy();
        gpuCost?.destroy();
    }
};

export { runPriorityPassLegacy };
