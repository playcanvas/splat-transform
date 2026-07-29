import { type GraphicsDevice } from 'playcanvas';

import { buildSplatCache, CACHE_STRIDE } from './edge-cost-cpu';
import { KNN_SENTINEL } from './knn-core';
import { type BlockView, gatherBlockView, type PriorityContext } from './priority';
import { GpuKnn } from '../gpu/gpu-knn';
import { WorkerQueue } from '../workers';

type PreparedBlock = BlockView & {
    cache: Float32Array;
    /** Local row ids; rows `[0, ownedCount)` are core queries. */
    neighbors: Uint32Array;
};

/**
 * Gather one core-plus-halo view and run exact KNN over that local view on
 * WebGPU. The returned candidate rows are fixed for the lifetime of the
 * block plan.
 *
 * @param ctx - Source, partition, and resident positions.
 * @param blockIndex - Core block index.
 * @param haloGlobals - Sorted immutable halo rows.
 * @param device - Required WebGPU device.
 * @param k - Neighbours per core query.
 * @returns Local view, field-L2 cache, and sentinel-padded KNN rows.
 */
const prepareGpuBlock = async (
    ctx: Pick<PriorityContext, 'source' | 'pool' | 'pos' | 'order' | 'blocks'>,
    blockIndex: number,
    haloGlobals: Uint32Array,
    device: GraphicsDevice,
    k: number
): Promise<PreparedBlock> => {
    const block = await gatherBlockView(ctx, blockIndex, haloGlobals, false, 3);
    const { view, ownedCount } = block;
    const totalCount = view.pos.length / 3;
    const x = new Float32Array(totalCount);
    const y = new Float32Array(totalCount);
    const z = new Float32Array(totalCount);
    const ids = new Uint32Array(totalCount);
    for (let i = 0; i < totalCount; i++) {
        ids[i] = i;
        x[i] = view.pos[i * 3];
        y[i] = view.pos[i * 3 + 1];
        z[i] = view.pos[i * 3 + 2];
    }
    const forest = await WorkerQueue.run('buildKdForestPart', { x, y, z, ids, shared: false }, [
        x.buffer as ArrayBuffer,
        y.buffer as ArrayBuffer,
        z.buffer as ArrayBuffer,
        ids.buffer as ArrayBuffer
    ]);

    const queryPos = view.pos.slice(0, ownedCount * 3);
    const queryIds = new Uint32Array(ownedCount);
    for (let i = 0; i < ownedCount; i++) queryIds[i] = i;
    const coreNeighbors = new Uint32Array(ownedCount * k);
    const gpuKnn = new GpuKnn(device, [forest], k);
    try {
        await gpuKnn.execute(queryPos, queryIds, ownedCount, coreNeighbors, 0);
    } finally {
        gpuKnn.destroy();
    }

    // Canonical (distance², local id) order keeps CPU/GPU test references and
    // repeated executions deterministic.
    const dist = new Float64Array(k);
    const cand = new Uint32Array(k);
    for (let q = 0; q < ownedCount; q++) {
        let count = 0;
        const qx = view.pos[q * 3], qy = view.pos[q * 3 + 1], qz = view.pos[q * 3 + 2];
        for (let s = 0; s < k; s++) {
            const id = coreNeighbors[q * k + s];
            if (id === KNN_SENTINEL) continue;
            const dx = view.pos[id * 3] - qx;
            const dy = view.pos[id * 3 + 1] - qy;
            const dz = view.pos[id * 3 + 2] - qz;
            const d2 = dx * dx + dy * dy + dz * dz;
            let at = count;
            while (at > 0 && (dist[at - 1] > d2 || (dist[at - 1] === d2 && cand[at - 1] > id))) {
                dist[at] = dist[at - 1];
                cand[at] = cand[at - 1];
                at--;
            }
            dist[at] = d2;
            cand[at] = id;
            count++;
        }
        for (let s = 0; s < k; s++) coreNeighbors[q * k + s] = s < count ? cand[s] : KNN_SENTINEL;
    }

    // GpuRecost addresses all local rows. Halo rows carry no candidate pools
    // because they are immutable and never refreshed.
    const neighbors = new Uint32Array(totalCount * k).fill(KNN_SENTINEL);
    neighbors.set(coreNeighbors);
    const cache = new Float32Array(totalCount * CACHE_STRIDE);
    buildSplatCache(view, cache);
    return { ...block, cache, neighbors };
};

export { prepareGpuBlock, type PreparedBlock };
