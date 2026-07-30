import {
    BUFFERUSAGE_COPY_DST,
    BUFFERUSAGE_COPY_SRC,
    GraphicsDevice,
    StorageBuffer
} from 'playcanvas';

import { makeKernel, type Kernel } from './compute-kernel';
import { type ForestPart } from '../decimate/knn-core';

/**
 * WGSL kernel: iterative KD-tree K-nearest-neighbours over one forest part.
 *
 * Each thread runs a depth-first traversal of a flattened KD subtree with a
 * fixed-size per-thread stack, carrying its query's top-K in the topDist /
 * topIdx buffers (reset per batch by a separate kernel): each part loads
 * and tightens them. A part whose AABB cannot improve the carried worst is
 * skipped before any traversal (the near-path descent is otherwise
 * unconditional — without the cull every part costs a full descent per
 * query). The caller dispatches each batch's HOME part first, so the top-K
 * is tight before any other part is tested and interior queries cull
 * everything else. Top-K is maintained unsorted with explicit worst-index
 * tracking, so the common "candidate rejected against worst" path is a
 * single compare.
 *
 * The part's root index and AABB are baked as compile-time constants (each
 * part is its own kernel), so only `queryCount` varies per dispatch.
 *
 * @param k - Compile-time K, the number of nearest neighbours per query.
 * @param stackSize - Compile-time per-thread DFS stack depth.
 * @param rootIdx - The part's group-local root node index.
 * @param aabb - The part's point AABB [minx, miny, minz, maxx, maxy, maxz].
 * @returns WGSL source.
 */
const knnWgsl = (k: number, stackSize: number, rootIdx: number, aabb: Float32Array) => /* wgsl */`
struct Uniforms {
    queryCount: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
// This batch's query positions (interleaved xyz) and global splat ids.
@group(0) @binding(1) var<storage, read> queryPos: array<f32>;
@group(0) @binding(2) var<storage, read> queryIds: array<u32>;
// The binding group's concatenated forest parts: node splat ids are GLOBAL,
// node child indices are group-local (offset applied at pack time),
// positions denormalized per node.
@group(0) @binding(3) var<storage, read> nodeSplatIdx: array<u32>;
@group(0) @binding(4) var<storage, read> nodePositions: array<f32>;
@group(0) @binding(5) var<storage, read> nodeChildren: array<u32>;
// Carried per-query top-K state (this batch's rows).
@group(0) @binding(6) var<storage, read_write> topDist: array<f32>;
@group(0) @binding(7) var<storage, read_write> topIdx: array<u32>;

const K: u32 = ${k}u;
const NULL_NODE: u32 = 0xFFFFFFFFu;
const F32_MAX: f32 = 3.4028234663852886e+38;
const STACK_SIZE: u32 = ${stackSize}u;
const ROOT_IDX: u32 = ${rootIdx}u;
const AABB_MIN: vec3f = vec3f(${aabb[0]}, ${aabb[1]}, ${aabb[2]});
const AABB_MAX: vec3f = vec3f(${aabb[3]}, ${aabb[4]}, ${aabb[5]});

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let bid = gid.x;
    if (bid >= uniforms.queryCount) { return; }

    let q3 = bid * 3u;
    let qx = queryPos[q3 + 0u];
    let qy = queryPos[q3 + 1u];
    let qz = queryPos[q3 + 2u];
    let qid = queryIds[bid];

    // Load the carried top-K (unsorted, worst tracked): distances first and
    // cull on the part AABB before touching anything else — a part that
    // cannot improve the top-K costs one distance-row read, not a tree
    // descent. (worst == F32_MAX means unfilled slots — never cull then.)
    var tIdx: array<u32, ${k}>;
    var tDist: array<f32, ${k}>;
    let base = bid * K;
    var worstIdx: u32 = 0u;
    for (var i: u32 = 0u; i < K; i++) {
        tDist[i] = topDist[base + i];
    }
    var worst: f32 = tDist[0];
    for (var i: u32 = 1u; i < K; i++) {
        if (tDist[i] > worst) { worst = tDist[i]; worstIdx = i; }
    }
    let dd = max(max(AABB_MIN - vec3f(qx, qy, qz), vec3f(qx, qy, qz) - AABB_MAX), vec3f(0.0));
    if (worst < F32_MAX && dot(dd, dd) >= worst) { return; }
    for (var i: u32 = 0u; i < K; i++) {
        tIdx[i] = topIdx[base + i];
    }

    // Stack: (nodeIdx, axis) packed as u32. axis ∈ {0,1,2} in top 2 bits,
    // group-local nodeIdx in low 30 — supports up to ~1B nodes per group.
    var stack: array<u32, ${stackSize}>;
    var sp: u32 = 0u;
    stack[0] = ROOT_IDX;   // axis=0 → no axis bits set
    sp = 1u;

    while (sp > 0u) {
        sp = sp - 1u;
        let packed = stack[sp];
        let nodeIdx = packed & 0x3FFFFFFFu;
        let axis = packed >> 30u;

        let np = nodeIdx * 3u;
        let nx = nodePositions[np + 0u];
        let ny = nodePositions[np + 1u];
        let nz = nodePositions[np + 2u];
        let splatId = nodeSplatIdx[nodeIdx];

        // Update top-K, skipping the query itself (by global id, so
        // coincident points stay valid mutual neighbours).
        if (splatId != qid) {
            let dx = nx - qx;
            let dy = ny - qy;
            let dz = nz - qz;
            let d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < worst) {
                tDist[worstIdx] = d2;
                tIdx[worstIdx] = splatId;
                var w: f32 = tDist[0];
                var wi: u32 = 0u;
                for (var i: u32 = 1u; i < K; i++) {
                    if (tDist[i] > w) { w = tDist[i]; wi = i; }
                }
                worst = w;
                worstIdx = wi;
            }
        }

        var qAxisVal: f32;
        var nAxisVal: f32;
        if (axis == 0u) { qAxisVal = qx; nAxisVal = nx; }
        else if (axis == 1u) { qAxisVal = qy; nAxisVal = ny; }
        else { qAxisVal = qz; nAxisVal = nz; }

        let delta = qAxisVal - nAxisVal;
        let nextAxis = select(axis + 1u, 0u, axis + 1u >= 3u);
        let nextAxisPacked = nextAxis << 30u;

        let nc = nodeIdx * 2u;
        let leftChild = nodeChildren[nc + 0u];
        let rightChild = nodeChildren[nc + 1u];
        let near = select(rightChild, leftChild, delta < 0.0);
        let far = select(leftChild, rightChild, delta < 0.0);

        if (far != NULL_NODE && delta * delta < worst) {
            stack[sp] = far | nextAxisPacked;
            sp = sp + 1u;
        }
        if (near != NULL_NODE) {
            stack[sp] = near | nextAxisPacked;
            sp = sp + 1u;
        }
    }

    // Store the carried state back. Unfilled slots (n-1 < K over the whole
    // forest) keep the NULL sentinel, matching the CPU path.
    for (var i: u32 = 0u; i < K; i++) {
        topDist[base + i] = tDist[i];
        topIdx[base + i] = tIdx[i];
    }
}
`;

// Reset the batch's carried top-K rows (runs before the batch's parts).
const knnResetWgsl = () => /* wgsl */`
struct Uniforms {
    slotCount: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> topDist: array<f32>;
@group(0) @binding(2) var<storage, read_write> topIdx: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= uniforms.slotCount) { return; }
    topDist[i] = 3.4028234663852886e+38;
    topIdx[i] = 0xFFFFFFFFu;
}
`;

/**
 * GPU K-nearest-neighbours over a forest of flat KD subtrees.
 *
 * The forest (built per generation by the `buildKdForestPart` worker task,
 * global splat ids in `nodeSplatIdx`) is uploaded once at construction:
 * parts are concatenated into binding groups sized under the device's
 * storage-binding limit, child indices offset to group-local at pack time.
 * Queries batch at 65,536; each batch dispatches every part in order (one
 * Compute per part — uniforms never vary within a submit) with the top-K
 * carried in buffers, then reads back ids only. Exactness: the carried
 * worst-distance bound makes the multi-part traversal equivalent to one
 * tree over the union.
 */
class GpuKnn {
    /**
     * @param queryPos - Interleaved xyz for this call's queries.
     * @param queryIds - Global splat id per query (self-exclusion).
     * @param queryCount - Query count.
     * @param outNeighbours - `queryCount * k` GLOBAL neighbour ids
     * (UNSORTED within a row; 0xFFFFFFFF sentinel for surplus slots).
     * @param homePart - Part index containing this call's queries; it is
     * traversed first so every other part sees a tight bound and culls.
     */
    execute: (
        queryPos: Float32Array,
        queryIds: Uint32Array,
        queryCount: number,
        outNeighbours: Uint32Array,
        homePart: number
    ) => Promise<void>;
    destroy: () => void;

    /**
     * @param device - PlayCanvas GraphicsDevice (WebGPU).
     * @param parts - The forest (node splat ids GLOBAL, children part-local).
     * @param k - Number of nearest neighbours per query.
     */
    constructor(device: GraphicsDevice, parts: ForestPart[], k: number) {
        const workgroupSize = 64;
        const queriesPerBatch = 1024 * workgroupSize;  // 65,536
        const stackSize = 48;

        // Group parts under the per-binding ceiling (positions, 12 B/node,
        // is the largest array).
        const limits = (device as unknown as { limits?: { maxStorageBufferBindingSize?: number, maxBufferSize?: number } }).limits;
        const maxBinding = Math.min(
            typeof limits?.maxStorageBufferBindingSize === 'number' ? limits.maxStorageBufferBindingSize : 128 * 2 ** 20,
            typeof limits?.maxBufferSize === 'number' ? limits.maxBufferSize : 256 * 2 ** 20
        );
        const maxNodesPerGroup = Math.floor(maxBinding / 12);

        type Group = { parts: { part: ForestPart, base: number }[], nodes: number };
        const groups: Group[] = [];
        for (const part of parts) {
            const n = part.nodeSplatIdx.length;
            if (n > maxNodesPerGroup) {
                throw new Error(`GpuKnn: a forest part (${n} nodes) exceeds the device binding limit — reduce the part size`);
            }
            if (n > 0x3FFFFFFF) {
                throw new Error(`GpuKnn: part exceeds the 30-bit node packing limit (${n} nodes)`);
            }
            let g = groups[groups.length - 1];
            if (!g || g.nodes + n > maxNodesPerGroup) {
                g = { parts: [], nodes: 0 };
                groups.push(g);
            }
            g.parts.push({ part, base: g.nodes });
            g.nodes += n;
        }

        // Shared per-batch buffers.
        const queryPosBuf = new StorageBuffer(device, queriesPerBatch * 3 * 4, BUFFERUSAGE_COPY_DST);
        const queryIdBuf = new StorageBuffer(device, queriesPerBatch * 4, BUFFERUSAGE_COPY_DST);
        const topDistBuf = new StorageBuffer(device, queriesPerBatch * k * 4, BUFFERUSAGE_COPY_DST);
        const topIdxBuf = new StorageBuffer(
            device,
            queriesPerBatch * k * 4,
            BUFFERUSAGE_COPY_SRC | BUFFERUSAGE_COPY_DST
        );
        const outScratch = new Uint32Array(queriesPerBatch * k);

        // Per-batch top-K reset (frees the part dispatch order per batch).
        const resetKernel = makeKernel(device, 'compute-knn-reset', knnResetWgsl(), ['slotCount'], [
            ['topDist', false],
            ['topIdx', false]
        ]);
        resetKernel.compute.setParameter('topDist', topDistBuf);
        resetKernel.compute.setParameter('topIdx', topIdxBuf);

        // Per group: concatenated node buffers (children offset to
        // group-local); per part: a Compute with its root/AABB baked in
        // (only queryCount varies per dispatch).
        const kernels: Kernel[] = [];
        const groupBufs: StorageBuffer[] = [];
        for (const g of groups) {
            const splatIdxBuf = new StorageBuffer(device, g.nodes * 4, BUFFERUSAGE_COPY_DST);
            const positionsBuf = new StorageBuffer(device, g.nodes * 3 * 4, BUFFERUSAGE_COPY_DST);
            const childrenBuf = new StorageBuffer(device, g.nodes * 2 * 4, BUFFERUSAGE_COPY_DST);
            groupBufs.push(splatIdxBuf, positionsBuf, childrenBuf);

            const childScratch = new Uint32Array(1 << 16);
            for (const { part, base } of g.parts) {
                const n = part.nodeSplatIdx.length;
                splatIdxBuf.write(base * 4, part.nodeSplatIdx, 0, n);
                positionsBuf.write(base * 3 * 4, part.nodePositions, 0, n * 3);
                // Children shift to group-local indices at upload.
                let remapped = childScratch;
                if (remapped.length < n * 2) remapped = new Uint32Array(n * 2);
                for (let i = 0; i < n * 2; i++) {
                    const c = part.nodeChildren[i];
                    remapped[i] = c === 0xFFFFFFFF ? c : c + base;
                }
                childrenBuf.write(base * 2 * 4, remapped, 0, n * 2);

                const source = knnWgsl(k, stackSize, base + part.rootIdx, part.aabb);
                const kernel = makeKernel(device, 'compute-knn-forest', source, ['queryCount'], [
                    ['queryPos', true],
                    ['queryIds', true],
                    ['nodeSplatIdx', true],
                    ['nodePositions', true],
                    ['nodeChildren', true],
                    ['topDist', false],
                    ['topIdx', false]
                ]);
                kernel.compute.setParameter('queryPos', queryPosBuf);
                kernel.compute.setParameter('queryIds', queryIdBuf);
                kernel.compute.setParameter('nodeSplatIdx', splatIdxBuf);
                kernel.compute.setParameter('nodePositions', positionsBuf);
                kernel.compute.setParameter('nodeChildren', childrenBuf);
                kernel.compute.setParameter('topDist', topDistBuf);
                kernel.compute.setParameter('topIdx', topIdxBuf);
                kernels.push(kernel);
            }
        }

        this.execute = async (
            queryPos: Float32Array,
            queryIds: Uint32Array,
            queryCount: number,
            outNeighbours: Uint32Array,
            homePart: number
        ) => {
            if (outNeighbours.length !== queryCount * k) {
                throw new Error(`GpuKnn: outNeighbours length ${outNeighbours.length} must be queryCount*k = ${queryCount * k}`);
            }

            // Home part first: it fills the top-K with true near neighbours,
            // so every other part's AABB test culls for interior queries.
            const order = [kernels[homePart], ...kernels.filter((_, i) => i !== homePart)];

            const numBatches = Math.ceil(queryCount / queriesPerBatch);
            for (let batch = 0; batch < numBatches; batch++) {
                const queryOffset = batch * queriesPerBatch;
                const batchCount = Math.min(queriesPerBatch, queryCount - queryOffset);
                const dispatchGroups = Math.ceil(batchCount / workgroupSize);

                queryPosBuf.write(0, queryPos, queryOffset * 3, batchCount * 3);
                queryIdBuf.write(0, queryIds, queryOffset, batchCount);

                resetKernel.compute.setParameter('slotCount', batchCount * k);
                resetKernel.compute.setupDispatch(Math.ceil((batchCount * k) / 256));
                for (const kernel of order) {
                    kernel.compute.setParameter('queryCount', batchCount);
                    kernel.compute.setupDispatch(dispatchGroups);
                }
                device.computeDispatch([resetKernel.compute, ...order.map(kn => kn.compute)], `knn-batch-${batch}`);

                const readBytes = batchCount * k * 4;
                await topIdxBuf.read(0, readBytes, outScratch, true);
                outNeighbours.set(outScratch.subarray(0, batchCount * k), queryOffset * k);
            }
        };

        this.destroy = () => {
            queryPosBuf.destroy();
            queryIdBuf.destroy();
            topDistBuf.destroy();
            topIdxBuf.destroy();
            for (const b of groupBufs) b.destroy();
            resetKernel.destroy();
            for (const kernel of kernels) kernel.destroy();
        };
    }
}

export { GpuKnn };
