import {
    BUFFERUSAGE_COPY_DST,
    BUFFERUSAGE_COPY_SRC,
    SHADERLANGUAGE_WGSL,
    SHADERSTAGE_COMPUTE,
    UNIFORMTYPE_UINT,
    BindGroupFormat,
    BindStorageBufferFormat,
    BindUniformBufferFormat,
    Compute,
    GraphicsDevice,
    Shader,
    StorageBuffer,
    UniformBufferFormat,
    UniformFormat
} from 'playcanvas';

import { Column, DataTable } from '../data-table/data-table';
import { KdTree } from '../spatial/kd-tree';

/**
 * WGSL kernel: iterative KD-tree K-nearest-neighbours.
 *
 * Each thread runs a depth-first traversal of the flattened KD-tree with a
 * fixed-size per-thread stack. Visits at most `O(K · log N)` nodes per
 * query thanks to the standard "skip the far subtree if its splitting plane
 * is farther than the current K-th best" pruning. Top-K is maintained
 * unsorted in per-thread storage with explicit worst-index tracking, so the
 * common-case "candidate is rejected against worst" path is a single
 * compare-and-branch (no dynamic-indexed shift).
 *
 * @param k - Compile-time K, the number of nearest neighbours per query.
 * @returns WGSL source.
 */
const knnWgsl = (k: number) => /* wgsl */`
struct Uniforms {
    queryOffset: u32,
    queryCount: u32,
    rootIdx: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
// Query positions (one per query splat).
@group(0) @binding(1) var<storage, read> px: array<f32>;
@group(0) @binding(2) var<storage, read> py: array<f32>;
@group(0) @binding(3) var<storage, read> pz: array<f32>;
// Flattened KD-tree.
@group(0) @binding(4) var<storage, read> nodeSplatIdx: array<u32>;
@group(0) @binding(5) var<storage, read> nodeX: array<f32>;
@group(0) @binding(6) var<storage, read> nodeY: array<f32>;
@group(0) @binding(7) var<storage, read> nodeZ: array<f32>;
@group(0) @binding(8) var<storage, read> nodeLeft: array<u32>;
@group(0) @binding(9) var<storage, read> nodeRight: array<u32>;
// Output: per query, k neighbour splat indices (unsorted).
@group(0) @binding(10) var<storage, read_write> outIndices: array<u32>;

const K: u32 = ${k}u;
const NULL_NODE: u32 = 0xFFFFFFFFu;
const F32_MAX: f32 = 3.4028234663852886e+38;
// log2(N) + slack — safe to ~2^40 nodes which is way past our limits.
const STACK_SIZE: u32 = 48u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let bid = gid.x;
    if (bid >= uniforms.queryCount) { return; }
    let q = bid + uniforms.queryOffset;

    let qx = px[q];
    let qy = py[q];
    let qz = pz[q];

    // Top-K state, unsorted. worstIdx points to the current K-th worst slot
    // so accepts replace it in O(1) and we recompute worst via a fixed loop.
    var topIdx: array<u32, ${k}>;
    var topDist: array<f32, ${k}>;
    var worst: f32 = F32_MAX;
    var worstIdx: u32 = 0u;
    for (var i: u32 = 0u; i < K; i++) {
        topDist[i] = F32_MAX;
        topIdx[i] = 0u;
    }

    // Stack: (nodeIdx, axis) packed as u32. axis ∈ {0,1,2} in top 2 bits,
    // nodeIdx in low 30 — supports up to ~1B nodes.
    var stack: array<u32, 48>;
    var sp: u32 = 0u;
    stack[0] = uniforms.rootIdx;   // axis=0 → no axis bits set
    sp = 1u;

    while (sp > 0u) {
        sp = sp - 1u;
        let packed = stack[sp];
        let nodeIdx = packed & 0x3FFFFFFFu;
        let axis = packed >> 30u;

        // Read the node's position + splat id.
        let nx = nodeX[nodeIdx];
        let ny = nodeY[nodeIdx];
        let nz = nodeZ[nodeIdx];
        let splatId = nodeSplatIdx[nodeIdx];

        // Update top-K, skipping the query itself.
        if (splatId != q) {
            let dx = nx - qx;
            let dy = ny - qy;
            let dz = nz - qz;
            let d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < worst) {
                topDist[worstIdx] = d2;
                topIdx[worstIdx] = splatId;
                // Recompute worst with a constant-bound loop (compiler can
                // unroll → all accesses to topDist resolve statically).
                var w: f32 = topDist[0];
                var wi: u32 = 0u;
                for (var i: u32 = 1u; i < K; i++) {
                    if (topDist[i] > w) { w = topDist[i]; wi = i; }
                }
                worst = w;
                worstIdx = wi;
            }
        }

        // Choose near/far children based on which side of the splitting
        // plane the query lies on. Walk near first (push far first so LIFO
        // pops near first), with pruning on far.
        var qAxisVal: f32;
        var nAxisVal: f32;
        if (axis == 0u) { qAxisVal = qx; nAxisVal = nx; }
        else if (axis == 1u) { qAxisVal = qy; nAxisVal = ny; }
        else { qAxisVal = qz; nAxisVal = nz; }

        let delta = qAxisVal - nAxisVal;
        let nextAxis = select(axis + 1u, 0u, axis + 1u >= 3u);
        let nextAxisPacked = nextAxis << 30u;

        let leftChild = nodeLeft[nodeIdx];
        let rightChild = nodeRight[nodeIdx];
        let near = select(rightChild, leftChild, delta < 0.0);
        let far = select(leftChild, rightChild, delta < 0.0);

        // Push far first iff its subtree could still hold a closer point
        // than the current K-th best.
        if (far != NULL_NODE && delta * delta < worst) {
            stack[sp] = far | nextAxisPacked;
            sp = sp + 1u;
        }
        if (near != NULL_NODE) {
            stack[sp] = near | nextAxisPacked;
            sp = sp + 1u;
        }
    }

    // Emit unsorted top-K (the decimator does not require sorted neighbours).
    let outBase = bid * K;
    for (var i: u32 = 0u; i < K; i++) {
        outIndices[outBase + i] = topIdx[i];
    }
}
`;

/**
 * GPU K-nearest-neighbours over a fixed point set using a flattened KD-tree.
 *
 * Algorithm: classic KD-tree DFS with bounded heap pruning, except the
 * recursion is unrolled into an explicit per-thread stack and the top-K is
 * maintained unsorted (with worst-index tracking) so the dominant
 * candidate-rejection path is a single compare. Same O(N log N) total work
 * as the CPU KD-tree the kernel mirrors, just parallelised across queries.
 *
 * Setup cost: O(N log N) for the CPU `KdTree` build + an O(N) DFS to
 * flatten into typed arrays.
 *
 * Memory footprint: ~24 N bytes for the flattened tree (3 floats + 3
 * u32 per node), plus query positions and the per-query output indices.
 */
class GpuKnn {
    /**
     * @param px - x coordinates of the N points (which double as queries).
     * @param py - y coordinates.
     * @param pz - z coordinates.
     * @param outNeighbours - destination for per-query K neighbour indices,
     *   length `N * k`. `outNeighbours[i * k + j]` is one of the k nearest
     *   neighbours of point i (UNSORTED). Excludes i itself.
     */
    execute: (
        px: Float32Array,
        py: Float32Array,
        pz: Float32Array,
        outNeighbours: Uint32Array
    ) => Promise<void>;
    destroy: () => void;

    /**
     * @param device - PlayCanvas GraphicsDevice (WebGPU).
     * @param maxN - Maximum number of points the index will handle.
     * @param k - Number of nearest neighbours per query.
     */
    constructor(device: GraphicsDevice, maxN: number, k: number) {
        const workgroupSize = 64;
        const queriesPerBatch = 1024 * workgroupSize;  // 65,536

        const bindGroupFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('px', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('py', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('pz', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('nodeSplatIdx', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('nodeX', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('nodeY', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('nodeZ', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('nodeLeft', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('nodeRight', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('outIndices', SHADERSTAGE_COMPUTE)
        ]);

        const shader = new Shader(device, {
            name: 'compute-knn-kdtree',
            shaderLanguage: SHADERLANGUAGE_WGSL,
            cshader: knnWgsl(k),
            // @ts-ignore
            computeUniformBufferFormats: {
                uniforms: new UniformBufferFormat(device, [
                    new UniformFormat('queryOffset', UNIFORMTYPE_UINT),
                    new UniformFormat('queryCount', UNIFORMTYPE_UINT),
                    new UniformFormat('rootIdx', UNIFORMTYPE_UINT)
                ])
            },
            // @ts-ignore
            computeBindGroupFormat: bindGroupFormat
        });

        const floatBufBytes = maxN * 4;
        const u32BufBytes = maxN * 4;
        const pxBuf = new StorageBuffer(device, floatBufBytes, BUFFERUSAGE_COPY_DST);
        const pyBuf = new StorageBuffer(device, floatBufBytes, BUFFERUSAGE_COPY_DST);
        const pzBuf = new StorageBuffer(device, floatBufBytes, BUFFERUSAGE_COPY_DST);
        const nSplatIdxBuf = new StorageBuffer(device, u32BufBytes, BUFFERUSAGE_COPY_DST);
        const nXBuf = new StorageBuffer(device, floatBufBytes, BUFFERUSAGE_COPY_DST);
        const nYBuf = new StorageBuffer(device, floatBufBytes, BUFFERUSAGE_COPY_DST);
        const nZBuf = new StorageBuffer(device, floatBufBytes, BUFFERUSAGE_COPY_DST);
        const nLeftBuf = new StorageBuffer(device, u32BufBytes, BUFFERUSAGE_COPY_DST);
        const nRightBuf = new StorageBuffer(device, u32BufBytes, BUFFERUSAGE_COPY_DST);

        const outBatchBytes = queriesPerBatch * k * 4;
        const outBuf = new StorageBuffer(
            device,
            outBatchBytes,
            BUFFERUSAGE_COPY_SRC | BUFFERUSAGE_COPY_DST
        );
        const outScratch = new Uint32Array(queriesPerBatch * k);

        const compute = new Compute(device, shader, 'compute-knn-kdtree');
        compute.setParameter('px', pxBuf);
        compute.setParameter('py', pyBuf);
        compute.setParameter('pz', pzBuf);
        compute.setParameter('nodeSplatIdx', nSplatIdxBuf);
        compute.setParameter('nodeX', nXBuf);
        compute.setParameter('nodeY', nYBuf);
        compute.setParameter('nodeZ', nZBuf);
        compute.setParameter('nodeLeft', nLeftBuf);
        compute.setParameter('nodeRight', nRightBuf);
        compute.setParameter('outIndices', outBuf);

        this.execute = async (
            px: Float32Array,
            py: Float32Array,
            pz: Float32Array,
            outNeighbours: Uint32Array
        ) => {
            const n = px.length;
            if (n > maxN) {
                throw new Error(`GpuKnn: N=${n} exceeds maxN=${maxN}`);
            }
            if (py.length !== n || pz.length !== n) {
                throw new Error('GpuKnn: px, py, pz must all have same length');
            }
            if (outNeighbours.length !== n * k) {
                throw new Error(`GpuKnn: outNeighbours length ${outNeighbours.length} must be N*k = ${n * k}`);
            }

            // Build the KD-tree on CPU. The existing `KdTree` constructor
            // accepts a DataTable with x/y/z columns first.
            const posTable = new DataTable([
                new Column('x', px),
                new Column('y', py),
                new Column('z', pz)
            ]);
            const tree = new KdTree(posTable);
            const flat = tree.flattenForGpu();

            // Upload everything in one go.
            pxBuf.write(0, px, 0, n);
            pyBuf.write(0, py, 0, n);
            pzBuf.write(0, pz, 0, n);
            nSplatIdxBuf.write(0, flat.nodeSplatIdx, 0, n);
            nXBuf.write(0, flat.nodeX, 0, n);
            nYBuf.write(0, flat.nodeY, 0, n);
            nZBuf.write(0, flat.nodeZ, 0, n);
            nLeftBuf.write(0, flat.nodeLeft, 0, n);
            nRightBuf.write(0, flat.nodeRight, 0, n);
            compute.setParameter('rootIdx', flat.rootIdx);

            const numBatches = Math.ceil(n / queriesPerBatch);
            for (let batch = 0; batch < numBatches; batch++) {
                const queryOffset = batch * queriesPerBatch;
                const queryCount = Math.min(queriesPerBatch, n - queryOffset);
                const groups = Math.ceil(queryCount / workgroupSize);

                compute.setParameter('queryOffset', queryOffset);
                compute.setParameter('queryCount', queryCount);

                compute.setupDispatch(groups);
                device.computeDispatch([compute], `knn-dispatch-${batch}`);

                const readBytes = queryCount * k * 4;
                await outBuf.read(0, readBytes, outScratch, true);
                outNeighbours.set(outScratch.subarray(0, queryCount * k), queryOffset * k);
            }
        };

        this.destroy = () => {
            pxBuf.destroy();
            pyBuf.destroy();
            pzBuf.destroy();
            nSplatIdxBuf.destroy();
            nXBuf.destroy();
            nYBuf.destroy();
            nZBuf.destroy();
            nLeftBuf.destroy();
            nRightBuf.destroy();
            outBuf.destroy();
            shader.destroy();
            bindGroupFormat.destroy();
        };
    }
}

export { GpuKnn };
