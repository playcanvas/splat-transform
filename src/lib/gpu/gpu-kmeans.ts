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
    FloatPacking,
    Shader,
    StorageBuffer,
    UniformBufferFormat,
    UniformFormat,
    GraphicsDevice
} from 'playcanvas';

/**
 * Flash-kmeans (arXiv 2603.09229) adapted to WebGPU: a fully GPU-resident
 * Lloyd loop. Per iteration, all stages are recorded with zero CPU
 * synchronization — FlashAssign (fused distance + online argmin over shared
 * memory centroid tiles), histogram, offset scan, counting-sort scatter
 * (the paper's sort-inverse mapping), segment reduction (one workgroup per
 * cluster, so the update needs no float atomics — WGSL has none), divide and
 * empty-cluster reseed. Labels and centroids are read back once, after the
 * final iteration.
 */

// points per thread in the assign kernel (register blocking): each thread
// keeps PPT points in registers and reuses every shared-memory centroid value
// across all of them, cutting shared-load traffic PPT×. Benched on M4 Max at
// d=45: ppt=2 is ~1.25× over ppt=1; ppt≥3 falls off a register-spill cliff
// (7× slower), so 2 is the cap outside the low-dimension case.
const pptForColumns = (numColumns: number) => {
    return numColumns <= 16 ? 4 : 2;
};

const roundUp = (value: number, multiple: number) => {
    return Math.ceil(value / multiple) * multiple;
};

// FlashAssign: streaming online argmin over centroid tiles staged through
// workgroup shared memory; the N×K distance matrix is never materialized.
// Threads are strided so the p-th point of the whole workgroup block is
// contiguous in memory (coalesced loads). The distance loop is scalar with a
// compile-time trip count — explicit vec4 widening benched slower (the kernel
// is ALU-bound; the compiler already vectorizes this loop).
const assignWgsl = (floatType: string, numColumns: number, tileSize: number, ppt: number) => /* wgsl */`
${floatType === 'f16' ? 'enable f16;' : ''}

struct Uniforms {
    localOffset: u32,       // first point of this dispatch, chunk-relative
    globalOffset: u32,      // first point of this dispatch, global (labels index)
    count: u32,             // points in this dispatch
    numCentroids: u32
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> points: array<${floatType}>;
@group(0) @binding(2) var<storage, read> centroids: array<${floatType}>;
@group(0) @binding(3) var<storage, read_write> labels: array<u32>;

const numColumns = ${numColumns}u;
const tileSize = ${tileSize}u;          // centroid rows per shared-memory tile, multiple of 64
const ppt = ${ppt}u;                    // points per thread
const F32_MAX: f32 = 3.4028234663852886e+38;

var<workgroup> tile: array<${floatType}, ${numColumns * tileSize}>;

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_index) local_id: u32,
    @builtin(workgroup_id) workgroup_id: vec3u
) {
    let blockBase = workgroup_id.x * (64u * ppt);

    // copy this thread's points into registers (strided: point p of this
    // thread is blockBase + p*64 + local_id)
    var pts: array<${floatType}, ${numColumns * ppt}>;
    for (var p = 0u; p < ppt; p++) {
        let pi = blockBase + p * 64u + local_id;
        if (pi < uniforms.count) {
            let src = (uniforms.localOffset + pi) * numColumns;
            for (var j = 0u; j < numColumns; j++) {
                pts[p * numColumns + j] = points[src + j];
            }
        }
    }

    var mind: array<f32, ${ppt}>;
    var mini: array<u32, ${ppt}>;
    for (var p = 0u; p < ppt; p++) {
        mind[p] = F32_MAX;
        mini[p] = 0u;
    }

    // work through the centroids in shared memory tiles
    let numTiles = (uniforms.numCentroids + tileSize - 1u) / tileSize;
    for (var t = 0u; t < numTiles; t++) {
        let tileBase = t * tileSize;

        // cooperative load: each thread copies tileSize/64 consecutive rows
        let rowsPerThread = tileSize / 64u;
        for (var r = 0u; r < rowsPerThread; r++) {
            let localRow = local_id * rowsPerThread + r;
            let srcRow = tileBase + localRow;
            if (srcRow < uniforms.numCentroids) {
                let src = srcRow * numColumns;
                let dst = localRow * numColumns;
                for (var j = 0u; j < numColumns; j++) {
                    tile[dst + j] = centroids[src + j];
                }
            }
        }
        workgroupBarrier();

        // online argmin: each centroid row is reused across all ppt points
        let cnt = min(tileSize, uniforms.numCentroids - tileBase);
        for (var c = 0u; c < cnt; c++) {
            let cb = c * numColumns;
            for (var p = 0u; p < ppt; p++) {
                var d = 0.0;
                let pb = p * numColumns;
                for (var j = 0u; j < numColumns; j++) {
                    let v = f32(pts[pb + j] - tile[cb + j]);
                    d += v * v;
                }
                if (d < mind[p]) {
                    mind[p] = d;
                    mini[p] = tileBase + c;
                }
            }
        }
        workgroupBarrier();
    }

    for (var p = 0u; p < ppt; p++) {
        let pi = blockBase + p * 64u + local_id;
        if (pi < uniforms.count) {
            labels[uniforms.globalOffset + pi] = mini[p];
        }
    }
}
`;

// per-cluster population count. Plain u32 atomics — the paper's contention
// concern applies to d-wide float scatters, not a 1-word histogram.
const histogramWgsl = () => /* wgsl */`
struct Uniforms {
    globalOffset: u32,
    count: u32
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> labels: array<u32>;
@group(0) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x < uniforms.count) {
        atomicAdd(&counts[labels[uniforms.globalOffset + gid.x]], 1u);
    }
}
`;

// single-workgroup exclusive scan of counts → segment offsets, plus a working
// copy for the scatter cursors (same 3-phase pattern as shaders/prefix-sum.ts).
const scanWgsl = () => /* wgsl */`
struct Uniforms {
    numCentroids: u32
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> cursors: array<u32>;

const SCAN_THREADS = 256u;

var<workgroup> scratch: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32) {
    let n = uniforms.numCentroids;
    let perThread = (n + SCAN_THREADS - 1u) / SCAN_THREADS;
    let base = tid * perThread;
    let end = min(base + perThread, n);

    var partial = 0u;
    for (var i = base; i < end; i++) {
        partial += counts[i];
    }
    scratch[tid] = partial;
    workgroupBarrier();

    if (tid == 0u) {
        var acc = 0u;
        for (var i = 0u; i < SCAN_THREADS; i++) {
            let v = scratch[i];
            scratch[i] = acc;
            acc += v;
        }
    }
    workgroupBarrier();

    var prefix = scratch[tid];
    for (var i = base; i < end; i++) {
        offsets[i] = prefix;
        cursors[i] = prefix;
        prefix += counts[i];
    }
}
`;

// counting-sort placement — the paper's sort-inverse mapping. Within-segment
// order is nondeterministic (atomic race), which only permutes float
// summation order in the reduce.
const scatterWgsl = () => /* wgsl */`
struct Uniforms {
    globalOffset: u32,
    count: u32
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> labels: array<u32>;
@group(0) @binding(2) var<storage, read_write> cursors: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> sortedIdx: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x < uniforms.count) {
        let i = uniforms.globalOffset + gid.x;
        let pos = atomicAdd(&cursors[labels[i]], 1u);
        sortedIdx[pos] = i;
    }
}
`;

// segment reduction: one workgroup owns one cluster, so partial sums
// accumulate in shared memory and land in sums[] with plain (non-atomic)
// adds — dispatch ordering serializes the per-chunk accumulation. Points
// outside the bound chunk are skipped and picked up by that chunk's dispatch.
const reduceWgsl = (floatType: string, numColumns: number) => /* wgsl */`
${floatType === 'f16' ? 'enable f16;' : ''}

struct Uniforms {
    numCentroids: u32,
    chunkStart: u32,
    chunkEnd: u32
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> points: array<${floatType}>;
@group(0) @binding(2) var<storage, read> sortedIdx: array<u32>;
@group(0) @binding(3) var<storage, read> offsets: array<u32>;
@group(0) @binding(4) var<storage, read> counts: array<u32>;
@group(0) @binding(5) var<storage, read_write> sums: array<f32>;

const numColumns = ${numColumns}u;

var<workgroup> partials: array<f32, ${numColumns * 64}>;

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_index) local_id: u32,
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u
) {
    // 2D dispatch: K can exceed the 65535 per-dimension workgroup limit
    let cluster = workgroup_id.x + workgroup_id.y * num_workgroups.x;
    if (cluster >= uniforms.numCentroids) {
        return;
    }

    let count = counts[cluster];
    let begin = offsets[cluster];

    // threads stride the cluster's segment of the inverse mapping
    var acc: array<f32, ${numColumns}>;
    for (var s = local_id; s < count; s += 64u) {
        let gi = sortedIdx[begin + s];
        if (gi >= uniforms.chunkStart && gi < uniforms.chunkEnd) {
            let src = (gi - uniforms.chunkStart) * numColumns;
            for (var j = 0u; j < numColumns; j++) {
                acc[j] += f32(points[src + j]);
            }
        }
    }
    for (var j = 0u; j < numColumns; j++) {
        partials[local_id * numColumns + j] = acc[j];
    }
    workgroupBarrier();

    // cross-thread reduce: thread j serially sums column j over the 64 partials
    for (var j = local_id; j < numColumns; j += 64u) {
        var s = 0.0;
        for (var t = 0u; t < 64u; t++) {
            s += partials[t * numColumns + j];
        }
        sums[cluster * numColumns + j] += s;
    }
}
`;

// centroid = sum / count for non-empty clusters (thread per element)
const divideWgsl = (numColumns: number) => /* wgsl */`
struct Uniforms {
    numCentroids: u32
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> sums: array<f32>;
@group(0) @binding(2) var<storage, read> counts: array<u32>;
@group(0) @binding(3) var<storage, read_write> centroids: array<f32>;

const numColumns = ${numColumns}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x < uniforms.numCentroids * numColumns) {
        let count = counts[gid.x / numColumns];
        if (count > 0u) {
            centroids[gid.x] = sums[gid.x] / f32(count);
        }
    }
}
`;

// empty clusters re-seed to a pseudo-random point (PCG hash of cluster id and
// a per-iteration seed), preserving the CPU path's semantics. The dispatch for
// chunk c only handles clusters whose hashed point lands in chunk c; disjoint
// from divide (count == 0 vs count > 0), so ordering between them is free.
const reseedWgsl = (floatType: string, numColumns: number) => /* wgsl */`
${floatType === 'f16' ? 'enable f16;' : ''}

struct Uniforms {
    numCentroids: u32,
    numPoints: u32,
    seed: u32,
    chunkStart: u32,
    chunkEnd: u32
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(2) var<storage, read> points: array<${floatType}>;
@group(0) @binding(3) var<storage, read_write> centroids: array<f32>;

const numColumns = ${numColumns}u;

fn pcgHash(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let cluster = gid.x;
    if (cluster < uniforms.numCentroids && counts[cluster] == 0u) {
        let p = pcgHash(cluster ^ uniforms.seed) % uniforms.numPoints;
        if (p >= uniforms.chunkStart && p < uniforms.chunkEnd) {
            let src = (p - uniforms.chunkStart) * numColumns;
            let dst = cluster * numColumns;
            for (var j = 0u; j < numColumns; j++) {
                centroids[dst + j] = f32(points[src + j]);
            }
        }
    }
}
`;

// mirror the canonical f32 centroids into the f16 copy read by assign
const convertWgsl = () => /* wgsl */`
enable f16;

struct Uniforms {
    totalElems: u32
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> centroids: array<f32>;
@group(0) @binding(2) var<storage, read_write> centroidsF16: array<f16>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x < uniforms.totalElems) {
        centroidsF16[gid.x] = f16(centroids[gid.x]);
    }
}
`;

type Kernel = {
    compute: Compute;
    destroy: () => void;
};

// shader + bind group format + compute boilerplate shared by all kernels
const makeKernel = (
    device: GraphicsDevice,
    name: string,
    source: string,
    uniformNames: string[],
    storageBindings: [string, boolean][]    // [name, readOnly]
): Kernel => {
    const bindGroupFormat = new BindGroupFormat(device, [
        new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
        ...storageBindings.map(([bname, readOnly]) => new BindStorageBufferFormat(bname, SHADERSTAGE_COMPUTE, readOnly))
    ]);

    const shader = new Shader(device, {
        name,
        shaderLanguage: SHADERLANGUAGE_WGSL,
        cshader: source,
        // @ts-ignore
        computeUniformBufferFormats: {
            uniforms: new UniformBufferFormat(device, uniformNames.map(u => new UniformFormat(u, UNIFORMTYPE_UINT)))
        },
        // @ts-ignore
        computeBindGroupFormat: bindGroupFormat
    });

    return {
        compute: new Compute(device, shader, name),
        destroy: () => {
            shader.destroy();
            bindGroupFormat.destroy();
        }
    };
};

class GpuKmeans {
    /**
     * Run the full Lloyd loop on the GPU. `points` is row-major interleaved
     * (numPoints×numColumns), `centroids` (numCentroids×numColumns) provides
     * the seeds and receives the final centroids, `labels` receives the final
     * assignment. Semantics match the CPU loop: iterations × (assign →
     * update), returning the last assign's labels and the post-update
     * centroids. `onIteration` fires as each iteration is submitted.
     * Requires numPoints >= numCentroids (callers early-out otherwise).
     */
    run: (
        points: Float32Array,
        numPoints: number,
        centroids: Float32Array,
        labels: Uint32Array,
        iterations: number,
        onIteration?: () => void
    ) => Promise<void>;
    destroy: () => void;

    constructor(device: GraphicsDevice, numColumns: number, numCentroids: number) {
        const useF16 = !!('supportsShaderF16' in device && device.supportsShaderF16);
        const floatType = useF16 ? 'f16' : 'f32';
        const bytesPerElem = useF16 ? 2 : 4;

        // @ts-ignore - wgpu is private on WebgpuGraphicsDevice but exposed in practice
        const wgpuLimits = (device as any).wgpu?.limits;

        // flush recorded work to the queue without any CPU synchronization
        const submit = () => (device as unknown as { submit: () => void }).submit();

        // centroid tile: large tiles amortize shared-memory loads but the
        // tile is the occupancy limiter — filling the 32KB workgroup-storage
        // limit leaves ~2 workgroups per core and starves the GPU of latency
        // hiding (measured 2× slower at d=45). Budget ~12KB (128 rows at
        // 45 columns f16, like the previous fixed tile) and shrink when a
        // row is too wide (the previous fixed 128 needed >16KB at 45
        // columns f32 — over the WebGPU minimum).
        const wgStorage: number = wgpuLimits?.maxComputeWorkgroupStorageSize ?? 16384;
        const tileBudget = Math.min(wgStorage, 12 * 1024);
        const tileSize = Math.max(64, Math.min(128, Math.floor(tileBudget / (numColumns * bytesPerElem * 64)) * 64));
        const ppt = pptForColumns(numColumns);

        const kernels = {
            assign: makeKernel(device, 'kmeans-assign', assignWgsl(floatType, numColumns, tileSize, ppt),
                ['localOffset', 'globalOffset', 'count', 'numCentroids'],
                [['points', true], ['centroids', true], ['labels', false]]),
            histogram: makeKernel(device, 'kmeans-histogram', histogramWgsl(),
                ['globalOffset', 'count'],
                [['labels', true], ['counts', false]]),
            scan: makeKernel(device, 'kmeans-scan', scanWgsl(),
                ['numCentroids'],
                [['counts', true], ['offsets', false], ['cursors', false]]),
            scatter: makeKernel(device, 'kmeans-scatter', scatterWgsl(),
                ['globalOffset', 'count'],
                [['labels', true], ['cursors', false], ['sortedIdx', false]]),
            reduce: makeKernel(device, 'kmeans-reduce', reduceWgsl(floatType, numColumns),
                ['numCentroids', 'chunkStart', 'chunkEnd'],
                [['points', true], ['sortedIdx', true], ['offsets', true], ['counts', true], ['sums', false]]),
            divide: makeKernel(device, 'kmeans-divide', divideWgsl(numColumns),
                ['numCentroids'],
                [['sums', true], ['counts', true], ['centroids', false]]),
            reseed: makeKernel(device, 'kmeans-reseed', reseedWgsl(floatType, numColumns),
                ['numCentroids', 'numPoints', 'seed', 'chunkStart', 'chunkEnd'],
                [['counts', true], ['points', true], ['centroids', false]]),
            convert: useF16 ? makeKernel(device, 'kmeans-convert', convertWgsl(),
                ['totalElems'],
                [['centroids', true], ['centroidsF16', false]]) : null
        };

        this.run = async (points, numPoints, centroids, labels, iterations, onIteration) => {
            // ---- chunk plan: each point chunk must fit a storage binding and
            // keep assign a 1D dispatch (65535 workgroups × 64 threads)
            const rowBytes = numColumns * bytesPerElem;
            const bindingLimit: number = wgpuLimits?.maxStorageBufferBindingSize ?? (128 * 1024 * 1024);
            const chunkCap = Math.min(4_000_000, Math.floor(bindingLimit / rowBytes)) & ~1;
            const numChunks = Math.ceil(numPoints / chunkCap);
            // resident when all chunks fit a 2GB budget, else stream through
            // two ping-pong buffers re-uploaded every iteration
            const resident = numPoints * rowBytes <= 2 * 1024 * 1024 * 1024;
            const chunkRows = (c: number) => Math.min(chunkCap, numPoints - c * chunkCap);

            // pack f16 once up front — uploads (including streaming re-uploads)
            // are then pure copies
            const packed = useF16 ? new Uint16Array(roundUp(numPoints * numColumns, 2)) : points;
            if (useF16) {
                const p16 = packed as Uint16Array;
                for (let i = 0, n = numPoints * numColumns; i < n; ++i) {
                    p16[i] = FloatPacking.float2Half(points[i]);
                }
            }

            // ---- buffers
            const pointBufs = resident ?
                Array.from({ length: numChunks }, (_, c) => new StorageBuffer(device, roundUp(chunkRows(c) * numColumns, 2) * bytesPerElem, BUFFERUSAGE_COPY_DST)) :
                Array.from({ length: 2 }, () => new StorageBuffer(device, roundUp(chunkCap * numColumns, 2) * bytesPerElem, BUFFERUSAGE_COPY_DST));
            const centroidsBuf = new StorageBuffer(device, numCentroids * numColumns * 4, BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC);
            const centroidsF16Buf = useF16 ? new StorageBuffer(device, roundUp(numCentroids * numColumns, 2) * 2, 0) : null;
            const labelsBuf = new StorageBuffer(device, numPoints * 4, BUFFERUSAGE_COPY_SRC);
            const sortedIdxBuf = new StorageBuffer(device, numPoints * 4, 0);
            const countsBuf = new StorageBuffer(device, numCentroids * 4, BUFFERUSAGE_COPY_DST);
            const offsetsBuf = new StorageBuffer(device, numCentroids * 4, 0);
            const cursorsBuf = new StorageBuffer(device, numCentroids * 4, 0);
            const sumsBuf = new StorageBuffer(device, numCentroids * numColumns * 4, BUFFERUSAGE_COPY_DST);

            try {
                // ---- static parameters
                const { assign, histogram, scan, scatter, reduce, divide, reseed, convert } = kernels;
                assign.compute.setParameter('centroids', useF16 ? centroidsF16Buf! : centroidsBuf);
                assign.compute.setParameter('labels', labelsBuf);
                assign.compute.setParameter('numCentroids', numCentroids);
                histogram.compute.setParameter('labels', labelsBuf);
                histogram.compute.setParameter('counts', countsBuf);
                scan.compute.setParameter('counts', countsBuf);
                scan.compute.setParameter('offsets', offsetsBuf);
                scan.compute.setParameter('cursors', cursorsBuf);
                scan.compute.setParameter('numCentroids', numCentroids);
                scatter.compute.setParameter('labels', labelsBuf);
                scatter.compute.setParameter('cursors', cursorsBuf);
                scatter.compute.setParameter('sortedIdx', sortedIdxBuf);
                reduce.compute.setParameter('sortedIdx', sortedIdxBuf);
                reduce.compute.setParameter('offsets', offsetsBuf);
                reduce.compute.setParameter('counts', countsBuf);
                reduce.compute.setParameter('sums', sumsBuf);
                reduce.compute.setParameter('numCentroids', numCentroids);
                divide.compute.setParameter('sums', sumsBuf);
                divide.compute.setParameter('counts', countsBuf);
                divide.compute.setParameter('centroids', centroidsBuf);
                divide.compute.setParameter('numCentroids', numCentroids);
                reseed.compute.setParameter('counts', countsBuf);
                reseed.compute.setParameter('centroids', centroidsBuf);
                reseed.compute.setParameter('numCentroids', numCentroids);
                reseed.compute.setParameter('numPoints', numPoints);
                convert?.compute.setParameter('centroids', centroidsBuf);
                convert?.compute.setParameter('centroidsF16', centroidsF16Buf);
                convert?.compute.setParameter('totalElems', numCentroids * numColumns);

                const uploadChunk = (buf: StorageBuffer, c: number) => {
                    buf.write(0, packed, c * chunkCap * numColumns, roundUp(chunkRows(c) * numColumns, useF16 ? 2 : 1));
                };

                const dispatchConvert = () => {
                    if (convert) {
                        convert.compute.setupDispatch(Math.ceil((numCentroids * numColumns) / 256));
                        device.computeDispatch([convert.compute], 'kmeans-convert');
                    }
                };

                // ---- one-time uploads
                centroidsBuf.write(0, centroids, 0, numCentroids * numColumns);
                dispatchConvert();
                if (resident) {
                    for (let c = 0; c < numChunks; c++) {
                        uploadChunk(pointBufs[c], c);
                    }
                }

                // assign sub-dispatch size: keeps each command buffer's GPU time
                // modest (watchdog margin) without any CPU synchronization
                const assignSub = 512 * 1024;

                for (let iter = 0; iter < iterations; iter++) {
                    countsBuf.clear();
                    sumsBuf.clear();

                    // FlashAssign per chunk (streaming re-uploads into ping-pong
                    // buffers; the submit between chunks keeps queue-ordered
                    // writes from racing ahead of recorded dispatches)
                    for (let c = 0; c < numChunks; c++) {
                        const buf = pointBufs[resident ? c : c % 2];
                        if (!resident) {
                            uploadChunk(buf, c);
                        }
                        const rows = chunkRows(c);
                        for (let s = 0; s < rows; s += assignSub) {
                            const count = Math.min(assignSub, rows - s);
                            assign.compute.setParameter('points', buf);
                            assign.compute.setParameter('localOffset', s);
                            assign.compute.setParameter('globalOffset', c * chunkCap + s);
                            assign.compute.setParameter('count', count);
                            assign.compute.setupDispatch(Math.ceil(count / (64 * ppt)));
                            device.computeDispatch([assign.compute], 'kmeans-assign');
                            submit();
                        }
                    }

                    // histogram → scan → scatter (labels-only, no point data).
                    // NOTE: re-dispatching one Compute with different uniform
                    // values only takes effect across a submit boundary — the
                    // engine's dynamic uniform-buffer bind group is not
                    // re-versioned within a submit, so back-to-back dispatches
                    // share one uniform snapshot. Every uniform-varying loop
                    // here therefore submits per dispatch (assign and reduce
                    // already do).
                    for (let c = 0; c < numChunks; c++) {
                        histogram.compute.setParameter('globalOffset', c * chunkCap);
                        histogram.compute.setParameter('count', chunkRows(c));
                        histogram.compute.setupDispatch(Math.ceil(chunkRows(c) / 256));
                        device.computeDispatch([histogram.compute], 'kmeans-histogram');
                        submit();
                    }
                    scan.compute.setupDispatch(1);
                    device.computeDispatch([scan.compute], 'kmeans-scan');
                    for (let c = 0; c < numChunks; c++) {
                        scatter.compute.setParameter('globalOffset', c * chunkCap);
                        scatter.compute.setParameter('count', chunkRows(c));
                        scatter.compute.setupDispatch(Math.ceil(chunkRows(c) / 256));
                        device.computeDispatch([scatter.compute], 'kmeans-scatter');
                        submit();
                    }

                    // sort-inverse update per chunk, then divide + reseed + mirror
                    const seed = Math.floor(Math.random() * 0xffffffff) >>> 0;
                    for (let c = 0; c < numChunks; c++) {
                        const buf = pointBufs[resident ? c : c % 2];
                        if (!resident) {
                            uploadChunk(buf, c);
                        }
                        const groups = numCentroids;
                        const y = Math.ceil(groups / 65535);
                        reduce.compute.setParameter('points', buf);
                        reduce.compute.setParameter('chunkStart', c * chunkCap);
                        reduce.compute.setParameter('chunkEnd', c * chunkCap + chunkRows(c));
                        reduce.compute.setupDispatch(Math.ceil(groups / y), y);
                        device.computeDispatch([reduce.compute], 'kmeans-reduce');
                        reseed.compute.setParameter('points', buf);
                        reseed.compute.setParameter('seed', seed);
                        reseed.compute.setParameter('chunkStart', c * chunkCap);
                        reseed.compute.setParameter('chunkEnd', c * chunkCap + chunkRows(c));
                        reseed.compute.setupDispatch(Math.ceil(numCentroids / 256));
                        device.computeDispatch([reseed.compute], 'kmeans-reseed');
                        submit();
                    }
                    divide.compute.setupDispatch(Math.ceil((numCentroids * numColumns) / 256));
                    device.computeDispatch([divide.compute], 'kmeans-divide');
                    dispatchConvert();
                    submit();

                    onIteration?.();
                }

                // ---- single blocking readback after the final iteration
                await labelsBuf.read(0, numPoints * 4, labels, true);
                await centroidsBuf.read(0, numCentroids * numColumns * 4, centroids, true);
            } finally {
                pointBufs.forEach(b => b.destroy());
                centroidsBuf.destroy();
                centroidsF16Buf?.destroy();
                labelsBuf.destroy();
                sortedIdxBuf.destroy();
                countsBuf.destroy();
                offsetsBuf.destroy();
                cursorsBuf.destroy();
                sumsBuf.destroy();
            }
        };

        this.destroy = () => {
            Object.values(kernels).forEach(k => k?.destroy());
        };
    }
}

export { GpuKmeans };
