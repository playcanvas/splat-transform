import type { Compute, GraphicsDevice } from 'playcanvas';
import { BUFFERUSAGE_COPY_DST, BUFFERUSAGE_COPY_SRC, FloatPacking, StorageBuffer } from 'playcanvas';

import { makeKernel } from './compute-kernel';
import type { Kernel } from './compute-kernel';

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

// Each assign sub-dispatch is its own command buffer, and macOS kills command
// buffers that run too long (the GPU watchdog). A sub-dispatch covers up to
// ASSIGN_SUB points (enough workgroups to fill a large GPU) against a range of
// centroids sized so points × centroids × columns stays within ASSIGN_WORK:
// ~10 ms on an M4 Max (measured ~3.6e12 per second at 64K × 45), roughly 50 ms
// on an 8-core M1. Splitting the centroids rather than the points keeps the
// GPU occupied (fewer points per dispatch measured 2× slower on an M4 Max).
// Small codebooks cover all centroids in one range.
const ASSIGN_WORK = 2 ** 35;
const ASSIGN_SUB = 512 * 1024;

// FlashAssign: streaming online argmin over centroid tiles staged through
// workgroup shared memory; the N×K distance matrix is never materialized.
// Threads are strided so the p-th point of the whole workgroup block is
// contiguous in memory (coalesced loads). The distance loop is scalar with a
// compile-time trip count — explicit vec4 widening benched slower (the kernel
// is ALU-bound; the compiler already vectorizes this loop).
const assignWgsl = (floatType: string, numColumns: number, tileSize: number, ppt: number) => /* wgsl */ `
${floatType === 'f16' ? 'enable f16;' : ''}

struct Uniforms {
    localOffset: u32,       // first point of this dispatch, chunk-relative
    globalOffset: u32,      // first point of this dispatch, global (labels index)
    count: u32,             // points in this dispatch
    numCentroids: u32,
    centroidBegin: u32,     // centroid range of this dispatch, tile aligned
    centroidEnd: u32
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> points: array<${floatType}>;
@group(0) @binding(2) var<storage, read> centroids: array<${floatType}>;
@group(0) @binding(3) var<storage, read_write> labels: array<u32>;
@group(0) @binding(4) var<storage, read_write> bestDist: array<f32>;    // carried between centroid ranges

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

    // resume the argmin from the previous centroid range
    var mind: array<f32, ${ppt}>;
    var mini: array<u32, ${ppt}>;
    for (var p = 0u; p < ppt; p++) {
        mind[p] = F32_MAX;
        mini[p] = 0u;
        let pi = blockBase + p * 64u + local_id;
        if (uniforms.centroidBegin > 0u && pi < uniforms.count) {
            mind[p] = bestDist[uniforms.globalOffset + pi];
            mini[p] = labels[uniforms.globalOffset + pi];
        }
    }

    // work through this dispatch's centroids in shared memory tiles
    let numTiles = (uniforms.centroidEnd - uniforms.centroidBegin + tileSize - 1u) / tileSize;
    for (var t = 0u; t < numTiles; t++) {
        let tileBase = uniforms.centroidBegin + t * tileSize;

        // cooperative load: each thread copies tileSize/64 consecutive rows
        let rowsPerThread = tileSize / 64u;
        for (var r = 0u; r < rowsPerThread; r++) {
            let localRow = local_id * rowsPerThread + r;
            let srcRow = tileBase + localRow;
            if (srcRow < uniforms.centroidEnd) {
                let src = srcRow * numColumns;
                let dst = localRow * numColumns;
                for (var j = 0u; j < numColumns; j++) {
                    tile[dst + j] = centroids[src + j];
                }
            }
        }
        workgroupBarrier();

        // online argmin: each centroid row is reused across all ppt points
        let cnt = min(tileSize, uniforms.centroidEnd - tileBase);
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
            if (uniforms.centroidEnd < uniforms.numCentroids) {
                bestDist[uniforms.globalOffset + pi] = mind[p];
            }
        }
    }
}
`;

// f32 FlashAssign. Two changes over the scalar form, measured 1.85× on an
// A10G (1M points, d=45, K=65536, labels identical): rows are widened to
// zero-padded vec4 in registers and shared memory (4-wide shared loads), and
// the distance is expanded as |c|² - 2x·c — |x|² is constant per point, so the
// argmin is unchanged — one FMA per element instead of a subtract and an FMA.
// Each tile row's |c|² is summed while the tile is staged. The expanded form's
// rounding error scales with |x|² rather than with the distance, so points and
// centroids are taken relative to a reference near the data (`center`): the
// error then follows the data's spread, not its offset from the origin.
const assignVec4Wgsl = (numColumns: number, tileSize: number, ppt: number, workgroupSize: number) => /* wgsl */ `
struct Uniforms {
    localOffset: u32,       // first point of this dispatch, chunk-relative
    globalOffset: u32,      // first point of this dispatch, global (labels index)
    count: u32,             // points in this dispatch
    numCentroids: u32,
    centroidBegin: u32,     // centroid range of this dispatch, tile aligned
    centroidEnd: u32
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> points: array<f32>;
@group(0) @binding(2) var<storage, read> centroids: array<f32>;
@group(0) @binding(3) var<storage, read_write> labels: array<u32>;
@group(0) @binding(4) var<storage, read> center: array<vec4f>;  // zero-padded reference row
@group(0) @binding(5) var<storage, read_write> bestDist: array<f32>;    // carried between centroid ranges

const numColumns = ${numColumns}u;
const quads = ${Math.ceil(numColumns / 4)}u;    // vec4s per padded row
const tileSize = ${tileSize}u;
const ppt = ${ppt}u;
const workgroupSize = ${workgroupSize}u;
const F32_MAX: f32 = 3.4028234663852886e+38;

var<workgroup> tile: array<vec4f, ${Math.ceil(numColumns / 4) * tileSize}>;
var<workgroup> tileNorm: array<f32, ${tileSize}>;

// quad q of the row starting at base, relative to the reference row and zero
// past the end of the row. The centering stays inside the bounds test so the
// padding lanes remain compile-time zero (their FMAs fold away; centering
// them at runtime measured 7% slower at d=45).
fn pointQuad(base: u32, q: u32) -> vec4f {
    var v = vec4f(0.0);
    for (var e = 0u; e < 4u; e++) {
        if (q * 4u + e < numColumns) {
            v[e] = points[base + q * 4u + e] - center[q][e];
        }
    }
    return v;
}

fn centroidQuad(base: u32, q: u32) -> vec4f {
    var v = vec4f(0.0);
    for (var e = 0u; e < 4u; e++) {
        if (q * 4u + e < numColumns) {
            v[e] = centroids[base + q * 4u + e] - center[q][e];
        }
    }
    return v;
}

@compute @workgroup_size(${workgroupSize})
fn main(
    @builtin(local_invocation_index) local_id: u32,
    @builtin(workgroup_id) workgroup_id: vec3u
) {
    let blockBase = workgroup_id.x * (workgroupSize * ppt);

    // copy this thread's points into registers (strided: point p of this
    // thread is blockBase + p*workgroupSize + local_id)
    var pts: array<vec4f, ${Math.ceil(numColumns / 4) * ppt}>;
    for (var p = 0u; p < ppt; p++) {
        let pi = blockBase + p * workgroupSize + local_id;
        if (pi < uniforms.count) {
            let src = (uniforms.localOffset + pi) * numColumns;
            for (var q = 0u; q < quads; q++) {
                pts[p * quads + q] = pointQuad(src, q);
            }
        }
    }

    // resume the argmin from the previous centroid range
    var mind: array<f32, ${ppt}>;
    var mini: array<u32, ${ppt}>;
    for (var p = 0u; p < ppt; p++) {
        mind[p] = F32_MAX;
        mini[p] = 0u;
        let pi = blockBase + p * workgroupSize + local_id;
        if (uniforms.centroidBegin > 0u && pi < uniforms.count) {
            mind[p] = bestDist[uniforms.globalOffset + pi];
            mini[p] = labels[uniforms.globalOffset + pi];
        }
    }

    let numTiles = (uniforms.centroidEnd - uniforms.centroidBegin + tileSize - 1u) / tileSize;
    for (var t = 0u; t < numTiles; t++) {
        let tileBase = uniforms.centroidBegin + t * tileSize;

        // stage the tile one row per thread, summing |c|² on the way
        for (var r = local_id; r < tileSize; r += workgroupSize) {
            let row = tileBase + r;
            var norm = 0.0;
            if (row < uniforms.centroidEnd) {
                let src = row * numColumns;
                for (var q = 0u; q < quads; q++) {
                    let c = centroidQuad(src, q);
                    tile[r * quads + q] = c;
                    norm += dot(c, c);
                }
            }
            tileNorm[r] = norm;
        }
        workgroupBarrier();

        let cnt = min(tileSize, uniforms.centroidEnd - tileBase);
        for (var c = 0u; c < cnt; c++) {
            let cb = c * quads;
            let norm = tileNorm[c];
            for (var p = 0u; p < ppt; p++) {
                var s = 0.0;
                let pb = p * quads;
                for (var q = 0u; q < quads; q++) {
                    s += dot(pts[pb + q], tile[cb + q]);
                }
                let d = norm - 2.0 * s;
                if (d < mind[p]) {
                    mind[p] = d;
                    mini[p] = tileBase + c;
                }
            }
        }
        workgroupBarrier();
    }

    for (var p = 0u; p < ppt; p++) {
        let pi = blockBase + p * workgroupSize + local_id;
        if (pi < uniforms.count) {
            labels[uniforms.globalOffset + pi] = mini[p];
            if (uniforms.centroidEnd < uniforms.numCentroids) {
                bestDist[uniforms.globalOffset + pi] = mind[p];
            }
        }
    }
}
`;

// per-cluster population count. Plain u32 atomics — the paper's contention
// concern applies to d-wide float scatters, not a 1-word histogram.
const histogramWgsl = () => /* wgsl */ `
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
const scanWgsl = () => /* wgsl */ `
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
const scatterWgsl = () => /* wgsl */ `
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
// Columns per cross-thread reduction pass: 64 threads x 48 f32 = 12KB of
// workgroup storage, inside the 16KB WebGPU minimum for any column count
// (72 columns at SH band 4 would need 18KB in one pass).
const REDUCE_COLUMN_BLOCK = 48;

// `weighted` (e-means): each point's contribution is scaled by its weight and
// the cluster's weight total lands in weightSums.
const reduceWgsl = (floatType: string, numColumns: number, weighted = false) => /* wgsl */ `
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
${
    weighted
        ? `@group(0) @binding(6) var<storage, read> weights: array<f32>;
@group(0) @binding(7) var<storage, read_write> weightSums: array<f32>;`
        : ''
}

const numColumns = ${numColumns}u;
const block = ${Math.min(numColumns, REDUCE_COLUMN_BLOCK)}u;

var<workgroup> partials: array<f32, ${Math.min(numColumns, REDUCE_COLUMN_BLOCK) * 64}>;

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
    var wacc = 0.0;
    for (var s = local_id; s < count; s += 64u) {
        let gi = sortedIdx[begin + s];
        if (gi >= uniforms.chunkStart && gi < uniforms.chunkEnd) {
            let src = (gi - uniforms.chunkStart) * numColumns;
            let w = ${weighted ? 'weights[gi]' : '1.0'};
            wacc += w;
            for (var j = 0u; j < numColumns; j++) {
                acc[j] += w * f32(points[src + j]);
            }
        }
    }
    // cross-thread reduce, one column block at a time: thread j serially sums
    // column j of the block over the 64 partials
    for (var b = 0u; b < numColumns; b += block) {
        let w = min(block, numColumns - b);
        for (var j = 0u; j < w; j++) {
            partials[local_id * block + j] = acc[b + j];
        }
        workgroupBarrier();
        for (var j = local_id; j < w; j += 64u) {
            var s = 0.0;
            for (var t = 0u; t < 64u; t++) {
                s += partials[t * block + j];
            }
            sums[cluster * numColumns + b + j] += s;
        }
        workgroupBarrier();
    }
${
    weighted
        ? `    partials[local_id] = wacc;
    workgroupBarrier();
    if (local_id == 0u) {
        var s = 0.0;
        for (var t = 0u; t < 64u; t++) {
            s += partials[t];
        }
        weightSums[cluster] += s;
    }`
        : ''
}
}
`;

// centroid = sum / count for non-empty clusters (thread per element); counts
// are u32 populations or (e-means) f32 weight totals
const divideWgsl = (numColumns: number, countType: 'u32' | 'f32' = 'u32') => /* wgsl */ `
struct Uniforms {
    numCentroids: u32
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> sums: array<f32>;
@group(0) @binding(2) var<storage, read> counts: array<${countType}>;
@group(0) @binding(3) var<storage, read_write> centroids: array<f32>;

const numColumns = ${numColumns}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) num_workgroups: vec3u) {
    // 2D dispatch for large palettes (see dispatchFlat)
    let i = gid.x + gid.y * num_workgroups.x * 256u;
    if (i < uniforms.numCentroids * numColumns) {
        let count = f32(counts[i / numColumns]);
        if (count > 0.0) {
            centroids[i] = sums[i] / count;
        }
    }
}
`;

// empty clusters re-seed to a pseudo-random point (PCG hash of cluster id and
// a per-iteration seed), preserving the CPU path's semantics. The dispatch for
// chunk c only handles clusters whose hashed point lands in chunk c; disjoint
// from divide (count == 0 vs count > 0), so ordering between them is free.
const reseedWgsl = (floatType: string, numColumns: number) => /* wgsl */ `
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
const convertWgsl = () => /* wgsl */ `
enable f16;

struct Uniforms {
    totalElems: u32
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> centroids: array<f32>;
@group(0) @binding(2) var<storage, read_write> centroidsF16: array<f16>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) num_workgroups: vec3u) {
    // 2D dispatch for large palettes (see dispatchFlat)
    let i = gid.x + gid.y * num_workgroups.x * 256u;
    if (i < uniforms.totalElems) {
        centroidsF16[i] = f16(centroids[i]);
    }
}
`;

// ---- e-means (Bessa et al., arXiv 2611.09501; reference implementation
// faiss/contrib/e_means, MIT). Mini-batch k-means whose centroids are
// exponential moving averages of batch means, with fitted scaling laws for the
// batch size and initial step, a cosine-annealed step, dead-cluster revival by
// splitting, and closing centroids taken as the exact means of the last
// epoch's assignments. Constants and laws below are the shipped faiss values.

const EMEANS_INIT_USAGE = 2 ** -16;
const EMEANS_ALPHA_MIN_FACTOR = 0.1;
const EMEANS_CHECK_UNUSED_EVERY = 5;
const EMEANS_UNUSED_THRESHOLD = 0.1;
const EMEANS_AUTO_QUALITY_MAX_D = 16;
// batch law B = C * k^bk * n^bn * T^bT: (C, bk, bn, bT)
const EMEANS_BATCH_LAWS = {
    throughput: [0.035134, 0.134749, 0.876473, 0.813344],
    quality: [0.00439175, 0.134749, 0.876473, 0.813344]
};
// step law alpha0 = C * k^ak * B^aB * n^an * T^aT: (C, ak, aB, an, aT)
const EMEANS_ALPHA_LAW = [0.8244, -0.4617, 0.9003, -0.4498, -0.1057];

// e-means keeps every point resident on the GPU (batches are ranges of one
// shuffled upload), unlike Lloyd's streaming mode
const EMEANS_RESIDENT_BYTES = 2 * 1024 * 1024 * 1024;

/**
 * Whether {@link GpuKmeans.runEmeans} can cluster this many points on the
 * device: the points must fit the resident budget and the labels one storage
 * binding.
 *
 * @param device - GPU device.
 * @param numPoints - Number of points.
 * @param numColumns - Dimensions per point.
 * @returns True when runEmeans accepts the input.
 * @ignore
 */
const emeansFits = (device: GraphicsDevice, numPoints: number, numColumns: number): boolean => {
    const f16 = !!('supportsShaderF16' in device && device.supportsShaderF16);
    const bindingLimit =
        (device as GraphicsDevice & { wgpu?: { limits?: { maxStorageBufferBindingSize?: number } } }).wgpu?.limits
            ?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
    return numPoints * numColumns * (f16 ? 2 : 4) <= EMEANS_RESIDENT_BYTES && numPoints * 4 <= bindingLimit;
};

/**
 * Points and weights shuffled once and resident on the GPU, shared by every
 * {@link GpuKmeans.runEmeans} call over the same data (e.g. the palette sizes
 * a colour-error search tries), which then skip the shuffle, the f16 packing
 * and the upload. Free with `destroy`.
 *
 * @ignore
 */
type EmeansPoints = {
    numPoints: number;
    numColumns: number;
    /** shuffled position → original row */
    perm: Uint32Array;
    /** rows per point buffer (each within one storage binding) */
    chunkCap: number;
    pointBufs: StorageBuffer[];
    weightsBuf: StorageBuffer;
    destroy: () => void;
};

/**
 * Shuffle, pack (f16 when the device supports it) and upload points and
 * weights for e-means. The input must pass {@link emeansFits}.
 *
 * @param device - GPU device.
 * @param points - Interleaved points (numPoints × numColumns).
 * @param numPoints - Number of points.
 * @param numColumns - Dimensions per point.
 * @param weights - Per-point weight; default 1.
 * @returns The resident points.
 * @ignore
 */
const uploadEmeansPoints = (
    device: GraphicsDevice,
    points: Float32Array,
    numPoints: number,
    numColumns: number,
    weights?: Float32Array
): EmeansPoints => {
    const useF16 = !!('supportsShaderF16' in device && device.supportsShaderF16);
    const bytesPerElem = useF16 ? 2 : 4;
    const rowBytes = numColumns * bytesPerElem;
    const bindingLimit =
        (device as GraphicsDevice & { wgpu?: { limits?: { maxStorageBufferBindingSize?: number } } }).wgpu?.limits
            ?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
    const chunkCap = Math.min(4_000_000, Math.floor(bindingLimit / rowBytes)) & ~1;
    const numChunks = Math.ceil(numPoints / chunkCap);
    if (numPoints * 4 > bindingLimit) {
        throw new Error(
            `GpuKmeans: labels buffer (${numPoints * 4} bytes) exceeds device maxStorageBufferBindingSize (${bindingLimit})`
        );
    }
    if (numPoints * rowBytes > EMEANS_RESIDENT_BYTES) {
        throw new Error(
            `GpuKmeans e-means: points (${numPoints * rowBytes} bytes) exceed the ${EMEANS_RESIDENT_BYTES}-byte resident budget`
        );
    }
    const chunkRows = (c: number) => Math.min(chunkCap, numPoints - c * chunkCap);

    // one shuffle up front (Fisher-Yates); batches are contiguous ranges of
    // the shuffled order
    const perm = new Uint32Array(numPoints);
    for (let i = 0; i < numPoints; i++) perm[i] = i;
    for (let i = numPoints - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        const t = perm[i];
        perm[i] = perm[j];
        perm[j] = t;
    }
    const packed = useF16
        ? new Uint16Array(roundUp(numPoints * numColumns, 2))
        : new Float32Array(numPoints * numColumns);
    for (let i = 0; i < numPoints; i++) {
        const src = perm[i] * numColumns;
        const dst = i * numColumns;
        if (useF16) {
            for (let j = 0; j < numColumns; j++) packed[dst + j] = FloatPacking.float2Half(points[src + j]);
        } else {
            for (let j = 0; j < numColumns; j++) packed[dst + j] = points[src + j];
        }
    }
    const shuffledWeights = new Float32Array(numPoints);
    for (let i = 0; i < numPoints; i++) shuffledWeights[i] = weights ? weights[perm[i]] : 1;

    const pointBufs = Array.from({ length: numChunks }, (_, c) => {
        const buf = new StorageBuffer(
            device,
            roundUp(chunkRows(c) * numColumns, 2) * bytesPerElem,
            BUFFERUSAGE_COPY_DST
        );
        buf.write(0, packed, c * chunkCap * numColumns, roundUp(chunkRows(c) * numColumns, useF16 ? 2 : 1));
        return buf;
    });
    const weightsBuf = new StorageBuffer(device, numPoints * 4, BUFFERUSAGE_COPY_DST);
    weightsBuf.write(0, shuffledWeights, 0, numPoints);

    return {
        numPoints,
        numColumns,
        perm,
        chunkCap,
        pointBufs,
        weightsBuf,
        destroy: () => {
            pointBufs.forEach((b) => b.destroy());
            weightsBuf.destroy();
        }
    };
};

const roundPowerOfTwo = (value: number) => {
    if (value <= 1) return 1;
    const low = 2 ** Math.floor(Math.log2(value));
    return value / low < (2 * low) / value ? low : 2 * low;
};

type EmeansRecipe = {
    batchSize: number;
    alpha0: number;
    /** rows visited per epoch: whole passes, an optional fractional pass, then a full closing pass */
    epochRows: number[];
    totalSteps: number;
};

/**
 * Resolve the e-means training plan (faiss `_resolve_recipe` / `_epoch_rows`).
 *
 * @param n - Number of points.
 * @param k - Number of clusters.
 * @param d - Dimensions.
 * @param epochs - Epoch budget T (may be fractional).
 * @returns The recipe.
 * @ignore
 */
const emeansRecipe = (n: number, k: number, d: number, epochs: number): EmeansRecipe => {
    const law = d <= EMEANS_AUTO_QUALITY_MAX_D ? 'quality' : 'throughput';
    const [cb, bk, bn, bt] = EMEANS_BATCH_LAWS[law];
    let batchSize = roundPowerOfTwo(cb * k ** bk * n ** bn * epochs ** bt);
    batchSize =
        law === 'throughput' ? Math.min(batchSize, roundPowerOfTwo(n / 2)) : Math.min(batchSize, Math.floor(n / 2));
    batchSize = Math.max(1, Math.min(batchSize, n));

    const [ca, ak, ab, an, at] = EMEANS_ALPHA_LAW;
    const alpha0 = Math.min(Math.max(ca * k ** ak * batchSize ** ab * n ** an * epochs ** at, 1e-4), 1);

    const whole = Math.floor(epochs);
    const remainder = Math.round((epochs - whole) * n);
    const epochRows = whole === 0 ? [remainder] : [...Array(whole - 1).fill(n), ...(remainder ? [remainder] : []), n];
    const totalSteps = epochRows.reduce((s, rows) => s + Math.ceil(rows / batchSize), 0);
    return { batchSize, alpha0, epochRows, totalSteps };
};

// cosine-anneal the EMA step from alpha0 towards alpha0 * EMEANS_ALPHA_MIN_FACTOR
const emeansAlpha = (alpha0: number, step: number, totalSteps: number) => {
    const fraction = Math.min(step, totalSteps) / Math.max(1, totalSteps);
    const cosine = 0.5 * (1 + Math.cos(Math.PI * fraction));
    return alpha0 * (EMEANS_ALPHA_MIN_FACTOR + (1 - EMEANS_ALPHA_MIN_FACTOR) * cosine);
};

// EMA update of one batch, thread per cluster (Algorithm 1 line 8, in the
// faiss mass/position form): N ← (1-α)N, then for clusters the batch touched
// N ← N + α·n and c ← c + (α·n/N)(s/n - c), with n the batch's weight total
// (the point count when unweighted). In the last epoch the batch sums also
// accumulate into the closing-mean totals. alpha arrives as f32 bits.
const emaWgsl = (numColumns: number) => /* wgsl */ `
struct Uniforms {
    numCentroids: u32,
    alphaBits: u32,
    replaceUsage: u32,      // alpha == 1: touched clusters forget their mass, untouched keep it
    lastEpoch: u32
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> sums: array<f32>;
@group(0) @binding(2) var<storage, read> weightSums: array<f32>;
@group(0) @binding(3) var<storage, read_write> centroids: array<f32>;
@group(0) @binding(4) var<storage, read_write> usage: array<f32>;
@group(0) @binding(5) var<storage, read_write> passSums: array<f32>;
@group(0) @binding(6) var<storage, read_write> passWeights: array<f32>;

const numColumns = ${numColumns}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) num_workgroups: vec3u) {
    // 2D dispatch for large palettes (see dispatchFlat)
    let j = gid.x + gid.y * num_workgroups.x * 256u;
    if (j >= uniforms.numCentroids) {
        return;
    }
    let alpha = bitcast<f32>(uniforms.alphaBits);
    let replace = uniforms.replaceUsage != 0u;
    let n = weightSums[j];
    var u = usage[j];
    if (!replace) {
        u = u * (1.0 - alpha);
    }
    if (n > 0.0) {
        let retained = select(u, 0.0, replace);
        let mass = retained + alpha * n;
        let w = alpha * n / mass;
        let inv = 1.0 / n;
        let base = j * numColumns;
        for (var c = 0u; c < numColumns; c++) {
            let mean = sums[base + c] * inv;
            let cur = centroids[base + c];
            centroids[base + c] = select(cur + w * (mean - cur), mean, w == 1.0);
        }
        u = mass;
        if (uniforms.lastEpoch != 0u) {
            passWeights[j] += n;
            for (var c = 0u; c < numColumns; c++) {
                passSums[base + c] += sums[base + c];
            }
        }
    }
    usage[j] = u;
}
`;

// dead-cluster revival (Appendix J): each (donor, target) pair splits the
// donor into two copies perturbed by ±epsilon (alternating sign per column)
// and halves its mass between them. A zero donor would give identical copies,
// so the target's first column is nudged by the smallest normal.
const splitWgsl = (numColumns: number, epsilon: number, minNormal: number) => /* wgsl */ `
struct Uniforms {
    numPairs: u32
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> pairs: array<u32>;     // donor0, target0, donor1, target1, ...
@group(0) @binding(2) var<storage, read_write> centroids: array<f32>;
@group(0) @binding(3) var<storage, read_write> usage: array<f32>;

const numColumns = ${numColumns}u;
const epsilon: f32 = ${epsilon.toExponential()};
const minNormal: f32 = ${minNormal.toExponential()};

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let p = gid.x;
    if (p >= uniforms.numPairs) {
        return;
    }
    let s = pairs[2u * p] * numColumns;
    let t = pairs[2u * p + 1u] * numColumns;
    var collapsed = true;
    for (var c = 0u; c < numColumns; c++) {
        let v = centroids[s + c];
        let offset = select(epsilon, -epsilon, (c & 1u) == 1u);
        let donor = v * (1.0 - offset);
        let child = v * (1.0 + offset);
        centroids[s + c] = donor;
        centroids[t + c] = child;
        if (donor != child) {
            collapsed = false;
        }
    }
    if (collapsed) {
        centroids[t] = centroids[t] + minNormal;
    }
    let half = usage[pairs[2u * p]] * 0.5;
    usage[pairs[2u * p]] = half;
    usage[pairs[2u * p + 1u]] = half;
}
`;

/**
 * Choose revival splits (faiss `_choose_splits`): clusters whose mass is below
 * EMEANS_UNUSED_THRESHOLD × the mean mass are targets; donors are sampled
 * without replacement in proportion to mass (Gumbel top-k).
 *
 * @param usage - Per-cluster EMA mass.
 * @param pairs - Receives donor/target pairs, interleaved.
 * @returns Number of pairs written.
 */
const chooseSplits = (usage: Float32Array, pairs: Uint32Array) => {
    const k = usage.length;
    let total = 0;
    for (let j = 0; j < k; j++) total += usage[j];
    const threshold = (total * EMEANS_UNUSED_THRESHOLD) / k;

    const targets: number[] = [];
    const donors: { j: number; key: number }[] = [];
    for (let j = 0; j < k; j++) {
        if (usage[j] < threshold) {
            targets.push(j);
        } else if (usage[j] > 0) {
            const u = Math.max(Math.random(), Number.MIN_VALUE);
            donors.push({ j, key: Math.log(usage[j]) - Math.log(-Math.log(u)) });
        }
    }
    const nsplit = Math.min(targets.length, donors.length);
    if (nsplit === 0) return 0;
    donors.sort((a, b) => b.key - a.key);
    for (let p = 0; p < nsplit; p++) {
        pairs[2 * p] = donors[p].j;
        pairs[2 * p + 1] = targets[p];
    }
    return nsplit;
};

/**
 * Dispatches `threads` invocations of a 256-wide kernel that reads its flat index as
 * x + y * width, splitting into rows when one dimension would exceed WebGPU's 65535
 * workgroups (e.g. a 262144-entry palette of 72 SH coefficients).
 *
 * @param compute - The compute to dispatch.
 * @param threads - Total invocations needed.
 */
const dispatchFlat = (compute: Compute, threads: number) => {
    const groups = Math.ceil(threads / 256);
    const rows = Math.ceil(groups / 65535);
    compute.setupDispatch(Math.ceil(groups / rows), rows);
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
    /**
     * Run e-means on the GPU: same inputs and outputs as `run`,
     * with an epoch budget instead of Lloyd iterations. Points are shuffled
     * once on upload; each epoch walks the shuffled order from a random
     * offset in batches (faiss reshuffles every epoch). Returned labels are
     * the last epoch's assignments and centroids their exact weighted
     * means. `weights` (one per point, default 1) scale each point's pull
     * on its centroid; assignment stays plain nearest-centroid. `onEpoch`
     * fires as each epoch is submitted. All points stay resident on the
     * GPU, so the input must pass {@link emeansFits}. `resident` (from
     * {@link uploadEmeansPoints} over the same points and weights) skips the
     * shuffle and upload; it stays owned by the caller.
     */
    runEmeans: (
        points: Float32Array,
        numPoints: number,
        centroids: Float32Array,
        labels: Uint32Array,
        epochs: number,
        onEpoch?: () => void,
        weights?: Float32Array,
        resident?: EmeansPoints
    ) => Promise<void>;
    destroy: () => void;

    constructor(device: GraphicsDevice, numColumns: number, numCentroids: number) {
        const useF16 = !!('supportsShaderF16' in device && device.supportsShaderF16);
        const floatType = useF16 ? 'f16' : 'f32';
        const bytesPerElem = useF16 ? 2 : 4;

        const wgpuLimits = (
            device as GraphicsDevice & {
                wgpu?: {
                    limits?: {
                        maxComputeWorkgroupStorageSize?: number;
                        maxComputeInvocationsPerWorkgroup?: number;
                        maxStorageBufferBindingSize?: number;
                    };
                };
            }
        ).wgpu?.limits;

        // flush recorded work to the queue without any CPU synchronization
        const submit = () => (device as unknown as { submit: () => void }).submit();

        // centroid tile: large tiles amortize shared-memory loads but the
        // tile is the occupancy limiter — filling the 32KB workgroup-storage
        // limit leaves ~2 workgroups per core and starves the GPU of latency
        // hiding (measured 2× slower at d=45). Budget ~12KB (128 rows at
        // 45 columns f16, like the previous fixed tile) and shrink when a
        // row is too wide (the previous fixed 128 needed >16KB at 45
        // columns f32 — over the WebGPU minimum). f32 rows are padded to
        // whole vec4s.
        const wgStorage: number = wgpuLimits?.maxComputeWorkgroupStorageSize ?? 16384;
        const tileBudget = Math.min(wgStorage, 12 * 1024);
        const tileRowBytes = useF16 ? numColumns * 2 : roundUp(numColumns, 4) * 4;
        // the f16 kernel loads tileSize/64 rows per thread, so its tile is a
        // multiple of 64; the f32 kernel strides its load and may go below 64
        // rows when a row is wide (SH band 4: 72 columns)
        const tileSize = useF16
            ? Math.max(64, Math.min(128, Math.floor(tileBudget / (tileRowBytes * 64)) * 64))
            : Math.max(16, Math.min(128, Math.floor(tileBudget / (tileRowBytes * 16)) * 16));
        const ppt = pptForColumns(numColumns);
        // the f32 assign benched fastest at 128 threads on an A10G (vs 64)
        const assignWorkgroupSize = useF16 ? 64 : 128;

        const kernels = {
            assign: makeKernel(
                device,
                'kmeans-assign',
                useF16
                    ? assignWgsl(floatType, numColumns, tileSize, ppt)
                    : assignVec4Wgsl(numColumns, tileSize, ppt, assignWorkgroupSize),
                ['localOffset', 'globalOffset', 'count', 'numCentroids', 'centroidBegin', 'centroidEnd'],
                [
                    ['points', true],
                    ['centroids', true],
                    ['labels', false],
                    ...(useF16 ? [] : [['center', true] as [string, boolean]]),
                    ['bestDist', false]
                ]
            ),
            histogram: makeKernel(
                device,
                'kmeans-histogram',
                histogramWgsl(),
                ['globalOffset', 'count'],
                [
                    ['labels', true],
                    ['counts', false]
                ]
            ),
            scan: makeKernel(
                device,
                'kmeans-scan',
                scanWgsl(),
                ['numCentroids'],
                [
                    ['counts', true],
                    ['offsets', false],
                    ['cursors', false]
                ]
            ),
            scatter: makeKernel(
                device,
                'kmeans-scatter',
                scatterWgsl(),
                ['globalOffset', 'count'],
                [
                    ['labels', true],
                    ['cursors', false],
                    ['sortedIdx', false]
                ]
            ),
            reduce: makeKernel(
                device,
                'kmeans-reduce',
                reduceWgsl(floatType, numColumns),
                ['numCentroids', 'chunkStart', 'chunkEnd'],
                [
                    ['points', true],
                    ['sortedIdx', true],
                    ['offsets', true],
                    ['counts', true],
                    ['sums', false]
                ]
            ),
            divide: makeKernel(
                device,
                'kmeans-divide',
                divideWgsl(numColumns),
                ['numCentroids'],
                [
                    ['sums', true],
                    ['counts', true],
                    ['centroids', false]
                ]
            ),
            reseed: makeKernel(
                device,
                'kmeans-reseed',
                reseedWgsl(floatType, numColumns),
                ['numCentroids', 'numPoints', 'seed', 'chunkStart', 'chunkEnd'],
                [
                    ['counts', true],
                    ['points', true],
                    ['centroids', false]
                ]
            ),
            convert: useF16
                ? makeKernel(
                      device,
                      'kmeans-convert',
                      convertWgsl(),
                      ['totalElems'],
                      [
                          ['centroids', true],
                          ['centroidsF16', false]
                      ]
                  )
                : null,
            reduceWeighted: makeKernel(
                device,
                'kmeans-reduce-weighted',
                reduceWgsl(floatType, numColumns, true),
                ['numCentroids', 'chunkStart', 'chunkEnd'],
                [
                    ['points', true],
                    ['sortedIdx', true],
                    ['offsets', true],
                    ['counts', true],
                    ['sums', false],
                    ['weights', true],
                    ['weightSums', false]
                ]
            ),
            close: makeKernel(
                device,
                'kmeans-close',
                divideWgsl(numColumns, 'f32'),
                ['numCentroids'],
                [
                    ['sums', true],
                    ['counts', true],
                    ['centroids', false]
                ]
            ),
            ema: makeKernel(
                device,
                'kmeans-ema',
                emaWgsl(numColumns),
                ['numCentroids', 'alphaBits', 'replaceUsage', 'lastEpoch'],
                [
                    ['sums', true],
                    ['weightSums', true],
                    ['centroids', false],
                    ['usage', false],
                    ['passSums', false],
                    ['passWeights', false]
                ]
            ),
            // f16 rounds faiss's 1e-4 split apart (assign would see an exact
            // tie and the child would never win a point), so widen it to a
            // few f16 ulps there
            split: makeKernel(
                device,
                'kmeans-split',
                splitWgsl(numColumns, useF16 ? 2e-3 : 1e-4, useF16 ? 6.103515625e-5 : 1.1754943508222875e-38),
                ['numPairs'],
                [
                    ['pairs', true],
                    ['centroids', false],
                    ['usage', false]
                ]
            )
        };

        this.run = async (points, numPoints, centroids, labels, iterations, onIteration) => {
            // ---- chunk plan: each point chunk must fit a storage binding and
            // keep assign a 1D dispatch (65535 workgroups × 64 threads)
            const rowBytes = numColumns * bytesPerElem;
            const bindingLimit: number = wgpuLimits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
            const chunkCap = Math.min(4_000_000, Math.floor(bindingLimit / rowBytes)) & ~1;
            const numChunks = Math.ceil(numPoints / chunkCap);

            // Pre-flight the whole-scene bindings so we fail with a clear
            // message instead of a driver-side error mid-run (mirrors
            // GpuEdgeCost). Point chunks are sliced under the limit above, but
            // labels and sortedIdx are bound whole — the counting sort is
            // global, so they can't be chunked. At 4 bytes/point they exceed a
            // 128 MiB binding beyond ~33.5M points.
            if (numPoints * 4 > bindingLimit) {
                throw new Error(
                    `GpuKmeans: labels/sortedIdx buffers (${numPoints * 4} bytes for ${numPoints} points) exceed ` +
                        `device maxStorageBufferBindingSize (${bindingLimit}) — reduce the input (e.g. write streamed ` +
                        'SOG / LOD units) or use a device with larger limits'
                );
            }
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
            const pointBufs = resident
                ? Array.from(
                      { length: numChunks },
                      (_, c) =>
                          new StorageBuffer(
                              device,
                              roundUp(chunkRows(c) * numColumns, 2) * bytesPerElem,
                              BUFFERUSAGE_COPY_DST
                          )
                  )
                : Array.from(
                      { length: 2 },
                      () =>
                          new StorageBuffer(
                              device,
                              roundUp(chunkCap * numColumns, 2) * bytesPerElem,
                              BUFFERUSAGE_COPY_DST
                          )
                  );
            const centroidsBuf = new StorageBuffer(
                device,
                numCentroids * numColumns * 4,
                BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC
            );
            const centroidsF16Buf = useF16
                ? new StorageBuffer(device, roundUp(numCentroids * numColumns, 2) * 2, 0)
                : null;
            const labelsBuf = new StorageBuffer(device, numPoints * 4, BUFFERUSAGE_COPY_SRC);
            // assign centroid range per sub-dispatch of `count` points: a
            // whole number of tiles, so the tiling matches the unsplit loop
            const assignRange = (count: number) => {
                return Math.max(tileSize, Math.floor(ASSIGN_WORK / (count * numColumns * tileSize)) * tileSize);
            };
            // the running best distance between centroid ranges is only needed
            // when the largest sub-dispatch splits (smaller ones split less)
            const splits = assignRange(Math.min(ASSIGN_SUB, chunkCap, numPoints)) < numCentroids;
            const bestDistBuf = new StorageBuffer(device, splits ? numPoints * 4 : 4, 0);
            const sortedIdxBuf = new StorageBuffer(device, numPoints * 4, 0);
            const countsBuf = new StorageBuffer(device, numCentroids * 4, BUFFERUSAGE_COPY_DST);
            const offsetsBuf = new StorageBuffer(device, numCentroids * 4, 0);
            const cursorsBuf = new StorageBuffer(device, numCentroids * 4, 0);
            const sumsBuf = new StorageBuffer(device, numCentroids * numColumns * 4, BUFFERUSAGE_COPY_DST);
            const centerBuf = useF16
                ? null
                : new StorageBuffer(device, roundUp(numColumns, 4) * 4, BUFFERUSAGE_COPY_DST);

            try {
                // ---- static parameters
                const { assign, histogram, scan, scatter, reduce, divide, reseed, convert } = kernels;
                assign.compute.setParameter('centroids', useF16 ? centroidsF16Buf! : centroidsBuf);
                assign.compute.setParameter('labels', labelsBuf);
                assign.compute.setParameter('bestDist', bestDistBuf);
                assign.compute.setParameter('numCentroids', numCentroids);
                if (centerBuf) {
                    // the f32 assign's reference row: the seeds' mean (the
                    // seeds are data points), zero-padded to whole vec4s
                    const center = new Float32Array(roundUp(numColumns, 4));
                    for (let j = 0; j < numColumns; j++) {
                        let sum = 0;
                        for (let i = 0; i < numCentroids; i++) sum += centroids[i * numColumns + j];
                        center[j] = sum / numCentroids;
                    }
                    centerBuf.write(0, center, 0, center.length);
                    assign.compute.setParameter('center', centerBuf);
                }
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
                        dispatchFlat(convert.compute, numCentroids * numColumns);
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

                // streaming-mode staging bound (see the iteration-boundary sync
                // below); 8 bytes keeps every mapAsync alignment rule satisfied
                const syncScratch = new Float32Array(2);

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
                        // one submit per sub-dispatch (uniform changes only take
                        // effect across a submit, see the NOTE below), which
                        // also keeps each command buffer under the watchdog
                        for (let s = 0; s < rows; s += ASSIGN_SUB) {
                            const count = Math.min(ASSIGN_SUB, rows - s);
                            const range = assignRange(count);
                            for (let begin = 0; begin < numCentroids; begin += range) {
                                assign.compute.setParameter('points', buf);
                                assign.compute.setParameter('localOffset', s);
                                assign.compute.setParameter('globalOffset', c * chunkCap + s);
                                assign.compute.setParameter('count', count);
                                assign.compute.setParameter('centroidBegin', begin);
                                assign.compute.setParameter('centroidEnd', Math.min(numCentroids, begin + range));
                                assign.compute.setupDispatch(Math.ceil(count / (assignWorkgroupSize * ppt)));
                                device.computeDispatch([assign.compute], 'kmeans-assign');
                                submit();
                            }
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
                    dispatchFlat(divide.compute, numCentroids * numColumns);
                    device.computeDispatch([divide.compute], 'kmeans-divide');
                    dispatchConvert();
                    submit();

                    // Streaming mode re-uploads every point chunk twice per
                    // iteration via queue.writeBuffer, which copies into
                    // driver-owned staging that is reclaimed only as the GPU
                    // consumes it — with zero CPU sync the loop would stage
                    // ~2× the point data PER ITERATION (tens of GB of host
                    // RAM). A tiny readback at the iteration boundary drains
                    // the queue, capping staging at one iteration's worth.
                    // Resident mode uploads once and stays fully zero-sync.
                    if (!resident) {
                        await centroidsBuf.read(0, 8, syncScratch, true);
                    }

                    onIteration?.();
                }

                // ---- single blocking readback after the final iteration. Both
                // copies are queued before awaiting, so neither waits on GPU
                // work submitted after this point (the LOD writer queues the
                // next unit's k-means behind this one).
                await Promise.all([
                    labelsBuf.read(0, numPoints * 4, labels, true),
                    centroidsBuf.read(0, numCentroids * numColumns * 4, centroids, true)
                ]);
            } finally {
                pointBufs.forEach((b) => b.destroy());
                centroidsBuf.destroy();
                centroidsF16Buf?.destroy();
                labelsBuf.destroy();
                bestDistBuf.destroy();
                sortedIdxBuf.destroy();
                countsBuf.destroy();
                offsetsBuf.destroy();
                cursorsBuf.destroy();
                sumsBuf.destroy();
                centerBuf?.destroy();
            }
        };

        this.runEmeans = async (points, numPoints, centroids, labels, epochs, onEpoch, weights, resident) => {
            const recipe = emeansRecipe(numPoints, numCentroids, numColumns, epochs);
            const { batchSize, alpha0, epochRows, totalSteps } = recipe;
            // faiss disables revival at d <= 2
            const revive = numColumns > 2;

            if (resident && (resident.numPoints !== numPoints || resident.numColumns !== numColumns)) {
                throw new Error('GpuKmeans e-means: resident points do not match the input');
            }
            const data = resident ?? uploadEmeansPoints(device, points, numPoints, numColumns, weights);
            const { perm, chunkCap, pointBufs, weightsBuf } = data;
            const chunkRows = (c: number) => Math.min(chunkCap, numPoints - c * chunkCap);

            // ---- buffers
            const centroidsBuf = new StorageBuffer(
                device,
                numCentroids * numColumns * 4,
                BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC
            );
            const centroidsF16Buf = useF16
                ? new StorageBuffer(device, roundUp(numCentroids * numColumns, 2) * 2, 0)
                : null;
            const labelsBuf = new StorageBuffer(device, numPoints * 4, BUFFERUSAGE_COPY_SRC);
            const assignRange = (count: number) => {
                return Math.max(tileSize, Math.floor(ASSIGN_WORK / (count * numColumns * tileSize)) * tileSize);
            };
            const splits = assignRange(Math.min(ASSIGN_SUB, chunkCap, batchSize)) < numCentroids;
            const bestDistBuf = new StorageBuffer(device, splits ? numPoints * 4 : 4, 0);
            // the counting sort only ever holds one batch
            const sortedIdxBuf = new StorageBuffer(device, batchSize * 4, 0);
            const countsBuf = new StorageBuffer(device, numCentroids * 4, BUFFERUSAGE_COPY_DST);
            const offsetsBuf = new StorageBuffer(device, numCentroids * 4, 0);
            const cursorsBuf = new StorageBuffer(device, numCentroids * 4, 0);
            const sumsBuf = new StorageBuffer(device, numCentroids * numColumns * 4, BUFFERUSAGE_COPY_DST);
            const centerBuf = useF16
                ? null
                : new StorageBuffer(device, roundUp(numColumns, 4) * 4, BUFFERUSAGE_COPY_DST);
            const usageBuf = new StorageBuffer(device, numCentroids * 4, BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC);
            const passSumsBuf = new StorageBuffer(device, numCentroids * numColumns * 4, BUFFERUSAGE_COPY_DST);
            const weightSumsBuf = new StorageBuffer(device, numCentroids * 4, BUFFERUSAGE_COPY_DST);
            const passWeightsBuf = new StorageBuffer(device, numCentroids * 4, BUFFERUSAGE_COPY_DST);
            const pairsBuf = new StorageBuffer(device, Math.max(8, numCentroids * 4), BUFFERUSAGE_COPY_DST);

            try {
                const {
                    assign,
                    histogram,
                    scan,
                    scatter,
                    reduceWeighted: reduce,
                    close,
                    convert,
                    ema,
                    split
                } = kernels;
                assign.compute.setParameter('centroids', useF16 ? centroidsF16Buf! : centroidsBuf);
                assign.compute.setParameter('labels', labelsBuf);
                assign.compute.setParameter('bestDist', bestDistBuf);
                assign.compute.setParameter('numCentroids', numCentroids);
                if (centerBuf) {
                    const center = new Float32Array(roundUp(numColumns, 4));
                    for (let j = 0; j < numColumns; j++) {
                        let sum = 0;
                        for (let i = 0; i < numCentroids; i++) sum += centroids[i * numColumns + j];
                        center[j] = sum / numCentroids;
                    }
                    centerBuf.write(0, center, 0, center.length);
                    assign.compute.setParameter('center', centerBuf);
                }
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
                reduce.compute.setParameter('weights', weightsBuf);
                reduce.compute.setParameter('weightSums', weightSumsBuf);
                ema.compute.setParameter('sums', sumsBuf);
                ema.compute.setParameter('weightSums', weightSumsBuf);
                ema.compute.setParameter('centroids', centroidsBuf);
                ema.compute.setParameter('usage', usageBuf);
                ema.compute.setParameter('passSums', passSumsBuf);
                ema.compute.setParameter('passWeights', passWeightsBuf);
                ema.compute.setParameter('numCentroids', numCentroids);
                split.compute.setParameter('pairs', pairsBuf);
                split.compute.setParameter('centroids', centroidsBuf);
                split.compute.setParameter('usage', usageBuf);
                // closing means over the last epoch's weighted totals
                close.compute.setParameter('sums', passSumsBuf);
                close.compute.setParameter('counts', passWeightsBuf);
                close.compute.setParameter('centroids', centroidsBuf);
                close.compute.setParameter('numCentroids', numCentroids);
                convert?.compute.setParameter('centroids', centroidsBuf);
                convert?.compute.setParameter('centroidsF16', centroidsF16Buf);
                convert?.compute.setParameter('totalElems', numCentroids * numColumns);

                const dispatchConvert = () => {
                    if (convert) {
                        dispatchFlat(convert.compute, numCentroids * numColumns);
                        device.computeDispatch([convert.compute], 'kmeans-convert');
                    }
                };

                // ---- one-time uploads (points and weights are resident already)
                centroidsBuf.write(0, centroids, 0, numCentroids * numColumns);
                usageBuf.write(0, new Float32Array(numCentroids).fill(EMEANS_INIT_USAGE), 0, numCentroids);
                passSumsBuf.clear();
                passWeightsBuf.clear();
                dispatchConvert();

                // a shuffled-order range [g0, g1) split at chunk boundaries
                type Piece = { chunk: number; local: number; global: number; count: number };
                const pushPieces = (out: Piece[], g0: number, g1: number) => {
                    for (let g = g0; g < g1;) {
                        const chunk = Math.floor(g / chunkCap);
                        const end = Math.min(g1, (chunk + 1) * chunkCap);
                        out.push({ chunk, local: g - chunk * chunkCap, global: g, count: end - g });
                        g = end;
                    }
                };

                const usage = new Float32Array(numCentroids);
                const pairs = new Uint32Array(Math.max(2, numCentroids));
                const alphaBits = new Uint32Array(1);
                const alphaView = new Float32Array(alphaBits.buffer);

                let step = 0;
                for (let epoch = 0; epoch < epochRows.length; epoch++) {
                    const rows = epochRows[epoch];
                    const lastEpoch = epoch === epochRows.length - 1;
                    const start = Math.floor(Math.random() * numPoints);

                    for (let b0 = 0; b0 < rows; b0 += batchSize) {
                        const b1 = Math.min(b0 + batchSize, rows);

                        // logical batch [b0, b1) from the epoch's start offset,
                        // wrapping around the end of the shuffled order
                        const pieces: Piece[] = [];
                        const p0 = (start + b0) % numPoints;
                        const p1 = p0 + (b1 - b0);
                        pushPieces(pieces, p0, Math.min(p1, numPoints));
                        if (p1 > numPoints) pushPieces(pieces, 0, p1 - numPoints);

                        countsBuf.clear();
                        sumsBuf.clear();
                        weightSumsBuf.clear();

                        // assign this batch against the current centroids
                        for (const piece of pieces) {
                            for (let s = 0; s < piece.count; s += ASSIGN_SUB) {
                                const count = Math.min(ASSIGN_SUB, piece.count - s);
                                const range = assignRange(count);
                                for (let begin = 0; begin < numCentroids; begin += range) {
                                    assign.compute.setParameter('points', pointBufs[piece.chunk]);
                                    assign.compute.setParameter('localOffset', piece.local + s);
                                    assign.compute.setParameter('globalOffset', piece.global + s);
                                    assign.compute.setParameter('count', count);
                                    assign.compute.setParameter('centroidBegin', begin);
                                    assign.compute.setParameter('centroidEnd', Math.min(numCentroids, begin + range));
                                    assign.compute.setupDispatch(Math.ceil(count / (assignWorkgroupSize * ppt)));
                                    device.computeDispatch([assign.compute], 'kmeans-assign');
                                    submit();
                                }
                            }
                        }

                        // batch counts and sums (same sort-inverse path as Lloyd)
                        for (const piece of pieces) {
                            histogram.compute.setParameter('globalOffset', piece.global);
                            histogram.compute.setParameter('count', piece.count);
                            histogram.compute.setupDispatch(Math.ceil(piece.count / 256));
                            device.computeDispatch([histogram.compute], 'kmeans-histogram');
                            submit();
                        }
                        scan.compute.setupDispatch(1);
                        device.computeDispatch([scan.compute], 'kmeans-scan');
                        for (const piece of pieces) {
                            scatter.compute.setParameter('globalOffset', piece.global);
                            scatter.compute.setParameter('count', piece.count);
                            scatter.compute.setupDispatch(Math.ceil(piece.count / 256));
                            device.computeDispatch([scatter.compute], 'kmeans-scatter');
                            submit();
                        }
                        const groups = numCentroids;
                        const y = Math.ceil(groups / 65535);
                        for (const chunk of new Set(pieces.map((p) => p.chunk))) {
                            reduce.compute.setParameter('points', pointBufs[chunk]);
                            reduce.compute.setParameter('chunkStart', chunk * chunkCap);
                            reduce.compute.setParameter('chunkEnd', chunk * chunkCap + chunkRows(chunk));
                            reduce.compute.setupDispatch(Math.ceil(groups / y), y);
                            device.computeDispatch([reduce.compute], 'kmeans-reduce');
                            submit();
                        }

                        // EMA step
                        const alpha = emeansAlpha(alpha0, step, totalSteps);
                        alphaView[0] = alpha;
                        ema.compute.setParameter('alphaBits', alphaBits[0]);
                        ema.compute.setParameter('replaceUsage', alpha === 1 ? 1 : 0);
                        ema.compute.setParameter('lastEpoch', lastEpoch ? 1 : 0);
                        dispatchFlat(ema.compute, numCentroids);
                        device.computeDispatch([ema.compute], 'kmeans-ema');
                        dispatchConvert();
                        submit();
                        step++;

                        // revival every few steps: needs the masses on the CPU
                        // to pick donors, so this is the loop's only sync point
                        if (revive && step % EMEANS_CHECK_UNUSED_EVERY === 0) {
                            await usageBuf.read(0, numCentroids * 4, usage, true);
                            const numPairs = chooseSplits(usage, pairs);
                            if (numPairs > 0) {
                                pairsBuf.write(0, pairs, 0, numPairs * 2);
                                split.compute.setParameter('numPairs', numPairs);
                                split.compute.setupDispatch(Math.ceil(numPairs / 64));
                                device.computeDispatch([split.compute], 'kmeans-split');
                                dispatchConvert();
                                submit();
                            }
                        }
                    }
                    onEpoch?.();
                }

                // closing means of the last epoch's assignments
                dispatchFlat(close.compute, numCentroids * numColumns);
                device.computeDispatch([close.compute], 'kmeans-close');
                submit();

                const shuffledLabels = new Uint32Array(numPoints);
                await Promise.all([
                    labelsBuf.read(0, numPoints * 4, shuffledLabels, true),
                    centroidsBuf.read(0, numCentroids * numColumns * 4, centroids, true)
                ]);
                for (let i = 0; i < numPoints; i++) labels[perm[i]] = shuffledLabels[i];
            } finally {
                if (!resident) data.destroy();
                centroidsBuf.destroy();
                centroidsF16Buf?.destroy();
                labelsBuf.destroy();
                bestDistBuf.destroy();
                sortedIdxBuf.destroy();
                countsBuf.destroy();
                offsetsBuf.destroy();
                cursorsBuf.destroy();
                sumsBuf.destroy();
                centerBuf?.destroy();
                usageBuf.destroy();
                passSumsBuf.destroy();
                weightSumsBuf.destroy();
                passWeightsBuf.destroy();
                pairsBuf.destroy();
            }
        };

        this.destroy = () => {
            Object.values(kernels).forEach((k) => k?.destroy());
        };
    }
}

export { GpuKmeans, emeansFits, uploadEmeansPoints, type EmeansPoints };
