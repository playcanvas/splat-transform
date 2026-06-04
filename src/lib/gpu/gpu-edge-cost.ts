import {
    BUFFERUSAGE_COPY_DST,
    BUFFERUSAGE_COPY_SRC,
    SHADERLANGUAGE_WGSL,
    SHADERSTAGE_COMPUTE,
    UNIFORMTYPE_FLOAT,
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

/**
 * Appearance columns per storage chunk. The kernel exposes three appearance
 * bindings (appA/appB/appC), so the layout holds up to 3·APP_CHUNK columns; at
 * 16 the widest chunk reaches the ~2 GB per-binding limit around ~33.5M splats.
 * The CPU-side packing in `data-table/decimate.ts` imports this same constant,
 * so the kernel strides and the host packing can't drift.
 */
export const APP_CHUNK = 16;

/**
 * WGSL kernel: per-edge KL-style cost (matches `computeEdgeCost` in
 * `data-table/decimate.ts`).
 *
 * Each thread = one edge (i, j). Reads the per-splat cache for both
 * endpoints, computes the merged Gaussian's covariance + determinant,
 * runs a single Monte-Carlo sample through both component gaussians
 * (the same `z` for both components, matching the CPU implementation),
 * and adds an L2 distance over the appearance (SH) coefficients.
 *
 * @param strideA - Live column count of appearance chunk A (0 if unused).
 * @param strideB - Live column count of appearance chunk B (0 if unused).
 * @param strideC - Live column count of appearance chunk C (0 if unused).
 * @returns WGSL source.
 */
const edgeCostWgsl = (strideA: number, strideB: number, strideC: number) => /* wgsl */`
struct Uniforms {
    edgeCount: u32,
    z0: f32,
    z1: f32,
    z2: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
// Edge list for the current dispatch batch only, split into two parallel
// arrays (avoids a host-side (i, j) interleave). The host uploads each batch's
// slice to offset 0, so we index edgesI/J[bid] directly — keeping these
// buffers batch-sized instead of N·k keeps them off the per-binding limit.
@group(0) @binding(1) var<storage, read> edgesI: array<u32>;
@group(0) @binding(2) var<storage, read> edgesJ: array<u32>;
// Per-splat geometry, interleaved 8-wide:
//   posScalars[8s + 0..2] = position xyz
//   posScalars[8s + 3]    = mass
//   posScalars[8s + 4]    = logdet
//   posScalars[8s + 5..7] = variances (vx, vy, vz)
@group(0) @binding(3) var<storage, read> posScalars: array<f32>;
// Row-major 3x3 rotation matrix per splat (9 floats per splat).
@group(0) @binding(4) var<storage, read> rotR: array<f32>;
// Appearance, split into up to three chunks (≤16 columns each) so no single
// binding exceeds maxStorageBufferBindingSize (~2 GB). Each chunk's stride is
// its live column count (STRIDE_A/B/C below); appA holds columns 0.., appB the
// next span, appC the next. Unused chunks have stride 0, are bound to a dummy
// buffer, and are never read.
@group(0) @binding(5) var<storage, read> appA: array<f32>;
@group(0) @binding(6) var<storage, read> appB: array<f32>;
@group(0) @binding(7) var<storage, read> appC: array<f32>;
// Output: cost per edge.
@group(0) @binding(8) var<storage, read_write> costs: array<f32>;

const EPS_COV: f32 = 1e-8;
const LOG2PI: f32 = 1.8378770664093453;
// Per-chunk appearance strides = live column count in each chunk (0 = unused,
// dummy-bound). Baked here because the column count is fixed for the lifetime
// of the kernel, so loop bounds and indexing resolve statically.
const STRIDE_A: u32 = ${strideA}u;
const STRIDE_B: u32 = ${strideB}u;
const STRIDE_C: u32 = ${strideC}u;

// Symmetric 3x3 covariance helpers — we pass them around as 6 f32 (xx, xy, xz, yy, yz, zz).

// Σ = R · diag(v) · R^T for row-major R (a 9-float array starting at offset r9).
// Variances v come from posScalars[s8 + 5..7]. Result is 6 floats:
// (xx, xy, xz, yy, yz, zz).
fn sigmaFromRotVar(r9: u32, s8: u32) -> array<f32, 6> {
    let r00 = rotR[r9 + 0u]; let r01 = rotR[r9 + 1u]; let r02 = rotR[r9 + 2u];
    let r10 = rotR[r9 + 3u]; let r11 = rotR[r9 + 4u]; let r12 = rotR[r9 + 5u];
    let r20 = rotR[r9 + 6u]; let r21 = rotR[r9 + 7u]; let r22 = rotR[r9 + 8u];
    let vx = posScalars[s8 + 5u];
    let vy = posScalars[s8 + 6u];
    let vz = posScalars[s8 + 7u];
    return array<f32, 6>(
        r00*r00*vx + r01*r01*vy + r02*r02*vz,   // xx
        r00*r10*vx + r01*r11*vy + r02*r12*vz,   // xy
        r00*r20*vx + r01*r21*vy + r02*r22*vz,   // xz
        r10*r10*vx + r11*r11*vy + r12*r12*vz,   // yy
        r10*r20*vx + r11*r21*vy + r12*r22*vz,   // yz
        r20*r20*vx + r21*r21*vy + r22*r22*vz    // zz
    );
}

// log N(x | mu, R · diag(v) · R^T) for a diagonally-decomposed covariance.
// invDiag is (1/vx, 1/vy, 1/vz); ld is logdet of the full covariance.
// Evaluates y = R^T * (x - mu) using columns of R (= rows of Rt).
fn gaussLogpdfDiagrot(
    x: vec3f, mu: vec3f, r9: u32,
    invDiag: vec3f, ld: f32
) -> f32 {
    let dx = x.x - mu.x;
    let dy = x.y - mu.y;
    let dz = x.z - mu.z;
    // y = R^T · d. R is row-major; column k of R is (R[k], R[k+3], R[k+6]).
    let y0 = dx * rotR[r9 + 0u] + dy * rotR[r9 + 3u] + dz * rotR[r9 + 6u];
    let y1 = dx * rotR[r9 + 1u] + dy * rotR[r9 + 4u] + dz * rotR[r9 + 7u];
    let y2 = dx * rotR[r9 + 2u] + dy * rotR[r9 + 5u] + dz * rotR[r9 + 8u];
    let quad = y0*y0*invDiag.x + y1*y1*invDiag.y + y2*y2*invDiag.z;
    return -0.5 * (3.0 * LOG2PI + ld + quad);
}

fn logAddExp(a: f32, b: f32) -> f32 {
    let m = max(a, b);
    return m + log(exp(a - m) + exp(b - m));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let bid = gid.x;
    if (bid >= uniforms.edgeCount) { return; }

    let i = edgesI[bid];
    let j = edgesJ[bid];

    let i8 = i * 8u;
    let j8 = j * 8u;
    let i9 = i * 9u;
    let j9 = j * 9u;

    let mu_i = vec3f(posScalars[i8 + 0u], posScalars[i8 + 1u], posScalars[i8 + 2u]);
    let mu_j = vec3f(posScalars[j8 + 0u], posScalars[j8 + 1u], posScalars[j8 + 2u]);

    let wi = posScalars[i8 + 3u];
    let wj = posScalars[j8 + 3u];
    let W = wi + wj;
    let Wsafe = select(1.0, W, W > 0.0);
    let pi_w_raw = wi / Wsafe;
    let pi_w = clamp(pi_w_raw, 1e-12, 1.0 - 1e-12);
    let pj_w = 1.0 - pi_w;
    let logPi = log(pi_w);
    let logPj = log(pj_w);

    // Merged mean.
    let mm = pi_w * mu_i + pj_w * mu_j;
    let di = mu_i - mm;
    let dj = mu_j - mm;

    // Σ_i and Σ_j from rotation + variances.
    let sig_i = sigmaFromRotVar(i9, i8);
    let sig_j = sigmaFromRotVar(j9, j8);

    // Merged covariance: pi*(Σ_i + δi·δiᵀ) + pj*(Σ_j + δj·δjᵀ), + EPS on diag.
    let s_xx = pi_w * (sig_i[0] + di.x*di.x) + pj_w * (sig_j[0] + dj.x*dj.x) + EPS_COV;
    let s_xy = pi_w * (sig_i[1] + di.x*di.y) + pj_w * (sig_j[1] + dj.x*dj.y);
    let s_xz = pi_w * (sig_i[2] + di.x*di.z) + pj_w * (sig_j[2] + dj.x*dj.z);
    let s_yy = pi_w * (sig_i[3] + di.y*di.y) + pj_w * (sig_j[3] + dj.y*dj.y) + EPS_COV;
    let s_yz = pi_w * (sig_i[4] + di.y*di.z) + pj_w * (sig_j[4] + dj.y*dj.z);
    let s_zz = pi_w * (sig_i[5] + di.z*di.z) + pj_w * (sig_j[5] + dj.z*dj.z) + EPS_COV;

    // det of symmetric 3x3.
    let det_m = s_xx * (s_yy*s_zz - s_yz*s_yz)
              - s_xy * (s_xy*s_zz - s_yz*s_xz)
              + s_xz * (s_xy*s_yz - s_yy*s_xz);
    let logdet_m = log(max(det_m, 1e-30));

    // Entropy of the merged Gaussian: H = 0.5 (k log(2π) + log|Σ_m| + k), k=3.
    let EpNegLogQ = 0.5 * (3.0 * LOG2PI + logdet_m + 3.0);

    // Read per-axis std for each input (variances live at posScalars[s8+5..7]).
    let vix = posScalars[i8 + 5u]; let viy = posScalars[i8 + 6u]; let viz = posScalars[i8 + 7u];
    let vjx = posScalars[j8 + 5u]; let vjy = posScalars[j8 + 6u]; let vjz = posScalars[j8 + 7u];
    let stdix = sqrt(max(vix, 0.0));
    let stdiy = sqrt(max(viy, 0.0));
    let stdiz = sqrt(max(viz, 0.0));
    let stdjx = sqrt(max(vjx, 0.0));
    let stdjy = sqrt(max(vjy, 0.0));
    let stdjz = sqrt(max(vjz, 0.0));

    // Inverse diagonals (1 / variance) for the log-pdf quadratic term.
    let invDi = vec3f(1.0 / max(vix, 1e-30), 1.0 / max(viy, 1e-30), 1.0 / max(viz, 1e-30));
    let invDj = vec3f(1.0 / max(vjx, 1e-30), 1.0 / max(vjy, 1e-30), 1.0 / max(vjz, 1e-30));
    let ldi = posScalars[i8 + 4u];
    let ldj = posScalars[j8 + 4u];

    let z0 = uniforms.z0;
    let z1 = uniforms.z1;
    let z2 = uniforms.z2;

    // Sample x = mu + R · diag(std) · z where z ~ N(0, I).
    // Row a of R is (rotR[r9+3a], rotR[r9+3a+1], rotR[r9+3a+2]).
    // x[a] = mu[a] + R[a][0]*std[0]*z[0] + R[a][1]*std[1]*z[1] + R[a][2]*std[2]*z[2].
    let xix = mu_i.x + z0 * stdix * rotR[i9 + 0u] + z1 * stdiy * rotR[i9 + 1u] + z2 * stdiz * rotR[i9 + 2u];
    let xiy = mu_i.y + z0 * stdix * rotR[i9 + 3u] + z1 * stdiy * rotR[i9 + 4u] + z2 * stdiz * rotR[i9 + 5u];
    let xiz = mu_i.z + z0 * stdix * rotR[i9 + 6u] + z1 * stdiy * rotR[i9 + 7u] + z2 * stdiz * rotR[i9 + 8u];
    let xi = vec3f(xix, xiy, xiz);

    let xjx = mu_j.x + z0 * stdjx * rotR[j9 + 0u] + z1 * stdjy * rotR[j9 + 1u] + z2 * stdjz * rotR[j9 + 2u];
    let xjy = mu_j.y + z0 * stdjx * rotR[j9 + 3u] + z1 * stdjy * rotR[j9 + 4u] + z2 * stdjz * rotR[j9 + 5u];
    let xjz = mu_j.z + z0 * stdjx * rotR[j9 + 6u] + z1 * stdjy * rotR[j9 + 7u] + z2 * stdjz * rotR[j9 + 8u];
    let xj = vec3f(xjx, xjy, xjz);

    // log p_ij at samples from component i.
    let logNiOnI = gaussLogpdfDiagrot(xi, mu_i, i9, invDi, ldi);
    let logNjOnI = gaussLogpdfDiagrot(xi, mu_j, j9, invDj, ldj);
    let logpOnI = logAddExp(logPi + logNiOnI, logPj + logNjOnI);

    let logNiOnJ = gaussLogpdfDiagrot(xj, mu_i, i9, invDi, ldi);
    let logNjOnJ = gaussLogpdfDiagrot(xj, mu_j, j9, invDj, ldj);
    let logpOnJ = logAddExp(logPi + logNiOnJ, logPj + logNjOnJ);

    let Ei = logpOnI;
    let Ej = logpOnJ;
    let EpLogp = pi_w * Ei + pj_w * Ej;
    let geo = EpLogp + EpNegLogQ;

    // Appearance L2 cost, summed across the (up to three) chunks. Each chunk's
    // stride is its live column count, so partial chunks store/read no padding.
    // A 0 stride yields 0 iterations, so the dummy-bound appB / appC are never
    // touched on inputs with fewer SH bands.
    var cSh: f32 = 0.0;
    let iA = i * STRIDE_A; let jA = j * STRIDE_A;
    for (var k: u32 = 0u; k < STRIDE_A; k = k + 1u) {
        let d = appA[iA + k] - appA[jA + k];
        cSh = cSh + d * d;
    }
    let iB = i * STRIDE_B; let jB = j * STRIDE_B;
    for (var k: u32 = 0u; k < STRIDE_B; k = k + 1u) {
        let d = appB[iB + k] - appB[jB + k];
        cSh = cSh + d * d;
    }
    let iC = i * STRIDE_C; let jC = j * STRIDE_C;
    for (var k: u32 = 0u; k < STRIDE_C; k = k + 1u) {
        let d = appC[iC + k] - appC[jC + k];
        cSh = cSh + d * d;
    }

    costs[bid] = geo + cSh;
}
`;

/**
 * Per-splat cache for the edge cost kernel. Packed layouts to stay within the
 * WebGPU per-stage storage-buffer count limit (8) and the per-binding size
 * limit (~2 GB) — appearance is split into 16-column chunks for the latter.
 */
interface EdgeCostCache {
    /** Per-splat geometry interleaved 8-wide: (x, y, z, mass, logdet, vx, vy, vz). */
    posScalars: Float32Array;
    /** Row-major 3×3 rotation per splat (length 9N). */
    rotR: Float32Array;
    /**
     * Appearance in up to three chunks of ≤16 columns. Chunk c has stride
     * width_c (= its live column count): appChunks[c][s * width_c + k].
     */
    appChunks: Float32Array[];
    /** Number of appearance columns C. */
    numAppCols: number;
    /** Number of splats. */
    numSplats: number;
}

/**
 * GPU edge-cost evaluator.
 *
 * Each compute thread evaluates the KL-style cost for one edge (i, j) by
 * reading the per-splat cache for both endpoints, computing the merged
 * Gaussian's covariance/determinant, running a single Monte-Carlo sample
 * through both component PDFs, and adding an L2 distance over the
 * appearance (SH) coefficients. Output is `costs[e] = cost for edge e`.
 *
 * Mirrors the CPU `computeEdgeCost` in `data-table/decimate.ts`.
 */
class GpuEdgeCost {
    /**
     * @param cache - Per-splat cache (uploaded once).
     * @param edgeI - Edge u indices (length E).
     * @param edgeJ - Edge v indices (length E).
     * @param z - Single Monte-Carlo sample (3 floats from N(0,1)).
     * @param outCosts - Destination for per-edge costs (length E).
     */
    execute: (
        cache: EdgeCostCache,
        edgeI: Uint32Array,
        edgeJ: Uint32Array,
        z: Float32Array,
        outCosts: Float32Array
    ) => Promise<void>;
    destroy: () => void;

    /**
     * @param device - PlayCanvas GraphicsDevice (WebGPU).
     * @param maxN - Maximum number of splats.
     * @param maxE - Maximum number of edges in a single dispatch.
     * @param maxAppCols - Maximum appearance column count (over all bands).
     */
    constructor(device: GraphicsDevice, maxN: number, maxE: number, maxAppCols: number) {
        const workgroupSize = 64;
        const edgesPerBatch = 1024 * workgroupSize;  // 65,536
        // Appearance is split at fixed APP_CHUNK-column boundaries, but each
        // chunk's *stride* is its live column count — only the last non-empty
        // chunk is ever partial, so partial chunks neither allocate nor upload
        // padding. The widest possible chunk reaches the ~2 GB limit at ~33.5M
        // splats, past the ~11.2M wall the single 48-col buffer hit. The three
        // kernel bindings (appA/appB/appC) cap the layout at three chunks.
        // e.g. [16, 11, 0] for 27 cols, [3, 0, 0] for DC-only.
        const appStrides = [0, 1, 2].map((ch) => {
            return Math.min(APP_CHUNK, Math.max(0, maxAppCols - ch * APP_CHUNK));
        });
        // Non-empty chunk count the kernel reads. execute() validates the cache
        // supplies exactly this many: a short count would leave a hoisted (reused
        // across iterations) appearance buffer holding the previous iteration's
        // data, which the kernel would then read as this iteration's appearance.
        const numAppChunks = appStrides.filter(stride => stride > 0).length;

        const bindGroupFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('edgesI', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('edgesJ', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('posScalars', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('rotR', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('appA', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('appB', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('appC', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('costs', SHADERSTAGE_COMPUTE)
        ]);

        const shader = new Shader(device, {
            name: 'compute-edge-cost',
            shaderLanguage: SHADERLANGUAGE_WGSL,
            cshader: edgeCostWgsl(appStrides[0], appStrides[1], appStrides[2]),
            // @ts-ignore
            computeUniformBufferFormats: {
                uniforms: new UniformBufferFormat(device, [
                    new UniformFormat('edgeCount', UNIFORMTYPE_UINT),
                    new UniformFormat('z0', UNIFORMTYPE_FLOAT),
                    new UniformFormat('z1', UNIFORMTYPE_FLOAT),
                    new UniformFormat('z2', UNIFORMTYPE_FLOAT)
                ])
            },
            // @ts-ignore
            computeBindGroupFormat: bindGroupFormat
        });

        // Pre-flight the largest per-N bindings against the device's storage
        // limit so we fail with a clear message instead of a driver-side error.
        // Edges are uploaded per batch (batch-sized buffers), so they can't hit
        // the limit; the widest appearance chunk and rotR are the candidates.
        // posScalars (8 floats/splat) is strictly smaller than rotR (9), so the
        // rotR check already bounds it — no separate check needed.
        const maxStorage = (device as any).limits?.maxStorageBufferBindingSize;
        if (typeof maxStorage === 'number') {
            const checkLimit = (label: string, bytes: number) => {
                if (bytes > maxStorage) {
                    throw new Error(
                        `GpuEdgeCost: ${label} buffer (${bytes} bytes) exceeds device ` +
                        `maxStorageBufferBindingSize (${maxStorage})`
                    );
                }
            };
            const maxChunkCols = Math.max(...appStrides);
            checkLimit(`appearance chunk (${maxN} splats × ${maxChunkCols} cols)`, maxN * maxChunkCols * 4);
            checkLimit(`rotR (${maxN} splats × 9)`, maxN * 9 * 4);
        }

        const posScalarsBuf = new StorageBuffer(device, maxN * 8 * 4, BUFFERUSAGE_COPY_DST);
        const rotRBuf = new StorageBuffer(device, maxN * 9 * 4, BUFFERUSAGE_COPY_DST);

        // One buffer per non-empty appearance chunk, sized to that chunk's live
        // column count; empty slots (inputs with fewer SH bands) share a small
        // dummy since WebGPU forbids a zero-size binding. The 3-binding layout
        // stays fixed regardless of band count.
        const appDummy = new StorageBuffer(device, 16, BUFFERUSAGE_COPY_DST);
        const appBufs: StorageBuffer[] = appStrides.map((width) => {
            return width > 0 ?
                new StorageBuffer(device, maxN * width * 4, BUFFERUSAGE_COPY_DST) :
                appDummy;
        });

        // Two parallel u32 buffers, sized to a single dispatch batch (not the
        // full N·k edge list): execute uploads each batch's slice before its
        // dispatch. Batch-sizing keeps these ~256 KB instead of N·k·4 — off the
        // ~2 GB per-binding limit (so edges never cap scene size) and ~1.6 GB
        // less VRAM at 13M splats. Two parallel arrays avoid a host-side pack.
        const edgesIBuf = new StorageBuffer(device, edgesPerBatch * 4, BUFFERUSAGE_COPY_DST);
        const edgesJBuf = new StorageBuffer(device, edgesPerBatch * 4, BUFFERUSAGE_COPY_DST);

        const outBuf = new StorageBuffer(
            device,
            edgesPerBatch * 4,
            BUFFERUSAGE_COPY_SRC | BUFFERUSAGE_COPY_DST
        );
        const outScratch = new Float32Array(edgesPerBatch);

        const compute = new Compute(device, shader, 'compute-edge-cost');
        compute.setParameter('edgesI', edgesIBuf);
        compute.setParameter('edgesJ', edgesJBuf);
        compute.setParameter('posScalars', posScalarsBuf);
        compute.setParameter('rotR', rotRBuf);
        compute.setParameter('appA', appBufs[0]);
        compute.setParameter('appB', appBufs[1]);
        compute.setParameter('appC', appBufs[2]);
        compute.setParameter('costs', outBuf);

        this.execute = async (
            cache: EdgeCostCache,
            edgeI: Uint32Array,
            edgeJ: Uint32Array,
            z: Float32Array,
            outCosts: Float32Array
        ) => {
            const n = cache.numSplats;
            const e = edgeI.length;

            if (n > maxN) throw new Error(`GpuEdgeCost: N=${n} exceeds maxN=${maxN}`);
            if (e > maxE) throw new Error(`GpuEdgeCost: E=${e} exceeds maxE=${maxE}`);
            if (cache.numAppCols !== maxAppCols) {
                throw new Error(`GpuEdgeCost: numAppCols=${cache.numAppCols} must equal maxAppCols=${maxAppCols} (baked into the kernel)`);
            }
            if (cache.appChunks.length !== numAppChunks) {
                throw new Error(`GpuEdgeCost: cache supplies ${cache.appChunks.length} appearance chunks but the kernel layout expects ${numAppChunks}`);
            }
            if (edgeJ.length !== e || outCosts.length !== e) {
                throw new Error('GpuEdgeCost: edgeI / edgeJ / outCosts must have same length');
            }
            if (z.length < 3) {
                throw new Error('GpuEdgeCost: z must have at least 3 elements');
            }

            // Upload per-splat cache. Each appearance chunk is row-major with
            // stride = its live column count, so we upload n*width per chunk.
            posScalarsBuf.write(0, cache.posScalars, 0, n * 8);
            rotRBuf.write(0, cache.rotR, 0, n * 9);
            for (let ch = 0; ch < cache.appChunks.length; ch++) {
                appBufs[ch].write(0, cache.appChunks[ch], 0, n * appStrides[ch]);
            }

            compute.setParameter('z0', z[0]);
            compute.setParameter('z1', z[1]);
            compute.setParameter('z2', z[2]);

            const numBatches = Math.ceil(e / edgesPerBatch);
            for (let batch = 0; batch < numBatches; batch++) {
                const edgeOffset = batch * edgesPerBatch;
                const edgeCount = Math.min(edgesPerBatch, e - edgeOffset);
                const groups = Math.ceil(edgeCount / workgroupSize);

                // Upload just this batch's edges to offset 0; the kernel indexes
                // edgesI/J[bid] within the batch.
                edgesIBuf.write(0, edgeI, edgeOffset, edgeCount);
                edgesJBuf.write(0, edgeJ, edgeOffset, edgeCount);

                compute.setParameter('edgeCount', edgeCount);

                compute.setupDispatch(groups);
                device.computeDispatch([compute], `edge-cost-dispatch-${batch}`);

                const readBytes = edgeCount * 4;
                await outBuf.read(0, readBytes, outScratch, true);
                outCosts.set(outScratch.subarray(0, edgeCount), edgeOffset);
            }
        };

        this.destroy = () => {
            posScalarsBuf.destroy();
            rotRBuf.destroy();
            for (const buf of appBufs) {
                if (buf !== appDummy) buf.destroy();
            }
            appDummy.destroy();
            edgesIBuf.destroy();
            edgesJBuf.destroy();
            outBuf.destroy();
            shader.destroy();
            bindGroupFormat.destroy();
        };
    }
}

export { GpuEdgeCost, type EdgeCostCache };
