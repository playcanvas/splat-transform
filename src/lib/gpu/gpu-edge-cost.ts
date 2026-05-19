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
 * WGSL kernel: per-edge KL-style cost (matches `computeEdgeCost` in
 * `data-table/decimate.ts`).
 *
 * Each thread = one edge (i, j). Reads the per-splat cache for both
 * endpoints, computes the merged Gaussian's covariance + determinant,
 * runs a single Monte-Carlo sample through both component gaussians
 * (the same `z` for both components, matching the CPU implementation),
 * and adds an L2 distance over the appearance (SH) coefficients.
 *
 * @returns WGSL source.
 */
const edgeCostWgsl = () => /* wgsl */`
struct Uniforms {
    edgeOffset: u32,
    edgeCount: u32,
    numAppCols: u32,
    z0: f32,
    z1: f32,
    z2: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
// Edge list, interleaved: edges[2e] = i, edges[2e+1] = j.
@group(0) @binding(1) var<storage, read> edges: array<u32>;
// Positions, interleaved xyz: positions[3s + 0/1/2].
@group(0) @binding(2) var<storage, read> positions: array<f32>;
// Row-major 3x3 rotation matrix per splat (9 floats per splat).
@group(0) @binding(3) var<storage, read> rotR: array<f32>;
// Per-splat scalars packed: splatScalars[5s + 0] = mass,
//                          splatScalars[5s + 1] = logdet,
//                          splatScalars[5s + 2..4] = variances (vx, vy, vz).
@group(0) @binding(4) var<storage, read> splatScalars: array<f32>;
// Appearance: row-major, app[s * C + k] = ch k of splat s.
@group(0) @binding(5) var<storage, read> appearance: array<f32>;
// Output: cost per edge.
@group(0) @binding(6) var<storage, read_write> costs: array<f32>;

const EPS_COV: f32 = 1e-8;
const LOG2PI: f32 = 1.8378770664093453;

// Symmetric 3x3 covariance helpers — we pass them around as 6 f32 (xx, xy, xz, yy, yz, zz).

// Σ = R · diag(v) · R^T for row-major R (a 9-float array starting at offset r9).
// Variances v come from splatScalars[s5 + 2..4]. Result is 6 floats:
// (xx, xy, xz, yy, yz, zz).
fn sigmaFromRotVar(r9: u32, s5: u32) -> array<f32, 6> {
    let r00 = rotR[r9 + 0u]; let r01 = rotR[r9 + 1u]; let r02 = rotR[r9 + 2u];
    let r10 = rotR[r9 + 3u]; let r11 = rotR[r9 + 4u]; let r12 = rotR[r9 + 5u];
    let r20 = rotR[r9 + 6u]; let r21 = rotR[r9 + 7u]; let r22 = rotR[r9 + 8u];
    let vx = splatScalars[s5 + 2u];
    let vy = splatScalars[s5 + 3u];
    let vz = splatScalars[s5 + 4u];
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
    let eIdx = bid + uniforms.edgeOffset;

    let i = edges[eIdx * 2u + 0u];
    let j = edges[eIdx * 2u + 1u];

    let i3 = i * 3u;
    let j3 = j * 3u;
    let i9 = i * 9u;
    let j9 = j * 9u;
    let i5 = i * 5u;
    let j5 = j * 5u;

    let mu_i = vec3f(positions[i3 + 0u], positions[i3 + 1u], positions[i3 + 2u]);
    let mu_j = vec3f(positions[j3 + 0u], positions[j3 + 1u], positions[j3 + 2u]);

    let wi = splatScalars[i5 + 0u];
    let wj = splatScalars[j5 + 0u];
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
    let sig_i = sigmaFromRotVar(i9, i5);
    let sig_j = sigmaFromRotVar(j9, j5);

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

    // Read per-axis std for each input (variances live at splatScalars[s5+2..4]).
    let vix = splatScalars[i5 + 2u]; let viy = splatScalars[i5 + 3u]; let viz = splatScalars[i5 + 4u];
    let vjx = splatScalars[j5 + 2u]; let vjy = splatScalars[j5 + 3u]; let vjz = splatScalars[j5 + 4u];
    let stdix = sqrt(max(vix, 0.0));
    let stdiy = sqrt(max(viy, 0.0));
    let stdiz = sqrt(max(viz, 0.0));
    let stdjx = sqrt(max(vjx, 0.0));
    let stdjy = sqrt(max(vjy, 0.0));
    let stdjz = sqrt(max(vjz, 0.0));

    // Inverse diagonals (1 / variance) for the log-pdf quadratic term.
    let invDi = vec3f(1.0 / max(vix, 1e-30), 1.0 / max(viy, 1e-30), 1.0 / max(viz, 1e-30));
    let invDj = vec3f(1.0 / max(vjx, 1e-30), 1.0 / max(vjy, 1e-30), 1.0 / max(vjz, 1e-30));
    let ldi = splatScalars[i5 + 1u];
    let ldj = splatScalars[j5 + 1u];

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

    // Appearance L2 cost.
    var cSh: f32 = 0.0;
    let C = uniforms.numAppCols;
    let baseI = i * C;
    let baseJ = j * C;
    for (var k: u32 = 0u; k < C; k = k + 1u) {
        let d = appearance[baseI + k] - appearance[baseJ + k];
        cSh = cSh + d * d;
    }

    costs[bid] = geo + cSh;
}
`;

/**
 * Per-splat cache for the edge cost kernel. Packed layouts to stay within
 * the WebGPU per-stage storage-buffer limit (max 10).
 */
interface EdgeCostCache {
    /** N positions interleaved xyz (length 3N). */
    positions: Float32Array;
    /** Row-major 3×3 rotation per splat (length 9N). */
    rotR: Float32Array;
    /** Per-splat scalars interleaved 5-wide: (mass, logdet, vx, vy, vz). */
    splatScalars: Float32Array;
    /** Appearance coefficients, row-major: appearance[s * C + k]. */
    appearance: Float32Array;
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

        const bindGroupFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('edges', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('positions', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('rotR', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('splatScalars', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('appearance', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('costs', SHADERSTAGE_COMPUTE)
        ]);

        const shader = new Shader(device, {
            name: 'compute-edge-cost',
            shaderLanguage: SHADERLANGUAGE_WGSL,
            cshader: edgeCostWgsl(),
            // @ts-ignore
            computeUniformBufferFormats: {
                uniforms: new UniformBufferFormat(device, [
                    new UniformFormat('edgeOffset', UNIFORMTYPE_UINT),
                    new UniformFormat('edgeCount', UNIFORMTYPE_UINT),
                    new UniformFormat('numAppCols', UNIFORMTYPE_UINT),
                    new UniformFormat('z0', UNIFORMTYPE_FLOAT),
                    new UniformFormat('z1', UNIFORMTYPE_FLOAT),
                    new UniformFormat('z2', UNIFORMTYPE_FLOAT)
                ])
            },
            // @ts-ignore
            computeBindGroupFormat: bindGroupFormat
        });

        const positionsBuf = new StorageBuffer(device, maxN * 3 * 4, BUFFERUSAGE_COPY_DST);
        const rotRBuf = new StorageBuffer(device, maxN * 9 * 4, BUFFERUSAGE_COPY_DST);
        const splatScalarsBuf = new StorageBuffer(device, maxN * 5 * 4, BUFFERUSAGE_COPY_DST);
        const appearanceBuf = new StorageBuffer(device, maxN * maxAppCols * 4, BUFFERUSAGE_COPY_DST);

        const edgesBuf = new StorageBuffer(device, maxE * 2 * 4, BUFFERUSAGE_COPY_DST);

        const outBuf = new StorageBuffer(
            device,
            edgesPerBatch * 4,
            BUFFERUSAGE_COPY_SRC | BUFFERUSAGE_COPY_DST
        );
        const outScratch = new Float32Array(edgesPerBatch);

        const compute = new Compute(device, shader, 'compute-edge-cost');
        compute.setParameter('edges', edgesBuf);
        compute.setParameter('positions', positionsBuf);
        compute.setParameter('rotR', rotRBuf);
        compute.setParameter('splatScalars', splatScalarsBuf);
        compute.setParameter('appearance', appearanceBuf);
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
            if (cache.numAppCols > maxAppCols) {
                throw new Error(`GpuEdgeCost: numAppCols=${cache.numAppCols} exceeds maxAppCols=${maxAppCols}`);
            }
            if (edgeJ.length !== e || outCosts.length !== e) {
                throw new Error('GpuEdgeCost: edgeI / edgeJ / outCosts must have same length');
            }
            if (z.length < 3) {
                throw new Error('GpuEdgeCost: z must have at least 3 elements');
            }

            // Upload per-splat cache.
            positionsBuf.write(0, cache.positions, 0, n * 3);
            rotRBuf.write(0, cache.rotR, 0, n * 9);
            splatScalarsBuf.write(0, cache.splatScalars, 0, n * 5);
            appearanceBuf.write(0, cache.appearance, 0, n * cache.numAppCols);

            // Upload edges interleaved (i, j).
            const edgePacked = new Uint32Array(e * 2);
            for (let k = 0; k < e; k++) {
                edgePacked[k * 2] = edgeI[k];
                edgePacked[k * 2 + 1] = edgeJ[k];
            }
            edgesBuf.write(0, edgePacked, 0, e * 2);

            compute.setParameter('numAppCols', cache.numAppCols);
            compute.setParameter('z0', z[0]);
            compute.setParameter('z1', z[1]);
            compute.setParameter('z2', z[2]);

            const numBatches = Math.ceil(e / edgesPerBatch);
            for (let batch = 0; batch < numBatches; batch++) {
                const edgeOffset = batch * edgesPerBatch;
                const edgeCount = Math.min(edgesPerBatch, e - edgeOffset);
                const groups = Math.ceil(edgeCount / workgroupSize);

                compute.setParameter('edgeOffset', edgeOffset);
                compute.setParameter('edgeCount', edgeCount);

                compute.setupDispatch(groups);
                device.computeDispatch([compute], `edge-cost-dispatch-${batch}`);

                const readBytes = edgeCount * 4;
                await outBuf.read(0, readBytes, outScratch, true);
                outCosts.set(outScratch.subarray(0, edgeCount), edgeOffset);
            }
        };

        this.destroy = () => {
            positionsBuf.destroy();
            rotRBuf.destroy();
            splatScalarsBuf.destroy();
            appearanceBuf.destroy();
            edgesBuf.destroy();
            outBuf.destroy();
            shader.destroy();
            bindGroupFormat.destroy();
        };
    }
}

export { GpuEdgeCost, type EdgeCostCache };
