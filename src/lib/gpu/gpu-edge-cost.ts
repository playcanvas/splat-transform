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

import { CACHE_STRIDE } from '../decimate/edge-cost-cpu';
import { gaussianL2Wgsl } from './shaders/chunks/gaussian-l2';

/** Per-splat interleaved stride in the `splat` storage buffer (= the CPU cost-cache layout). */
export const SPLAT_STRIDE = CACHE_STRIDE;

/**
 * WGSL kernel: per-slot L2 field-error cost (mirrors `computeEdgeCost` in
 * `decimate/edge-cost-cpu.ts`).
 *
 * Dense-slot edge model: slot s belongs to owned row qi = s / K and holds a
 * view-local neighbour row (or the 0xFFFFFFFF sentinel for an empty slot,
 * whose cost output is never read — the reduction skips sentinel slots by
 * id). Each thread = one slot.
 * It reads the per-splat cache for both endpoints, moment-matches the merged
 * Gaussian m (mean, covariance, opacity), and evaluates
 *   E = ‖αᵢcᵢGᵢ + αⱼcⱼGⱼ − αₘcₘGₘ‖²
 * in closed form via Gaussian–Gaussian L2 products ⟨G_a,G_b⟩. No Monte-Carlo
 * sampling and no appearance loop — the amplitude is α × base colour, so only
 * the packed base colour (3) and covariance (6) per splat are needed.
 *
 * @param k - Compile-time K, neighbour slots per owned row.
 * @returns WGSL source.
 */
const edgeCostWgsl = (k: number) => /* wgsl */`
struct Uniforms {
    slotBase: u32,
    slotCount: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
// Neighbour rows for the current dispatch batch (host uploads each batch to
// offset 0): view-local row per slot, 0xFFFFFFFF for empty slots.
@group(0) @binding(1) var<storage, read> nbRow: array<u32>;
// Per-splat cache, interleaved ${SPLAT_STRIDE}-wide:
//   [0..2]   mean xyz
//   [3..8]   covariance Σ (xx, xy, xz, yy, yz, zz)
//   [9]      sqrtDet = √|Σ|
//   [10]     alpha (linear opacity)
//   [11]     mass (area·α merge weight)
//   [12..14] base colour (r, g, b)
//   [15]     |base|²
@group(0) @binding(2) var<storage, read> splat: array<f32>;
// Output: cost per slot.
@group(0) @binding(3) var<storage, read_write> costs: array<f32>;

const K: u32 = ${k}u;
const SENTINEL: u32 = 0xFFFFFFFFu;
${gaussianL2Wgsl}

// Gaussian cross-product ⟨G_a,G_b⟩ for symmetric M = Σ_a+Σ_b (6: xx,xy,xz,yy,yz,zz)
// and mean offset d, scaled by √|Σ_a|·√|Σ_b| (passed as sdA·sdB).
fn crossG(sdA: f32, sdB: f32, m: array<f32, 6>, d: vec3f) -> f32 {
    let c00 = m[3] * m[5] - m[4] * m[4];
    let c01 = m[2] * m[4] - m[1] * m[5];
    let c02 = m[1] * m[4] - m[2] * m[3];
    let c11 = m[0] * m[5] - m[2] * m[2];
    let c12 = m[1] * m[2] - m[0] * m[4];
    let c22 = m[0] * m[3] - m[1] * m[1];
    let det = max(m[0] * c00 + m[1] * c01 + m[2] * c02, 1e-30);
    let quadAdj = c00 * d.x * d.x + c11 * d.y * d.y + c22 * d.z * d.z +
        2.0 * (c01 * d.x * d.y + c02 * d.x * d.z + c12 * d.y * d.z);
    let quad = quadAdj / det;
    return TWO_PI_1_5 * sdA * sdB / sqrt(det) * exp(-0.5 * quad);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let bid = gid.x;
    if (bid >= uniforms.slotCount) { return; }

    // Empty slot: the reduction skips sentinel slots by id, so the cost value
    // is never read (WGSL cannot materialize an f32 infinity to park here).
    let row = nbRow[bid];
    if (row == SENTINEL) {
        costs[bid] = 0.0;
        return;
    }

    let io = ((uniforms.slotBase + bid) / K) * ${SPLAT_STRIDE}u;
    let jo = row * ${SPLAT_STRIDE}u;

    let mui = vec3f(splat[io + 0u], splat[io + 1u], splat[io + 2u]);
    let si = array<f32, 6>(splat[io + 3u], splat[io + 4u], splat[io + 5u], splat[io + 6u], splat[io + 7u], splat[io + 8u]);
    let sdi = splat[io + 9u]; let ai = splat[io + 10u]; let mi = splat[io + 11u];
    let bi = vec3f(splat[io + 12u], splat[io + 13u], splat[io + 14u]); let bni = splat[io + 15u];

    let muj = vec3f(splat[jo + 0u], splat[jo + 1u], splat[jo + 2u]);
    let sj = array<f32, 6>(splat[jo + 3u], splat[jo + 4u], splat[jo + 5u], splat[jo + 6u], splat[jo + 7u], splat[jo + 8u]);
    let sdj = splat[jo + 9u]; let aj = splat[jo + 10u]; let mj = splat[jo + 11u];
    let bj = vec3f(splat[jo + 12u], splat[jo + 13u], splat[jo + 14u]); let bnj = splat[jo + 15u];

    let W = mi + mj;
    let pi_ = select(0.5, mi / W, W > 0.0);
    let pj_ = 1.0 - pi_;

    let mm = pi_ * mui + pj_ * muj;
    let di = mui - mm;
    let dj = muj - mm;

    // Merged covariance: Σ pₖ(δₖδₖᵀ + Σₖ) + EPS_COV·I.
    let sm = array<f32, 6>(
        pi_ * (di.x * di.x + si[0]) + pj_ * (dj.x * dj.x + sj[0]) + EPS_COV,
        pi_ * (di.x * di.y + si[1]) + pj_ * (dj.x * dj.y + sj[1]),
        pi_ * (di.x * di.z + si[2]) + pj_ * (dj.x * dj.z + sj[2]),
        pi_ * (di.y * di.y + si[3]) + pj_ * (dj.y * dj.y + sj[3]) + EPS_COV,
        pi_ * (di.y * di.z + si[4]) + pj_ * (dj.y * dj.z + sj[4]),
        pi_ * (di.z * di.z + si[5]) + pj_ * (dj.z * dj.z + sj[5]) + EPS_COV
    );

    let detm = max(
        sm[0] * (sm[3] * sm[5] - sm[4] * sm[4]) - sm[1] * (sm[1] * sm[5] - sm[4] * sm[2]) + sm[2] * (sm[1] * sm[4] - sm[3] * sm[2]),
        1e-30
    );
    let sqrtDetM = sqrt(detm);

    // Merged opacity: mass-conserving, capped — needs merged scales (eigenvalues
    // of Σ_m, Smith's closed form for a symmetric 3×3).
    let e = eig3(sm);
    let s0 = sqrt(max(e.x, 1e-18));
    let s1 = sqrt(max(e.y, 1e-18));
    let s2 = sqrt(max(e.z, 1e-18));
    let am = min(1.0, W / max(ellipsoidArea(s0, s1, s2), 1e-30));

    // Base-colour dots (merged colour = mass-weighted average).
    let bij = dot(bi, bj);
    let bim = pi_ * bni + pj_ * bij;
    let bjm = pi_ * bij + pj_ * bnj;
    let bnm = pi_ * pi_ * bni + 2.0 * pi_ * pj_ * bij + pj_ * pj_ * bnj;

    let selfI = ai * ai * bni * PI_1_5 * sdi;
    let selfJ = aj * aj * bnj * PI_1_5 * sdj;
    let selfM = am * am * bnm * PI_1_5 * sqrtDetM;

    let mij = array<f32, 6>(si[0] + sj[0], si[1] + sj[1], si[2] + sj[2], si[3] + sj[3], si[4] + sj[4], si[5] + sj[5]);
    let cIJ = crossG(sdi, sdj, mij, mui - muj);
    let mim = array<f32, 6>(si[0] + sm[0], si[1] + sm[1], si[2] + sm[2], si[3] + sm[3], si[4] + sm[4], si[5] + sm[5]);
    let cIM = crossG(sdi, sqrtDetM, mim, di);
    let mjm = array<f32, 6>(sj[0] + sm[0], sj[1] + sm[1], sj[2] + sm[2], sj[3] + sm[3], sj[4] + sm[4], sj[5] + sm[5]);
    let cJM = crossG(sdj, sqrtDetM, mjm, dj);

    let E = selfI + selfJ + selfM +
        2.0 * ai * aj * bij * cIJ -
        2.0 * ai * am * bim * cIM -
        2.0 * aj * am * bjm * cJM;

    // Clamp float-noise negatives on both squared norms (the colour distance
    // uses separately-rounded norms and can round slightly negative) but
    // preserve NaN/Inf (degenerate input → the host fails loud on no finite
    // merges). |Δbase|² = bni + bnj − 2·(bi·bj).
    let dc2 = bni + bnj - 2.0 * bij;
    costs[bid] = select(E, 0.0, E < 0.0) + COLOR_WEIGHT * select(dc2, 0.0, dc2 < 0.0);
}
`;

/**
 * GPU edge-cost evaluator.
 *
 * Each compute thread evaluates the L2 field-error cost for one dense slot
 * (owned row qi = slot / K vs its slot's neighbour row), mirroring the CPU
 * `computeEdgeCost` in `decimate/edge-cost-cpu.ts`. Empty (sentinel) slots
 * cost +Inf. Output is `costs[s] = cost for slot s`.
 */
class GpuEdgeCost {
    execute: (
        splatData: Float32Array,
        numSplats: number,
        nbRows: Uint32Array,
        outCosts: Float32Array
    ) => Promise<void>;
    destroy: () => void;

    /**
     * @param device - PlayCanvas GraphicsDevice (WebGPU).
     * @param maxN - Maximum number of splats.
     * @param k - Neighbour slots per owned row.
     */
    constructor(device: GraphicsDevice, maxN: number, k: number) {
        const workgroupSize = 64;
        // Slots per dispatch: bounded by the 65,535 workgroups-per-dimension
        // limit. One blocking readback per ~4.2M slots (a 2M-row block costs
        // 8 round trips, not 512 as with 65,536-edge batches).
        const slotsPerBatch = 65535 * workgroupSize;  // 4,194,240

        const bindGroupFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('nbRow', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('splat', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('costs', SHADERSTAGE_COMPUTE)
        ]);

        const shader = new Shader(device, {
            name: 'compute-edge-cost',
            shaderLanguage: SHADERLANGUAGE_WGSL,
            cshader: edgeCostWgsl(k),
            // @ts-ignore
            computeUniformBufferFormats: {
                uniforms: new UniformBufferFormat(device, [
                    new UniformFormat('slotBase', UNIFORMTYPE_UINT),
                    new UniformFormat('slotCount', UNIFORMTYPE_UINT)
                ])
            },
            // @ts-ignore
            computeBindGroupFormat: bindGroupFormat
        });

        // Pre-flight the per-N splat buffer against the device's storage limit
        // so we fail with a clear message instead of a driver-side error.
        const maxStorage = (device as any).limits?.maxStorageBufferBindingSize;
        if (typeof maxStorage === 'number') {
            const bytes = maxN * SPLAT_STRIDE * 4;
            if (bytes > maxStorage) {
                throw new Error(
                    `GpuEdgeCost: splat buffer (${maxN} splats × ${SPLAT_STRIDE} floats = ${bytes} bytes) ` +
                    `exceeds device maxStorageBufferBindingSize (${maxStorage})`
                );
            }
        }

        const splatBuf = new StorageBuffer(device, maxN * SPLAT_STRIDE * 4, BUFFERUSAGE_COPY_DST);

        // Neighbour-row buffer sized to a single dispatch batch (not the full
        // N·k slot list): execute uploads each batch's slice before its
        // dispatch, keeping it off the ~2 GB per-binding limit.
        const nbRowBuf = new StorageBuffer(device, slotsPerBatch * 4, BUFFERUSAGE_COPY_DST);

        const outBuf = new StorageBuffer(
            device,
            slotsPerBatch * 4,
            BUFFERUSAGE_COPY_SRC | BUFFERUSAGE_COPY_DST
        );
        const outScratch = new Float32Array(slotsPerBatch);

        const compute = new Compute(device, shader, 'compute-edge-cost');
        compute.setParameter('nbRow', nbRowBuf);
        compute.setParameter('splat', splatBuf);
        compute.setParameter('costs', outBuf);

        this.execute = async (
            splatData: Float32Array,
            numSplats: number,
            nbRows: Uint32Array,
            outCosts: Float32Array
        ) => {
            const n = numSplats;
            const s = nbRows.length;

            if (n > maxN) throw new Error(`GpuEdgeCost: N=${n} exceeds maxN=${maxN}`);
            if (outCosts.length !== s) {
                throw new Error('GpuEdgeCost: nbRows / outCosts must have same length');
            }
            if (s % k !== 0) throw new Error(`GpuEdgeCost: slot count ${s} must be a multiple of k=${k}`);

            splatBuf.write(0, splatData, 0, n * SPLAT_STRIDE);

            const numBatches = Math.ceil(s / slotsPerBatch);
            for (let batch = 0; batch < numBatches; batch++) {
                const slotBase = batch * slotsPerBatch;
                const slotCount = Math.min(slotsPerBatch, s - slotBase);
                const groups = Math.ceil(slotCount / workgroupSize);

                nbRowBuf.write(0, nbRows, slotBase, slotCount);

                compute.setParameter('slotBase', slotBase);
                compute.setParameter('slotCount', slotCount);

                compute.setupDispatch(groups);
                device.computeDispatch([compute], `edge-cost-dispatch-${batch}`);

                const readBytes = slotCount * 4;
                await outBuf.read(0, readBytes, outScratch, true);
                outCosts.set(outScratch.subarray(0, slotCount), slotBase);
            }
        };

        this.destroy = () => {
            splatBuf.destroy();
            nbRowBuf.destroy();
            outBuf.destroy();
            shader.destroy();
            bindGroupFormat.destroy();
        };
    }
}

export { GpuEdgeCost };
