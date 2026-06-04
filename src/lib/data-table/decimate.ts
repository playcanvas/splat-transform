import { Column, DataTable } from './data-table';
import { APP_CHUNK, type EdgeCostCache, GpuEdgeCost } from '../gpu/gpu-edge-cost';
import { GpuKnn } from '../gpu/gpu-knn';
import { KdTree } from '../spatial/kd-tree';
import {
    isFloatBitsNonFinite,
    RadixSortScratch,
    radixSortIndicesByFloat
} from '../spatial/radix-sort';
import { type DeviceCreator } from '../types';
import { logger } from '../utils';

const LOG2PI = Math.log(2 * Math.PI);
const KNN_K = 16;
const MC_SAMPLES = 1;
const EPS_COV = 1e-8;
const PROGRESS_TICKS = 100;

// ---------- sigmoid / logit ----------

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

const logit = (p: number) => {
    p = Math.max(1e-7, Math.min(1 - 1e-7, p));
    return Math.log(p / (1 - p));
};

const logAddExp = (a: number, b: number) => {
    if (a === -Infinity) return b;
    if (b === -Infinity) return a;
    const m = Math.max(a, b);
    return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
};

// ---------- PRNG ----------

const mulberry32 = (seed: number) => {
    return () => {
        let t = (seed += 0x6d2b79f5);
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
};

const makeGaussianSamples = (n: number, seed: number): Float64Array[] => {
    const rand = mulberry32(seed >>> 0);
    const out: Float64Array[] = [];
    while (out.length < n) {
        const u1 = Math.max(rand(), 1e-12);
        const u2 = rand();
        const u3 = Math.max(rand(), 1e-12);
        const u4 = rand();
        const r1 = Math.sqrt(-2 * Math.log(u1));
        const t1 = 2 * Math.PI * u2;
        const r2 = Math.sqrt(-2 * Math.log(u3));
        const t2 = 2 * Math.PI * u4;
        out.push(new Float64Array([r1 * Math.cos(t1), r1 * Math.sin(t1), r2 * Math.cos(t2)]));
    }
    return out;
};

// ---------- 3x3 matrix helpers (row-major, 9 floats) ----------

const quatToRotmat = (qw: number, qx: number, qy: number, qz: number, out: Float32Array | Float64Array, o: number) => {
    const xx = qx * qx, yy = qy * qy, zz = qz * qz;
    const wx = qw * qx, wy = qw * qy, wz = qw * qz;
    const xy = qx * qy, xz = qx * qz, yz = qy * qz;
    out[o] = 1 - 2 * (yy + zz);
    out[o + 1] = 2 * (xy - wz);
    out[o + 2] = 2 * (xz + wy);
    out[o + 3] = 2 * (xy + wz);
    out[o + 4] = 1 - 2 * (xx + zz);
    out[o + 5] = 2 * (yz - wx);
    out[o + 6] = 2 * (xz - wy);
    out[o + 7] = 2 * (yz + wx);
    out[o + 8] = 1 - 2 * (xx + yy);
};

const sigmaFromRotVar = (R: Float32Array | Float64Array, r: number, vx: number, vy: number, vz: number, out: Float32Array | Float64Array, o: number) => {
    const r00 = R[r], r01 = R[r + 1], r02 = R[r + 2];
    const r10 = R[r + 3], r11 = R[r + 4], r12 = R[r + 5];
    const r20 = R[r + 6], r21 = R[r + 7], r22 = R[r + 8];
    out[o] = r00 * r00 * vx + r01 * r01 * vy + r02 * r02 * vz;
    out[o + 1] = r00 * r10 * vx + r01 * r11 * vy + r02 * r12 * vz;
    out[o + 2] = r00 * r20 * vx + r01 * r21 * vy + r02 * r22 * vz;
    out[o + 3] = out[o + 1];
    out[o + 4] = r10 * r10 * vx + r11 * r11 * vy + r12 * r12 * vz;
    out[o + 5] = r10 * r20 * vx + r11 * r21 * vy + r12 * r22 * vz;
    out[o + 6] = out[o + 2];
    out[o + 7] = out[o + 5];
    out[o + 8] = r20 * r20 * vx + r21 * r21 * vy + r22 * r22 * vz;
};

const det3 = (A: Float64Array, o: number) => {
    return (
        A[o] * (A[o + 4] * A[o + 8] - A[o + 5] * A[o + 7]) -
        A[o + 1] * (A[o + 3] * A[o + 8] - A[o + 5] * A[o + 6]) +
        A[o + 2] * (A[o + 3] * A[o + 7] - A[o + 4] * A[o + 6])
    );
};

const gaussLogpdfDiagrot = (
    x: number, y: number, z: number,
    mx: number, my: number, mz: number,
    R: Float32Array | Float64Array, ro: number,
    invx: number, invy: number, invz: number, logdet: number
) => {
    const dx = x - mx, dy = y - my, dz = z - mz;
    const y0 = dx * R[ro] + dy * R[ro + 3] + dz * R[ro + 6];
    const y1 = dx * R[ro + 1] + dy * R[ro + 4] + dz * R[ro + 7];
    const y2 = dx * R[ro + 2] + dy * R[ro + 5] + dz * R[ro + 8];
    const quad = y0 * y0 * invx + y1 * y1 * invy + y2 * y2 * invz;
    return -0.5 * (3 * LOG2PI + logdet + quad);
};

// Jacobi eigendecomposition for 3x3 symmetric matrix (full 9-element row-major).
// `A` and `V` are caller-provided 9-element scratch buffers; `Ain` is copied
// into `A` on entry, and on return `A`'s diagonal holds the eigenvalues and
// `V` holds the eigenvectors as columns. Returns nothing — caller reads from
// the buffers it passed in. This shape exists to avoid the per-call
// allocations the merge loop would otherwise burn at ~Nlog2(N) rate.
const eigenSymmetric3x3 = (Ain: Float64Array, A: Float64Array, V: Float64Array) => {
    A.set(Ain);
    V[0] = 1; V[1] = 0; V[2] = 0;
    V[3] = 0; V[4] = 1; V[5] = 0;
    V[6] = 0; V[7] = 0; V[8] = 1;

    for (let iter = 0; iter < 24; iter++) {
        let p = 0, q = 1;
        let maxAbs = Math.abs(A[1]);
        if (Math.abs(A[2]) > maxAbs) {
            p = 0; q = 2; maxAbs = Math.abs(A[2]);
        }
        if (Math.abs(A[5]) > maxAbs) {
            p = 1; q = 2; maxAbs = Math.abs(A[5]);
        }
        if (maxAbs < 1e-12) break;

        const pp = 3 * p + p, qq = 3 * q + q, pq = 3 * p + q;
        const app = A[pp], aqq = A[qq], apq = A[pq];
        const tau = (aqq - app) / (2 * apq);
        const t = Math.sign(tau) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = t * c;

        for (let k = 0; k < 3; k++) {
            if (k === p || k === q) continue;
            const kp = 3 * k + p, kq = 3 * k + q;
            const pk = 3 * p + k, qk = 3 * q + k;
            const akp = A[kp], akq = A[kq];
            A[kp] = c * akp - s * akq;
            A[pk] = A[kp];
            A[kq] = s * akp + c * akq;
            A[qk] = A[kq];
        }
        A[pp] = c * c * app - 2 * s * c * apq + s * s * aqq;
        A[qq] = s * s * app + 2 * s * c * apq + c * c * aqq;
        A[pq] = 0; A[3 * q + p] = 0;

        for (let k = 0; k < 3; k++) {
            const kp = 3 * k + p, kq = 3 * k + q;
            const vkp = V[kp], vkq = V[kq];
            V[kp] = c * vkp - s * vkq;
            V[kq] = s * vkp + c * vkq;
        }
    }
};

// Writes the unit quaternion (w, x, y, z) to `out[oo..oo+3]`. The previous
// version returned a fresh Float64Array per call — at ~9M merge calls that's
// 280 MB of throwaway garbage per decimate.
const rotmatToQuat = (R: Float64Array, o: number, out: Float64Array, oo: number) => {
    const m00 = R[o], m11 = R[o + 4], m22 = R[o + 8];
    const tr = m00 + m11 + m22;
    let qw: number, qx: number, qy: number, qz: number;

    if (tr > 0) {
        const S = Math.sqrt(tr + 1) * 2;
        qw = 0.25 * S;
        qx = (R[o + 7] - R[o + 5]) / S;
        qy = (R[o + 2] - R[o + 6]) / S;
        qz = (R[o + 3] - R[o + 1]) / S;
    } else if (R[o] > R[o + 4] && R[o] > R[o + 8]) {
        const S = Math.sqrt(1 + R[o] - R[o + 4] - R[o + 8]) * 2;
        qw = (R[o + 7] - R[o + 5]) / S;
        qx = 0.25 * S;
        qy = (R[o + 1] + R[o + 3]) / S;
        qz = (R[o + 2] + R[o + 6]) / S;
    } else if (R[o + 4] > R[o + 8]) {
        const S = Math.sqrt(1 + R[o + 4] - R[o] - R[o + 8]) * 2;
        qw = (R[o + 2] - R[o + 6]) / S;
        qx = (R[o + 1] + R[o + 3]) / S;
        qy = 0.25 * S;
        qz = (R[o + 5] + R[o + 7]) / S;
    } else {
        const S = Math.sqrt(1 + R[o + 8] - R[o] - R[o + 4]) * 2;
        qw = (R[o + 3] - R[o + 1]) / S;
        qx = (R[o + 2] + R[o + 6]) / S;
        qy = (R[o + 5] + R[o + 7]) / S;
        qz = 0.25 * S;
    }

    const n = Math.hypot(qw, qx, qy, qz);
    const inv = 1 / Math.max(n, 1e-12);
    out[oo] = qw * inv;
    out[oo + 1] = qx * inv;
    out[oo + 2] = qy * inv;
    out[oo + 3] = qz * inv;
};

// ====================== ELLIPSOID AREA ======================

// Knud Thomsen p=1.6075 approximation for ellipsoid surface area.
// Used as the per-splat screen-projection weight in the pairwise merge.
const ELLIPSOID_P = 1.6075;
const ellipsoidArea = (sx: number, sy: number, sz: number): number => {
    const a = Math.pow(sx * sy, ELLIPSOID_P);
    const b = Math.pow(sx * sz, ELLIPSOID_P);
    const c = Math.pow(sy * sz, ELLIPSOID_P);
    return 4 * Math.PI * Math.pow((a + b + c) / 3, 1 / ELLIPSOID_P);
};

// ====================== PER-SPLAT CACHE ======================

// Float32 throughout — at 17.9M splats the cache hits ~2.5GB at Float64 and the
// downstream cost is a noisy single-MC-sample heuristic anyway, so f32
// precision is fine. `Rt` is dropped (derivable from `R` via index swap at
// call sites); `invdiag` is kept since it amortises ~6 divisions per edge
// across ~k edges per splat.
interface SplatCache {
    R: Float32Array;       // row-major 3×3 rotation per splat
    v: Float32Array;       // variances (scale^2 + eps) per axis
    invdiag: Float32Array; // 1 / v per axis, precomputed
    logdet: Float32Array;
    sigma: Float32Array;   // full 9-element covariance per splat
    mass: Float32Array;
}

// `forGpu` skips the cache fields the GPU kernel recomputes on the fly —
// `invdiag` (1/v, computed in the kernel) and `sigma` (= R·diag(v)·Rᵀ, also
// computed in the kernel). On a 17.9M-splat run that's ~860 MB less held for
// the entire decimate.
const buildPerSplatCache = (
    n: number,
    cop: any, cs0: any, cs1: any, cs2: any,
    cr0: any, cr1: any, cr2: any, cr3: any,
    forGpu: boolean
): SplatCache => {
    const R = new Float32Array(n * 9);
    const v = new Float32Array(n * 3);
    const invdiag = forGpu ? new Float32Array(0) : new Float32Array(n * 3);
    const logdet = new Float32Array(n);
    const sigma = forGpu ? new Float32Array(0) : new Float32Array(n * 9);
    const mass = new Float32Array(n);

    for (let i = 0; i < n; i++) {
        const i3 = 3 * i;
        const i9 = 9 * i;

        const linAlpha = sigmoid(cop[i]);
        const sx = Math.max(Math.exp(cs0[i] as number), 1e-12);
        const sy = Math.max(Math.exp(cs1[i] as number), 1e-12);
        const sz = Math.max(Math.exp(cs2[i] as number), 1e-12);

        const vx = sx * sx + EPS_COV;
        const vy = sy * sy + EPS_COV;
        const vz = sz * sz + EPS_COV;

        v[i3] = vx; v[i3 + 1] = vy; v[i3 + 2] = vz;
        if (!forGpu) {
            invdiag[i3] = 1 / Math.max(vx, 1e-30);
            invdiag[i3 + 1] = 1 / Math.max(vy, 1e-30);
            invdiag[i3 + 2] = 1 / Math.max(vz, 1e-30);
        }
        logdet[i] = Math.log(Math.max(vx, 1e-30)) + Math.log(Math.max(vy, 1e-30)) + Math.log(Math.max(vz, 1e-30));

        // Normalize quaternion before building rotation
        let qw = cr0[i] as number, qx = cr1[i] as number, qy = cr2[i] as number, qz = cr3[i] as number;
        const qn = Math.hypot(qw, qx, qy, qz);
        const invq = 1 / Math.max(qn, 1e-12);
        qw *= invq; qx *= invq; qy *= invq; qz *= invq;

        quatToRotmat(qw, qx, qy, qz, R, i9);
        if (!forGpu) {
            sigmaFromRotVar(R, i9, vx, vy, vz, sigma, i9);
        }

        // Area·α weighting — matches the merge formula in `momentMatch`, so
        // the KL cost in `computeEdgeCost` (which uses `cache.mass` to derive
        // its pi/pj mixture weights) optimises the same merge model that's
        // actually applied. Volume·α here would mis-score pairs whose
        // volume/area ratio differs strongly between members.
        mass[i] = linAlpha * ellipsoidArea(sx, sy, sz) + 1e-12;
    }

    return { R, v, invdiag, logdet, sigma, mass };
};

// ====================== COST FUNCTION ======================

// Reusable scratch shared by `computeEdgeCost` and `momentMatch`. Allocated
// once per `simplifyGaussians` invocation (not per call) so concurrent
// invocations don't share state. Per-call allocation at ~9M merges/iter
// would burn ~5 GB of throwaway garbage; per-decimate allocation is ~600 B.
interface MergeScratch {
    sigm: Float64Array;  // computeEdgeCost: merged covariance
    sigI: Float64Array;  // momentMatch: input i covariance
    sigJ: Float64Array;  // momentMatch: input j covariance
    rI: Float64Array;    // momentMatch: input i rotation
    rJ: Float64Array;    // momentMatch: input j rotation
    sig: Float64Array;   // momentMatch: merged covariance
    rM: Float64Array;    // momentMatch: merged rotation
    eigA: Float64Array;  // eigenSymmetric3x3: working matrix
    eigV: Float64Array;  // eigenSymmetric3x3: eigenvectors
}

const createMergeScratch = (): MergeScratch => ({
    sigm: new Float64Array(9),
    sigI: new Float64Array(9),
    sigJ: new Float64Array(9),
    rI: new Float64Array(9),
    rJ: new Float64Array(9),
    sig: new Float64Array(9),
    rM: new Float64Array(9),
    eigA: new Float64Array(9),
    eigV: new Float64Array(9)
});

const computeEdgeCost = (
    i: number, j: number,
    cx: any, cy: any, cz: any,
    cache: SplatCache,
    Z: Float64Array[],
    appData: any[], appColCount: number,
    scratch: MergeScratch
): number => {
    const i3 = 3 * i, j3 = 3 * j;
    const i9 = 9 * i, j9 = 9 * j;

    const mux = cx[i] as number, muy = cy[i] as number, muz = cz[i] as number;
    const mvx = cx[j] as number, mvy = cy[j] as number, mvz = cz[j] as number;

    const wi = cache.mass[i], wj = cache.mass[j];
    const W = wi + wj;
    const Wsafe = W > 0 ? W : 1;

    let pi = wi / Wsafe;
    pi = Math.max(1e-12, Math.min(1 - 1e-12, pi));
    const pj = 1 - pi;
    const logPi = Math.log(pi);
    const logPj = Math.log(pj);

    // Merged mean
    const mmx = pi * mux + pj * mvx;
    const mmy = pi * muy + pj * mvy;
    const mmz = pi * muz + pj * mvz;

    const dix = mux - mmx, diy = muy - mmy, diz = muz - mmz;
    const djx = mvx - mmx, djy = mvy - mmy, djz = mvz - mmz;

    // Merged covariance (full 9-element, reuse preallocated buffer)
    const sigm = scratch.sigm;
    for (let a = 0; a < 9; a++) {
        sigm[a] = pi * cache.sigma[i9 + a] + pj * cache.sigma[j9 + a];
    }
    sigm[0] += pi * dix * dix + pj * djx * djx;
    sigm[1] += pi * dix * diy + pj * djx * djy;
    sigm[2] += pi * dix * diz + pj * djx * djz;
    sigm[3] += pi * diy * dix + pj * djy * djx;
    sigm[4] += pi * diy * diy + pj * djy * djy;
    sigm[5] += pi * diy * diz + pj * djy * djz;
    sigm[6] += pi * diz * dix + pj * djz * djx;
    sigm[7] += pi * diz * diy + pj * djz * djy;
    sigm[8] += pi * diz * diz + pj * djz * djz;

    // Force symmetry + regularize
    sigm[1] = sigm[3] = 0.5 * (sigm[1] + sigm[3]);
    sigm[2] = sigm[6] = 0.5 * (sigm[2] + sigm[6]);
    sigm[5] = sigm[7] = 0.5 * (sigm[5] + sigm[7]);
    sigm[0] += EPS_COV;
    sigm[4] += EPS_COV;
    sigm[8] += EPS_COV;

    const detm = Math.max(det3(sigm, 0), 1e-30);
    const logdetm = Math.log(detm);

    // E_p[-log q_m] computed analytically as entropy of merged Gaussian
    const EpNegLogQ = 0.5 * (3 * LOG2PI + logdetm + 3);

    // Sample from each component separately with same z-vectors
    const stdix = Math.sqrt(Math.max(cache.v[i3], 0));
    const stdiy = Math.sqrt(Math.max(cache.v[i3 + 1], 0));
    const stdiz = Math.sqrt(Math.max(cache.v[i3 + 2], 0));
    const stdjx = Math.sqrt(Math.max(cache.v[j3], 0));
    const stdjy = Math.sqrt(Math.max(cache.v[j3 + 1], 0));
    const stdjz = Math.sqrt(Math.max(cache.v[j3 + 2], 0));

    let sumLogpOnI = 0;
    let sumLogpOnJ = 0;

    for (let s = 0; s < Z.length; s++) {
        const z0 = Z[s][0], z1 = Z[s][1], z2 = Z[s][2];

        // Sample x = mu + R · diag(std) · z (same form as the GPU kernel).
        // Each row a of R sits at R[i9 + 3a .. i9 + 3a + 2].
        const xix = mux + z0 * stdix * cache.R[i9 + 0] + z1 * stdiy * cache.R[i9 + 1] + z2 * stdiz * cache.R[i9 + 2];
        const xiy = muy + z0 * stdix * cache.R[i9 + 3] + z1 * stdiy * cache.R[i9 + 4] + z2 * stdiz * cache.R[i9 + 5];
        const xiz = muz + z0 * stdix * cache.R[i9 + 6] + z1 * stdiy * cache.R[i9 + 7] + z2 * stdiz * cache.R[i9 + 8];

        const xjx = mvx + z0 * stdjx * cache.R[j9 + 0] + z1 * stdjy * cache.R[j9 + 1] + z2 * stdjz * cache.R[j9 + 2];
        const xjy = mvy + z0 * stdjx * cache.R[j9 + 3] + z1 * stdjy * cache.R[j9 + 4] + z2 * stdjz * cache.R[j9 + 5];
        const xjz = mvz + z0 * stdjx * cache.R[j9 + 6] + z1 * stdjy * cache.R[j9 + 7] + z2 * stdjz * cache.R[j9 + 8];

        // Evaluate log p_ij at samples from component i
        const logNiOnI = gaussLogpdfDiagrot(xix, xiy, xiz, mux, muy, muz,
            cache.R, i9, cache.invdiag[i3], cache.invdiag[i3 + 1], cache.invdiag[i3 + 2], cache.logdet[i]);
        const logNjOnI = gaussLogpdfDiagrot(xix, xiy, xiz, mvx, mvy, mvz,
            cache.R, j9, cache.invdiag[j3], cache.invdiag[j3 + 1], cache.invdiag[j3 + 2], cache.logdet[j]);
        sumLogpOnI += logAddExp(logPi + logNiOnI, logPj + logNjOnI);

        // Evaluate log p_ij at samples from component j
        const logNiOnJ = gaussLogpdfDiagrot(xjx, xjy, xjz, mux, muy, muz,
            cache.R, i9, cache.invdiag[i3], cache.invdiag[i3 + 1], cache.invdiag[i3 + 2], cache.logdet[i]);
        const logNjOnJ = gaussLogpdfDiagrot(xjx, xjy, xjz, mvx, mvy, mvz,
            cache.R, j9, cache.invdiag[j3], cache.invdiag[j3 + 1], cache.invdiag[j3 + 2], cache.logdet[j]);
        sumLogpOnJ += logAddExp(logPi + logNiOnJ, logPj + logNjOnJ);
    }

    const Ei = sumLogpOnI / Z.length;
    const Ej = sumLogpOnJ / Z.length;
    const EpLogp = pi * Ei + pj * Ej;
    const geo = EpLogp + EpNegLogQ;

    // Appearance cost
    let cSh = 0;
    for (let k = 0; k < appColCount; k++) {
        const d = (appData[k][i] as number) - (appData[k][j] as number);
        cSh += d * d;
    }

    return geo + cSh;
};

// ====================== MERGE ======================

// Pairwise merge derived from sparkjsdev/spark's `gsplat.rs::new_merged`:
//   - weights are α · ellipsoid-area (the screen-projected "ink" currency),
//     which preserves color contribution from thin/anisotropic splats that
//     volume-weighted NanoGS would otherwise drown out
//   - merged covariance is the area·α-weighted sum of (δδᵀ + Σ_k), giving
//     the right merged shape via the law of total variance
//   - merged opacity is mass-conserving: α_m = (Σ α_k · area_k) / area_merged,
//     clamped at 1 (the `spark-no-inflate` variant — no scale inflation and
//     no out-of-range α, so the output PLY/SOG stays standard logit-encoded)
//   - no Spark filter floor: we're pair-wise merging, not building an LOD
//     grid, so the inter-mean filter regularization adds blur for no benefit
const momentMatch = (
    i: number, j: number,
    cx: any, cy: any, cz: any,
    cop: any, cs0: any, cs1: any, cs2: any,
    cr0: any, cr1: any, cr2: any, cr3: any,
    out: { mu: Float64Array; sc: Float64Array; q: Float64Array; op: number; sh: Float64Array },
    appData: any[], appColCount: number,
    scratch: MergeScratch
) => {
    const sxi = Math.max(Math.exp(cs0[i] as number), 1e-12);
    const syi = Math.max(Math.exp(cs1[i] as number), 1e-12);
    const szi = Math.max(Math.exp(cs2[i] as number), 1e-12);
    const sxj = Math.max(Math.exp(cs0[j] as number), 1e-12);
    const syj = Math.max(Math.exp(cs1[j] as number), 1e-12);
    const szj = Math.max(Math.exp(cs2[j] as number), 1e-12);

    const alphaI = sigmoid(cop[i] as number);
    const alphaJ = sigmoid(cop[j] as number);

    // Area·α weighting (NanoGS used (2π)^1.5·sx·sy·sz·α; we use ellipsoid area).
    const Ai = ellipsoidArea(sxi, syi, szi);
    const Aj = ellipsoidArea(sxj, syj, szj);
    const wi = alphaI * Ai + 1e-30;
    const wj = alphaJ * Aj + 1e-30;
    const W = wi + wj;
    const pi = wi / W;
    const pj = wj / W;

    // Merged mean (weighted)
    const mxi = cx[i] as number, myi = cy[i] as number, mzi = cz[i] as number;
    const mxj = cx[j] as number, myj = cy[j] as number, mzj = cz[j] as number;
    const mux = pi * mxi + pj * mxj;
    const muy = pi * myi + pj * myj;
    const muz = pi * mzi + pj * mzj;

    // Per-call scratch buffers — owned by `simplifyGaussians` and passed in.
    const SigI = scratch.sigI;
    const SigJ = scratch.sigJ;
    const Ri = scratch.rI;
    const Rj = scratch.rJ;

    let qwi = cr0[i] as number, qxi = cr1[i] as number, qyi = cr2[i] as number, qzi = cr3[i] as number;
    const ni = 1 / Math.max(Math.hypot(qwi, qxi, qyi, qzi), 1e-12);
    qwi *= ni; qxi *= ni; qyi *= ni; qzi *= ni;
    let qwj = cr0[j] as number, qxj = cr1[j] as number, qyj = cr2[j] as number, qzj = cr3[j] as number;
    const nj = 1 / Math.max(Math.hypot(qwj, qxj, qyj, qzj), 1e-12);
    qwj *= nj; qxj *= nj; qyj *= nj; qzj *= nj;

    quatToRotmat(qwi, qxi, qyi, qzi, Ri, 0);
    quatToRotmat(qwj, qxj, qyj, qzj, Rj, 0);
    sigmaFromRotVar(Ri, 0, sxi * sxi, syi * syi, szi * szi, SigI, 0);
    sigmaFromRotVar(Rj, 0, sxj * sxj, syj * syj, szj * szj, SigJ, 0);

    const dix = mxi - mux, diy = myi - muy, diz = mzi - muz;
    const djx = mxj - mux, djy = myj - muy, djz = mzj - muz;

    // Merged covariance (weighted sum of δδᵀ + Σ_k) — scratch.
    const Sig = scratch.sig;
    Sig[0] = pi * (dix * dix + SigI[0]) + pj * (djx * djx + SigJ[0]);
    Sig[1] = pi * (dix * diy + SigI[1]) + pj * (djx * djy + SigJ[1]);
    Sig[2] = pi * (dix * diz + SigI[2]) + pj * (djx * djz + SigJ[2]);
    Sig[3] = Sig[1];
    Sig[4] = pi * (diy * diy + SigI[4]) + pj * (djy * djy + SigJ[4]);
    Sig[5] = pi * (diy * diz + SigI[5]) + pj * (djy * djz + SigJ[5]);
    Sig[6] = Sig[2];
    Sig[7] = Sig[5];
    Sig[8] = pi * (diz * diz + SigI[8]) + pj * (djz * djz + SigJ[8]);
    Sig[0] += EPS_COV;
    Sig[4] += EPS_COV;
    Sig[8] += EPS_COV;

    // Eigendecompose → scales (= √λ) and rotation. `eigA` ends with the
    // eigenvalues on its diagonal (positions 0/4/8); `eigV` holds the
    // eigenvectors as columns.
    const eigA = scratch.eigA;
    const eigV = scratch.eigV;
    eigenSymmetric3x3(Sig, eigA, eigV);
    const vecs = eigV;

    // Sort eigenvalue indices descending. Hand-unrolled to avoid the
    // `[0,1,2].sort(...)` + `.map(...)` allocations.
    const v0 = eigA[0], v1 = eigA[4], v2 = eigA[8];
    let o0: number, o1: number, o2: number;
    if (v0 >= v1) {
        if (v1 >= v2)      {
            o0 = 0; o1 = 1; o2 = 2;
        } else if (v0 >= v2) {
            o0 = 0; o1 = 2; o2 = 1;
        } else               {
            o0 = 2; o1 = 0; o2 = 1;
        }
    } else {
        if (v0 >= v2)      {
            o0 = 1; o1 = 0; o2 = 2;
        } else if (v1 >= v2) {
            o0 = 1; o1 = 2; o2 = 0;
        } else               {
            o0 = 2; o1 = 1; o2 = 0;
        }
    }
    const ev0 = Math.max(eigA[3 * o0 + o0], 1e-18);
    const ev1 = Math.max(eigA[3 * o1 + o1], 1e-18);
    const ev2 = Math.max(eigA[3 * o2 + o2], 1e-18);

    const s0 = Math.sqrt(ev0);
    const s1 = Math.sqrt(ev1);
    const s2 = Math.sqrt(ev2);

    // Mass-conserving opacity, capped at 1 (no scale inflation). No lower
    // clamp here — `logit()` at the storage step already pins p ≥ 1e-7.
    const alphaM = Math.min(1, W / Math.max(ellipsoidArea(s0, s1, s2), 1e-30));

    // Build rotation matrix from sorted eigenvectors (right-handed) — scratch.
    const Rm = scratch.rM;
    Rm[0] = vecs[o0]; Rm[1] = vecs[o1]; Rm[2] = vecs[o2];
    Rm[3] = vecs[3 + o0]; Rm[4] = vecs[3 + o1]; Rm[5] = vecs[3 + o2];
    Rm[6] = vecs[6 + o0]; Rm[7] = vecs[6 + o1]; Rm[8] = vecs[6 + o2];
    if (det3(Rm, 0) < 0) {
        Rm[2] *= -1; Rm[5] *= -1; Rm[8] *= -1;
    }
    rotmatToQuat(Rm, 0, out.q, 0);

    out.mu[0] = mux; out.mu[1] = muy; out.mu[2] = muz;
    out.sc[0] = Math.log(s0);
    out.sc[1] = Math.log(s1);
    out.sc[2] = Math.log(s2);
    out.op = alphaM;

    // Color: weight-normalized (area·α weighted) average
    for (let k = 0; k < appColCount; k++) {
        out.sh[k] = pi * (appData[k][i] as number) + pj * (appData[k][j] as number);
    }
};

// ====================== VISIBILITY PRUNING ======================

const sortByVisibility = (dataTable: DataTable, indices: Uint32Array): void => {
    const opacityCol = dataTable.getColumnByName('opacity');
    const scale0Col = dataTable.getColumnByName('scale_0');
    const scale1Col = dataTable.getColumnByName('scale_1');
    const scale2Col = dataTable.getColumnByName('scale_2');

    if (!opacityCol || !scale0Col || !scale1Col || !scale2Col) {
        logger.debug('missing required columns for visibility sorting (opacity, scale_0, scale_1, scale_2)');
        return;
    }
    if (indices.length === 0) return;

    const opacity = opacityCol.data;
    const scale0 = scale0Col.data;
    const scale1 = scale1Col.data;
    const scale2 = scale2Col.data;

    const scores = new Float32Array(indices.length);
    for (let i = 0; i < indices.length; i++) {
        const ri = indices[i];
        scores[i] = (1 / (1 + Math.exp(-opacity[ri]))) * Math.exp(scale0[ri] + scale1[ri] + scale2[ri]);
    }

    const order = new Uint32Array(indices.length);
    for (let i = 0; i < order.length; i++) order[i] = i;
    order.sort((a, b) => scores[b] - scores[a]);

    const tmp = indices.slice();
    for (let i = 0; i < indices.length; i++) indices[i] = tmp[order[i]];
};

// ====================== MAIN: simplifyGaussians ======================

/**
 * Simplifies a Gaussian splat DataTable to a target number of splats by
 * progressive pair-wise merging of nearest-neighbor gaussians.
 *
 * The merge math is derived from sparkjsdev/spark `gsplat.rs::new_merged`
 * (area·α weighted, mass-conserving opacity, no filter floor, opacity
 * clamped at 1). It preserves color fidelity through multiple iterations
 * far better than the NanoGS Porter-Duff opacity + volume-weighted formula
 * does — color drift / over-saturation is essentially gone even at
 * aggressive reductions.
 *
 * @param dataTable - The input splat DataTable.
 * @param targetCount - The desired number of output splats.
 * @param createDevice - Optional factory yielding a `GraphicsDevice`. When supplied, KNN and edge-cost run on the GPU; otherwise the CPU KD-tree path is used.
 * @returns A new DataTable with approximately `targetCount` splats.
 */
const simplifyGaussians = async (
    dataTable: DataTable,
    targetCount: number,
    createDevice?: DeviceCreator
): Promise<DataTable> => {
    const N = dataTable.numRows;
    if (N <= targetCount || targetCount <= 0) {
        return targetCount <= 0 ? dataTable.clone({ rows: [] }) : dataTable;
    }

    // Mirrors the factory contract used by `filterFloaters` — the caller
    // hands us a `DeviceCreator` so we can create the device lazily (and
    // skip creation entirely on the early-return path above). We don't
    // destroy the device; per the `DeviceCreator` contract in types.ts,
    // the caller owns its lifecycle and is responsible for caching/reuse.
    const device = createDevice ? await createDevice() : undefined;

    const requiredCols = ['x', 'y', 'z', 'opacity', 'scale_0', 'scale_1', 'scale_2',
        'rot_0', 'rot_1', 'rot_2', 'rot_3'];
    for (const name of requiredCols) {
        if (!dataTable.hasColumn(name)) {
            logger.debug(`missing required column '${name}', falling back to visibility pruning`);
            const indices = new Uint32Array(N);
            for (let i = 0; i < N; i++) indices[i] = i;
            sortByVisibility(dataTable, indices);
            return dataTable.clone({ rows: indices.subarray(0, targetCount) });
        }
    }

    // Identify appearance columns
    const allAppearanceCols: string[] = [];
    for (const name of ['f_dc_0', 'f_dc_1', 'f_dc_2']) {
        if (dataTable.hasColumn(name)) allAppearanceCols.push(name);
    }
    for (let i = 0; i < 45; i++) {
        const name = `f_rest_${i}`;
        if (dataTable.hasColumn(name)) allAppearanceCols.push(name);
    }

    let current: DataTable = dataTable;

    // Pre-generate MC samples
    const Z = makeGaussianSamples(MC_SAMPLES, 0);

    // Reusable scratch buffers for the per-iteration edge-cost sort.
    const sortScratch = new RadixSortScratch();

    // Per-call merge scratch — see `MergeScratch` above. Lives until the end
    // of this `simplifyGaussians` invocation; not shared with concurrent calls.
    const mergeScratch = createMergeScratch();

    // Hoist GPU resources above the merging loop. Empirically this lowers peak
    // RSS by ~1.4 GB on a 17.9M-splat run vs. per-iteration construction —
    // the per-iteration version has a transient destroy/create overlap where
    // GpuKnn buffers haven't been reclaimed by the WebGPU backend before
    // GpuEdgeCost allocates fresh ones. Hoisting reuses the same backing
    // storage across iterations. `initialMaxE` is the true upper bound on edge
    // count, n·k (not n·k/2 — that's only the expected count; per-iteration
    // variance lets the actual edgeCount exceed n·k/2 by a few percent). It's a
    // validation bound, not a buffer size: GpuEdgeCost uploads edges per
    // dispatch batch, so its edge buffers are batch-sized regardless.
    const initialN = current.numRows;
    const kEffMax = Math.min(Math.max(1, KNN_K), Math.max(1, initialN - 1));
    const initialMaxE = initialN * kEffMax;
    const numAppColsMax = allAppearanceCols.length;
    const z = new Float32Array(3);
    z[0] = Z[0][0]; z[1] = Z[0][1]; z[2] = Z[0][2];

    // Iterative merging.
    // Each iteration roughly halves the row count (greedy disjoint pair
    // selection picks up to n/2 merges), so log2(n / target) is a good
    // upper bound for the [N/T] series numbering. Iterations beyond the
    // estimate render unnumbered, which is fine.
    const estIterations = Math.max(1, Math.ceil(Math.log2(current.numRows / targetCount)));
    let iterationIndex = 0;

    // GPU resources live inside the try block so that if the second
    // constructor throws (e.g. the appearance-buffer preflight in
    // GpuEdgeCost), the partially-built first constructor's resources get
    // cleaned up by the finally.
    let gpuKnn: GpuKnn | undefined;
    let gpuCost: GpuEdgeCost | undefined;

    try {
        if (device) {
            gpuKnn = new GpuKnn(device, initialN, kEffMax);
            gpuCost = new GpuEdgeCost(device, initialN, initialMaxE, numAppColsMax);
        }

        while (current.numRows > targetCount) {
            const n = current.numRows;
            // kEff fixed at kEffMax so the hoisted GPU resources (which bake k
            // into the shader and into buffer sizing) work across every iteration.
            // When n - 1 < kEffMax, both KNN paths fill the surplus slots with the
            // sentinel 0xFFFFFFFF and the edge-extraction loop ignores them.
            const kEff = kEffMax;

            iterationIndex++;
            const g = iterationIndex <= estIterations ?
                logger.group('Decimate iteration', {
                    index: iterationIndex,
                    total: estIterations
                }) :
                logger.group('Decimate iteration');

            const cx = current.getColumnByName('x')!.data;
            const cy = current.getColumnByName('y')!.data;
            const cz = current.getColumnByName('z')!.data;
            const cop = current.getColumnByName('opacity')!.data;
            const cs0 = current.getColumnByName('scale_0')!.data;
            const cs1 = current.getColumnByName('scale_1')!.data;
            const cs2 = current.getColumnByName('scale_2')!.data;
            const cr0 = current.getColumnByName('rot_0')!.data;
            const cr1 = current.getColumnByName('rot_1')!.data;
            const cr2 = current.getColumnByName('rot_2')!.data;
            const cr3 = current.getColumnByName('rot_3')!.data;

            const cacheSub = logger.group('Building per-splat cache');
            const cache = buildPerSplatCache(n, cop, cs0, cs1, cs2, cr0, cr1, cr2, cr3, !!device);
            cacheSub.end();

            // Flat per-query KNN output. allKnn[i*kEff + j] is one of the kEff
            // nearest neighbours of i, excluding i itself. CPU path emits sorted
            // ascending by distance; GPU path emits an unsorted top-K. Neither
            // ordering matters downstream — the edge-extraction loop only filters
            // j > i. Sentinel 0xFFFFFFFF marks unfilled slots when fewer than kEff
            // non-self neighbours exist. `let` so we can drop the reference after
            // edge extraction — letting it go out of scope at end of iteration
            // doesn't help because V8 grows the heap before collecting.
            let allKnn = new Uint32Array(n * kEff);

            const pxArr = cx instanceof Float32Array ? cx : new Float32Array(cx as any);
            const pyArr = cy instanceof Float32Array ? cy : new Float32Array(cy as any);
            const pzArr = cz instanceof Float32Array ? cz : new Float32Array(cz as any);

            if (device) {
                const knnSub = logger.group('Finding nearest neighbors (GPU)');
                await gpuKnn!.execute(pxArr, pyArr, pzArr, allKnn);
                knnSub.end();
            } else {
                const kdSub = logger.group('Building KD-tree');
                const posTable = new DataTable([
                    new Column('x', pxArr),
                    new Column('y', pyArr),
                    new Column('z', pzArr)
                ]);
                const kdTree = new KdTree(posTable);
                kdSub.end();

                const queryPoint = new Float32Array(3);
                const knnInterval = Math.max(1, Math.ceil(n / PROGRESS_TICKS));
                const knnTicks = Math.ceil(n / knnInterval);
                const knnBar = logger.bar('Finding nearest neighbors', knnTicks);
                for (let i = 0; i < n; i++) {
                    queryPoint[0] = cx[i] as number;
                    queryPoint[1] = cy[i] as number;
                    queryPoint[2] = cz[i] as number;
                    // Request kEff+1 because the KD-tree returns the query itself
                    // (distance 0) as the first match.
                    const knn = kdTree.findKNearest(queryPoint, kEff + 1);
                    let outPos = 0;
                    for (let ki = 0; ki < knn.indices.length && outPos < kEff; ki++) {
                        const j = knn.indices[ki];
                        if (j === i) continue;
                        allKnn[i * kEff + outPos] = j;
                        outPos++;
                    }
                    while (outPos < kEff) {
                        allKnn[i * kEff + outPos] = 0xFFFFFFFF;
                        outPos++;
                    }
                    if ((i + 1) % knnInterval === 0) knnBar.tick();
                }
                if (n % knnInterval !== 0) knnBar.tick();
                knnBar.end();
            }

            // Extract directed edges (i < j) from the K-NN graph.
            let edgeCapacity = Math.ceil(n * kEff / 2);
            let edgeU = new Uint32Array(edgeCapacity);
            let edgeV = new Uint32Array(edgeCapacity);
            let edgeCount = 0;
            for (let i = 0; i < n; i++) {
                const base = i * kEff;
                for (let ki = 0; ki < kEff; ki++) {
                    const j = allKnn[base + ki];
                    if (j === 0xFFFFFFFF || j <= i) continue;
                    if (edgeCount === edgeCapacity) {
                        edgeCapacity *= 2;
                        const newU = new Uint32Array(edgeCapacity);
                        const newV = new Uint32Array(edgeCapacity);
                        newU.set(edgeU);
                        newV.set(edgeV);
                        edgeU = newU;
                        edgeV = newV;
                    }
                    edgeU[edgeCount] = i;
                    edgeV[edgeCount] = j;
                    edgeCount++;
                }
            }

            // allKnn was only read by the edge-extraction loop above. Drop the
            // reference so V8 can reclaim the 1.14 GB (at N=17.9M, k=16) — see
            // the comment at the declaration.
            allKnn = new Uint32Array(0);

            // The loop only runs while n > targetCount, and a non-degenerate
            // k-NN graph over n >= 2 splats always yields edges (splat 0's
            // neighbours have higher indices). Zero edges here therefore means
            // the neighbour step produced nothing usable — on the GPU path a
            // swallowed out-of-memory leaving the k-NN buffer zeroed, on the CPU
            // path a degenerate input — never a legitimate "can't decimate
            // further". Fail rather than return an incompletely-decimated scene,
            // regardless of how many earlier iterations succeeded.
            if (edgeCount === 0) {
                g.end();
                const cause = device ?
                    'the GPU step likely failed (e.g. out-of-memory)' :
                    'the input is likely degenerate (e.g. non-finite positions)';
                throw new Error(
                    `decimation produced no edges from the k-NN graph at ${n} splats (target ${targetCount}) — ` +
                    `${cause}. Refusing to return an incompletely-decimated scene.`
                );
            }

            const appData: any[] = [];
            for (let ai = 0; ai < allAppearanceCols.length; ai++) {
                const col = current.getColumnByName(allAppearanceCols[ai]);
                if (col) appData.push(col.data);
            }

            const mergesNeeded = n - targetCount;
            let costs = new Float32Array(edgeCount);

            if (device) {
                // GPU path: pack the per-splat cache into the layout the kernel
                // expects — position + scalars interleaved 8-wide in one buffer,
                // appearance split into 16-column chunks (each chunk stays under
                // the ~2 GB per-binding limit). `R` is already Float32, so the
                // rotation passes through directly.
                const costSub = logger.group('Computing edge costs (GPU)');

                const C = appData.length;
                const numChunks = Math.ceil(C / APP_CHUNK);

                const posScalars = new Float32Array(n * 8);
                for (let s = 0; s < n; s++) {
                    const o = s * 8;
                    posScalars[o + 0] = cx[s] as number;
                    posScalars[o + 1] = cy[s] as number;
                    posScalars[o + 2] = cz[s] as number;
                    posScalars[o + 3] = cache.mass[s];
                    posScalars[o + 4] = cache.logdet[s];
                    posScalars[o + 5] = cache.v[s * 3 + 0];
                    posScalars[o + 6] = cache.v[s * 3 + 1];
                    posScalars[o + 7] = cache.v[s * 3 + 2];
                }

                // Pack appearance into chunks of ≤APP_CHUNK columns. Each
                // chunk's stride is its live width (only the final chunk may be
                // partial), so no padding is stored or uploaded. APP_CHUNK is
                // the same constant the kernel bakes its strides from, so the
                // host packing and the kernel layout stay in lockstep.
                const appChunks: Float32Array[] = [];
                for (let ch = 0; ch < numChunks; ch++) {
                    const kStart = ch * APP_CHUNK;
                    const width = Math.min(APP_CHUNK, C - kStart);
                    const chunk = new Float32Array(n * width);
                    for (let s = 0; s < n; s++) {
                        const dst = s * width;
                        for (let kk = 0; kk < width; kk++) {
                            chunk[dst + kk] = appData[kStart + kk][s] as number;
                        }
                    }
                    appChunks.push(chunk);
                }

                const cacheGpu: EdgeCostCache = {
                    posScalars,
                    rotR: cache.R,
                    appChunks,
                    numAppCols: C,
                    numSplats: n
                };

                // Trim edge buffers to the valid prefix (edgeU/edgeV were grown).
                const edgeITrim = edgeU.subarray(0, edgeCount);
                const edgeJTrim = edgeV.subarray(0, edgeCount);

                await gpuCost!.execute(cacheGpu, edgeITrim, edgeJTrim, z, costs);
                costSub.end();
            } else {
                const costInterval = Math.max(1, Math.ceil(edgeCount / PROGRESS_TICKS));
                const costTicks = Math.ceil(edgeCount / costInterval);
                const costBar = logger.bar('Computing edge costs', costTicks);
                for (let e = 0; e < edgeCount; e++) {
                    costs[e] = computeEdgeCost(edgeU[e], edgeV[e], cx, cy, cz,
                        cache, Z, appData, appData.length, mergeScratch);
                    if ((e + 1) % costInterval === 0) costBar.tick();
                }
                if (edgeCount % costInterval !== 0) costBar.tick();
                costBar.end();
            }

            // Release the per-splat cache fields that are no longer needed —
            // the merge loop only reads `cache.mass` for dominant-side
            // selection. At N=17.9M that's ~1.7 GB of buffers (R, v, invdiag,
            // logdet, sigma) we can drop before the sort phase pushes peak.
            cache.R = new Float32Array(0);
            cache.v = new Float32Array(0);
            cache.invdiag = new Float32Array(0);
            cache.logdet = new Float32Array(0);
            cache.sigma = new Float32Array(0);

            // Sort and greedy disjoint pair selection. We pass `costs` directly as
            // the parallel-key buffer (an edgeCount-sized Float32 scratch would
            // otherwise add ~580 MB to peak at 17.9M splats). Non-finite costs are
            // replaced by +Inf in place so they sort to the end; the greedy loop
            // caps at `validCount` to ignore them. `costs` is iteration-local and
            // not read again after this point, so mutating it is safe.
            const pairSelectSub = logger.group('Selecting pairs');
            let sorted = new Uint32Array(edgeCount);
            const costBits = new Uint32Array(costs.buffer, costs.byteOffset, costs.length);
            let validCount = edgeCount;
            for (let i = 0; i < edgeCount; i++) {
                sorted[i] = i;
                if (isFloatBitsNonFinite(costBits[i])) {
                    costs[i] = Infinity;
                    validCount--;
                }
            }
            radixSortIndicesByFloat(sorted, costs, edgeCount, sortScratch);

            const used = new Uint8Array(n);
            const pairs: [number, number][] = [];

            for (let t = 0; t < validCount; t++) {
                const e = sorted[t];
                const u = edgeU[e], v = edgeV[e];
                if (used[u] || used[v]) continue;
                used[u] = 1; used[v] = 1;
                pairs.push([u, v]);
                if (pairs.length >= mergesNeeded) break;
            }
            pairSelectSub.end();

            // Release the sort/edge scratch — none of it is read during the
            // merge below. At N=17.9M this is ~2.85 GB (edgeU/V 1.14 GB,
            // sorted 0.57 GB, costs 0.57 GB) that V8 would otherwise keep
            // pinned through the merge.
            edgeU = new Uint32Array(0);
            edgeV = new Uint32Array(0);
            costs = new Float32Array(0);
            sorted = new Uint32Array(0);

            // pairs.length === 0 only happens when every edge cost is non-finite
            // (any finite cost yields at least one pair), i.e. a total
            // cost-computation failure — GPU corruption on the GPU path, or
            // non-finite inputs on either path. As above, the loop guarantees
            // n > targetCount, so this is always a failure to reach the target,
            // not a legitimate stop. Fatal regardless of prior progress.
            if (pairs.length === 0) {
                g.end();
                const cause = device ?
                    'the GPU step likely failed (e.g. out-of-memory) or produced non-finite costs' :
                    'cost computation produced only non-finite values (e.g. non-finite inputs)';
                throw new Error(
                    `decimation found no valid merge pairs among ${edgeCount} edges at ${n} splats (target ${targetCount}) — ` +
                    `${cause}. Refusing to return an incompletely-decimated scene.`
                );
            }

            // Allocate output table
            const allocSub = logger.group('Allocating output table');
            const usedSet = new Uint8Array(n);
            for (let p = 0; p < pairs.length; p++) {
                usedSet[pairs[p][0]] = 1;
                usedSet[pairs[p][1]] = 1;
            }

            const keepIndices: number[] = [];
            for (let i = 0; i < n; i++) {
                if (!usedSet[i]) keepIndices.push(i);
            }

            const outCount = keepIndices.length + pairs.length;
            const cols = current.columns;
            const newColumns: Column[] = [];
            for (let ci = 0; ci < cols.length; ci++) {
                const c = cols[ci];
                newColumns.push(new Column(c.name, new (c.data.constructor as any)(outCount)));
            }
            const newTable = new DataTable(newColumns, dataTable.transform);
            allocSub.end();

            // Copy unmerged splats
            const copySub = logger.group('Copying kept splats');
            let dst = 0;
            for (let t = 0; t < keepIndices.length; t++, dst++) {
                const src = keepIndices[t];
                for (let c = 0; c < cols.length; c++) {
                    newTable.columns[c].data[dst] = cols[c].data[src] as number;
                }
            }
            copySub.end();

            // Merge pairs
            const mergeOut = {
                mu: new Float64Array(3),
                sc: new Float64Array(3),
                q: new Float64Array(4),
                op: 0,
                sh: new Float64Array(allAppearanceCols.length)
            };

            const dstXCol = newTable.getColumnByName('x')!;
            const dstYCol = newTable.getColumnByName('y')!;
            const dstZCol = newTable.getColumnByName('z')!;
            const dstS0Col = newTable.getColumnByName('scale_0')!;
            const dstS1Col = newTable.getColumnByName('scale_1')!;
            const dstS2Col = newTable.getColumnByName('scale_2')!;
            const dstR0Col = newTable.getColumnByName('rot_0')!;
            const dstR1Col = newTable.getColumnByName('rot_1')!;
            const dstR2Col = newTable.getColumnByName('rot_2')!;
            const dstR3Col = newTable.getColumnByName('rot_3')!;
            const dstOpCol = newTable.getColumnByName('opacity')!;
            const dstAppCols = allAppearanceCols.map(name => newTable.getColumnByName(name));

            const handledCols = new Set([
                'x', 'y', 'z', 'opacity', 'scale_0', 'scale_1', 'scale_2',
                'rot_0', 'rot_1', 'rot_2', 'rot_3', ...allAppearanceCols
            ]);
            const unhandledColPairs = cols
            .filter(col => !handledCols.has(col.name))
            .map(col => ({ src: col, dst: newTable.getColumnByName(col.name)! }))
            .filter(pair => pair.dst);

            const mergeInterval = Math.max(1, Math.ceil(pairs.length / PROGRESS_TICKS));
            const mergeTicks = Math.ceil(pairs.length / mergeInterval);
            const mergeBar = logger.bar('Merging splats', mergeTicks);
            for (let p = 0; p < pairs.length; p++, dst++) {
                const pi = pairs[p][0], pj = pairs[p][1];

                momentMatch(pi, pj, cx, cy, cz, cop, cs0, cs1, cs2, cr0, cr1, cr2, cr3,
                    mergeOut, appData, appData.length, mergeScratch);

                dstXCol.data[dst] = mergeOut.mu[0];
                dstYCol.data[dst] = mergeOut.mu[1];
                dstZCol.data[dst] = mergeOut.mu[2];
                dstS0Col.data[dst] = mergeOut.sc[0];
                dstS1Col.data[dst] = mergeOut.sc[1];
                dstS2Col.data[dst] = mergeOut.sc[2];
                dstR0Col.data[dst] = mergeOut.q[0];
                dstR1Col.data[dst] = mergeOut.q[1];
                dstR2Col.data[dst] = mergeOut.q[2];
                dstR3Col.data[dst] = mergeOut.q[3];
                dstOpCol.data[dst] = logit(Math.max(0, Math.min(1, mergeOut.op)));

                for (let k = 0; k < dstAppCols.length; k++) {
                    if (dstAppCols[k]) dstAppCols[k]!.data[dst] = mergeOut.sh[k];
                }

                const dominant = cache.mass[pi] >= cache.mass[pj] ? pi : pj;
                for (let u = 0; u < unhandledColPairs.length; u++) {
                    unhandledColPairs[u].dst.data[dst] = unhandledColPairs[u].src.data[dominant] as number;
                }

                if ((p + 1) % mergeInterval === 0) mergeBar.tick();
            }
            if (pairs.length % mergeInterval !== 0) mergeBar.tick();
            mergeBar.end();

            current = newTable;
            g.end();
        }

        return current;
    } finally {
        gpuKnn?.destroy();
        gpuCost?.destroy();
    }
};

export { sortByVisibility, simplifyGaussians };
