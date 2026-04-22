import { Column, DataTable } from './data-table';
import { KdTree } from '../spatial/kd-tree';
import { logger } from '../utils';

const TWO_PI_POW_1_5 = Math.pow(2 * Math.PI, 1.5);
const LOG2PI = Math.log(2 * Math.PI);
const OPACITY_PRUNE_THRESHOLD = 0.1;
const KNN_K = 16;
const MC_SAMPLES = 1;
const EPS_COV = 1e-8;
const PROGRESS_TICKS = 100;

// Radix sort edge indices by their Float32 costs.
// Converts floats to sortable uint32 keys (preserving order), then does
// 4-pass LSD radix sort with 8-bit radix. Returns the number of valid
// (finite-cost) edges written to `out`.
const radixSortIndicesByFloat = (out: Uint32Array, count: number, keys: Float32Array): number => {
    const keyBits = new Uint32Array(keys.buffer, keys.byteOffset, keys.length);

    const sortKeys = new Uint32Array(count);
    let validCount = 0;
    for (let i = 0; i < count; i++) {
        const bits = keyBits[i];
        if ((bits & 0x7F800000) === 0x7F800000) continue;
        sortKeys[validCount] = (bits & 0x80000000) ? ~bits >>> 0 : (bits | 0x80000000) >>> 0;
        out[validCount] = i;
        validCount++;
    }

    if (validCount <= 1) return validCount;

    const n = validCount;
    const temp = new Uint32Array(n);
    const tempKeys = new Uint32Array(n);
    const counts = new Uint32Array(256);

    for (let pass = 0; pass < 4; pass++) {
        const shift = pass << 3;
        const srcIdx = (pass & 1) ? temp : out;
        const dstIdx = (pass & 1) ? out : temp;
        const srcK = (pass & 1) ? tempKeys : sortKeys;
        const dstK = (pass & 1) ? sortKeys : tempKeys;

        counts.fill(0);
        for (let i = 0; i < n; i++) {
            counts[(srcK[i] >>> shift) & 0xFF]++;
        }

        let sum = 0;
        for (let b = 0; b < 256; b++) {
            const c = counts[b];
            counts[b] = sum;
            sum += c;
        }

        for (let i = 0; i < n; i++) {
            const bucket = (srcK[i] >>> shift) & 0xFF;
            const pos = counts[bucket]++;
            dstIdx[pos] = srcIdx[i];
            dstK[pos] = srcK[i];
        }
    }

    return validCount;
};

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

const quatToRotmat = (qw: number, qx: number, qy: number, qz: number, out: Float64Array, o: number) => {
    const ww = qw * qw, xx = qx * qx, yy = qy * qy, zz = qz * qz;
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

const transpose3 = (src: Float64Array, so: number, dst: Float64Array, doff: number) => {
    dst[doff] = src[so]; dst[doff + 1] = src[so + 3]; dst[doff + 2] = src[so + 6];
    dst[doff + 3] = src[so + 1]; dst[doff + 4] = src[so + 4]; dst[doff + 5] = src[so + 7];
    dst[doff + 6] = src[so + 2]; dst[doff + 7] = src[so + 5]; dst[doff + 8] = src[so + 8];
};

const sigmaFromRotVar = (R: Float64Array, r: number, vx: number, vy: number, vz: number, out: Float64Array, o: number) => {
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
    R: Float64Array, ro: number,
    invx: number, invy: number, invz: number, logdet: number
) => {
    const dx = x - mx, dy = y - my, dz = z - mz;
    const y0 = dx * R[ro] + dy * R[ro + 3] + dz * R[ro + 6];
    const y1 = dx * R[ro + 1] + dy * R[ro + 4] + dz * R[ro + 7];
    const y2 = dx * R[ro + 2] + dy * R[ro + 5] + dz * R[ro + 8];
    const quad = y0 * y0 * invx + y1 * y1 * invy + y2 * y2 * invz;
    return -0.5 * (3 * LOG2PI + logdet + quad);
};

// Jacobi eigendecomposition for 3x3 symmetric matrix (full 9-element row-major)
const eigenSymmetric3x3 = (Ain: Float64Array) => {
    const A = new Float64Array(Ain);
    const V = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);

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

    return { values: [A[0], A[4], A[8]], vectors: V };
};

const rotmatToQuat = (R: Float64Array, o: number): Float64Array => {
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
    return new Float64Array([qw * inv, qx * inv, qy * inv, qz * inv]);
};

// ====================== PER-SPLAT CACHE ======================

interface SplatCache {
    R: Float64Array;
    Rt: Float64Array;
    v: Float64Array;       // variances (scale^2 + eps) per axis
    invdiag: Float64Array;
    logdet: Float64Array;
    sigma: Float64Array;   // full 9-element covariance
    mass: Float64Array;
}

const buildPerSplatCache = (
    n: number,
    cx: any, cy: any, cz: any,
    cop: any, cs0: any, cs1: any, cs2: any,
    cr0: any, cr1: any, cr2: any, cr3: any
): SplatCache => {
    const R = new Float64Array(n * 9);
    const Rt = new Float64Array(n * 9);
    const v = new Float64Array(n * 3);
    const invdiag = new Float64Array(n * 3);
    const logdet = new Float64Array(n);
    const sigma = new Float64Array(n * 9);
    const mass = new Float64Array(n);

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
        invdiag[i3] = 1 / Math.max(vx, 1e-30);
        invdiag[i3 + 1] = 1 / Math.max(vy, 1e-30);
        invdiag[i3 + 2] = 1 / Math.max(vz, 1e-30);
        logdet[i] = Math.log(Math.max(vx, 1e-30)) + Math.log(Math.max(vy, 1e-30)) + Math.log(Math.max(vz, 1e-30));

        // Normalize quaternion before building rotation
        let qw = cr0[i] as number, qx = cr1[i] as number, qy = cr2[i] as number, qz = cr3[i] as number;
        const qn = Math.hypot(qw, qx, qy, qz);
        const invq = 1 / Math.max(qn, 1e-12);
        qw *= invq; qx *= invq; qy *= invq; qz *= invq;

        quatToRotmat(qw, qx, qy, qz, R, i9);
        transpose3(R, i9, Rt, i9);
        sigmaFromRotVar(R, i9, vx, vy, vz, sigma, i9);

        mass[i] = TWO_PI_POW_1_5 * linAlpha * sx * sy * sz + 1e-12;
    }

    return { R, Rt, v, invdiag, logdet, sigma, mass };
};

// ====================== COST FUNCTION ======================

const _Sigm = new Float64Array(9);

const computeEdgeCost = (
    i: number, j: number,
    cx: any, cy: any, cz: any,
    cache: SplatCache,
    Z: Float64Array[],
    appData: any[], appColCount: number
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
    for (let a = 0; a < 9; a++) {
        _Sigm[a] = pi * cache.sigma[i9 + a] + pj * cache.sigma[j9 + a];
    }
    _Sigm[0] += pi * dix * dix + pj * djx * djx;
    _Sigm[1] += pi * dix * diy + pj * djx * djy;
    _Sigm[2] += pi * dix * diz + pj * djx * djz;
    _Sigm[3] += pi * diy * dix + pj * djy * djx;
    _Sigm[4] += pi * diy * diy + pj * djy * djy;
    _Sigm[5] += pi * diy * diz + pj * djy * djz;
    _Sigm[6] += pi * diz * dix + pj * djz * djx;
    _Sigm[7] += pi * diz * diy + pj * djz * djy;
    _Sigm[8] += pi * diz * diz + pj * djz * djz;

    // Force symmetry + regularize
    _Sigm[1] = _Sigm[3] = 0.5 * (_Sigm[1] + _Sigm[3]);
    _Sigm[2] = _Sigm[6] = 0.5 * (_Sigm[2] + _Sigm[6]);
    _Sigm[5] = _Sigm[7] = 0.5 * (_Sigm[5] + _Sigm[7]);
    _Sigm[0] += EPS_COV;
    _Sigm[4] += EPS_COV;
    _Sigm[8] += EPS_COV;

    const detm = Math.max(det3(_Sigm, 0), 1e-30);
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

        // x_i = mu_i + R_i^T * diag(std_i) * z
        const xix = mux + z0 * stdix * cache.Rt[i9] + z1 * stdiy * cache.Rt[i9 + 3] + z2 * stdiz * cache.Rt[i9 + 6];
        const xiy = muy + z0 * stdix * cache.Rt[i9 + 1] + z1 * stdiy * cache.Rt[i9 + 4] + z2 * stdiz * cache.Rt[i9 + 7];
        const xiz = muz + z0 * stdix * cache.Rt[i9 + 2] + z1 * stdiy * cache.Rt[i9 + 5] + z2 * stdiz * cache.Rt[i9 + 8];

        // x_j = mu_j + R_j^T * diag(std_j) * z
        const xjx = mvx + z0 * stdjx * cache.Rt[j9] + z1 * stdjy * cache.Rt[j9 + 3] + z2 * stdjz * cache.Rt[j9 + 6];
        const xjy = mvy + z0 * stdjx * cache.Rt[j9 + 1] + z1 * stdjy * cache.Rt[j9 + 4] + z2 * stdjz * cache.Rt[j9 + 7];
        const xjz = mvz + z0 * stdjx * cache.Rt[j9 + 2] + z1 * stdjy * cache.Rt[j9 + 5] + z2 * stdjz * cache.Rt[j9 + 8];

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

// ====================== MERGE (MPMM) ======================

const momentMatch = (
    i: number, j: number,
    cx: any, cy: any, cz: any,
    cop: any, cs0: any, cs1: any, cs2: any,
    cr0: any, cr1: any, cr2: any, cr3: any,
    out: { mu: Float64Array; sc: Float64Array; q: Float64Array; op: number; sh: Float64Array },
    appData: any[], appColCount: number
) => {
    const sxi = Math.max(Math.exp(cs0[i] as number), 1e-12);
    const syi = Math.max(Math.exp(cs1[i] as number), 1e-12);
    const szi = Math.max(Math.exp(cs2[i] as number), 1e-12);
    const sxj = Math.max(Math.exp(cs0[j] as number), 1e-12);
    const syj = Math.max(Math.exp(cs1[j] as number), 1e-12);
    const szj = Math.max(Math.exp(cs2[j] as number), 1e-12);

    const alphaI = sigmoid(cop[i] as number);
    const alphaJ = sigmoid(cop[j] as number);

    const wi = TWO_PI_POW_1_5 * alphaI * sxi * syi * szi + 1e-12;
    const wj = TWO_PI_POW_1_5 * alphaJ * sxj * syj * szj + 1e-12;
    const W = Math.max(wi + wj, 1e-12);

    // Merged mean
    const mux = (wi * (cx[i] as number) + wj * (cx[j] as number)) / W;
    const muy = (wi * (cy[i] as number) + wj * (cy[j] as number)) / W;
    const muz = (wi * (cz[i] as number) + wj * (cz[j] as number)) / W;

    // Build per-splat covariance matrices
    const SigI = new Float64Array(9);
    const SigJ = new Float64Array(9);
    const Ri = new Float64Array(9);
    const Rj = new Float64Array(9);

    let qwi = cr0[i] as number, qxi = cr1[i] as number, qyi = cr2[i] as number, qzi = cr3[i] as number;
    let ni = Math.hypot(qwi, qxi, qyi, qzi);
    ni = 1 / Math.max(ni, 1e-12);
    qwi *= ni; qxi *= ni; qyi *= ni; qzi *= ni;

    let qwj = cr0[j] as number, qxj = cr1[j] as number, qyj = cr2[j] as number, qzj = cr3[j] as number;
    let nj = Math.hypot(qwj, qxj, qyj, qzj);
    nj = 1 / Math.max(nj, 1e-12);
    qwj *= nj; qxj *= nj; qyj *= nj; qzj *= nj;

    quatToRotmat(qwi, qxi, qyi, qzi, Ri, 0);
    quatToRotmat(qwj, qxj, qyj, qzj, Rj, 0);
    sigmaFromRotVar(Ri, 0, sxi * sxi, syi * syi, szi * szi, SigI, 0);
    sigmaFromRotVar(Rj, 0, sxj * sxj, syj * syj, szj * szj, SigJ, 0);

    const dix = (cx[i] as number) - mux, diy = (cy[i] as number) - muy, diz = (cz[i] as number) - muz;
    const djx = (cx[j] as number) - mux, djy = (cy[j] as number) - muy, djz = (cz[j] as number) - muz;

    // Merged covariance
    const Sig = new Float64Array(9);
    for (let a = 0; a < 9; a++) {
        Sig[a] = (wi * SigI[a] + wj * SigJ[a]) / W;
    }
    Sig[0] += (wi * dix * dix + wj * djx * djx) / W;
    Sig[1] += (wi * dix * diy + wj * djx * djy) / W;
    Sig[2] += (wi * dix * diz + wj * djx * djz) / W;
    Sig[3] += (wi * diy * dix + wj * djy * djx) / W;
    Sig[4] += (wi * diy * diy + wj * djy * djy) / W;
    Sig[5] += (wi * diy * diz + wj * djy * djz) / W;
    Sig[6] += (wi * diz * dix + wj * djz * djx) / W;
    Sig[7] += (wi * diz * diy + wj * djz * djy) / W;
    Sig[8] += (wi * diz * diz + wj * djz * djz) / W;

    // Force symmetry + regularize
    Sig[1] = Sig[3] = 0.5 * (Sig[1] + Sig[3]);
    Sig[2] = Sig[6] = 0.5 * (Sig[2] + Sig[6]);
    Sig[5] = Sig[7] = 0.5 * (Sig[5] + Sig[7]);
    Sig[0] += EPS_COV;
    Sig[4] += EPS_COV;
    Sig[8] += EPS_COV;

    // Eigendecomposition
    const ev = eigenSymmetric3x3(Sig);
    let vals = ev.values;
    const vecs = ev.vectors;

    // Sort eigenvalues descending
    const order = [0, 1, 2].sort((a, b) => vals[b] - vals[a]);
    vals = order.map(k => Math.max(vals[k], 1e-18));

    // Build rotation matrix with sorted eigenvectors as columns
    const Rm = new Float64Array(9);
    for (let c = 0; c < 3; c++) {
        const src = order[c];
        Rm[0 + c] = vecs[0 + src];
        Rm[3 + c] = vecs[3 + src];
        Rm[6 + c] = vecs[6 + src];
    }

    if (det3(Rm, 0) < 0) {
        Rm[2] *= -1; Rm[5] *= -1; Rm[8] *= -1;
    }

    const q = rotmatToQuat(Rm, 0);

    out.mu[0] = mux; out.mu[1] = muy; out.mu[2] = muz;
    out.sc[0] = Math.log(Math.sqrt(vals[0]));
    out.sc[1] = Math.log(Math.sqrt(vals[1]));
    out.sc[2] = Math.log(Math.sqrt(vals[2]));
    out.q[0] = q[0]; out.q[1] = q[1]; out.q[2] = q[2]; out.q[3] = q[3];

    // Porter-Duff over opacity
    out.op = alphaI + alphaJ - alphaI * alphaJ;

    // Mass-weighted appearance
    for (let k = 0; k < appColCount; k++) {
        out.sh[k] = (wi * (appData[k][i] as number) + wj * (appData[k][j] as number)) / W;
    }
};

// ====================== SORT BY VISIBILITY (legacy) ======================

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
 * Simplifies a Gaussian splat DataTable to a target number of splats using the
 * NanoGS progressive pairwise merging algorithm.
 *
 * Reference: "NanoGS: Training-Free Gaussian Splat Simplification" (Xiong et al.)
 *
 * @param dataTable - The input splat DataTable.
 * @param targetCount - The desired number of output splats.
 * @returns A new DataTable with approximately `targetCount` splats.
 */
const simplifyGaussians = (dataTable: DataTable, targetCount: number): DataTable => {
    const N = dataTable.numRows;
    if (N <= targetCount || targetCount <= 0) {
        return targetCount <= 0 ? dataTable.clone({ rows: [] }) : dataTable;
    }

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

    // Step 1: Opacity pruning
    const opacityData = dataTable.getColumnByName('opacity')!.data;
    const opsSorted = new Array(N);
    for (let i = 0; i < N; i++) opsSorted[i] = sigmoid(opacityData[i]);
    opsSorted.sort((a: number, b: number) => a - b);
    const median = opsSorted[N >> 1];
    const pruneThreshold = Math.min(OPACITY_PRUNE_THRESHOLD, median);

    const keptIndices: number[] = [];
    for (let i = 0; i < N; i++) {
        if (sigmoid(opacityData[i]) >= pruneThreshold) keptIndices.push(i);
    }

    let current: DataTable;
    if (keptIndices.length < N && keptIndices.length > targetCount) {
        current = dataTable.clone({ rows: keptIndices });
    } else {
        current = dataTable;
    }

    // Pre-generate MC samples
    const Z = makeGaussianSamples(MC_SAMPLES, 0);

    // Step 2: Iterative merging.
    // Each iteration roughly halves the row count (greedy disjoint pair
    // selection picks up to n/2 merges), so log2(n / target) is a good
    // upper bound for the [N/T] series numbering. Iterations beyond the
    // estimate render unnumbered, which is fine.
    const estIterations = Math.max(1, Math.ceil(Math.log2(current.numRows / targetCount)));
    let iterationIndex = 0;

    while (current.numRows > targetCount) {
        const n = current.numRows;
        const kEff = Math.min(Math.max(1, KNN_K), Math.max(1, n - 1));

        iterationIndex++;
        const g = iterationIndex <= estIterations ?
            logger.group('Decimate iteration', {
                index: iterationIndex,
                total: estIterations
            }) :
            logger.group('Decimate iteration');

        const kdSub = logger.group('Building KD-tree');

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

        const cache = buildPerSplatCache(n, cx, cy, cz, cop, cs0, cs1, cs2, cr0, cr1, cr2, cr3);

        const posTable = new DataTable([
            new Column('x', cx instanceof Float32Array ? cx : new Float32Array(cx as any)),
            new Column('y', cy instanceof Float32Array ? cy : new Float32Array(cy as any)),
            new Column('z', cz instanceof Float32Array ? cz : new Float32Array(cz as any))
        ]);
        const kdTree = new KdTree(posTable);
        kdSub.end();

        let edgeCapacity = Math.ceil(n * kEff / 2);
        let edgeU = new Uint32Array(edgeCapacity);
        let edgeV = new Uint32Array(edgeCapacity);
        let edgeCount = 0;
        const queryPoint = new Float32Array(3);

        const knnInterval = Math.max(1, Math.ceil(n / PROGRESS_TICKS));
        const knnTicks = Math.ceil(n / knnInterval);
        const knnBar = logger.bar('Finding nearest neighbors', knnTicks);
        for (let i = 0; i < n; i++) {
            queryPoint[0] = cx[i] as number;
            queryPoint[1] = cy[i] as number;
            queryPoint[2] = cz[i] as number;
            const knn = kdTree.findKNearest(queryPoint, kEff + 1);
            for (let ki = 0; ki < knn.indices.length; ki++) {
                const j = knn.indices[ki];
                if (j <= i) continue;
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
            if ((i + 1) % knnInterval === 0) knnBar.tick();
        }
        if (n % knnInterval !== 0) knnBar.tick();
        knnBar.end();

        if (edgeCount === 0) {
            g.end();
            break;
        }

        const appData: any[] = [];
        for (let ai = 0; ai < allAppearanceCols.length; ai++) {
            const col = current.getColumnByName(allAppearanceCols[ai]);
            if (col) appData.push(col.data);
        }

        const mergesNeeded = n - targetCount;
        const costs = new Float32Array(edgeCount);

        const costInterval = Math.max(1, Math.ceil(edgeCount / PROGRESS_TICKS));
        const costTicks = Math.ceil(edgeCount / costInterval);
        const costBar = logger.bar('Computing edge costs', costTicks);
        for (let e = 0; e < edgeCount; e++) {
            costs[e] = computeEdgeCost(edgeU[e], edgeV[e], cx, cy, cz,
                cache, Z, appData, appData.length);
            if ((e + 1) % costInterval === 0) costBar.tick();
        }
        if (edgeCount % costInterval !== 0) costBar.tick();
        costBar.end();

        // Sort and greedy disjoint pair selection
        const sorted = new Uint32Array(edgeCount);
        const validCount = radixSortIndicesByFloat(sorted, edgeCount, costs);

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

        if (pairs.length === 0) {
            g.end();
            break;
        }

        // Mark which indices are consumed by merging
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

        // Copy unmerged splats
        let dst = 0;
        for (let t = 0; t < keepIndices.length; t++, dst++) {
            const src = keepIndices[t];
            for (let c = 0; c < cols.length; c++) {
                newTable.columns[c].data[dst] = cols[c].data[src] as number;
            }
        }

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
                mergeOut, appData, appData.length);

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
};

export { sortByVisibility, simplifyGaussians };
