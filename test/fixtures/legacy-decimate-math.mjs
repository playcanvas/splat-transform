/**
 * Verbatim reference copies of the legacy decimation math from
 * src/lib/data-table/decimate.ts (deleted when the chunk-native decimation
 * landed), adapted only to read splats from a SplatView-shaped object:
 *
 *   view = { pos: Float32Array (3/splat), geo: Float32Array (8/splat:
 *            rot_0..3, scale_0..2, opacity), color: Float32Array
 *            (colorDim/splat), colorDim }
 *
 * These are the parity references: the new implementation's n=2 merge and
 * per-edge costs must match these formulas exactly.
 */

const LOG2PI = Math.log(2 * Math.PI);
const EPS_COV = 1e-8;

// ---------- sigmoid / logit ----------

export const sigmoid = x => 1 / (1 + Math.exp(-x));

export const logit = (p) => {
    p = Math.max(1e-7, Math.min(1 - 1e-7, p));
    return Math.log(p / (1 - p));
};

const logAddExp = (a, b) => {
    if (a === -Infinity) return b;
    if (b === -Infinity) return a;
    const m = Math.max(a, b);
    return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
};

// ---------- PRNG ----------

const mulberry32 = (seed) => {
    return () => {
        let t = (seed += 0x6d2b79f5);
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
};

export const makeGaussianSamples = (n, seed) => {
    const rand = mulberry32(seed >>> 0);
    const out = [];
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

const quatToRotmat = (qw, qx, qy, qz, out, o) => {
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

const sigmaFromRotVar = (R, r, vx, vy, vz, out, o) => {
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

const det3 = (A, o) => {
    return (
        A[o] * (A[o + 4] * A[o + 8] - A[o + 5] * A[o + 7]) -
        A[o + 1] * (A[o + 3] * A[o + 8] - A[o + 5] * A[o + 6]) +
        A[o + 2] * (A[o + 3] * A[o + 7] - A[o + 4] * A[o + 6])
    );
};

const gaussLogpdfDiagrot = (x, y, z, mx, my, mz, R, ro, invx, invy, invz, logdet) => {
    const dx = x - mx, dy = y - my, dz = z - mz;
    const y0 = dx * R[ro] + dy * R[ro + 3] + dz * R[ro + 6];
    const y1 = dx * R[ro + 1] + dy * R[ro + 4] + dz * R[ro + 7];
    const y2 = dx * R[ro + 2] + dy * R[ro + 5] + dz * R[ro + 8];
    const quad = y0 * y0 * invx + y1 * y1 * invy + y2 * y2 * invz;
    return -0.5 * (3 * LOG2PI + logdet + quad);
};

const eigenSymmetric3x3 = (Ain, A, V) => {
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

const rotmatToQuat = (R, o, out, oo) => {
    const m00 = R[o], m11 = R[o + 4], m22 = R[o + 8];
    const tr = m00 + m11 + m22;
    let qw, qx, qy, qz;

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

const ELLIPSOID_P = 1.6075;
export const ellipsoidArea = (sx, sy, sz) => {
    const a = Math.pow(sx * sy, ELLIPSOID_P);
    const b = Math.pow(sx * sz, ELLIPSOID_P);
    const c = Math.pow(sy * sz, ELLIPSOID_P);
    return 4 * Math.PI * Math.pow((a + b + c) / 3, 1 / ELLIPSOID_P);
};

// ====================== VIEW → LEGACY COLUMNS ======================

// The legacy functions take per-column accessor arrays (cx, cy, ... appData).
// Extract them once per view (memoized) so tests can call the adapters in a loop.
const columnsCache = new WeakMap();

export const viewColumns = (view) => {
    let cols = columnsCache.get(view);
    if (cols) return cols;
    const n = view.pos.length / 3;
    const col = f => Float32Array.from({ length: n }, (_, i) => f(i));
    cols = {
        n,
        cx: col(i => view.pos[i * 3]),
        cy: col(i => view.pos[i * 3 + 1]),
        cz: col(i => view.pos[i * 3 + 2]),
        cr0: col(i => view.geo[i * 8]),
        cr1: col(i => view.geo[i * 8 + 1]),
        cr2: col(i => view.geo[i * 8 + 2]),
        cr3: col(i => view.geo[i * 8 + 3]),
        cs0: col(i => view.geo[i * 8 + 4]),
        cs1: col(i => view.geo[i * 8 + 5]),
        cs2: col(i => view.geo[i * 8 + 6]),
        cop: col(i => view.geo[i * 8 + 7]),
        appData: Array.from({ length: view.colorDim }, (_, k) => col(i => view.color[i * view.colorDim + k]))
    };
    columnsCache.set(view, cols);
    return cols;
};

// ====================== PER-SPLAT CACHE (legacy buildPerSplatCache) ======================

const cacheCache = new WeakMap();

export const legacyBuildCache = (view) => {
    let cache = cacheCache.get(view);
    if (cache) return cache;
    const { n, cop, cs0, cs1, cs2, cr0, cr1, cr2, cr3 } = viewColumns(view);
    const R = new Float32Array(n * 9);
    const v = new Float32Array(n * 3);
    const invdiag = new Float32Array(n * 3);
    const logdet = new Float32Array(n);
    const sigma = new Float32Array(n * 9);
    const mass = new Float32Array(n);

    for (let i = 0; i < n; i++) {
        const i3 = 3 * i;
        const i9 = 9 * i;

        const linAlpha = sigmoid(cop[i]);
        const sx = Math.max(Math.exp(cs0[i]), 1e-12);
        const sy = Math.max(Math.exp(cs1[i]), 1e-12);
        const sz = Math.max(Math.exp(cs2[i]), 1e-12);

        const vx = sx * sx + EPS_COV;
        const vy = sy * sy + EPS_COV;
        const vz = sz * sz + EPS_COV;

        v[i3] = vx; v[i3 + 1] = vy; v[i3 + 2] = vz;
        invdiag[i3] = 1 / Math.max(vx, 1e-30);
        invdiag[i3 + 1] = 1 / Math.max(vy, 1e-30);
        invdiag[i3 + 2] = 1 / Math.max(vz, 1e-30);
        logdet[i] = Math.log(Math.max(vx, 1e-30)) + Math.log(Math.max(vy, 1e-30)) + Math.log(Math.max(vz, 1e-30));

        let qw = cr0[i], qx = cr1[i], qy = cr2[i], qz = cr3[i];
        const qn = Math.hypot(qw, qx, qy, qz);
        const invq = 1 / Math.max(qn, 1e-12);
        qw *= invq; qx *= invq; qy *= invq; qz *= invq;

        quatToRotmat(qw, qx, qy, qz, R, i9);
        sigmaFromRotVar(R, i9, vx, vy, vz, sigma, i9);

        mass[i] = linAlpha * ellipsoidArea(sx, sy, sz) + 1e-12;
    }

    cache = { R, v, invdiag, logdet, sigma, mass };
    cacheCache.set(view, cache);
    return cache;
};

// ====================== COST FUNCTION (legacy computeEdgeCost) ======================

const createMergeScratch = () => ({
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

const sharedScratch = createMergeScratch();

export const legacyEdgeCost = (view, i, j, Z) => {
    const { cx, cy, cz, appData } = viewColumns(view);
    const cache = legacyBuildCache(view);
    const scratch = sharedScratch;
    const appColCount = appData.length;

    const i3 = 3 * i, j3 = 3 * j;
    const i9 = 9 * i, j9 = 9 * j;

    const mux = cx[i], muy = cy[i], muz = cz[i];
    const mvx = cx[j], mvy = cy[j], mvz = cz[j];

    const wi = cache.mass[i], wj = cache.mass[j];
    const W = wi + wj;
    const Wsafe = W > 0 ? W : 1;

    let pi = wi / Wsafe;
    pi = Math.max(1e-12, Math.min(1 - 1e-12, pi));
    const pj = 1 - pi;
    const logPi = Math.log(pi);
    const logPj = Math.log(pj);

    const mmx = pi * mux + pj * mvx;
    const mmy = pi * muy + pj * mvy;
    const mmz = pi * muz + pj * mvz;

    const dix = mux - mmx, diy = muy - mmy, diz = muz - mmz;
    const djx = mvx - mmx, djy = mvy - mmy, djz = mvz - mmz;

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

    sigm[1] = sigm[3] = 0.5 * (sigm[1] + sigm[3]);
    sigm[2] = sigm[6] = 0.5 * (sigm[2] + sigm[6]);
    sigm[5] = sigm[7] = 0.5 * (sigm[5] + sigm[7]);
    sigm[0] += EPS_COV;
    sigm[4] += EPS_COV;
    sigm[8] += EPS_COV;

    const detm = Math.max(det3(sigm, 0), 1e-30);
    const logdetm = Math.log(detm);

    const EpNegLogQ = 0.5 * (3 * LOG2PI + logdetm + 3);

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

        const xix = mux + z0 * stdix * cache.R[i9 + 0] + z1 * stdiy * cache.R[i9 + 1] + z2 * stdiz * cache.R[i9 + 2];
        const xiy = muy + z0 * stdix * cache.R[i9 + 3] + z1 * stdiy * cache.R[i9 + 4] + z2 * stdiz * cache.R[i9 + 5];
        const xiz = muz + z0 * stdix * cache.R[i9 + 6] + z1 * stdiy * cache.R[i9 + 7] + z2 * stdiz * cache.R[i9 + 8];

        const xjx = mvx + z0 * stdjx * cache.R[j9 + 0] + z1 * stdjy * cache.R[j9 + 1] + z2 * stdjz * cache.R[j9 + 2];
        const xjy = mvy + z0 * stdjx * cache.R[j9 + 3] + z1 * stdjy * cache.R[j9 + 4] + z2 * stdjz * cache.R[j9 + 5];
        const xjz = mvz + z0 * stdjx * cache.R[j9 + 6] + z1 * stdjy * cache.R[j9 + 7] + z2 * stdjz * cache.R[j9 + 8];

        const logNiOnI = gaussLogpdfDiagrot(xix, xiy, xiz, mux, muy, muz,
            cache.R, i9, cache.invdiag[i3], cache.invdiag[i3 + 1], cache.invdiag[i3 + 2], cache.logdet[i]);
        const logNjOnI = gaussLogpdfDiagrot(xix, xiy, xiz, mvx, mvy, mvz,
            cache.R, j9, cache.invdiag[j3], cache.invdiag[j3 + 1], cache.invdiag[j3 + 2], cache.logdet[j]);
        sumLogpOnI += logAddExp(logPi + logNiOnI, logPj + logNjOnI);

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

    let cSh = 0;
    for (let k = 0; k < appColCount; k++) {
        const d = appData[k][i] - appData[k][j];
        cSh += d * d;
    }

    return geo + cSh;
};

// ====================== MERGE (legacy momentMatch) ======================

export const legacyMomentMatch = (view, i, j) => {
    const { cx, cy, cz, cop, cs0, cs1, cs2, cr0, cr1, cr2, cr3, appData } = viewColumns(view);
    const appColCount = appData.length;
    const scratch = sharedScratch;
    const out = {
        mu: new Float64Array(3),
        sc: new Float64Array(3),
        q: new Float64Array(4),
        op: 0,
        sh: new Float64Array(appColCount)
    };

    const sxi = Math.max(Math.exp(cs0[i]), 1e-12);
    const syi = Math.max(Math.exp(cs1[i]), 1e-12);
    const szi = Math.max(Math.exp(cs2[i]), 1e-12);
    const sxj = Math.max(Math.exp(cs0[j]), 1e-12);
    const syj = Math.max(Math.exp(cs1[j]), 1e-12);
    const szj = Math.max(Math.exp(cs2[j]), 1e-12);

    const alphaI = sigmoid(cop[i]);
    const alphaJ = sigmoid(cop[j]);

    const Ai = ellipsoidArea(sxi, syi, szi);
    const Aj = ellipsoidArea(sxj, syj, szj);
    const wi = alphaI * Ai + 1e-30;
    const wj = alphaJ * Aj + 1e-30;
    const W = wi + wj;
    const pi = wi / W;
    const pj = wj / W;

    const mxi = cx[i], myi = cy[i], mzi = cz[i];
    const mxj = cx[j], myj = cy[j], mzj = cz[j];
    const mux = pi * mxi + pj * mxj;
    const muy = pi * myi + pj * myj;
    const muz = pi * mzi + pj * mzj;

    const SigI = scratch.sigI;
    const SigJ = scratch.sigJ;
    const Ri = scratch.rI;
    const Rj = scratch.rJ;

    let qwi = cr0[i], qxi = cr1[i], qyi = cr2[i], qzi = cr3[i];
    const ni = 1 / Math.max(Math.hypot(qwi, qxi, qyi, qzi), 1e-12);
    qwi *= ni; qxi *= ni; qyi *= ni; qzi *= ni;
    let qwj = cr0[j], qxj = cr1[j], qyj = cr2[j], qzj = cr3[j];
    const nj = 1 / Math.max(Math.hypot(qwj, qxj, qyj, qzj), 1e-12);
    qwj *= nj; qxj *= nj; qyj *= nj; qzj *= nj;

    quatToRotmat(qwi, qxi, qyi, qzi, Ri, 0);
    quatToRotmat(qwj, qxj, qyj, qzj, Rj, 0);
    sigmaFromRotVar(Ri, 0, sxi * sxi, syi * syi, szi * szi, SigI, 0);
    sigmaFromRotVar(Rj, 0, sxj * sxj, syj * syj, szj * szj, SigJ, 0);

    const dix = mxi - mux, diy = myi - muy, diz = mzi - muz;
    const djx = mxj - mux, djy = myj - muy, djz = mzj - muz;

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

    const eigA = scratch.eigA;
    const eigV = scratch.eigV;
    eigenSymmetric3x3(Sig, eigA, eigV);
    const vecs = eigV;

    const v0 = eigA[0], v1 = eigA[4], v2 = eigA[8];
    let o0, o1, o2;
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

    const alphaM = Math.min(1, W / Math.max(ellipsoidArea(s0, s1, s2), 1e-30));

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

    for (let k = 0; k < appColCount; k++) {
        out.sh[k] = pi * appData[k][i] + pj * appData[k][j];
    }

    return out;
};
