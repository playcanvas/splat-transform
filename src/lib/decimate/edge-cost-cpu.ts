/**
 * CPU edge-cost path for chunk-native decimation — a port of the legacy
 * `computeEdgeCost` + `buildPerSplatCache` (CPU variant) over a
 * {@link SplatView}. Formula-identical to the legacy implementation and to
 * the `GpuEdgeCost` WGSL kernel; used as the no-device fallback and by the
 * GPU parity tests.
 *
 * Engine-free.
 */

import {
    EPS_COV,
    LOG2PI,
    logAddExp,
    sigmoid,
    ellipsoidArea,
    quatToRotmat,
    sigmaFromRotVar,
    det3,
    gaussLogpdfDiagrot,
    type SplatView,
    type MergeScratch
} from './moment-match';

/**
 * Per-splat derived quantities for the cost function (legacy
 * `buildPerSplatCache`, forGpu = false). `mass` uses the cost-path epsilon
 * (+1e-12), matching legacy exactly.
 */
type CostCache = {
    R: Float32Array;
    v: Float32Array;
    invdiag: Float32Array;
    logdet: Float32Array;
    sigma: Float32Array;
    mass: Float32Array;
};

const buildCostCache = (view: SplatView): CostCache => {
    const { geo } = view;
    const n = geo.length / 8;
    const R = new Float32Array(n * 9);
    const v = new Float32Array(n * 3);
    const invdiag = new Float32Array(n * 3);
    const logdet = new Float32Array(n);
    const sigma = new Float32Array(n * 9);
    const mass = new Float32Array(n);

    for (let i = 0; i < n; i++) {
        const i3 = 3 * i;
        const i8 = 8 * i;
        const i9 = 9 * i;

        const linAlpha = sigmoid(geo[i8 + 7]);
        const sx = Math.max(Math.exp(geo[i8 + 4]), 1e-12);
        const sy = Math.max(Math.exp(geo[i8 + 5]), 1e-12);
        const sz = Math.max(Math.exp(geo[i8 + 6]), 1e-12);

        const vx = sx * sx + EPS_COV;
        const vy = sy * sy + EPS_COV;
        const vz = sz * sz + EPS_COV;

        v[i3] = vx; v[i3 + 1] = vy; v[i3 + 2] = vz;
        invdiag[i3] = 1 / Math.max(vx, 1e-30);
        invdiag[i3 + 1] = 1 / Math.max(vy, 1e-30);
        invdiag[i3 + 2] = 1 / Math.max(vz, 1e-30);
        logdet[i] = Math.log(Math.max(vx, 1e-30)) + Math.log(Math.max(vy, 1e-30)) + Math.log(Math.max(vz, 1e-30));

        let qw = geo[i8], qx = geo[i8 + 1], qy = geo[i8 + 2], qz = geo[i8 + 3];
        const qn = Math.hypot(qw, qx, qy, qz);
        const invq = 1 / Math.max(qn, 1e-12);
        qw *= invq; qx *= invq; qy *= invq; qz *= invq;

        quatToRotmat(qw, qx, qy, qz, R, i9);
        sigmaFromRotVar(R, i9, vx, vy, vz, sigma, i9);

        mass[i] = linAlpha * ellipsoidArea(sx, sy, sz) + 1e-12;
    }

    return { R, v, invdiag, logdet, sigma, mass };
};

/**
 * Edge cost between splats `i` and `j` of the view: KL-style geometric term
 * (single MC sample) + L2 over the color/SH coefficients. Legacy
 * `computeEdgeCost`, verbatim.
 *
 * @param view - Splat columns.
 * @param cache - Per-splat cache from {@link buildCostCache}.
 * @param i - First splat (view row).
 * @param j - Second splat (view row).
 * @param Z - MC samples (legacy: one sample, seed 0).
 * @param scratch - Merge scratch (uses `sigm`).
 * @returns The edge cost.
 */
const computeEdgeCostView = (
    view: SplatView,
    cache: CostCache,
    i: number,
    j: number,
    Z: Float64Array[],
    scratch: MergeScratch
): number => {
    const { pos, color, colorDim } = view;
    const i3 = 3 * i, j3 = 3 * j;
    const i9 = 9 * i, j9 = 9 * j;

    const mux = pos[i3], muy = pos[i3 + 1], muz = pos[i3 + 2];
    const mvx = pos[j3], mvy = pos[j3 + 1], mvz = pos[j3 + 2];

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
    const geoCost = EpLogp + EpNegLogQ;

    let cSh = 0;
    for (let c = 0; c < colorDim; c++) {
        const d = color[i * colorDim + c] - color[j * colorDim + c];
        cSh += d * d;
    }

    return geoCost + cSh;
};

export { buildCostCache, computeEdgeCostView, type CostCache };
