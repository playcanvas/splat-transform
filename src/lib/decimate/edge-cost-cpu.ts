/**
 * CPU edge-cost path for chunk-native decimation.
 *
 * PROTOTYPE — principled L2 field error. The cost of merging splats i and j
 * into the moment-matched splat m is the squared L2 norm of the difference
 * between the original emitted field (αc·G for each splat, G the unnormalized
 * Gaussian kernel) and the merged field:
 *
 *     E = ∫ ‖ αᵢcᵢGᵢ + αⱼcⱼGⱼ − αₘcₘGₘ ‖² dx
 *
 * which expands to a closed form in Gaussian–Gaussian L2 products
 * ⟨G_a,G_b⟩ = (2π)^{3/2}|Σ_a|^{1/2}|Σ_b|^{1/2}|Σ_a+Σ_b|^{-1/2}
 *             exp(−½ dᵀ(Σ_a+Σ_b)⁻¹ d),   d = μ_a − μ_b.
 *
 * Every term scales with √|Σ| (Gaussian volume), so merging large (distant
 * sky) Gaussians costs far more than a geometrically-similar tiny merge —
 * unlike the scale-invariant KL cost this replaces. The metric is ≥ 0 by
 * construction (a squared norm). Merged parameters (μ, Σ, α, colour) are the
 * faithful moment-matched values from {@link mergeGroup}, so the cost predicts
 * the real output error.
 *
 * Modelling choices (prototype): amplitude is α × base colour (0.5 + C0·f_dc),
 * i.e. higher-order SH is carried through the merge but not scored in the
 * ranking cost.
 *
 * Engine-free.
 */

import {
    EPS_COV,
    sigmoid,
    ellipsoidArea,
    quatToRotmat,
    sigmaFromRotVar,
    type SplatView
} from './moment-match';

/** SH band-0 constant (f_dc → base colour: 0.5 + C0·f_dc). */
const C0 = 0.28209479177387814;

/**
 * Scale-free colour dissimilarity weight. Unlike the field-L2's own colour
 * sensitivity (which vanishes ∝σ³ for faint splats), this term keeps
 * light-vs-dark pairing selective at any scale — without it, fine texture
 * (validated on grass/snow speckle) merges colour-blind and washes to mush.
 * Calibrated as λ=1e-6 over raw f_dc coefficients (crop+full-scale study);
 * base colour = 0.5 + C0·f_dc and C0² = 1/(4π), so in base-colour space the
 * constant is 4π·1e-6. Mirrored in the GpuEdgeCost WGSL kernel.
 */
const COLOR_WEIGHT = 4 * Math.PI * 1e-6;

/** ⟨G,G⟩ self-product constant: (2π)^{3/2}·2^{-3/2} = π^{3/2}. */
const PI_1_5 = Math.PI ** 1.5;

/** Cross-product constant (2π)^{3/2}. */
const TWO_PI_1_5 = (2 * Math.PI) ** 1.5;

/**
 * Per-splat derived quantities for the L2 field cost. `sig` holds each
 * covariance's 6 unique components [xx, xy, xz, yy, yz, zz]; `sqrtDet` is
 * √|Σ| = sx·sy·sz; `alpha` is the linear opacity; `base` is the 3-channel base
 * colour; `baseN2` is |base|²; `mass` is the area·α merge weight.
 */
type CostCache = {
    sig: Float32Array;
    sqrtDet: Float32Array;
    alpha: Float32Array;
    base: Float32Array;
    baseN2: Float32Array;
    mass: Float32Array;
};

const buildCostCache = (view: SplatView): CostCache => {
    const { geo, color, colorDim } = view;
    const n = geo.length / 8;
    const sig = new Float32Array(n * 6);
    const sqrtDet = new Float32Array(n);
    const alpha = new Float32Array(n);
    const base = new Float32Array(n * 3);
    const baseN2 = new Float32Array(n);
    const mass = new Float32Array(n);
    const R = new Float32Array(9);
    const S = new Float32Array(9);

    for (let i = 0; i < n; i++) {
        const i8 = 8 * i;

        const a = sigmoid(geo[i8 + 7]);
        const sx = Math.max(Math.exp(geo[i8 + 4]), 1e-12);
        const sy = Math.max(Math.exp(geo[i8 + 5]), 1e-12);
        const sz = Math.max(Math.exp(geo[i8 + 6]), 1e-12);

        let qw = geo[i8], qx = geo[i8 + 1], qy = geo[i8 + 2], qz = geo[i8 + 3];
        const invq = 1 / Math.max(Math.hypot(qw, qx, qy, qz), 1e-12);
        qw *= invq; qx *= invq; qy *= invq; qz *= invq;

        quatToRotmat(qw, qx, qy, qz, R, 0);
        sigmaFromRotVar(R, 0, sx * sx, sy * sy, sz * sz, S, 0);

        const i6 = 6 * i;
        sig[i6] = S[0]; sig[i6 + 1] = S[1]; sig[i6 + 2] = S[2];
        sig[i6 + 3] = S[4]; sig[i6 + 4] = S[5]; sig[i6 + 5] = S[8];

        sqrtDet[i] = sx * sy * sz;
        alpha[i] = a;
        mass[i] = a * ellipsoidArea(sx, sy, sz) + 1e-30;

        const i3 = 3 * i;
        const br = 0.5 + C0 * color[i * colorDim];
        const bg = 0.5 + C0 * color[i * colorDim + 1];
        const bb = 0.5 + C0 * color[i * colorDim + 2];
        base[i3] = br; base[i3 + 1] = bg; base[i3 + 2] = bb;
        baseN2[i] = br * br + bg * bg + bb * bb;
    }

    return { sig, sqrtDet, alpha, base, baseN2, mass };
};

// dᵀ(A)⁻¹d and √|A| for a symmetric 3×3 A = [xx,xy,xz,yy,yz,zz]; returns the
// Gaussian cross-product ⟨G_a,G_b⟩ scaled by the caller's √|Σ_a|·√|Σ_b|.
const crossG = (
    sdA: number, sdB: number,
    xx: number, xy: number, xz: number, yy: number, yz: number, zz: number,
    dx: number, dy: number, dz: number
): number => {
    // Signed cofactors (adj is symmetric); det via row-0 expansion.
    const c00 = yy * zz - yz * yz;
    const c01 = xz * yz - xy * zz;
    const c02 = xy * yz - xz * yy;
    const c11 = xx * zz - xz * xz;
    const c12 = xy * xz - xx * yz;
    const c22 = xx * yy - xy * xy;
    const det = Math.max(xx * c00 + xy * c01 + xz * c02, 1e-30);
    const quadAdj = c00 * dx * dx + c11 * dy * dy + c22 * dz * dz +
        2 * (c01 * dx * dy + c02 * dx * dz + c12 * dy * dz);
    const quad = quadAdj / det;
    return TWO_PI_1_5 * sdA * sdB / Math.sqrt(det) * Math.exp(-0.5 * quad);
};

/**
 * L2 field-error cost of merging splats `i` and `j` of the view.
 *
 * @param view - Splat columns.
 * @param cache - Per-splat cache from {@link buildCostCache}.
 * @param i - First splat (view row).
 * @param j - Second splat (view row).
 * @returns The edge cost (≥ 0).
 */
const computeEdgeCostView = (
    view: SplatView,
    cache: CostCache,
    i: number,
    j: number
): number => {
    const { pos } = view;
    const { sig, sqrtDet, alpha, base, baseN2, mass } = cache;

    const i3 = 3 * i, j3 = 3 * j;
    const i6 = 6 * i, j6 = 6 * j;

    const mux = pos[i3], muy = pos[i3 + 1], muz = pos[i3 + 2];
    const mvx = pos[j3], mvy = pos[j3 + 1], mvz = pos[j3 + 2];

    const mi = mass[i], mj = mass[j];
    const W = mi + mj;
    const pi = W > 0 ? mi / W : 0.5;
    const pj = 1 - pi;

    // Merged mean.
    const mmx = pi * mux + pj * mvx;
    const mmy = pi * muy + pj * mvy;
    const mmz = pi * muz + pj * mvz;

    const dix = mux - mmx, diy = muy - mmy, diz = muz - mmz;
    const djx = mvx - mmx, djy = mvy - mmy, djz = mvz - mmz;

    // Merged covariance: Σ pₖ(δₖδₖᵀ + Σₖ) + EPS_COV·I  (law of total variance).
    const sxx = pi * (dix * dix + sig[i6]) + pj * (djx * djx + sig[j6]) + EPS_COV;
    const sxy = pi * (dix * diy + sig[i6 + 1]) + pj * (djx * djy + sig[j6 + 1]);
    const sxz = pi * (dix * diz + sig[i6 + 2]) + pj * (djx * djz + sig[j6 + 2]);
    const syy = pi * (diy * diy + sig[i6 + 3]) + pj * (djy * djy + sig[j6 + 3]) + EPS_COV;
    const syz = pi * (diy * diz + sig[i6 + 4]) + pj * (djy * djz + sig[j6 + 4]);
    const szz = pi * (diz * diz + sig[i6 + 5]) + pj * (djz * djz + sig[j6 + 5]) + EPS_COV;

    const detm = Math.max(
        sxx * (syy * szz - syz * syz) - sxy * (sxy * szz - syz * sxz) + sxz * (sxy * syz - syy * sxz),
        1e-30
    );
    const sqrtDetM = Math.sqrt(detm);

    // Merged opacity: mass-conserving, capped at 1 (needs merged scales, i.e.
    // eigenvalues of Σ_m — Smith's closed form for a symmetric 3×3).
    const q = (sxx + syy + szz) / 3;
    const p1 = sxy * sxy + sxz * sxz + syz * syz;
    let e0: number, e1: number, e2: number;
    if (p1 <= 1e-30) {
        e0 = sxx; e1 = syy; e2 = szz;
    } else {
        const p2 = (sxx - q) * (sxx - q) + (syy - q) * (syy - q) + (szz - q) * (szz - q) + 2 * p1;
        const p = Math.sqrt(p2 / 6);
        const ip = 1 / p;
        const b00 = (sxx - q) * ip, b11 = (syy - q) * ip, b22 = (szz - q) * ip;
        const b01 = sxy * ip, b02 = sxz * ip, b12 = syz * ip;
        const detB = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02) + b02 * (b01 * b12 - b11 * b02);
        let r = detB / 2;
        r = r < -1 ? -1 : (r > 1 ? 1 : r);
        const phi = Math.acos(r) / 3;
        e0 = q + 2 * p * Math.cos(phi);
        e2 = q + 2 * p * Math.cos(phi + 2 * Math.PI / 3);
        e1 = 3 * q - e0 - e2;
    }
    const s0 = Math.sqrt(Math.max(e0, 1e-18));
    const s1 = Math.sqrt(Math.max(e1, 1e-18));
    const s2 = Math.sqrt(Math.max(e2, 1e-18));
    const alphaM = Math.min(1, W / Math.max(ellipsoidArea(s0, s1, s2), 1e-30));

    // Base colours and their dots (merged colour = mass-weighted average).
    const bir = base[i3], big = base[i3 + 1], bib = base[i3 + 2];
    const bjr = base[j3], bjg = base[j3 + 1], bjb = base[j3 + 2];
    const bij = bir * bjr + big * bjg + bib * bjb;   // base_i · base_j
    const bni = baseN2[i];                            // base_i · base_i
    const bnj = baseN2[j];                            // base_j · base_j
    const bim = pi * bni + pj * bij;                  // base_i · base_m
    const bjm = pi * bij + pj * bnj;                  // base_j · base_m
    const bnm = pi * pi * bni + 2 * pi * pj * bij + pj * pj * bnj; // base_m · base_m

    const ai = alpha[i], aj = alpha[j], am = alphaM;
    const sdi = sqrtDet[i], sdj = sqrtDet[j];

    // Self terms ⟨G,G⟩ = π^{3/2}·√|Σ|.
    const selfI = ai * ai * bni * PI_1_5 * sdi;
    const selfJ = aj * aj * bnj * PI_1_5 * sdj;
    const selfM = am * am * bnm * PI_1_5 * sqrtDetM;

    // Cross terms (amplitude dot × Gaussian overlap).
    const cIJ = crossG(sdi, sdj,
        sig[i6] + sig[j6], sig[i6 + 1] + sig[j6 + 1], sig[i6 + 2] + sig[j6 + 2],
        sig[i6 + 3] + sig[j6 + 3], sig[i6 + 4] + sig[j6 + 4], sig[i6 + 5] + sig[j6 + 5],
        mux - mvx, muy - mvy, muz - mvz);
    const cIM = crossG(sdi, sqrtDetM,
        sig[i6] + sxx, sig[i6 + 1] + sxy, sig[i6 + 2] + sxz,
        sig[i6 + 3] + syy, sig[i6 + 4] + syz, sig[i6 + 5] + szz,
        dix, diy, diz);
    const cJM = crossG(sdj, sqrtDetM,
        sig[j6] + sxx, sig[j6 + 1] + sxy, sig[j6 + 2] + sxz,
        sig[j6 + 3] + syy, sig[j6 + 4] + syz, sig[j6 + 5] + szz,
        djx, djy, djz);

    const E = selfI + selfJ + selfM +
        2 * ai * aj * bij * cIJ -
        2 * ai * am * bim * cIM -
        2 * aj * am * bjm * cJM;

    // Clamp float-noise negatives (E is a squared norm ≥ 0) but preserve
    // NaN/Inf from degenerate inputs so the caller can fail loud.
    // |Δbase|² = |b_i|² + |b_j|² − 2·b_i·b_j — no extra loads needed.
    return (E < 0 ? 0 : E) + COLOR_WEIGHT * (bni + bnj - 2 * bij);
};

export { buildCostCache, computeEdgeCostView, COLOR_WEIGHT, type CostCache };
