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
 * Floats per splat in the interleaved cost cache (see {@link buildSplatCache}
 * for the layout). Single source of truth — the GPU kernels and the re-costed
 * selection state share this exact layout.
 */
const CACHE_STRIDE = 16;

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
 * Build the per-splat cost cache: {@link CACHE_STRIDE} interleaved floats per
 * splat, written into `out` (reusable scratch; only the leading n·16 floats
 * are touched). Layout per row:
 *
 *   [0..2]   mean xyz
 *   [3..8]   covariance Σ (xx, xy, xz, yy, yz, zz) — unregularized
 *   [9]      sqrtDet = √|Σ| = sx·sy·sz
 *   [10]     alpha (linear opacity)
 *   [11]     mass (area·α merge weight)
 *   [12..14] base colour (0.5 + C0·f_dc per channel)
 *   [15]     |base|²
 *
 * This exact buffer uploads to the GPU kernels and persists per-gaussian for
 * the re-costed selection, so CPU and GPU read identical per-splat inputs.
 *
 * @param view - Splat columns (only the first 3 colour components are read).
 * @param out - Destination, at least n·{@link CACHE_STRIDE} floats.
 * @returns The splat count n.
 */
const buildSplatCache = (view: SplatView, out: Float32Array): number => {
    const { pos, geo, color, colorDim } = view;
    const n = geo.length / 8;
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

        const o = i * CACHE_STRIDE;
        const i3 = 3 * i;
        out[o] = pos[i3]; out[o + 1] = pos[i3 + 1]; out[o + 2] = pos[i3 + 2];
        out[o + 3] = S[0]; out[o + 4] = S[1]; out[o + 5] = S[2];
        out[o + 6] = S[4]; out[o + 7] = S[5]; out[o + 8] = S[8];
        out[o + 9] = sx * sy * sz;
        out[o + 10] = a;
        out[o + 11] = a * ellipsoidArea(sx, sy, sz) + 1e-30;

        const br = 0.5 + C0 * color[i * colorDim];
        const bg = 0.5 + C0 * color[i * colorDim + 1];
        const bb = 0.5 + C0 * color[i * colorDim + 2];
        out[o + 12] = br; out[o + 13] = bg; out[o + 14] = bb;
        out[o + 15] = br * br + bg * bg + bb * bb;
    }

    return n;
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
 * L2 field-error cost of merging splats `i` and `j`.
 *
 * @param cache - Per-splat cache from {@link buildSplatCache}.
 * @param i - First splat (cache row).
 * @param j - Second splat (cache row).
 * @returns The edge cost (≥ 0).
 */
const computeEdgeCost = (
    cache: Float32Array,
    i: number,
    j: number
): number => {
    const io = i * CACHE_STRIDE, jo = j * CACHE_STRIDE;

    const mux = cache[io], muy = cache[io + 1], muz = cache[io + 2];
    const mvx = cache[jo], mvy = cache[jo + 1], mvz = cache[jo + 2];

    const mi = cache[io + 11], mj = cache[jo + 11];
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
    const sxx = pi * (dix * dix + cache[io + 3]) + pj * (djx * djx + cache[jo + 3]) + EPS_COV;
    const sxy = pi * (dix * diy + cache[io + 4]) + pj * (djx * djy + cache[jo + 4]);
    const sxz = pi * (dix * diz + cache[io + 5]) + pj * (djx * djz + cache[jo + 5]);
    const syy = pi * (diy * diy + cache[io + 6]) + pj * (djy * djy + cache[jo + 6]) + EPS_COV;
    const syz = pi * (diy * diz + cache[io + 7]) + pj * (djy * djz + cache[jo + 7]);
    const szz = pi * (diz * diz + cache[io + 8]) + pj * (djz * djz + cache[jo + 8]) + EPS_COV;

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
    const bir = cache[io + 12], big = cache[io + 13], bib = cache[io + 14];
    const bjr = cache[jo + 12], bjg = cache[jo + 13], bjb = cache[jo + 14];
    const bij = bir * bjr + big * bjg + bib * bjb;   // base_i · base_j
    const bni = cache[io + 15];                       // base_i · base_i
    const bnj = cache[jo + 15];                       // base_j · base_j
    const bim = pi * bni + pj * bij;                  // base_i · base_m
    const bjm = pi * bij + pj * bnj;                  // base_j · base_m
    const bnm = pi * pi * bni + 2 * pi * pj * bij + pj * pj * bnj; // base_m · base_m

    const ai = cache[io + 10], aj = cache[jo + 10], am = alphaM;
    const sdi = cache[io + 9], sdj = cache[jo + 9];

    // Self terms ⟨G,G⟩ = π^{3/2}·√|Σ|.
    const selfI = ai * ai * bni * PI_1_5 * sdi;
    const selfJ = aj * aj * bnj * PI_1_5 * sdj;
    const selfM = am * am * bnm * PI_1_5 * sqrtDetM;

    // Cross terms (amplitude dot × Gaussian overlap).
    const cIJ = crossG(sdi, sdj,
        cache[io + 3] + cache[jo + 3], cache[io + 4] + cache[jo + 4], cache[io + 5] + cache[jo + 5],
        cache[io + 6] + cache[jo + 6], cache[io + 7] + cache[jo + 7], cache[io + 8] + cache[jo + 8],
        mux - mvx, muy - mvy, muz - mvz);
    const cIM = crossG(sdi, sqrtDetM,
        cache[io + 3] + sxx, cache[io + 4] + sxy, cache[io + 5] + sxz,
        cache[io + 6] + syy, cache[io + 7] + syz, cache[io + 8] + szz,
        dix, diy, diz);
    const cJM = crossG(sdj, sqrtDetM,
        cache[jo + 3] + sxx, cache[jo + 4] + sxy, cache[jo + 5] + sxz,
        cache[jo + 6] + syy, cache[jo + 7] + syz, cache[jo + 8] + szz,
        djx, djy, djz);

    const E = selfI + selfJ + selfM +
        2 * ai * aj * bij * cIJ -
        2 * ai * am * bim * cIM -
        2 * aj * am * bjm * cJM;

    // Clamp float-noise negatives (both terms are squared norms ≥ 0; the
    // colour distance uses separately-rounded f32 norms so it too can round
    // slightly negative) but preserve NaN/Inf from degenerate inputs so the
    // caller can fail loud. |Δbase|² = |b_i|² + |b_j|² − 2·b_i·b_j.
    const dc2 = bni + bnj - 2 * bij;
    return (E < 0 ? 0 : E) + COLOR_WEIGHT * (dc2 < 0 ? 0 : dc2);
};

export { buildSplatCache, computeEdgeCost, CACHE_STRIDE, COLOR_WEIGHT };
