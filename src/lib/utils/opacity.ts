/**
 * True opacity `A` (a source's opacity slot) is stored and drawn up to this value,
 * `(255 / 204)^2 = 1.25^2`, the top of the 8-bit encoding. Merges can owe far more (the
 * moment-matched mass double-counts children that overlap along a view), but on five scenes at deep
 * LOD levels the render improves as `A` is capped, down to ~1.55, and the decimator's own cascade is
 * unaffected by capping it there (scenes/alpha-study/RESULTS.md).
 */
const OPACITY_MAX = 1.5625;

/**
 * Encode true opacity as one byte, `b = round(204 * sqrt(A))`: the square root gives fine steps at
 * the low opacities that overlapping translucent layers are built from, and byte 204 is exactly
 * `A = 1`, where trained scenes keep their saturated gaussians. Values above OPACITY_MAX are clamped.
 *
 * @param a - True opacity.
 * @returns The byte, 0..255.
 */
const encodeOpacityByte = (a: number): number => {
    if (!(a > 0)) return 0;
    return Math.min(255, Math.round(Math.sqrt(a) * 204));
};

/**
 * Decode a byte of {@link encodeOpacityByte}: `A = (b / 204)^2`, which is `(1.25 * v)^2` with `v`
 * the byte as a unit value `b / 255`.
 *
 * @param b - The byte, 0..255.
 * @returns True opacity, 0..OPACITY_MAX.
 */
const decodeOpacityByte = (b: number): number => {
    const s = b / 204;
    return s * s;
};

// Abramowitz and Stegun 7.1.26, within 1.5e-7.
const erf = (x: number): number => {
    const t = 1 / (1 + 0.3275911 * Math.abs(x));
    const y =
        1 -
        ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
            t *
            Math.exp(-x * x);
    return x >= 0 ? y : -y;
};

/**
 * Constants of {@link plateauCore} for a renderer truncating at `cutoff` σ: `u0 = sqrt(r0² + q t) - r0`.
 * A unit-peak splat drawn as `exp(-ρ²/2) - f`, `f = exp(-cutoff² / 2)`, has its radius remapped past an
 * opaque core of radius `u0` (a fraction of the cutoff): `ρ → max(0, ρ - u0 cutoff) / (1 - u0)`. Its
 * drawn mass, over a plain gaussian's, is `1 + (a u0² + b u0) / G1`, with `G0` and `G1` the integrals of
 * `g(x) = exp(-cutoff² x² / 2) - f` and `x g(x)` over `x` in [0, 1], `a = g(0)/2 - G0 + G1` and
 * `b = G0 - 2 G1`; the root drawing `1 + t` times the mass is `r0 = b / 2a`, `q = G1 / a`. A renderer that
 * also renormalizes by `1 / (1 - f)` scales every splat alike, so the constants are unaffected.
 *
 * @param cutoff - The renderer's truncation radius, in σ.
 * @returns The constants r0 and q.
 */
const plateauCoreConstants = (cutoff: number): { r0: number; q: number } => {
    const f = Math.exp((-cutoff * cutoff) / 2);
    const g0 = (Math.sqrt(Math.PI / 2) / cutoff) * erf(cutoff / Math.SQRT2) - f;
    const g1 = (1 - f) / (cutoff * cutoff) - f / 2;
    const a = (1 - f) / 2 - g0 + g1;
    const b = g0 - 2 * g1;
    return { r0: b / (2 * a), q: g1 / a };
};

/**
 * Core radius that draws true opacity `A` with exactly its mass: a unit-peak splat whose inner `u0` of
 * the cutoff radius is opaque and whose plain gaussian falloff is stretched over the rest (see
 * {@link plateauCoreConstants}). Its footprint, culls and extent are the plain gaussian's; 0 for `A ≤ 1`.
 *
 * @param a - True opacity, capped at OPACITY_MAX.
 * @param cutoff - The renderer's truncation radius, in σ.
 * @returns The core radius u0, as a fraction of the cutoff (0..0.24 at 3σ).
 */
const plateauCore = (a: number, cutoff: number): number => {
    const { r0, q } = plateauCoreConstants(cutoff);
    const t = Math.min(Math.max(a - 1, 0), OPACITY_MAX - 1);
    return Math.sqrt(r0 * r0 + q * t) - r0;
};

// The largest f32 below 1, which opacity above 1 takes in the logit: logit ~16.64.
const BELOW_ONE = 1 - 2 ** -24;

/**
 * The PLY `opacity` logit for true opacity `A`, as standard (logit) PLY output and
 * DataTables store it: `logit(A)`, with `A` exactly 1 as `+Infinity` and exactly 0 (or
 * below) as `-Infinity`, so infinite logits round-trip (in f32 every logit above ~17.3
 * reads as `A = 1`, and every one below ~-104 as `A = 0`), and `A` above 1 as the
 * logit of the largest f32 below 1, the largest finite logit an f32 opacity reads
 * as. NaN stays NaN.
 *
 * @param a - True opacity.
 * @returns The logit.
 */
const opacityToLogit = (a: number): number => {
    if (a === 1) return Infinity;
    if (a <= 0) return -Infinity;
    const p = a > 1 ? BELOW_ONE : a;
    return Math.log(p / (1 - p));
};

export { OPACITY_MAX, encodeOpacityByte, decodeOpacityByte, opacityToLogit, plateauCoreConstants, plateauCore };
