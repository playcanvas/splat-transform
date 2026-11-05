/**
 * True opacity `A = sigmoid(opacity) + opacity_excess` is stored and drawn up to this value,
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

/**
 * Squeeze of the plateau profile: its unsqueezed extent, `(D - 1) + cutoff / sqrt(D)` σ where it
 * reaches the plain gaussian's floor, over the plain gaussian's `cutoff` σ. The profile is drawn at
 * radius `r * e`, so it ends exactly where a plain gaussian does and its footprint, culls and
 * extent are the plain gaussian's. `e(1) = 1`.
 *
 * @param d - Plateau shape D, at least 1.
 * @param cutoff - The renderer's truncation radius, in σ.
 * @returns The squeeze e(D).
 */
const plateauSqueeze = (d: number, cutoff: number): number => (d - 1 + cutoff / Math.sqrt(d)) / cutoff;

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
 * Drawn mass of a unit-peak splat of plateau shape D (D = 1 for a plain gaussian), as a renderer
 * truncating at `cutoff` σ draws it: the squeezed profile `exp(-D/2 * max(0, r * e - (D - 1))^2)`
 * (see {@link plateauSqueeze}) with the floor `f = exp(-cutoff^2 / 2)` subtracted, so it reaches 0 at
 * `cutoff` σ. Integrated in closed form over the opaque core and the gaussian edge; a renderer that
 * also renormalizes by `1 / (1 - f)` scales every splat alike, so ratios are unaffected.
 *
 * @param d - Plateau shape D, at least 1.
 * @param cutoff - The renderer's truncation radius, in σ.
 * @returns The drawn mass, in σ².
 */
const plateauDrawnMass = (d: number, cutoff: number): number => {
    const f = Math.exp((-cutoff * cutoff) / 2);
    const p = d - 1;
    const e = plateauSqueeze(d, cutoff);
    const edge = cutoff / Math.sqrt(d);
    const core = Math.PI * p * p * (1 - f);
    const tail =
        2 *
        Math.PI *
        ((1 - f) / d -
            (f * edge * edge) / 2 +
            p * (Math.sqrt(Math.PI / (2 * d)) * erf(cutoff / Math.SQRT2) - f * edge));
    return (core + tail) / (e * e);
};

/**
 * Plateau shape that draws true opacity `A` with exactly its mass: the D whose drawn mass is
 * `min(A, OPACITY_MAX)` times a plain gaussian's (see {@link plateauDrawnMass}); 1 for `A ≤ 1`.
 * Solved by bisection: callers tabulate it once rather than per gaussian.
 *
 * @param a - True opacity.
 * @param cutoff - The renderer's truncation radius, in σ.
 * @returns The shape D, 1..~1.76.
 */
const plateauShape = (a: number, cutoff: number): number => {
    const m = Math.min(a, OPACITY_MAX);
    if (!(m > 1)) return 1;
    const plain = plateauDrawnMass(1, cutoff);
    let lo = 1;
    let hi = 3;
    for (let k = 0; k < 48; k++) {
        const mid = 0.5 * (lo + hi);
        if (plateauDrawnMass(mid, cutoff) / plain < m) lo = mid;
        else hi = mid;
    }
    return 0.5 * (lo + hi);
};

export { OPACITY_MAX, encodeOpacityByte, decodeOpacityByte, plateauSqueeze, plateauDrawnMass, plateauShape };
