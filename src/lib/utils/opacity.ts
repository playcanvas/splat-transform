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
 * Mass (integrated opacity, in units of a unit-peak gaussian's) of the plateau profile
 * `exp(-D/2 * max(0, r - (D - 1))^2)` that draws over-unity opacity: an opaque core of radius
 * `(D - 1)σ` joined smoothly to a gaussian edge. `I(1) = 1`.
 *
 * @param d - Plateau shape D, at least 1.
 * @returns The mass I(D).
 */
const plateauMass = (d: number): number => {
    const p = d - 1;
    return 0.5 * p * p + 1 / d + 0.5 * p * Math.sqrt((2 * Math.PI) / d);
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

/**
 * Plateau shape that draws true opacity `A` with exactly its mass when squeezed (see
 * {@link plateauSqueeze}): `I(D) / e(D)^2 = min(A, OPACITY_MAX)`; 1 for `A ≤ 1`. Solved by
 * bisection: callers tabulate it once rather than per gaussian.
 *
 * @param a - True opacity.
 * @param cutoff - The renderer's truncation radius, in σ.
 * @returns The shape D, 1..~1.84.
 */
const plateauShape = (a: number, cutoff: number): number => {
    const m = Math.min(a, OPACITY_MAX);
    if (!(m > 1)) return 1;
    let lo = 1;
    let hi = 3;
    for (let k = 0; k < 48; k++) {
        const mid = 0.5 * (lo + hi);
        if (plateauMass(mid) / plateauSqueeze(mid, cutoff) ** 2 < m) lo = mid;
        else hi = mid;
    }
    return 0.5 * (lo + hi);
};

export { OPACITY_MAX, encodeOpacityByte, decodeOpacityByte, plateauMass, plateauSqueeze, plateauShape };
