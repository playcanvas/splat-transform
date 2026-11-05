/**
 * True opacity `A = sigmoid(opacity) + opacity_excess` is stored and drawn up to this value. Merges
 * can owe far more (the moment-matched mass double-counts children that overlap along a view), but
 * on five scenes at deep LOD levels the render improves as `A` is capped, down to ~1.6, and the
 * decimator's own cascade is unaffected by capping it there (scenes/alpha-study/RESULTS.md).
 */
const OPACITY_MAX = 1.6;

/** Byte of the 8-bit opacity encoding at `A = 1`; bytes above it carry `A` in (1, OPACITY_MAX]. */
const OPACITY_BYTE_ONE = 224;

/**
 * Encode true opacity as one byte: `b ≤ 224` holds `A ≤ 1` as `sqrt(A)` (fine steps at the low
 * opacities that overlapping translucent layers are built from; 0 and 1 exact), `b > 224` holds
 * `A` in (1, OPACITY_MAX] linearly. Values above OPACITY_MAX are clamped.
 *
 * @param a - True opacity.
 * @returns The byte, 0..255.
 */
const encodeOpacityByte = (a: number): number => {
    if (!(a > 0)) return 0;
    if (a <= 1) return Math.round(Math.sqrt(a) * OPACITY_BYTE_ONE);
    const over = (Math.min(a, OPACITY_MAX) - 1) / (OPACITY_MAX - 1);
    return OPACITY_BYTE_ONE + Math.round(over * (255 - OPACITY_BYTE_ONE));
};

/**
 * Decode a byte of {@link encodeOpacityByte}.
 *
 * @param b - The byte, 0..255.
 * @returns True opacity, 0..OPACITY_MAX.
 */
const decodeOpacityByte = (b: number): number => {
    if (b <= OPACITY_BYTE_ONE) {
        const s = b / OPACITY_BYTE_ONE;
        return s * s;
    }
    return 1 + ((OPACITY_MAX - 1) * (b - OPACITY_BYTE_ONE)) / (255 - OPACITY_BYTE_ONE);
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
 * Plateau shape that draws true opacity `A` with exactly its mass, `D = I⁻¹(min(A, OPACITY_MAX))`;
 * 1 for `A ≤ 1`. Solved by bisection: callers tabulate it once rather than per gaussian.
 *
 * @param a - True opacity.
 * @returns The shape D, 1..I⁻¹(OPACITY_MAX) (≈ 1.79).
 */
const plateauShape = (a: number): number => {
    const m = Math.min(a, OPACITY_MAX);
    if (!(m > 1)) return 1;
    let lo = 1;
    let hi = 1 + Math.sqrt(2 * m);
    for (let k = 0; k < 48; k++) {
        const mid = 0.5 * (lo + hi);
        if (plateauMass(mid) < m) lo = mid;
        else hi = mid;
    }
    return 0.5 * (lo + hi);
};

export { OPACITY_MAX, OPACITY_BYTE_ONE, encodeOpacityByte, decodeOpacityByte, plateauMass, plateauShape };
