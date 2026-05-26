import {
    AA_DILATION_COV,
    DISCRIMINANT_FLOOR,
    GAUSSIAN_FLOOR,
    JACOBIAN_LIMIT_FACTOR,
    MIN_ALPHA,
    MIN_TRANSMITTANCE,
    OPACITY_CAP,
    POLE_EPS,
    RADIUS_FADE_END_FRAC,
    RADIUS_FADE_START_FRAC,
    SIGMA_CUTOFF,
    TILE_SIZE
} from '../../../render/config';

/**
 * Format a JS number as a WGSL `f32` literal. Adds an explicit `.0` so
 * integer-valued constants like `3` aren't parsed as `AbstractInt` —
 * keeps shaders readable when the constant flips to a fractional value.
 *
 * @param n - Numeric value to format.
 * @returns WGSL literal string with explicit `.0` for integer values.
 */
const wgslF32 = (n: number): string => {
    const s = n.toString();
    return s.includes('.') || s.includes('e') || s.includes('E') ? s : `${s}.0`;
};

/**
 * Shared render-time tunables declared as WGSL `const` so every shader can
 * reference them as plain identifiers. Bound to the JS-side values in
 * [render/config.ts](../../render/config.ts) — the single source of truth.
 *
 * The PlayCanvas WGSL preprocessor's `cdefines` mechanism only registers
 * symbols for `#ifdef` checks; it does not substitute bare identifiers
 * with their values. WGSL `const` declarations cover that gap and can
 * additionally be used inside `@workgroup_size(...)` and other const-
 * expression contexts.
 *
 * Included via `#include "constants"` (see `sharedCincludes` in
 * `gpu-splat-rasterizer.ts`).
 */
const constantsChunk = /* wgsl */`
const TILE_SIZE: u32 = ${TILE_SIZE}u;
const SIGMA_CUTOFF: f32 = ${wgslF32(SIGMA_CUTOFF)};
const GAUSSIAN_FLOOR: f32 = ${wgslF32(GAUSSIAN_FLOOR)};
const AA_DILATION_COV: f32 = ${wgslF32(AA_DILATION_COV)};
const DISCRIMINANT_FLOOR: f32 = ${wgslF32(DISCRIMINANT_FLOOR)};
const JACOBIAN_LIMIT_FACTOR: f32 = ${wgslF32(JACOBIAN_LIMIT_FACTOR)};
const MIN_ALPHA: f32 = ${wgslF32(MIN_ALPHA)};
const MIN_TRANSMITTANCE: f32 = ${wgslF32(MIN_TRANSMITTANCE)};
const OPACITY_CAP: f32 = ${wgslF32(OPACITY_CAP)};
const RADIUS_FADE_START_FRAC: f32 = ${wgslF32(RADIUS_FADE_START_FRAC)};
const RADIUS_FADE_END_FRAC: f32 = ${wgslF32(RADIUS_FADE_END_FRAC)};
const POLE_EPS: f32 = ${wgslF32(POLE_EPS)};
`;

export { constantsChunk, wgslF32 };
