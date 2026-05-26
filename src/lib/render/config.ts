/**
 * Centralised render-time tunables. Constants that today live as magic
 * numbers in the project/rasterize WGSL shaders, in `gaussian-aabb.ts`'s
 * extent computation, and in the orchestrator's far-plane logic.
 *
 * Two goals:
 *
 *   1. Single source of truth — one place to discover every knob, so the
 *      "reference 3DGS standard" values are documented together rather
 *      than scattered across files.
 *
 *   2. A natural switching point for a future "reference" vs "fast"
 *      profile — for now everything is the reference-3DGS value.
 *
 * Most of these match the values used by the original INRIA 3DGS CUDA
 * rasterizer. Touching any of them is an "exceed reference" decision and
 * needs to be reflected in regenerated golden fixtures.
 */

/**
 * Half-pixel² covariance dilation added to the 2D projected covariance's
 * diagonal before inversion. Reduces aliasing at the cost of inflating
 * every splat's screen footprint by ≤1 pixel. Matches the INRIA reference.
 */
export const AA_DILATION_COV = 0.3;

/**
 * Maximum tan(half-FOV) factor that the EWA Jacobian's `(cx/cz, cy/cz)`
 * is clamped to before forming the projection. Splats outside this cone
 * use a distorted Jacobian. Matches the INRIA reference (`1.3 × half-FOV`).
 */
export const JACOBIAN_LIMIT_FACTOR = 1.3;

/**
 * Per-splat alpha cap. Prevents perfectly opaque splats so the
 * transmittance chain can't collapse to exactly zero in one step. INRIA
 * reference.
 */
export const OPACITY_CAP = 0.99;

/**
 * Per-splat alpha cutoff. Contributions below this are dropped — saves
 * work for splats whose evaluated 2D Gaussian falls below quantization
 * threshold. INRIA reference.
 */
export const MIN_ALPHA = 1.0 / 255.0;

/**
 * Per-pixel transmittance early-out. Once `T` falls below this the pixel
 * stops accumulating; remaining ~0.01% of mass is discarded. INRIA
 * reference.
 */
export const MIN_TRANSMITTANCE = 1e-4;

/**
 * Floor on `½·trace² − det` before sqrt when deriving the projected
 * covariance's larger eigenvalue. Bounds the radius from below by
 * `ceil(3·sqrt(½·trace + sqrt(floor)))` ≈ 1 pixel. INRIA reference.
 */
export const DISCRIMINANT_FLOOR = 0.1;

/**
 * Number of standard deviations at which the rendered Gaussian's tail
 * is truncated. Used for both:
 *   - the AABB half-extents in `computeGaussianExtents` (BVH input)
 *   - the screen-space radius in the project shader (rasterize bbox)
 *
 * Keeping them in sync ensures a splat included by the BVH cull is
 * actually rasterized; lowering this here without also lowering it in
 * the project shader's `radius = ceil(SIGMA_CUTOFF · sqrt(λmax))` would
 * cause silently-clipped tails.
 */
export const SIGMA_CUTOFF = 3.0;

/**
 * Value of the unit gaussian at the truncation radius. Subtracted from
 * `exp(power)` in the rasterizer so each splat's alpha reaches exactly
 * 0 at `SIGMA_CUTOFF · σ` instead of clipping at ≈ 1.1% (which would
 * leave a faint ring at the splat boundary). Matches the PlayCanvas
 * engine's edge compensation.
 */
export const GAUSSIAN_FLOOR = Math.exp(-0.5 * SIGMA_CUTOFF * SIGMA_CUTOFF);

/**
 * Floor on the far-plane distance, expressed as a multiple of the near
 * plane. If every scene-AABB corner sits behind the camera the
 * computed far would otherwise be ≤ near; this prevents a degenerate
 * frustum. Magic number; not from any reference.
 */
export const FAR_PLANE_NEAR_FACTOR = 100;

/**
 * Tile size in pixels for the rasterize workgroup. Must match the
 * WGSL shader's hard-coded `workgroup_size(16, 16, 1)`. Changing here
 * also requires updating the shader templates.
 */
export const TILE_SIZE = 16;

/**
 * Equirect projection pole-exclusion factor. The equirect Jacobian's
 * latitude derivative diverges as the splat approaches the zenith /
 * nadir (rxz → 0). We clamp `rxz` from below at `POLE_EPS · r` so the
 * Jacobian (and hence the projected covariance, screen-space radius,
 * and per-tile coverage) stays bounded for splats arbitrarily close to
 * a pole. Splats *exactly* on the pole get a finite (but narrow) screen
 * footprint at a longitude determined by tiny numerical noise — in
 * practice this band is rarely visible because polar splats tend to be
 * far from the camera. Lowering the floor sharpens detail near the
 * poles at the cost of more aggressive footprint stretching for nearby
 * splats; 0.005 (~0.29°) is a reasonable balance.
 */
export const POLE_EPS = 0.005;


/**
 * Total GPU memory budget for ALL pair-sized buffers combined: the
 * tile-key and splat-value buffers the rasterizer owns (2× pairsCap × 4 B)
 * plus the four internal ping-pong buffers `ComputeRadixSort` allocates
 * (`_keys0`, `_keys1`, `_values0`, `_values1` — another 4× pairsCap × 4 B).
 * That's 6 × pairsCap × 4 B = `PAIR_BUFFER_TOTAL_BYTES_PER_ELEMENT × pairsCap`.
 *
 * `chunkCap` is sized to keep this total under budget. The radix-sort
 * scratch is the dominant share; budgeting only the local buffers (as
 * earlier revisions did) under-counts actual GPU memory by ~3×.
 */
export const PAIR_BUFFER_BUDGET_BYTES = 768 * 1024 * 1024;

/**
 * Total bytes per pair *element* across all six pair-sized buffers
 * (4 B × 6 buffers). Used by the orchestrator's chunkCap calculation
 * so the math reflects actual GPU memory consumed by the sort
 * pipeline.
 */
export const PAIR_BUFFER_TOTAL_BYTES_PER_ELEMENT = 4 * 6;

/**
 * Screen-radius fade thresholds, expressed as fractions of image
 * height. Defends against outlier splats with large world-space scale
 * or near-camera placement that would otherwise project to a screen-
 * spanning footprint.
 *
 * A hard clamp produces a visible "pop" as the camera approaches a
 * splat that grows past the cap (the splat suddenly stops getting any
 * bigger). Instead we *linearly fade out* the splat's alpha between
 * `RADIUS_FADE_START_FRAC × imageHeight` (alpha × 1) and
 * `RADIUS_FADE_END_FRAC × imageHeight` (alpha × 0). Beyond the END
 * threshold the splat is discarded entirely.
 *
 * Image-height-relative so the SAME world-space splats fade at every
 * render resolution — a splat that doesn't fade at 1080p won't get
 * dropped at 8K just because its pixel radius is 4× bigger. The values
 * are calibrated so that the original 1080p thresholds (1024 px /
 * 2048 px) reproduce, while 8K renders fade only the splats that
 * would also fade at 1080p.
 *
 * Inspired by PlayCanvas engine's `min(1024.0, viewport)` axis cap
 * (see `gsplatCorner.js`), but with the cap softened into a fade.
 */
export const RADIUS_FADE_START_FRAC = 1024 / 1080;
export const RADIUS_FADE_END_FRAC = 2048 / 1080;
