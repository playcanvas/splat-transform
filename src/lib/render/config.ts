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
 * Max per-splat tile coverage budgeted for binning. Sets the GPU's
 * tile-data buffer capacity at `chunkCap × MAX_COVERAGE_PER_SPLAT` u32s.
 * Pathological splats with screen radius > ~`√n · TILE_SIZE` tiles get
 * truncated by the CPU binner (counted via the logger warning).
 *
 * 64 covers any reasonable splat (a 16×16-tile coverage = 256-px screen
 * radius, well above the typical 3σ projection of a real splat).
 */
export const MAX_COVERAGE_PER_SPLAT = 64;

/**
 * Screen-radius fade thresholds, in pixels. Defends against outlier
 * splats with large world-space scale or near-camera placement that
 * would otherwise project to a screen-spanning footprint.
 *
 * A hard clamp produces a visible "pop" as the camera approaches a
 * splat that grows past the cap (the splat suddenly stops getting any
 * bigger). Instead we *linearly fade out* the splat's alpha between
 * `RADIUS_FADE_START_PX` (alpha × 1) and `RADIUS_FADE_END_PX` (alpha
 * × 0). Beyond `RADIUS_FADE_END_PX` the splat is discarded entirely.
 *
 * The bounding bbox is clamped at `RADIUS_FADE_END_PX` so the binner
 * doesn't allocate tile coverage for splats with effectively zero
 * contribution.
 *
 * Inspired by PlayCanvas engine's `min(1024.0, viewport)` axis cap
 * (see `gsplatCorner.js`), but with the cap softened into a fade.
 */
export const RADIUS_FADE_START_PX = 1024;
export const RADIUS_FADE_END_PX = 2048;
