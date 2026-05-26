/**
 * Binned rasterize shader. Each workgroup handles one tile and only walks
 * the splats that have been pre-binned into it (tile-bin pre-pass on CPU
 * or GPU). Replaces the "walk all splats per pixel" loop in a non-binned
 * rasterizer with "walk this tile's slice", which is the asymptotic
 * fix for performance at high splat counts.
 *
 * The slice is stored in two buffers:
 *   - `tileOffsets[T + 1]` — exclusive prefix sum: tile T's slice is
 *     `tileData[tileOffsets[T] .. tileOffsets[T + 1])`.
 *   - `tileData[]` — splat indices, grouped by tile, depth-sorted within
 *     each tile (the orchestrator's CPU pre-sort + stable per-splat
 *     binning produces this layout for free).
 *
 * Projection-mode variation: `PROJECTION_EQUIRECT` wraps the per-pixel
 * `dx = px - splat.x` into `[-W/2, W/2]` so a tile on the opposite side
 * of the ±π longitude seam evaluates against the splat's nearer copy.
 * Without the flag the raw delta is used.
 *
 * @returns WGSL source for the binned-rasterize compute shader.
 */
const rasterizeBinnedWgsl = () => /* wgsl */`
#include "uniformsStruct"
#include "constants"

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> projected: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> runningState: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> tileOffsets: array<u32>;
@group(0) @binding(4) var<storage, read> sortedSplatIndices: array<u32>;

@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
fn main(
    @builtin(workgroup_id) wgId: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    if (wgId.x >= uniforms.groupTilesX || wgId.y >= uniforms.groupTilesY) { return; }

    let tileIdx = wgId.y * uniforms.groupTilesX + wgId.x;
    let sliceStart = tileOffsets[tileIdx];
    let sliceEnd = tileOffsets[tileIdx + 1u];

    let localPixelX = wgId.x * TILE_SIZE + lid.x;
    let localPixelY = wgId.y * TILE_SIZE + lid.y;
    let groupPixelW = uniforms.groupTilesX * TILE_SIZE;

    let imagePixelX = uniforms.groupPixelOriginX + localPixelX;
    let imagePixelY = uniforms.groupPixelOriginY + localPixelY;
    if (imagePixelX >= uniforms.imageWidth || imagePixelY >= uniforms.imageHeight) { return; }

    let pixelIdx = localPixelY * groupPixelW + localPixelX;
    var state = runningState[pixelIdx];
    var color = state.rgb;
    var T = state.a;

    if (T < MIN_TRANSMITTANCE) { return; }

    let px = f32(imagePixelX) + 0.5;
    let py = f32(imagePixelY) + 0.5;

#ifdef PROJECTION_EQUIRECT
    let imgWf2 = f32(uniforms.imageWidth);
    let halfImgW = imgWf2 * 0.5;
#endif
    for (var i: u32 = sliceStart; i < sliceEnd; i = i + 1u) {
        if (T < MIN_TRANSMITTANCE) { break; }
        let splatIdx = sortedSplatIndices[i];
        let v0 = projected[splatIdx * 3u + 0u];
#ifdef PROJECTION_EQUIRECT
        // Equirect: a splat near the ±π longitude seam is tile-binned on
        // both sides of the image. Wrap dx into [-W/2, W/2] so a tile on
        // the opposite side of the seam pulls the splat's footprint from
        // the correct (nearer) copy.
        var dx = px - v0.x;
        if (dx > halfImgW) { dx = dx - imgWf2; }
        else if (dx < -halfImgW) { dx = dx + imgWf2; }
#else
        let dx = px - v0.x;
#endif
        let dy = py - v0.y;
        let r = v0.z;
        if (r <= 0.0 || abs(dx) > r || abs(dy) > r) { continue; }
        let v1 = projected[splatIdx * 3u + 1u];
        let power = -0.5 * (v1.x * dx * dx + 2.0 * v1.y * dx * dy + v1.z * dy * dy);
        if (power > 0.0) { continue; }
        // Subtract GAUSSIAN_FLOOR so each splat's alpha reaches 0 exactly
        // at the 3σ truncation radius instead of clipping at ~1.1% —
        // eliminates faint ring artifacts at splat edges. Matches the
        // PlayCanvas engine.
        let alpha = min(OPACITY_CAP, v1.w * max(0.0, exp(power) - GAUSSIAN_FLOOR));
        if (alpha < MIN_ALPHA) { continue; }
        let weight = T * alpha;
        let v2 = projected[splatIdx * 3u + 2u];
        color = color + weight * v2.rgb;
        T = T * (1.0 - alpha);
    }

    runningState[pixelIdx] = vec4<f32>(color, T);
}
`;

export { rasterizeBinnedWgsl };
