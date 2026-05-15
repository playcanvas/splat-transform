/**
 * WGSL compute shader for tile-based gaussian splat rasterization.
 *
 * Layout assumptions match preprocess.ts:
 * - `splats`: flat array<f32> with 12 floats per record:
 *     [centerX, centerY, covInvA, covInvB, covInvC, r, g, b, alpha, _, _, _]
 *   Records are pre-sorted by (tile, depth ascending) so each tile's range
 *   can be walked front-to-back.
 * - `tileRanges`: flat array<u32> with 2 u32 per tile: [start, end).
 * - `output`: array<u32> of packed RGBA8 pixels (R in low byte, A in high).
 *
 * Workgroup size = TILE_SIZE × TILE_SIZE (must match preprocess.ts).
 * One workgroup per tile; one thread per pixel.
 */
const TILE_SIZE = 16;

const rasterizeWgsl = () => /* wgsl */`
struct Uniforms {
    width: u32,
    height: u32,
    tilesX: u32,
    tilesY: u32,
    bgR: f32,
    bgG: f32,
    bgB: f32,
    bgA: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> splats: array<f32>;
@group(0) @binding(2) var<storage, read> tileRanges: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(${TILE_SIZE}, ${TILE_SIZE}, 1)
fn main(
    @builtin(workgroup_id) wgId: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tileX = wgId.x;
    let tileY = wgId.y;
    let pixelX = tileX * ${TILE_SIZE}u + lid.x;
    let pixelY = tileY * ${TILE_SIZE}u + lid.y;

    if (pixelX >= uniforms.width || pixelY >= uniforms.height) {
        return;
    }

    let tileIdx = tileY * uniforms.tilesX + tileX;
    let rangeStart = tileRanges[tileIdx * 2u];
    let rangeEnd = tileRanges[tileIdx * 2u + 1u];

    let px = f32(pixelX) + 0.5;
    let py = f32(pixelY) + 0.5;

    var color = vec3<f32>(0.0, 0.0, 0.0);
    var T = 1.0;

    let stride = 12u;

    for (var i = rangeStart; i < rangeEnd; i = i + 1u) {
        if (T < 1e-4) {
            break;
        }
        let base = i * stride;
        let cx = splats[base];
        let cy = splats[base + 1u];
        let a = splats[base + 2u];
        let b = splats[base + 3u];
        let c = splats[base + 4u];

        let dx = px - cx;
        let dy = py - cy;
        let power = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy);
        if (power > 0.0) {
            continue;
        }
        let baseAlpha = splats[base + 8u];
        let alpha = min(0.99, baseAlpha * exp(power));
        if (alpha < (1.0 / 255.0)) {
            continue;
        }
        let weight = T * alpha;
        color.r = color.r + weight * splats[base + 5u];
        color.g = color.g + weight * splats[base + 6u];
        color.b = color.b + weight * splats[base + 7u];
        T = T * (1.0 - alpha);
    }

    // Composite background under the running transmittance.
    color.r = color.r + T * uniforms.bgR;
    color.g = color.g + T * uniforms.bgG;
    color.b = color.b + T * uniforms.bgB;
    let alphaOut = (1.0 - T) + T * uniforms.bgA;

    let r = u32(clamp(color.r, 0.0, 1.0) * 255.0 + 0.5);
    let g = u32(clamp(color.g, 0.0, 1.0) * 255.0 + 0.5);
    let bch = u32(clamp(color.b, 0.0, 1.0) * 255.0 + 0.5);
    let aOut = u32(clamp(alphaOut, 0.0, 1.0) * 255.0 + 0.5);

    let pixelIdx = pixelY * uniforms.width + pixelX;
    output[pixelIdx] = r | (g << 8u) | (bch << 16u) | (aOut << 24u);
}
`;

export { rasterizeWgsl, TILE_SIZE };
