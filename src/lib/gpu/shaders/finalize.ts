/**
 * Packs the running-state (linear color + residual transmittance) into a
 * single RGBA8-packed u32 per group pixel. Composites the user-supplied
 * background under the residual transmittance so the final image carries
 * the chosen `bgR/bgG/bgB/bgA` everywhere the splats didn't fully cover.
 *
 * @returns WGSL source for the finalize compute shader.
 */
const finalizeWgsl = () => /* wgsl */`
#include "uniformsStruct"
#include "constants"

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> runningState: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
fn main(
    @builtin(workgroup_id) wgId: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    if (wgId.x >= uniforms.groupTilesX || wgId.y >= uniforms.groupTilesY) { return; }

    let localPixelX = wgId.x * TILE_SIZE + lid.x;
    let localPixelY = wgId.y * TILE_SIZE + lid.y;
    let groupPixelW = uniforms.groupTilesX * TILE_SIZE;

    let pixelIdx = localPixelY * groupPixelW + localPixelX;
    let state = runningState[pixelIdx];

    let color = state.rgb + state.a * vec3<f32>(uniforms.bgR, uniforms.bgG, uniforms.bgB);
    let alphaOut = (1.0 - state.a) + state.a * uniforms.bgA;

    let r = u32(clamp(color.r, 0.0, 1.0) * 255.0 + 0.5);
    let g = u32(clamp(color.g, 0.0, 1.0) * 255.0 + 0.5);
    let bch = u32(clamp(color.b, 0.0, 1.0) * 255.0 + 0.5);
    let aOut = u32(clamp(alphaOut, 0.0, 1.0) * 255.0 + 0.5);

    output[pixelIdx] = r | (g << 8u) | (bch << 16u) | (aOut << 24u);
}
`;

export { finalizeWgsl };
