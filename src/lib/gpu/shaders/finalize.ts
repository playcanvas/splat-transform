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
#include "packRGBA8"

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

    output[pixelIdx] = packRGBA8(vec4<f32>(color, alphaOut));
}
`;

export { finalizeWgsl };
