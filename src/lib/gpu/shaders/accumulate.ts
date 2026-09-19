/**
 * Motion-blur slice accumulation. Runs in place of finalize once per
 * shutter slice: composites the group's running state over the background
 * exactly as finalize does, clamps it to [0, 1] like the packed slice
 * finalize would have produced, and adds it into an f32 accumulator
 * (`uniforms.sliceIndex` 0 initialises it). On the last slice
 * (`sliceIndex + 1 == sliceCount`) the mean is packed to RGBA8, so a frame
 * costs one readback however many slices it has and no per-slice
 * quantization.
 *
 * @returns WGSL source for the accumulate compute shader.
 */
const accumulateWgsl = () => /* wgsl */`
#include "uniformsStruct"
#include "constants"
#include "packRGBA8"

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> runningState: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> accum: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;

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
    let slice = clamp(vec4<f32>(color, alphaOut), vec4<f32>(0.0), vec4<f32>(1.0));

    let sum = select(accum[pixelIdx], vec4<f32>(0.0), uniforms.sliceIndex == 0u) + slice;
    accum[pixelIdx] = sum;
    if (uniforms.sliceIndex + 1u == uniforms.sliceCount) {
        output[pixelIdx] = packRGBA8(sum / f32(uniforms.sliceCount));
    }
}
`;

export { accumulateWgsl };
