/**
 * Integrates aperture and shutter samples in premultiplied linear light.
 * Each view reconstructs the splats' gamma-space colours, then converts
 * to linear before accumulation. Only the final mean is encoded and
 * clamped to RGBA8, preserving highlights above one between samples.
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

fn decodeSRGB(color: vec3<f32>) -> vec3<f32> {
    return select(pow((color + 0.055) / 1.055, vec3<f32>(2.4)), color / 12.92, color <= vec3<f32>(0.04045));
}

fn encodeSRGB(color: vec3<f32>) -> vec3<f32> {
    return select(1.055 * pow(color, vec3<f32>(1.0 / 2.4)) - 0.055, 12.92 * color, color <= vec3<f32>(0.0031308));
}

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

    let color = state.rgb + state.a * uniforms.bgA * vec3<f32>(uniforms.bgR, uniforms.bgG, uniforms.bgB);
    let alphaOut = (1.0 - state.a) + state.a * uniforms.bgA;
    let straightColor = color / max(alphaOut, 1e-8);
    let slice = vec4<f32>(decodeSRGB(max(straightColor, vec3<f32>(0.0))) * alphaOut, alphaOut);

    let sum = select(accum[pixelIdx], vec4<f32>(0.0), uniforms.sliceIndex == 0u) + slice;
    accum[pixelIdx] = sum;
    if (uniforms.sliceIndex + 1u == uniforms.sliceCount) {
        let mean = sum / f32(uniforms.sliceCount);
        output[pixelIdx] = packRGBA8(vec4<f32>(encodeSRGB(mean.rgb / max(mean.a, 1e-8)), mean.a));
    }
}
`;

export { accumulateWgsl };
