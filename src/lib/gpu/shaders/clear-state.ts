/**
 * Resets the running state of every pixel in the active group to "nothing
 * painted yet": colour 0 and transmittance 1. Runs on the GPU at `beginGroup`
 * so a group of any size costs no upload — a 4096-square group is 268 MB of
 * state, which would otherwise be written from the CPU for every image.
 *
 * Each thread clears four pixels so a maximal group stays within the dispatch
 * limit of 65535 workgroups.
 *
 * @returns WGSL source for the clear-state compute shader.
 */
const clearStateWgsl = () => /* wgsl */`
#include "uniformsStruct"
#include "constants"

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> runningState: array<vec4<f32>>;

const PIXELS_PER_THREAD: u32 = 4u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let numPixels = uniforms.groupTilesX * uniforms.groupTilesY * TILE_SIZE * TILE_SIZE;
    let base = gid.x * PIXELS_PER_THREAD;
    for (var k: u32 = 0u; k < PIXELS_PER_THREAD; k = k + 1u) {
        let i = base + k;
        if (i < numPixels) {
            runningState[i] = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
    }
}
`;

export { clearStateWgsl };
