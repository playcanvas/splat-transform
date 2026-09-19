/**
 * Depth sort keys for the resident scene: one thread per gaussian, reading
 * its centre from the column-major base buffer and writing a u32 key whose
 * unsigned order is the front-to-back order of the chunked path's CPU sort.
 *
 * - pinhole: camera-space depth `forward · (p − eye)`;
 * - pinhole with MOTION_BLUR: the mean of the shutter-open and shutter-close
 *   depths, so a moving gaussian composites at its mid-shutter depth;
 * - PROJECTION_EQUIRECT: radial distance squared (monotone in the distance).
 *
 * No near-plane test: every gaussian is keyed, and the project shader's own
 * near test invalidates the ones behind the camera wherever they sort. A
 * stable radix sort over these keys with identity payload breaks ties by
 * row index, exactly as the CPU sort does.
 *
 * Bindings: 0 uniforms; 1 splatsBase (read), the column-major base
 * attributes with x, y, z first; 2 sortKeys (read_write), one key per
 * gaussian.
 *
 * @returns WGSL source.
 */
const depthKeysWgsl = (): string => /* wgsl */`
#include "uniformsStruct"

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> splatsBase: array<f32>;
@group(0) @binding(2) var<storage, read_write> sortKeys: array<u32>;

fn attr(c: u32, s: u32) -> f32 {
    return splatsBase[c * uniforms.numSplats + s];
}

// Map an f32 to a u32 that sorts unsigned in the float's numeric order:
// flip the sign bit of non-negative values, every bit of negative ones.
fn sortableKey(v: f32) -> u32 {
    let bits = bitcast<u32>(v);
    return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wgId: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) numWg: vec3<u32>
) {
    let i = (wgId.y * numWg.x + wgId.x) * 64u + lid.x;
    if (i >= uniforms.numSplats) { return; }

    // Same expressions as the project shader, so the two agree bit for bit.
    let wx = attr(0u, i) - uniforms.eyeX;
    let wy = attr(1u, i) - uniforms.eyeY;
    let wz = attr(2u, i) - uniforms.eyeZ;

#ifdef PROJECTION_EQUIRECT
    let depth = wx * wx + wy * wy + wz * wz;
#else
    let cz = uniforms.forwardX * wx + uniforms.forwardY * wy + uniforms.forwardZ * wz;
    #ifdef MOTION_BLUR
    let wxB = attr(0u, i) - uniforms.eyeBX;
    let wyB = attr(1u, i) - uniforms.eyeBY;
    let wzB = attr(2u, i) - uniforms.eyeBZ;
    let czB = uniforms.forwardBX * wxB + uniforms.forwardBY * wyB + uniforms.forwardBZ * wzB;
    let depth = 0.5 * (cz + czB);
    #else
    let depth = cz;
    #endif
#endif

    sortKeys[i] = sortableKey(depth);
}
`;

export { depthKeysWgsl };
