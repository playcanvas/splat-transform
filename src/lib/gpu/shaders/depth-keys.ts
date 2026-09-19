/**
 * Depth sort keys for the scene rasterizer: one thread per gaussian, reading
 * its centre from the resident position layer and writing a u32 key whose
 * unsigned order is the front-to-back order of the chunked path's CPU sort.
 *
 * - pinhole: camera-space depth `forward · (p − eye)`;
 * - equirect: radial distance squared (monotone in the distance);
 * - with MOTION_BLUR, the mean of the shutter-open and shutter-close values,
 *   so a moving gaussian composites at its mid-shutter depth.
 *
 * Gaussians the project shader will invalidate (behind the near plane at
 * either shutter pose, or with a NaN position) get the largest key and sort
 * to the tail; the visible ones are counted into `visibleCount`, so a caller
 * streaming attributes in sorted order can stop at the last visible one. The
 * near tests are the project shader's, so the two agree. A stable radix
 * sort over these keys with identity payload breaks ties by row index,
 * exactly as the CPU sort does.
 *
 * Bindings: 0 uniforms; 1 position (read), 3 floats per gaussian;
 * 2 sortKeys (read_write), one key per gaussian; 3 visibleCount
 * (read_write), a single u32 the caller zeroes before the dispatch.
 *
 * @returns WGSL source.
 */
const depthKeysWgsl = (): string => /* wgsl */`
#include "uniformsStruct"

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> position: array<f32>;
@group(0) @binding(2) var<storage, read_write> sortKeys: array<u32>;
@group(0) @binding(3) var<storage, read_write> visibleCount: atomic<u32>;

var<workgroup> wgVisible: atomic<u32>;

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
    if (lid.x == 0u) { atomicStore(&wgVisible, 0u); }
    workgroupBarrier();

    let i = (wgId.y * numWg.x + wgId.x) * 64u + lid.x;
    if (i < uniforms.numSplats) {
        // Same expressions as the project shader, so the two agree bit for bit.
        let posX = position[i * 3u];
        let posY = position[i * 3u + 1u];
        let posZ = position[i * 3u + 2u];
        let wx = posX - uniforms.eyeX;
        let wy = posY - uniforms.eyeY;
        let wz = posZ - uniforms.eyeZ;
        var visible = false;
        var depth = 0.0;

#ifdef PROJECTION_EQUIRECT
        let r2 = wx * wx + wy * wy + wz * wz;
        visible = r2 > uniforms.near * uniforms.near;
        depth = r2;
    #ifdef MOTION_BLUR
        let wxB = posX - uniforms.eyeBX;
        let wyB = posY - uniforms.eyeBY;
        let wzB = posZ - uniforms.eyeBZ;
        let r2B = wxB * wxB + wyB * wyB + wzB * wzB;
        visible = visible && (r2B > uniforms.near * uniforms.near);
        depth = 0.5 * (r2 + r2B);
    #endif
#else
        let cz = uniforms.forwardX * wx + uniforms.forwardY * wy + uniforms.forwardZ * wz;
        visible = cz > uniforms.near;
        depth = cz;
    #ifdef MOTION_BLUR
        let wxB = posX - uniforms.eyeBX;
        let wyB = posY - uniforms.eyeBY;
        let wzB = posZ - uniforms.eyeBZ;
        let czB = uniforms.forwardBX * wxB + uniforms.forwardBY * wyB + uniforms.forwardBZ * wzB;
        visible = visible && (czB > uniforms.near);
        depth = 0.5 * (cz + czB);
    #endif
#endif

        var key = 0xFFFFFFFFu;
        if (visible) {
            key = sortableKey(depth);
            atomicAdd(&wgVisible, 1u);
        }
        sortKeys[i] = key;
    }

    workgroupBarrier();
    if (lid.x == 0u) {
        let n = atomicLoad(&wgVisible);
        if (n > 0u) { atomicAdd(&visibleCount, n); }
    }
}
`;

export { depthKeysWgsl };
