/**
 * Tile-bin emit-pairs shader. For each projected splat, emits
 * `coverage[i]` (tile, splat) pairs into two parallel buffers,
 * starting at `emitOffset[i]`:
 *
 *   tileKeys [emitOffset[i] + j] = tileIdx
 *   splatValues[emitOffset[i] + j] = splatIdx (= i)
 *
 * The orchestrator sizes maxCoveragePerSplat to cover a sub-frame's
 * entire tile area, so a splat's `coverage[i]` always equals its full
 * bbox-in-group tile count — no truncation, no seams. The walk emits
 * row-major over the bbox-in-group. A subsequent key+value radix sort
 * groups pairs by tile; within each tile, the splatIdx-as-value sort
 * preserves the chunk's depth order (splatIdx is monotonic in depth
 * from the CPU pre-sort).
 *
 * Projection-mode variation: `PROJECTION_EQUIRECT` swaps in the
 * tile-walk-equirect chunk, which walks the un-clamped X range and
 * applies a modular wrap so a splat near the ±π longitude seam emits
 * tile keys on both sides of the image. Without the flag the
 * tile-walk-pinhole chunk walks the clamped bbox directly.
 *
 * @returns WGSL source for the emit-pairs compute shader.
 */
const tileBinEmitPairsWgsl = () => /* wgsl */`
#include "uniformsStruct"
#include "constants"

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> projected: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> emitOffset: array<u32>;
@group(0) @binding(3) var<storage, read> coverage: array<u32>;
@group(0) @binding(4) var<storage, read_write> tileKeys: array<u32>;
@group(0) @binding(5) var<storage, read_write> splatValues: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= uniforms.chunkSize) { return; }
    let cap = coverage[i];
    if (cap == 0u) { return; }
    let v0 = projected[i * 3u + 0u];
    let radius = v0.z;
    if (radius <= 0.0) { return; }
    let sX = v0.x;
    let sY = v0.y;
    let tsz: f32 = f32(TILE_SIZE);
    // Group-local tile indices (see project shader for rationale).
    let gox = f32(uniforms.groupPixelOriginX);
    let goy = f32(uniforms.groupPixelOriginY);
#ifdef PROJECTION_EQUIRECT
    #include "tileWalkEquirect"
#else
    #include "tileWalkPinhole"
#endif
}
`;

export { tileBinEmitPairsWgsl };
