/**
 * For every adjacent pair of sorted keys where the high-bit tile index
 * differs, atomicMin's the current position into `tileOffsets[t]` for
 * every tile `t` in `(prevTile, curTile]`. Combined with the sentinel
 * init this gives `tileOffsets[T]` = first index in `sortedKeys` whose
 * tile bits equal T (or the sentinel if T is empty).
 *
 * Dispatched indirectly with workgroup count `ceil(totalPairs / 64)` so
 * that we don't waste invocations on the unused tail of the pairs
 * buffer.
 *
 * @returns WGSL source for the find-boundaries compute shader.
 */
const findBoundariesWgsl = () => /* wgsl */`
#include "uniformsStruct"
#include "constants"

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> totalPairs: array<u32>;
@group(0) @binding(2) var<storage, read> sortedTileKeys: array<u32>;
@group(0) @binding(3) var<storage, read_write> tileOffsets: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wgId: vec3<u32>,
    @builtin(num_workgroups) numWg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    // 2-D dispatch (prepare-indirect splits to stay under the 65535
    // per-axis workgroup-count limit); linearise here.
    let linearWg = wgId.x + wgId.y * numWg.x;
    let i = linearWg * 64u + lid.x;
    let n = totalPairs[0];
    if (i >= n) { return; }

    // Reference uniforms once so the binding isn't dead-code-stripped
    // (keeps the BG format in sync with the shader expectations).
    let _u = uniforms.groupTilesX;

    let curTile = sortedTileKeys[i];
    // Sentinel for "no previous tile" — overflow makes prevTile+1 = 0u
    // so the for loop below cleanly handles the i = 0 case.
    let prevTileBits = select(0xFFFFFFFFu, sortedTileKeys[i - 1u], i > 0u);
    if (curTile == prevTileBits) { return; }
    for (var t: u32 = prevTileBits + 1u; t <= curTile; t = t + 1u) {
        atomicMin(&tileOffsets[t], i);
    }
}
`;

export { findBoundariesWgsl };
