/**
 * Initialises `tileOffsets[0 .. numTiles]` to the sentinel value
 * `totalPairs[0]` (= past-the-end). `findBoundaries` then atomicMin's
 * the actual first-pair-index for every non-empty tile; tiles with no
 * pairs keep the sentinel, which collapses to a zero-length slice when
 * the rasterize-binned shader reads `tileOffsets[T] .. tileOffsets[T+1]`.
 *
 * @returns WGSL source for the init-tile-offsets compute shader.
 */
const initTileOffsetsWgsl = () => /* wgsl */`
#include "uniformsStruct"
#include "constants"

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> totalPairs: array<u32>;
@group(0) @binding(2) var<storage, read_write> tileOffsets: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let numTiles = uniforms.groupTilesX * uniforms.groupTilesY;
    if (i > numTiles) { return; }
    tileOffsets[i] = totalPairs[0];
}
`;

export { initTileOffsetsWgsl };
