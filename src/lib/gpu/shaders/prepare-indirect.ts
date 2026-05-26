/**
 * Prepares indirect-dispatch arguments by reading `totalPairs[0]` and
 * writing workgroup counts into two slots of the device's
 * `indirectDispatchBuffer` (each slot is 3 × u32 = `(x, y, z)`):
 *
 *   - `sortSlot`: workgroup count for `ComputeRadixSort.sortIndirect`,
 *     computed as `ceil(totalPairs / 2048)` (matches the radix sort's
 *     16×16 thread × 8 elements / thread = 2048 elements/workgroup).
 *   - `boundariesSlot`: workgroup count for `findBoundaries`, computed
 *     as `ceil(totalPairs / 64)`.
 *
 * Slot byte offsets are passed in via two u32 uniforms in a small ad-hoc
 * uniform block (NOT the shared `Uniforms` struct, because the slot
 * indices vary per chunk while the shared uniforms are set per group).
 *
 * @returns WGSL source for the prepare-indirect compute shader.
 */
const prepareIndirectWgsl = () => /* wgsl */`
struct PrepareUniforms {
    sortSlotBase: u32,
    boundariesSlotBase: u32,
}

@group(0) @binding(0) var<uniform> uniforms: PrepareUniforms;
@group(0) @binding(1) var<storage, read> totalPairs: array<u32>;
@group(0) @binding(2) var<storage, read_write> indirectBuffer: array<u32>;

const SORT_ELEMENTS_PER_WG: u32 = 2048u;
const BOUNDARIES_THREADS_PER_WG: u32 = 64u;
// WebGPU spec minimum for maxComputeWorkgroupsPerDimension. Any larger
// 1-D dispatch must be tiled into 2-D so both axes stay <= this bound.
// The consumer shaders linearise via WORKGROUP_ID = w_id.x + w_id.y * w_dim.x.
const MAX_DIM: u32 = 65535u;

fn splitWg(count: u32) -> vec2<u32> {
    if (count <= MAX_DIM) {
        return vec2<u32>(count, 1u);
    }
    let y = (count + MAX_DIM - 1u) / MAX_DIM;
    let x = (count + y - 1u) / y;
    return vec2<u32>(x, y);
}

@compute @workgroup_size(1)
fn main() {
    let n = totalPairs[0];
    let sortWg = (n + SORT_ELEMENTS_PER_WG - 1u) / SORT_ELEMENTS_PER_WG;
    let bndWg = (n + BOUNDARIES_THREADS_PER_WG - 1u) / BOUNDARIES_THREADS_PER_WG;
    let sortDim = splitWg(sortWg);
    let bndDim = splitWg(bndWg);
    let s = uniforms.sortSlotBase;
    let b = uniforms.boundariesSlotBase;
    indirectBuffer[s + 0u] = sortDim.x;
    indirectBuffer[s + 1u] = sortDim.y;
    indirectBuffer[s + 2u] = 1u;
    indirectBuffer[b + 0u] = bndDim.x;
    indirectBuffer[b + 1u] = bndDim.y;
    indirectBuffer[b + 2u] = 1u;
}
`;

export { prepareIndirectWgsl };
