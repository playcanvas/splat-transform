/**
 * Single-workgroup exclusive prefix-sum of the project shader's
 * per-splat `coverage[]` into `emitOffset[]`. Also writes the total
 * pair count into `totalPairs[0]` so downstream kernels and the radix
 * sort can size their dispatches without a CPU round trip.
 *
 * Layout: 256 threads, each processes `SCAN_PER_THREAD` elements
 * serially (chosen so that 256 × SCAN_PER_THREAD ≥ chunkCap). Phase 1
 * computes a per-thread partial sum; phase 2 has thread 0 scan the 256
 * partials in shared memory (negligible vs the 800-element block work);
 * phase 3 each thread re-walks its block writing the exclusive prefix.
 *
 * @param scanPerThread - Per-thread element budget; must satisfy `256 * scanPerThread >= chunkCap`.
 * @returns WGSL source for the prefix-sum compute shader.
 */
const prefixSumWgsl = (scanPerThread: number) => /* wgsl */`
#include "uniformsStruct"
#include "constants"

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> coverage: array<u32>;
@group(0) @binding(2) var<storage, read_write> emitOffset: array<u32>;
@group(0) @binding(3) var<storage, read_write> totalPairs: array<u32>;

const SCAN_THREADS: u32 = 256u;
const SCAN_PER_THREAD: u32 = ${scanPerThread}u;

var<workgroup> scratch: array<u32, SCAN_THREADS>;

@compute @workgroup_size(SCAN_THREADS)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n = uniforms.chunkSize;
    let base = tid * SCAN_PER_THREAD;

    var partial: u32 = 0u;
    for (var i: u32 = 0u; i < SCAN_PER_THREAD; i = i + 1u) {
        let idx = base + i;
        if (idx < n) {
            partial = partial + coverage[idx];
        }
    }
    scratch[tid] = partial;
    workgroupBarrier();

    if (tid == 0u) {
        var acc: u32 = 0u;
        for (var i: u32 = 0u; i < SCAN_THREADS; i = i + 1u) {
            let v = scratch[i];
            scratch[i] = acc;
            acc = acc + v;
        }
        totalPairs[0] = acc;
    }
    workgroupBarrier();

    var prefix: u32 = scratch[tid];
    for (var i: u32 = 0u; i < SCAN_PER_THREAD; i = i + 1u) {
        let idx = base + i;
        if (idx < n) {
            emitOffset[idx] = prefix;
            prefix = prefix + coverage[idx];
        }
    }
}
`;

export { prefixSumWgsl };
