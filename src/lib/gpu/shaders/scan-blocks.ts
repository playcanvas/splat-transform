/**
 * First half of the resident path's coverage prefix sum. Each workgroup
 * scans one block of `SCAN_BLOCK` splats: it writes the block-local
 * exclusive prefix of `coverage[]` into `emitOffset[]` and the block's total
 * into `blockSums[blockId]`. `scanSumsWgsl` then scans the block totals, and
 * the emit shader adds a splat's block prefix to its local offset.
 *
 * The block size is fixed so the CPU can cut the depth-sorted list into
 * pair-budget-sized ranges at block boundaries from the block prefixes
 * alone. Dispatched over `ceil(chunkSize / SCAN_BLOCK)` workgroups, tiled
 * in 2-D past the per-dimension limit.
 *
 * @returns WGSL source for the block-scan compute shader.
 */
const SCAN_THREADS = 256;
const SCAN_PER_THREAD = 8;
/** Splats per scan block; must match the `>> 11u` in the emit shader's `pairBase`. */
const SCAN_BLOCK = SCAN_THREADS * SCAN_PER_THREAD;

const scanBlocksWgsl = () => /* wgsl */`
#include "uniformsStruct"
#include "constants"

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> coverage: array<u32>;
@group(0) @binding(2) var<storage, read_write> emitOffset: array<u32>;
@group(0) @binding(3) var<storage, read_write> blockSums: array<u32>;

const SCAN_THREADS: u32 = ${SCAN_THREADS}u;
const SCAN_PER_THREAD: u32 = ${SCAN_PER_THREAD}u;
const SCAN_BLOCK: u32 = ${SCAN_BLOCK}u;

var<workgroup> partial: array<u32, SCAN_THREADS>;

@compute @workgroup_size(SCAN_THREADS)
fn main(
    @builtin(workgroup_id) wgId: vec3<u32>,
    @builtin(num_workgroups) numWg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let block = wgId.y * numWg.x + wgId.x;
    let n = uniforms.chunkSize;
    let numBlocks = (n + SCAN_BLOCK - 1u) / SCAN_BLOCK;
    if (block >= numBlocks) { return; }

    let tid = lid.x;
    let base = block * SCAN_BLOCK + tid * SCAN_PER_THREAD;

    var sum: u32 = 0u;
    for (var k: u32 = 0u; k < SCAN_PER_THREAD; k = k + 1u) {
        let idx = base + k;
        if (idx < n) { sum = sum + coverage[idx]; }
    }
    partial[tid] = sum;
    workgroupBarrier();

    // Inclusive Hillis–Steele scan of the per-thread sums.
    for (var off: u32 = 1u; off < SCAN_THREADS; off = off << 1u) {
        var t: u32 = 0u;
        if (tid >= off) { t = partial[tid - off]; }
        workgroupBarrier();
        partial[tid] = partial[tid] + t;
        workgroupBarrier();
    }

    var run = partial[tid] - sum;
    for (var k: u32 = 0u; k < SCAN_PER_THREAD; k = k + 1u) {
        let idx = base + k;
        if (idx < n) {
            emitOffset[idx] = run;
            run = run + coverage[idx];
        }
    }
    if (tid == SCAN_THREADS - 1u) {
        blockSums[block] = partial[tid];
    }
}
`;

/**
 * Second half of the resident prefix sum: one workgroup turns `blockSums[]`
 * into exclusive block prefixes in place and writes the grand total to
 * `totalPairs[0]`. Each thread handles a contiguous run of blocks, so any
 * block count works; the per-thread runs are scanned across the workgroup.
 *
 * @returns WGSL source for the block-sum scan compute shader.
 */
const scanSumsWgsl = () => /* wgsl */`
#include "uniformsStruct"
#include "constants"

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> blockSums: array<u32>;
@group(0) @binding(2) var<storage, read_write> totalPairs: array<u32>;

const SCAN_THREADS: u32 = ${SCAN_THREADS}u;
const SCAN_BLOCK: u32 = ${SCAN_BLOCK}u;

var<workgroup> partial: array<u32, SCAN_THREADS>;

@compute @workgroup_size(SCAN_THREADS)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let numBlocks = (uniforms.chunkSize + SCAN_BLOCK - 1u) / SCAN_BLOCK;
    let per = (numBlocks + SCAN_THREADS - 1u) / SCAN_THREADS;
    let start = tid * per;

    var sum: u32 = 0u;
    for (var k: u32 = 0u; k < per; k = k + 1u) {
        let idx = start + k;
        if (idx < numBlocks) { sum = sum + blockSums[idx]; }
    }
    partial[tid] = sum;
    workgroupBarrier();

    for (var off: u32 = 1u; off < SCAN_THREADS; off = off << 1u) {
        var t: u32 = 0u;
        if (tid >= off) { t = partial[tid - off]; }
        workgroupBarrier();
        partial[tid] = partial[tid] + t;
        workgroupBarrier();
    }

    var run = partial[tid] - sum;
    for (var k: u32 = 0u; k < per; k = k + 1u) {
        let idx = start + k;
        if (idx < numBlocks) {
            let v = blockSums[idx];
            blockSums[idx] = run;
            run = run + v;
        }
    }
    if (tid == SCAN_THREADS - 1u) {
        totalPairs[0] = partial[tid];
    }
}
`;

export { scanBlocksWgsl, scanSumsWgsl, SCAN_BLOCK };
