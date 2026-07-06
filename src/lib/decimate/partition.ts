import { quickselect } from '../utils';

/**
 * Resident per-gaussian position columns — the only whole-scene data
 * decimation keeps in memory (12 B/gaussian).
 */
type ResidentPositions = {
    x: Float32Array;
    y: Float32Array;
    z: Float32Array;
};

/**
 * One spatial block of the KD partition: gaussians `order[start..end)`,
 * sorted ascending (for gather coalescing), with the block's position AABB
 * as `[minx, miny, minz, maxx, maxy, maxz]`.
 */
type BlockRange = {
    start: number;
    end: number;
    aabb: Float32Array;
};

/**
 * KD-partition the resident positions into spatial blocks of at most
 * `blockSize` gaussians by recursive median splits on the largest AABB axis
 * (quickselect, in place on one index array). Blocks are an IO pattern only —
 * with globally exact KNN, block boundaries cannot affect the decimation
 * result.
 *
 * @param pos - Resident positions.
 * @param blockSize - Maximum gaussians per block.
 * @returns The permuted index array and the block ranges over it.
 */
const kdPartition = (pos: ResidentPositions, blockSize: number): { order: Uint32Array; blocks: BlockRange[] } => {
    const n = pos.x.length;
    const order = new Uint32Array(n);
    for (let i = 0; i < n; i++) order[i] = i;
    const blocks: BlockRange[] = [];
    const cols = [pos.x, pos.y, pos.z];

    const aabbOf = (start: number, end: number): Float32Array => {
        const a = new Float32Array([Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity]);
        for (let i = start; i < end; i++) {
            const g = order[i];
            for (let c = 0; c < 3; c++) {
                const v = cols[c][g];
                if (v < a[c]) a[c] = v;
                if (v > a[3 + c]) a[3 + c] = v;
            }
        }
        return a;
    };

    const recurse = (start: number, end: number): void => {
        const aabb = aabbOf(start, end);
        if (end - start <= blockSize) {
            order.subarray(start, end).sort();
            blocks.push({ start, end, aabb });
            return;
        }
        let axis = 0, ext = -Infinity;
        for (let c = 0; c < 3; c++) {
            const e = aabb[3 + c] - aabb[c];
            if (e > ext) {
                ext = e;
                axis = c;
            }
        }
        const mid = start + ((end - start) >> 1);
        quickselect(cols[axis], order.subarray(start, end), mid - start);
        recurse(start, mid);
        recurse(mid, end);
    };
    if (n > 0) recurse(0, n);
    return { order, blocks };
};

/**
 * Count the coalesced runs a block's sorted source rows form under the
 * reader's gap-merge threshold — the spatial-coherence signal. A coherent
 * (Morton-ordered / block-ordered) file yields a handful of runs per block;
 * a training-order file yields ~one run per row, which is the cue to
 * recommend a one-time `--morton-order` prepass.
 *
 * @param sortedIndices - Row indices, ascending, typically `order`.
 * @param start - Range start (inclusive).
 * @param end - Range end (exclusive).
 * @param mergeGapRows - Merge adjacent indices when the gap is at most this many rows.
 * @returns The number of coalesced runs.
 */
const coherenceRuns = (sortedIndices: Uint32Array, start: number, end: number, mergeGapRows: number): number => {
    let runs = end > start ? 1 : 0;
    for (let i = start + 1; i < end; i++) {
        if (sortedIndices[i] - sortedIndices[i - 1] > mergeGapRows) runs++;
    }
    return runs;
};

export { kdPartition, coherenceRuns, type BlockRange, type ResidentPositions };
