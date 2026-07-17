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

/** Outlier fence: expand the sampled per-axis quantile interval this much. */
const OUTLIER_FENCE_FACTOR = 4;

/** Treat out-of-fence points as flyaways only while they are rare. */
const OUTLIER_MAX_FRACTION = 0.01;

/** Position sample cap for the fence quantiles. */
const OUTLIER_SAMPLE_CAP = 1 << 20;

// Per-axis fence [lo, hi] from strided-sample quantiles: mid ± factor × the
// 0.1–99.9% half-spread. An axis with no spread stays unfenced (±Infinity).
const outlierFence = (pos: ResidentPositions): { lo: number[]; hi: number[] } => {
    const n = pos.x.length;
    const stride = Math.max(1, Math.floor(n / OUTLIER_SAMPLE_CAP));
    const cols = [pos.x, pos.y, pos.z];
    const lo = [-Infinity, -Infinity, -Infinity];
    const hi = [Infinity, Infinity, Infinity];
    const samp = new Float32Array(Math.ceil(n / stride) || 1);
    for (let c = 0; c < 3; c++) {
        let m = 0;
        for (let i = 0; i < n; i += stride) samp[m++] = cols[c][i];
        const s = samp.subarray(0, m).sort();
        const qlo = s[Math.min(m - 1, Math.floor(0.001 * m))];
        const qhi = s[Math.min(m - 1, Math.floor(0.999 * m))];
        const half = (qhi - qlo) / 2;
        if (!(half > 0)) continue;
        const mid = (qlo + qhi) / 2;
        lo[c] = mid - OUTLIER_FENCE_FACTOR * half;
        hi[c] = mid + OUTLIER_FENCE_FACTOR * half;
    }
    return { lo, hi };
};

/**
 * KD-partition the resident positions into spatial blocks of at most
 * `blockSize` gaussians by recursive median splits on the largest AABB axis
 * (quickselect, in place on one index array). Rare flyaway positions are set
 * aside into trailing residual block(s) first, so core blocks keep tight
 * AABBs — flyaways otherwise stretch AABBs scene-wide, which wrecks the
 * density-based halo estimate and AABB-distance pruning downstream. Blocks
 * are an IO pattern only — with globally exact KNN, block boundaries cannot
 * affect the decimation result.
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

    // Residual split: fence classification must stay rare — a scene that is
    // mostly "outliers" is just sparse, and splitting it would recreate the
    // stretched-AABB problem inside the residual.
    let coreEnd = n;
    if (n > 0) {
        const { lo, hi } = outlierFence(pos);
        let out = 0;
        for (let i = 0; i < n; i++) {
            if (cols[0][i] < lo[0] || cols[0][i] > hi[0] ||
                cols[1][i] < lo[1] || cols[1][i] > hi[1] ||
                cols[2][i] < lo[2] || cols[2][i] > hi[2]) out++;
        }
        if (out > 0 && out <= n * OUTLIER_MAX_FRACTION) {
            coreEnd = n - out;
            let c = 0, o = coreEnd;
            for (let i = 0; i < n; i++) {
                if (cols[0][i] < lo[0] || cols[0][i] > hi[0] ||
                    cols[1][i] < lo[1] || cols[1][i] > hi[1] ||
                    cols[2][i] < lo[2] || cols[2][i] > hi[2]) order[o++] = i;
                else order[c++] = i;
            }
        }
    }
    if (coreEnd > 0) recurse(0, coreEnd);
    if (coreEnd < n) recurse(coreEnd, n);
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
