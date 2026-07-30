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
    /** Rare out-of-fence residual block. */
    residual: boolean;
};

type OutlierFence = {
    lo: number[];
    hi: number[];
};

type PartitionResult = {
    order: Uint32Array;
    blocks: BlockRange[];
    fence: OutlierFence;
};

/** Outlier fence: expand the sampled per-axis quantile interval this much. */
const OUTLIER_FENCE_FACTOR = 4;

/** Treat out-of-fence points as flyaways only while they are rare. */
const OUTLIER_MAX_FRACTION = 0.01;

/** Position sample cap for the fence quantiles. */
const OUTLIER_SAMPLE_CAP = 1 << 20;

// Per-axis fence [lo, hi] from strided-sample quantiles: mid ± factor × the
// 0.1–99.9% half-spread. An axis with no spread stays unfenced (±Infinity).
const outlierFence = (pos: ResidentPositions): OutlierFence => {
    const n = pos.x.length;
    const stride = Math.max(1, Math.ceil(n / OUTLIER_SAMPLE_CAP));
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
 * density-based halo estimate and AABB-distance pruning downstream. With
 * globally exact KNN, block boundaries cannot change which merges are
 * possible or their costs — but they do set output row order, and selection
 * tie-breaks between quantized-equal costs can resolve differently under a
 * different partition.
 *
 * @param pos - Resident positions.
 * @param blockSize - Maximum gaussians per block.
 * @param generation - Generation number used for deterministic split jitter.
 * @returns The permuted index array and the block ranges over it.
 */
const kdPartition = (pos: ResidentPositions, blockSize: number, generation: number | null = null): PartitionResult => {
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

    const recurse = (start: number, end: number, depth: number, branch: number, residual: boolean): void => {
        const aabb = aabbOf(start, end);
        if (end - start <= blockSize) {
            order.subarray(start, end).sort();
            blocks.push({ start, end, aabb, residual });
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
        // Alternating 3/8 and 5/8 quantiles move core boundaries between
        // generations without weakening the maximum leaf-size guarantee.
        // The hash uses only stable structural inputs, so repeated runs are
        // deterministic.
        const count = end - start;
        const upper = generation !== null && ((generation + depth + branch) & 1) !== 0;
        const fraction = generation === null ? 1 / 2 : (upper ? 5 / 8 : 3 / 8);
        const offset = Math.max(1, Math.min(count - 1, Math.floor(count * fraction)));
        const mid = start + offset;
        quickselect(cols[axis], order.subarray(start, end), mid - start);
        recurse(start, mid, depth + 1, branch << 1, residual);
        recurse(mid, end, depth + 1, (branch << 1) | 1, residual);
    };

    // Residual split: fence classification must stay rare — a scene that is
    // mostly "outliers" is just sparse, and splitting it would recreate the
    // stretched-AABB problem inside the residual.
    let coreEnd = n;
    const fence = outlierFence(pos);
    if (n > blockSize) {
        const { lo, hi } = fence;
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
    if (coreEnd > 0) recurse(0, coreEnd, 0, 0, false);
    if (coreEnd < n) recurse(coreEnd, n, 0, 1, true);
    return { order, blocks, fence };
};

const aabbDistanceSquared = (a: Float32Array, b: Float32Array): number => {
    let d2 = 0;
    for (let c = 0; c < 3; c++) {
        const d = a[3 + c] < b[c] ? b[c] - a[3 + c] : (b[3 + c] < a[c] ? a[c] - b[3 + c] : 0);
        d2 += d * d;
    }
    return d2;
};

const pointAabbDistanceSquared = (x: number, y: number, z: number, a: Float32Array): number => {
    const dx = x < a[0] ? a[0] - x : (x > a[3] ? x - a[3] : 0);
    const dy = y < a[1] ? a[1] - y : (y > a[4] ? y - a[4] : 0);
    const dz = z < a[2] ? a[2] - z : (z > a[5] ? z - a[5] : 0);
    return dx * dx + dy * dy + dz * dz;
};

/**
 * Build a block-local immutable halo without scanning the scene. Candidate
 * blocks are ordered by AABB distance and only their rows are inspected.
 * Normal cores use an in-fence density radius; residual flyaway cores gather
 * nearest block populations directly so a scene-scale radius is never
 * inferred from their extents.
 *
 * @param pos - Resident positions.
 * @param partition - Partition result sharing the sampled outlier fence.
 * @param blockIndex - Core block.
 * @param maxHaloRows - Device working-set halo cap.
 * @returns Sorted global halo rows and cap diagnostics.
 */
const buildBlockHalo = (
    pos: ResidentPositions,
    partition: PartitionResult,
    blockIndex: number,
    maxHaloRows: number
): { rows: Uint32Array; capped: boolean; radius: number } => {
    const { order, blocks, fence } = partition;
    const core = blocks[blockIndex];
    const coreCount = core.end - core.start;
    const cap = Math.max(0, Math.min(coreCount, maxHaloRows));
    if (cap === 0 || blocks.length === 1) return { rows: new Uint32Array(0), capped: false, radius: 0 };

    const neighbours = blocks
    .map((block, index) => ({ block, index, d2: index === blockIndex ? Infinity : aabbDistanceSquared(core.aabb, block.aabb) }))
    .filter(v => v.index !== blockIndex)
    .sort((a, b) => a.d2 - b.d2 || a.index - b.index);

    const selected: number[] = [];
    let capped = false;
    let radius = 0;

    if (core.residual) {
        for (const { block } of neighbours) {
            for (let i = block.start; i < block.end; i++) {
                if (selected.length === cap) {
                    capped = true;
                    break;
                }
                selected.push(order[i]);
            }
            if (selected.length === cap) break;
        }
    } else {
        let inFence = 0;
        const a = new Float64Array([Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity]);
        for (let i = core.start; i < core.end; i++) {
            const g = order[i];
            const x = pos.x[g], y = pos.y[g], z = pos.z[g];
            if (x < fence.lo[0] || x > fence.hi[0] ||
                y < fence.lo[1] || y > fence.hi[1] ||
                z < fence.lo[2] || z > fence.hi[2]) continue;
            inFence++;
            a[0] = Math.min(a[0], x); a[1] = Math.min(a[1], y); a[2] = Math.min(a[2], z);
            a[3] = Math.max(a[3], x); a[4] = Math.max(a[4], y); a[5] = Math.max(a[5], z);
        }
        if (inFence > 0) {
            const ex = Math.max(0, a[3] - a[0]);
            const ey = Math.max(0, a[4] - a[1]);
            const ez = Math.max(0, a[5] - a[2]);
            const extents = [ex, ey, ez].sort((u, v) => v - u);
            const scale = extents[0] > 0 ?
                (extents[2] > extents[0] * 1e-6 ?
                    Math.cbrt(extents[0] * extents[1] * extents[2] / inFence) :
                    (extents[1] > extents[0] * 1e-6 ?
                        Math.sqrt(extents[0] * extents[1] / inFence) :
                        extents[0] / inFence)) :
                0;
            radius = 2.5 * scale;
        }
        const r2 = radius * radius;
        for (const { block, d2 } of neighbours) {
            if (d2 > r2) break;
            for (let i = block.start; i < block.end; i++) {
                const g = order[i];
                if (pointAabbDistanceSquared(pos.x[g], pos.y[g], pos.z[g], core.aabb) > r2) continue;
                if (selected.length === cap) {
                    capped = true;
                    break;
                }
                selected.push(g);
            }
            if (selected.length === cap) break;
        }
    }

    const rows = Uint32Array.from(selected);
    rows.sort();
    return { rows, capped, radius };
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

export {
    kdPartition,
    buildBlockHalo,
    coherenceRuns,
    type BlockRange,
    type OutlierFence,
    type PartitionResult,
    type ResidentPositions
};
