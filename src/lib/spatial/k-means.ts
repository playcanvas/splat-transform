import type { GraphicsDevice } from 'playcanvas';

import { GpuKmeans, emeansFits, uploadEmeansPoints } from '../gpu';

import { KdTree } from './kd-tree';

// use floyd's algorithm to pick m unique random indices from 0..n-1
const pickRandomIndices = (n: number, m: number) => {
    const chosen = new Set<number>();
    for (let j = n - m; j < n; j++) {
        const t = Math.floor(Math.random() * (j + 1));
        chosen.add(chosen.has(t) ? j : t);
    }
    return [...chosen];
};

// seed centroids by copying k random points (interleaved in/out)
const initCentroids = (points: Float32Array, numRows: number, nc: number, centroids: Float32Array, k: number) => {
    const indices = pickRandomIndices(numRows, k);
    for (let i = 0; i < k; ++i) {
        const src = indices[i] * nc;
        const dst = i * nc;
        for (let j = 0; j < nc; ++j) centroids[dst + j] = points[src + j];
    }
};

// 1D case: quantile-based init handles skewed data better. nc === 1, so the
// interleaved buffer is already a flat column.
const initCentroids1D = (points: Float32Array, numRows: number, centroids: Float32Array, k: number) => {
    const sorted = points.slice(0, numRows).sort();
    for (let i = 0; i < k; ++i) {
        // place centroid at the center of its expected cluster region
        const quantile = (2 * i + 1) / (2 * k);
        const index = Math.min(Math.floor(quantile * numRows), numRows - 1);
        centroids[i] = sorted[index];
    }
};

// CPU assignment fallback (no GPU): build a kd-tree over the centroids and find
// each point's nearest. points/centroids are row-major interleaved.
const assignCpu = (
    points: Float32Array,
    numRows: number,
    nc: number,
    centroids: Float32Array,
    k: number,
    labels: Uint32Array
) => {
    // de-interleave centroids into columns for the KdTree (k is small)
    const cols: Float32Array[] = [];
    for (let j = 0; j < nc; ++j) {
        const col = new Float32Array(k);
        for (let i = 0; i < k; ++i) col[i] = centroids[i * nc + j];
        cols.push(col);
    }
    const kdTree = new KdTree(cols);

    const point = new Float32Array(nc);
    for (let r = 0; r < numRows; ++r) {
        const base = r * nc;
        for (let j = 0; j < nc; ++j) point[j] = points[base + j];
        labels[r] = kdTree.findNearest(point).index;
    }
};

/**
 * k-means over row-major interleaved points (a numRows×numColumns Float32Array).
 * Returns interleaved centroids (k×numColumns) and per-point labels.
 *
 * @param points - Interleaved point data.
 * @param numRows - Number of points.
 * @param numColumns - Dimensions per point.
 * @param k - Number of clusters.
 * @param iterations - Lloyd iterations to run.
 * @param device - Optional GPU device; falls back to a CPU kd-tree assign.
 * @param onIteration - Optional callback fired per iteration (progress). No
 * logging happens here, so a clustering can run while another scope is open.
 * @returns Interleaved `centroids` (k×numColumns) and `labels`.
 * @ignore
 */
const kmeansInterleaved = async (
    points: Float32Array,
    numRows: number,
    numColumns: number,
    k: number,
    iterations: number,
    device?: GraphicsDevice,
    onIteration?: () => void
): Promise<{ centroids: Float32Array; labels: Uint32Array }> => {
    const nc = numColumns;

    // too few data points: each point is its own centroid
    if (numRows < k) {
        return {
            centroids: points.slice(0, numRows * nc),
            labels: new Uint32Array(numRows).map((_, i) => i)
        };
    }

    // construct + seed centroids
    const centroids = new Float32Array(k * nc);
    if (nc === 1) {
        initCentroids1D(points, numRows, centroids, k);
    } else {
        initCentroids(points, numRows, nc, centroids, k);
    }

    const labels = new Uint32Array(numRows);

    if (device) {
        // flash-kmeans: the whole Lloyd loop runs on the GPU with a single
        // readback of labels + centroids after the final iteration. Kernels
        // are built per call — caching them across calls was benched (M4,
        // 12-unit LOD run) at no measurable win; the driver shader cache
        // already absorbs the recompile.
        const gpuKmeans = new GpuKmeans(device, nc, k);
        try {
            await gpuKmeans.run(points, numRows, centroids, labels, iterations, onIteration);
        } finally {
            gpuKmeans.destroy();
        }
    } else {
        // recompute scratch (reused across iterations): per-cluster column sums
        const sums = new Float64Array(k * nc);
        const counts = new Uint32Array(k);

        for (let step = 0; step < iterations; ++step) {
            assignCpu(points, numRows, nc, centroids, k, labels);

            // recompute centroids in one vectorized pass: accumulate per-cluster
            // column sums into typed arrays, then divide by the cluster count.
            sums.fill(0);
            counts.fill(0);
            for (let r = 0; r < numRows; ++r) {
                const c = labels[r];
                counts[c]++;
                const sb = c * nc;
                const pb = r * nc;
                for (let j = 0; j < nc; ++j) sums[sb + j] += points[pb + j];
            }
            for (let i = 0; i < k; ++i) {
                const cb = i * nc;
                if (counts[i] === 0) {
                    // re-seed empty cluster to a random point to avoid a zero vector
                    const src = Math.floor(Math.random() * numRows) * nc;
                    for (let j = 0; j < nc; ++j) centroids[cb + j] = points[src + j];
                } else {
                    const inv = 1 / counts[i];
                    for (let j = 0; j < nc; ++j) centroids[cb + j] = sums[cb + j] * inv;
                }
            }
            onIteration?.();
        }
    }

    return { centroids, labels };
};

// e-means epochs per palette probe. Matches Lloyd-10 on most scenes (2-3
// epochs measured); an under-converged probe only costs a slightly larger
// palette, since the search checks the error it actually reached.
const COLOR_SEARCH_EPOCHS = 3;
// most palette sizes tried before settling for the smallest that met the target
const COLOR_SEARCH_MAX_PROBES = 6;
// stop once the smallest passing size is within this ratio of the largest failing one
const COLOR_SEARCH_TOLERANCE = 1.25;
// ...or once a passing size's error is within this fraction of the target: the
// error falls slowly with size (about size^-0.15), so a size 25% smaller would
// gain only ~3% error, the same precision as the ratio above
const COLOR_SEARCH_ERROR_TOLERANCE = 0.02;
// extrapolated jumps cover this fraction of the predicted change in log size:
// the error falls faster at larger sizes (log-log slope about -0.12 at 4K, -0.2
// at 64K+), so a full jump from the slope measured so far overshoots, and the
// oversized passing palette it lands on can use up the probe budget
const COLOR_SEARCH_DAMPING = 0.7;
// growing a palette (when 4096 entries don't meet the target): e-means epochs
// per growth step, warm from the previous palette (one leaves ~2x more entries
// for the same error)
const COLOR_GROW_EPOCHS = 2;
// largest and smallest growth per step
const COLOR_GROW_MAX_STEP = 2;
const COLOR_GROW_MIN_STEP = 1.1;
// log-log slope of error against size for the first growth step (-0.12 to -0.2
// measured); later steps use the slope measured between the last two sizes
const COLOR_GROW_FIRST_SLOPE = -0.15;
// guard on growth steps (4096 to 262144 takes about 7)
const COLOR_GROW_MAX_STEPS = 32;
// palette rows hold 64 entries: sizes are whole rows
const PALETTE_ROW = 64;

/**
 * Weighted, view-averaged RMS colour error per channel, in 8-bit levels, of
 * replacing each point's SH rest coefficients with its palette entry. The
 * real SH basis is orthonormal, so the squared colour error of one channel
 * averaged over view directions is the sum of its squared coefficient errors
 * over 4π. Points are interleaved with numColumns = 3 × coefficients per channel.
 *
 * @param points - Interleaved SH rest coefficients.
 * @param numRows - Number of points.
 * @param numColumns - Coefficients per point.
 * @param centroids - Interleaved palette.
 * @param labels - Palette entry per point.
 * @param weights - Per-point weight.
 * @returns RMS colour error in 8-bit levels.
 * @ignore
 */
const shColorError = (
    points: Float32Array,
    numRows: number,
    numColumns: number,
    centroids: Float32Array,
    labels: Uint32Array,
    weights: Float32Array
): number => {
    let sum = 0;
    let total = 0;
    for (let i = 0; i < numRows; i++) {
        const p = i * numColumns;
        const c = labels[i] * numColumns;
        let d = 0;
        for (let j = 0; j < numColumns; j++) {
            const v = points[p + j] - centroids[c + j];
            d += v * v;
        }
        sum += weights[i] * d;
        total += weights[i];
    }
    if (total <= 0) return 0;
    // mean squared error per channel = (per-point squared error / 3 channels)
    return 255 * Math.sqrt(sum / total / 3 / (4 * Math.PI));
};

type PaletteProbe = { k: number; error: number };

// Grow a k-entry palette to `next` entries by splitting the entries with the most
// weighted squared error: each new entry starts at one of that entry's points,
// sampled by weight (sampling by error, as k-means++ does, favours outliers and
// leaves the new entries nearly unused). Grown palettes match fresh 3-epoch ones
// of the same size, on real SH data and on tightly clustered data alike.
const splitEntries = (
    points: Float32Array,
    numRows: number,
    nc: number,
    centroids: Float32Array,
    labels: Uint32Array,
    weights: Float32Array,
    k: number,
    next: number
): Float32Array => {
    const sse = new Float64Array(k);
    const keys = new Float64Array(k).fill(-Infinity);
    const picks = new Int32Array(k).fill(-1);
    for (let i = 0; i < numRows; i++) {
        const c = labels[i];
        let d = 0;
        for (let j = 0; j < nc; j++) {
            const v = points[i * nc + j] - centroids[c * nc + j];
            d += v * v;
        }
        const w = weights[i];
        sse[c] += w * d;
        if (w > 0) {
            // weighted reservoir of one (Efraimidis-Spirakis): largest u^(1/w)
            const key = Math.log(Math.random()) / w;
            if (key > keys[c]) {
                keys[c] = key;
                picks[c] = i;
            }
        }
    }
    const order = Uint32Array.from({ length: k }, (_, i) => i).sort((a, b) => sse[b] - sse[a]);
    const grown = new Float32Array(next * nc);
    grown.set(centroids.subarray(0, k * nc));
    let n = k;
    for (let i = 0; i < k && n < next; i++) {
        const c = order[i];
        if (picks[c] < 0 || sse[c] === 0) continue;
        grown.set(points.subarray(picks[c] * nc, (picks[c] + 1) * nc), n * nc);
        n++;
    }
    // too few entries with any error to split: new entries at random points
    while (n < next) {
        const row = Math.floor(Math.random() * numRows);
        grown.set(points.subarray(row * nc, (row + 1) * nc), n * nc);
        n++;
    }
    return grown;
};

/**
 * Size an SH palette from a colour-error target with GPU e-means (weighted),
 * keeping the smallest palette found whose weighted colour error
 * ({@link shColorError}) is at most `target`. A 4096-entry palette is clustered
 * first. If it misses the target, the palette grows: the entries with the most
 * error are split and the palette refined warm, by steps sized from the log-log
 * slope of error against size, until it meets the target. If it meets it, smaller
 * sizes are clustered afresh, stepping by the measured slope (damped) and
 * interpolating between a size that meets the target and one that doesn't. The
 * points are uploaded once for all of it. When no size up to `maxK` meets the
 * target, the `maxK` palette is returned. A palette of every point (maxK ≥
 * numRows) is exact.
 *
 * @param points - Interleaved SH rest coefficients (numRows × numColumns).
 * @param numRows - Number of points.
 * @param numColumns - Coefficients per point.
 * @param weights - Per-point weight, both for the clustering and the error.
 * @param target - Colour error target in 8-bit levels.
 * @param maxK - Largest palette allowed.
 * @param device - GPU device.
 * @param onProbe - Optional callback fired after each palette size is tried (at most
 *   {@link COLOR_SEARCH_MAX_PROBES} times, for a progress bar).
 * @returns The palette, labels, and every size tried with its error.
 * @ignore
 */
const kmeansToColorError = async (
    points: Float32Array,
    numRows: number,
    numColumns: number,
    weights: Float32Array,
    target: number,
    maxK: number,
    device: GraphicsDevice,
    onProbe?: () => void
): Promise<{ centroids: Float32Array; labels: Uint32Array; probes: PaletteProbe[] }> => {
    // every point its own entry: exact
    if (maxK >= numRows) {
        const exact = {
            centroids: points.slice(0, numRows * numColumns),
            labels: new Uint32Array(numRows).map((_, i) => i)
        };
        const probe = { k: numRows, error: 0 };
        // fewer entries may still meet the target; search below numRows
        maxK = Math.floor((numRows - 1) / PALETTE_ROW) * PALETTE_ROW;
        if (maxK < PALETTE_ROW) return { ...exact, probes: [probe] };
        const result = await kmeansToColorError(points, numRows, numColumns, weights, target, maxK, device, onProbe);
        const best = result.probes.find((p) => p.error <= target);
        return best ? result : { ...exact, probes: [...result.probes, probe] };
    }

    const clamp = (k: number) => Math.min(maxK, Math.max(PALETTE_ROW, Math.ceil(k / PALETTE_ROW) * PALETTE_ROW));
    const probes: PaletteProbe[] = [];
    let best: { centroids: Float32Array; labels: Uint32Array; k: number } | null = null;
    let largest: { centroids: Float32Array; labels: Uint32Array; k: number } | null = null;

    // the points and weights are shuffled and uploaded once for every size tried
    const resident = uploadEmeansPoints(device, points, numRows, numColumns, weights);

    let reported = 0;
    // cluster `centroids` (k entries, updated in place) for `epochs`; record the size
    const cluster = async (centroids: Float32Array, k: number, epochs: number) => {
        const labels = new Uint32Array(numRows);
        const gpu = new GpuKmeans(device, numColumns, k);
        try {
            await gpu.runEmeans(points, numRows, centroids, labels, epochs, undefined, weights, resident);
        } finally {
            gpu.destroy();
        }
        const error = shColorError(points, numRows, numColumns, centroids, labels, weights);
        probes.push({ k, error });
        if (error <= target && (!best || k < best.k)) best = { centroids, labels, k };
        if (!largest || k > largest.k) largest = { centroids, labels, k };
        if (reported++ < COLOR_SEARCH_MAX_PROBES) onProbe?.();
        return { labels, error };
    };

    // a fresh palette of k entries seeded at random points
    const evaluate = async (k: number) => {
        const centroids = new Float32Array(k * numColumns);
        initCentroids(points, numRows, numColumns, centroids, k);
        return { centroids, ...(await cluster(centroids, k, COLOR_SEARCH_EPOCHS)) };
    };

    // log-log interpolation (or extrapolation) of the size reaching `target`
    const sizeFor = (a: PaletteProbe, b: PaletteProbe) => {
        const slope = (Math.log(b.error) - Math.log(a.error)) / (Math.log(b.k) - Math.log(a.k));
        if (!(slope < 0) || !Number.isFinite(slope)) return null;
        return Math.exp(Math.log(a.k) + (Math.log(target) - Math.log(a.error)) / slope);
    };

    // a damped step from size `from` towards the predicted size `to`
    const damped = (from: number, to: number) => from * (to / from) ** COLOR_SEARCH_DAMPING;

    // the first size met the target: search smaller sizes
    const shrink = async () => {
        while (probes.length < COLOR_SEARCH_MAX_PROBES) {
            const passing = probes.filter((p) => p.error <= target).sort((a, b) => a.k - b.k);
            const failing = probes.filter((p) => p.error > target).sort((a, b) => b.k - a.k);
            const hi = passing[0];
            const lo = failing[0];
            // close enough to the target that a smaller size can't gain much
            if (hi.error >= target * (1 - COLOR_SEARCH_ERROR_TOLERANCE)) return;
            let next: number;
            if (lo) {
                if (hi.k <= lo.k * COLOR_SEARCH_TOLERANCE) return;
                const guess = sizeFor(lo, hi) ?? Math.sqrt(lo.k * hi.k);
                next = clamp(Math.min(hi.k / 1.05, Math.max(lo.k * 1.05, guess)));
                if (next <= lo.k || next >= hi.k) return;
            } else {
                // all good enough: ÷4 first, then a damped step by the measured slope, at least 1.2×
                if (hi.k <= PALETTE_ROW) return;
                const guess = passing.length > 1 ? sizeFor(passing[0], passing[1]) : null;
                next = clamp(guess ? Math.min(hi.k / 1.2, Math.max(hi.k / 16, damped(hi.k, guess))) : hi.k / 4);
            }
            if (probes.some((p) => p.k === next)) return;
            await evaluate(next);
        }
    };

    try {
        let k = clamp(4096);
        let { centroids, labels, error }: { centroids: Float32Array; labels: Uint32Array; error: number } =
            await evaluate(k);
        if (error <= target) {
            await shrink();
        } else {
            // grow this palette until it meets the target
            for (let step = 0; error > target && k < maxK && step < COLOR_GROW_MAX_STEPS; step++) {
                const prev = probes[probes.length - 2];
                const slope = prev
                    ? (Math.log(error) - Math.log(prev.error)) / (Math.log(k) - Math.log(prev.k))
                    : COLOR_GROW_FIRST_SLOPE;
                const factor = slope < 0 ? (target / error) ** (1 / slope) : COLOR_GROW_MAX_STEP;
                const next = clamp(k * Math.min(COLOR_GROW_MAX_STEP, Math.max(COLOR_GROW_MIN_STEP, factor)));
                if (next <= k) break;
                centroids = splitEntries(points, numRows, numColumns, centroids, labels, weights, k, next);
                k = next;
                ({ labels, error } = await cluster(centroids, k, COLOR_GROW_EPOCHS));
            }
        }
    } finally {
        resident.destroy();
    }

    const chosen = best ?? largest!;
    return { centroids: chosen.centroids, labels: chosen.labels, probes };
};

/**
 * Whether {@link kmeansToColorError} can run on this input: its e-means
 * keeps every point resident on the GPU.
 *
 * @param device - GPU device.
 * @param numRows - Number of points.
 * @param numColumns - Coefficients per point.
 * @returns True when the colour-error search fits the device.
 * @ignore
 */
const colorErrorSearchFits = (device: GraphicsDevice, numRows: number, numColumns: number) =>
    emeansFits(device, numRows, numColumns);

export {
    kmeansInterleaved,
    kmeansToColorError,
    colorErrorSearchFits,
    shColorError,
    COLOR_SEARCH_MAX_PROBES,
    type PaletteProbe
};
