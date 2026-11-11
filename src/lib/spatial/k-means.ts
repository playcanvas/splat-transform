import type { GraphicsDevice } from 'playcanvas';

import { GpuKmeans, emeansFits } from '../gpu';

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

/**
 * Size an SH palette from a colour-error target: cluster with GPU e-means
 * (weighted) at a few palette sizes, interpolating log error against log size
 * between them, and keep the smallest size found whose weighted colour error
 * ({@link shColorError}) is at most `target`. When no size up to `maxK` meets
 * it, the `maxK` palette is returned. A palette of every point (maxK ≥
 * numRows) is exact.
 *
 * @param points - Interleaved SH rest coefficients (numRows × numColumns).
 * @param numRows - Number of points.
 * @param numColumns - Coefficients per point.
 * @param weights - Per-point weight, both for the clustering and the error.
 * @param target - Colour error target in 8-bit levels.
 * @param maxK - Largest palette allowed.
 * @param device - GPU device.
 * @param onProbe - Optional callback fired after each palette size is tried.
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

    const evaluate = async (k: number) => {
        const centroids = new Float32Array(k * numColumns);
        initCentroids(points, numRows, numColumns, centroids, k);
        const labels = new Uint32Array(numRows);
        const gpu = new GpuKmeans(device, numColumns, k);
        try {
            await gpu.runEmeans(points, numRows, centroids, labels, COLOR_SEARCH_EPOCHS, undefined, weights);
        } finally {
            gpu.destroy();
        }
        const error = shColorError(points, numRows, numColumns, centroids, labels, weights);
        probes.push({ k, error });
        if (error <= target && (!best || k < best.k)) best = { centroids, labels, k };
        if (!largest || k > largest.k) largest = { centroids, labels, k };
        onProbe?.();
    };

    // log-log interpolation (or extrapolation) of the size reaching `target`
    const sizeFor = (a: PaletteProbe, b: PaletteProbe) => {
        const slope = (Math.log(b.error) - Math.log(a.error)) / (Math.log(b.k) - Math.log(a.k));
        if (!(slope < 0) || !Number.isFinite(slope)) return null;
        return Math.exp(Math.log(a.k) + (Math.log(target) - Math.log(a.error)) / slope);
    };

    let k = clamp(4096);
    for (let i = 0; i < COLOR_SEARCH_MAX_PROBES; i++) {
        await evaluate(k);
        const passing = probes.filter((p) => p.error <= target).sort((a, b) => a.k - b.k);
        const failing = probes.filter((p) => p.error > target).sort((a, b) => b.k - a.k);
        const hi = passing[0];
        const lo = failing[0];
        let next: number;
        if (hi && lo) {
            if (hi.k <= lo.k * COLOR_SEARCH_TOLERANCE) break;
            const guess = sizeFor(lo, hi) ?? Math.sqrt(lo.k * hi.k);
            next = clamp(Math.min(hi.k / 1.05, Math.max(lo.k * 1.05, guess)));
            if (next <= lo.k || next >= hi.k) break;
        } else if (lo) {
            // all too coarse: grow, at least 1.5×, by the local slope when known
            if (lo.k >= maxK) break;
            const guess = failing.length > 1 ? sizeFor(failing[1], failing[0]) : null;
            next = clamp(Math.max(lo.k * 1.5, Math.min(lo.k * 16, guess ?? lo.k * 4)));
        } else {
            // all good enough: shrink, at least 1.5×
            if (hi.k <= PALETTE_ROW) break;
            const guess = passing.length > 1 ? sizeFor(passing[0], passing[1]) : null;
            next = clamp(Math.min(hi.k / 1.5, Math.max(hi.k / 16, guess ?? hi.k / 4)));
        }
        if (probes.some((p) => p.k === next)) break;
        k = next;
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
