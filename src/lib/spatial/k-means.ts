import { GraphicsDevice } from 'playcanvas';

import { KdTree } from './kd-tree';
import { Column, DataTable } from '../data-table';
import { GpuKmeans } from '../gpu';
import { logger } from '../utils';

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
const assignCpu = (points: Float32Array, numRows: number, nc: number, centroids: Float32Array, k: number, labels: Uint32Array) => {
    // de-interleave centroids into columns for the KdTree (k is small)
    const cols: Column[] = [];
    for (let j = 0; j < nc; ++j) {
        const col = new Float32Array(k);
        for (let i = 0; i < k; ++i) col[i] = centroids[i * nc + j];
        cols.push(new Column(`c${j}`, col));
    }
    const kdTree = new KdTree(new DataTable(cols));

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
 * @returns Interleaved `centroids` (k×numColumns) and `labels`.
 * @ignore
 */
const kmeansInterleaved = async (
    points: Float32Array,
    numRows: number,
    numColumns: number,
    k: number,
    iterations: number,
    device?: GraphicsDevice
): Promise<{ centroids: Float32Array, labels: Uint32Array }> => {
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

    logger.debug(`running k-means clustering: dims=${nc} points=${numRows} clusters=${k} iterations=${iterations}`);

    const bar = logger.bar('k-means', iterations);
    if (device) {
        // flash-kmeans: the whole Lloyd loop runs on the GPU with a single
        // readback of labels + centroids after the final iteration
        const gpuKmeans = new GpuKmeans(device, nc, k);
        try {
            await gpuKmeans.run(points, numRows, centroids, labels, iterations, () => bar.tick());
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
            bar.tick();
        }
    }

    bar.end();

    return { centroids, labels };
};

/**
 * DataTable wrapper around {@link kmeansInterleaved} for the legacy writer.
 * Interleaves the input columns, clusters, then de-interleaves the centroids
 * back into a DataTable with the same column layout.
 * @ignore
 */
const kmeans = async (points: DataTable, k: number, iterations: number, device?: GraphicsDevice) => {
    const numRows = points.numRows;
    const nc = points.numColumns;

    // interleave columns into row-major order
    const flat = new Float32Array(numRows * nc);
    const cols = points.columns.map(c => c.data);
    for (let r = 0; r < numRows; ++r) {
        const base = r * nc;
        for (let j = 0; j < nc; ++j) flat[base + j] = cols[j][r];
    }

    const { centroids, labels } = await kmeansInterleaved(flat, numRows, nc, k, iterations, device);

    // de-interleave centroids back into the original column layout
    const kRows = centroids.length / nc;
    const outCols = points.columns.map((c, j) => {
        const d = new Float32Array(kRows);
        for (let i = 0; i < kRows; ++i) d[i] = centroids[i * nc + j];
        return new Column(c.name, d);
    });

    return { centroids: new DataTable(outCols), labels };
};

export { kmeans, kmeansInterleaved };
