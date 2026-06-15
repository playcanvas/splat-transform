import type { TypedArray } from '../data-table/data-table';

/**
 * Raw-column form of quantize1d's input and output. This module is bundled
 * into the worker (see src/lib/workers), so it must stay free of runtime
 * imports - DataTable in particular drags in Transform and the playcanvas
 * engine. The DataTable-typed wrapper lives in quantize-1d.ts.
 */
type QuantizedColumns = {
    centroids: Float32Array;
    labels: { name: string, data: Uint8Array }[];
};

/**
 * Optimal 1D quantization using dynamic programming on a histogram.
 *
 * Pools all columns into a single 1D dataset, sorts the values, bins them
 * using a blend of uniform and quantile positioning, then uses DP to find k
 * centroids that minimize weighted sum-of-squared-errors (SSE).
 *
 * Bin positions are an adaptive blend of uniform (value-space) and
 * quantile (rank-space) positioning. The blend ratio is computed from
 * the data's IQR-to-range ratio: extreme outlier distributions (small
 * IQR relative to range) use near-pure quantile to give the dense
 * center adequate bins, while moderate-tail distributions reduce
 * quantile bias (but keep at least 50% quantile weighting).
 *
 * Bin weights use sub-linear density weighting: weight = count^alpha.
 * With alpha < 1, sparse tail regions earn meaningful influence on
 * centroid placement.
 *
 * @param columns - Named columns pooled into 1D.
 * @param k - Number of codebook entries (default 256).
 * @param alpha - Density weight exponent. 0 = uniform (each bin equal),
 * 0.5 = sqrt (balanced), 1.0 = standard MSE (dense regions dominate).
 * Default 0.5.
 * @returns Object with `centroids` (k Float32 values, sorted ascending) and
 * `labels` (same column layout as input, each holding Uint8Array indices
 * into the codebook).
 */
const quantize1dColumns = (columns: { name: string, data: TypedArray }[], k = 256, alpha = 0.5): QuantizedColumns => {
    const numColumns = columns.length;
    const numRows = numColumns > 0 ? columns[0].data.length : 0;

    // pool all columns into a flat 1D array
    const N = numRows * numColumns;

    if (N === 0) {
        return {
            centroids: new Float32Array(k),
            labels: columns.map(c => ({ name: c.name, data: new Uint8Array(numRows) }))
        };
    }

    const data = new Float32Array(N);
    for (let i = 0; i < numColumns; ++i) {
        data.set(columns[i].data, i * numRows);
    }

    // sort a copy for histogram binning (keep original for label assignment)
    const sortedData = new Float32Array(data);
    sortedData.sort();

    const vMin = sortedData[0];
    const vMax = sortedData[N - 1];

    // handle degenerate case where all values are identical
    if (vMax - vMin < 1e-20) {
        const centroids = new Float32Array(k);
        centroids.fill(vMin);

        return {
            centroids,
            labels: columns.map(c => ({ name: c.name, data: new Uint8Array(numRows) }))
        };
    }

    // build histogram using blended uniform/quantile bin positions
    const H = Math.min(1024, N);
    const vRange = vMax - vMin;

    // adaptive blend ratio: when outliers are extreme (IQR << range), lean
    // strongly toward quantile to give the dense center adequate bins; when
    // the distribution has moderate tails (IQR ~ range), reduce quantile
    // bias somewhat, but keep at least 50% quantile to preserve density
    const iqr = sortedData[Math.floor(N * 0.75)] - sortedData[Math.floor(N * 0.25)];
    const beta = Math.max(0.5, Math.min(0.999, 1 - iqr / vRange));

    const counts = new Float64Array(H);
    const sums = new Float64Array(H);

    for (let i = 0; i < N; ++i) {
        const uniformPos = (sortedData[i] - vMin) / vRange;
        const quantilePos = i / N;
        const bin = Math.min(H - 1, Math.floor(H * (beta * quantilePos + (1 - beta) * uniformPos)));
        counts[bin]++;
        sums[bin] += sortedData[i];
    }

    const centers = new Float64Array(H);
    for (let i = 0; i < H; ++i) {
        centers[i] = counts[i] > 0 ? sums[i] / counts[i] : vMin + (i + 0.5) / H * vRange;
    }

    // compute weights: w = count^alpha (sub-linear density weighting)
    const weights = new Float64Array(H);
    for (let i = 0; i < H; ++i) {
        weights[i] = counts[i] > 0 ? Math.pow(counts[i], alpha) : 0;
    }

    // prefix sums for O(1) range cost queries
    //   cost(a,b) = sum_wxx - sum_wx^2 / sum_w
    //   centroid(a,b) = sum_wx / sum_w
    const prefW = new Float64Array(H + 1);
    const prefWX = new Float64Array(H + 1);
    const prefWXX = new Float64Array(H + 1);
    for (let i = 0; i < H; ++i) {
        prefW[i + 1] = prefW[i] + weights[i];
        prefWX[i + 1] = prefWX[i] + weights[i] * centers[i];
        prefWXX[i + 1] = prefWXX[i] + weights[i] * centers[i] * centers[i];
    }

    const rangeCost = (a: number, b: number): number => {
        const w = prefW[b + 1] - prefW[a];
        if (w <= 0) return 0;
        const wx = prefWX[b + 1] - prefWX[a];
        const wxx = prefWXX[b + 1] - prefWXX[a];
        return wxx - (wx * wx) / w;
    };

    const rangeMean = (a: number, b: number): number => {
        const w = prefW[b + 1] - prefW[a];
        if (w <= 0) return (centers[a] + centers[b]) * 0.5;
        return (prefWX[b + 1] - prefWX[a]) / w;
    };

    const nonEmpty = counts.reduce((n, c) => n + (c > 0 ? 1 : 0), 0);
    const effectiveK = Math.min(k, nonEmpty);

    // DP: dp[m][j] = min weighted SSE of quantizing bins 0..j into m centroids
    // Use two rows to save memory (only need previous row)
    const INF = 1e30;
    let dpPrev = new Float64Array(H).fill(INF);
    let dpCurr = new Float64Array(H).fill(INF);
    const splitTable = new Array(effectiveK + 1);

    // base case: m = 1
    const split1 = new Int32Array(H);
    for (let j = 0; j < H; ++j) {
        dpPrev[j] = rangeCost(0, j);
        split1[j] = -1;
    }
    splitTable[1] = split1;

    // fill DP for m = 2..effectiveK
    for (let m = 2; m <= effectiveK; ++m) {
        dpCurr.fill(INF);
        const splitM = new Int32Array(H);

        for (let j = m - 1; j < H; ++j) {
            let bestCost = INF;
            let bestS = m - 2;

            for (let s = m - 2; s < j; ++s) {
                const cost = dpPrev[s] + rangeCost(s + 1, j);
                if (cost < bestCost) {
                    bestCost = cost;
                    bestS = s;
                }
            }

            dpCurr[j] = bestCost;
            splitM[j] = bestS;
        }

        splitTable[m] = splitM;

        // swap rows
        const tmp = dpPrev;
        dpPrev = dpCurr;
        dpCurr = tmp;
    }

    // backtrack to find centroid values
    const centroidValues = new Float32Array(effectiveK);
    let j = H - 1;
    for (let m = effectiveK; m >= 1; --m) {
        const s = m > 1 ? splitTable[m][j] : -1;
        centroidValues[m - 1] = rangeMean(s + 1, j);
        j = s;
    }

    // sort centroids (should already be sorted, but ensure)
    centroidValues.sort();

    // pad to k entries if effectiveK < k (duplicate last centroid)
    const finalCentroids = new Float32Array(k);
    finalCentroids.set(centroidValues);
    for (let i = effectiveK; i < k; ++i) {
        finalCentroids[i] = centroidValues[effectiveK - 1];
    }

    // assign each data point to nearest centroid via binary search
    const labels = new Uint8Array(N);
    for (let i = 0; i < N; ++i) {
        const v = data[i];

        // binary search for nearest centroid
        let lo = 0;
        let hi = k - 1;
        while (lo < hi) {
            const mid = (lo + hi) >> 1;
            // compare against midpoint between centroids mid and mid+1
            if (v < (finalCentroids[mid] + finalCentroids[mid + 1]) * 0.5) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        labels[i] = lo;
    }

    return {
        centroids: finalCentroids,
        labels: columns.map((c, i) => ({
            name: c.name,
            data: labels.slice(i * numRows, (i + 1) * numRows)
        }))
    };
};

export { quantize1dColumns, type QuantizedColumns };
