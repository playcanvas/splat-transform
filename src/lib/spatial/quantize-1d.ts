import { Column, DataTable } from '../data-table/data-table';

/**
 * Optimal 1D quantization using dynamic programming on a histogram.
 *
 * Pools all columns of the input DataTable into a single 1D dataset,
 * bins values into a histogram, then uses DP to find k centroids that
 * minimize weighted sum-of-squared-errors (SSE).
 *
 * Bin weights use sub-linear density weighting: weight = count^alpha.
 * With alpha < 1, sparse tail regions of the distribution earn
 * meaningful influence on centroid placement, preventing the
 * catastrophic clipping of extreme values that standard k-means
 * (alpha = 1) exhibits on heavy-tailed distributions.
 *
 * @param dataTable - Input data table whose columns are pooled into 1D.
 * @param k - Number of codebook entries (default 256).
 * @param alpha - Density weight exponent. 0 = uniform (each bin equal),
 * 0.5 = sqrt (balanced), 1.0 = standard MSE (dense regions dominate).
 * Default 0.5.
 * @returns Object with `centroids` (DataTable with one 'data' column of
 * k Float32 values, sorted ascending) and `labels` (DataTable with same
 * column layout as input, each column containing Uint8Array indices into
 * the codebook).
 */
const quantize1d = (dataTable: DataTable, k = 256, alpha = 0.5) => {
    const { numColumns, numRows } = dataTable;

    // pool all columns into a flat 1D array
    const N = numRows * numColumns;
    const data = new Float32Array(N);
    for (let i = 0; i < numColumns; ++i) {
        data.set(dataTable.getColumn(i).data, i * numRows);
    }

    // find global min/max
    let dataMin = Infinity;
    let dataMax = -Infinity;
    for (let i = 0; i < N; ++i) {
        const v = data[i];
        if (v < dataMin) dataMin = v;
        if (v > dataMax) dataMax = v;
    }

    // handle degenerate case where all values are identical
    if (dataMax - dataMin < 1e-20) {
        const centroids = new DataTable([new Column('data', new Float32Array(k))]);
        centroids.getColumn(0).data.fill(dataMin);

        const result = new DataTable(dataTable.columnNames.map(name => new Column(name, new Uint8Array(numRows))));
        return { centroids, labels: result };
    }

    // build histogram
    const H = 1024;
    const binWidth = (dataMax - dataMin) / H;
    const counts = new Float64Array(H);
    const sums = new Float64Array(H);

    for (let i = 0; i < N; ++i) {
        const bin = Math.min(H - 1, Math.floor((data[i] - dataMin) / binWidth));
        counts[bin]++;
        sums[bin] += data[i];
    }

    // compute bin centers (mean of values in each bin, or geometric center if empty)
    const centers = new Float64Array(H);
    for (let i = 0; i < H; ++i) {
        centers[i] = counts[i] > 0 ? sums[i] / counts[i] : dataMin + (i + 0.5) * binWidth;
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

    // clamp k to number of non-empty bins
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

    // build output in the same format as cluster1d
    const centroids = new DataTable([new Column('data', finalCentroids)]);

    const result = new DataTable(dataTable.columnNames.map(name => new Column(name, new Uint8Array(numRows))));
    for (let i = 0; i < numColumns; ++i) {
        result.getColumn(i).data.set(labels.subarray(i * numRows, (i + 1) * numRows));
    }

    return { centroids, labels: result };
};

export { quantize1d };
