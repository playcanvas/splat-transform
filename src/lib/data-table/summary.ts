import { Column, DataTable, TypedArray } from './data-table';

/**
 * Statistical summary for a single column.
 */
type ColumnStats = {
    /** Minimum value (excluding NaN/Inf). */
    min: number;
    /** Maximum value (excluding NaN/Inf). */
    max: number;
    /** Median value. */
    median: number;
    /** Arithmetic mean. */
    mean: number;
    /** Standard deviation. */
    stdDev: number;
    /** Count of NaN values. */
    nanCount: number;
    /** Count of Infinity values. */
    infCount: number;
};

/**
 * Statistical summary for an entire DataTable.
 */
type SummaryData = {
    /** Summary format version. */
    version: number;
    /** Total number of rows. */
    rowCount: number;
    /** Per-column statistics keyed by column name. */
    columns: Record<string, ColumnStats>;
};

const PRECISION = 6;

const round = (value: number): number => {
    if (!Number.isFinite(value)) return value;
    return Math.round(value * Math.pow(10, PRECISION)) / Math.pow(10, PRECISION);
};

/**
 * QuickSelect algorithm to find the k-th smallest element in O(n) average time.
 * Modifies the array in place (partial reordering).
 */
const quickSelect = (arr: TypedArray, k: number, left: number, right: number): number => {
    while (left < right) {
        // Use median-of-three pivot selection for better performance
        const mid = (left + right) >>> 1;
        if (arr[mid] < arr[left]) {
            const t = arr[left]; arr[left] = arr[mid]; arr[mid] = t;
        }
        if (arr[right] < arr[left]) {
            const t = arr[left]; arr[left] = arr[right]; arr[right] = t;
        }
        if (arr[right] < arr[mid]) {
            const t = arr[mid]; arr[mid] = arr[right]; arr[right] = t;
        }

        const pivot = arr[mid];
        let i = left;
        let j = right;

        while (i <= j) {
            while (arr[i] < pivot) i++;
            while (arr[j] > pivot) j--;
            if (i <= j) {
                const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
                i++;
                j--;
            }
        }

        if (k <= j) {
            right = j;
        } else if (k >= i) {
            left = i;
        } else {
            break;
        }
    }
    return arr[k];
};

const computeColumnStats = (column: Column): ColumnStats => {
    const data = column.data;
    const len = data.length;

    // First pass: count valid values, compute min/max/sum, count NaN/Inf
    let nanCount = 0;
    let infCount = 0;
    let validCount = 0;
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;

    for (let i = 0; i < len; i++) {
        const v = data[i];
        if (Number.isNaN(v)) {
            nanCount++;
        } else if (!Number.isFinite(v)) {
            infCount++;
        } else {
            validCount++;
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
        }
    }

    // Handle case where all values are NaN/Inf
    if (validCount === 0) {
        return {
            min: NaN,
            max: NaN,
            median: NaN,
            mean: NaN,
            stdDev: NaN,
            nanCount,
            infCount
        };
    }

    const mean = sum / validCount;

    // Second pass: copy valid values to typed array and compute stdDev
    // Use the same typed array type as the source to preserve precision and save memory
    const TypedArrayCtor = data.constructor as new (length: number) => TypedArray;
    const validValues = new TypedArrayCtor(validCount);
    let sumSquaredDiff = 0;
    let idx = 0;

    for (let i = 0; i < len; i++) {
        const v = data[i];
        if (Number.isFinite(v)) {
            validValues[idx++] = v;
            const diff = v - mean;
            sumSquaredDiff += diff * diff;
        }
    }

    const stdDev = Math.sqrt(sumSquaredDiff / validCount);

    // Compute median using QuickSelect - O(n) instead of O(n log n)
    const mid = validCount >>> 1;
    let median: number;

    if (validCount % 2 === 0) {
        // For even count, need both middle values
        const lower = quickSelect(validValues, mid - 1, 0, validCount - 1);
        const upper = quickSelect(validValues, mid, 0, validCount - 1);
        median = (lower + upper) / 2;
    } else {
        median = quickSelect(validValues, mid, 0, validCount - 1);
    }

    return {
        min: round(min),
        max: round(max),
        median: round(median),
        mean: round(mean),
        stdDev: round(stdDev),
        nanCount,
        infCount
    };
};

/**
 * Computes statistical summary for all columns in a DataTable.
 *
 * For each column, calculates min, max, median, mean, standard deviation,
 * and counts of NaN/Infinity values. Useful for data validation and analysis.
 *
 * @param dataTable - The DataTable to analyze.
 * @returns Summary data with per-column statistics.
 *
 * @example
 * ```ts
 * const summary = computeSummary(dataTable);
 * console.log(summary.rowCount);
 * console.log(summary.columns['x'].mean);
 * console.log(summary.columns['opacity'].nanCount);
 * ```
 */
const computeSummary = (dataTable: DataTable): SummaryData => {
    const columns: Record<string, ColumnStats> = {};

    for (const column of dataTable.columns) {
        columns[column.name] = computeColumnStats(column);
    }

    return {
        version: 1,
        rowCount: dataTable.numRows,
        columns
    };
};

export { computeSummary, type ColumnStats, type SummaryData };
