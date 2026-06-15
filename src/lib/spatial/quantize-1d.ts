import { quantize1dColumns } from './quantize-1d-core';
import { Column, DataTable } from '../data-table/data-table';

/**
 * Optimal 1D quantization using dynamic programming on a histogram.
 * DataTable wrapper around quantize1dColumns (see quantize-1d-core.ts for
 * the algorithm).
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
    const { centroids, labels } = quantize1dColumns(
        dataTable.columns.map(c => ({ name: c.name, data: c.data })),
        k,
        alpha
    );

    return {
        centroids: new DataTable([new Column('data', centroids)]),
        labels: new DataTable(labels.map(c => new Column(c.name, c.data)))
    };
};

export { quantize1d };
