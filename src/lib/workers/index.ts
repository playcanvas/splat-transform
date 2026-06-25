import { WorkerQueue } from './worker-queue';
import { Column, DataTable, type TypedArray } from '../data-table/data-table';
import { quantize1dColumns, type QuantizedColumns } from '../spatial/quantize-1d-core';

/**
 * Typed client wrappers around WorkerQueue tasks. These own the marshalling
 * between rich types (DataTable) and the transferable shapes the worker
 * protocol uses, keeping call sites one-liners.
 */

/**
 * quantize1d over raw named columns, preferring a worker thread. This is the
 * column-native form (no DataTable); see {@link runQuantize1d} for the
 * DataTable wrapper used by the legacy writer.
 *
 * @param columns - Input columns pooled into 1D.
 * @param k - Number of codebook entries.
 * @param alpha - Density weight exponent.
 * @returns Centroids (k Float32 codebook) and per-column Uint8 labels.
 * @ignore
 */
const runQuantize1dColumns = async (columns: { name: string, data: TypedArray }[], k?: number, alpha?: number): Promise<QuantizedColumns> => {
    if (WorkerQueue.isInline) {
        // zero-copy: no point marshalling when running on this thread anyway
        return quantize1dColumns(columns, k, alpha);
    }

    // compact copies: column data may be views into larger shared buffers,
    // and the originals must remain usable after the transfer
    const copies = columns.map(c => ({ name: c.name, data: c.data.slice() }));
    return await WorkerQueue.run('quantize1d', { columns: copies, k, alpha }, copies.map(c => c.data.buffer as ArrayBuffer));
};

/**
 * quantize1d, preferring a worker thread (DataTable wrapper around
 * {@link runQuantize1dColumns}).
 *
 * @param dataTable - Input data table whose columns are pooled into 1D.
 * @param k - Number of codebook entries.
 * @param alpha - Density weight exponent.
 * @returns Centroids and labels (see quantize1d).
 * @ignore
 */
const runQuantize1d = async (dataTable: DataTable, k?: number, alpha?: number): Promise<{ centroids: DataTable, labels: DataTable }> => {
    const { centroids, labels } = await runQuantize1dColumns(
        dataTable.columns.map(c => ({ name: c.name, data: c.data })), k, alpha
    );

    return {
        centroids: new DataTable([new Column('data', centroids)]),
        labels: new DataTable(labels.map(c => new Column(c.name, c.data)))
    };
};

/**
 * Lossless WebP encode, preferring a worker thread.
 *
 * @param rgba - RGBA pixel data. Transferred: unusable after this call.
 * @param width - Image width in pixels.
 * @param height - Image height in pixels.
 * @returns The encoded WebP data.
 * @ignore
 */
const runEncodeWebp = (rgba: Uint8Array, width: number, height: number): Promise<Uint8Array> => {
    return WorkerQueue.run('encodeWebp', { rgba, width, height }, [rgba.buffer as ArrayBuffer]);
};

export { WorkerQueue, runQuantize1d, runQuantize1dColumns, runEncodeWebp };
