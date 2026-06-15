import { WorkerQueue } from './worker-queue';
import { Column, DataTable } from '../data-table/data-table';
import { quantize1d } from '../spatial/quantize-1d';

/**
 * Typed client wrappers around WorkerQueue tasks. These own the marshalling
 * between rich types (DataTable) and the transferable shapes the worker
 * protocol uses, keeping call sites one-liners.
 */

/**
 * quantize1d, preferring a worker thread.
 *
 * @param dataTable - Input data table whose columns are pooled into 1D.
 * @param k - Number of codebook entries.
 * @param alpha - Density weight exponent.
 * @returns Centroids and labels (see quantize1d).
 * @ignore
 */
const runQuantize1d = async (dataTable: DataTable, k?: number, alpha?: number): Promise<{ centroids: DataTable, labels: DataTable }> => {
    if (WorkerQueue.isInline) {
        // zero-copy: no point marshalling when running on this thread anyway
        return quantize1d(dataTable, k, alpha);
    }

    // compact copies: column data may be views into larger shared buffers,
    // and the originals must remain usable after the transfer
    const columns = dataTable.columns.map(c => ({ name: c.name, data: c.data.slice() }));
    const result = await WorkerQueue.run('quantize1d', { columns, k, alpha }, columns.map(c => c.data.buffer as ArrayBuffer));

    return {
        centroids: new DataTable([new Column('data', result.centroids)]),
        labels: new DataTable(result.labels.map(c => new Column(c.name, c.data)))
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

export { WorkerQueue, runQuantize1d, runEncodeWebp };
