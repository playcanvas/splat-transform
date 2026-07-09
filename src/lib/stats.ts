import { type ChunkDataPool, type ChunkSource, createChunkDataPool } from './chunk';
import { dataTableToChunkSource } from './compat/data-table';
import { DataTable } from './data-table';
import { computeSourceStats, type SourceStats } from './ops';

/**
 * Compute per-LOD, per-column statistics for splat data in a single streaming
 * pass — exact min/max/mean/stdDev/NaN/Inf, an approximate median, and a
 * 16-bin histogram per column, in columnar form (see {@link SourceStats}).
 *
 * Accepts either a `ChunkSource` (read chunk-by-chunk, constant memory) or a
 * legacy `DataTable` (bridged transiently; yields a single LOD). Values are the
 * raw, unbaked values — any pending transform is not applied.
 *
 * @param input - The source or table to analyze (left unchanged).
 * @param pool - Optional pool for the temporary read buffers; defaults to a fresh pool.
 * @returns The per-LOD statistics.
 *
 * @example
 * ```ts
 * const stats = await computeStats(dataTable);
 * const { columns, mean } = stats.lods[0];
 * console.log(mean[columns.indexOf('opacity')]);
 * ```
 */
const computeStats = async (input: ChunkSource | DataTable, pool?: ChunkDataPool): Promise<SourceStats> => {
    if (input instanceof DataTable) {
        const src = dataTableToChunkSource(input);
        try {
            return await computeSourceStats(src, pool ?? createChunkDataPool());
        } finally {
            await src.close();
        }
    }
    return computeSourceStats(input, pool ?? createChunkDataPool());
};

export { computeStats };
export type { LodStats, LodStatsData, SourceStats } from './ops';
