import { type ChunkDataPool, type ChunkSource } from './chunk';
import { dataTableToChunkSource, materializeToDataTable } from './compat/data-table';
import {
    computeSourceStats,
    filterByValueRows,
    filterBoxRows,
    filterNaNRows,
    filterSphereRows,
    filterSource,
    mapSource,
    reduceBandsSource
} from './ops';
import { processDataTable, type ProcessAction, type ProcessOptions } from './process';
import { formatSourceInfo, formatSourceStats } from './source-info';
import { logger, Transform } from './utils';

/**
 * The `ProcessAction` kinds `processSource` can apply natively on the streaming
 * chunk path. Anything else (decimate, lod, mortonOrder, the GPU voxel filters)
 * is applied by {@link processSourceBridged} as a `processDataTable` island.
 */
const SOURCE_ACTION_KINDS: ReadonlySet<ProcessAction['kind']> = new Set([
    'translate', 'rotate', 'scale',
    'filterNaN', 'filterByValue', 'filterBox', 'filterSphere', 'filterBands',
    'stats', 'info', 'param'
]);

/**
 * Apply a sequence of processing actions to a {@link ChunkSource}, the streaming
 * analog of `processDataTable`. Transforms compose lazily onto the pending
 * `meta.transform` (via {@link mapSource}); filters scan the source and return a
 * filtered view (via {@link filterSource}); `stats` streams a one-pass
 * per-LOD accumulation (via {@link computeSourceStats}) — a diagnostic pass
 * that leaves the data unchanged.
 *
 * Supports only the {@link SOURCE_ACTION_KINDS}; throws on any unsupported
 * action rather than silently dropping it ({@link processSourceBridged} owns
 * the `processDataTable` fallback for everything else).
 *
 * @param source - The input source.
 * @param actions - Actions to apply in order.
 * @param pool - Pool for the filter passes' temporary read buffers.
 * @returns The processed source (a view chain over `source`).
 */
const processSource = async (
    source: ChunkSource,
    actions: ProcessAction[],
    pool: ChunkDataPool
): Promise<ChunkSource> => {
    let src = source;

    for (const action of actions) {
        switch (action.kind) {
            case 'translate':
                src = mapSource(src, new Transform(action.value));
                break;
            case 'rotate':
                src = mapSource(src, new Transform().fromEulers(action.value.x, action.value.y, action.value.z));
                break;
            case 'scale':
                src = mapSource(src, new Transform(undefined, undefined, action.value));
                break;
            case 'filterNaN':
                src = filterSource(src, await filterNaNRows(src, pool), pool);
                break;
            case 'filterByValue':
                src = filterSource(src, await filterByValueRows(src, pool, action), pool);
                break;
            case 'filterBox':
                src = filterSource(src, await filterBoxRows(src, pool, action), pool);
                break;
            case 'filterSphere':
                src = filterSource(src, await filterSphereRows(src, pool, action), pool);
                break;
            case 'filterBands':
                src = reduceBandsSource(src, action.value, pool);
                break;
            case 'stats':
                // One streaming pass per LOD; no materialization. Reflects the
                // source's current, unbaked values — matching processDataTable
                // for every ordering except a transform-baking filterByValue
                // immediately followed by stats (a rare case).
                logger.output(formatSourceStats(src.meta, await computeSourceStats(src, pool), action.format));
                break;
            case 'info':
                // Structural metadata only (meta-level) — no materialization; the
                // source passes through unchanged.
                logger.output(formatSourceInfo(src.meta, action.format));
                break;
            case 'param':
                break; // generator params: no-op here, as in processDataTable
            default:
                throw new Error(`processSource: unsupported action '${action.kind}'`);
        }
    }

    return src;
};

/**
 * Apply an ordered action list to a {@link ChunkSource}, streaming the
 * chunk-native runs and bridging only the DataTable-only runs. Consecutive
 * actions are grouped into maximal same-mode runs (order preserved): a
 * chunk-native run ({@link SOURCE_ACTION_KINDS}) goes through {@link processSource};
 * a DataTable-only run (decimate, mortonOrder, the GPU voxel filters, …)
 * materializes once, runs `processDataTable`, and re-bridges to a source via
 * `dataTableToChunkSource`. So the not-yet-chunked ops do their work inline as
 * islands and everything around them keeps streaming.
 *
 * @param source - The input source (consumed; the returned source owns it).
 * @param actions - Actions to apply in order.
 * @param pool - Pool for the chunk-native passes and the bridge's chunk size.
 * @param options - Process options (e.g. `createDevice` for the GPU islands).
 * @returns The processed source.
 */
const processSourceBridged = async (
    source: ChunkSource,
    actions: ProcessAction[],
    pool: ChunkDataPool,
    options?: ProcessOptions
): Promise<ChunkSource> => {
    let src = source;
    let i = 0;
    while (i < actions.length) {
        const chunkNative = SOURCE_ACTION_KINDS.has(actions[i].kind);
        let j = i + 1;
        while (j < actions.length && SOURCE_ACTION_KINDS.has(actions[j].kind) === chunkNative) {
            j++;
        }
        const run = actions.slice(i, j);
        if (chunkNative) {
            src = await processSource(src, run, pool);
        } else {
            // DataTable island: materialize the current (streaming) source, apply
            // the run on the table, and re-bridge back to a source to keep going.
            const dt = await materializeToDataTable(src, pool);
            await src.close();
            src = dataTableToChunkSource(await processDataTable(dt, run, options), pool.chunkSize);
        }
        i = j;
    }
    return src;
};

export { processSource, processSourceBridged, SOURCE_ACTION_KINDS };
