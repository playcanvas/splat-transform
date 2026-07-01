import { type ChunkDataPool, type ChunkSource, type ChunkSourceMetadata } from './chunk';
import { columnNamesFromMeta, dataTableToChunkSource, materializeToDataTable } from './compat/data-table';
import {
    filterByValueRows,
    filterBoxRows,
    filterNaNRows,
    filterSphereRows,
    filterSource,
    mapSource,
    reduceBandsSource
} from './ops';
import { processDataTable, type ProcessAction, type ProcessOptions } from './process';
import { fmtCount, logger, Transform } from './utils';

/**
 * The `ProcessAction` kinds `processSource` can apply natively on the streaming
 * chunk path. Anything else (decimate, lod, mortonOrder, the GPU voxel filters)
 * is applied by {@link processSourceBridged} as a `processDataTable` island.
 */
const SOURCE_ACTION_KINDS: ReadonlySet<ProcessAction['kind']> = new Set([
    'translate', 'rotate', 'scale',
    'filterNaN', 'filterByValue', 'filterBox', 'filterSphere', 'filterBands',
    'summary', 'info', 'param'
]);

/**
 * Whether every action can be applied on the chunk path by {@link processSource}.
 * @param actions - The actions to check.
 * @returns `true` if all are {@link SOURCE_ACTION_KINDS}.
 */
const canProcessSource = (actions: ProcessAction[]): boolean => actions.every(a => SOURCE_ACTION_KINDS.has(a.kind));

/**
 * Render a source's structural metadata as text for the `info` action — per-LOD
 * counts, SH bands, and the canonical column list, all from `meta` (no data read).
 * @param meta - The source metadata.
 * @returns A text block for `logger.output`.
 */
const formatSourceInfo = (meta: ChunkSourceMetadata): string => {
    const lods = meta.numLods > 1 ?
        `${meta.numLods} (${meta.lodCounts.map(c => fmtCount(c)).join(', ')})` :
        '1';
    return [
        '# File info',
        `gaussians: ${fmtCount(meta.numGaussians)}`,
        `lods: ${lods}`,
        `sh bands: ${meta.shBands}`,
        `columns: ${columnNamesFromMeta(meta).join(', ')}`
    ].join('\n');
};

/**
 * Apply a sequence of processing actions to a {@link ChunkSource}, the streaming
 * analog of `processDataTable`. Transforms compose lazily onto the pending
 * `meta.transform` (via {@link mapSource}); filters scan the source and return a
 * filtered view (via {@link filterSource}); `summary` materializes transiently
 * and reuses the `DataTable` summary (median needs every value, so there is no
 * streaming win — it is a diagnostic pass that leaves the data unchanged).
 *
 * Supports only the {@link SOURCE_ACTION_KINDS}; callers gate with
 * {@link canProcessSource} and fall back to `processDataTable` otherwise. Throws
 * on any unsupported action rather than silently dropping it.
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
            case 'summary': {
                // Reuse the DataTable summary verbatim (same formatting / logging).
                // Reflects the source's current, unbaked values — matching
                // processDataTable for every ordering except a transform-baking
                // filterByValue immediately followed by summary (a rare case).
                const dt = await materializeToDataTable(src, pool);
                await processDataTable(dt, [{ kind: 'summary' }]);
                break;
            }
            case 'info':
                // Structural metadata only (meta-level) — no materialization; the
                // source passes through unchanged.
                logger.output(formatSourceInfo(src.meta));
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

export { processSource, processSourceBridged, canProcessSource, SOURCE_ACTION_KINDS };
