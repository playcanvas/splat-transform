import { materializeToDataTable } from './compat/data-table';
import {
    filterByValueRows,
    filterBoxRows,
    filterNaNRows,
    filterSphereRows,
    filterSource,
    mapSource
} from './ops';
import { processDataTable, type ProcessAction } from './process';
import { type ChunkDataPool, type ChunkSource } from './source';
import { Transform } from './utils';

/**
 * The `ProcessAction` kinds `processSource` can apply on the streaming chunk
 * path. Anything else (decimate, lod, mortonOrder, filterBands, the GPU voxel
 * filters) still routes through the `DataTable` `processDataTable` pipeline.
 */
const SOURCE_ACTION_KINDS: ReadonlySet<ProcessAction['kind']> = new Set([
    'translate', 'rotate', 'scale',
    'filterNaN', 'filterByValue', 'filterBox', 'filterSphere',
    'summary', 'param'
]);

/**
 * Whether every action can be applied on the chunk path by {@link processSource}.
 * @param actions - The actions to check.
 * @returns `true` if all are {@link SOURCE_ACTION_KINDS}.
 */
const canProcessSource = (actions: ProcessAction[]): boolean => actions.every(a => SOURCE_ACTION_KINDS.has(a.kind));

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
            case 'summary': {
                // Reuse the DataTable summary verbatim (same formatting / logging).
                // Reflects the source's current, unbaked values — matching
                // processDataTable for every ordering except a transform-baking
                // filterByValue immediately followed by summary (a rare case).
                const dt = await materializeToDataTable(src, pool);
                await processDataTable(dt, [{ kind: 'summary' }]);
                break;
            }
            case 'param':
                break; // generator params: no-op here, as in processDataTable
            default:
                throw new Error(`processSource: unsupported action '${action.kind}'`);
        }
    }

    return src;
};

export { processSource, canProcessSource, SOURCE_ACTION_KINDS };
