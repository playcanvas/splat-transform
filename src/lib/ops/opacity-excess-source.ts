import { OPACITY_EXCESS_COLUMN, otherLayout } from '../chunk';
import type { ChunkDataPool, ChunkLayer, ChunkSource, ChunkSourceMetadata, ReadRequest } from '../chunk';

/**
 * Present a source with an `opacity_excess` column, as a lazy view: sources
 * that already carry one are returned unchanged; for the rest the column is
 * appended to the `other` layer and reads as 0 (nothing owed), which is exactly
 * what its absence means. Decimation runs its input through this so every
 * merge can read and write the column without special-casing its absence, and
 * its output always carries it.
 *
 * Only the `other` layer changes: when requested, the parent's own `other`
 * columns (if any) are read into a temp and repacked with the new word last.
 *
 * @param src - Parent source.
 * @param pool - Pool for the parent's `other` temporaries; `chunkSize` must match the parent's.
 * @returns A source whose `other` layer includes `opacity_excess`.
 */
const withOpacityExcess = (src: ChunkSource, pool: ChunkDataPool): ChunkSource => {
    const parentOther = src.meta.availableLayers.has('other') ? src.meta.layouts.other : undefined;
    if (parentOther?.fields[OPACITY_EXCESS_COLUMN]) return src;

    const extraColumns = [...src.meta.extraColumns, { name: OPACITY_EXCESS_COLUMN, type: 'float32' as const }];
    const layout = otherLayout(extraColumns);
    const meta: ChunkSourceMetadata = {
        ...src.meta,
        extraColumns,
        availableLayers: new Set<ChunkLayer>([...src.meta.availableLayers, 'other']),
        layouts: { ...src.meta.layouts, other: layout }
    };

    const inSw = parentOther ? parentOther.stride >>> 2 : 0;
    const outSw = layout.stride >>> 2;

    const read = async (request: ReadRequest): Promise<void> => {
        const out = request.other;
        if (!out) {
            await src.read(request);
            return;
        }
        const count = out.count;
        const dst = new Uint32Array(out.data, 0, count * outSw);
        if (!parentOther) {
            // the new column is the only one
            await src.read({ ...request, other: undefined });
            dst.fill(0);
            return;
        }
        const tmp = pool.acquire('other', parentOther, count);
        try {
            await src.read({ ...request, other: tmp });
            const srcU = new Uint32Array(tmp.data, 0, count * inSw);
            for (let i = 0; i < count; i++) {
                dst.set(srcU.subarray(i * inSw, (i + 1) * inSw), i * outSw);
                dst[i * outSw + inSw] = 0; // +0.0f
            }
        } finally {
            tmp.release();
        }
    };

    return { meta, read, close: () => src.close() };
};

export { withOpacityExcess };
