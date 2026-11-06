import { OPACITY_LINEAR_COLUMN, otherLayout } from '../chunk';
import type { ChunkDataPool, ChunkLayer, ChunkSource, ChunkSourceMetadata, ReadRequest } from '../chunk';

/**
 * Present a source with an `opacity_linear` column, as a lazy view: sources
 * that already carry one are returned unchanged; for the rest the column is
 * appended to the `other` layer and reads as `sigmoid(opacity)`, which is
 * exactly what its absence means. Decimation runs its input through this so
 * every merge can read and write the column without special-casing its
 * absence, and its output always carries it.
 *
 * Only the `other` layer changes: when requested, the opacity logits are read
 * (into a temp unless the caller asked for `geometric` too) and the parent's
 * own `other` columns (if any) are repacked with the new word last.
 *
 * @param src - Parent source.
 * @param pool - Pool for the temporaries; `chunkSize` must match the parent's.
 * @returns A source whose `other` layer includes `opacity_linear`.
 */
const withOpacityLinear = (src: ChunkSource, pool: ChunkDataPool): ChunkSource => {
    const parentOther = src.meta.availableLayers.has('other') ? src.meta.layouts.other : undefined;
    if (parentOther?.fields[OPACITY_LINEAR_COLUMN]) return src;

    const extraColumns = [...src.meta.extraColumns, { name: OPACITY_LINEAR_COLUMN, type: 'float32' as const }];
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
        const geoTmp = request.geometric ? null : pool.acquire('geometric', src.meta.layouts.geometric!, count);
        const otherTmp = parentOther ? pool.acquire('other', parentOther, count) : null;
        try {
            await src.read({
                ...request,
                geometric: request.geometric ?? geoTmp!,
                other: otherTmp ?? undefined
            });
            const geo = new Float32Array((request.geometric ?? geoTmp!).data, 0, count * 8);
            const dst = new Uint32Array(out.data, 0, count * outSw);
            const dstF = new Float32Array(out.data, 0, count * outSw);
            const srcU = otherTmp ? new Uint32Array(otherTmp.data, 0, count * inSw) : null;
            for (let i = 0; i < count; i++) {
                if (srcU) dst.set(srcU.subarray(i * inSw, (i + 1) * inSw), i * outSw);
                dstF[i * outSw + inSw] = 1 / (1 + Math.exp(-geo[i * 8 + 7]));
            }
        } finally {
            geoTmp?.release();
            otherTmp?.release();
        }
    };

    return { meta, read, close: () => src.close() };
};

export { withOpacityLinear };
