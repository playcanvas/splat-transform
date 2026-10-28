import { OPACITY_EXCESS_COLUMN } from '../chunk';
import type { ChunkDataPool, ChunkSource, ReadRequest } from '../chunk';

/**
 * Render-side view of a source with excess opacity: the geometric layer's
 * opacity slot holds the LINEAR true opacity `A = sigmoid(opacity) +
 * opacity_excess` instead of a logit, so a single float reaches the GPU and
 * the project shader (compiled with `LINEAR_OPACITY`) can draw `A > 1` with
 * the over-unity profile. Every other layer and field passes through.
 *
 * Only for the renderer's own consumption: the result is not a valid source
 * for writers, which expect a logit there.
 *
 * @param src - The scene.
 * @param pool - Pool for the excess-column temporaries.
 * @returns The view, or null when the source has no excess column (render it as is).
 */
const withLinearOpacity = (src: ChunkSource, pool: ChunkDataPool): ChunkSource | null => {
    const other = src.meta.availableLayers.has('other') ? src.meta.layouts.other : undefined;
    const field = other?.fields[OPACITY_EXCESS_COLUMN];
    if (!other || !field) return null;
    const sw = other.stride >>> 2;
    const word = field.byteOffset >>> 2;

    const read = async (request: ReadRequest): Promise<void> => {
        const geo = request.geometric;
        if (!geo) {
            await src.read(request);
            return;
        }
        // read the excess into a temp unless the caller asked for `other` too
        const tmp = request.other ? null : pool.acquire('other', other, geo.count);
        try {
            await src.read(tmp ? { ...request, other: tmp } : request);
            const ex = new Float32Array((tmp ?? request.other)!.data, 0, geo.count * sw);
            const g = new Float32Array(geo.data, 0, geo.count * 8);
            for (let i = 0; i < geo.count; i++) {
                const e = ex[i * sw + word];
                g[i * 8 + 7] = 1 / (1 + Math.exp(-g[i * 8 + 7])) + (e > 0 ? e : 0);
            }
        } finally {
            tmp?.release();
        }
    };

    return { meta: src.meta, read, close: () => src.close() };
};

export { withLinearOpacity };
