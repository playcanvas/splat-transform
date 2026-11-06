import { OPACITY_LINEAR_COLUMN } from '../chunk';
import type { ChunkDataPool, ChunkSource, ReadRequest } from '../chunk';

/**
 * Render-side view of a source with an `opacity_linear` column: the geometric
 * layer's opacity slot holds that LINEAR true opacity `A` instead of a logit
 * (the column always wins over `opacity`), so a single float reaches the GPU
 * and the project shader (compiled with `LINEAR_OPACITY`) can draw `A > 1`
 * with the over-unity profile. Every other layer and field passes through.
 *
 * Only for the renderer's own consumption: the result is not a valid source
 * for writers, which expect a logit there.
 *
 * @param src - The scene.
 * @param pool - Pool for the column temporaries.
 * @param maxOpacity - Clamp `A` to this (1 draws every splat at most opaque).
 * @returns The view, or null when the source has no `opacity_linear` column (render it as is).
 */
const withLinearOpacity = (src: ChunkSource, pool: ChunkDataPool, maxOpacity = Infinity): ChunkSource | null => {
    const other = src.meta.availableLayers.has('other') ? src.meta.layouts.other : undefined;
    const field = other?.fields[OPACITY_LINEAR_COLUMN];
    if (!other || !field) return null;
    const sw = other.stride >>> 2;
    const word = field.byteOffset >>> 2;

    const read = async (request: ReadRequest): Promise<void> => {
        const geo = request.geometric;
        if (!geo) {
            await src.read(request);
            return;
        }
        // read the column into a temp unless the caller asked for `other` too
        const tmp = request.other ? null : pool.acquire('other', other, geo.count);
        try {
            await src.read(tmp ? { ...request, other: tmp } : request);
            const lin = new Float32Array((tmp ?? request.other)!.data, 0, geo.count * sw);
            const g = new Float32Array(geo.data, 0, geo.count * 8);
            for (let i = 0; i < geo.count; i++) {
                const a = lin[i * sw + word];
                g[i * 8 + 7] = a > 0 ? Math.min(a, maxOpacity) : 0;
            }
        } finally {
            tmp?.release();
        }
    };

    return { meta: src.meta, read, close: () => src.close() };
};

export { withLinearOpacity };
