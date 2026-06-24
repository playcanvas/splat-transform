import {
    InMemoryChunkSource,
    type ChunkData,
    type ChunkLayer,
    type ChunkReadRequest,
    type ChunkSource
} from '../source';

/**
 * Reorder a resident source's gaussians by a permutation, as a lazy view.
 *
 * Output gaussian `i` is parent gaussian `order[i]`. The parent must be a
 * resident {@link InMemoryChunkSource} (e.g. from `compact()`), so each output
 * row is a direct buffer copy from the parent — no re-decode and no extra
 * scene-sized allocation. Layout, layers, transform and counts are unchanged;
 * only row order differs.
 *
 * Single-LOD only — ordering is a LOD-0 concept (LOD partitioning is separate).
 *
 * @param parent - Resident source to reorder.
 * @param order - Permutation of `[0, numGaussians)`; `order[i]` is the parent row placed at output row `i`.
 * @returns A derived source serving the reordered gaussians chunk-by-chunk.
 */
const permuteSource = (parent: ChunkSource, order: Uint32Array): ChunkSource => {
    if (!(parent instanceof InMemoryChunkSource)) {
        throw new Error('permuteSource: parent must be a resident InMemoryChunkSource (compact() it first)');
    }
    if (parent.meta.numLods !== 1) {
        throw new Error('permuteSource: only single-LOD sources are supported');
    }
    if (order.length !== parent.meta.numGaussians) {
        throw new Error(`permuteSource: order length ${order.length} != numGaussians ${parent.meta.numGaussians}`);
    }

    const meta = parent.meta; // permutation preserves layout / layers / transform / counts
    const { chunkSize, numGaussians } = meta;

    const read = (request: ChunkReadRequest): Promise<void> => {
        const base = request.chunkIndex * chunkSize;
        const count = Math.min(chunkSize, numGaussians - base);

        const fill = (cd: ChunkData | undefined, layer: ChunkLayer): void => {
            if (!cd) return;
            parent.gatherRows(layer, 0, order, base, count, cd.data);
        };

        fill(request.position, 'position');
        fill(request.geometric, 'geometric');
        fill(request.color, 'color');
        fill(request.other, 'other');
        return Promise.resolve();
    };

    return { meta, read, close: () => parent.close() };
};

export { permuteSource };
