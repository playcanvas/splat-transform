import {
    type ChunkReadRequest,
    type ChunkSource,
    type ChunkSourceMetadata
} from '../source';

/**
 * Gather a source's gaussians by an ordered index list, as a lazy view.
 *
 * Output gaussian `i` is parent gaussian `order[i]`. The parent must support
 * random-access gather ({@link ChunkSource.readRows}) — a resident
 * `InMemoryChunkSource` (direct buffer copy, e.g. from `compact()`) or a
 * fixed-stride file source like PLY (byte-range reads per row). Layout, layers
 * and transform are inherited. A lazy combinator without `readRows` must be
 * `compact()`-ed first.
 *
 * `order` is a full-length permutation (output count == parent count) or a
 * shorter ordered subset (output count == `order.length`) — the source-native
 * equivalent of the LOD writer's per-unit ordered-subset gather. Indices may
 * repeat and need not be sorted; the only constraint is `order.length` cannot
 * exceed the parent's gaussian count.
 *
 * Single-LOD only — ordering is a LOD-0 concept (LOD partitioning is separate).
 *
 * @param parent - `readRows`-capable source to gather from.
 * @param order - Ordered parent row indices; `order[i]` is placed at output row `i`. Length must be `<= parent.meta.numGaussians`.
 * @returns A derived source serving the gathered gaussians chunk-by-chunk.
 */
const permuteSource = (parent: ChunkSource, order: Uint32Array): ChunkSource => {
    if (!parent.readRows) {
        throw new Error('permuteSource: parent must support random-access gather (readRows) — compact() a lazy source first');
    }
    if (parent.meta.numLods !== 1) {
        throw new Error('permuteSource: only single-LOD sources are supported');
    }
    if (order.length > parent.meta.numGaussians) {
        throw new Error(`permuteSource: order length ${order.length} exceeds numGaussians ${parent.meta.numGaussians}`);
    }

    const readRows = parent.readRows.bind(parent);
    const chunkSize = parent.meta.chunkSize;
    const outCount = order.length;
    // Gather inherits layout / layers / transform; only the counts shrink to the
    // gathered size (a full-length order leaves them unchanged).
    const meta: ChunkSourceMetadata = {
        ...parent.meta,
        numGaussians: outCount,
        numLods: 1,
        lodCounts: [outCount],
        numChunks: [Math.ceil(outCount / chunkSize)]
    };

    const read = (request: ChunkReadRequest): Promise<void> => {
        const base = request.chunkIndex * chunkSize;
        return readRows({
            indices: order,
            indexOffset: base,
            count: Math.min(chunkSize, outCount - base),
            position: request.position,
            geometric: request.geometric,
            color: request.color,
            other: request.other
        });
    };

    return { meta, read, close: () => parent.close() };
};

export { permuteSource };
