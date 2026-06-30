import {
    type ChunkReadRequest,
    type ChunkSource,
    type ChunkSourceMetadata
} from '../source';

/**
 * Gather a source's gaussians by an ordered index list, as a lazy view.
 *
 * Output gaussian `i` is parent gaussian `order[i]` of LOD `opts.lod` (default
 * 0). The parent must support random-access gather ({@link ChunkSource.readRows})
 * — a resident `InMemoryChunkSource` (direct buffer copy, e.g. from `compact()`)
 * or a fixed-stride file source like PLY (byte-range reads per row). Layout,
 * layers and transform are inherited. A lazy combinator without `readRows` must
 * be `compact()`-ed first.
 *
 * `order` is a full-length permutation (output count == the LOD's count) or a
 * shorter ordered subset (output count == `order.length`) — the source-native
 * equivalent of the LOD writer's per-unit ordered-subset gather. Indices are
 * relative to the chosen LOD, may repeat and need not be sorted; the only
 * constraint is `order.length` cannot exceed that LOD's gaussian count.
 *
 * The output is single-LOD (the gathered subset). To gather from a structural
 * LOD of a multi-LOD source, pass `opts.lod` — the LOD writer gathers each output
 * unit from its level this way (replacing the old per-gaussian lod tag).
 *
 * @param parent - `readRows`-capable source to gather from.
 * @param order - Ordered row indices within the chosen LOD; `order[i]` is placed at output row `i`.
 * @param opts - Gather options.
 * @param opts.lod - The parent LOD to gather from (default 0).
 * @returns A derived source serving the gathered gaussians chunk-by-chunk.
 */
const permuteSource = (parent: ChunkSource, order: Uint32Array, opts?: { lod?: number }): ChunkSource => {
    if (!parent.readRows) {
        throw new Error('permuteSource: parent must support random-access gather (readRows) — compact() a lazy source first');
    }
    const lod = opts?.lod ?? 0;
    if (lod < 0 || lod >= parent.meta.numLods) {
        throw new Error(`permuteSource: lod ${lod} out of range (numLods ${parent.meta.numLods})`);
    }
    const parentCount = parent.meta.lodCounts[lod];
    if (order.length > parentCount) {
        throw new Error(`permuteSource: order length ${order.length} exceeds lod ${lod} count ${parentCount}`);
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
            lod,
            position: request.position,
            geometric: request.geometric,
            color: request.color,
            other: request.other
        });
    };

    return { meta, read, close: () => parent.close() };
};

export { permuteSource };
