import {
    type ChunkSource,
    type ChunkSourceMetadata,
    type ReadRequest
} from '../chunk';

/**
 * Gather a source's gaussians by an ordered index list, as a lazy view.
 *
 * Output gaussian `i` is parent gaussian `order[i]` of LOD `opts.lod` (default
 * 0), gathered from the parent via {@link ChunkSource.read} with an index list.
 * Layout, layers and transform are inherited.
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
 * @param parent - Source to gather from.
 * @param order - Ordered row indices within the chosen LOD; `order[i]` is placed at output row `i`.
 * @param opts - Gather options.
 * @param opts.lod - The parent LOD to gather from (default 0).
 * @returns A derived source serving the gathered gaussians chunk-by-chunk.
 */
const permuteSource = (parent: ChunkSource, order: Uint32Array, opts?: { lod?: number }): ChunkSource => {
    const lod = opts?.lod ?? 0;
    if (lod < 0 || lod >= parent.meta.numLods) {
        throw new Error(`permuteSource: lod ${lod} out of range (numLods ${parent.meta.numLods})`);
    }
    const parentCount = parent.meta.lodCounts[lod];
    if (order.length > parentCount) {
        throw new Error(`permuteSource: order length ${order.length} exceeds lod ${lod} count ${parentCount}`);
    }

    const parentRead = parent.read.bind(parent);
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

    const read = (request: ReadRequest): Promise<void> => {
        if ('indices' in request) {
            // Gather of a gather: compose the permutation — output row j is parent
            // row order[request.indices[indexOffset + j]].
            const { indices, indexOffset, count } = request;
            const mapped = new Uint32Array(count);
            for (let j = 0; j < count; j++) mapped[j] = order[indices[indexOffset + j]];
            return parentRead({
                indices: mapped,
                indexOffset: 0,
                count,
                lod,
                position: request.position,
                geometric: request.geometric,
                color: request.color,
                other: request.other
            });
        }
        const base = request.chunkIndex * chunkSize;
        return parentRead({
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
