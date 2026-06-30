import {
    type ChunkSource,
    type ChunkSourceMetadata,
    type ReadRequest
} from '../chunk';

/**
 * Stack N single-LOD sources into one structural multi-LOD source: output LOD
 * `i` is `sources[i]`. `read` dispatches by `request.lod` to the matching source
 * (read at its own LOD 0). `numGaussians` is LOD 0's count; `lodCounts[i]` is
 * `sources[i]`'s gaussian count.
 *
 * This is how per-detail-level inputs become one structural scene for the LOD
 * writer — multi-PLY `--lod` tags (one source per tagged level) or a DataTable
 * split by its `lod` column — replacing the old per-gaussian lod tag array. LOD
 * is a structural axis here; no source carries a per-gaussian LOD tag.
 *
 * All inputs must share layout (chunk size, SH bands, layers, extras, transform)
 * and be single-LOD; the stacked metadata is inherited from `sources[0]`.
 *
 * @param sources - The per-LOD single-LOD sources, in output-LOD order.
 * @returns A multi-LOD source dispatching by LOD to the inputs.
 */
const stackLods = (sources: ChunkSource[]): ChunkSource => {
    if (sources.length === 0) {
        throw new Error('stackLods: at least one source is required');
    }
    const ref = sources[0].meta;
    for (const s of sources) {
        if (s.meta.numLods !== 1) {
            throw new Error('stackLods: every input must be single-LOD');
        }
        if (s.meta.chunkSize !== ref.chunkSize || s.meta.shBands !== ref.shBands) {
            throw new Error('stackLods: inputs must share chunk size and SH band count');
        }
    }

    const lodCounts = sources.map(s => s.meta.numGaussians);
    const meta: ChunkSourceMetadata = {
        ...ref,
        numGaussians: lodCounts[0],
        numLods: sources.length,
        lodCounts,
        numChunks: sources.map(s => s.meta.numChunks[0])
    };

    const pick = (lod: number): ChunkSource => {
        const s = sources[lod];
        if (!s) {
            throw new Error(`stackLods: lod ${lod} out of range (numLods ${sources.length})`);
        }
        return s;
    };

    // Dispatch by LOD; the chosen source serves the request (chunk or gather) at
    // its own LOD 0.
    const read = (request: ReadRequest): Promise<void> => {
        return pick(request.lod ?? 0).read({ ...request, lod: 0 });
    };

    return {
        meta,
        read,
        close: async () => {
            await Promise.all(sources.map(s => s.close()));
        }
    };
};

export { stackLods };
