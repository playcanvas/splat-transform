import {
    type ChunkLayer,
    type ChunkSource,
    type ChunkSourceMetadata,
    type ReadRequest
} from '../chunk';

const LAYERS: ChunkLayer[] = ['position', 'geometric', 'color', 'other'];

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
    const layerKey = (m: ChunkSourceMetadata) => LAYERS.filter(l => m.availableLayers.has(l)).join(',');
    const extraKey = (m: ChunkSourceMetadata) => m.extraColumns.map(e => `${e.name}:${e.type}`).join(',');
    for (const s of sources) {
        if (s.meta.numLods !== 1) {
            throw new Error('stackLods: every input must be single-LOD');
        }
        if (s.meta.chunkSize !== ref.chunkSize || s.meta.shBands !== ref.shBands) {
            throw new Error('stackLods: inputs must share chunk size and SH band count');
        }
        if (layerKey(s.meta) !== layerKey(ref)) {
            throw new Error(`stackLods: available-layer mismatch between LOD levels ([${layerKey(s.meta)}] vs [${layerKey(ref)}])`);
        }
        if (extraKey(s.meta) !== extraKey(ref)) {
            throw new Error('stackLods: extra-column mismatch between LOD levels');
        }
        // The stacked metadata (including `transform`) is inherited from
        // sources[0] and the writer bakes ONE delta over every level's reads —
        // a per-level transform mismatch would silently misplace levels.
        //
        // FOLLOW-UP (deliberate restriction, not a design endpoint): LOD levels
        // can come from separate PLY inputs carrying arbitrary per-input
        // transforms, and the API shouldn't disallow that. Likely direction:
        // store a transform per LOD in ChunkSourceMetadata and teach
        // bakeTransform / geometry consumers to select by request.lod, so
        // callers no longer need to pre-bake diverged levels (as the CLI LOD
        // path does today).
        if (!s.meta.transform.equals(ref.transform)) {
            throw new Error('stackLods: transform mismatch between LOD levels — bake to a common space first');
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
