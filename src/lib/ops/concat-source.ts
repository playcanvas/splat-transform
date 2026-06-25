import {
    type ChunkData,
    type ChunkDataPool,
    type ChunkLayer,
    type ChunkReadRequest,
    type ChunkSource,
    type ChunkSourceMetadata
} from '../source';

const LAYERS: ChunkLayer[] = ['position', 'geometric', 'color', 'other'];

// Mutable form of a read request, built layer-by-layer before passing to a
// source's `read` (whose `ChunkReadRequest` fields are readonly).
type MutableReadRequest = { chunkIndex: number; lod: number } & { [L in ChunkLayer]?: ChunkData };

/**
 * Concatenate several sources end-to-end into one, as a lazy view.
 *
 * Output gaussians are the inputs' gaussians in order: all of `sources[0]`, then
 * all of `sources[1]`, and so on. Every source must agree on layout (chunk size,
 * SH bands, available layers, extra columns) and on the pending coordinate-space
 * **transform** — concatenating data in mismatched spaces is silently wrong, so
 * a transform mismatch throws (the caller must bake to a common space first).
 *
 * Single-LOD only. Reads stitch contiguous row ranges: an output chunk is filled
 * by block-copying the overlapping span out of each contributing source chunk
 * (order is preserved, so each overlap is a contiguous byte range — one
 * `set()` per layer, not a per-row gather). Source chunks are read on demand;
 * peak extra memory is one source chunk-set of temporaries.
 *
 * @param sources - The sources to concatenate (at least one).
 * @param pool - Pool for the temporary source read buffers; `chunkSize` must match the sources'.
 * @returns A derived source serving the concatenated gaussians chunk-by-chunk.
 */
const concatSource = (sources: ChunkSource[], pool: ChunkDataPool): ChunkSource => {
    if (sources.length === 0) {
        throw new Error('concatSource: at least one source is required');
    }

    const ref = sources[0].meta;
    const layerKey = (m: ChunkSourceMetadata) => LAYERS.filter(l => m.availableLayers.has(l)).join(',');
    const extraKey = (m: ChunkSourceMetadata) => m.extraColumns.map(e => `${e.name}:${e.type}`).join(',');
    const refLayers = layerKey(ref);
    const refExtras = extraKey(ref);

    for (let i = 1; i < sources.length; i++) {
        const m = sources[i].meta;
        if (m.numLods !== 1 || ref.numLods !== 1) {
            throw new Error('concatSource: only single-LOD sources are supported');
        }
        if (m.chunkSize !== ref.chunkSize) {
            throw new Error(`concatSource: chunkSize mismatch (${m.chunkSize} vs ${ref.chunkSize})`);
        }
        if (m.shBands !== ref.shBands) {
            throw new Error(`concatSource: SH band mismatch (${m.shBands} vs ${ref.shBands})`);
        }
        if (layerKey(m) !== refLayers) {
            throw new Error(`concatSource: available-layer mismatch ([${layerKey(m)}] vs [${refLayers}])`);
        }
        if (extraKey(m) !== refExtras) {
            throw new Error('concatSource: extra-column mismatch between sources');
        }
        if (!m.transform.equals(ref.transform)) {
            throw new Error('concatSource: transform mismatch between sources — bake to a common space first');
        }
    }

    const S = ref.chunkSize;
    // Per-source gaussian counts and the output-row offset each source begins at.
    const counts = sources.map(s => s.meta.numGaussians);
    const starts: number[] = [];
    let total = 0;
    for (const c of counts) {
        starts.push(total);
        total += c;
    }

    const meta: ChunkSourceMetadata = {
        ...ref,
        numGaussians: total,
        numLods: 1,
        lodCounts: [total],
        numChunks: [Math.ceil(total / S)]
    };

    const read = async (request: ChunkReadRequest): Promise<void> => {
        const outBase = request.chunkIndex * S;
        const outCount = Math.min(S, total - outBase);

        const wanted: { layer: ChunkLayer; out: ChunkData }[] = [];
        if (request.position) wanted.push({ layer: 'position', out: request.position });
        if (request.geometric) wanted.push({ layer: 'geometric', out: request.geometric });
        if (request.color) wanted.push({ layer: 'color', out: request.color });
        if (request.other) wanted.push({ layer: 'other', out: request.other });
        if (wanted.length === 0) return;

        // Find the source containing the first output row of this chunk.
        let si = 0;
        while (si < sources.length && starts[si] + counts[si] <= outBase) si++;

        let filled = 0;
        while (filled < outCount) {
            const globalRow = outBase + filled;
            const localRow = globalRow - starts[si];
            const srcChunk = Math.floor(localRow / S);
            const srcChunkStart = srcChunk * S;
            const srcChunkCount = Math.min(S, counts[si] - srcChunkStart);
            const rowInChunk = localRow - srcChunkStart;
            // Rows we can take from this source chunk without overrunning it or
            // the remaining output room.
            const take = Math.min(srcChunkCount - rowInChunk, outCount - filled);

            const temps: { out: ChunkData; tmp: ChunkData }[] = [];
            const req: MutableReadRequest = { chunkIndex: srcChunk, lod: 0 };
            for (const { layer, out } of wanted) {
                const tmp = pool.acquire(layer, ref.layouts[layer]!, srcChunkCount);
                req[layer] = tmp;
                temps.push({ out, tmp });
            }
            await sources[si].read(req);

            for (const { out, tmp } of temps) {
                const stride = tmp.stride;
                const srcU8 = new Uint8Array(tmp.data, rowInChunk * stride, take * stride);
                new Uint8Array(out.data).set(srcU8, filled * stride);
            }

            for (const { tmp } of temps) tmp.release();

            filled += take;
            // Advance to the next source if we exhausted this one mid-chunk.
            if (filled < outCount && localRow + take >= counts[si]) si++;
        }
    };

    const close = async (): Promise<void> => {
        await Promise.all(sources.map(s => s.close()));
    };

    return { meta, read, close };
};

export { concatSource };
