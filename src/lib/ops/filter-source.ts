import {
    type ChunkData,
    type ChunkDataPool,
    type ChunkLayer,
    type ReadRequest,
    type ChunkSource,
    type ChunkSourceMetadata
} from '../chunk';

// Mutable form of a chunk read request, for building one layer-by-layer before
// passing it to a parent's `read` (whose `ReadRequest` fields are readonly).
type MutableReadRequest = { chunkIndex: number; lod: number } & { [L in ChunkLayer]?: ChunkData };

/**
 * Select a subset of a source's gaussians by index, as a lazy view.
 *
 * `indices` is an **ascending** list of parent (LOD-0) gaussian indices to keep;
 * output gaussian `i` is parent gaussian `indices[i]`. Output is single-LOD with
 * `numGaussians = indices.length`; layout, layers and the pending transform are
 * inherited unchanged (filtering removes rows, not columns or coordinate space).
 *
 * Reads stream the parent monotonically: an output chunk maps to a contiguous,
 * ascending run of parent indices, so it touches a contiguous span of parent
 * chunks. Each contributing parent chunk is read once — filling every requested
 * layer in a single `parent.read` — and the kept rows are gathered into the
 * output buffer. Peak extra memory is one parent chunk-set of temporaries.
 *
 * The producers in `filter-mask.ts` emit ascending indices; the ascending
 * invariant is required (not validated here) — an unsorted list yields garbage.
 *
 * @param parent - The source to filter (read at LOD 0).
 * @param indices - Ascending parent gaussian indices to keep.
 * @param pool - Pool for the temporary parent read buffers; `chunkSize` must match the parent's.
 * @returns A derived source serving the selected gaussians chunk-by-chunk.
 */
const filterSource = (parent: ChunkSource, indices: Uint32Array, pool: ChunkDataPool): ChunkSource => {
    const parentMeta = parent.meta;
    const S = parentMeta.chunkSize;
    const parentN = parentMeta.numGaussians;
    const numOut = indices.length;
    const numChunks = Math.ceil(numOut / S);

    const meta: ChunkSourceMetadata = {
        ...parentMeta,
        numGaussians: numOut,
        numLods: 1,
        lodCounts: [numOut],
        numChunks: [numChunks]
    };

    const read = async (request: ReadRequest): Promise<void> => {
        if ('indices' in request) {
            // Gather: output row j is kept index `indices[request.indices[..]]`.
            // Compose the kept-list with the requested indices and gather those
            // parent rows in one pass (every source supports gather via `read`).
            const { indices: reqIdx, indexOffset, count } = request;
            const mapped = new Uint32Array(count);
            for (let j = 0; j < count; j++) mapped[j] = indices[reqIdx[indexOffset + j]];
            await parent.read({
                indices: mapped,
                indexOffset: 0,
                count,
                lod: 0,
                position: request.position,
                geometric: request.geometric,
                color: request.color,
                other: request.other
            });
            return;
        }

        const outStart = request.chunkIndex * S;
        const outCount = Math.min(S, numOut - outStart);

        // Which layers the caller wants filled, paired with their output buffer.
        const wanted: { layer: ChunkLayer; out: ChunkData }[] = [];
        if (request.position) wanted.push({ layer: 'position', out: request.position });
        if (request.geometric) wanted.push({ layer: 'geometric', out: request.geometric });
        if (request.color) wanted.push({ layer: 'color', out: request.color });
        if (request.other) wanted.push({ layer: 'other', out: request.other });
        if (wanted.length === 0) return;

        // Walk this output chunk's parent indices in ascending order, grouping
        // the run that falls inside each parent chunk so we read it once.
        let j = 0;
        while (j < outCount) {
            const pc = Math.floor(indices[outStart + j] / S);
            const pcStart = pc * S;
            const pcEnd = pcStart + Math.min(S, parentN - pcStart);

            let jEnd = j;
            while (jEnd < outCount && indices[outStart + jEnd] < pcEnd) jEnd++;
            const groupCount = jEnd - j;
            const parentCount = pcEnd - pcStart;

            // Read all wanted layers of this parent chunk into temporaries.
            const temps: { layer: ChunkLayer; out: ChunkData; tmp: ChunkData }[] = [];
            const req: MutableReadRequest = { chunkIndex: pc, lod: 0 };
            for (const { layer, out } of wanted) {
                const tmp = pool.acquire(layer, parentMeta.layouts[layer]!, parentCount);
                req[layer] = tmp;
                temps.push({ layer, out, tmp });
            }
            await parent.read(req);

            // Gather the kept rows of this group into each output buffer at row j,
            // a tight per-row 32-bit-word copy (stride is always a u32 multiple).
            for (const { out, tmp } of temps) {
                const sw = tmp.stride >>> 2;
                const srcU32 = new Uint32Array(tmp.data);
                const dstU32 = new Uint32Array(out.data);
                for (let g = 0; g < groupCount; g++) {
                    const so = (indices[outStart + j + g] - pcStart) * sw;
                    const dof = (j + g) * sw;
                    for (let w = 0; w < sw; w++) dstU32[dof + w] = srcU32[so + w];
                }
            }

            for (const { tmp } of temps) tmp.release();
            j = jEnd;
        }
    };

    return { meta, read, close: () => parent.close() };
};

export { filterSource };
