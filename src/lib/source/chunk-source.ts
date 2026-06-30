import { type Transform } from '../utils';
import { type ChunkData } from './chunk-data';
import { type LayerLayout } from './chunk-data-pool';
import { type ExtraColumn, type ChunkLayer, type SHBands } from './layout';

/**
 * Static description of a {@link ChunkSource}'s contents — what's in it and
 * how it's laid out. Populated at `open()` time; never changes thereafter.
 *
 * `chunkSize` is the gaussian count per chunk; all chunks are this size except
 * the final one in each LOD, which holds `lodCounts[lod] % chunkSize` gaussians
 * (or `chunkSize` if the count divides evenly).
 *
 * `layouts` exposes the byte stride and named field map for each available
 * layer, used by callers when acquiring `ChunkData` buffers from a {@link ChunkDataPool}.
 *
 * LOD is a structural axis: a chunk belongs to exactly one LOD. Sources never
 * carry a per-gaussian LOD tag.
 */
type ChunkSourceMetadata = {
    readonly numGaussians: number;
    readonly numLods: number;
    /** Gaussian counts per LOD. `lodCounts[0]` matches `numGaussians`. */
    readonly lodCounts: ReadonlyArray<number>;
    /** Gaussians per chunk (all chunks are this size except the last per LOD). */
    readonly chunkSize: number;
    /** Number of chunks per LOD: `Math.ceil(lodCounts[lod] / chunkSize)`. */
    readonly numChunks: ReadonlyArray<number>;
    /** SH band count present in the source. */
    readonly shBands: SHBands;
    /** Extra non-standard columns mapped to the `other` layer. */
    readonly extraColumns: ReadonlyArray<ExtraColumn>;
    /** Coordinate-space transform; applied lazily when consumed. */
    readonly transform: Transform;
    /** Which layers the source can serve. */
    readonly availableLayers: ReadonlySet<ChunkLayer>;
    /** Per-layer stride + field map. Keyed by layer; only present for available layers. */
    readonly layouts: Readonly<Partial<Record<ChunkLayer, LayerLayout>>>;
};

/**
 * A single read request to a {@link ChunkSource}.
 *
 * The caller passes destination buffers for whichever layers it wants filled
 * for the given `(chunkIndex, lod)`. Layers omitted from the request are
 * skipped. All passed buffers must have the same `count`, which must equal
 * `meta.chunkSize` for non-final chunks or the trailing count for the last.
 */
type ChunkReadRequest = {
    readonly chunkIndex: number;
    readonly lod?: number;
    readonly position?: ChunkData;
    readonly geometric?: ChunkData;
    readonly color?: ChunkData;
    readonly other?: ChunkData;
};

/**
 * A random-access (scatter-gather) read request to a {@link ChunkSource}.
 *
 * Fills output rows `[0, count)` of the passed layer buffers from source rows
 * `indices[indexOffset .. indexOffset + count)` of LOD `lod` (default 0). Indices
 * are arbitrary and need not be sorted; this is the per-row analog of
 * {@link ChunkReadRequest} and underpins the LOD writer's "positions resident,
 * heavy data fetched per output chunk" gather — for a fixed-stride file source
 * each row is a byte-range read, so a unit pulls only its own gaussians (≈ 1×
 * total reads, no whole-scene residency). LOD is structural: a gather targets one
 * LOD's gaussians (sources never carry a per-gaussian LOD tag).
 */
type RowReadRequest = {
    readonly indices: Uint32Array;
    readonly indexOffset: number;
    readonly count: number;
    /** Which LOD to gather from (default 0). */
    readonly lod?: number;
    readonly position?: ChunkData;
    readonly geometric?: ChunkData;
    readonly color?: ChunkData;
    readonly other?: ChunkData;
};

/**
 * Lazy, chunked, layered view onto gaussian splat data.
 *
 * Sources are opened over a file (or derived from another source via a
 * combinator) and expose only metadata up front — no gaussian data is loaded
 * at open time except for formats whose decode is fundamentally whole-blob
 * (SPZ, MJS). Data is materialized into caller-allocated `ChunkData` buffers
 * on demand via {@link ChunkSource.read}.
 *
 * Memory ownership is on the caller: buffers are acquired from a
 * `ChunkDataPool`, filled by `read`, used, and released back to the pool. The
 * source itself never holds long-lived buffer memory on the caller's behalf.
 */
interface ChunkSource {
    readonly meta: ChunkSourceMetadata;

    /**
     * Fill the caller's destination buffers with data for the given chunk
     * index. ChunkLayer fields present in the request are filled; absent layers
     * are skipped. All passed buffers must share the same `count` matching
     * the source's reported chunk size for the requested index.
     */
    read(request: ChunkReadRequest): Promise<void>;

    /**
     * Optional random-access gather: fill the request's layer buffers from
     * arbitrary source rows (see {@link RowReadRequest}). Implemented by sources
     * that support efficient per-row access — resident `InMemoryChunkSource` and
     * fixed-stride file sources (PLY). Combinators and streaming/whole-blob
     * sources omit it; callers needing it on such a source must `compact()` first.
     */
    readRows?(request: RowReadRequest): Promise<void>;

    /**
     * Release any open file handles or internal decode state.
     * Idempotent; safe to call multiple times.
     */
    close(): Promise<void>;
}

export { type ChunkSource, type ChunkReadRequest, type RowReadRequest, type ChunkSourceMetadata };
