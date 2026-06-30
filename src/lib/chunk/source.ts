import { type Transform } from '../utils';
import { type ChunkData } from './data';
import { type ExtraColumn, type ChunkLayer, type SHBands, type LayerLayout } from './layout';

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
 * Fields common to every {@link ReadRequest}: which LOD to read and the
 * destination buffers for whichever layers the caller wants filled. Layers
 * omitted from the request are skipped.
 */
type ReadTarget = {
    /** Which LOD to read (default 0). */
    readonly lod?: number;
    readonly position?: ChunkData;
    readonly geometric?: ChunkData;
    readonly color?: ChunkData;
    readonly other?: ChunkData;
};

/**
 * A read request to a {@link ChunkSource}, selecting source rows in one of two
 * ways:
 *
 * - **Chunk** (`chunkIndex`): the contiguous run
 *   `[chunkIndex·chunkSize, +chunkSize)` of the chosen LOD, filled into output
 *   rows `[0, count)`. All passed buffers must have `count` equal to
 *   `meta.chunkSize` for non-final chunks or the trailing count for the last.
 * - **Gather** (`indices`/`indexOffset`/`count`): arbitrary source rows
 *   `indices[indexOffset .. indexOffset + count)` of the chosen LOD, filled into
 *   output rows `[0, count)`. Indices need not be sorted and may repeat.
 *
 * Gather underpins the LOD writer's "positions resident, heavy data fetched per
 * output chunk" pass — for a fixed-stride file source each row is a byte-range
 * read, so a unit pulls only its own gaussians (≈ 1× total reads, no whole-scene
 * residency). The two are the same operation with a different row selection; the
 * decode is identical, which is why a source serves both from one `read`.
 *
 * The arms are disjoint on the `indices` key, so an implementation discriminates
 * with `'indices' in request` (gather) vs the chunk path otherwise.
 */
type ReadRequest =
    | (ReadTarget & { readonly chunkIndex: number })
    | (ReadTarget & { readonly indices: Uint32Array; readonly indexOffset: number; readonly count: number });

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
     * Fill the caller's destination buffers from the source, selecting rows
     * either by chunk index or by an explicit index list (see
     * {@link ReadRequest}). Layers present in the request are filled; absent
     * layers are skipped. All passed buffers must share the same `count` —
     * the chunk size for a chunk request, or `count` for a gather.
     *
     * Every source supports both selections: a chunk is a contiguous range and
     * a gather is an arbitrary one, but the per-row decode is the same.
     */
    read(request: ReadRequest): Promise<void>;

    /**
     * Release any open file handles or internal decode state.
     * Idempotent; safe to call multiple times.
     */
    close(): Promise<void>;
}

export { type ChunkSource, type ReadRequest, type ReadTarget, type ChunkSourceMetadata };
