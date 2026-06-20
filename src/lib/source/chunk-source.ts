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
     * Release any open file handles or internal decode state.
     * Idempotent; safe to call multiple times.
     */
    close(): Promise<void>;
}

export { type ChunkSource, type ChunkReadRequest, type ChunkSourceMetadata };
