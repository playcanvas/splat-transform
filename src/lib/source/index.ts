// Layout & type constants
export {
    type ChunkLayer,
    type SHBands,
    type ChunkField,
    type ChunkFieldMap,
    type ExtraColumn,
    SH_REST_COUNTS,
    POSITION_STRIDE,
    GEOMETRIC_STRIDE,
    DEFAULT_CHUNK_SIZE,
    colorStride,
    positionFields,
    geometricFields,
    colorFields,
    otherLayout
} from './layout';

// ChunkData + pool
export { type ChunkData } from './chunk-data';
export { type ChunkDataPool, type LayerLayout, createChunkDataPool } from './chunk-data-pool';

// Source contract
export { type ChunkSource, type ReadRequest, type ReadTarget, type ChunkSourceMetadata } from './chunk-source';

// In-memory backing + compact
export { InMemoryChunkSource, createInMemoryChunkSource, compact } from './in-memory-chunk-source';

// Residency: LRU decode cache with a user byte budget
export { cached } from './cached';
