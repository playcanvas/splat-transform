// Layout & type constants
export {
    type ChunkLayer,
    type SHBands,
    type ChunkField,
    type ChunkFieldMap,
    type LayerLayout,
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
export { type ChunkData } from './data';
export { type ChunkDataPool, createChunkDataPool } from './pool';

// Source contract
export { type ChunkSource, type ReadRequest, type ReadTarget, type ChunkSourceMetadata } from './source';

// In-memory backing + compact
export { InMemoryChunkSource, createInMemoryChunkSource, compact } from './in-memory';

// Residency: LRU decode cache with a user byte budget
export { cached } from './cached';
