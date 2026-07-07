// ---------------------------------------------------------------------------
// Chunk-source pipeline (the primary API): scenes flow through lazy, chunked
// `ChunkSource`s so resident memory is bounded by chunk size, not scene size.
// ---------------------------------------------------------------------------

// Core contract + pool
export { createChunkDataPool } from './chunk';
export type {
    ChunkSource, ChunkSourceMetadata, ReadRequest, ReadTarget,
    ChunkData, ChunkDataPool,
    ChunkLayer, SHBands, ChunkField, ChunkFieldMap, LayerLayout, ExtraColumn
} from './chunk';

// Structural combinators (lazy views over sources)
export { bakeTransform, concatSource, selectLod, stackLods } from './ops';

// Action processing over a source: `processSource` streams and throws on
// actions that need the DataTable bridge; `processSourceBridged` handles every
// action, materializing only the DataTable-only runs as islands.
export { processSource, processSourceBridged } from './process-source';
export type {
    ProcessAction,
    ProcessOptions,
    Translate,
    Rotate,
    Scale,
    FilterNaN,
    FilterByValue,
    FilterBands,
    FilterBox,
    FilterSphere,
    FilterFloaters,
    FilterCluster,
    Param as ProcessParam,
    Stats,
    Info,
    MortonOrder,
    Decimate
} from './process';

// Chunk-native decimation
export { decimateSource } from './decimate';
export type { DecimateOptions, DecimateSpill } from './decimate';

// Statistics
export { computeStats } from './stats';
export type { LodStats, LodStatsData, SourceStats } from './stats';

// High-level read/write
export { readFile, readFileInfo, getInputFormat } from './read';
export type { InputFormat, ReadFileOptions, FileInfo } from './read';
export { writeFile, writeSource, getOutputFormat } from './write';
export type { OutputFormat, WriteOptions, WriteSourceOptions } from './write';
export { writeLodSource } from './writers';
export type { WriteLodSourceOptions } from './writers';

// ---------------------------------------------------------------------------
// DataTable compat (secondary API, for consumers mid-migration): the legacy
// whole-scene table, its processor, and the bridges to/from the chunk-source
// world. Everything here materializes the full scene in memory.
// ---------------------------------------------------------------------------
export { Column, DataTable, combine } from './data-table';
export type { TypedArray, ColumnType, Row } from './data-table';
export { dataTableToChunkSource, materializeToDataTable } from './compat/data-table';
export { processDataTable } from './process';

// Individual readers (advanced use; ply/splat/spz return a `ChunkSource`,
// ksplat/mjs/sog return a `DataTable`)
export { readKsplat, readMjs, readPly, readSog, readSplat, readSpz } from './readers';

// Individual writers (advanced use; DataTable-input compat set)
export { writeSog, writeSpz, writePly, writeCompressedPly, writeCsv, writeHtml, writeImage, writeGlb, writeVoxel } from './writers';
export type { WriteImageOptions, WriteVoxelOptions, VoxelMetadata } from './writers';

// ---------------------------------------------------------------------------
// Infrastructure
// ---------------------------------------------------------------------------

// Utils
export {
    fmtBytes, fmtCount, fmtTime,
    logger, TextRenderer, Transform, WebPCodec
} from './utils';
export type { Bar, Group, LogEvent, Logger, MessageKind, Renderer, TextRendererOptions, Verbosity } from './utils';

// Worker pool for CPU-heavy tasks
export { WorkerQueue } from './workers';

// File system abstractions
export { ReadStream, BufferedReadStream, MemoryReadFileSystem, UrlReadFileSystem, ZipReadFileSystem } from './io/read';
export type { ReadSource, ReadFileSystem, ProgressCallback, ZipEntry } from './io/read';
export { MemoryFileSystem, ZipFileSystem } from './io/write';
export type { FileSystem, Writer } from './io/write';

// Types
export type { CollisionMeshShape, Options, Param, DeviceCreator } from './types';

// Version
export { version, revision } from './version';
