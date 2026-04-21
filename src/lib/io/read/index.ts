// Core interfaces and base classes
export { ReadStream, type ReadSource, type ReadFileSystem, type ProgressCallback, readFile } from './file-system';
export { BufferedReadStream } from './buffered-read-stream';
export { CombineProgress } from './combine-progress';

// Platform-agnostic path utilities
export { basename, dirname, join } from 'pathe';

// Filesystem implementations (platform-agnostic only)
export { MemoryReadFileSystem } from './memory-file-system';
export { UrlReadFileSystem } from './url-file-system';
export { ZipReadFileSystem, type ZipEntry } from './zip-file-system';
