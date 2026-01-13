// Core interfaces and base classes
export { ReadStream, type ReadSource, type ReadFileSystem, type ProgressCallback, readFile } from './file-system';

// Platform-agnostic path utilities
export { dirname, join } from '../../utils/path';

// Filesystem implementations
export { MemoryReadFileSystem } from './memory-file-system';
export { NodeReadFileSystem } from '../../node-file-system';
export { UrlReadFileSystem } from './url-file-system';
export { ZipReadFileSystem } from './zip-file-system';
