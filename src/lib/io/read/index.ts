// Core interfaces and base classes
export { ReadStream, type ReadSource, type ReadFileSystem, type ProgressCallback, readFile } from './file-system';

// Platform-agnostic path utilities
export { dirname, join } from 'pathe';

// Filesystem implementations (platform-agnostic only)
export { MemoryReadFileSystem } from './memory-file-system';
export { UrlReadFileSystem } from './url-file-system';
export { ZipReadFileSystem, type ZipEntry } from './zip-file-system';
