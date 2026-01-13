// Core interfaces and base classes
export { ReadStream } from './read-stream';
export { type ReadSource } from './read-source';
export { type ReadFileSystem, type ProgressCallback, readFile } from './read-file-system';

// Platform-agnostic path utilities
export { dirname, join } from '../path-utils';

// Memory source implementation
export { MemoryReadStream, MemoryReadSource, MemoryReadFileSystem } from './memory-source';

// Node.js file source implementation
export { NodeReadStream, NodeReadSource, NodeReadFileSystem } from './node-source';

// URL/fetch source implementation
export { UrlReadStream, UrlReadSource, UrlReadFileSystem } from './url-source';

// Zip filesystem implementation
export { ZipReadFileSystem, ZipEntrySource, type ZipEntry } from './zip-read-file-system';
