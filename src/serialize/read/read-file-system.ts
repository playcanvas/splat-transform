import { ReadSource } from './read-source';

/**
 * Progress callback for tracking read operations.
 * @param bytesLoaded - Bytes loaded so far
 * @param totalBytes - Total bytes if known, undefined otherwise
 */
type ProgressCallback = (bytesLoaded: number, totalBytes: number | undefined) => void;

/**
 * Interface for a file system that can create readable sources.
 * Implementations exist for various backends (URL, Node FS, Zip, Memory).
 */
interface ReadFileSystem {
    /**
     * Create a readable source for the given path/identifier.
     * @param filename - Path or identifier for the resource
     * @param progress - Optional callback for progress reporting
     * @returns Promise resolving to a ReadSource
     */
    createSource(filename: string, progress?: ProgressCallback): Promise<ReadSource>;
}

/**
 * Read an entire file into memory.
 * Convenience helper that handles source creation, reading, and cleanup.
 * @param fs - The file system to read from
 * @param filename - Path or identifier for the resource
 * @returns Promise resolving to file contents as Uint8Array
 */
const readFile = async (fs: ReadFileSystem, filename: string): Promise<Uint8Array> => {
    const source = await fs.createSource(filename);
    try {
        const stream = source.read();
        return await stream.readAll();
    } finally {
        source.close();
    }
};

export { type ReadFileSystem, type ProgressCallback, readFile };
