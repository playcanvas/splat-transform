/**
 * Abstract base class for streaming data from a source.
 * Uses a pull-based model where the consumer provides the buffer.
 * @ignore
 */
abstract class ReadStream {
    /**
     * Size hint for buffer pre-allocation in readAll().
     * May be undefined if size is unknown.
     */
    readonly expectedSize: number | undefined;

    /**
     * Total bytes read from this stream so far.
     */
    bytesRead: number = 0;

    /**
     * @param expectedSize - Optional size hint for buffer pre-allocation
     */
    constructor(expectedSize?: number) {
        this.expectedSize = expectedSize;
    }

    /**
     * Pull data into the provided buffer.
     * @param target - Buffer to fill with data
     * @returns Number of bytes read, or 0 for EOF
     */
    abstract pull(target: Uint8Array): Promise<number>;

    /**
     * Read entire stream into a single buffer.
     * Uses expectedSize hint if available, grows dynamically if needed.
     * @returns Complete data as Uint8Array
     */
    async readAll(): Promise<Uint8Array> {
        const capacity = this.expectedSize ?? 65536;
        let buffer = new Uint8Array(capacity);
        let length = 0;

        while (true) {
            // Grow buffer if full
            if (length >= buffer.length) {
                const newBuffer = new Uint8Array(buffer.length * 2);
                newBuffer.set(buffer);
                buffer = newBuffer;
            }

            const n = await this.pull(buffer.subarray(length));
            if (n === 0) break;
            length += n;
        }

        // Return exact-sized view
        return buffer.subarray(0, length);
    }

    /**
     * Release resources and abort any pending operations.
     */
    close(): void {
        // Base implementation does nothing - subclasses can override
    }
}

/**
 * Interface representing a readable data source.
 * Provides size information and creates streams for reading.
 * @ignore
 */
interface ReadSource {
    /**
     * The size of the source in bytes, or undefined if unknown.
     * For compressed sources (e.g., gzipped HTTP), this may be approximate.
     */
    readonly size: number | undefined;

    /**
     * Whether range reads are supported.
     * If false, read() must be called with no arguments or start=0.
     */
    readonly seekable: boolean;

    /**
     * Create a stream for reading data, optionally with a byte range.
     * @param start - Starting byte offset (inclusive), defaults to 0
     * @param end - Ending byte offset (exclusive), defaults to size/EOF
     * @returns A ReadStream for pulling data
     * @throws Error if range requested on non-seekable source
     */
    read(start?: number, end?: number): ReadStream;

    /**
     * Release any resources held by this source.
     */
    close(): void;
}

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

export { ReadStream, type ReadSource, type ReadFileSystem, type ProgressCallback, readFile };
