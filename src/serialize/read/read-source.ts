import { ReadStream } from './read-stream';

/**
 * Interface representing a readable data source.
 * Provides size information and creates streams for reading.
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

export { type ReadSource };
