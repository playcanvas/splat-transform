import { type ReadSource, type ReadStream } from '../io/read';
import { type ChunkSource, type ChunkSourceMetadata, type ReadRequest } from '../source';

/**
 * Read exactly `length` bytes from `stream` into `buffer` at `offset`. A stream
 * may satisfy a read in several pulls; returns the number of bytes actually read
 * (less than `length` only at EOF).
 *
 * @param stream - The stream to read from.
 * @param buffer - Destination buffer.
 * @param offset - Byte offset in `buffer` to write the first byte.
 * @param length - Number of bytes to read.
 * @returns The number of bytes read.
 */
const readExact = async (stream: ReadStream, buffer: Uint8Array, offset: number, length: number): Promise<number> => {
    let total = 0;
    while (total < length) {
        const n = await stream.pull(buffer.subarray(offset + total, offset + length));
        if (n === 0) break;
        total += n;
    }
    return total;
};

/**
 * Build a {@link ChunkSource} backed by a {@link ReadSource}. `close()` releases
 * the source, so every file-backed reader owns its source's lifetime uniformly
 * (a reader can't forget to close it). Used by the lazy PLY/splat/SPZ readers.
 *
 * @param source - The read source whose lifetime this `ChunkSource` owns.
 * @param meta - The source metadata.
 * @param read - The read implementation (serves both chunk and gather requests).
 * @returns A `ChunkSource` whose `close()` closes `source`.
 */
const fileChunkSource = (
    source: ReadSource,
    meta: ChunkSourceMetadata,
    read: (request: ReadRequest) => Promise<void>
): ChunkSource => ({
    meta,
    read,
    close: () => {
        source.close();
        return Promise.resolve();
    }
});

export { readExact, fileChunkSource };
