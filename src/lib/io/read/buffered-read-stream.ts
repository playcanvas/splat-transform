import { ReadStream } from './file-system';

/**
 * ReadStream wrapper that adds read-ahead buffering to reduce async overhead.
 * Reads larger chunks from the inner stream and buffers excess data for
 * subsequent small reads. Useful for sources with high per-call overhead.
 *
 * @example
 * // Wrap a stream with 4MB read-ahead buffering
 * const buffered = new BufferedReadStream(rawStream, 4 * 1024 * 1024);
 * const data = await buffered.readAll();
 */
class BufferedReadStream extends ReadStream {
    private inner: ReadStream;
    private chunkSize: number;

    // Buffer state
    private buffer: Uint8Array | null = null;
    private bufferOffset = 0;

    /**
     * Create a caching wrapper around a stream.
     * @param inner - The underlying stream to read from
     * @param chunkSize - Minimum bytes to read at once from inner stream (default 64KB)
     */
    constructor(inner: ReadStream, chunkSize: number = 65536) {
        super(inner.expectedSize);
        this.inner = inner;
        this.chunkSize = chunkSize;
    }

    async pull(target: Uint8Array): Promise<number> {
        // Early return for zero-length requests (e.g., EOF check from readAll)
        if (target.length === 0) {
            return 0;
        }

        let written = 0;

        // Serve from buffer first
        if (this.buffer && this.bufferOffset < this.buffer.length) {
            const available = this.buffer.length - this.bufferOffset;
            const toCopy = Math.min(available, target.length);
            target.set(this.buffer.subarray(this.bufferOffset, this.bufferOffset + toCopy));
            this.bufferOffset += toCopy;
            written += toCopy;
            this.bytesRead += toCopy;

            // Clear exhausted buffer
            if (this.bufferOffset >= this.buffer.length) {
                this.buffer = null;
                this.bufferOffset = 0;
            }

            if (written >= target.length) {
                return written;
            }
        }

        // Read a chunk from inner stream
        const remaining = target.length - written;
        const readSize = Math.max(this.chunkSize, remaining);
        const chunk = new Uint8Array(readSize);
        const n = await this.inner.pull(chunk);

        if (n === 0) {
            return written;
        }

        // Copy what we need to target
        const toCopy = Math.min(n, remaining);
        target.set(chunk.subarray(0, toCopy), written);
        written += toCopy;
        this.bytesRead += toCopy;

        // Cache the excess
        if (toCopy < n) {
            this.buffer = chunk.subarray(0, n);
            this.bufferOffset = toCopy;
        }

        return written;
    }

    close(): void {
        this.inner.close();
        this.buffer = null;
    }
}

export { BufferedReadStream };
