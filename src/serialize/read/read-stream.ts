/**
 * Abstract base class for streaming data from a source.
 * Uses a pull-based model where the consumer provides the buffer.
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

export { ReadStream };
