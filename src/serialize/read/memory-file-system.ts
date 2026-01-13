import { type ReadFileSystem, type ProgressCallback, type ReadSource, ReadStream } from './file-system';

/**
 * ReadStream implementation for reading from memory buffers.
 */
class MemoryReadStream extends ReadStream {
    private data: Uint8Array;
    private offset: number;
    private end: number;

    constructor(data: Uint8Array, start: number, end: number) {
        super(end - start);
        this.data = data;
        this.offset = start;
        this.end = end;
    }

    pull(target: Uint8Array): Promise<number> {
        const remaining = this.end - this.offset;
        if (remaining <= 0) {
            return Promise.resolve(0);
        }

        const bytesToCopy = Math.min(target.length, remaining);
        target.set(this.data.subarray(this.offset, this.offset + bytesToCopy));
        this.offset += bytesToCopy;
        this.bytesRead += bytesToCopy;
        return Promise.resolve(bytesToCopy);
    }
}

/**
 * ReadSource implementation wrapping a Uint8Array or ArrayBuffer.
 * Size is always exact. Always seekable.
 */
class MemoryReadSource implements ReadSource {
    readonly size: number;
    readonly seekable: boolean = true;

    private data: Uint8Array;
    private closed: boolean = false;

    constructor(data: Uint8Array | ArrayBuffer) {
        this.data = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
        this.size = this.data.length;
    }

    read(start: number = 0, end: number = this.size): ReadStream {
        if (this.closed) {
            throw new Error('Source has been closed');
        }

        // Clamp range to valid bounds
        const clampedStart = Math.max(0, Math.min(start, this.size));
        const clampedEnd = Math.max(clampedStart, Math.min(end, this.size));

        return new MemoryReadStream(this.data, clampedStart, clampedEnd);
    }

    close(): void {
        this.closed = true;
    }
}

/**
 * ReadFileSystem for reading from named memory buffers.
 * Useful for testing or when data is already in memory.
 */
class MemoryReadFileSystem implements ReadFileSystem {
    private buffers: Map<string, Uint8Array> = new Map();

    /**
     * Store a named buffer.
     * @param name - Name/path for the buffer
     * @param data - Data to store
     */
    set(name: string, data: Uint8Array): void {
        this.buffers.set(name, data);
    }

    /**
     * Get a stored buffer by name.
     * @param name - Name/path of the buffer
     * @returns The stored data or undefined
     */
    get(name: string): Uint8Array | undefined {
        return this.buffers.get(name);
    }

    createSource(filename: string, progress?: ProgressCallback): Promise<ReadSource> {
        const data = this.buffers.get(filename);
        if (!data) {
            return Promise.reject(new Error(`Entry not found: ${filename}`));
        }

        // Report complete progress immediately since data is already in memory
        if (progress) {
            progress(data.length, data.length);
        }

        return Promise.resolve(new MemoryReadSource(data));
    }
}

export { MemoryReadFileSystem };
