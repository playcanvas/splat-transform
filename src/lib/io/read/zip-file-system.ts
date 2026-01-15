import { type ReadFileSystem, type ProgressCallback, type ReadSource, ReadStream } from './file-system';

/**
 * Metadata for a zip file entry.
 */
type ZipEntry = {
    name: string;
    compressedSize: number;
    uncompressedSize: number;
    offset: number;        // Local header offset
    method: number;        // 0=store, 8=deflate
};

/**
 * ReadStream for reading stored (uncompressed) zip entries.
 */
class StoredEntryReadStream extends ReadStream {
    private sourceStream: ReadStream;

    constructor(sourceStream: ReadStream, expectedSize: number) {
        super(expectedSize);
        this.sourceStream = sourceStream;
    }

    async pull(target: Uint8Array): Promise<number> {
        const n = await this.sourceStream.pull(target);
        this.bytesRead += n;
        return n;
    }

    close(): void {
        this.sourceStream.close();
    }
}

/**
 * ReadStream for reading deflated (compressed) zip entries.
 * Uses DecompressionStream for streaming decompression.
 */
class DeflateEntryReadStream extends ReadStream {
    private sourceStream: ReadStream;
    private reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
    private buffer: Uint8Array | null = null;
    private bufferOffset: number = 0;
    private closed: boolean = false;
    private initialized: boolean = false;

    constructor(sourceStream: ReadStream, expectedSize: number) {
        super(expectedSize);
        this.sourceStream = sourceStream;
    }

    private init(): void {
        if (this.reader || this.closed) return;

        // Create a ReadableStream from our source
        const sourceStream = this.sourceStream;
        const inputStream = new ReadableStream<Uint8Array>({
            async pull(controller) {
                const chunk = new Uint8Array(32768);
                const n = await sourceStream.pull(chunk);
                if (n === 0) {
                    controller.close();
                } else {
                    controller.enqueue(chunk.subarray(0, n));
                }
            }
        });

        // Pipe through decompression
        // Type assertion needed due to TypeScript's strict typing of DecompressionStream
        const decompressedStream = inputStream.pipeThrough(
            new DecompressionStream('deflate-raw') as unknown as TransformStream<Uint8Array, Uint8Array>
        );

        this.reader = decompressedStream.getReader();
    }

    async pull(target: Uint8Array): Promise<number> {
        if (this.closed) return 0;

        // Initialize on first pull
        if (!this.initialized) {
            this.init();
            this.initialized = true;
        }

        if (!this.reader) return 0;

        let targetOffset = 0;

        // First, consume any leftover data from previous read
        if (this.buffer && this.bufferOffset < this.buffer.length) {
            const remaining = this.buffer.length - this.bufferOffset;
            const toCopy = Math.min(remaining, target.length);
            target.set(this.buffer.subarray(this.bufferOffset, this.bufferOffset + toCopy));
            this.bufferOffset += toCopy;
            targetOffset += toCopy;
            this.bytesRead += toCopy;

            if (targetOffset >= target.length) {
                return targetOffset;
            }

            if (this.bufferOffset >= this.buffer.length) {
                this.buffer = null;
                this.bufferOffset = 0;
            }
        }

        // Read from decompressed stream
        while (targetOffset < target.length) {
            const { done, value } = await this.reader.read();

            if (done || !value) {
                break;
            }

            const toCopy = Math.min(value.length, target.length - targetOffset);
            target.set(value.subarray(0, toCopy), targetOffset);
            targetOffset += toCopy;
            this.bytesRead += toCopy;

            // Store leftover for next pull
            if (toCopy < value.length) {
                this.buffer = value;
                this.bufferOffset = toCopy;
                break;
            }
        }

        return targetOffset;
    }

    close(): void {
        this.closed = true;
        if (this.reader) {
            this.reader.cancel();
            this.reader = null;
        }
        this.sourceStream.close();
    }
}

/**
 * ReadSource for a single zip entry.
 */
class ZipEntrySource implements ReadSource {
    readonly size: number;
    readonly seekable: boolean;

    private source: ReadSource;
    private entry: ZipEntry;
    private dataOffset: number;
    private closed: boolean = false;

    constructor(source: ReadSource, entry: ZipEntry, dataOffset: number) {
        this.source = source;
        this.entry = entry;
        this.dataOffset = dataOffset;
        this.size = entry.uncompressedSize;
        // Only stored entries support seeking
        this.seekable = entry.method === 0;
    }

    read(start?: number, end?: number): ReadStream {
        if (this.closed) {
            throw new Error('Source has been closed');
        }

        if (this.entry.method === 0) {
            // Stored (uncompressed) - direct range read
            const rangeStart = start ?? 0;
            const rangeEnd = end ?? this.entry.uncompressedSize;
            const sourceStream = this.source.read(
                this.dataOffset + rangeStart,
                this.dataOffset + rangeEnd
            );
            return new StoredEntryReadStream(sourceStream, rangeEnd - rangeStart);
        }

        if (this.entry.method === 8) {
            // Deflate - must read from start and decompress
            if (start !== undefined && start !== 0) {
                throw new Error('Range reads not supported on compressed zip entries');
            }

            const sourceStream = this.source.read(
                this.dataOffset,
                this.dataOffset + this.entry.compressedSize
            );
            return new DeflateEntryReadStream(sourceStream, this.entry.uncompressedSize);
        }

        throw new Error(`Unsupported compression method: ${this.entry.method}`);
    }

    close(): void {
        this.closed = true;
    }
}

/**
 * Virtual filesystem for reading files from a zip archive.
 * Wraps any ReadSource and provides memory-efficient streaming access.
 */
class ZipReadFileSystem implements ReadFileSystem {
    private source: ReadSource;
    private entries: Map<string, ZipEntry> | null = null;
    private parsePromise: Promise<Map<string, ZipEntry>> | null = null;
    private closed: boolean = false;

    constructor(source: ReadSource) {
        this.source = source;
    }

    /**
     * Parse the central directory from the zip file.
     * Called lazily on first createSource() or list() call.
     * @returns Map of entry names to ZipEntry metadata
     */
    private async parseDirectory(): Promise<Map<string, ZipEntry>> {
        if (this.entries) {
            return this.entries;
        }

        const size = this.source.size;
        if (size === undefined) {
            throw new Error('Cannot read zip from source with unknown size');
        }

        // Read the last 65KB to find the End of Central Directory record
        const eocdSearchSize = Math.min(65536 + 22, size);
        const eocdStream = this.source.read(size - eocdSearchSize, size);
        const eocdData = await eocdStream.readAll();
        eocdStream.close();

        // Find EOCD signature (0x06054b50)
        let eocdOffset = -1;
        for (let i = eocdData.length - 22; i >= 0; i--) {
            if (eocdData[i] === 0x50 &&
                eocdData[i + 1] === 0x4b &&
                eocdData[i + 2] === 0x05 &&
                eocdData[i + 3] === 0x06) {
                eocdOffset = i;
                break;
            }
        }

        if (eocdOffset < 0) {
            throw new Error('End of central directory not found - invalid zip file');
        }

        const eocdView = new DataView(eocdData.buffer, eocdData.byteOffset + eocdOffset, 22);
        const entryCount = eocdView.getUint16(10, true);
        const cdSize = eocdView.getUint32(12, true);
        const cdOffset = eocdView.getUint32(16, true);

        // Read central directory
        const cdStream = this.source.read(cdOffset, cdOffset + cdSize);
        const cdData = await cdStream.readAll();
        cdStream.close();

        const entries = new Map<string, ZipEntry>();
        let offset = 0;

        for (let i = 0; i < entryCount; i++) {
            if (offset + 46 > cdData.length) {
                throw new Error('Truncated central directory');
            }

            const cdView = new DataView(cdData.buffer, cdData.byteOffset + offset);
            const sig = cdView.getUint32(0, true);

            if (sig !== 0x02014b50) {
                throw new Error('Invalid central directory entry signature');
            }

            const gpFlags = cdView.getUint16(8, true);
            const method = cdView.getUint16(10, true);
            const compressedSize = cdView.getUint32(20, true);
            const uncompressedSize = cdView.getUint32(24, true);
            const nameLen = cdView.getUint16(28, true);
            const extraLen = cdView.getUint16(30, true);
            const commentLen = cdView.getUint16(32, true);
            const localHeaderOffset = cdView.getUint32(42, true);

            const nameBytes = cdData.subarray(offset + 46, offset + 46 + nameLen);
            const utf8 = (gpFlags & 0x800) !== 0;
            const name = new TextDecoder(utf8 ? 'utf-8' : 'ascii').decode(nameBytes);

            entries.set(name, {
                name,
                compressedSize,
                uncompressedSize,
                offset: localHeaderOffset,
                method
            });

            offset += 46 + nameLen + extraLen + commentLen;
        }

        this.entries = entries;
        return entries;
    }

    /**
     * Get the data offset for an entry by reading its local header.
     * @param entry - Zip entry metadata
     * @returns Byte offset where entry data begins
     */
    private async getDataOffset(entry: ZipEntry): Promise<number> {
        // Read local file header to get variable-length fields
        const headerStream = this.source.read(entry.offset, entry.offset + 30);
        const headerData = await headerStream.readAll();
        headerStream.close();

        const headerView = new DataView(headerData.buffer, headerData.byteOffset, 30);
        const sig = headerView.getUint32(0, true);

        if (sig !== 0x04034b50) {
            throw new Error('Invalid local file header signature');
        }

        const nameLen = headerView.getUint16(26, true);
        const extraLen = headerView.getUint16(28, true);

        return entry.offset + 30 + nameLen + extraLen;
    }

    async createSource(filename: string, _progress?: ProgressCallback): Promise<ReadSource> {
        if (this.closed) {
            throw new Error('Source has been closed');
        }

        if (!this.parsePromise) {
            this.parsePromise = this.parseDirectory();
        }

        const entries = await this.parsePromise;
        const entry = entries.get(filename);

        if (!entry) {
            throw new Error(`Entry not found: ${filename}`);
        }

        const dataOffset = await this.getDataOffset(entry);
        return new ZipEntrySource(this.source, entry, dataOffset);
    }

    /**
     * List all entries in the zip file.
     * @returns Array of entry names
     */
    async list(): Promise<string[]> {
        if (this.closed) {
            throw new Error('Source has been closed');
        }

        if (!this.parsePromise) {
            this.parsePromise = this.parseDirectory();
        }

        const entries = await this.parsePromise;
        return Array.from(entries.keys());
    }

    /**
     * Get entry metadata.
     * @param filename - Entry name
     * @returns Entry metadata or undefined if not found
     */
    async getEntry(filename: string): Promise<ZipEntry | undefined> {
        if (!this.parsePromise) {
            this.parsePromise = this.parseDirectory();
        }

        const entries = await this.parsePromise;
        return entries.get(filename);
    }

    /**
     * Close the zip filesystem and underlying source.
     */
    close(): void {
        this.closed = true;
        this.source.close();
    }
}

export { ZipReadFileSystem, type ZipEntry };
