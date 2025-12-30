import { ReadFileSystem } from './read-file-system';
import { ReadSource } from './read-source';

type ZipEntryInfo = {
    offset: number;
    compressedSize: number;
    uncompressedSize: number;
    compression: 'none' | 'deflate';
};

const inflate = async (compressed: Uint8Array): Promise<Uint8Array> => {
    const ds = new DecompressionStream('deflate-raw');
    const out = new Blob([compressed as ArrayBufferView<ArrayBuffer>]).stream().pipeThrough(ds);
    const ab = await new Response(out).arrayBuffer();
    return new Uint8Array(ab); // uncompressed file bytes
};

// Parse ZIP file using EOCD/CDR approach
const parseZipEntries = (data: Uint8Array): Map<string, ZipEntryInfo> => {
    const dataView = new DataView(data.buffer, data.byteOffset, data.byteLength);

    const u16 = (offset: number) => dataView.getUint16(offset, true);
    const u32 = (offset: number) => dataView.getUint32(offset, true);

    // Read End of Central Directory Record (last 22 bytes)
    const eocdOffset = data.byteLength - 22;
    const eocdMagic = u32(eocdOffset);

    if (eocdMagic !== 0x06054b50) {
        throw new Error('Invalid zip file: EOCD not found');
    }

    const numFiles = u16(eocdOffset + 8);
    const cdOffset = u32(eocdOffset + 16);

    if (cdOffset === 0xffffffff) {
        throw new Error('Invalid zip file: Zip64 not supported');
    }

    // Parse Central Directory Records
    const entries = new Map<string, ZipEntryInfo>();
    let offset = cdOffset;

    for (let i = 0; i < numFiles; i++) {
        const cdrMagic = u32(offset);
        if (cdrMagic !== 0x02014b50) {
            throw new Error('Invalid zip file: CDR not found');
        }

        const compressionMethod = u16(offset + 10);
        const compressedSize = u32(offset + 20);
        const uncompressedSize = u32(offset + 24);
        const filenameLength = u16(offset + 28);
        const extraFieldLength = u16(offset + 30);
        const fileCommentLength = u16(offset + 32);
        const lfhOffset = u32(offset + 42);

        const filename = new TextDecoder().decode(
            data.subarray(offset + 46, offset + 46 + filenameLength)
        );

        // Determine compression type
        let compression: 'none' | 'deflate';
        if (compressionMethod === 0) {
            compression = 'none';
        } else if (compressionMethod === 8) {
            compression = 'deflate';
        } else {
            throw new Error(`Unsupported ZIP compression method: ${compressionMethod}`);
        }

        // Read Local File Header to get actual data offset
        const lfhFilenameLength = u16(lfhOffset + 26);
        const lfhExtraLength = u16(lfhOffset + 28);
        const dataOffset = lfhOffset + 30 + lfhFilenameLength + lfhExtraLength;

        entries.set(filename, {
            offset: dataOffset,
            compressedSize,
            uncompressedSize,
            compression
        });

        // Move to next CDR
        offset += 46 + filenameLength + extraFieldLength + fileCommentLength;
    }

    return entries;
};

// Minimal ZIP reader supporting STORED (method 0) and DEFLATE (method 8).
// Implements ReadFileSystem to allow reading files from within the ZIP archive.
class ZipReadFileSystem implements ReadFileSystem {
    private data: Uint8Array;
    private entries: Map<string, ZipEntryInfo>;

    constructor(data: Uint8Array) {
        this.data = data;
        this.entries = parseZipEntries(data);
    }

    createReader(filename: string): ReadSource {
        const entry = this.entries.get(filename);
        if (!entry) {
            throw new Error(`File not found in ZIP: ${filename}`);
        }

        const compressedData = this.data.subarray(
            entry.offset,
            entry.offset + entry.compressedSize
        );

        if (entry.compression === 'none') {
            return ReadSource.fromOptions({ source: compressedData });
        }

        // For deflate, we need async decompression - return a ReadSource with factory
        return new ReadSource(async () => {
            const decompressed = await inflate(compressedData);
            return { source: decompressed };
        });
    }

    getFiles(path: string): string[] {
        const prefix = path.endsWith('/') ? path : `${path}/`;
        return [...this.entries.keys()].filter(name => path === '' || path === '.' || name === path || name.startsWith(prefix)
        );
    }
}

export { ZipReadFileSystem };
