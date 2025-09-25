import { FileHandle } from 'node:fs/promises';

type ZipEntry = {
    name: string;
    size: number;
    readData: () => Promise<Uint8Array>;
};

// Minimal ZIP reader supporting STORED (method 0), data descriptor (0x08074b50), and UTF-8 filenames.
class ZipReader {
    private file: FileHandle;
    private cursor: number = 0;
    private size: number = 0;

    constructor(file: FileHandle, fileSize?: number) {
        this.file = file;
        this.size = fileSize ?? 0;
    }

    private async readAt(pos: number, len: number): Promise<Uint8Array> {
        const buf = new Uint8Array(len);
        const { bytesRead } = await this.file.read(buf, 0, len, pos);
        if (bytesRead !== len) throw new Error('Unexpected EOF while reading ZIP');
        return buf;
    }

    private async read(len: number): Promise<Uint8Array> {
        const buf = await this.readAt(this.cursor, len);
        this.cursor += len;
        return buf;
    }

    private dv(u8: Uint8Array) {
        return new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    }

    async list(): Promise<ZipEntry[]> {
        // To keep simple and compatible with our writer that streams local headers first, we'll
        // sequentially parse local headers and data descriptors until we hit the central directory
        // which we can ignore for listing (we already have file names and sizes by then).
        if (this.size === 0) {
            const stat = await this.file.stat();
            this.size = stat.size;
        }

        this.cursor = 0;
        const entries: ZipEntry[] = [];

        while (this.cursor + 30 <= this.size) {
            const header = await this.read(30);
            const dv = this.dv(header);
            const sig = dv.getUint32(0, true);
            if (sig === 0x02014b50 || sig === 0x06054b50) {
                // central directory or EOCD reached
                break;
            }
            if (sig !== 0x04034b50) {
                // not a local file header; stop
                break;
            }

            const gpFlags = dv.getUint16(6, true);
            const method = dv.getUint16(8, true);
            const nameLen = dv.getUint16(26, true);
            const extraLen = dv.getUint16(28, true);

            const nameBytes = await this.read(nameLen);
            const extra = await this.read(extraLen);
            const utf8 = (gpFlags & 0x800) !== 0;
            const name = new TextDecoder(utf8 ? 'utf-8' : 'ascii').decode(nameBytes);

            if (method !== 0) {
                throw new Error(`Unsupported ZIP compression method: ${method} (only STORE=0 supported)`);
            }

            let size = dv.getUint32(22, true);
            let crc = dv.getUint32(14, true);
            let useDescriptor = false;
            if (gpFlags & 0x8) {
                // data descriptor follows the file data; sizes are zero in local header
                useDescriptor = true;
            }

            const dataOffset = this.cursor;

            if (!useDescriptor) {
                // known size, can skip data directly
                const start = dataOffset;
                const end = start + size;
                this.cursor = end;
                {
                    const entrySize = size;
                    const entryStart = start;
                    entries.push({
                        name,
                        size: entrySize,
                        readData: () => this.readAt(entryStart, entrySize)
                    });
                }
            } else {
                // Need to scan until data descriptor signature 0x08074b50
                // Our writer writes descriptor immediately after data. We'll read forward to find it.
                // For performance, read in chunks.
                const chunk = 64 * 1024;
                let pos = dataOffset;
                let found = false;
                const sigBytes = new Uint8Array([0x50, 0x4b, 0x07, 0x08]);
                while (pos < this.size) {
                    const len = Math.min(chunk, this.size - pos);
                    const buf = await this.readAt(pos, len);
                    // search signature
                    for (let i = 0; i + 16 <= buf.length; i++) {
                        if (buf[i] === sigBytes[0] && buf[i + 1] === sigBytes[1] && buf[i + 2] === sigBytes[2] && buf[i + 3] === sigBytes[3]) {
                            const view = new DataView(buf.buffer, buf.byteOffset + i, 16);
                            const _sig = view.getUint32(0, true);
                            if (_sig === 0x08074b50) {
                                crc = view.getUint32(4, true);
                                size = view.getUint32(8, true);
                                const compSize = view.getUint32(12, true);
                                // According to spec, size and compSize are equal for STORE
                                const endOfData = pos + i; // descriptor starts here
                                // Update cursor to after descriptor
                                this.cursor = endOfData + 16;
                                {
                                    const entrySize = size;
                                    const entryStart = dataOffset;
                                    entries.push({
                                        name,
                                        size: entrySize,
                                        readData: () => this.readAt(entryStart, entrySize)
                                    });
                                }
                                found = true;
                                break;
                            }
                        }
                    }
                    if (found) break;
                    pos += len;
                }
                if (!found) throw new Error('ZIP data descriptor not found');
            }
        }

        return entries;
    }
}

export { ZipReader, ZipEntry };
