import { Crc } from './crc';
import type { FileSystem, Writer } from './file-system';

// https://gist.github.com/rvaiya/4a2192df729056880a027789ae3cd4b7
// zip64: https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT (4.3.9, 4.3.14-4.3.16, 4.5.3)

type ZipEntry = {
    filename: Uint8Array;
    crc: Crc;
    sizeBytes: number;
    // byte offset of the entry's local header within the archive (central
    // directory records point back at it)
    headerOffset: number;
};

// Writer for a single zip entry
class ZipEntryWriter implements Writer {
    write: (data: Uint8Array) => Promise<void>;
    close: () => Promise<void>;
    abort: () => Promise<void>;
    get bytesWritten(): number {
        return this.entry.sizeBytes;
    }
    private entry: ZipEntry;

    constructor(outputWriter: Writer, entry: ZipEntry) {
        this.entry = entry;

        this.write = async (data: Uint8Array) => {
            entry.sizeBytes += data.length;
            entry.crc.update(data);
            await outputWriter.write(data);
        };

        this.close = async () => {
            // no-op, finalization is handled by ZipFileSystem
        };

        this.abort = async () => {
            // a partially-written entry can't be removed from the stream, so
            // the whole archive is unsalvageable — abort the underlying writer
            await outputWriter.abort();
        };
    }
}

/**
 * A file system that writes files into a ZIP archive.
 *
 * Creates a ZIP file containing all written files. Used internally
 * for bundled output formats like .sog files. Archives past the classic
 * 4 GiB / 65535-entry limits are written with zip64 records; smaller
 * archives use the classic layout only.
 *
 * @example
 * ```ts
 * const outputWriter = await fs.createWriter('bundle.zip');
 * const zipFs = new ZipFileSystem(outputWriter);
 *
 * // Write files into the zip
 * const writer = await zipFs.createWriter('data.json');
 * await writer.write(jsonData);
 * await writer.close();
 *
 * // Finalize the zip
 * await zipFs.close();
 * ```
 */
class ZipFileSystem implements FileSystem {
    /**
     * Sizes and offsets at or above this value are written in zip64 form
     * (0xFFFFFFFF in the u32 field, real value in a zip64 record). Tests lower
     * it to exercise zip64 without writing 4 GiB.
     *
     * @ignore
     */
    static zip64Limit = 0xffffffff;

    close: () => Promise<void>;
    createWriter: (filename: string) => Promise<Writer>;
    mkdir: (path: string) => Promise<void>;

    constructor(writer: Writer) {
        const zip64Limit = ZipFileSystem.zip64Limit;
        const textEncoder = new TextEncoder();
        const files: ZipEntry[] = [];
        let activeEntry: ZipEntry | null = null;
        // running byte offset into the archive (next local header goes here;
        // at close time this is where the central directory starts)
        let offset = 0;

        const date = new Date();
        const dosTime = (date.getHours() << 11) | (date.getMinutes() << 5) | Math.floor(date.getSeconds() / 2);
        const dosDate = ((date.getFullYear() - 1980) << 9) | ((date.getMonth() + 1) << 5) | date.getDate();

        const writeEntryHeader = async (filename: string) => {
            const filenameBuf = textEncoder.encode(filename);
            const nameLen = filenameBuf.length;

            const header = new Uint8Array(30 + nameLen);
            const view = new DataView(header.buffer);

            view.setUint32(0, 0x04034b50, true);
            view.setUint16(4, 20, true); // version needed to extract = 2.0
            view.setUint16(6, 0x8 | 0x800, true); // indicate crc and size comes after, utf-8 encoding
            view.setUint16(8, 0, true); // method = 0 (store)
            view.setUint16(10, dosTime, true);
            view.setUint16(12, dosDate, true);
            view.setUint16(26, nameLen, true);
            header.set(filenameBuf, 30);

            const entry: ZipEntry = { filename: filenameBuf, crc: new Crc(), sizeBytes: 0, headerOffset: offset };
            offset += header.length;

            await writer.write(header);

            files.push(entry);
            return entry;
        };

        const writeEntryFooter = async (entry: ZipEntry) => {
            const { crc, sizeBytes } = entry;
            // The local header is written before the size is known, so an
            // entry that outgrows u32 gets the zip64 (8-byte size) data
            // descriptor and its central directory record carries the zip64
            // sizes - the same scheme as Go's archive/zip and Java's
            // ZipOutputStream.
            const zip64 = sizeBytes >= zip64Limit;
            const data = new Uint8Array(zip64 ? 24 : 16);
            const view = new DataView(data.buffer);
            view.setUint32(0, 0x08074b50, true);
            view.setUint32(4, crc.value(), true);
            if (zip64) {
                view.setBigUint64(8, BigInt(sizeBytes), true);
                view.setBigUint64(16, BigInt(sizeBytes), true);
            } else {
                view.setUint32(8, sizeBytes, true);
                view.setUint32(12, sizeBytes, true);
            }
            offset += sizeBytes + data.length;
            await writer.write(data);
        };

        this.createWriter = async (filename: string): Promise<Writer> => {
            // Close previous entry if exists
            if (activeEntry) {
                await writeEntryFooter(activeEntry);
                activeEntry = null;
            }

            // Start new entry
            const entry = await writeEntryHeader(filename);
            activeEntry = entry;

            return new ZipEntryWriter(writer, entry);
        };

        this.mkdir = async (_path: string): Promise<void> => {
            // No-op for zip - directories are created implicitly from file paths
        };

        this.close = async () => {
            // Close last entry if exists
            if (activeEntry) {
                await writeEntryFooter(activeEntry);
                activeEntry = null;
            }

            // central directory starts where entry data ended
            const cdOffset = offset;

            // Write central directory records
            for (const file of files) {
                const { filename, crc, sizeBytes, headerOffset } = file;
                const nameLen = filename.length;

                // zip64 extended information extra field: holds only the
                // overflowed fields, in fixed order (sizes, then header offset)
                const sizeZip64 = sizeBytes >= zip64Limit;
                const offsetZip64 = headerOffset >= zip64Limit;
                const zip64Len = (sizeZip64 ? 16 : 0) + (offsetZip64 ? 8 : 0);
                const extraLen = zip64Len ? 4 + zip64Len : 0;
                const version = zip64Len ? 45 : 20;

                const cdr = new Uint8Array(46 + nameLen + extraLen);
                const view = new DataView(cdr.buffer);
                view.setUint32(0, 0x02014b50, true);
                view.setUint16(4, version, true);
                view.setUint16(6, version, true);
                view.setUint16(8, 0x8 | 0x800, true);
                view.setUint16(10, 0, true);
                view.setUint16(12, dosTime, true);
                view.setUint16(14, dosDate, true);
                view.setUint32(16, crc.value(), true);
                view.setUint32(20, sizeZip64 ? 0xffffffff : sizeBytes, true);
                view.setUint32(24, sizeZip64 ? 0xffffffff : sizeBytes, true);
                view.setUint16(28, nameLen, true);
                view.setUint16(30, extraLen, true);
                view.setUint32(42, offsetZip64 ? 0xffffffff : headerOffset, true);
                cdr.set(filename, 46);

                if (zip64Len) {
                    let p = 46 + nameLen;
                    view.setUint16(p, 0x0001, true);
                    view.setUint16(p + 2, zip64Len, true);
                    p += 4;
                    if (sizeZip64) {
                        view.setBigUint64(p, BigInt(sizeBytes), true); // uncompressed
                        view.setBigUint64(p + 8, BigInt(sizeBytes), true); // compressed
                        p += 16;
                    }
                    if (offsetZip64) {
                        view.setBigUint64(p, BigInt(headerOffset), true);
                    }
                }

                offset += cdr.length;
                await writer.write(cdr);
            }

            const cdSize = offset - cdOffset;
            const numFiles = files.length;

            // Write zip64 end of central directory record and locator ahead of
            // the classic record when any of its fields overflow
            if (numFiles >= 0xffff || cdSize >= zip64Limit || cdOffset >= zip64Limit) {
                const zip64Eocd = new Uint8Array(56 + 20);
                const zip64View = new DataView(zip64Eocd.buffer);
                zip64View.setUint32(0, 0x06064b50, true);
                zip64View.setBigUint64(4, 44n, true); // size of the rest of the record
                zip64View.setUint16(12, 45, true);
                zip64View.setUint16(14, 45, true);
                zip64View.setBigUint64(24, BigInt(numFiles), true);
                zip64View.setBigUint64(32, BigInt(numFiles), true);
                zip64View.setBigUint64(40, BigInt(cdSize), true);
                zip64View.setBigUint64(48, BigInt(cdOffset), true);

                // locator
                zip64View.setUint32(56, 0x07064b50, true);
                zip64View.setBigUint64(64, BigInt(offset), true); // zip64 record offset
                zip64View.setUint32(72, 1, true); // total number of disks

                await writer.write(zip64Eocd);
            }

            // Write end of central directory record
            const eocd = new Uint8Array(22);
            const eocdView = new DataView(eocd.buffer);
            eocdView.setUint32(0, 0x06054b50, true);
            eocdView.setUint16(8, Math.min(numFiles, 0xffff), true);
            eocdView.setUint16(10, Math.min(numFiles, 0xffff), true);
            eocdView.setUint32(12, cdSize >= zip64Limit ? 0xffffffff : cdSize, true);
            eocdView.setUint32(16, cdOffset >= zip64Limit ? 0xffffffff : cdOffset, true);

            await writer.write(eocd);

            // Close the underlying writer
            await writer.close();
        };
    }
}

export { ZipFileSystem };
