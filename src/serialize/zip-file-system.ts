import { Crc } from './crc';
import { FileSystem, Writer } from './file-system';

// https://gist.github.com/rvaiya/4a2192df729056880a027789ae3cd4b7

type ZipEntry = {
    filename: Uint8Array;
    crc: Crc;
    sizeBytes: number;
};

// Writer for a single zip entry
class ZipEntryWriter implements Writer {
    write: (data: Uint8Array) => Promise<void>;
    close: () => Promise<void>;

    constructor(outputWriter: Writer, entry: ZipEntry) {
        this.write = async (data: Uint8Array) => {
            entry.sizeBytes += data.length;
            entry.crc.update(data);
            await outputWriter.write(data);
        };

        this.close = async () => {
            // no-op, finalization is handled by ZipFileSystem
        };
    }
}

// FileSystem implementation that writes files into a zip archive
class ZipFileSystem implements FileSystem {
    close: () => Promise<void>;
    createWriter: (filename: string) => Promise<Writer>;
    mkdir: (path: string) => Promise<void>;

    constructor(writer: Writer) {
        const textEncoder = new TextEncoder();
        const files: ZipEntry[] = [];
        let activeEntry: ZipEntry | null = null;

        const date = new Date();
        const dosTime = (date.getHours() << 11) | (date.getMinutes() << 5) | Math.floor(date.getSeconds() / 2);
        const dosDate = ((date.getFullYear() - 1980) << 9) | ((date.getMonth() + 1) << 5) | date.getDate();

        const writeEntryHeader = async (filename: string) => {
            const filenameBuf = textEncoder.encode(filename);
            const nameLen = filenameBuf.length;

            const header = new Uint8Array(30 + nameLen);
            const view = new DataView(header.buffer);

            view.setUint32(0, 0x04034b50, true);
            view.setUint16(4, 20, true);            // version needed to extract = 2.0
            view.setUint16(6, 0x8 | 0x800, true);   // indicate crc and size comes after, utf-8 encoding
            view.setUint16(8, 0, true);             // method = 0 (store)
            view.setUint16(10, dosTime, true);
            view.setUint16(12, dosDate, true);
            view.setUint16(26, nameLen, true);
            header.set(filenameBuf, 30);

            await writer.write(header);

            const entry: ZipEntry = { filename: filenameBuf, crc: new Crc(), sizeBytes: 0 };
            files.push(entry);
            return entry;
        };

        const writeEntryFooter = async (entry: ZipEntry) => {
            const { crc, sizeBytes } = entry;
            const data = new Uint8Array(16);
            const view = new DataView(data.buffer);
            view.setUint32(0, 0x08074b50, true);
            view.setUint32(4, crc.value(), true);
            view.setUint32(8, sizeBytes, true);
            view.setUint32(12, sizeBytes, true);
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

            // Write central directory records
            let offset = 0;
            for (const file of files) {
                const { filename, crc, sizeBytes } = file;
                const nameLen = filename.length;

                const cdr = new Uint8Array(46 + nameLen);
                const view = new DataView(cdr.buffer);
                view.setUint32(0, 0x02014b50, true);
                view.setUint16(4, 20, true);
                view.setUint16(6, 20, true);
                view.setUint16(8, 0x8 | 0x800, true);
                view.setUint16(10, 0, true);
                view.setUint16(12, dosTime, true);
                view.setUint16(14, dosDate, true);
                view.setUint32(16, crc.value(), true);
                view.setUint32(20, sizeBytes, true);
                view.setUint32(24, sizeBytes, true);
                view.setUint16(28, nameLen, true);
                view.setUint32(42, offset, true);
                cdr.set(filename, 46);

                await writer.write(cdr);

                offset += 30 + nameLen + sizeBytes + 16; // 30 local header + name + data + 16 descriptor
            }

            const filenameLength = files.reduce((tot, file) => tot + file.filename.length, 0);
            const dataLength = files.reduce((tot, file) => tot + file.sizeBytes, 0);

            // Write end of central directory record
            const eocd = new Uint8Array(22);
            const eocdView = new DataView(eocd.buffer);
            eocdView.setUint32(0, 0x06054b50, true);
            eocdView.setUint16(8, files.length, true);
            eocdView.setUint16(10, files.length, true);
            eocdView.setUint32(12, filenameLength + files.length * 46, true);
            eocdView.setUint32(16, filenameLength + files.length * (30 + 16) + dataLength, true);

            await writer.write(eocd);

            // Close the underlying writer
            await writer.close();
        };
    }
}

export { ZipFileSystem };
