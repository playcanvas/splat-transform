import { FileHandle } from 'node:fs/promises';

// defines the interface for a stream writer class. all functions are async.
interface Writer {
    // write data to the stream
    write(data: Uint8Array): void | Promise<void>;

    // close the writing stream. return value depends on writer implementation.
    close(): any | Promise<any>;
}

// write data to a file stream
class FileWriter implements Writer {
    write: (data: Uint8Array) => void;
    close: () => void;

    constructor(stream: FileHandle) {
        let cursor = 0;

        this.write = async (data: Uint8Array) => {
            cursor += data.byteLength;
            await stream.write(data);
        };

        this.close = async () => {
            await stream.truncate(cursor);
            return true;
        };
    }
}

// write data to a memory buffer
class BufferWriter implements Writer {
    write: (data: Uint8Array) => void;
    close: () => Uint8Array[];

    constructor() {
        const buffers: Uint8Array[] = [];
        let buffer: Uint8Array;
        let cursor = 0;

        this.write = (data: Uint8Array) => {
            let readcursor = 0;

            while (readcursor < data.byteLength) {
                const readSize = data.byteLength - readcursor;

                // allocate buffer
                if (!buffer) {
                    buffer = new Uint8Array(Math.max(5 * 1024 * 1024, readSize));
                }

                const writeSize = buffer.byteLength - cursor;
                const copySize = Math.min(readSize, writeSize);

                buffer.set(data.subarray(readcursor, readcursor + copySize), cursor);

                readcursor += copySize;
                cursor += copySize;

                if (cursor === buffer.byteLength) {
                    buffers.push(buffer);
                    buffer = null;
                    cursor = 0;
                }
            }
        };

        this.close = () => {
            if (buffer) {
                buffers.push(new Uint8Array(buffer.buffer, 0, cursor));
                buffer = null;
                cursor = 0;
            }
            return buffers;
        };
    }
}

class ProgressWriter implements Writer {
    write: (data: Uint8Array) => void;
    close: () => any;

    constructor(writer: Writer, totalBytes: number, progress?: (progress: number, total: number) => void) {
        let cursor = 0;

        this.write = async (data: Uint8Array) => {
            cursor += data.byteLength;
            await writer.write(data);
            progress?.(cursor, totalBytes);
        };

        this.close = () => {
            if (cursor !== totalBytes) {
                throw new Error(`ProgressWriter: expected ${totalBytes} bytes, but wrote ${cursor} bytes`);
            }
            progress?.(cursor, totalBytes);
            return totalBytes;
        };
    }
}

export { Writer, FileWriter, BufferWriter, ProgressWriter };
