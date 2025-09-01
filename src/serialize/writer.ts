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

export { Writer, FileWriter };
