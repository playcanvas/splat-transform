import { FileSystem, Writer } from './file-system';

// write data to a memory buffer
class MemoryWriter implements Writer {
    write: (data: Uint8Array) => Promise<void>;
    close: () => Promise<void>;

    constructor(onclose: (buffers: Uint8Array[]) => void) {
        const buffers: Uint8Array[] = [];
        let buffer: Uint8Array;
        let cursor = 0;

        this.write = async (data: Uint8Array) => {
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

        this.close = async () => {
            if (buffer) {
                buffers.push(new Uint8Array(buffer.buffer, 0, cursor));
                buffer = null;
                cursor = 0;
            }
            onclose(buffers);
        };
    }
}

class MemoryFileSystem implements FileSystem {
    results: Map<string, Uint8Array> = new Map();

    async createWriter(filename: string): Promise<Writer> {
        return new MemoryWriter((result: Uint8Array[]) => {
            // combine buffers
            if (result.length === 1) {
                this.results.set(filename, result[0]);
            } else {
                const combined = new Uint8Array(result.reduce((total, buf) => total + buf.byteLength, 0));
                let offset = 0;
                for (let i = 0; i < result.length; ++i) {
                    combined.set(result[i], offset);
                    offset += result[i].byteLength;
                }
                this.results.set(filename, combined);
            }
        });
    }

    async mkdir(path: string): Promise<void> {
        // no-op
    }
}

export { MemoryFileSystem };
