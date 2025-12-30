// user can provide a readable stream, but we need to know the size (even approximate size)
// in order to show progress.
type SizedStream = {
    stream: ReadableStream<Uint8Array>;
    totalBytes?: number;
    approximateTotalBytes?: number;
    close?: () => void;
};

// totalBytes might be approximate
type ProgressCallback = (readBytes: number, totalBytes?: number) => void;

class ReadStream {
    private sizedStream: SizedStream;
    private onProgress?: ProgressCallback;
    private reader: ReadableStreamDefaultReader<Uint8Array>;
    private incoming: Uint8Array[] = [];
    private incomingBytes = 0;
    private closed = false;

    readBytes = 0;

    constructor(sizedStream: SizedStream, onProgress?: ProgressCallback) {
        this.sizedStream = sizedStream;
        this.onProgress = onProgress;
        this.reader = sizedStream.stream.getReader();
    }

    get totalBytes() {
        return this.sizedStream.totalBytes;
    }

    get approximateTotalBytes() {
        return this.sizedStream.approximateTotalBytes;
    }

    get estimatedTotalBytes(): number | undefined {
        return this.totalBytes ?? this.approximateTotalBytes;
    }

    async pull(target: Uint8Array): Promise<number> {
        // read chunks until we have enough bytes
        while (this.incomingBytes < target.byteLength) {
            const { done, value } = await this.reader.read();
            if (done) {
                break;
            }
            this.incoming.push(value);
            this.incomingBytes += value.byteLength;
        }

        // copy data into target
        let offset = 0;
        while (offset < target.byteLength && this.incoming.length > 0) {
            const chunk = this.incoming[0];
            const copyBytes = Math.min(chunk.byteLength, target.byteLength - offset);
            target.set(chunk.subarray(0, copyBytes), offset);
            offset += copyBytes;
            this.incomingBytes -= copyBytes;

            if (copyBytes < chunk.byteLength) {
                // remove copied bytes from chunk
                this.incoming[0] = chunk.subarray(copyBytes);
            } else {
                // remove chunk
                this.incoming.shift();
            }
        }

        this.readBytes += offset;
        this.onProgress?.(this.readBytes, this.estimatedTotalBytes);

        // auto-close when stream is exhausted
        if (offset === 0) {
            this.close();
        }

        return offset;
    }

    close() {
        if (!this.closed) {
            this.closed = true;
            this.reader.cancel();
            this.sizedStream.close?.();
        }
    }
}

export {
    type SizedStream,
    type ProgressCallback,
    ReadStream
};
