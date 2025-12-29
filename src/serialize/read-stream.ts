type ResponseTypes = Response | Request | string;
type ReadSourceType = ResponseTypes | ArrayBuffer | Blob | ReadableStream<Uint8Array>;
type PullFunc = (target: Uint8Array) => Promise<number>;
type ProgressCallback = (readBytes: number, totalBytes?: number) => void;

type ReadStreamOptions = {
    totalBytes?: number;
    approximateTotalBytes?: number;
    pull: PullFunc;
    close?: () => void;
    onProgress?: ProgressCallback;
};

class ReadStream {
    readBytes: number;
    options: ReadStreamOptions;

    constructor(options: ReadStreamOptions) {
        this.readBytes = 0;
        this.options = options;
    }

    get totalBytes() {
        return this.options.totalBytes;
    }

    get approximateTotalBytes() {
        return this.options.approximateTotalBytes;
    }

    get estimatedTotalBytes(): number | undefined {
        return this.totalBytes ?? this.approximateTotalBytes;
    }

    async pull(target: Uint8Array): Promise<number> {
        const result = await this.options.pull(target);
        this.readBytes += result;
        this.options.onProgress?.(this.readBytes, this.estimatedTotalBytes);
        return result;
    }

    close() {
        this.options.close?.();
    }
}

type WrapReadableStreamOptions = {
    totalBytes?: number;
    approximateTotalBytes?: number;
    stream: ReadableStream<Uint8Array>;
    onProgress?: ProgressCallback;
};

const wrapReadableStream = (options: WrapReadableStreamOptions): ReadStream => {
    const incoming: Uint8Array[] = [];
    let incomingBytes = 0;

    const reader = options.stream.getReader();

    const pull = async (target: Uint8Array): Promise<number> => {
        // read the next result.byteLength bytes into result
        while (incomingBytes < target.byteLength) {
            const { done, value } = await reader.read();
            if (done) {
                break;
            }
            incoming.push(value);
            incomingBytes += value.byteLength;
        }

        // copy data into result
        let offset = 0;
        while (offset < target.byteLength && incoming.length > 0) {
            const chunk = incoming[0];
            const copyBytes = Math.min(chunk.byteLength, target.byteLength - offset);
            target.set(chunk.subarray(0, copyBytes), offset);
            offset += copyBytes;
            incomingBytes -= copyBytes;

            if (copyBytes < chunk.byteLength) {
                // remove copied bytes from chunk
                incoming[0] = chunk.subarray(copyBytes);
            } else {
                // remove chunk
                incoming.shift();
            }
        }

        return offset;
    };

    const close = () => {
        reader.cancel();
    };

    return new ReadStream({
        totalBytes: options.totalBytes,
        approximateTotalBytes: options.approximateTotalBytes,
        pull,
        close,
        onProgress: options.onProgress
    });
};

const getResponse = async (source: ResponseTypes): Promise<Response> => {
    if (source instanceof Response) {
        return source;
    }
    const response = await fetch(source instanceof Request ? source : new Request(source));
    if (!response.ok) {
        throw new Error(`Failed to fetch: ${response.status} ${response.statusText}`);
    }
    return response;
};

type ReadSourceOptions = {
    source: ReadSourceType;
    totalBytes?: number;
    approximateTotalBytes?: number;
    onProgress?: ProgressCallback;
};

class ReadSource {
    options: ReadSourceOptions;
    private consumed = false;

    constructor(options: ReadSourceOptions) {
        this.options = options;
    }

    private markConsumed() {
        if (this.consumed) {
            throw new Error('ReadSource has already been consumed');
        }
        this.consumed = true;
    }

    async getReadStream(): Promise<ReadStream> {
        this.markConsumed();
        const { source, onProgress } = this.options;

        if (source instanceof ArrayBuffer) {
            const totalBytes = source.byteLength;
            let cursor = 0;
            const pull = async (target: Uint8Array) => {
                const bytes = Math.min(target.byteLength, totalBytes - cursor);
                target.set(new Uint8Array(source, cursor, bytes));
                cursor += bytes;
                return bytes;
            };
            return new ReadStream({ totalBytes, pull, onProgress });
        } else if (source instanceof Blob) {
            return wrapReadableStream({
                totalBytes: source.size,
                stream: source.stream(),
                onProgress
            });
        } else if (source instanceof ReadableStream) {
            return wrapReadableStream({
                totalBytes: this.options.totalBytes,
                approximateTotalBytes: this.options.approximateTotalBytes,
                stream: source,
                onProgress
            });
        }

        const response = await getResponse(source as ResponseTypes);

        if (!response.body) {
            throw new Error('Response has no body');
        }

        const contentLength = response.headers.get('Content-Length');

        return wrapReadableStream({
            approximateTotalBytes: contentLength ? parseInt(contentLength, 10) : undefined,
            stream: response.body,
            onProgress
        });
    }

    async arrayBuffer(): Promise<ArrayBuffer> {
        this.markConsumed();
        const { source, onProgress } = this.options;

        if (source instanceof ArrayBuffer) {
            onProgress?.(source.byteLength, source.byteLength);
            return source;
        } else if (source instanceof Blob) {
            const buffer = await source.arrayBuffer();
            onProgress?.(buffer.byteLength, buffer.byteLength);
            return buffer;
        } else if (source instanceof ReadableStream) {
            // read entire stream into a single ArrayBuffer
            const reader = source.getReader();
            const chunks: Uint8Array[] = [];
            let totalLength = 0;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                chunks.push(value);
                totalLength += value.byteLength;
                onProgress?.(totalLength, this.options.totalBytes ?? this.options.approximateTotalBytes);
            }

            const result = new Uint8Array(totalLength);
            let offset = 0;
            for (const chunk of chunks) {
                result.set(chunk, offset);
                offset += chunk.byteLength;
            }

            return result.buffer;
        }

        const response = await getResponse(source as ResponseTypes);
        const buffer = await response.arrayBuffer();
        onProgress?.(buffer.byteLength, buffer.byteLength);
        return buffer;
    }
}

export { ReadStream, ReadSource, type ReadSourceOptions, type ProgressCallback };
