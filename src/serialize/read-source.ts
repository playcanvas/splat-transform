import { type SizedStream, type ProgressCallback, ReadStream } from './read-stream';

// string is a fetch URL
type ReadSourceType = Response | Request | string | ArrayBuffer | Uint8Array | Blob | SizedStream;

type ReadSourceOptions = {
    source: ReadSourceType;
    onProgress?: ProgressCallback;
};

type ReadSourceOptionsFactory = () => ReadSourceOptions | Promise<ReadSourceOptions | null> | null;

type ResponseTypes = Response | Request | string;

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

class ReadSource {
    private factory: ReadSourceOptionsFactory;

    constructor(factory: ReadSourceOptionsFactory) {
        this.factory = factory;
    }

    static fromOptions(options: ReadSourceOptions): ReadSource {
        let opts: ReadSourceOptions | null = options;
        return new ReadSource(() => {
            const result = opts;
            opts = null;
            return result;
        });
    }

    async getReadStream(): Promise<ReadStream> {
        const options = await this.factory();
        if (!options) {
            throw new Error('ReadSource has already been consumed');
        }
        const { source, onProgress } = options;

        if (source instanceof ArrayBuffer || source instanceof Uint8Array) {
            const data = source instanceof Uint8Array ? source : new Uint8Array(source);
            return new ReadStream({
                stream: new ReadableStream({
                    start(controller) {
                        controller.enqueue(data);
                        controller.close();
                    }
                }),
                totalBytes: data.byteLength
            }, onProgress);
        } else if (source instanceof Blob) {
            return new ReadStream({
                stream: source.stream(),
                totalBytes: source.size
            }, onProgress);
        } else if (typeof source === 'object' && 'stream' in source) {
            return new ReadStream(source, onProgress);
        }

        const response = await getResponse(source as ResponseTypes);

        if (!response.body) {
            throw new Error('Response has no body');
        }

        const contentLength = response.headers.get('Content-Length');

        return new ReadStream({
            stream: response.body,
            approximateTotalBytes: contentLength ? parseInt(contentLength, 10) : undefined
        }, onProgress);
    }

    async arrayBuffer(): Promise<ArrayBuffer> {
        const options = await this.factory();
        if (!options) {
            throw new Error('ReadSource has already been consumed');
        }
        const { source, onProgress } = options;

        if (source instanceof ArrayBuffer) {
            onProgress?.(source.byteLength, source.byteLength);
            return source;
        } else if (source instanceof Uint8Array) {
            onProgress?.(source.byteLength, source.byteLength);
            // Return just the viewed portion (Uint8Array may be a view into larger buffer)
            return source.buffer.slice(source.byteOffset, source.byteOffset + source.byteLength) as ArrayBuffer;
        } else if (source instanceof Blob) {
            const buffer = await source.arrayBuffer();
            onProgress?.(buffer.byteLength, buffer.byteLength);
            return buffer;
        } else if (typeof source === 'object' && 'stream' in source) {
            // read entire stream into a single ArrayBuffer
            const reader = source.stream.getReader();
            const chunks: Uint8Array[] = [];
            let totalLength = 0;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                chunks.push(value);
                totalLength += value.byteLength;
                onProgress?.(totalLength, source.totalBytes ?? source.approximateTotalBytes);
            }

            source.close?.();

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

export {
    type ReadSourceOptions,
    type ReadSourceOptionsFactory,
    ReadSource
};
