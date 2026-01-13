import { ReadFileSystem, ProgressCallback } from './read-file-system';
import { ReadSource } from './read-source';
import { ReadStream } from './read-stream';

/**
 * ReadStream implementation for reading from fetch responses.
 */
class UrlReadStream extends ReadStream {
    private reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
    private response: Response | null = null;
    private buffer: Uint8Array | null = null;
    private bufferOffset: number = 0;
    private closed: boolean = false;
    private progress: ProgressCallback | undefined;
    private totalSize: number | undefined;

    constructor(response: Response, expectedSize?: number, progress?: ProgressCallback) {
        super(expectedSize);
        this.response = response;
        this.totalSize = expectedSize;
        this.progress = progress;

        if (response.body) {
            this.reader = response.body.getReader();
        }
    }

    async pull(target: Uint8Array): Promise<number> {
        if (this.closed || !this.reader) {
            return 0;
        }

        let targetOffset = 0;

        // First, consume any leftover data from previous read
        if (this.buffer && this.bufferOffset < this.buffer.length) {
            const remaining = this.buffer.length - this.bufferOffset;
            const toCopy = Math.min(remaining, target.length);
            target.set(this.buffer.subarray(this.bufferOffset, this.bufferOffset + toCopy));
            this.bufferOffset += toCopy;
            targetOffset += toCopy;
            this.bytesRead += toCopy;

            // Report progress
            if (this.progress) {
                this.progress(this.bytesRead, this.totalSize);
            }

            // If we filled the target, return
            if (targetOffset >= target.length) {
                return targetOffset;
            }

            // Clear exhausted buffer
            if (this.bufferOffset >= this.buffer.length) {
                this.buffer = null;
                this.bufferOffset = 0;
            }
        }

        // Read from stream
        while (targetOffset < target.length) {
            const { done, value } = await this.reader.read();

            if (done || !value) {
                break;
            }

            const toCopy = Math.min(value.length, target.length - targetOffset);
            target.set(value.subarray(0, toCopy), targetOffset);
            targetOffset += toCopy;
            this.bytesRead += toCopy;

            // Report progress
            if (this.progress) {
                this.progress(this.bytesRead, this.totalSize);
            }

            // Store leftover for next pull
            if (toCopy < value.length) {
                this.buffer = value;
                this.bufferOffset = toCopy;
                break;
            }
        }

        return targetOffset;
    }

    close(): void {
        this.closed = true;
        if (this.reader) {
            this.reader.cancel();
            this.reader = null;
        }
        this.response = null;
        this.buffer = null;
    }
}

/**
 * A ReadStream that lazily initiates the fetch on first pull.
 */
class LazyUrlReadStream extends ReadStream {
    private url: string;
    private headers: Record<string, string>;
    private innerStream: UrlReadStream | null = null;
    private fetchPromise: Promise<UrlReadStream> | null = null;
    private closed: boolean = false;
    private progress: ProgressCallback | undefined;

    constructor(url: string, headers: Record<string, string>, expectedSize?: number, progress?: ProgressCallback) {
        super(expectedSize);
        this.url = url;
        this.headers = headers;
        this.progress = progress;
    }

    private async ensureStream(): Promise<UrlReadStream | null> {
        if (this.closed) {
            return null;
        }

        if (this.innerStream) {
            return this.innerStream;
        }

        if (!this.fetchPromise) {
            this.fetchPromise = (async () => {
                const response = await fetch(this.url, { headers: this.headers });
                if (!response.ok) {
                    throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
                }
                return new UrlReadStream(response, this.expectedSize, this.progress);
            })();
        }

        this.innerStream = await this.fetchPromise;
        return this.innerStream;
    }

    async pull(target: Uint8Array): Promise<number> {
        const stream = await this.ensureStream();
        if (!stream) {
            return 0;
        }

        const result = await stream.pull(target);
        this.bytesRead = stream.bytesRead;
        return result;
    }

    close(): void {
        this.closed = true;
        if (this.innerStream) {
            this.innerStream.close();
            this.innerStream = null;
        }
    }
}

/**
 * ReadSource implementation for URLs using fetch with Range headers.
 * Size is from Content-Length (may be approximate for compressed responses).
 * Seekable via Range header requests.
 */
class UrlReadSource implements ReadSource {
    readonly size: number | undefined;
    readonly seekable: boolean = true;

    private url: string;
    private closed: boolean = false;
    private progress: ProgressCallback | undefined;

    constructor(url: string, size: number | undefined, progress?: ProgressCallback) {
        this.url = url;
        this.size = size;
        this.progress = progress;
    }

    read(start?: number, end?: number): ReadStream {
        if (this.closed) {
            throw new Error('Source has been closed');
        }

        // Calculate expected size for this range
        let expectedSize: number | undefined;
        if (start !== undefined || end !== undefined) {
            const rangeStart = start ?? 0;
            const rangeEnd = end ?? this.size;
            if (rangeEnd !== undefined) {
                expectedSize = rangeEnd - rangeStart;
            }
        } else {
            expectedSize = this.size;
        }

        // Create fetch promise with appropriate headers
        const headers: Record<string, string> = {};
        if (start !== undefined || end !== undefined) {
            const rangeStart = start ?? 0;
            const rangeEnd = end !== undefined ? end - 1 : ''; // HTTP Range is inclusive
            headers.Range = `bytes=${rangeStart}-${rangeEnd}`;
        }

        // We need to return a ReadStream synchronously, so we create one that
        // will fetch on first pull
        return new LazyUrlReadStream(this.url, headers, expectedSize, this.progress);
    }

    close(): void {
        this.closed = true;
    }
}

/**
 * ReadFileSystem for reading from URLs using fetch.
 * Supports optional base URL for relative paths.
 */
class UrlReadFileSystem implements ReadFileSystem {
    private baseUrl: string;

    /**
     * @param baseUrl - Optional base URL to prepend to filenames
     */
    constructor(baseUrl: string = '') {
        this.baseUrl = baseUrl;
    }

    async createSource(filename: string, progress?: ProgressCallback): Promise<ReadSource> {
        const url = this.baseUrl ? new URL(filename, this.baseUrl).href : filename;

        // Make a HEAD request to get the size without downloading the body
        const headResponse = await fetch(url, { method: 'HEAD' });
        if (!headResponse.ok) {
            throw new Error(`HTTP error ${headResponse.status}: ${headResponse.statusText}`);
        }

        const contentLength = headResponse.headers.get('Content-Length');
        const size = contentLength ? parseInt(contentLength, 10) : undefined;

        // Report initial progress
        if (progress) {
            progress(0, size);
        }

        return new UrlReadSource(url, size, progress);
    }
}

export { UrlReadStream, UrlReadSource, UrlReadFileSystem };
