import { FileHandle, open, stat } from 'node:fs/promises';

import { ReadFileSystem, ProgressCallback } from './read-file-system';
import { ReadSource } from './read-source';
import { ReadStream } from './read-stream';

/**
 * ReadStream implementation for reading from Node.js file handles.
 */
class NodeReadStream extends ReadStream {
    private fileHandle: FileHandle;
    private position: number;
    private end: number;
    private closed: boolean = false;

    constructor(fileHandle: FileHandle, start: number, end: number) {
        super(end - start);
        this.fileHandle = fileHandle;
        this.position = start;
        this.end = end;
    }

    async pull(target: Uint8Array): Promise<number> {
        if (this.closed) {
            return 0;
        }

        const remaining = this.end - this.position;
        if (remaining <= 0) {
            return 0;
        }

        const bytesToRead = Math.min(target.length, remaining);
        const { bytesRead } = await this.fileHandle.read(target, 0, bytesToRead, this.position);

        this.position += bytesRead;
        this.bytesRead += bytesRead;
        return bytesRead;
    }

    close(): void {
        this.closed = true;
    }
}

/**
 * ReadSource implementation for Node.js file handles.
 * Size is exact from stat(). Always seekable via positioned reads.
 */
class NodeReadSource implements ReadSource {
    readonly size: number;
    readonly seekable: boolean = true;

    private fileHandle: FileHandle;
    private closed: boolean = false;

    constructor(fileHandle: FileHandle, size: number) {
        this.fileHandle = fileHandle;
        this.size = size;
    }

    read(start: number = 0, end: number = this.size): ReadStream {
        if (this.closed) {
            throw new Error('Source has been closed');
        }

        // Clamp range to valid bounds
        const clampedStart = Math.max(0, Math.min(start, this.size));
        const clampedEnd = Math.max(clampedStart, Math.min(end, this.size));

        return new NodeReadStream(this.fileHandle, clampedStart, clampedEnd);
    }

    close(): void {
        this.closed = true;
        this.fileHandle.close();
    }
}

/**
 * ReadFileSystem for reading from the local filesystem using Node.js fs module.
 */
class NodeReadFileSystem implements ReadFileSystem {
    async createSource(filename: string, progress?: ProgressCallback): Promise<ReadSource> {
        const fileStats = await stat(filename);
        const fileHandle = await open(filename, 'r');

        // Report initial progress
        if (progress) {
            progress(0, fileStats.size);
        }

        return new NodeReadSource(fileHandle, fileStats.size);
    }
}

export { NodeReadStream, NodeReadSource, NodeReadFileSystem };
