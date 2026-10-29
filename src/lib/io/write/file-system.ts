// defines the interface for a stream writer class
interface Writer {
    // total bytes successfully passed to write() so far
    readonly bytesWritten: number;

    // write data to the stream
    write(data: Uint8Array): void | Promise<void>;

    // close the writing stream, committing the output. return value depends
    // on writer implementation.
    close(): void | Promise<void>;

    // abandon the stream WITHOUT committing: release resources and discard
    // partial output where the implementation can (e.g. delete the temp file
    // instead of renaming it over the destination). Error paths call this
    // instead of close() so a failed write never publishes a truncated file.
    abort(): void | Promise<void>;
}

interface FileSystem {
    // create a writer for the given filename
    createWriter(filename: string): Writer | Promise<Writer>;

    // create a directory at the given path
    mkdir(path: string): Promise<void>;
}

export { type FileSystem, type Writer };
