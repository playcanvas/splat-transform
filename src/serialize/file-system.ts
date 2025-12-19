// defines the interface for a stream writer class
interface Writer {
    // write data to the stream
    write(data: Uint8Array): Promise<void>;

    // close the writing stream. return value depends on writer implementation.
    close(): Promise<void>;
}

interface FileSystem {
    // create a writer for the given filename
    createWriter(filename: string): Promise<Writer>;

    // create a directory at the given path
    mkdir(path: string): Promise<void>;
}

export { type FileSystem, type Writer };
