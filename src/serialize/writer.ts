// defines the interface for a stream writer class. all functions are async.
interface Writer {
    // write data to the stream
    write(data: Uint8Array): void | Promise<void>;

    // close the writing stream. return value depends on writer implementation.
    close(): any | Promise<any>;
}

export { type Writer };
