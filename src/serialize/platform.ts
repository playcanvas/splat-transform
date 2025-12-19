import { Writer } from './writer';

interface Platform {
    // create a writer for the given filename
    createWriter(filename: string): Promise<Writer>;

    // create a directory at the given path
    mkdir(path: string): Promise<void>;
}

export { type Platform };
