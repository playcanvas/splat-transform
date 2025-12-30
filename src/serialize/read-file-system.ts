import { ReadSource } from './read-source';

interface ReadFileSystem {
    createReader(filename: string): ReadSource | Promise<ReadSource>;

    getFiles(path: string): string[] | Promise<string[]>;
}

export { type ReadFileSystem };
