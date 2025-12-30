import { ReadSource } from './read-source';

interface ReadFileSystem {
    createReader(filename: string): ReadSource | Promise<ReadSource>;
}

export { type ReadFileSystem };
