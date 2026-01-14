import { DataTable } from './data-table/data-table';
import { ReadFileSystem } from './io/read';
import { ZipReadFileSystem } from './io/read/zip-file-system';
import { readKsplat } from './readers/read-ksplat';
import { readLcc } from './readers/read-lcc';
import { readMjs } from './readers/read-mjs';
import { readPly } from './readers/read-ply';
import { readSog } from './readers/read-sog';
import { readSplat } from './readers/read-splat';
import { readSpz } from './readers/read-spz';
import { Options, Param } from './types';
import { logger } from './utils/logger';

type InputFormat = 'mjs' | 'ksplat' | 'splat' | 'sog' | 'ply' | 'spz' | 'lcc';

const getInputFormat = (filename: string): InputFormat => {
    const lowerFilename = filename.toLowerCase();

    if (lowerFilename.endsWith('.mjs')) {
        return 'mjs';
    } else if (lowerFilename.endsWith('.ksplat')) {
        return 'ksplat';
    } else if (lowerFilename.endsWith('.splat')) {
        return 'splat';
    } else if (lowerFilename.endsWith('.sog') || lowerFilename.endsWith('meta.json')) {
        return 'sog';
    } else if (lowerFilename.endsWith('.ply')) {
        return 'ply';
    } else if (lowerFilename.endsWith('.spz')) {
        return 'spz';
    } else if (lowerFilename.endsWith('.lcc')) {
        return 'lcc';
    }

    throw new Error(`Unsupported input file type: ${filename}`);
};

type ReadFileOptions = {
    filename: string;
    inputFormat: InputFormat;
    options: Options;
    params: Param[];
    fileSystem: ReadFileSystem;
};

const readFile = async (readFileOptions: ReadFileOptions): Promise<DataTable[]> => {
    const { filename, inputFormat, options, params, fileSystem } = readFileOptions;

    let result: DataTable[];

    logger.log(`reading '${filename}'...`);

    if (inputFormat === 'mjs') {
        result = [await readMjs(filename, params)];
    } else if (inputFormat === 'sog') {
        const lowerFilename = filename.toLowerCase();
        if (lowerFilename.endsWith('.sog')) {
            const source = await fileSystem.createSource(filename);
            const zipFs = new ZipReadFileSystem(source);
            try {
                result = [await readSog(zipFs, 'meta.json')];
            } finally {
                zipFs.close();
            }
        } else {
            result = [await readSog(fileSystem, filename)];
        }
    } else if (inputFormat === 'lcc') {
        // LCC uses ReadFileSystem for multi-file access
        result = await readLcc(fileSystem, filename, options);
    } else {
        // All other formats use ReadSource
        const source = await fileSystem.createSource(filename);
        try {
            if (inputFormat === 'ply') {
                result = [await readPly(source)];
            } else if (inputFormat === 'ksplat') {
                result = [await readKsplat(source)];
            } else if (inputFormat === 'splat') {
                result = [await readSplat(source)];
            } else if (inputFormat === 'spz') {
                result = [await readSpz(source)];
            }
        } finally {
            source.close();
        }
    }

    return result;
};

export { readFile, getInputFormat, type InputFormat };
