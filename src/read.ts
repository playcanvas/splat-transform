import { open } from 'node:fs/promises';

import { DataTable } from './data-table/data-table';
import { readKsplat } from './readers/read-ksplat';
import { readLcc } from './readers/read-lcc';
import { readMjs } from './readers/read-mjs';
import { readPly } from './readers/read-ply';
import { readSog } from './readers/read-sog';
import { readSplat } from './readers/read-splat';
import { readSpz } from './readers/read-spz';
import { ReadFileSystem } from './serialize/read-file-system';
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
};

const readFile = async (readFileOptions: ReadFileOptions, fs: ReadFileSystem): Promise<DataTable[]> => {
    const { filename, inputFormat, options, params } = readFileOptions;

    let result: DataTable[];

    logger.info(`reading '${filename}'...`);

    if (inputFormat === 'mjs') {
        result = [await readMjs(filename, params)];
    } else if (inputFormat === 'ksplat') {
        result = [await readKsplat(filename, fs)];
    } else if (inputFormat === 'splat') {
        result = [await readSplat(filename, fs)];
    } else if (inputFormat === 'ply') {
        result = [await readPly(filename, fs)];
    } else if (inputFormat === 'spz') {
        result = [await readSpz(filename, fs)];
    } else {
        // sog and lcc need FileHandle for random access
        const inputFile = await open(filename, 'r');

        if (inputFormat === 'sog') {
            result = [await readSog(inputFile, filename)];
        } else if (inputFormat === 'lcc') {
            result = await readLcc(inputFile, filename, options);
        }

        await inputFile.close();
    }

    return result;
};

export { readFile, getInputFormat, type InputFormat };
