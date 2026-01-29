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

/**
 * Supported input file formats for Gaussian splat data.
 *
 * - `ply` - PLY format (standard 3DGS training output)
 * - `splat` - Antimatter15 splat format
 * - `ksplat` - Kevin Kwok's compressed splat format
 * - `spz` - Niantic Labs compressed format
 * - `sog` - PlayCanvas SOG format (WebP-compressed)
 * - `lcc` - XGrids LCC format
 * - `mjs` - JavaScript module generator
 */
type InputFormat = 'mjs' | 'ksplat' | 'splat' | 'sog' | 'ply' | 'spz' | 'lcc';

/**
 * Determines the input format based on file extension.
 *
 * @param filename - The filename to analyze.
 * @returns The detected input format.
 * @throws Error if the file extension is not recognized.
 *
 * @example
 * ```ts
 * const format = getInputFormat('scene.ply');  // returns 'ply'
 * const format2 = getInputFormat('scene.splat');  // returns 'splat'
 * ```
 */
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

/**
 * Options for reading a Gaussian splat file.
 */
type ReadFileOptions = {
    /** Path to the input file. */
    filename: string;
    /** The format of the input file. */
    inputFormat: InputFormat;
    /** Processing options. */
    options: Options;
    /** Parameters for generator modules (.mjs files). */
    params: Param[];
    /** File system abstraction for reading files. */
    fileSystem: ReadFileSystem;
};

/**
 * Reads a Gaussian splat file and returns its data as one or more DataTables.
 *
 * Supports multiple input formats including PLY, splat, ksplat, spz, SOG, and LCC.
 * Some formats (like LCC) may return multiple DataTables for different LOD levels.
 *
 * @param readFileOptions - Options specifying the file to read and how to read it.
 * @returns Promise resolving to an array of DataTables containing the splat data.
 *
 * @example
 * ```ts
 * import { readFile, getInputFormat, UrlReadFileSystem } from '@playcanvas/splat-transform';
 *
 * const filename = 'scene.ply';
 * const fileSystem = new UrlReadFileSystem('https://example.com/');
 * const tables = await readFile({
 *     filename,
 *     inputFormat: getInputFormat(filename),
 *     options: {},
 *     params: [],
 *     fileSystem
 * });
 * ```
 */
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

export { readFile, getInputFormat, type InputFormat, type ReadFileOptions };
