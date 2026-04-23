import { DataTable } from './data-table';
import { ReadFileSystem, ZipReadFileSystem } from './io/read';
import { readKsplat, readLcc, readMjs, readPly, readSog, readSplat, readSpz } from './readers';
import { Options, Param } from './types';

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
// Strip a trailing `?...` querystring and/or `#...` fragment so that
// extension sniffing works for http(s):// URL inputs (e.g. presigned
// URLs like `scene.sog?token=...`).
const stripQueryAndHash = (filename: string): string => {
    const q = filename.search(/[?#]/);
    return q < 0 ? filename : filename.slice(0, q);
};

const getInputFormat = (filename: string): InputFormat => {
    const lowerFilename = stripQueryAndHash(filename).toLowerCase();

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
 * Per-format progress (decoding bars, multi-payload bars) is emitted directly
 * by each reader through the global {@link logger}; install a renderer via
 * `logger.setRenderer(...)` to consume those events.
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

    if (inputFormat === 'mjs') {
        result = [await readMjs(filename, params)];
    } else if (inputFormat === 'sog') {
        const lowerFilename = stripQueryAndHash(filename).toLowerCase();
        if (lowerFilename.endsWith('.sog')) {
            // Outer .sog is a ZIP container - mount it and let the inner SOG
            // reader drive its own decode bar against the zipped payloads.
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
        result = await readLcc(fileSystem, filename, options);
    } else {
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
