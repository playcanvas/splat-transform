import { DataTable } from './data-table';
import { basename, type ProgressCallback, type ReadSource, ReadFileSystem, ZipReadFileSystem } from './io/read';
import { readKsplat, readLcc, readMjs, readPly, readSog, readSplat, readSpz } from './readers';
import { Options, Param } from './types';
import { logger } from './utils';

/**
 * Open a single source while driving a per-file progress bar named after the
 * file's basename. The optional `onProgress` callback is also forwarded so
 * library consumers receive the same `(loaded, total)` stream.
 *
 * @param fileSystem - File system to open the source on.
 * @param filename - Path to the file to open.
 * @param onProgress - Optional callback invoked with `(loaded, total)` from
 * the source's read events.
 * @returns The opened source and a function to end the associated bar.
 */
const openWithBar = async (
    fileSystem: ReadFileSystem,
    filename: string,
    onProgress?: ProgressCallback
): Promise<{ source: ReadSource; endBar: () => void }> => {
    const bar = logger.bar(basename(filename), 100);
    let prev = 0;
    const source = await fileSystem.createSource(filename, (loaded, total) => {
        onProgress?.(loaded, total);
        if (!total) return;
        const ticks = Math.floor((loaded / total) * 100);
        const delta = ticks - prev;
        prev = ticks;
        if (delta > 0) bar.tick(delta);
    });
    return { source, endBar: () => bar.end() };
};

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
    /** Optional callback for read progress reporting. */
    onProgress?: ProgressCallback;
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
    const { filename, inputFormat, options, params, fileSystem, onProgress } = readFileOptions;

    let result: DataTable[];

    logger.debug(`reading '${filename}'`);

    if (inputFormat === 'mjs') {
        result = [await readMjs(filename, params)];
    } else if (inputFormat === 'sog') {
        const lowerFilename = filename.toLowerCase();
        if (lowerFilename.endsWith('.sog')) {
            // Outer .sog ZIP read drives a single bar named after the archive;
            // inner readSog runs against ZipReadFileSystem with per-file bars
            // suppressed (the actual disk reads are tracked by the outer bar).
            const { source, endBar } = await openWithBar(fileSystem, filename, onProgress);
            const zipFs = new ZipReadFileSystem(source);
            try {
                result = [await readSog(zipFs, 'meta.json', undefined, false)];
            } finally {
                zipFs.close();
                endBar();
            }
        } else {
            // Loose SOG: reader draws one bar per payload file.
            result = [await readSog(fileSystem, filename, onProgress)];
        }
    } else if (inputFormat === 'lcc') {
        // LCC reader draws bars for its large payload files (data.bin, shcoef.bin).
        result = await readLcc(fileSystem, filename, options, onProgress);
    } else {
        // Single-source formats: one bar named after the file.
        const { source, endBar } = await openWithBar(fileSystem, filename, onProgress);
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
            endBar();
        }
    }

    return result;
};

export { readFile, getInputFormat, type InputFormat, type ReadFileOptions };
