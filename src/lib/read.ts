import { type ChunkSource, createChunkDataPool } from './chunk';
import { dataTableToChunkSource } from './compat/data-table';
import { ReadFileSystem, ZipReadFileSystem } from './io/read';
import { readKsplat, readLcc, readLcc2, readMjs, readPly, readSogSource, readSplat, readSpz } from './readers';
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
 * - `lcc2` - XGrids LCC2 (octree) format
 * - `mjs` - JavaScript module generator
 */
type InputFormat = 'mjs' | 'ksplat' | 'splat' | 'sog' | 'ply' | 'spz' | 'lcc' | 'lcc2';

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
// Strip a trailing `?...` querystring and/or `#...` fragment from the
// *basename* so that extension sniffing works for URL-shaped inputs:
//   - full URLs:   `https://host/scene.sog?token=...`
//   - CLI splits:  `scene.sog?token=...` (resolveInput passes the bare leaf
//                  + query down to readFile so the initial fetch carries it)
// Only the basename (text after the last `/` or `\`) is considered, so
// POSIX paths containing `?` or `#` in *directory* segments are left
// untouched. Local files literally named with `?`/`#` in the basename are
// an unsupported edge case (the extension would be ambiguous anyway).
const stripQueryAndHash = (filename: string): string => {
    const lastSep = Math.max(filename.lastIndexOf('/'), filename.lastIndexOf('\\'));
    const basenameStart = lastSep + 1;
    const q = filename.slice(basenameStart).search(/[?#]/);
    return q < 0 ? filename : filename.slice(0, basenameStart + q);
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
    } else if (lowerFilename.endsWith('.lcc2')) {
        return 'lcc2';
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
 * Reads a Gaussian splat file and returns its data as one or more
 * {@link ChunkSource}s (one per LOD/sub-table for multi-table formats like LCC).
 *
 * Readers are chunk-native: `ply`/`splat`/`spz` return **lazy** sources whose
 * `close()` releases the underlying file; whole-blob formats (`sog`/`mjs`/`ksplat`)
 * and the LCC families are decoded eagerly and returned as resident sources.
 * Callers that need a `DataTable` materialize at their own boundary (and call
 * `source.close()` when done).
 *
 * Per-format progress (decoding bars, multi-payload bars) is emitted directly
 * by each reader through the global {@link logger}; install a renderer via
 * `logger.setRenderer(...)` to consume those events.
 *
 * @param readFileOptions - Options specifying the file to read and how to read it.
 * @returns Promise resolving to an array of chunk sources containing the splat data.
 *
 * @example
 * ```ts
 * import { readFile, getInputFormat, UrlReadFileSystem } from '@playcanvas/splat-transform';
 *
 * const filename = 'scene.ply';
 * const fileSystem = new UrlReadFileSystem('https://example.com/');
 * const sources = await readFile({
 *     filename,
 *     inputFormat: getInputFormat(filename),
 *     options: {},
 *     params: [],
 *     fileSystem
 * });
 * ```
 */
const readFile = async (readFileOptions: ReadFileOptions): Promise<ChunkSource[]> => {
    const { filename, inputFormat, options, params, fileSystem } = readFileOptions;

    // Whole-blob / multi-source formats: the reader opens and closes its own
    // source(s) internally and returns DataTable(s); wrap each as a resident
    // ChunkSource so readFile uniformly yields sources.
    if (inputFormat === 'mjs') {
        return [dataTableToChunkSource(await readMjs(filename, params))];
    }
    if (inputFormat === 'sog') {
        const lowerFilename = stripQueryAndHash(filename).toLowerCase();
        const pool = createChunkDataPool();
        if (lowerFilename.endsWith('.sog')) {
            // Outer .sog is a ZIP container - mount it and decode chunk-native
            // (textures resident, rows expanded on demand). readSogSource loads and
            // decodes every texture up front, so the zip can close once it returns.
            const source = await fileSystem.createSource(filename);
            const zipFs = new ZipReadFileSystem(source);
            try {
                return [await readSogSource(zipFs, 'meta.json', pool)];
            } finally {
                zipFs.close();
            }
        }
        return [await readSogSource(fileSystem, filename, pool)];
    }
    if (inputFormat === 'lcc') {
        return (await readLcc(fileSystem, filename, options)).map(dt => dataTableToChunkSource(dt));
    }
    if (inputFormat === 'lcc2') {
        return (await readLcc2(fileSystem, filename, options)).map(dt => dataTableToChunkSource(dt));
    }

    // Single-source binary formats over a seekable ReadSource. Lazy readers
    // (ply/splat/spz) hand the source's lifetime to the returned ChunkSource —
    // its close() releases it — so the source is NOT closed here on success;
    // ksplat is eager and is done with the source once decoded. On error the
    // source is released (close() is idempotent).
    const source = await fileSystem.createSource(filename);
    const pool = createChunkDataPool();
    try {
        if (inputFormat === 'ply') return [await readPly(source, pool)];
        if (inputFormat === 'splat') return [await readSplat(source, pool)];
        if (inputFormat === 'spz') return [await readSpz(source, pool)];
        if (inputFormat === 'ksplat') {
            const dataTable = await readKsplat(source);
            source.close();
            return [dataTableToChunkSource(dataTable)];
        }
        throw new Error(`Unsupported input format: ${inputFormat}`);
    } catch (e) {
        source.close();
        throw e;
    }
};

export { readFile, getInputFormat, type InputFormat, type ReadFileOptions };
