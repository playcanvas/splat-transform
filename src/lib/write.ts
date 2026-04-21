import { basename } from 'pathe';

import { DataTable } from './data-table';
import { type FileSystem } from './io/write';
import { type DeviceCreator, type Options } from './types';
import { logger } from './utils';
import { writeCompressedPly, writeCsv, writeGlb, writeHtml, writeLod, writePly, writeSog, writeVoxel } from './writers';

/**
 * Run a writer function while displaying a single per-file progress bar named
 * after the file's basename. The bar ticks once on successful completion,
 * giving an atomic 0->100% indicator for write paths that don't expose
 * incremental progress.
 *
 * @param filename - Path to the file being written.
 * @param fn - The async writer body to execute.
 * @returns Whatever `fn` returns.
 */
const withFileBar = async <T>(filename: string, fn: () => Promise<T>): Promise<T> => {
    const bar = logger.bar(basename(filename), 1);
    try {
        const result = await fn();
        bar.tick(1);
        return result;
    } finally {
        bar.end();
    }
};

/**
 * Supported output file formats for Gaussian splat data.
 *
 * - `ply` - Standard PLY format
 * - `compressed-ply` - Compressed PLY format
 * - `glb` - Binary glTF with KHR_gaussian_splatting extension
 * - `csv` - CSV text format (for debugging/analysis)
 * - `sog` - PlayCanvas SOG format (separate files)
 * - `sog-bundle` - PlayCanvas SOG format (bundled into single .sog file)
 * - `lod` - Multi-LOD format with chunked data
 * - `html` - Self-contained HTML viewer (separate assets)
 * - `html-bundle` - Self-contained HTML viewer (all assets embedded)
 * - `voxel` - Sparse voxel octree format for collision detection
 */
type OutputFormat = 'csv' | 'sog' | 'sog-bundle' | 'lod' | 'compressed-ply' | 'ply' | 'glb' | 'html' | 'html-bundle' | 'voxel';

/**
 * Options for writing a Gaussian splat file.
 */
type WriteOptions = {
    /** Path to the output file. */
    filename: string;
    /** The format to write. */
    outputFormat: OutputFormat;
    /** The splat data to write. */
    dataTable: DataTable;
    /** Optional environment/skybox splat data (for LOD format). */
    envDataTable?: DataTable;
    /** Processing options. */
    options: Options;
    /** Optional function to create a GPU device for compression. */
    createDevice?: DeviceCreator;
};

/**
 * Determines the output format based on file extension and options.
 *
 * @param filename - The filename to analyze.
 * @param options - Options that may affect format selection.
 * @returns The detected output format.
 * @throws Error if the file extension is not recognized.
 *
 * @example
 * ```ts
 * const format = getOutputFormat('scene.ply', {});  // returns 'ply'
 * const format2 = getOutputFormat('scene.sog', {});  // returns 'sog-bundle'
 * ```
 */
const getOutputFormat = (filename: string, options: Options): OutputFormat => {
    const lowerFilename = filename.toLowerCase();

    if (lowerFilename.endsWith('.csv')) {
        return 'csv';
    } else if (lowerFilename.endsWith('.voxel.json')) {
        return 'voxel';
    } else if (lowerFilename.endsWith('lod-meta.json')) {
        return 'lod';
    } else if (lowerFilename.endsWith('.sog')) {
        return 'sog-bundle';
    } else if (lowerFilename.endsWith('meta.json')) {
        return 'sog';
    } else if (lowerFilename.endsWith('.compressed.ply')) {
        return 'compressed-ply';
    } else if (lowerFilename.endsWith('.ply')) {
        return 'ply';
    } else if (lowerFilename.endsWith('.glb')) {
        return 'glb';
    } else if (lowerFilename.endsWith('.html')) {
        return options.unbundled ? 'html' : 'html-bundle';
    }

    throw new Error(`Unsupported output file type: ${filename}`);
};

/**
 * Writes Gaussian splat data to a file in the specified format.
 *
 * Supports multiple output formats including PLY, compressed PLY, CSV, SOG, LOD, and HTML.
 *
 * @param writeOptions - Options specifying the data and format to write.
 * @param fs - File system abstraction for writing files.
 *
 * @example
 * ```ts
 * import { writeFile, getOutputFormat, MemoryFileSystem } from '@playcanvas/splat-transform';
 *
 * const fs = new MemoryFileSystem();
 * await writeFile({
 *     filename: 'output.sog',
 *     outputFormat: getOutputFormat('output.sog', {}),
 *     dataTable: myDataTable,
 *     options: { iterations: 8 }
 * }, fs);
 * ```
 */
const writeFile = async (writeOptions: WriteOptions, fs: FileSystem) => {
    const { filename, outputFormat, dataTable, envDataTable, options, createDevice } = writeOptions;

    // Single-file writers get an atomic per-file bar wrapped here. Multi-file
    // writers (sog, sog-bundle, lod) draw their own per-file bars internally.
    switch (outputFormat) {
        case 'csv':
            await withFileBar(filename, () => writeCsv({ filename, dataTable }, fs));
            break;
        case 'sog':
        case 'sog-bundle':
            await writeSog({
                filename,
                dataTable,
                bundle: outputFormat === 'sog-bundle',
                iterations: options.iterations,
                createDevice
            }, fs);
            break;
        case 'lod':
            await writeLod({
                filename,
                dataTable,
                envDataTable,
                iterations: options.iterations,
                createDevice,
                chunkCount: options.lodChunkCount,
                chunkExtent: options.lodChunkExtent
            }, fs);
            break;
        case 'compressed-ply':
            await withFileBar(filename, () => writeCompressedPly({ filename, dataTable }, fs));
            break;
        case 'ply':
            await withFileBar(filename, () => writePly({
                filename,
                plyData: {
                    comments: [],
                    elements: [{
                        name: 'vertex',
                        dataTable
                    }]
                }
            }, fs));
            break;
        case 'glb':
            await withFileBar(filename, () => writeGlb({ filename, dataTable }, fs));
            break;
        case 'html':
        case 'html-bundle':
            await withFileBar(filename, () => writeHtml({
                filename,
                dataTable,
                viewerSettingsJson: options.viewerSettingsJson,
                bundle: outputFormat === 'html-bundle',
                iterations: options.iterations,
                createDevice
            }, fs));
            break;
        case 'voxel':
            await withFileBar(filename, () => writeVoxel({
                filename,
                dataTable,
                voxelResolution: options.voxelResolution,
                opacityCutoff: options.opacityCutoff,
                navExteriorRadius: options.navExteriorRadius,
                floorFill: options.floorFill,
                floorFillDilation: options.floorFillDilation,
                navCapsule: options.navCapsule,
                navSeed: options.navSeed,
                collisionMesh: options.collisionMesh,
                meshSimplifyError: options.meshSimplifyError,
                createDevice
            }, fs));
            break;
    }
};

export { getOutputFormat, writeFile, type OutputFormat, type WriteOptions };
