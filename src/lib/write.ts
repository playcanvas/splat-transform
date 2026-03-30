import { DataTable } from './data-table/data-table';
import { type FileSystem } from './io/write';
import { Options } from './types';
import { logger } from './utils/logger';
import { writeCompressedPly } from './writers/write-compressed-ply';
import { writeCsv } from './writers/write-csv';
import { writeGlb } from './writers/write-glb';
import { writeHtml } from './writers/write-html';
import { writeLod } from './writers/write-lod';
import { writePly } from './writers/write-ply';
import { writeSog, type DeviceCreator } from './writers/write-sog';
import { writeVoxel } from './writers/write-voxel';

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

    logger.log(`writing '${filename}'...`);

    // write the file data
    switch (outputFormat) {
        case 'csv':
            await writeCsv({ filename, dataTable }, fs);
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
            await writeCompressedPly({ filename, dataTable }, fs);
            break;
        case 'ply':
            await writePly({
                filename,
                plyData: {
                    comments: [],
                    elements: [{
                        name: 'vertex',
                        dataTable: dataTable
                    }]
                }
            }, fs);
            break;
        case 'glb':
            await writeGlb({ filename, dataTable }, fs);
            break;
        case 'html':
        case 'html-bundle':
            await writeHtml({
                filename,
                dataTable,
                viewerSettingsJson: options.viewerSettingsJson,
                bundle: outputFormat === 'html-bundle',
                iterations: options.iterations,
                createDevice
            }, fs);
            break;
        case 'voxel': {
            const enableNav = options.navSimplify !== false;
            const navCapsule = enableNav ? (options.navCapsule ?? { height: 1.6, radius: 0.2 }) : undefined;
            const navSeed = enableNav ? (options.navSeed ?? { x: 0, y: 0, z: 0 }) : undefined;
            await writeVoxel({
                filename,
                dataTable,
                voxelResolution: options.voxelResolution,
                opacityCutoff: options.opacityCutoff,
                navExteriorRadius: options.navExteriorRadius,
                navCapsule,
                navSeed,
                collisionMesh: options.collisionMesh,
                meshSimplifyError: options.meshSimplifyError,
                createDevice
            }, fs);
            break;
        }
    }
};

export { getOutputFormat, writeFile, type OutputFormat, type WriteOptions };
