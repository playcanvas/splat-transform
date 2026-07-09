import { type ChunkDataPool, type ChunkLayer, type ChunkSource } from './chunk';
import { materializeToDataTable } from './compat/data-table';
import { DataTable } from './data-table';
import { type FileSystem } from './io/write';
import { type DeviceCreator, type Options } from './types';
import { writeCompressedPly, writeCsv, writeGlb, writeHtml, writeImage, writePly, writeSog, writeSogSource, writeSpz, writeVoxel } from './writers';
import { writeCompressedPlySource } from './writers/write-compressed-ply';
import { writePlyStreaming } from './writers/write-ply-streaming';
import { writeSplatStreaming } from './writers/write-splat-streaming';

/**
 * Supported output file formats for Gaussian splat data.
 *
 * - `ply` - Standard PLY format
 * - `compressed-ply` - Compressed PLY format
 * - `splat` - antimatter15 / PlayCanvas viewer `.splat` format
 * - `spz` - Niantic Labs SPZ format
 * - `glb` - Binary glTF with KHR_gaussian_splatting extension
 * - `csv` - CSV text format (for debugging/analysis)
 * - `sog` - PlayCanvas SOG format (separate files)
 * - `sog-bundle` - PlayCanvas SOG format (bundled into single .sog file)
 * - `lod` - Multi-LOD format with chunked data
 * - `html` - Self-contained HTML viewer (separate assets)
 * - `html-bundle` - Self-contained HTML viewer (all assets embedded)
 * - `voxel` - Sparse voxel octree format for collision detection
 * - `image` - Rasterized RGBA image (lossless WebP) rendered from a camera view
 */
type OutputFormat = 'csv' | 'sog' | 'sog-bundle' | 'lod' | 'compressed-ply' | 'ply' | 'splat' | 'spz' | 'glb' | 'html' | 'html-bundle' | 'voxel' | 'image';

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
    } else if (lowerFilename.endsWith('.splat')) {
        return 'splat';
    } else if (lowerFilename.endsWith('.spz')) {
        return 'spz';
    } else if (lowerFilename.endsWith('.glb')) {
        return 'glb';
    } else if (lowerFilename.endsWith('.html')) {
        return options.unbundled ? 'html' : 'html-bundle';
    } else if (lowerFilename.endsWith('.webp')) {
        return 'image';
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
    const { filename, outputFormat, dataTable, options, createDevice } = writeOptions;

    // Each writer is responsible for opening its own `Writing` log group and
    // emitting `filename (size)` info entries per output file.
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
                iterations: options.iterations ?? 10,
                createDevice
            }, fs);
            break;
        case 'lod':
            throw new Error('lod-meta.json output is written from a multi-LOD ChunkSource via writeLodSource, not from a DataTable.');
        case 'compressed-ply':
            await writeCompressedPly({ filename, dataTable }, fs);
            break;
        case 'splat':
            throw new Error('splat output is written from a ChunkSource via writeSource, not from a DataTable.');
        case 'ply':
            await writePly({
                filename,
                plyData: {
                    comments: [],
                    elements: [{
                        name: 'vertex',
                        dataTable
                    }]
                }
            }, fs);
            break;
        case 'spz':
            await writeSpz({
                filename,
                dataTable,
                version: options.spzVersion ?? 4
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
                iterations: options.iterations ?? 10,
                createDevice
            }, fs);
            break;
        case 'voxel':
            await writeVoxel({
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
                createDevice
            }, fs);
            break;
        case 'image':
            await writeImage({
                filename,
                dataTable,
                projection: options.renderProjection,
                cameraPosition: options.renderCameraPosition,
                lookAt: options.renderLookAt,
                up: options.renderUp,
                fov: options.renderFov,
                width: options.renderWidth,
                height: options.renderHeight,
                near: options.renderNear,
                background: options.renderBackground,
                fStop: options.renderFStop,
                focusDistance: options.renderFocusDistance,
                sensorSize: options.renderSensorSize,
                cameraEndPosition: options.renderCameraEndPosition,
                lookAtEnd: options.renderLookAtEnd,
                upEnd: options.renderUpEnd,
                shutter: options.renderShutter,
                motionSamples: options.renderMotionSamples,
                createDevice
            }, fs);
            break;
    }
};

/**
 * Options for {@link writeSource} — the chunk-native write entry.
 */
type WriteSourceOptions = {
    /** Path to the output file. */
    filename: string;
    /** The format to write (single-scene formats; `lod` goes via `writeLodSource`). */
    outputFormat: OutputFormat;
    /** The source to write (the caller owns its lifetime / `close()`). */
    source: ChunkSource;
    /** Pool for the streaming writers and the materialize bridge. */
    pool: ChunkDataPool;
    /** Processing options. */
    options: Options;
    /** Optional function to create a GPU device. */
    createDevice?: DeviceCreator;
};

/**
 * Write a {@link ChunkSource} to a file — the chunk-native sibling of
 * {@link writeFile}. Formats with a streaming writer (`ply`/`sog`/`compressed-ply`)
 * consume the source directly; formats without one yet materialize to a
 * `DataTable` right at the writer and delegate to {@link writeFile} — the inline
 * bridge around the not-yet-chunked writer.
 *
 * `lod` output is written via `writeLodSource` (multi-LOD + env), not here.
 *
 * @param writeSourceOptions - The source, format and options to write.
 * @param fs - File system abstraction for writing files.
 */
const writeSource = async (writeSourceOptions: WriteSourceOptions, fs: FileSystem): Promise<void> => {
    const { filename, outputFormat, source, pool, options, createDevice } = writeSourceOptions;

    switch (outputFormat) {
        case 'ply':
            await writePlyStreaming(source, pool, { filename }, fs);
            break;
        case 'sog':
        case 'sog-bundle':
            await writeSogSource(source, pool, {
                filename,
                bundle: outputFormat === 'sog-bundle',
                iterations: options.iterations ?? 10,
                createDevice
            }, fs);
            break;
        case 'compressed-ply':
            await writeCompressedPlySource(source, pool, { filename }, fs);
            break;
        case 'splat':
            await writeSplatStreaming(source, pool, { filename }, fs);
            break;
        case 'lod':
            throw new Error('writeSource: lod output must be written via writeLodSource');
        case 'voxel': {
            // Voxelization consumes only position + geometric (see writeVoxel:
            // x/y/z, rot, scale, opacity — no color/SH). Materialize just those
            // layers so color and SH are never loaded (they were previously read
            // into the full table and discarded).
            const dataTable = await materializeToDataTable(source, pool, new Set<ChunkLayer>(['position', 'geometric']));
            await writeFile({ filename, outputFormat, dataTable, options, createDevice }, fs);
            break;
        }
        default: {
            // No streaming writer yet — materialize and delegate to the DataTable
            // writer (the inline bridge around the unconverted writer).
            const dataTable = await materializeToDataTable(source, pool);
            await writeFile({ filename, outputFormat, dataTable, options, createDevice }, fs);
        }
    }
};

export { getOutputFormat, writeFile, writeSource, type OutputFormat, type WriteOptions, type WriteSourceOptions };
