import { type ChunkDataPool, type ChunkLayer, type ChunkSource } from './chunk';
import { materializeToDataTable } from './compat/data-table';
import { type FileSystem } from './io/write';
import { type DeviceCreator, type Options } from './types';
import { writeCsv, writeGlb, writeHtml, writeImage, writeSogSource, writeSpz, writeVoxel } from './writers';
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
 * Render a source to an image file (single frame or camera track).
 *
 * @param filename - Output filename.
 * @param source - The scene.
 * @param pool - Pool for the source's read buffers.
 * @param options - Processing options (the `render*` fields).
 * @param fs - File system to write through.
 * @param createDevice - GPU device factory.
 */
const writeImageSource = async (filename: string, source: ChunkSource, pool: ChunkDataPool, options: Options, fs: FileSystem, createDevice?: DeviceCreator): Promise<void> => {
    await writeImage({
        filename,
        source,
        pool,
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
        webpEffort: options.renderWebpEffort,
        cameraTrack: options.renderCameraTrack,
        frames: options.renderFrames,
        residentBudget: options.renderResidentBudget,
        createDevice
    }, fs);
};

/**
 * Options for {@link writeSource}.
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
 * Write a {@link ChunkSource} to a file. Formats with a source writer
 * (`ply`, `sog`, `compressed-ply`, `splat`, `image`) consume the source
 * directly; the rest (`csv`, `spz`, `glb`, `html`, `voxel`) still take a
 * `DataTable`, so the source is materialized right here and the table stays
 * a private detail of those writers until each is ported.
 *
 * `lod` output is written via `writeLodSource` (multi-LOD + env), not here.
 *
 * Each writer is responsible for opening its own `Writing` log group and
 * emitting `filename (size)` info entries per output file.
 *
 * @param writeSourceOptions - The source, format and options to write.
 * @param fs - File system abstraction for writing files.
 */
const writeSource = async (writeSourceOptions: WriteSourceOptions, fs: FileSystem): Promise<void> => {
    const { filename, outputFormat, source, pool, options, createDevice } = writeSourceOptions;
    const { model } = source.meta;

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
        case 'image':
            await writeImageSource(filename, source, pool, options, fs, createDevice);
            break;
        case 'voxel': {
            // Voxelization consumes only position + geometric (see writeVoxel:
            // x/y/z, rot, scale, opacity — no color/SH). Materialize just those
            // layers so color and SH are never loaded.
            const dataTable = await materializeToDataTable(source, pool, new Set<ChunkLayer>(['position', 'geometric']));
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
        }
        case 'csv':
            await writeCsv({ filename, dataTable: await materializeToDataTable(source, pool) }, fs);
            break;
        case 'spz':
            await writeSpz({
                filename,
                dataTable: await materializeToDataTable(source, pool),
                model,
                version: options.spzVersion ?? 4
            }, fs);
            break;
        case 'glb':
            await writeGlb({ filename, dataTable: await materializeToDataTable(source, pool) }, fs);
            break;
        case 'html':
        case 'html-bundle':
            await writeHtml({
                filename,
                dataTable: await materializeToDataTable(source, pool),
                viewerSettingsJson: options.viewerSettingsJson,
                bundle: outputFormat === 'html-bundle',
                iterations: options.iterations ?? 10,
                createDevice
            }, fs);
            break;
    }
};

export { getOutputFormat, writeSource, type OutputFormat, type WriteSourceOptions };
