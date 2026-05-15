import { basename } from 'pathe';
import { Vec3 } from 'playcanvas';

import { logWrittenFile } from './utils';
import { convertToSpace, DataTable } from '../data-table';
import { type FileSystem, writeFile } from '../io/write';
import { renderSplats } from '../render';
import { type RenderCamera } from '../render/camera';
import type { DeviceCreator } from '../types';
import { logger, Transform, WebPCodec } from '../utils';

/**
 * Options for writing a rendered splat image.
 */
type WriteImageOptions = {
    /** Output filename ending in `.webp`. */
    filename: string;

    /** Gaussian splat data to render. */
    dataTable: DataTable;

    /** Camera position in world space. Default: (2, 1, -2). */
    cameraPosition?: { x: number; y: number; z: number };

    /** Point the camera looks at, in world space. Default: (0, 0, 0). */
    lookAt?: { x: number; y: number; z: number };

    /** World-space up vector. Default: (0, 1, 0). */
    up?: { x: number; y: number; z: number };

    /** Vertical field of view in degrees. Default: 60. */
    fov?: number;

    /** Output image width in pixels. Default: 1280. */
    width?: number;

    /** Output image height in pixels. Default: 720. */
    height?: number;

    /** Near clip distance in world units. Splats with camera-space depth <= near are culled. Default: 0.2 (matches the reference 3DGS rasterizer). */
    near?: number;

    /** RGBA background, each channel in [0, 1]. Default: (0, 0, 0, 1). */
    background?: { r: number; g: number; b: number; a: number };

    /** Function returning a GraphicsDevice. Required — rasterization runs on GPU. */
    createDevice?: DeviceCreator;
};

/**
 * Renders the splat scene to a lossless WebP image written via `fs`.
 *
 * @param options - Render parameters and target filename.
 * @param fs - File system abstraction for writing the output.
 *
 * @example
 * ```ts
 * await writeImage({
 *     filename: 'view.webp',
 *     dataTable,
 *     cameraPosition: { x: 0, y: 0, z: 5 },
 *     fov: 60,
 *     width: 1920, height: 1080,
 *     createDevice: async () => myDevice
 * }, fs);
 * ```
 */
const writeImage = async (options: WriteImageOptions, fs: FileSystem): Promise<void> => {
    const {
        filename,
        dataTable,
        cameraPosition = { x: 2, y: 1, z: -2 },
        lookAt = { x: 0, y: 0, z: 0 },
        up = { x: 0, y: 1, z: 0 },
        fov = 60,
        width = 1280,
        height = 720,
        near = 0.2,
        background = { r: 0, g: 0, b: 0, a: 1 },
        createDevice
    } = options;

    if (!createDevice) {
        throw new Error('writeImage requires a createDevice function for GPU rasterization');
    }
    if (fov <= 0 || fov >= 180) {
        throw new Error(`Invalid fov: ${fov}. Must be in (0, 180).`);
    }

    const g = logger.group('Render');

    const camera: RenderCamera = {
        position: new Vec3(cameraPosition.x, cameraPosition.y, cameraPosition.z),
        target: new Vec3(lookAt.x, lookAt.y, lookAt.z),
        up: new Vec3(up.x, up.y, up.z),
        fovY: (fov * Math.PI) / 180,
        width,
        height,
        near
    };

    const device = await createDevice();

    // The renderer's camera, defaults and convention all live in PlayCanvas
    // default space. Convert the DataTable from its source-space (e.g.
    // Transform.PLY) to identity so positions, rotations and SH bands align.
    // `inPlace=true` is safe: the CLI builds a fresh combined DataTable per
    // write and library callers passing data into a writer accept that the
    // writer may consume the table.
    const pcDataTable = convertToSpace(dataTable, Transform.IDENTITY, true);

    logger.info(`${width}x${height} fov ${fov}°`);

    const rgba = await renderSplats(device, pcDataTable, camera, background);

    const encodingGroup = logger.group('Encoding');
    const codec = await WebPCodec.create();
    const webp = codec.encodeLosslessRGBA(rgba, width, height);
    encodingGroup.end();

    await writeFile(fs, filename, webp);
    logWrittenFile(basename(filename), webp.byteLength);

    g.end();
};

export { writeImage, type WriteImageOptions };
