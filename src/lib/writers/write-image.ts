import { basename } from 'pathe';
import { Vec3 } from 'playcanvas';

import { logWrittenFile } from './utils';
import { convertToSpace, DataTable } from '../data-table';
import { type FileSystem, writeFile } from '../io/write';
import { renderSplats } from '../render';
import { type Projection, type RenderCamera } from '../render/camera';
import type { DeviceCreator } from '../types';
import { logger, Transform, WebPCodec } from '../utils';

// Cache the WebP codec across invocations; `WebPCodec.create()` instantiates
// the WASM module which is expensive to repeat. Same pattern as write-sog.ts.
let webPCodec: WebPCodec | undefined;

/**
 * Options for writing a rendered splat image.
 */
type WriteImageOptions = {
    /** Output filename ending in `.webp`. */
    filename: string;

    /** Gaussian splat data to render. */
    dataTable: DataTable;

    /**
     * Camera projection mode. Default: `'pinhole'`.
     *
     * - `'pinhole'` — perspective camera using `fov`.
     * - `'equirect'` — full 360° × 180° equirectangular panorama from
     *   `cameraPosition`. Ignores `fov`. Requires `width === 2 × height`;
     *   default resolution is 2048 × 1024.
     */
    projection?: Projection;

    /** Camera position in world space. Default: (2, 1, -2). */
    cameraPosition?: { x: number; y: number; z: number };

    /** Point the camera looks at, in world space. Default: (0, 0, 0). */
    lookAt?: { x: number; y: number; z: number };

    /** World-space up vector. Default: (0, 1, 0). */
    up?: { x: number; y: number; z: number };

    /** Vertical field of view in degrees. Default: 60. Unused for `equirect`. */
    fov?: number;

    /** Output image width in pixels. Default: 1280 (pinhole) or 2048 (equirect). */
    width?: number;

    /** Output image height in pixels. Default: 720 (pinhole) or 1024 (equirect). */
    height?: number;

    /** Near clip distance in world units. Splats with camera-space depth <= near are culled. Default: 0.2 (matches the reference 3DGS rasterizer). */
    near?: number;

    /** RGBA background, each channel in [0, 1]. Default: (0, 0, 0, 1). */
    background?: { r: number; g: number; b: number; a: number };

    /**
     * Aperture as a photographic f-number (e.g. 2.8, 5.6, 11). Enables
     * defocus blur / depth-of-field: smaller numbers = stronger blur.
     * Defaults to disabled. Pinhole only — passing this with
     * `projection: 'equirect'` is an error.
     */
    fNumber?: number;

    /**
     * Camera-space Z of the focus plane in world units. Defaults to the
     * distance from `cameraPosition` to `lookAt` along the forward axis
     * (i.e. focus on the look-at point) when `fNumber` is set. Has no
     * effect without `fNumber`. Pinhole only — passing this with
     * `projection: 'equirect'` is an error.
     */
    focusDistance?: number;

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
        projection = 'pinhole',
        cameraPosition = { x: 2, y: 1, z: -2 },
        lookAt = { x: 0, y: 0, z: 0 },
        up = { x: 0, y: 1, z: 0 },
        near = 0.2,
        background = { r: 0, g: 0, b: 0, a: 1 },
        fNumber,
        focusDistance,
        createDevice
    } = options;

    if (!createDevice) {
        throw new Error('writeImage requires a createDevice function for GPU rasterization');
    }

    let { fov, width, height } = options;
    if (projection === 'equirect') {
        if (fov !== undefined) {
            throw new Error('writeImage: --fov is not valid with --projection equirect (the projection covers a full 360°×180° sphere).');
        }
        if (fNumber !== undefined) {
            throw new Error('writeImage: --f-number is not valid with --projection equirect (defocus blur needs a focal length, which the equirect projection does not have).');
        }
        if (focusDistance !== undefined) {
            throw new Error('writeImage: --focus-distance is not valid with --projection equirect.');
        }
        if (width === undefined && height === undefined) {
            width = 2048;
            height = 1024;
        } else if (width === undefined || height === undefined) {
            throw new Error('writeImage: equirect requires either both width and height, or neither (defaults to 2048x1024).');
        }
        if (width !== 2 * height) {
            throw new Error(`writeImage: equirect requires width === 2 × height (got ${width}x${height}).`);
        }
    } else {
        fov ??= 60;
        width ??= 1280;
        height ??= 720;
        if (fov <= 0 || fov >= 180) {
            throw new Error(`Invalid fov: ${fov}. Must be in (0, 180).`);
        }
        if (fNumber !== undefined && !(fNumber > 0)) {
            throw new Error(`Invalid f-number: ${fNumber}. Must be > 0.`);
        }
        if (focusDistance !== undefined && !(focusDistance > 0)) {
            throw new Error(`Invalid focus-distance: ${focusDistance}. Must be > 0.`);
        }
    }

    const g = logger.group('Render');

    const fovY = projection === 'equirect' ? 0 : (fov! * Math.PI) / 180;

    // Resolve DoF for pinhole only. apertureScale carries the pixel-space
    // CoC-per-unit-defocus scalar that the project shader consumes; we
    // convert the user-facing photographic f-number using the rendered
    // focal length (`focalY` = (height/2) / tan(fovY/2)`), matching the
    // formula in `buildCameraBasis`. Focusing defaults to the look-at
    // point — the most natural pick — measured as the forward-axis
    // projection of `lookAt - cameraPosition`. apertureScale = 0 (the
    // default when --f-number is omitted) makes the rasterizer's DoF
    // path collapse to a no-op, preserving bit-identical default
    // renders.
    let resolvedFocusDistance = 0;
    let resolvedApertureScale = 0;
    if (projection !== 'equirect' && fNumber !== undefined) {
        const focalYPx = (height / 2) / Math.tan(fovY * 0.5);
        resolvedApertureScale = focalYPx / fNumber;
        if (focusDistance !== undefined) {
            resolvedFocusDistance = focusDistance;
        } else {
            const fwdX = lookAt.x - cameraPosition.x;
            const fwdY = lookAt.y - cameraPosition.y;
            const fwdZ = lookAt.z - cameraPosition.z;
            const fwdLen = Math.hypot(fwdX, fwdY, fwdZ);
            if (fwdLen === 0) {
                throw new Error('writeImage: cannot derive default --focus-distance because --camera equals --look-at.');
            }
            // forward · (lookAt - cameraPosition) where forward is unit:
            // = fwdLen (the basis forward is the same vector normalized).
            resolvedFocusDistance = fwdLen;
        }
    }

    const camera: RenderCamera = {
        projection,
        position: new Vec3(cameraPosition.x, cameraPosition.y, cameraPosition.z),
        target: new Vec3(lookAt.x, lookAt.y, lookAt.z),
        up: new Vec3(up.x, up.y, up.z),
        fovY,
        width,
        height,
        near,
        focusDistance: resolvedFocusDistance,
        apertureScale: resolvedApertureScale
    };

    const device = await createDevice();

    // The renderer's camera, defaults and convention all live in PlayCanvas
    // default space. Convert the DataTable from its source-space (e.g.
    // Transform.PLY) to identity so positions, rotations and SH bands align.
    // `inPlace=true` is safe: the CLI builds a fresh combined DataTable per
    // write and library callers passing data into a writer accept that the
    // writer may consume the table.
    const pcDataTable = convertToSpace(dataTable, Transform.IDENTITY, true);

    if (projection === 'equirect') {
        logger.info(`${width}x${height} equirect`);
    } else if (resolvedApertureScale > 0) {
        logger.info(`${width}x${height} fov ${fov}° f/${fNumber} focus ${resolvedFocusDistance.toFixed(3)}`);
    } else {
        logger.info(`${width}x${height} fov ${fov}°`);
    }

    const rgba = await renderSplats(device, pcDataTable, camera, background);

    const encodingGroup = logger.group('Encoding');
    if (!webPCodec) {
        webPCodec = await WebPCodec.create();
    }
    const webp = webPCodec.encodeLosslessRGBA(rgba, width, height);
    encodingGroup.end();

    await writeFile(fs, filename, webp);
    logWrittenFile(basename(filename), webp.byteLength);

    g.end();
};

export { writeImage, type WriteImageOptions };
