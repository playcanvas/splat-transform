import { basename } from 'pathe';
import { Vec3 } from 'playcanvas';

import { logWrittenFile } from './utils';
import { convertToSpace, DataTable } from '../data-table';
import { type FileSystem, writeFile } from '../io/write';
import { renderSplats } from '../render';
import { type Projection, type RenderCamera } from '../render/camera';
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

    /** Vertical field of view in degrees. Default: 60 for `pinhole`. Must be omitted for `equirect` (throws if supplied). */
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
     * Aperture as a photographic f-stop (e.g. 2.8, 5.6, 11). Enables
     * defocus blur / depth-of-field: smaller numbers = stronger blur.
     * Defaults to disabled. Pinhole only — passing this with
     * `projection: 'equirect'` is an error.
     */
    fStop?: number;

    /**
     * Camera-space Z of the focus plane in world units. Defaults to the
     * distance from `cameraPosition` to `lookAt` along the forward axis
     * (i.e. focus on the look-at point) when `fStop` is set. Has no
     * effect without `fStop`. Pinhole only — passing this with
     * `projection: 'equirect'` is an error.
     */
    focusDistance?: number;

    /**
     * Vertical sensor height in world units, used to give `fStop` a
     * defined physical meaning. Default `0.024` matches a 35mm
     * full-frame sensor when world units are meters. Scale this with
     * your scene's units (e.g. world unit = decimeter → 0.24, world
     * unit = millimeter → 24). Has no effect without `fStop`.
     */
    sensorSize?: number;

    /**
     * End camera position for motion blur. When set, enables camera
     * motion blur: the camera moves from (`cameraPosition`, `lookAt`,
     * `up`) at shutter-open to (`cameraEndPosition`, `lookAtEnd`,
     * `upEnd`) at shutter-close, and every gaussian is integrated over
     * its motion in `motionSamples` shutter slices. Pinhole only.
     */
    cameraEndPosition?: { x: number; y: number; z: number };

    /**
     * End look-at target for motion blur. Defaults to `lookAt` when
     * motion blur is enabled.
     */
    lookAtEnd?: { x: number; y: number; z: number };

    /**
     * End up vector for motion blur. Defaults to `up` when motion blur
     * is enabled.
     */
    upEnd?: { x: number; y: number; z: number };

    /**
     * Shutter fraction in `[0, 1]`. Portion of the start→end segment
     * actually integrated, centered on the midpoint (standard
     * shutter-angle convention: 1.0 = full motion, 0.5 = 180° shutter).
     * Default: `1`. Only meaningful with `cameraEndPosition`.
     */
    shutter?: number;

    /**
     * Number of shutter slices for motion blur; cost is N× a single
     * render. Each slice integrates every gaussian's motion exactly, so
     * streaks are smooth at any N; more slices refine the compositing
     * between overlapping gaussians (1 slice is a single pass with a
     * small exposure bias, 4 is visually converged on typical scenes).
     * Default: `4`. Only meaningful with `cameraEndPosition`.
     */
    motionSamples?: number;

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
        fStop,
        focusDistance,
        sensorSize = 0.024,
        cameraEndPosition,
        lookAtEnd,
        upEnd,
        shutter,
        motionSamples,
        createDevice
    } = options;

    if (!createDevice) {
        throw new Error('writeImage requires a createDevice function for GPU rasterization');
    }

    let { fov, width, height } = options;
    if (projection === 'equirect') {
        if (fov !== undefined) {
            throw new Error('writeImage: --camera-fov is not valid with --projection equirect (the projection covers a full 360°×180° sphere).');
        }
        if (fStop !== undefined) {
            throw new Error('writeImage: --f-stop is not valid with --projection equirect (defocus blur needs a focal length, which the equirect projection does not have).');
        }
        if (focusDistance !== undefined) {
            throw new Error('writeImage: --focus-distance is not valid with --projection equirect.');
        }
        if (options.sensorSize !== undefined) {
            throw new Error('writeImage: --sensor-size is not valid with --projection equirect.');
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
        if (fStop !== undefined && !(fStop > 0)) {
            throw new Error(`Invalid f-stop: ${fStop}. Must be > 0.`);
        }
        if (focusDistance !== undefined && !(focusDistance > 0)) {
            throw new Error(`Invalid focus-distance: ${focusDistance}. Must be > 0.`);
        }
        if (!(sensorSize > 0)) {
            throw new Error(`Invalid sensor-size: ${sensorSize}. Must be > 0.`);
        }
    }

    // Motion blur: enabled iff `--camera-pos-end` is supplied. The end pose
    // for missing `camera-target-end` / `camera-up-end` defaults to the start pose so
    // pure translations don't need redundant flags.
    const motionEnabled = cameraEndPosition !== undefined;
    const motionN = motionEnabled ? (motionSamples ?? 4) : 1;
    const motionShutter = motionEnabled ? (shutter ?? 1) : 0;
    if (motionEnabled && (motionShutter < 0 || motionShutter > 1)) {
        throw new Error(`writeImage: --shutter must be in [0, 1], got ${motionShutter}.`);
    }
    if (motionEnabled && (!Number.isInteger(motionN) || motionN < 1)) {
        throw new Error(`writeImage: --motion-samples must be a positive integer, got ${motionN}.`);
    }
    if (motionEnabled && projection === 'equirect') {
        throw new Error('writeImage: motion blur (--camera-pos-end) is not supported with --projection equirect.');
    }
    const camStart = cameraPosition;
    const camEnd = cameraEndPosition ?? cameraPosition;
    const lookStart = lookAt;
    const lookEnd = lookAtEnd ?? lookAt;
    const upStart = up;
    const upEndR = upEnd ?? up;

    const g = logger.group('Render');

    const fovY = projection === 'equirect' ? 0 : (fov! * Math.PI) / 180;

    // Precompute focal scaling used by the DoF aperture-scale formula. Only
    // depends on intrinsics (sensor, fov, height), not on the camera pose,
    // so it's safe to share across motion-blur sub-frames.
    const dofEnabled = projection !== 'equirect' && fStop !== undefined;
    const focalRealWorld = dofEnabled ? (sensorSize / 2) / Math.tan(fovY * 0.5) : 0;
    const focalYPx = dofEnabled ? (height / 2) / Math.tan(fovY * 0.5) : 0;

    // Resolve DoF for pinhole only. The project shader consumes a single
    // pre-baked scalar `apertureScale` (pixel CoC per unit relative
    // defocus) and the focus distance. Physical CoC for a thin lens is:
    //
    //     CoC_pixels = (focal_real² / (N · focus)) × |1 − focus/cz|
    //                  × image_height / sensor_height
    //
    // where focal_real is the real lens focal length implied by
    // `fovY` and `sensorSize`. Apply image_height / sensor_height to
    // convert physical CoC (sensor units) to pixels. Defaulting
    // `sensorSize` to 0.024 makes f-stops behave like a 35mm
    // full-frame camera when world units are meters; scale to suit
    // non-meter scenes. Focus defaults to the look-at point — which,
    // under motion blur, moves with the interpolated camera pose
    // (recomputed per sub-frame below).
    const buildCamera = (pos: { x: number; y: number; z: number },
        tgt: { x: number; y: number; z: number },
        u: { x: number; y: number; z: number }): RenderCamera => {
        let fDist = 0;
        let aScale = 0;
        if (dofEnabled) {
            if (focusDistance !== undefined) {
                fDist = focusDistance;
            } else {
                const fwdLen = Math.hypot(tgt.x - pos.x, tgt.y - pos.y, tgt.z - pos.z);
                if (fwdLen === 0) {
                    throw new Error('writeImage: cannot derive default --focus-distance because --camera-pos equals --camera-target.');
                }
                fDist = fwdLen;
            }
            aScale = focalRealWorld * focalYPx / (fStop! * fDist);
        }
        return {
            projection,
            position: new Vec3(pos.x, pos.y, pos.z),
            target: new Vec3(tgt.x, tgt.y, tgt.z),
            up: new Vec3(u.x, u.y, u.z),
            fovY,
            width: width!,
            height: height!,
            near,
            focusDistance: fDist,
            apertureScale: aScale
        };
    };

    const device = await createDevice();

    // The renderer's camera, defaults and convention all live in PlayCanvas
    // default space. Convert the DataTable from its source-space (e.g.
    // Transform.PLY) to identity so positions, rotations and SH bands align.
    // `inPlace=true` is safe: the CLI builds a fresh combined DataTable per
    // write and library callers passing data into a writer accept that the
    // writer may consume the table.
    const pcDataTable = convertToSpace(dataTable, Transform.IDENTITY, true);

    // Pre-resolve the start-pose DoF info for the info log line.
    const startCamera = buildCamera(camStart, lookStart, upStart);

    if (projection === 'equirect') {
        logger.info(`${width}x${height} equirect`);
    } else if (startCamera.apertureScale! > 0) {
        logger.info(`${width}x${height} fov ${fov}° f/${fStop} focus ${startCamera.focusDistance!.toFixed(3)} sensor ${sensorSize}`);
    } else {
        logger.info(`${width}x${height} fov ${fov}°`);
    }
    if (motionEnabled) {
        logger.info(`motion blur: ${motionN} slice${motionN === 1 ? '' : 's'}, shutter ${motionShutter}`);
    }

    // Pose along the start→end segment at parameter t. Normalized lerp for
    // `up` so it stays unit-length when the start/end up vectors differ in
    // direction.
    const poseAt = (t: number) => {
        const pos = {
            x: camStart.x + (camEnd.x - camStart.x) * t,
            y: camStart.y + (camEnd.y - camStart.y) * t,
            z: camStart.z + (camEnd.z - camStart.z) * t
        };
        const tgt = {
            x: lookStart.x + (lookEnd.x - lookStart.x) * t,
            y: lookStart.y + (lookEnd.y - lookStart.y) * t,
            z: lookStart.z + (lookEnd.z - lookStart.z) * t
        };
        const ux = upStart.x + (upEndR.x - upStart.x) * t;
        const uy = upStart.y + (upEndR.y - upStart.y) * t;
        const uz = upStart.z + (upEndR.z - upStart.z) * t;
        const ulen = Math.hypot(ux, uy, uz) || 1;
        return { pos, tgt, up: { x: ux / ulen, y: uy / ulen, z: uz / ulen } };
    };
    // Shutter window, centered on the segment midpoint.
    const halfWin = motionShutter / 2;
    const t0 = 0.5 - halfWin;
    const t1 = 0.5 + halfWin;

    // Camera for a motion-blur pass over [tA, tB]: shutter-open pose at
    // tA, shutter-close pose at tB, focus distance and aperture scale from
    // the slice midpoint.
    const toVec3 = (v: { x: number; y: number; z: number }) => new Vec3(v.x, v.y, v.z);
    const sliceCamera = (tA: number, tB: number): RenderCamera => {
        const open = poseAt(tA);
        const close = poseAt(tB);
        const mid = poseAt(0.5 * (tA + tB));
        return {
            ...buildCamera(mid.pos, mid.tgt, mid.up),
            position: toVec3(open.pos),
            target: toVec3(open.tgt),
            up: toVec3(open.up),
            shutterClose: {
                position: toVec3(close.pos),
                target: toVec3(close.tgt),
                up: toVec3(close.up)
            }
        };
    };

    let rgba: Uint8Array;
    if (!motionEnabled) {
        rgba = await renderSplats(device, pcDataTable, startCamera, background);
    } else if (motionN === 1) {
        rgba = await renderSplats(device, pcDataTable, sliceCamera(t0, t1), background);
    } else {
        // Camera motion blur over N shutter slices, averaged in float to
        // avoid 8-bit truncation per slice. Each slice integrates every
        // gaussian's motion exactly, so the streak is smooth at any N;
        // more slices refine the per-instant compositing between
        // overlapping gaussians, which a single slice approximates with a
        // fixed order and time-averaged alphas.
        const pixels = width! * height! * 4;
        const accum = new Float32Array(pixels);
        for (let i = 0; i < motionN; i++) {
            const subCamera = sliceCamera(t0 + (t1 - t0) * i / motionN, t0 + (t1 - t0) * (i + 1) / motionN);
            const frame = await renderSplats(device, pcDataTable, subCamera, background);
            for (let p = 0; p < pixels; p++) accum[p] += frame[p];
        }
        rgba = new Uint8Array(pixels);
        const inv = 1 / motionN;
        for (let p = 0; p < pixels; p++) rgba[p] = Math.round(accum[p] * inv);
    }

    const encodingGroup = logger.group('Encoding');
    const webPCodec = await WebPCodec.create(); // cheap: create() memoizes the wasm module
    const webp = webPCodec.encodeLosslessRGBA(rgba, width, height);
    encodingGroup.end();

    await writeFile(fs, filename, webp);
    logWrittenFile(basename(filename), webp.byteLength);

    g.end();
};

export { writeImage, type WriteImageOptions };
