import { basename } from 'pathe';
import { Vec3 } from 'playcanvas';

import { frameFilename, logWrittenFile } from './utils';
import { type ChunkDataPool, type ChunkSource } from '../chunk';
import { computeWriteTransform } from '../data-table';
import { type FileSystem, writeFile } from '../io/write';
import { SceneRenderer, type SceneTier } from '../render';
import { type Projection, type RenderCamera } from '../render/camera';
import { type CameraTrack } from '../render/camera-track';
import type { DeviceCreator } from '../types';
import { fmtBytes, fmtTime, logger, Transform, WebPCodec } from '../utils';
import { runEncodeWebp, WorkerQueue } from '../workers';

type Vec3Like = { x: number; y: number; z: number };

/**
 * Frames of a sequence allowed to be encoding (queued or on a worker thread)
 * while the next one renders. Bounds the RGBA frames held in memory.
 */
const MAX_PENDING_ENCODES = 4;

/** Renders averaged per motion-blurred frame unless the caller sets `motionSamples`. */
const DEFAULT_MOTION_SAMPLES = 1;

/**
 * Options for writing a rendered splat image.
 */
type WriteImageOptions = {
    /**
     * Output filename ending in `.webp`. With `cameraTrack`, each frame is
     * written as `<stem>.NNNN.webp` (zero-padded frame index).
     */
    filename: string;

    /** The scene to render (its LOD 0). Its pending transform is honoured by moving the camera into the scene's space. */
    source: ChunkSource;
    /** Pool for the source's read buffers; its chunk size must be at least the source's. */
    pool: ChunkDataPool;

    /**
     * Camera projection mode. Default: `'pinhole'`.
     *
     * - `'pinhole'` — perspective camera using `fov`.
     * - `'equirect'` — full 360° × 180° equirectangular panorama from
     *   `cameraPosition`. Ignores `fov`. Requires `width === 2 × height`;
     *   default resolution is 2048 × 1024.
     */
    projection?: Projection;

    /** Camera position in world space. Default: (2, 1, -2). Ignored with `cameraTrack`. */
    cameraPosition?: Vec3Like;

    /** Point the camera looks at, in world space. Default: (0, 0, 0). Ignored with `cameraTrack`. */
    lookAt?: Vec3Like;

    /** World-space up vector. Default: (0, 1, 0). */
    up?: Vec3Like;

    /**
     * Vertical field of view in degrees. Default: 60 for `pinhole`. Must be
     * omitted for `equirect` (throws if supplied). With `cameraTrack`, the
     * track's per-pose fov takes precedence.
     */
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
     * `upEnd`) at shutter-close, and the frame averages renders at
     * `motionSamples` instants across the shutter. Not valid with
     * `cameraTrack`, whose motion blur comes from `shutter`.
     */
    cameraEndPosition?: Vec3Like;

    /**
     * End look-at target for motion blur. Defaults to `lookAt` when
     * motion blur is enabled.
     */
    lookAtEnd?: Vec3Like;

    /**
     * End up vector for motion blur. Defaults to `up` when motion blur
     * is enabled.
     */
    upEnd?: Vec3Like;

    /**
     * Shutter fraction in `[0, 1]`. For a start→end segment, the portion
     * actually integrated, centered on the midpoint (standard
     * shutter-angle convention: 1.0 = full motion, 0.5 = 180° shutter);
     * default `1`. For a `cameraTrack`, setting it enables motion blur
     * over that fraction of the frame interval, centered on each frame;
     * default off.
     */
    shutter?: number;

    /**
     * Renders averaged per motion-blurred frame, at evenly spaced instants
     * across the shutter; cost is N× a single render. Each instant is
     * composited exactly, so the mean converges to the true time average
     * as N grows; too few instants show as discrete copies wherever the
     * motion between them exceeds a couple of pixels. Default: `1`.
     * Only meaningful when motion blur is enabled.
     */
    motionSamples?: number;

    /**
     * Camera animation to render as a frame sequence. Poses come from the
     * track (position, target and fov per frame; `up` still applies) and
     * the scene stays resident on the GPU across frames.
     */
    cameraTrack?: CameraTrack;

    /** Inclusive frame range of `cameraTrack` to render. Default: every frame. */
    frames?: [number, number];

    /**
     * WebP lossless compression effort, 0–9. Every level is lossless; higher
     * levels shrink the file at a steep cost in encode time (about 8× slower
     * from 0 to 6 for roughly 20% smaller output). Default: `0`.
     */
    webpEffort?: number;

    /**
     * Most bytes to hold GPU-resident for the scene. The resident path
     * uploads the scene once and renders every pass from it; scenes over
     * this budget, or over the device's binding limits, or that the device
     * refuses to allocate, stream through the chunked path instead.
     * Default: no budget.
     */
    residentBudget?: number;

    /** Function returning a GraphicsDevice. Required — rasterization runs on GPU. */
    createDevice?: DeviceCreator;
};

/** A camera pose in the scene's space: position, target, up and vertical fov in degrees. */
type Pose = { pos: Vec3Like; tgt: Vec3Like; up: Vec3Like; fov: number };

/**
 * Renders the splat scene to a lossless WebP image written via `fs`, or to
 * a sequence of them along a camera track.
 *
 * @param options - Render parameters and target filename.
 * @param fs - File system abstraction for writing the output.
 *
 * @example
 * ```ts
 * await writeImage({
 *     filename: 'view.webp',
 *     source,
 *     pool,
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
        source,
        pool,
        projection = 'pinhole',
        cameraPosition = { x: 2, y: 1, z: -2 },
        lookAt = { x: 0, y: 0, z: 0 },
        up = { x: 0, y: 1, z: 0 },
        background = { r: 0, g: 0, b: 0, a: 1 },
        fStop,
        cameraEndPosition,
        lookAtEnd,
        upEnd,
        shutter,
        motionSamples,
        cameraTrack,
        frames,
        webpEffort = 0,
        residentBudget,
        createDevice
    } = options;
    let { near = 0.2, focusDistance, sensorSize = 0.024 } = options;

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

    // Motion blur. Along a start→end segment it is enabled by
    // `--camera-pos-end`, with the end pose defaulting to the start pose
    // for missing `camera-target-end` / `camera-up-end` so pure translations
    // don't need redundant flags. Along a track it is enabled by `--shutter`.
    if (cameraTrack && cameraEndPosition) {
        throw new Error('writeImage: --camera-pos-end is not valid with --camera-track; motion blur along a track comes from --shutter.');
    }
    const motionEnabled = cameraTrack ? (shutter !== undefined && shutter > 0) : cameraEndPosition !== undefined;
    const motionN = motionEnabled ? (motionSamples ?? DEFAULT_MOTION_SAMPLES) : 1;
    const motionShutter = motionEnabled ? (shutter ?? 1) : 0;
    if (motionEnabled && (motionShutter < 0 || motionShutter > 1)) {
        throw new Error(`writeImage: --shutter must be in [0, 1], got ${motionShutter}.`);
    }
    if (motionEnabled && (!Number.isInteger(motionN) || motionN < 1)) {
        throw new Error(`writeImage: --motion-samples must be a positive integer, got ${motionN}.`);
    }

    // Frame range: a single frame without a track, else the track's frames.
    let frameStart = 0;
    let frameEnd = 0;
    if (cameraTrack) {
        frameStart = frames?.[0] ?? 0;
        frameEnd = frames?.[1] ?? cameraTrack.frameCount - 1;
        if (!Number.isInteger(frameStart) || !Number.isInteger(frameEnd) || frameStart < 0 || frameEnd < frameStart) {
            throw new Error(`writeImage: invalid frame range ${frameStart}-${frameEnd}.`);
        }
        if (frameEnd >= cameraTrack.frameCount) {
            throw new Error(`writeImage: frame range ${frameStart}-${frameEnd} exceeds the track's ${cameraTrack.frameCount} frames (0-${cameraTrack.frameCount - 1}).`);
        }
    }
    const frameCount = frameEnd - frameStart + 1;
    // The camera, its defaults and the renderer's conventions live in the
    // PlayCanvas default space, while the scene stays in its source space
    // (e.g. Transform.PLY). Rather than baking every gaussian into the
    // camera's space, move the camera into the scene's: positions and
    // targets through the inverse transform, directions through its
    // rotation, and lengths (near plane, focus, sensor) by its scale. The
    // image is the same; the per-render pass over the whole scene is not.
    const delta = computeWriteTransform(source.meta.transform, Transform.IDENTITY);
    const toData = delta ? delta.clone().invert() : null;
    const tmp = new Vec3();
    const toDataPoint = (p: Vec3Like): Vec3Like => {
        if (!toData) return p;
        toData.transformPoint(tmp.set(p.x, p.y, p.z), tmp);
        return { x: tmp.x, y: tmp.y, z: tmp.z };
    };
    const toDataDir = (d: Vec3Like): Vec3Like => {
        if (!toData) return d;
        toData.rotation.transformVector(tmp.set(d.x, d.y, d.z), tmp);
        return { x: tmp.x, y: tmp.y, z: tmp.z };
    };
    if (toData) {
        near *= toData.scale;
        sensorSize *= toData.scale;
        if (focusDistance !== undefined) focusDistance *= toData.scale;
    }
    const camStart = toDataPoint(cameraPosition);
    const camEnd = toDataPoint(cameraEndPosition ?? cameraPosition);
    const lookStart = toDataPoint(lookAt);
    const lookEnd = toDataPoint(lookAtEnd ?? lookAt);
    const upStart = toDataDir(up);
    const upEndR = toDataDir(upEnd ?? up);
    const optionFov = projection === 'equirect' ? 0 : fov!;

    // Pose at time t. Along a segment, t ∈ [0, 1] from the start to the end
    // pose (normalized lerp for `up` so it stays unit-length when the two
    // differ in direction) and the shutter window is centered on 0.5. Along
    // a track, t is a frame time and the window is centered on the frame.
    const poseAt: (t: number) => Pose = cameraTrack ?
        (t) => {
            const p = cameraTrack.poseAt(t);
            return { pos: toDataPoint(p.position), tgt: toDataPoint(p.target), up: upStart, fov: projection === 'equirect' ? 0 : p.fov };
        } :
        (t) => {
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
            return { pos, tgt, up: { x: ux / ulen, y: uy / ulen, z: uz / ulen }, fov: optionFov };
        };

    const g = logger.group('Render');

    // Resolve DoF for pinhole only. The project shader consumes a single
    // pre-baked scalar `apertureScale` (pixel CoC per unit relative
    // defocus) and the focus distance. Physical CoC for a thin lens is:
    //
    //     CoC_pixels = (focal_real² / (N · focus)) × |1 − focus/cz|
    //                  × image_height / sensor_height
    //
    // where focal_real is the real lens focal length implied by the pose's
    // fov and `sensorSize`. Apply image_height / sensor_height to convert
    // physical CoC (sensor units) to pixels. Defaulting `sensorSize` to
    // 0.024 makes f-stops behave like a 35mm full-frame camera when world
    // units are meters; scale to suit non-meter scenes. Focus defaults to
    // the look-at point — which, under motion blur or along a track, moves
    // with the pose.
    const dofEnabled = projection !== 'equirect' && fStop !== undefined;
    const buildCamera = (pose: Pose): RenderCamera => {
        const { pos, tgt, up: u } = pose;
        const fovY = projection === 'equirect' ? 0 : (pose.fov * Math.PI) / 180;
        if (projection !== 'equirect' && !(fovY > 0 && fovY < Math.PI)) {
            throw new Error(`writeImage: invalid fov ${pose.fov}° on the camera track.`);
        }
        let fDist = 0;
        let aScale = 0;
        if (dofEnabled) {
            if (focusDistance !== undefined) {
                fDist = focusDistance;
            } else {
                const fwdLen = Math.hypot(tgt.x - pos.x, tgt.y - pos.y, tgt.z - pos.z);
                if (fwdLen === 0) {
                    throw new Error('writeImage: cannot derive default --focus-distance because the camera position equals its target.');
                }
                fDist = fwdLen;
            }
            const focalRealWorld = (sensorSize / 2) / Math.tan(fovY * 0.5);
            const focalYPx = (height! / 2) / Math.tan(fovY * 0.5);
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

    // Pre-resolve the first pose for the info log line.
    const firstPose = poseAt(cameraTrack ? frameStart : 0);
    const startCamera = buildCamera(firstPose);

    if (projection === 'equirect') {
        logger.info(`${width}x${height} equirect`);
    } else if (startCamera.apertureScale! > 0) {
        logger.info(`${width}x${height} fov ${+firstPose.fov.toFixed(3)}° f/${fStop} focus ${startCamera.focusDistance!.toFixed(3)} sensor ${options.sensorSize ?? 0.024}`);
    } else {
        logger.info(`${width}x${height} fov ${+firstPose.fov.toFixed(3)}°`);
    }
    if (cameraTrack) {
        logger.info(`camera track: frames ${frameStart}-${frameEnd} of ${cameraTrack.frameCount} at ${cameraTrack.frameRate} fps`);
    }
    if (motionEnabled) {
        logger.info(`motion blur: shutter ${motionShutter}, ${motionN} sample${motionN === 1 ? '' : 's'}`);
    }

    // Resident when the scene fits the device and the budget: one upload,
    // then every pass sends only the camera. Otherwise the renderer streams
    // the gaussians from the source per pass in depth order, sorting on the
    // GPU while the positions fit and on the CPU past that.
    const scene = new SceneRenderer(device, source, pool, {
        projection,
        width: width!,
        height: height!,
        background,
        residentBudget
    });
    const tierNote: Record<SceneTier, string> = {
        'resident': 'scene resident on GPU',
        'streamed-gpu': 'scene streamed per pass, positions and depth sort on GPU',
        'streamed-cpu': 'scene streamed per pass, depth sort on CPU'
    };

    try {
        const tier = await scene.upload();
        logger.info(`${tierNote[tier]} (${fmtBytes(scene.gpuBytes)})`);
        const renderView = (camera: RenderCamera): Promise<Uint8Array> => scene.render(camera);

        // One output frame centered on time `center`, its shutter spanning
        // ±halfWin around it. Along a track the window is clipped to the
        // track's frames: a looping timeline wraps from its last frame back
        // to its first, and a shutter reaching into that jump would ghost the
        // end frames of a clip that plays once.
        const renderFrame = (center: number, halfWin: number): Promise<Uint8Array> => {
            if (!motionEnabled) {
                return renderView(buildCamera(poseAt(center)));
            }
            const t0 = cameraTrack ? Math.max(0, center - halfWin) : center - halfWin;
            const t1 = cameraTrack ? Math.min(cameraTrack.frameCount - 1, center + halfWin) : center + halfWin;
            // Frame averaging: render the pose at N evenly spaced instants
            // across the shutter and accumulate them on the GPU in float,
            // quantizing once. Each instant composites exactly, so the mean
            // converges to the true time average as N grows; too few show as
            // discrete copies where the motion between them exceeds a couple
            // of pixels.
            const instants: RenderCamera[] = [];
            for (let i = 0; i < motionN; i++) {
                instants.push(buildCamera(poseAt(t0 + (t1 - t0) * (i + 0.5) / motionN)));
            }
            return motionN === 1 ? renderView(instants[0]) : scene.renderSlices(instants);
        };

        const webPCodec = await WebPCodec.create(); // cheap: create() memoizes the wasm module
        const halfWin = motionShutter / 2;

        if (!cameraTrack) {
            // A still is the start pose; a blurred frame centres on the segment.
            const rgba = await renderFrame(motionEnabled ? 0.5 : 0, halfWin);
            const encodingGroup = logger.group('Encoding');
            const webp = webPCodec.encodeLosslessRGBA(rgba, width, height, width * 4, webpEffort);
            encodingGroup.end();
            await writeFile(fs, filename, webp);
            logWrittenFile(basename(filename), webp.byteLength);
        } else {
            // Each frame encodes on a worker thread (and writes) while the
            // next one renders; the RGBA buffer is transferred, not copied.
            // Encode failures surface at the next frame or at the end.
            const bar = logger.bar('frames', frameCount);
            const tStart = performance.now();
            let renderMs = 0;
            let totalBytes = 0;
            let encodeError: unknown = null;
            const pending = new Set<Promise<void>>();
            const encodeFrame = async (rgba: Uint8Array, frameFile: string): Promise<void> => {
                try {
                    const webp = await runEncodeWebp(rgba, width, height, webpEffort);
                    await writeFile(fs, frameFile, webp);
                    totalBytes += webp.byteLength;
                } catch (e) {
                    encodeError ??= e;
                }
            };
            for (let f = frameStart; f <= frameEnd; f++) {
                const t0 = performance.now();
                const rgba = await renderFrame(f, halfWin);
                renderMs += performance.now() - t0;
                const job: Promise<void> = encodeFrame(rgba, frameFilename(filename, f, frameEnd)).finally(() => pending.delete(job));
                pending.add(job);
                if (pending.size >= MAX_PENDING_ENCODES) await Promise.race(pending);
                if (encodeError) throw encodeError;
                bar.update(f - frameStart + 1);
            }
            await Promise.all(pending);
            if (encodeError) throw encodeError;
            bar.end();
            const first = basename(frameFilename(filename, frameStart, frameEnd));
            const last = basename(frameFilename(filename, frameEnd, frameEnd));
            const where = WorkerQueue.isInline ? 'inline' : 'on worker threads';
            logger.info(`${frameCount} frames ${first} … ${last} (${fmtBytes(totalBytes)}): render ${fmtTime(renderMs)}, total ${fmtTime(performance.now() - tStart)} with encode ${where}`);
        }
    } finally {
        scene.destroy();
    }

    g.end();
};

export { writeImage, type WriteImageOptions };
