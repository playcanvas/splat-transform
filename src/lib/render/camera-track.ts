/**
 * Camera animation tracks for multi-frame rendering.
 *
 * A {@link CameraTrack} answers "where is the camera at frame t" for any
 * fractional frame time, which is what the image writer needs both for the
 * frames themselves and for the shutter slices of motion blur between them.
 *
 * Three JSON sources are understood by {@link loadCameraTrack}:
 *
 * - The supersplat editor's project document (`document.json` inside an
 *   `.ssproj`): `poseSets[0].poses` keyed by frame, with `timeline.frames`,
 *   `frameRate`, `smoothness` and `loop`. Evaluated exactly as the editor's
 *   timeline does, so rendered frames match what the editor shows.
 * - The supersplat viewer's `settings.json`: the first of `animTracks`
 *   (keyframe times in frames, `spline` or `step` interpolation, loop mode).
 * - A plain per-frame list: `{ frameRate?, frames: [{ position, target, fov?, up? }] }`,
 *   linearly interpolated between entries for fractional times.
 *
 * All poses are in the PlayCanvas default (viewer/editor) space, like the
 * writer's camera options; the target doubles as the defocus focus point.
 * Only the frame list can tilt the camera: a frame's `up` sets its up
 * vector (roll about the view direction); poses without one, and the editor
 * and viewer formats, which carry none, use the caller's default up.
 */

type Vec3Like = { x: number; y: number; z: number };

/** A camera pose on a track: position, look-at target, up vector and vertical fov in degrees. */
type TrackPose = {
    position: Vec3Like;
    target: Vec3Like;
    up: Vec3Like;
    fov: number;
};

const DEFAULT_UP: Vec3Like = { x: 0, y: 1, z: 0 };

/** A camera animation track evaluated in frame time. */
interface CameraTrack {
    /** Frames per second. */
    frameRate: number;
    /** Frames the track defines, rendered as `0 … frameCount − 1`. */
    frameCount: number;
    /** Pose at a (fractional) frame time. */
    poseAt(frame: number): TrackPose;
}

/**
 * Cubic Hermite spline over multi-dimensional points; a port of the
 * supersplat viewer's `CubicSpline` so tracks evaluate identically here.
 * Knots store (in-tangent, value, out-tangent) per dimension.
 */
class CubicSpline {
    times: number[];
    knots: number[];
    dim: number;

    constructor(times: number[], knots: number[]) {
        this.times = times;
        this.knots = knots;
        this.dim = knots.length / times.length / 3;
    }

    evaluate(time: number, result: number[]): void {
        const { times } = this;
        const last = times.length - 1;
        if (time <= times[0]) {
            this.getKnot(0, result);
        } else if (time >= times[last]) {
            this.getKnot(last, result);
        } else {
            let seg = 0;
            while (time >= times[seg + 1]) {
                seg++;
            }
            this.evaluateSegment(seg, (time - times[seg]) / (times[seg + 1] - times[seg]), result);
        }
    }

    getKnot(index: number, result: number[]): void {
        const { knots, dim } = this;
        const idx = index * 3 * dim;
        for (let i = 0; i < dim; ++i) {
            result[i] = knots[idx + i * 3 + 1];
        }
    }

    evaluateSegment(segment: number, t: number, result: number[]): void {
        const { knots, dim } = this;
        const t2 = t * t;
        const twot = t + t;
        const omt = 1 - t;
        const omt2 = omt * omt;
        let idx = segment * dim * 3;
        for (let i = 0; i < dim; ++i) {
            const p0 = knots[idx + 1];
            const m0 = knots[idx + 2];
            const m1 = knots[idx + dim * 3];
            const p1 = knots[idx + dim * 3 + 1];
            idx += 3;
            result[i] = p0 * ((1 + twot) * omt2) + m0 * (t * omt2) + p1 * (t2 * (3 - twot)) + m1 * (t2 * (t - 1));
        }
    }

    // smoothness: 0 = linear, 1 = smooth
    static calcKnots(times: number[], points: number[], smoothness: number): number[] {
        const n = times.length;
        const dim = points.length / n;
        const knots = new Array<number>(n * dim * 3);
        for (let i = 0; i < n; i++) {
            const t = times[i];
            for (let j = 0; j < dim; j++) {
                const idx = i * dim + j;
                const p = points[idx];
                let tangent;
                if (i === 0) {
                    tangent = (points[idx + dim] - p) / (times[i + 1] - t);
                } else if (i === n - 1) {
                    tangent = (p - points[idx - dim]) / (t - times[i - 1]);
                } else {
                    tangent = (points[idx + dim] - points[idx - dim]) / (times[i + 1] - times[i - 1]);
                }
                const inScale = i > 0 ? times[i] - times[i - 1] : times[1] - times[0];
                const outScale = i < n - 1 ? times[i + 1] - times[i] : times[i] - times[i - 1];
                knots[idx * 3] = tangent * inScale * smoothness;
                knots[idx * 3 + 1] = p;
                knots[idx * 3 + 2] = tangent * outScale * smoothness;
            }
        }
        return knots;
    }

    static fromPoints(times: number[], points: number[], smoothness = 1): CubicSpline {
        return new CubicSpline(times, CubicSpline.calcKnots(times, points, smoothness));
    }

    // create a looping spline by duplicating animation points at the end and beginning
    static fromPointsLooping(length: number, times: number[], points: number[], smoothness = 1): CubicSpline {
        if (times.length < 2) {
            return CubicSpline.fromPoints(times, points);
        }
        const dim = points.length / times.length;
        const newTimes = times.slice();
        const newPoints = points.slice();
        newTimes.push(length + times[0], length + times[1]);
        newPoints.push(...points.slice(0, dim * 2));
        newTimes.splice(0, 0, times[times.length - 2] - length, times[times.length - 1] - length);
        newPoints.splice(0, 0, ...points.slice(points.length - dim * 2));
        return CubicSpline.fromPoints(newTimes, newPoints, smoothness);
    }
}

type Keyframes = {
    /** Key times in frames, ascending. */
    times: number[];
    /** Per key: position xyz, target xyz, fov. */
    points: number[];
};

const vec3Of = (v: unknown, what: string): Vec3Like => {
    if (!Array.isArray(v) || v.length !== 3 || !v.every(Number.isFinite)) {
        throw new Error(`camera track: ${what} must be an array of three numbers`);
    }
    return { x: v[0], y: v[1], z: v[2] };
};

const finiteNumber = (v: unknown, what: string, dflt?: number): number => {
    if (v === undefined && dflt !== undefined) return dflt;
    if (typeof v !== 'number' || !Number.isFinite(v)) {
        throw new Error(`camera track: ${what} must be a number`);
    }
    return v;
};

/**
 * Track over a spline (or a single held pose); shared by the editor and viewer formats.
 *
 * @param keys - Key times (frames) and per-key pose components.
 * @param frameRate - Frames per second.
 * @param frameCount - Frames the track defines.
 * @param smoothness - Spline tangent scale (0 linear, 1 smooth).
 * @param loopLength - Period in frames for a looping spline, or null to hold the end poses.
 * @param step - Hold each key until the next instead of interpolating.
 * @param up - Up vector shared by every pose (the formats have no per-key up).
 * @returns The track.
 */
const splineTrack = (keys: Keyframes, frameRate: number, frameCount: number, smoothness: number, loopLength: number | null, step: boolean, up: Vec3Like): CameraTrack => {
    const { times, points } = keys;
    const result = new Array<number>(7);
    const toPose = (): TrackPose => ({
        position: { x: result[0], y: result[1], z: result[2] },
        target: { x: result[3], y: result[4], z: result[5] },
        up,
        fov: result[6]
    });
    if (times.length === 1) {
        const p: TrackPose = { position: { x: points[0], y: points[1], z: points[2] }, target: { x: points[3], y: points[4], z: points[5] }, up, fov: points[6] };
        return { frameRate, frameCount, poseAt: () => p };
    }
    if (step) {
        return {
            frameRate,
            frameCount,
            poseAt: (frame) => {
                let k = 0;
                while (k + 1 < times.length && frame >= times[k + 1]) k++;
                for (let d = 0; d < 7; d++) result[d] = points[k * 7 + d];
                return toPose();
            }
        };
    }
    const spline = loopLength !== null ?
        CubicSpline.fromPointsLooping(loopLength, times, points, smoothness) :
        CubicSpline.fromPoints(times, points, smoothness);
    return {
        frameRate,
        frameCount,
        poseAt: (frame) => {
            spline.evaluate(frame, result);
            return toPose();
        }
    };
};

/**
 * Supersplat editor project document: `poseSets[0].poses` on the `timeline`.
 * Mirrors the editor's `CameraAnimTrack.rebuildSpline`: poses at or past
 * `timeline.frames` are dropped, keys sort by frame, a looping timeline wraps
 * the spline over its length, otherwise the end poses hold.
 *
 * @param doc - Parsed `document.json`.
 * @param defaultFov - Fallback vertical fov in degrees when neither a pose nor the document carries one.
 * @param up - Up vector for every pose.
 * @returns The track.
 */
const fromEditorDocument = (doc: any, defaultFov: number, up: Vec3Like): CameraTrack => {
    const timeline = doc.timeline ?? {};
    const frameCount = Math.floor(finiteNumber(timeline.frames, 'timeline.frames'));
    const frameRate = finiteNumber(timeline.frameRate, 'timeline.frameRate', 30);
    const smoothness = finiteNumber(timeline.smoothness, 'timeline.smoothness', 1);
    const loop = timeline.loop ?? true;
    const docFov = typeof doc.camera?.fov === 'number' ? doc.camera.fov : defaultFov;
    const poses: any[] = doc.poseSets?.[0]?.poses ?? [];
    if (poses.length === 0) {
        throw new Error('camera track: the project has no camera poses');
    }
    const ordered = poses
    .map((p, i) => ({ frame: finiteNumber(p.frame, `poses[${i}].frame`, i * frameRate), p }))
    .filter(e => e.frame < frameCount)
    .sort((a, b) => a.frame - b.frame);
    if (ordered.length === 0) {
        throw new Error('camera track: every pose lies past the end of the timeline');
    }
    const times: number[] = [];
    const points: number[] = [];
    for (const { frame, p } of ordered) {
        const position = vec3Of(p.position, 'pose position');
        const target = vec3Of(p.target, 'pose target');
        times.push(frame);
        points.push(position.x, position.y, position.z, target.x, target.y, target.z, finiteNumber(p.fov, 'pose fov', docFov));
    }
    return splineTrack({ times, points }, frameRate, frameCount, smoothness, loop ? frameCount : null, false, up);
};

/**
 * Supersplat viewer settings: the first animation track. Mirrors the
 * viewer's `AnimState.fromTrack` (times are in frames; `repeat` wraps the
 * spline over the duration, an extra frame when the last key sits exactly
 * at the end; `none` and `pingpong` hold the end poses).
 *
 * @param settings - Parsed viewer `settings.json`.
 * @param up - Up vector for every pose.
 * @returns The track.
 */
const fromViewerSettings = (settings: any, up: Vec3Like): CameraTrack => {
    const track = settings.animTracks?.[0];
    if (!track) {
        throw new Error('camera track: the settings have no animation tracks');
    }
    const frameRate = finiteNumber(track.frameRate, 'animTrack.frameRate', 30);
    const duration = finiteNumber(track.duration, 'animTrack.duration');
    const smoothness = finiteNumber(track.smoothness, 'animTrack.smoothness', 1);
    const times: number[] = track.keyframes?.times ?? [];
    const { position = [], target = [], fov = [] } = track.keyframes?.values ?? {};
    if (times.length === 0 || position.length !== times.length * 3 || target.length !== times.length * 3) {
        throw new Error('camera track: malformed animTrack keyframes');
    }
    const points: number[] = [];
    for (let i = 0; i < times.length; i++) {
        points.push(position[i * 3], position[i * 3 + 1], position[i * 3 + 2]);
        points.push(target[i * 3], target[i * 3 + 1], target[i * 3 + 2]);
        points.push(finiteNumber(fov[i], `animTrack fov[${i}]`, 60));
    }
    const extra = duration === times[times.length - 1] / frameRate ? 1 : 0;
    const loopLength = track.loopMode === 'repeat' ? (duration + extra) * frameRate : null;
    const frameCount = Math.round(duration * frameRate);
    return splineTrack({ times, points }, frameRate, frameCount, smoothness, loopLength, track.interpolation === 'step', up);
};

/**
 * Plain per-frame list, linearly interpolated between entries so shutter
 * slices can fall between frames (the up vector by normalized lerp, so it
 * stays unit-length when neighbouring frames differ in direction).
 *
 * @param json - Parsed `{ frameRate?, frames[] }` object.
 * @param defaultFov - Fallback vertical fov in degrees for frames without one.
 * @param defaultUp - Fallback up vector for frames without one.
 * @returns The track.
 */
const fromFrameList = (json: any, defaultFov: number, defaultUp: Vec3Like): CameraTrack => {
    const frames: any[] = json.frames;
    if (!Array.isArray(frames) || frames.length === 0) {
        throw new Error('camera track: `frames` must be a non-empty array');
    }
    const frameRate = finiteNumber(json.frameRate, 'frameRate', 30);
    const poses: TrackPose[] = frames.map((f, i) => ({
        position: vec3Of(f.position, `frames[${i}].position`),
        target: vec3Of(f.target, `frames[${i}].target`),
        up: f.up === undefined ? defaultUp : vec3Of(f.up, `frames[${i}].up`),
        fov: finiteNumber(f.fov, `frames[${i}].fov`, defaultFov)
    }));
    const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
    return {
        frameRate,
        frameCount: poses.length,
        poseAt: (frame) => {
            const f = Math.min(Math.max(frame, 0), poses.length - 1);
            const i0 = Math.floor(f);
            const i1 = Math.min(i0 + 1, poses.length - 1);
            const t = f - i0;
            const a = poses[i0], b = poses[i1];
            const ux = lerp(a.up.x, b.up.x, t), uy = lerp(a.up.y, b.up.y, t), uz = lerp(a.up.z, b.up.z, t);
            const ulen = Math.hypot(ux, uy, uz) || 1;
            return {
                position: { x: lerp(a.position.x, b.position.x, t), y: lerp(a.position.y, b.position.y, t), z: lerp(a.position.z, b.position.z, t) },
                target: { x: lerp(a.target.x, b.target.x, t), y: lerp(a.target.y, b.target.y, t), z: lerp(a.target.z, b.target.z, t) },
                up: { x: ux / ulen, y: uy / ulen, z: uz / ulen },
                fov: lerp(a.fov, b.fov, t)
            };
        }
    };
};

/**
 * Build a camera track from parsed JSON, detecting the source by shape.
 *
 * @param json - Parsed contents of an editor `document.json`, a viewer
 * `settings.json`, or a plain `{ frameRate?, frames[] }` list.
 * @param defaultFov - Vertical fov in degrees for poses that carry none.
 * @param defaultUp - Up vector for poses that carry none (only frame-list
 * entries can carry their own). Default: world +Y.
 * @returns The track.
 */
const loadCameraTrack = (json: unknown, defaultFov: number, defaultUp: Vec3Like = DEFAULT_UP): CameraTrack => {
    if (!json || typeof json !== 'object') {
        throw new Error('camera track: expected a JSON object');
    }
    const j = json as any;
    if (Array.isArray(j.poseSets) && j.timeline) return fromEditorDocument(j, defaultFov, defaultUp);
    if (Array.isArray(j.animTracks)) return fromViewerSettings(j, defaultUp);
    if (Array.isArray(j.frames)) return fromFrameList(j, defaultFov, defaultUp);
    throw new Error('camera track: unrecognised format (expected an editor document with poseSets/timeline, viewer settings with animTracks, or a frames list)');
};

export { loadCameraTrack, CubicSpline, type CameraTrack, type TrackPose };
