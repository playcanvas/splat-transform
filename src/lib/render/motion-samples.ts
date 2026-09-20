import { type CameraBasis, type RenderCamera, buildCameraBasis } from './camera';

/** Screen-space step between consecutive shutter samples the default count aims for, in pixels. */
const MOTION_PX_PER_SAMPLE = 2;

/** Most renders averaged per motion-blurred frame. */
const MAX_MOTION_SAMPLES = 64;

type Point = { x: number; y: number; z: number };

/**
 * Screen position of a world point under a camera, or null when the point
 * is behind the near plane (pinhole) or inside the near sphere (equirect).
 *
 * @param camera - The camera.
 * @param basis - Its basis.
 * @param p - The point.
 * @returns Pixel coordinates, or null.
 */
const project = (camera: RenderCamera, basis: CameraBasis, p: Point): { x: number; y: number } | null => {
    const wx = p.x - basis.eye.x, wy = p.y - basis.eye.y, wz = p.z - basis.eye.z;
    const cx = basis.right.x * wx + basis.right.y * wy + basis.right.z * wz;
    const cy = basis.down.x * wx + basis.down.y * wy + basis.down.z * wz;
    const cz = basis.forward.x * wx + basis.forward.y * wy + basis.forward.z * wz;
    if ((camera.projection ?? 'pinhole') === 'pinhole') {
        if (!(cz > camera.near)) return null;
        return { x: basis.focalX * cx / cz + camera.width * 0.5, y: basis.focalY * cy / cz + camera.height * 0.5 };
    }
    const r = Math.hypot(cx, cy, cz);
    if (!(r > camera.near)) return null;
    return {
        x: (Math.atan2(cx, cz) / (2 * Math.PI) + 0.5) * camera.width,
        y: (Math.asin(Math.min(1, Math.max(-1, cy / r))) / Math.PI + 0.5) * camera.height
    };
};

/**
 * Renders to average for a motion-blurred frame whose shutter opens at
 * `open` and closes at `close`: enough that consecutive instants move the
 * picture about {@link MOTION_PX_PER_SAMPLE} pixels. Nothing is known about
 * the scene here, so the motion is measured on the look-at point and on six
 * points around it, one either way along each camera axis, at the radius a
 * subject filling the frame would have (half the visible height at the
 * look-at distance; half that distance for equirect). Points behind either
 * pose are skipped; equirect displacement takes the short way round the
 * seam.
 *
 * @param open - Camera at shutter open.
 * @param close - Camera at shutter close.
 * @returns Sample count in [1, {@link MAX_MOTION_SAMPLES}].
 */
const motionSampleCount = (open: RenderCamera, close: RenderCamera): number => {
    const a = buildCameraBasis(open);
    const b = buildCameraBasis(close);
    const t = open.target;
    const d = Math.hypot(t.x - open.position.x, t.y - open.position.y, t.z - open.position.z);
    const equirect = (open.projection ?? 'pinhole') === 'equirect';
    const r = equirect ? 0.5 * d : d * Math.tan(open.fovY * 0.5);
    const points: Point[] = [{ x: t.x, y: t.y, z: t.z }];
    for (const axis of [a.right, a.down, a.forward]) {
        for (const s of [-r, r]) {
            points.push({ x: t.x + s * axis.x, y: t.y + s * axis.y, z: t.z + s * axis.z });
        }
    }
    let maxPx = 0;
    for (const p of points) {
        const pa = project(open, a, p);
        const pb = project(close, b, p);
        if (!pa || !pb) continue;
        let dx = pb.x - pa.x;
        if (equirect) {
            const w = open.width;
            if (dx > w * 0.5) dx -= w; else if (dx < -w * 0.5) dx += w;
        }
        maxPx = Math.max(maxPx, Math.hypot(dx, pb.y - pa.y));
    }
    return Math.min(MAX_MOTION_SAMPLES, Math.max(1, Math.ceil(maxPx / MOTION_PX_PER_SAMPLE)));
};

export { motionSampleCount, MOTION_PX_PER_SAMPLE, MAX_MOTION_SAMPLES };
