import { Quat, Vec3 } from 'playcanvas';

import { logger, type Transform } from './utils';

/**
 * The optional `camera` block of a SOG `meta.json` (and of a streamed SOG's
 * `lod-meta.json`): one camera that says how the scene is meant to be opened.
 * Purely advisory — a reader that doesn't want it ignores it.
 *
 * Positions and distances are in the same coordinates as the file's gaussians
 * (scene units; metres for a metric capture). Every field is optional, and keys
 * this type doesn't list are carried through untouched.
 */
type SogCamera = {
    /**
     * The camera's local axes. `opencv` (+x right, +y down, +z forward) is what
     * COLMAP and 3DGS trainers use.
     */
    convention?: string;
    /**
     * The intended framing: `camera` (open at the rest camera and keep its
     * frustum) or `display` (orbit the subject; `rest` is the initial viewpoint).
     */
    rig?: string;
    /** The rest pose: camera centre, and camera-to-world rotation as `[x, y, z, w]`. */
    rest?: { position?: number[]; rotation?: number[] };
    /** Pinhole intrinsics in pixels of a `width` × `height` image. */
    intrinsics?: { fx?: number; fy?: number; cx?: number; cy?: number; width?: number; height?: number };
    /** A stereo capture's second eye sits at `+x · baseline_m` in the rest camera's frame. */
    stereo?: { baseline_m?: number };
    /**
     * Scene facts a viewer can't recover from the geometry: the focus / orbit
     * point (in the rest camera's frame), the subject depth and a depth range.
     */
    focus?: { point?: number[]; subject_m?: number; near_m?: number; far_m?: number };
};

const isObject = (v: unknown): v is Record<string, any> => typeof v === 'object' && v !== null && !Array.isArray(v);
const isVec = (v: unknown, n: number): v is number[] => Array.isArray(v) && v.length === n && v.every(Number.isFinite);

/**
 * Validate a `camera` value read from a meta file, returning a deep copy (so a
 * later transform never writes into the parsed document), or `undefined` with a
 * warning when it isn't an object.
 *
 * @param value - The parsed `camera` value.
 * @returns The camera block, or `undefined`.
 * @ignore
 */
const readSogCamera = (value: unknown): SogCamera | undefined => {
    if (value === undefined) return undefined;
    if (!isObject(value)) {
        logger.warn('ignoring meta.json \'camera\': expected an object');
        return undefined;
    }
    return structuredClone(value) as SogCamera;
};

/**
 * Re-express a camera block under a coordinate-space transform: the rest pose
 * gets the full transform, and scene distances (`focus`, `stereo`) the uniform
 * scale. Intrinsics and unknown keys are unchanged. Returns a new object.
 *
 * @param camera - The camera block.
 * @param transform - The transform to apply.
 * @returns The transformed camera block.
 */
const transformSogCamera = (camera: SogCamera, transform: Transform): SogCamera => {
    const result = structuredClone(camera);
    if (transform.isIdentity()) return result;

    const { rest, stereo, focus } = result;
    if (rest && isVec(rest.position, 3)) {
        const p = transform.transformPoint(new Vec3(rest.position[0], rest.position[1], rest.position[2]), new Vec3());
        rest.position = [p.x, p.y, p.z];
    }
    if (rest && isVec(rest.rotation, 4)) {
        const [x, y, z, w] = rest.rotation;
        const q = new Quat().mul2(transform.rotation, new Quat(x, y, z, w)).normalize();
        rest.rotation = [q.x, q.y, q.z, q.w];
    }

    const s = transform.scale;
    if (s !== 1) {
        if (stereo && Number.isFinite(stereo.baseline_m)) stereo.baseline_m *= s;
        if (focus) {
            if (isVec(focus.point, 3)) focus.point = focus.point.map(v => v * s);
            if (Number.isFinite(focus.subject_m)) focus.subject_m *= s;
            if (Number.isFinite(focus.near_m)) focus.near_m *= s;
            if (Number.isFinite(focus.far_m)) focus.far_m *= s;
        }
    }
    return result;
};

// Rotation matrix given as rows -> unit quaternion [x, y, z, w].
const quatFromRows = (m: number[][]): number[] => {
    const [[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]] = m;
    const trace = m00 + m11 + m22;
    let x, y, z, w;
    if (trace > 0) {
        const s = 0.5 / Math.sqrt(trace + 1);
        w = 0.25 / s;
        x = (m21 - m12) * s;
        y = (m02 - m20) * s;
        z = (m10 - m01) * s;
    } else if (m00 > m11 && m00 > m22) {
        const s = 2 * Math.sqrt(1 + m00 - m11 - m22);
        w = (m21 - m12) / s;
        x = 0.25 * s;
        y = (m01 + m10) / s;
        z = (m02 + m20) / s;
    } else if (m11 > m22) {
        const s = 2 * Math.sqrt(1 + m11 - m00 - m22);
        w = (m02 - m20) / s;
        x = (m01 + m10) / s;
        y = 0.25 * s;
        z = (m12 + m21) / s;
    } else {
        const s = 2 * Math.sqrt(1 + m22 - m00 - m11);
        w = (m10 - m01) / s;
        x = (m02 + m20) / s;
        y = (m12 + m21) / s;
        z = 0.25 * s;
    }
    const len = Math.hypot(x, y, z, w);
    return [x / len, y / len, z / len, w / len];
};

/**
 * Build a camera block from one entry of a 3DGS training `cameras.json` (the
 * INRIA layout: `position`, a camera-to-world `rotation` matrix given as rows,
 * pixel `fx`/`fy`, `width`/`height`). Those poses are in the trained PLY's own
 * coordinates with OpenCV camera axes, so they are used as-is. The file has no
 * principal point, so it is taken as the image centre.
 *
 * @param cameras - The parsed `cameras.json` (an array of cameras).
 * @param index - Which camera to use. Default: 0.
 * @returns The camera block.
 * @throws If the entry is missing or malformed.
 */
const sogCameraFromCamerasJson = (cameras: unknown, index = 0): SogCamera => {
    if (!Array.isArray(cameras)) {
        throw new Error('cameras.json: expected an array of cameras');
    }
    const cam = cameras[index];
    if (!isObject(cam)) {
        throw new Error(`cameras.json: no camera at index ${index} (${cameras.length} cameras)`);
    }
    const { position, rotation, fx, fy, width, height } = cam;
    if (!isVec(position, 3) || !Array.isArray(rotation) || rotation.length !== 3 || !rotation.every(r => isVec(r, 3)) ||
        ![fx, fy, width, height].every(v => Number.isFinite(v) && v > 0)) {
        throw new Error(`cameras.json: camera ${index} needs position, a 3x3 rotation, fx, fy, width and height`);
    }
    return {
        convention: 'opencv',
        rest: { position: [...position], rotation: quatFromRows(rotation) },
        intrinsics: { fx, fy, cx: width / 2, cy: height / 2, width, height }
    };
};

export { readSogCamera, sogCameraFromCamerasJson, transformSogCamera };
export type { SogCamera };
