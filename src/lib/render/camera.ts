import { Vec3 } from 'playcanvas';

/**
 * Parameters describing a pinhole camera for headless splat rendering.
 *
 * Convention: PlayCanvas right-handed, Y-up world. Camera-space axes are
 * `right` (image x increases), `down` (image y increases), `forward`
 * (positive depth in front of the camera). The basis is built from a
 * `target - eye` direction plus a world-up vector and assumes the camera
 * looks roughly opposite the world up's perpendicular plane in a
 * PlayCanvas-style scene (so e.g. camera at +Z looking at the origin
 * places world `+X` on the right of the image).
 */
type RenderCamera = {
    /** Camera position in world space. */
    position: Vec3;
    /** Point the camera looks at, in world space. */
    target: Vec3;
    /** World-space up vector (used to define camera roll). */
    up: Vec3;
    /** Vertical field of view in radians. */
    fovY: number;
    /** Output image width in pixels. */
    width: number;
    /** Output image height in pixels. */
    height: number;
    /** Near clipping distance in world units. Splats with depth <= near are culled. */
    near: number;
};

/**
 * Derived camera basis and intrinsics in the conventions described on
 * RenderCamera. The 3x3 view rotation rows are (right, down, forward); the
 * translation is -R * eye. Focal lengths are in pixel units.
 */
type CameraBasis = {
    /** Eye position (copy of camera.position). */
    eye: Vec3;
    /** Camera-space +X axis in world coords (rightward in image). */
    right: Vec3;
    /** Camera-space +Y axis in world coords (downward in image). */
    down: Vec3;
    /** Camera-space +Z axis in world coords (into the scene). */
    forward: Vec3;
    /** Horizontal focal length in pixels. */
    focalX: number;
    /** Vertical focal length in pixels. */
    focalY: number;
};

const sub = (a: Vec3, b: Vec3) => new Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
const cross = (a: Vec3, b: Vec3) => new Vec3(
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x
);
const normalize = (v: Vec3): Vec3 => {
    const len = Math.hypot(v.x, v.y, v.z);
    if (len === 0) return new Vec3(0, 0, 0);
    return new Vec3(v.x / len, v.y / len, v.z / len);
};

const buildCameraBasis = (camera: RenderCamera): CameraBasis => {
    const forward = normalize(sub(camera.target, camera.position));
    if (forward.x === 0 && forward.y === 0 && forward.z === 0) {
        throw new Error('Camera target equals camera position');
    }

    // right = forward × up_world (image x increases to the right).
    // For a PlayCanvas-style camera at +Z looking toward the origin (forward
    // pointing in world -Z), this yields right = world +X, so world +X
    // appears on the right side of the image. The opposite ordering would
    // be correct for a CV-style camera looking down +Z; we choose this one
    // because the renderer always consumes PlayCanvas-identity-space data.
    let right = cross(forward, camera.up);
    if (right.x === 0 && right.y === 0 && right.z === 0) {
        throw new Error('Camera up vector is parallel to view direction');
    }
    right = normalize(right);

    // down = forward × right (image y increases downward)
    const down = cross(forward, right);

    const halfTanY = Math.tan(camera.fovY * 0.5);
    const focalY = (camera.height * 0.5) / halfTanY;
    const focalX = focalY;

    return {
        eye: new Vec3(camera.position.x, camera.position.y, camera.position.z),
        right,
        down,
        forward,
        focalX,
        focalY
    };
};

export { type RenderCamera, type CameraBasis, buildCameraBasis };
