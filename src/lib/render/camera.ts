import { Vec3 } from 'playcanvas';

/**
 * Camera projection mode for headless splat rendering.
 *
 * - `pinhole` — standard perspective projection with `fovY` as the
 *   vertical field of view. Splats are culled by camera-space `cz <= near`.
 * - `equirect` — full 360° × 180° equirectangular panorama from the
 *   camera position. `fovY` is unused (the projection covers the entire
 *   sphere); `width` must equal `2 × height` (standard 2:1 panorama
 *   aspect). Splats are culled by radial distance `r <= near`.
 *
 * For both modes the `right`/`down`/`forward` basis derived from
 * `position`, `target`, and `up` defines the orientation. In equirect
 * mode, image-x = 0 maps to azimuth -π (behind the camera, left edge)
 * and image-x = width/2 maps to the forward direction; image-y = 0 is
 * the zenith and image-y = height is the nadir.
 */
type Projection = 'pinhole' | 'equirect';

/**
 * Parameters describing a camera for headless splat rendering.
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
    /**
     * Projection mode. Defaults to `'pinhole'` if omitted — back-compatible
     * with callers (JS, or TS compiled against the pre-equirect type) that
     * don't know the field exists.
     */
    projection?: Projection;
    /** Camera position in world space. */
    position: Vec3;
    /** Point the camera looks at, in world space. */
    target: Vec3;
    /** World-space up vector (used to define camera roll). */
    up: Vec3;
    /** Vertical field of view in radians. Used only for `pinhole` projection. */
    fovY: number;
    /** Output image width in pixels. Equirect requires `width === 2 × height`. */
    width: number;
    /** Output image height in pixels. */
    height: number;
    /** Near clipping distance in world units. For pinhole, splats with `cz <= near` are culled; for equirect, splats with radial `r <= near`. */
    near: number;
    /**
     * Camera-space Z of the focus plane in world units. Pinhole only;
     * ignored for equirect. Optional — only meaningful when
     * `apertureScale > 0`.
     */
    focusDistance?: number;
    /**
     * DoF strength as a pixel-space scalar: the CoC radius in pixels
     * when `|1 − focusDistance/cz| = 1`. Pinhole only; ignored for
     * equirect. Default `0` disables defocus.
     */
    apertureScale?: number;
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

    // Pinhole focal lengths (pixels). Equirect ignores these — the
    // equirect shader path derives its scale from imageWidth/Height
    // directly — so we leave them at zero rather than evaluating
    // `tan(fovY/2)`, which is meaningless when the caller has no FOV
    // to provide.
    let focalX = 0;
    let focalY = 0;
    // Default to pinhole when projection is omitted — keeps existing JS
    // callers (no TS type-checking against the new field) working.
    if ((camera.projection ?? 'pinhole') === 'pinhole') {
        const halfTanY = Math.tan(camera.fovY * 0.5);
        focalY = (camera.height * 0.5) / halfTanY;
        focalX = focalY;
    }

    return {
        eye: new Vec3(camera.position.x, camera.position.y, camera.position.z),
        right,
        down,
        forward,
        focalX,
        focalY
    };
};

export { type Projection, type RenderCamera, type CameraBasis, buildCameraBasis };
