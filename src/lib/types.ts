/**
 * Collision mesh shape generated alongside voxel output.
 *
 * - `smooth` - marching cubes with lossless coplanar merge.
 * - `faces` - direct watertight voxel-boundary faces.
 */
type CollisionMeshShape = 'smooth' | 'faces';

/**
 * Options for read/write operations.
 */
type Options = {
    /** Number of iterations for SOG SH compression (higher = better quality). Default: 10 */
    iterations: number;

    /** LOD levels to read from LCC input */
    lodSelect: number[];

    /** Viewer settings JSON for HTML output */
    viewerSettingsJson?: any;

    /** Whether to generate unbundled HTML output with separate files */
    unbundled: boolean;

    /** Approximate number of Gaussians per LOD chunk (in thousands). Default: 512 */
    lodChunkCount: number;

    /** Approximate size of an LOD chunk in world units (meters). Default: 16 */
    lodChunkExtent: number;

    /** SPZ format version to write. Default: 4. */
    spzVersion?: 3 | 4;

    /** Size of each voxel in world units for voxel output. Default: 0.05 */
    voxelResolution?: number;

    /** Opacity threshold for solid voxels - voxels below this are considered empty. Default: 0.1 */
    opacityCutoff?: number;

    /** Exterior fill radius in world units. Enables exterior fill when set. Requires navSeed. */
    navExteriorRadius?: number;

    /** Fill each voxel column upward from the bottom until hitting solid. Runs before carve. Default: false */
    floorFill?: boolean;

    /** When `floorFill` is enabled, dilation radius in world units used to identify "interior" XZ columns to patch. Empty XZ areas larger than `2 * floorFillDilation` from any solid column are treated as exterior and left empty. Default: 0 (patch every empty column). */
    floorFillDilation?: number;

    /** Capsule dimensions for carve. Height of 0 disables carve. Requires navSeed. */
    navCapsule?: { height: number; radius: number };

    /** Seed position in world space for exterior fill and carve flood fill. */
    navSeed?: { x: number; y: number; z: number };

    /** When set, a collision mesh (.collision.glb) is generated alongside the voxel output. `true` is equivalent to `smooth`. */
    collisionMesh?: boolean | CollisionMeshShape;

    /** Camera projection for image output: `'pinhole'` (default) or `'equirect'` (360°×180° panorama). */
    renderProjection?: 'pinhole' | 'equirect';

    /** Camera position (world space) for image output. Default: (2, 1, -2). */
    renderCameraPosition?: { x: number; y: number; z: number };

    /** Camera look-at target (world space) for image output. Default: (0, 0, 0). */
    renderLookAt?: { x: number; y: number; z: number };

    /** World-space up vector for image output. Default: (0, 1, 0). */
    renderUp?: { x: number; y: number; z: number };

    /** Vertical field of view in degrees for image output. Default: 60. */
    renderFov?: number;

    /** Output image width in pixels. Default: 1280. */
    renderWidth?: number;

    /** Output image height in pixels. Default: 720. */
    renderHeight?: number;

    /** Near clip distance for image output. Default: 0.2. */
    renderNear?: number;

    /** RGBA background (each channel in [0, 1]) for image output. Default: (0, 0, 0, 1). */
    renderBackground?: { r: number; g: number; b: number; a: number };

    /**
     * Aperture as a photographic f-stop (e.g. 2.8, 5.6, 11) for image
     * output. Enables defocus blur / depth-of-field: smaller numbers =
     * stronger blur. Defaults to disabled. Pinhole projection only.
     */
    renderFStop?: number;

    /**
     * Camera-space Z of the focus plane in world units for image output.
     * Defaults to the distance from the camera to the look-at point when
     * `renderFStop` is set. No effect without `renderFStop`. Pinhole
     * projection only.
     */
    renderFocusDistance?: number;

    /**
     * Vertical sensor height in world units. Calibrates `renderFStop`
     * to your world scale. Default `0.024` (35mm full-frame in meters).
     * No effect without `renderFStop`. Pinhole projection only.
     */
    renderSensorSize?: number;

    /**
     * End camera position for motion blur. When set, enables camera
     * motion blur: the renderer averages multiple sub-frames with the
     * camera interpolated between `renderCameraPosition` (shutter open)
     * and `renderCameraEndPosition` (shutter close).
     */
    renderCameraEndPosition?: { x: number; y: number; z: number };

    /**
     * End look-at target for motion blur. Defaults to `renderLookAt`
     * when motion blur is enabled. Only meaningful with
     * `renderCameraEndPosition`.
     */
    renderLookAtEnd?: { x: number; y: number; z: number };

    /**
     * End up vector for motion blur. Defaults to `renderUp` when
     * motion blur is enabled. Only meaningful with
     * `renderCameraEndPosition`.
     */
    renderUpEnd?: { x: number; y: number; z: number };

    /**
     * Shutter fraction in `[0, 1]`. Controls what portion of the
     * start→end camera segment is integrated, centered on the midpoint
     * (standard shutter-angle convention: 1.0 = full motion, 0.5 = 180°
     * shutter). Default: `1`. No effect without `renderCameraEndPosition`.
     */
    renderShutter?: number;

    /**
     * Number of sub-frames to accumulate for motion blur. More samples =
     * smoother streaks at proportionally higher cost. Default: `16`.
     * No effect without `renderCameraEndPosition`.
     */
    renderMotionSamples?: number;
};

/**
 * Parameter passed to MJS generator scripts.
 * @ignore
 */
type Param = {
    name: string;
    value: string;
};

/**
 * A function that creates a PlayCanvas GraphicsDevice on demand.
 *
 * Used for GPU-accelerated operations such as SOG compression and voxelization.
 * The application is responsible for caching if needed.
 *
 * @returns Promise resolving to a GraphicsDevice instance.
 */
type DeviceCreator = () => Promise<import('playcanvas').GraphicsDevice>;

export type { CollisionMeshShape, Options, Param, DeviceCreator };
