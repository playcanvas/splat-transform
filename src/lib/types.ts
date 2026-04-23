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

    /** Shape of the collision mesh (.collision.glb). When set, a collision mesh is generated. `edge` = axis-aligned greedy voxel surface, `smooth` = marching cubes followed by lossless coplanar merge. */
    meshType?: 'edge' | 'smooth';
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

export type { Options, Param, DeviceCreator };
