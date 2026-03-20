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

    /** Opacity threshold for solid voxels - voxels below this are considered empty. Default: 0.5 */
    opacityCutoff?: number;

    /** Whether to generate a collision mesh (.collision.glb) alongside voxel output. Default: false */
    collisionMesh?: boolean;

    /** Ratio of triangles to keep when simplifying the collision mesh (0-1). Default: 0.25 */
    meshSimplify?: number;

    /** Capsule dimensions for navigation simplification. When set with navSeed, simplifies voxel output. */
    navCapsule?: { height: number; radius: number };

    /** Seed position in world space for navigation flood fill. Required when navCapsule is set. */
    navSeed?: { x: number; y: number; z: number };
};

/**
 * Parameter passed to MJS generator scripts.
 * @ignore
 */
type Param = {
    name: string;
    value: string;
};

export type { Options, Param };
