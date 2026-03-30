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

    /** Enable navigation simplification with default capsule (height 1.6, radius 0.2) and seed (0,0,0). Default: true (set to false to disable). */
    navSimplify?: boolean;

    /** Capsule dimensions (height, radius) for navigation simplification. Default: { height: 1.6, radius: 0.2 } */
    navCapsule?: { height: number; radius: number };

    /** Seed position in world space for navigation flood fill. Default: { x: 0, y: 0, z: 0 } */
    navSeed?: { x: number; y: number; z: number };

    /** Exterior fill radius in world units. When set, fills empty space outside enclosed levels. */
    navExteriorRadius?: number;

    /** Whether to generate a collision mesh (.collision.glb) alongside voxel output. Default: false */
    collisionMesh?: boolean;

    /** Maximum geometric error for collision mesh simplification as a fraction of voxelResolution. Default: 0.08 */
    meshSimplifyError?: number;
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
