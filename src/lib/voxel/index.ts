// Voxelization module for Gaussian splat scenes

export {
    computeGaussianExtents,
    getGaussianAABB,
    gaussianOverlapsBox
} from './gaussian-aabb.js';

export type { GaussianExtentsResult } from './gaussian-aabb.js';

export { GaussianBVH } from './gaussian-bvh.js';

export type { GaussianBVHNode, BVHBounds } from './gaussian-bvh.js';

export { GpuVoxelization } from './gpu-voxelization.js';

export type { VoxelizationResult } from './gpu-voxelization.js';

export {
    // Morton code functions
    xyzToMorton,
    mortonToXYZ,

    // Utility functions
    popcount,
    isSolid,
    isEmpty,
    getChildOffset,

    // Accumulator
    BlockAccumulator,

    // Octree construction
    buildSparseOctree,
    alignGridBounds,

    // Constants
    SOLID_LEAF_MARKER,
    MIXED_LEAF_MARKER
} from './sparse-octree.js';

export type { SparseOctree, Bounds } from './sparse-octree.js';
