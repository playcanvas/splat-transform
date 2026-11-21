import type { GpuDilation } from '../gpu';

import { gpuDilate3 } from './dilation';
import type { SparseVoxelGrid } from './sparse-voxel-grid';

/**
 * Number of voxels the solid voxels are grown by to close the gaps up to a
 * given width: a gap of up to `2 * halfExtent` voxels closes.
 *
 * @param width - Width of the widest gap to close, in world units.
 * @param voxelResolution - Size of a voxel in world units.
 * @returns Half-extent of the closing in voxels, at least 1.
 */
const closeGapsHalfExtent = (width: number, voxelResolution: number): number => {
    return Math.max(1, Math.ceil(width / (2 * voxelResolution) - 1e-6));
};

/**
 * Close the holes and narrow gaps in the solid voxels, by a morphological
 * closing: the solid voxels are dilated by a box, then eroded by the same
 * box, the erosion done as a dilation of the empty voxels. A gap of up to
 * `width` between solid voxels fills, while the surfaces around it stay where
 * they are. Unlike a dilation, which grows every surface, the closing only
 * adds voxels in holes, gaps and concave corners, and never removes any.
 *
 * Splats which cover a surface thinly voxelize into a surface full of small
 * holes. A hole through a floor one voxel thick has only 4 of its 6
 * neighbors solid, so the isolated voxel fill of the voxelization leaves it
 * open, and anything colliding with the voxels falls through it.
 *
 * The erosion counts the voxels outside the grid as solid, so the gaps within
 * `closeGapsHalfExtent` voxels of the grid boundary close as if the boundary
 * was solid. Give the grid a margin of empty voxels that wide to avoid it.
 *
 * @param grid - Input grid (read-only).
 * @param voxelResolution - Size of a voxel in world units.
 * @param width - Width of the widest gap to close, in world units.
 * @param gpu - GPU dilation context.
 * @returns Newly allocated closed grid.
 */
const closeGaps = async (
    grid: SparseVoxelGrid,
    voxelResolution: number,
    width: number,
    gpu: GpuDilation
): Promise<SparseVoxelGrid> => {
    if (!Number.isFinite(voxelResolution) || voxelResolution <= 0) {
        throw new Error(`closeGaps: voxelResolution must be finite and > 0, got ${voxelResolution}`);
    }
    if (!Number.isFinite(width) || width <= 0) {
        throw new Error(`closeGaps: width must be finite and > 0, got ${width}`);
    }

    const halfExtent = closeGapsHalfExtent(width, voxelResolution);
    const { nbx, nby, nbz } = grid;

    const dilated = await gpuDilate3(gpu, grid, halfExtent, halfExtent);
    const empty = dilated.cropToInverted(0, 0, 0, nbx, nby, nbz);
    dilated.releaseStorage();

    const emptyDilated = await gpuDilate3(gpu, empty, halfExtent, halfExtent);
    empty.releaseStorage();
    const closed = emptyDilated.cropToInverted(0, 0, 0, nbx, nby, nbz);
    emptyDilated.releaseStorage();

    return closed;
};

export { closeGaps, closeGapsHalfExtent };
