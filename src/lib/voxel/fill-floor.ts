import { BlockMaskBuffer } from './block-mask-buffer';
import { sparseDilate3 } from './dilation';
import type { NavSimplifyResult } from './fill-exterior';
import type { Bounds } from '../data-table';
import { SparseVoxelGrid } from './sparse-voxel-grid';
import { logger } from '../utils';

/**
 * Fill below the floor surface to block outdoor scene edges.
 *
 * For each voxel column (ix, iz), scans upward from the bottom of a
 * dilation-bridged copy of the grid to find the first solid voxel (the floor).
 * Everything below the floor is filled with solid in the original grid.
 * Columns with no floor are filled entirely with solid, blocking the void.
 *
 * The XZ dilation bridges small horizontal holes in the floor surface the
 * same way {@link fillExterior} bridges wall gaps. The dilation radius
 * controls how large a gap can be bridged.
 *
 * Since the SVO stores solid blocks as a single sentinel marker and does not
 * store empty blocks, filling large volumes with solid is essentially free.
 *
 * @param buffer - Voxelized scene data.
 * @param gridBounds - Axis-aligned bounds of the voxel grid.
 * @param voxelResolution - Size of each voxel in world units.
 * @param dilation - XZ dilation radius in world units for bridging floor gaps.
 * @returns Modified buffer with below-floor space filled solid.
 */
const fillFloor = (
    buffer: BlockMaskBuffer,
    gridBounds: Bounds,
    voxelResolution: number,
    dilation: number
): NavSimplifyResult => {
    if (!Number.isFinite(voxelResolution) || voxelResolution <= 0) {
        throw new Error(`fillFloor: voxelResolution must be finite and > 0, got ${voxelResolution}`);
    }
    if (!Number.isFinite(dilation) || dilation <= 0) {
        throw new Error(`fillFloor: dilation must be finite and > 0, got ${dilation}`);
    }

    const nx = Math.round((gridBounds.max.x - gridBounds.min.x) / voxelResolution);
    const ny = Math.round((gridBounds.max.y - gridBounds.min.y) / voxelResolution);
    const nz = Math.round((gridBounds.max.z - gridBounds.min.z) / voxelResolution);

    if (nx % 4 !== 0 || ny % 4 !== 0 || nz % 4 !== 0) {
        throw new Error(`Grid dimensions must be multiples of 4, got ${nx}x${ny}x${nz}`);
    }

    if (buffer.count === 0) {
        return { buffer, gridBounds };
    }

    const halfExtent = Math.ceil(dilation / voxelResolution);
    const nbx = nx >> 2;
    const nby = ny >> 2;
    const nbz = nz >> 2;

    logger.progress.begin(5);
    let progressComplete = false;

    try {
        const grid = SparseVoxelGrid.fromBuffer(buffer, nx, ny, nz);
        logger.progress.step();

        const dilated = sparseDilate3(grid, halfExtent, 0);

        for (let iz = 0; iz < nz; iz++) {
            for (let ix = 0; ix < nx; ix++) {
                let floorY = ny;
                for (let iy = 0; iy < ny; iy++) {
                    if (dilated.getVoxel(ix, iy, iz)) {
                        floorY = iy;
                        break;
                    }
                }
                for (let iy = 0; iy < floorY; iy++) {
                    grid.setVoxel(ix, iy, iz);
                }
            }
        }

        logger.progress.step();
        progressComplete = true;

        return {
            buffer: grid.toBuffer(0, 0, 0, nbx, nby, nbz),
            gridBounds
        };
    } finally {
        if (!progressComplete) {
            logger.progress.cancel();
        }
    }
};

export { fillFloor };
