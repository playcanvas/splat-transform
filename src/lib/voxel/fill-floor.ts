import { BlockMaskBuffer } from './block-mask-buffer';
import type { NavSimplifyResult } from './fill-exterior';
import type { Bounds } from '../data-table';
import {
    BLOCK_EMPTY,
    BLOCK_MIXED,
    BLOCK_SOLID,
    SOLID_HI,
    SOLID_LO,
    SparseVoxelGrid
} from './sparse-voxel-grid';
import { logger } from '../utils';

/**
 * Fill each voxel column upward from the bottom until hitting an existing
 * solid voxel or the top of the grid. Intended to run after carveInterior
 * to seal the floor of the navigable region.
 *
 * Iterates at block granularity for performance, with voxel-accurate
 * filling at mixed-block boundaries.
 *
 * @param buffer - Voxelized scene data.
 * @param gridBounds - Axis-aligned bounds of the voxel grid.
 * @param voxelResolution - Size of each voxel in world units.
 * @returns Modified buffer with columns filled from bottom to first solid.
 */
const fillFloor = (
    buffer: BlockMaskBuffer,
    gridBounds: Bounds,
    voxelResolution: number
): NavSimplifyResult => {
    if (!Number.isFinite(voxelResolution) || voxelResolution <= 0) {
        throw new Error(`fillFloor: voxelResolution must be finite and > 0, got ${voxelResolution}`);
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

    const nbx = nx >> 2;
    const nby = ny >> 2;
    const nbz = nz >> 2;

    logger.progress.begin(2);
    let progressComplete = false;

    try {
        const grid = SparseVoxelGrid.fromBuffer(buffer, nx, ny, nz);
        logger.progress.step();

        for (let bz = 0; bz < nbz; bz++) {
            for (let bx = 0; bx < nbx; bx++) {
                let filling = 0xFFFF;

                for (let by = 0; by < nby && filling; by++) {
                    const blockIdx = bx + by * nbx + bz * (nbx * nby);
                    const bt = grid.blockType[blockIdx];

                    if (bt === BLOCK_SOLID) {
                        break;
                    }

                    if (bt === BLOCK_EMPTY) {
                        if (filling === 0xFFFF) {
                            grid.orBlock(blockIdx, SOLID_LO, SOLID_HI);
                        } else {
                            let lo = 0, hi = 0;
                            for (let lz = 0; lz < 4; lz++) {
                                for (let lx = 0; lx < 4; lx++) {
                                    if (!(filling & (1 << (lz * 4 + lx)))) continue;
                                    for (let ly = 0; ly < 4; ly++) {
                                        const bitIdx = lx + (ly << 2) + (lz << 4);
                                        if (bitIdx < 32) lo |= (1 << bitIdx);
                                        else hi |= (1 << (bitIdx - 32));
                                    }
                                }
                            }
                            grid.orBlock(blockIdx, lo >>> 0, hi >>> 0);
                        }
                        continue;
                    }

                    // BLOCK_MIXED: per-voxel accuracy
                    const s = grid.masks.slot(blockIdx);
                    const existLo = grid.masks.lo[s];
                    const existHi = grid.masks.hi[s];

                    let fillLo = 0;
                    let fillHi = 0;

                    for (let lz = 0; lz < 4; lz++) {
                        for (let lx = 0; lx < 4; lx++) {
                            const subCol = 1 << (lz * 4 + lx);
                            if (!(filling & subCol)) continue;

                            for (let ly = 0; ly < 4; ly++) {
                                const bitIdx = lx + (ly << 2) + (lz << 4);
                                const inHi = bitIdx >= 32;
                                const word = inHi ? existHi : existLo;
                                const bit = 1 << (inHi ? bitIdx - 32 : bitIdx);

                                if (word & bit) {
                                    filling &= ~subCol;
                                    break;
                                }

                                if (inHi) fillHi |= bit;
                                else fillLo |= bit;
                            }
                        }
                    }

                    if (fillLo || fillHi) {
                        grid.orBlock(blockIdx, fillLo >>> 0, fillHi >>> 0);
                    }
                }
            }
        }

        const result = grid.toBuffer(0, 0, 0, nbx, nby, nbz);
        logger.progress.step();
        progressComplete = true;

        return { buffer: result, gridBounds };
    } finally {
        if (!progressComplete) {
            logger.progress.cancel();
        }
    }
};

export { fillFloor };
