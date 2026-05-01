import { BlockMaskBuffer } from './block-mask-buffer';
import { gpuDilate3 } from './dilation';
import type { NavSimplifyResult } from './fill-exterior';
import { sparseOrGrids } from './grid-ops';
import type { Bounds } from '../data-table';
import type { GpuDilation } from '../gpu';
import {
    BLOCK_EMPTY,
    BLOCK_MIXED,
    BLOCK_SOLID,
    SOLID_HI,
    SOLID_LO,
    SparseVoxelGrid,
    readBlockType
} from './sparse-voxel-grid';
import { logger } from '../utils';

/**
 * Floor-fill via XZ dilate -> per-column upward walk -> XZ dilate -> OR.
 *
 * Mirrors the shape of `fillExterior` (dilate -> traverse -> dilate -> OR) but
 * the traversal is a per-(lx, lz) upward walk through empty space instead of a
 * 3D boundary BFS, and the dilations operate only in X and Z.
 *
 * Steps with `r = ceil(dilation / voxelResolution)`:
 *   1. `S_xz = sparseDilate3(S, r, 0)` closes any XZ holes in horizontal
 *      surfaces smaller than `2 * r`.
 *   2. For every (lx, lz), walk `y = 0` upward through `S_xz`. Mark each
 *      visited empty voxel into `foundEmpty`. Stop on the first solid voxel
 *      of `S_xz` or at the grid top.
 *   3. `dilatedFound = sparseDilate3(foundEmpty, r, 0)` spreads the found
 *      under-surface volume back out in XZ to cover the kernel halo.
 *   4. `output = S | dilatedFound` adds the dilated under-surface region as
 *      solid on top of the original solids.
 *
 * Intended to run before `carve`: it seals the under-side of the floor
 * (and patches small XZ holes via the dilation), and the carve handles the
 * remaining hole plugging via its 3D dilate + capsule BFS.
 *
 * With `r = 0` the dilations are skipped and the algorithm degrades to
 * "fill the under-side of every column up to the first solid", matching the
 * original (pre-dilation) `fillFloor` behavior.
 *
 * @param buffer - Voxelized scene data.
 * @param gridBounds - Axis-aligned bounds of the voxel grid.
 * @param voxelResolution - Size of each voxel in world units.
 * @param dilation - XZ dilation radius in world units. 0 disables dilation.
 * @returns Modified buffer with under-surface regions filled.
 */
const fillFloor = async (
    buffer: BlockMaskBuffer,
    gridBounds: Bounds,
    voxelResolution: number,
    dilation: number,
    gpu: GpuDilation
): Promise<NavSimplifyResult> => {
    if (!Number.isFinite(voxelResolution) || voxelResolution <= 0) {
        throw new Error(`fillFloor: voxelResolution must be finite and > 0, got ${voxelResolution}`);
    }
    if (!Number.isFinite(dilation) || dilation < 0) {
        throw new Error(`fillFloor: dilation must be finite and >= 0, got ${dilation}`);
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
    const bStride = nbx * nby;

    const r = dilation > 0 ? Math.ceil(dilation / voxelResolution) : 0;

    logger.debug(`fill floor: ${nx}x${ny}x${nz} grid, dilation radius ${r} voxels`);

    const grid = SparseVoxelGrid.fromBuffer(buffer, nx, ny, nz);
    const dilatedSolid = r > 0 ? await gpuDilate3(gpu, grid, r, 0) : grid;

    const foundEmpty = new SparseVoxelGrid(nx, ny, nz);

    const walkBar = logger.bar('Column walk', nbx * nbz);
    const dilatedTypes = dilatedSolid.types;
    for (let bz = 0; bz < nbz; bz++) {
        for (let bx = 0; bx < nbx; bx++) {
            let walking = 0xFFFF;

            for (let by = 0; by < nby && walking; by++) {
                const blockIdx = bx + by * nbx + bz * bStride;
                const bt = readBlockType(dilatedTypes, blockIdx);

                if (bt === BLOCK_SOLID) {
                    break;
                }

                if (bt === BLOCK_EMPTY) {
                    if (walking === 0xFFFF) {
                        foundEmpty.orBlock(blockIdx, SOLID_LO, SOLID_HI);
                    } else {
                        let lo = 0, hi = 0;
                        for (let lz = 0; lz < 4; lz++) {
                            for (let lx = 0; lx < 4; lx++) {
                                if (!(walking & (1 << (lz * 4 + lx)))) continue;
                                for (let ly = 0; ly < 4; ly++) {
                                    const bitIdx = lx + (ly << 2) + (lz << 4);
                                    if (bitIdx < 32) lo |= (1 << bitIdx);
                                    else hi |= (1 << (bitIdx - 32));
                                }
                            }
                        }
                        foundEmpty.orBlock(blockIdx, lo >>> 0, hi >>> 0);
                    }
                    continue;
                }

                // BLOCK_MIXED: per-voxel walk
                const s = dilatedSolid.masks.slot(blockIdx);
                const dLo = dilatedSolid.masks.lo[s];
                const dHi = dilatedSolid.masks.hi[s];

                let foundLo = 0;
                let foundHi = 0;

                for (let lz = 0; lz < 4; lz++) {
                    for (let lx = 0; lx < 4; lx++) {
                        const subCol = 1 << (lz * 4 + lx);
                        if (!(walking & subCol)) continue;

                        for (let ly = 0; ly < 4; ly++) {
                            const bitIdx = lx + (ly << 2) + (lz << 4);
                            const inHi = bitIdx >= 32;
                            const word = inHi ? dHi : dLo;
                            const bit = 1 << (inHi ? bitIdx - 32 : bitIdx);

                            if (word & bit) {
                                walking &= ~subCol;
                                break;
                            }

                            if (inHi) foundHi |= bit;
                            else foundLo |= bit;
                        }
                    }
                }

                if (foundLo || foundHi) {
                    foundEmpty.orBlock(blockIdx, foundLo >>> 0, foundHi >>> 0);
                }
            }
            walkBar.tick();
        }
    }
    walkBar.end();

    if (r > 0) dilatedSolid.clear();

    const dilatedFound = r > 0 ? await gpuDilate3(gpu, foundEmpty, r, 0) : foundEmpty;

    // grid is the original voxelization; not read after this OR. Pass
    // consumeA=true so sparseOrGrids mutates it in place rather than
    // cloning — saves a full SparseVoxelGrid clone right at the end of
    // the fill-floor phase.
    const combineBar = logger.bar('Combining', grid.types.length);
    const combined = sparseOrGrids(grid, dilatedFound, true, done => combineBar.update(done));
    combineBar.end();

    const buildBar = logger.bar('Building buffer', nbz);
    const result = combined.toBuffer(0, 0, 0, nbx, nby, nbz, false, done => buildBar.update(done));
    buildBar.end();

    return { buffer: result, gridBounds };
};

export { fillFloor };
