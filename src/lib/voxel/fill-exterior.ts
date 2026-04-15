import { Vec3 } from 'playcanvas';

import { BlockMaskBuffer } from './block-mask-buffer';
import { sparseDilate3 } from './dilation';
import { twoLevelBFS } from './flood-fill';
import { sparseOrGrids } from './grid-ops';
import type { Bounds } from '../data-table';
import {
    BLOCK_MIXED,
    BLOCK_SOLID,
    FACE_MASKS_HI,
    FACE_MASKS_LO,
    SOLID_HI,
    SOLID_LO,
    SparseVoxelGrid
} from './sparse-voxel-grid';
import { logger } from '../utils';

type NavSeed = {
    x: number;
    y: number;
    z: number;
};

type NavSimplifyResult = {
    buffer: BlockMaskBuffer;
    gridBounds: Bounds;
};

const fillExterior = (
    buffer: BlockMaskBuffer,
    gridBounds: Bounds,
    voxelResolution: number,
    dilation: number,
    seed: NavSeed
): NavSimplifyResult => {
    if (!Number.isFinite(voxelResolution) || voxelResolution <= 0) {
        throw new Error(`fillExterior: voxelResolution must be finite and > 0, got ${voxelResolution}`);
    }
    if (!Number.isFinite(dilation) || dilation <= 0) {
        throw new Error(`fillExterior: dilation must be finite and > 0, got ${dilation}`);
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

    logger.progress.begin(4);
    let progressComplete = false;

    try {
        const gridOriginal = SparseVoxelGrid.fromBuffer(buffer, nx, ny, nz);
        logger.progress.step();

        const dilated = sparseDilate3(gridOriginal, halfExtent, halfExtent);

        const bStride = nbx * nby;
        const blockSeeds: number[] = [];
        const faceVoxelSeeds: { ix: number; iy: number; iz: number }[] = [];

        const seedBoundaryBlock = (blockIdx: number, bx: number, by: number, bz: number, face: number): void => {
            const bt = dilated.blockType[blockIdx];
            if (bt === BLOCK_SOLID) return;
            if (bt === BLOCK_MIXED) {
                const ms = dilated.masks.slot(blockIdx);
                const faceLo = FACE_MASKS_LO[face];
                const faceHi = FACE_MASKS_HI[face];
                let freeLo = (faceLo & ~dilated.masks.lo[ms]) >>> 0;
                let freeHi = (faceHi & ~dilated.masks.hi[ms]) >>> 0;
                if (freeLo === 0 && freeHi === 0) return;
                const baseIx = bx << 2;
                const baseIy = by << 2;
                const baseIz = bz << 2;
                while (freeLo) {
                    const bp = 31 - Math.clz32(freeLo & -freeLo);
                    faceVoxelSeeds.push({ ix: baseIx + (bp & 3), iy: baseIy + ((bp >> 2) & 3), iz: baseIz + (bp >> 4) });
                    freeLo &= freeLo - 1;
                }
                while (freeHi) {
                    const bp = 31 - Math.clz32(freeHi & -freeHi);
                    const bi = bp + 32;
                    faceVoxelSeeds.push({ ix: baseIx + (bi & 3), iy: baseIy + ((bi >> 2) & 3), iz: baseIz + (bi >> 4) });
                    freeHi &= freeHi - 1;
                }
                return;
            }
            blockSeeds.push(blockIdx);
        };

        for (let bz = 0; bz < nbz; bz++) {
            for (let by = 0; by < nby; by++) {
                seedBoundaryBlock(by * nbx + bz * bStride, 0, by, bz, 0);
            }
        }

        for (let bz = 0; bz < nbz; bz++) {
            for (let by = 0; by < nby; by++) {
                seedBoundaryBlock((nbx - 1) + by * nbx + bz * bStride, nbx - 1, by, bz, 1);
            }
        }

        for (let bz = 0; bz < nbz; bz++) {
            for (let bx = 0; bx < nbx; bx++) {
                seedBoundaryBlock(bx + bz * bStride, bx, 0, bz, 2);
            }
        }

        for (let bz = 0; bz < nbz; bz++) {
            for (let bx = 0; bx < nbx; bx++) {
                seedBoundaryBlock(bx + (nby - 1) * nbx + bz * bStride, bx, nby - 1, bz, 3);
            }
        }

        for (let by = 0; by < nby; by++) {
            for (let bx = 0; bx < nbx; bx++) {
                seedBoundaryBlock(bx + by * nbx, bx, by, 0, 4);
            }
        }

        for (let by = 0; by < nby; by++) {
            for (let bx = 0; bx < nbx; bx++) {
                seedBoundaryBlock(bx + by * nbx + (nbz - 1) * bStride, bx, by, nbz - 1, 5);
            }
        }

        const visited = twoLevelBFS(dilated, blockSeeds, faceVoxelSeeds, nx, ny, nz);

        const seedIx = Math.floor((seed.x - gridBounds.min.x) / voxelResolution);
        const seedIy = Math.floor((seed.y - gridBounds.min.y) / voxelResolution);
        const seedIz = Math.floor((seed.z - gridBounds.min.z) / voxelResolution);

        if (seedIx >= 0 && seedIx < nx && seedIy >= 0 && seedIy < ny && seedIz >= 0 && seedIz < nz) {
            if (visited.getVoxel(seedIx, seedIy, seedIz)) {
                logger.log('fillExterior: seed reachable from outside, skipping');
                logger.progress.cancel();
                progressComplete = true;
                return { buffer, gridBounds };
            }
        } else {
            logger.log('fillExterior: seed outside grid bounds, skipping exterior fill');
            logger.progress.cancel();
            progressComplete = true;
            return { buffer, gridBounds };
        }

        logger.progress.step();

        const dilatedVisited = sparseDilate3(visited, halfExtent, halfExtent);

        const combined = sparseOrGrids(gridOriginal, dilatedVisited);
        logger.progress.step();

        let minIx = nx, minIy = ny, minIz = nz;
        let maxIx = 0, maxIy = 0, maxIz = 0;

        for (let bz = 0; bz < nbz; bz++) {
            for (let by = 0; by < nby; by++) {
                for (let bx = 0; bx < nbx; bx++) {
                    const blockIdx = bx + by * nbx + bz * combined.bStride;
                    const bt = combined.blockType[blockIdx];
                    if (bt === BLOCK_SOLID) continue;
                    if (bt === BLOCK_MIXED) {
                        const cs = combined.masks.slot(blockIdx);
                        if (combined.masks.lo[cs] === SOLID_LO && combined.masks.hi[cs] === SOLID_HI) continue;
                    }
                    const baseX = bx << 2;
                    const baseY = by << 2;
                    const baseZ = bz << 2;
                    if (baseX < minIx) minIx = baseX;
                    if (baseX + 3 > maxIx) maxIx = baseX + 3;
                    if (baseY < minIy) minIy = baseY;
                    if (baseY + 3 > maxIy) maxIy = baseY + 3;
                    if (baseZ < minIz) minIz = baseZ;
                    if (baseZ + 3 > maxIz) maxIz = baseZ + 3;
                }
            }
        }

        if (minIx > maxIx) {
            logger.warn('fillExterior: no navigable cells remain, returning empty result');
            logger.progress.step();
            progressComplete = true;
            return {
                buffer: new BlockMaskBuffer(),
                gridBounds: { min: gridBounds.min.clone(), max: gridBounds.min.clone() }
            };
        }

        const MARGIN = 1;
        const cropMinBx = Math.max(0, (minIx >> 2) - MARGIN);
        const cropMinBy = Math.max(0, (minIy >> 2) - MARGIN);
        const cropMinBz = Math.max(0, (minIz >> 2) - MARGIN);
        const cropMaxBx = Math.min(nbx, (maxIx >> 2) + 1 + MARGIN);
        const cropMaxBy = Math.min(nby, (maxIy >> 2) + 1 + MARGIN);
        const cropMaxBz = Math.min(nbz, (maxIz >> 2) + 1 + MARGIN);

        const blockSize = 4 * voxelResolution;
        const croppedMin = new Vec3(
            gridBounds.min.x + cropMinBx * blockSize,
            gridBounds.min.y + cropMinBy * blockSize,
            gridBounds.min.z + cropMinBz * blockSize
        );
        const croppedBounds: Bounds = {
            min: croppedMin,
            max: new Vec3(
                croppedMin.x + (cropMaxBx - cropMinBx) * blockSize,
                croppedMin.y + (cropMaxBy - cropMinBy) * blockSize,
                croppedMin.z + (cropMaxBz - cropMinBz) * blockSize
            )
        };

        logger.progress.step();
        progressComplete = true;

        return {
            buffer: combined.toBuffer(
                cropMinBx, cropMinBy, cropMinBz,
                cropMaxBx, cropMaxBy, cropMaxBz
            ),
            gridBounds: croppedBounds
        };

    } finally {
        if (!progressComplete) {
            logger.progress.cancel();
        }
    }
};

export { fillExterior };
export type { NavSeed, NavSimplifyResult };
