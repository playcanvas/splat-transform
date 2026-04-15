/**
 * Tests for fillFloor -- fills below the floor surface to block outdoor edges.
 *
 * Constructs small voxel scenes using BlockMaskBuffer, runs fillFloor,
 * and verifies that below-floor and no-floor columns are filled with solid.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { BlockMaskBuffer } from '../src/lib/voxel/block-mask-buffer.js';
import { SparseVoxelGrid } from '../src/lib/voxel/sparse-voxel-grid.js';
import { xyzToMorton, popcount } from '../src/lib/voxel/morton.js';
import { alignGridBounds } from '../src/lib/voxel/voxelize.js';
import { fillFloor } from '../src/lib/voxel/fill-floor.js';

const SOLID_LO = 0xFFFFFFFF >>> 0;
const SOLID_HI = 0xFFFFFFFF >>> 0;

function countSolidVoxels(acc) {
    let count = 0;
    const solid = acc.getSolidBlocks();
    count += solid.length * 64;
    const mixed = acc.getMixedBlocks();
    for (let i = 0; i < mixed.morton.length; i++) {
        count += popcount(mixed.masks[i * 2]) + popcount(mixed.masks[i * 2 + 1]);
    }
    return count;
}

function bufferToGrid(buffer, gridBounds, voxelResolution) {
    const nx = Math.round((gridBounds.max.x - gridBounds.min.x) / voxelResolution);
    const ny = Math.round((gridBounds.max.y - gridBounds.min.y) / voxelResolution);
    const nz = Math.round((gridBounds.max.z - gridBounds.min.z) / voxelResolution);
    return SparseVoxelGrid.fromBuffer(buffer, nx, ny, nz);
}

/**
 * Build a flat floor: a single layer of solid blocks at a given Y block level.
 * The floor spans the full XZ extent of the grid.
 */
function buildFloor(sizeBlocksXZ, sizeBlocksY, floorBlockY, voxelResolution) {
    const acc = new BlockMaskBuffer();
    for (let bz = 0; bz < sizeBlocksXZ; bz++) {
        for (let bx = 0; bx < sizeBlocksXZ; bx++) {
            acc.addBlock(xyzToMorton(bx, floorBlockY, bz), SOLID_LO, SOLID_HI);
        }
    }
    const worldX = sizeBlocksXZ * 4 * voxelResolution;
    const worldY = sizeBlocksY * 4 * voxelResolution;
    const worldZ = sizeBlocksXZ * 4 * voxelResolution;
    const gridBounds = alignGridBounds(0, 0, 0, worldX, worldY, worldZ, voxelResolution);
    return { acc, gridBounds };
}

/**
 * Build a floor with a gap: one block in the middle is missing.
 */
function buildFloorWithGap(sizeBlocksXZ, sizeBlocksY, floorBlockY, gapBx, gapBz, voxelResolution) {
    const acc = new BlockMaskBuffer();
    for (let bz = 0; bz < sizeBlocksXZ; bz++) {
        for (let bx = 0; bx < sizeBlocksXZ; bx++) {
            if (bx === gapBx && bz === gapBz) continue;
            acc.addBlock(xyzToMorton(bx, floorBlockY, bz), SOLID_LO, SOLID_HI);
        }
    }
    const worldX = sizeBlocksXZ * 4 * voxelResolution;
    const worldY = sizeBlocksY * 4 * voxelResolution;
    const worldZ = sizeBlocksXZ * 4 * voxelResolution;
    const gridBounds = alignGridBounds(0, 0, 0, worldX, worldY, worldZ, voxelResolution);
    return { acc, gridBounds };
}

describe('fillFloor', function () {
    const voxelResolution = 0.25;
    const dilation = 1.0;

    describe('flat floor', function () {
        it('should fill below the floor with solid', function () {
            const sizeXZ = 4;
            const sizeY = 6;
            const floorBlockY = 2;
            const { acc, gridBounds } = buildFloor(sizeXZ, sizeY, floorBlockY, voxelResolution);

            const inputCount = countSolidVoxels(acc);
            const result = fillFloor(acc, gridBounds, voxelResolution, dilation);
            const resultCount = countSolidVoxels(result.buffer);

            assert.ok(resultCount > inputCount,
                `Should have more solid voxels after fill (input: ${inputCount}, result: ${resultCount})`);
        });

        it('should make all voxels below the floor solid', function () {
            const sizeXZ = 4;
            const sizeY = 6;
            const floorBlockY = 2;
            const { acc, gridBounds } = buildFloor(sizeXZ, sizeY, floorBlockY, voxelResolution);

            const result = fillFloor(acc, gridBounds, voxelResolution, dilation);
            const grid = bufferToGrid(result.buffer, result.gridBounds, voxelResolution);

            const nx = grid.nx;
            const nz = grid.nz;
            const floorVoxelY = floorBlockY * 4;

            for (let iz = 0; iz < nz; iz++) {
                for (let ix = 0; ix < nx; ix++) {
                    for (let iy = 0; iy < floorVoxelY; iy++) {
                        assert.strictEqual(grid.getVoxel(ix, iy, iz), 1,
                            `Voxel (${ix}, ${iy}, ${iz}) below floor should be solid`);
                    }
                }
            }
        });

        it('should not fill above the floor', function () {
            const sizeXZ = 4;
            const sizeY = 6;
            const floorBlockY = 2;
            const { acc, gridBounds } = buildFloor(sizeXZ, sizeY, floorBlockY, voxelResolution);

            const result = fillFloor(acc, gridBounds, voxelResolution, dilation);
            const grid = bufferToGrid(result.buffer, result.gridBounds, voxelResolution);

            const ny = grid.ny;
            const floorTopVoxelY = (floorBlockY + 1) * 4;

            let solidAboveFloor = 0;
            for (let iz = 0; iz < grid.nz; iz++) {
                for (let ix = 0; ix < grid.nx; ix++) {
                    for (let iy = floorTopVoxelY; iy < ny; iy++) {
                        if (grid.getVoxel(ix, iy, iz)) solidAboveFloor++;
                    }
                }
            }

            assert.strictEqual(solidAboveFloor, 0,
                'No voxels above the floor layer should be solid');
        });
    });

    describe('no-floor columns', function () {
        it('should fill columns with no floor entirely with solid', function () {
            const sizeXZ = 6;
            const sizeY = 4;
            const floorBlockY = 1;

            const acc = new BlockMaskBuffer();
            for (let bz = 0; bz < 3; bz++) {
                for (let bx = 0; bx < 3; bx++) {
                    acc.addBlock(xyzToMorton(bx, floorBlockY, bz), SOLID_LO, SOLID_HI);
                }
            }
            const worldX = sizeXZ * 4 * voxelResolution;
            const worldY = sizeY * 4 * voxelResolution;
            const gridBounds = alignGridBounds(0, 0, 0, worldX, worldY, worldX, voxelResolution);

            const result = fillFloor(acc, gridBounds, voxelResolution, dilation);
            const grid = bufferToGrid(result.buffer, result.gridBounds, voxelResolution);

            const emptyIx = 5 * 4;
            const emptyIz = 5 * 4;

            let solidCount = 0;
            for (let iy = 0; iy < grid.ny; iy++) {
                if (grid.getVoxel(emptyIx, iy, emptyIz)) solidCount++;
            }

            assert.strictEqual(solidCount, grid.ny,
                `No-floor column at (${emptyIx}, *, ${emptyIz}) should be entirely solid (got ${solidCount}/${grid.ny})`);
        });
    });

    describe('floor with gap', function () {
        it('should bridge small gaps with dilation', function () {
            const sizeXZ = 6;
            const sizeY = 6;
            const floorBlockY = 2;
            const gapBx = 3;
            const gapBz = 3;
            const { acc, gridBounds } = buildFloorWithGap(sizeXZ, sizeY, floorBlockY, gapBx, gapBz, voxelResolution);

            const largeDilation = 2.0;
            const result = fillFloor(acc, gridBounds, voxelResolution, largeDilation);
            const grid = bufferToGrid(result.buffer, result.gridBounds, voxelResolution);

            const gapIx = gapBx * 4;
            const gapIz = gapBz * 4;
            const floorVoxelY = floorBlockY * 4;

            let solidBelowGap = 0;
            for (let iy = 0; iy < floorVoxelY; iy++) {
                if (grid.getVoxel(gapIx, iy, gapIz)) solidBelowGap++;
            }

            assert.ok(solidBelowGap > 0,
                'Dilation should bridge the gap, so below-gap voxels should be filled');
        });
    });

    describe('empty buffer', function () {
        it('should return the same buffer reference when input is empty', function () {
            const acc = new BlockMaskBuffer();
            assert.strictEqual(acc.count, 0);
            const gridBounds = alignGridBounds(0, 0, 0, 1, 1, 1, voxelResolution);

            const result = fillFloor(acc, gridBounds, voxelResolution, dilation);
            assert.strictEqual(result.buffer, acc, 'Empty input should return same buffer reference');
        });
    });

    describe('parameter validation', function () {
        it('should throw for zero voxel resolution', function () {
            const acc = new BlockMaskBuffer();
            acc.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);
            const gridBounds = alignGridBounds(0, 0, 0, 1, 1, 1, 0.25);
            assert.throws(
                () => fillFloor(acc, gridBounds, 0, dilation),
                /voxelResolution must be finite and > 0/
            );
        });

        it('should throw for zero dilation', function () {
            const acc = new BlockMaskBuffer();
            acc.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);
            const gridBounds = alignGridBounds(0, 0, 0, 1, 1, 1, 0.25);
            assert.throws(
                () => fillFloor(acc, gridBounds, 0.25, 0),
                /dilation must be finite and > 0/
            );
        });

        it('should throw for negative dilation', function () {
            const acc = new BlockMaskBuffer();
            acc.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);
            const gridBounds = alignGridBounds(0, 0, 0, 1, 1, 1, 0.25);
            assert.throws(
                () => fillFloor(acc, gridBounds, 0.25, -1),
                /dilation must be finite and > 0/
            );
        });
    });

    describe('gridBounds preserved', function () {
        it('should return the same gridBounds as input', function () {
            const sizeXZ = 4;
            const sizeY = 4;
            const { acc, gridBounds } = buildFloor(sizeXZ, sizeY, 1, voxelResolution);

            const result = fillFloor(acc, gridBounds, voxelResolution, dilation);

            assert.strictEqual(result.gridBounds, gridBounds,
                'gridBounds should be the same object reference');
        });
    });
});
