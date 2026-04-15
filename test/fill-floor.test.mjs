/**
 * Tests for fillFloor -- fills gap/edge columns to block outdoor scene edges.
 *
 * The new implementation only fills columns where the original floor is absent
 * but neighbors within the dilation radius have a floor (gap/edge columns).
 * Columns that already have a floor are left unmodified.
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

    describe('complete floor (no gaps)', function () {
        it('should not modify a floor that has no gaps', function () {
            const sizeXZ = 4;
            const sizeY = 6;
            const floorBlockY = 2;
            const { acc, gridBounds } = buildFloor(sizeXZ, sizeY, floorBlockY, voxelResolution);

            const inputCount = countSolidVoxels(acc);
            const result = fillFloor(acc, gridBounds, voxelResolution, dilation);
            const resultCount = countSolidVoxels(result.buffer);

            assert.strictEqual(resultCount, inputCount,
                'A complete floor should not be modified');
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

    describe('edge columns within dilation radius', function () {
        it('should fill no-floor columns adjacent to the floor edge', function () {
            // Floor covers blocks (0..2, 1, 0..2), grid extends to block 5.
            // Block 3 is 1 block past the floor edge. With dilation=1.0 and
            // voxelResolution=0.25, halfExtent = ceil(4) = 4 voxels = 1 block.
            // The no-floor column at block 3 voxel 12 is within dilation of
            // the floor edge at voxel 11.
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

            const largeDilation = 2.0;
            const result = fillFloor(acc, gridBounds, voxelResolution, largeDilation);
            const grid = bufferToGrid(result.buffer, result.gridBounds, voxelResolution);

            // Check a column just past the floor edge, within dilation reach.
            // Floor edge is at voxel x=11. halfExtent = ceil(2.0/0.25) = 8.
            // Voxel x=12 (block 3) is 1 past the edge, well within radius.
            const edgeIx = 3 * 4;
            const edgeIz = 1 * 4;
            const floorVoxelY = floorBlockY * 4;

            let solidCount = 0;
            for (let iy = 0; iy < floorVoxelY; iy++) {
                if (grid.getVoxel(edgeIx, iy, edgeIz)) solidCount++;
            }

            assert.ok(solidCount > 0,
                `Edge column at (${edgeIx}, *, ${edgeIz}) within dilation radius should be filled below floor (got ${solidCount}/${floorVoxelY})`);
        });

        it('should not fill columns beyond dilation radius', function () {
            const sizeXZ = 10;
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

            // halfExtent = ceil(1.0 / 0.25) = 4. Floor edge at voxel 11.
            // Column at voxel 20 (block 5) is 9 voxels past the edge, beyond radius.
            const farIx = 5 * 4;
            const farIz = 5 * 4;

            let solidCount = 0;
            for (let iy = 0; iy < grid.ny; iy++) {
                if (grid.getVoxel(farIx, iy, farIz)) solidCount++;
            }

            assert.strictEqual(solidCount, 0,
                `Column at (${farIx}, *, ${farIz}) beyond dilation radius should not be filled`);
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

        it('should add voxels only at gap columns, not below existing floor', function () {
            const sizeXZ = 6;
            const sizeY = 6;
            const floorBlockY = 2;
            const gapBx = 3;
            const gapBz = 3;
            const { acc, gridBounds } = buildFloorWithGap(sizeXZ, sizeY, floorBlockY, gapBx, gapBz, voxelResolution);

            const largeDilation = 2.0;
            const result = fillFloor(acc, gridBounds, voxelResolution, largeDilation);
            const grid = bufferToGrid(result.buffer, result.gridBounds, voxelResolution);

            // A column that already has floor should not have fill below it
            const floorIx = 0;
            const floorIz = 0;
            const floorVoxelY = floorBlockY * 4;

            let belowFloor = 0;
            for (let iy = 0; iy < floorVoxelY; iy++) {
                if (grid.getVoxel(floorIx, iy, floorIz)) belowFloor++;
            }

            assert.strictEqual(belowFloor, 0,
                'Floor-having columns should not have fill below them');
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
