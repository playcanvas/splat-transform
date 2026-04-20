/**
 * Tests for fillFloor -- fills each voxel column upward from the bottom
 * until hitting an existing solid voxel or the top of the grid.
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

function makeGridBounds(nbx, nby, nbz, voxelResolution) {
    const worldX = nbx * 4 * voxelResolution;
    const worldY = nby * 4 * voxelResolution;
    const worldZ = nbz * 4 * voxelResolution;
    return alignGridBounds(0, 0, 0, worldX, worldY, worldZ, voxelResolution);
}

describe('fillFloor', function () {
    const voxelResolution = 0.25;

    describe('empty column fill', function () {
        it('should fill entirely empty columns solid from bottom to top', function () {
            const nbx = 2, nby = 4, nbz = 2;
            const gridBounds = makeGridBounds(nbx, nby, nbz, voxelResolution);

            // Only one block at (0, 2, 0) is solid; all other columns are empty.
            const acc = new BlockMaskBuffer();
            acc.addBlock(xyzToMorton(0, 2, 0), SOLID_LO, SOLID_HI);

            const result = fillFloor(acc, gridBounds, voxelResolution);
            const grid = bufferToGrid(result.buffer, result.gridBounds, voxelResolution);

            // Column at (4, *, 4) has no solid -- should be filled entirely.
            for (let iy = 0; iy < grid.ny; iy++) {
                assert.strictEqual(grid.getVoxel(4, iy, 4), 1,
                    `Empty column voxel (4, ${iy}, 4) should be solid`);
            }
        });
    });

    describe('fill below solid', function () {
        it('should fill below the floor block and not above', function () {
            const nbx = 2, nby = 4, nbz = 2;
            const floorBlockY = 2;
            const gridBounds = makeGridBounds(nbx, nby, nbz, voxelResolution);

            // Solid floor at y-block 2 across all XZ
            const acc = new BlockMaskBuffer();
            for (let bz = 0; bz < nbz; bz++) {
                for (let bx = 0; bx < nbx; bx++) {
                    acc.addBlock(xyzToMorton(bx, floorBlockY, bz), SOLID_LO, SOLID_HI);
                }
            }

            const result = fillFloor(acc, gridBounds, voxelResolution);
            const grid = bufferToGrid(result.buffer, result.gridBounds, voxelResolution);

            const floorVoxelY = floorBlockY * 4;
            const floorTopVoxelY = (floorBlockY + 1) * 4;

            for (let iz = 0; iz < grid.nz; iz++) {
                for (let ix = 0; ix < grid.nx; ix++) {
                    // Below floor should be filled solid
                    for (let iy = 0; iy < floorVoxelY; iy++) {
                        assert.strictEqual(grid.getVoxel(ix, iy, iz), 1,
                            `Voxel (${ix}, ${iy}, ${iz}) below floor should be solid`);
                    }
                    // Floor itself should remain solid
                    for (let iy = floorVoxelY; iy < floorTopVoxelY; iy++) {
                        assert.strictEqual(grid.getVoxel(ix, iy, iz), 1,
                            `Floor voxel (${ix}, ${iy}, ${iz}) should remain solid`);
                    }
                    // Above floor should remain empty
                    for (let iy = floorTopVoxelY; iy < grid.ny; iy++) {
                        assert.strictEqual(grid.getVoxel(ix, iy, iz), 0,
                            `Voxel (${ix}, ${iy}, ${iz}) above floor should be empty`);
                    }
                }
            }
        });

        it('should stop at the first solid voxel per sub-column in a mixed block', function () {
            const nbx = 1, nby = 3, nbz = 1;
            const gridBounds = makeGridBounds(nbx, nby, nbz, voxelResolution);

            // Place a single solid voxel at (0, 5, 0) in block (0, 1, 0).
            // bitIdx = lx + (ly << 2) + (lz << 4) = 0 + (1 << 2) + 0 = 4
            const acc = new BlockMaskBuffer();
            const lo = (1 << 4) >>> 0;
            acc.addBlock(xyzToMorton(0, 1, 0), lo, 0);

            const result = fillFloor(acc, gridBounds, voxelResolution);
            const grid = bufferToGrid(result.buffer, result.gridBounds, voxelResolution);

            // Sub-column (lx=0, lz=0): solid at iy=5, should fill iy=0..4
            for (let iy = 0; iy < 5; iy++) {
                assert.strictEqual(grid.getVoxel(0, iy, 0), 1,
                    `Voxel (0, ${iy}, 0) below the solid at iy=5 should be filled`);
            }
            assert.strictEqual(grid.getVoxel(0, 5, 0), 1, 'The original solid voxel should remain');
            // Above the solid should be empty
            for (let iy = 6; iy < grid.ny; iy++) {
                assert.strictEqual(grid.getVoxel(0, iy, 0), 0,
                    `Voxel (0, ${iy}, 0) above the solid at iy=5 should be empty`);
            }

            // Sub-column (lx=1, lz=0) has no solid; should be filled entirely
            for (let iy = 0; iy < grid.ny; iy++) {
                assert.strictEqual(grid.getVoxel(1, iy, 0), 1,
                    `Voxel (1, ${iy}, 0) in a no-solid sub-column should be solid`);
            }
        });
    });

    describe('already solid from bottom', function () {
        it('should not modify a fully solid grid', function () {
            const nbx = 2, nby = 2, nbz = 2;
            const gridBounds = makeGridBounds(nbx, nby, nbz, voxelResolution);

            const acc = new BlockMaskBuffer();
            for (let bz = 0; bz < nbz; bz++) {
                for (let bx = 0; bx < nbx; bx++) {
                    for (let by = 0; by < nby; by++) {
                        acc.addBlock(xyzToMorton(bx, by, bz), SOLID_LO, SOLID_HI);
                    }
                }
            }

            const inputCount = countSolidVoxels(acc);
            const result = fillFloor(acc, gridBounds, voxelResolution);
            const resultCount = countSolidVoxels(result.buffer);

            assert.strictEqual(resultCount, inputCount,
                'A fully solid grid should not be modified');
        });

        it('should not add voxels when the bottom block is already solid', function () {
            const nbx = 1, nby = 3, nbz = 1;
            const gridBounds = makeGridBounds(nbx, nby, nbz, voxelResolution);

            // Solid block at bottom (by=0), empty above
            const acc = new BlockMaskBuffer();
            acc.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);

            const inputCount = countSolidVoxels(acc);
            const result = fillFloor(acc, gridBounds, voxelResolution);
            const resultCount = countSolidVoxels(result.buffer);

            assert.strictEqual(resultCount, inputCount,
                'Column with solid at bottom should not gain any voxels');
        });
    });

    describe('empty buffer', function () {
        it('should return the same buffer reference when input is empty', function () {
            const acc = new BlockMaskBuffer();
            assert.strictEqual(acc.count, 0);
            const gridBounds = makeGridBounds(2, 2, 2, voxelResolution);

            const result = fillFloor(acc, gridBounds, voxelResolution);
            assert.strictEqual(result.buffer, acc, 'Empty input should return same buffer reference');
        });
    });

    describe('parameter validation', function () {
        it('should throw for zero voxel resolution', function () {
            const acc = new BlockMaskBuffer();
            acc.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);
            const gridBounds = makeGridBounds(1, 1, 1, 0.25);
            assert.throws(
                () => fillFloor(acc, gridBounds, 0),
                /voxelResolution must be finite and > 0/
            );
        });

        it('should throw for negative voxel resolution', function () {
            const acc = new BlockMaskBuffer();
            acc.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);
            const gridBounds = makeGridBounds(1, 1, 1, 0.25);
            assert.throws(
                () => fillFloor(acc, gridBounds, -1),
                /voxelResolution must be finite and > 0/
            );
        });
    });

    describe('gridBounds preserved', function () {
        it('should return the same gridBounds as input', function () {
            const nbx = 2, nby = 2, nbz = 2;
            const gridBounds = makeGridBounds(nbx, nby, nbz, voxelResolution);

            const acc = new BlockMaskBuffer();
            acc.addBlock(xyzToMorton(0, 1, 0), SOLID_LO, SOLID_HI);

            const result = fillFloor(acc, gridBounds, voxelResolution);

            assert.strictEqual(result.gridBounds, gridBounds,
                'gridBounds should be the same object reference');
        });
    });

    describe('multiple solids in column', function () {
        it('should stop at the first (lowest) solid and not fill between solids', function () {
            const nbx = 1, nby = 4, nbz = 1;
            const gridBounds = makeGridBounds(nbx, nby, nbz, voxelResolution);

            // Solid blocks at by=1 and by=3, empty at by=0 and by=2
            const acc = new BlockMaskBuffer();
            acc.addBlock(xyzToMorton(0, 1, 0), SOLID_LO, SOLID_HI);
            acc.addBlock(xyzToMorton(0, 3, 0), SOLID_LO, SOLID_HI);

            const result = fillFloor(acc, gridBounds, voxelResolution);
            const grid = bufferToGrid(result.buffer, result.gridBounds, voxelResolution);

            // by=0 should be filled (below the first solid at by=1)
            for (let iy = 0; iy < 4; iy++) {
                assert.strictEqual(grid.getVoxel(0, iy, 0), 1,
                    `Voxel (0, ${iy}, 0) below first solid should be filled`);
            }
            // by=1 (first solid) remains solid
            for (let iy = 4; iy < 8; iy++) {
                assert.strictEqual(grid.getVoxel(0, iy, 0), 1,
                    `Voxel (0, ${iy}, 0) in first solid block should remain`);
            }
            // by=2 should remain empty (above first solid, fill stopped)
            for (let iy = 8; iy < 12; iy++) {
                assert.strictEqual(grid.getVoxel(0, iy, 0), 0,
                    `Voxel (0, ${iy}, 0) between solids should remain empty`);
            }
        });
    });
});
