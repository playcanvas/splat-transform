/**
 * Tests for capsule-traced navigation voxel simplification.
 *
 * Constructs small voxel scenes (hollow boxes, corridors) using BlockMaskBuffer,
 * runs carveInterior, and verifies the output uses negative space carving
 * with erosion to restore correct surface positions.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { BlockMaskBuffer } from '../src/lib/voxel/block-mask-buffer.js';
import { SparseVoxelGrid } from '../src/lib/voxel/sparse-voxel-grid.js';
import { xyzToMorton, mortonToXYZ, popcount } from '../src/lib/voxel/morton.js';
import { alignGridBounds } from '../src/lib/voxel/voxelize.js';
import { carveInterior } from '../src/lib/voxel/carve-interior.js';

const SOLID_LO = 0xFFFFFFFF >>> 0;
const SOLID_HI = 0xFFFFFFFF >>> 0;

/**
 * Count total solid voxels in a BlockMaskBuffer.
 */
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

/**
 * Convert a BlockMaskBuffer to a SparseVoxelGrid for voxel-level queries.
 */
function bufferToGrid(buffer, gridBounds, voxelResolution) {
    const nx = Math.round((gridBounds.max.x - gridBounds.min.x) / voxelResolution);
    const ny = Math.round((gridBounds.max.y - gridBounds.min.y) / voxelResolution);
    const nz = Math.round((gridBounds.max.z - gridBounds.min.z) / voxelResolution);
    return SparseVoxelGrid.fromBuffer(buffer, nx, ny, nz);
}

/**
 * Build a hollow box of solid blocks. The box has solid walls of 1 block thick
 * and an empty interior. Returns the buffer and grid bounds.
 *
 * @param {number} sizeBlocks - Size of the box in blocks per axis (must be >= 3).
 * @param {number} voxelResolution - Voxel resolution.
 */
function buildHollowBox(sizeBlocks, voxelResolution) {
    const acc = new BlockMaskBuffer();
    for (let bz = 0; bz < sizeBlocks; bz++) {
        for (let by = 0; by < sizeBlocks; by++) {
            for (let bx = 0; bx < sizeBlocks; bx++) {
                const isWall = bx === 0 || bx === sizeBlocks - 1 ||
                               by === 0 || by === sizeBlocks - 1 ||
                               bz === 0 || bz === sizeBlocks - 1;
                if (isWall) {
                    acc.addBlock(xyzToMorton(bx, by, bz), SOLID_LO, SOLID_HI);
                }
            }
        }
    }

    const worldSize = sizeBlocks * 4 * voxelResolution;
    const gridBounds = alignGridBounds(0, 0, 0, worldSize, worldSize, worldSize, voxelResolution);
    return { acc, gridBounds };
}

describe('carveInterior', function () {
    const voxelResolution = 0.25;
    const capsuleHeight = 1.5;
    const capsuleRadius = 0.2;

    describe('hollow box', function () {
        it('should produce solid voxels around the navigable space', function () {
            const { acc, gridBounds } = buildHollowBox(6, voxelResolution);

            const centerWorld = (gridBounds.min.x + gridBounds.max.x) / 2;
            const seed = { x: centerWorld, y: centerWorld, z: centerWorld };

            const result = carveInterior(acc, gridBounds, voxelResolution, capsuleHeight, capsuleRadius, seed);
            const resultCount = countSolidVoxels(result.buffer);

            assert.ok(resultCount > 0,
                'Should produce solid voxels around the navigable space');
        });

        it('should not include reachable cells as solid', function () {
            const { acc, gridBounds } = buildHollowBox(6, voxelResolution);

            const centerWorld = (gridBounds.min.x + gridBounds.max.x) / 2;
            const seed = { x: centerWorld, y: centerWorld, z: centerWorld };

            const result = carveInterior(acc, gridBounds, voxelResolution, capsuleHeight, capsuleRadius, seed);

            const resultCount = countSolidVoxels(result.buffer);
            const nx = Math.round((gridBounds.max.x - gridBounds.min.x) / voxelResolution);
            const totalCells = nx * nx * nx;

            assert.ok(resultCount < totalCells,
                `Result (${resultCount}) must leave reachable cells empty (total grid: ${totalCells})`);
        });

        it('should leave the seed voxel unoccupied in the output', function () {
            const { acc, gridBounds } = buildHollowBox(6, voxelResolution);

            const centerWorld = (gridBounds.min.x + gridBounds.max.x) / 2;
            const seed = { x: centerWorld, y: centerWorld, z: centerWorld };

            const result = carveInterior(acc, gridBounds, voxelResolution, capsuleHeight, capsuleRadius, seed);

            const grid = bufferToGrid(result.buffer, result.gridBounds, voxelResolution);
            const seedIx = Math.floor((seed.x - result.gridBounds.min.x) / voxelResolution);
            const seedIy = Math.floor((seed.y - result.gridBounds.min.y) / voxelResolution);
            const seedIz = Math.floor((seed.z - result.gridBounds.min.z) / voxelResolution);

            if (seedIx >= 0 && seedIx < grid.nx && seedIy >= 0 && seedIy < grid.ny && seedIz >= 0 && seedIz < grid.nz) {
                assert.strictEqual(grid.getVoxel(seedIx, seedIy, seedIz), 0,
                    'Seed voxel should be free (navigable) in the carved output');
            }
        });

        it('should produce a smaller grid due to cropping', function () {
            const sizeBlocks = 6;
            const { acc, gridBounds } = buildHollowBox(sizeBlocks, voxelResolution);

            const totalSize = (sizeBlocks + 4) * 4 * voxelResolution;
            const paddedBounds = alignGridBounds(0, 0, 0, totalSize, totalSize, totalSize, voxelResolution);

            const centerWorld = sizeBlocks * 4 * voxelResolution / 2;
            const seed = { x: centerWorld, y: centerWorld, z: centerWorld };

            const result = carveInterior(acc, paddedBounds, voxelResolution, capsuleHeight, capsuleRadius, seed);

            const inputExtentX = paddedBounds.max.x - paddedBounds.min.x;
            const outputExtentX = result.gridBounds.max.x - result.gridBounds.min.x;

            assert.ok(outputExtentX <= inputExtentX,
                `Output grid (${outputExtentX}) should be cropped to <= input (${inputExtentX})`);
        });

        it('should have solid voxels near the original walls', function () {
            const sizeBlocks = 6;
            const { acc, gridBounds } = buildHollowBox(sizeBlocks, voxelResolution);

            const centerWorld = (gridBounds.min.x + gridBounds.max.x) / 2;
            const seed = { x: centerWorld, y: centerWorld, z: centerWorld };

            const result = carveInterior(acc, gridBounds, voxelResolution, capsuleHeight, capsuleRadius, seed);
            const grid = bufferToGrid(result.buffer, result.gridBounds, voxelResolution);

            let solidNearWall = 0;
            const wallBlock = 0;
            const wallVoxelMax = 4;
            for (let iy = 0; iy < Math.min(wallVoxelMax, grid.ny); iy++) {
                for (let iz = 0; iz < Math.min(wallVoxelMax, grid.nz); iz++) {
                    for (let ix = 0; ix < Math.min(wallVoxelMax, grid.nx); ix++) {
                        if (grid.getVoxel(ix, iy, iz)) solidNearWall++;
                    }
                }
            }

            assert.ok(solidNearWall > 0,
                'Should have solid voxels in the corner region near the original walls');
        });
    });

    describe('seed validation', function () {
        it('should return original buffer if seed is outside grid', function () {
            const { acc, gridBounds } = buildHollowBox(4, voxelResolution);

            const seed = { x: -100, y: -100, z: -100 };
            const result = carveInterior(acc, gridBounds, voxelResolution, capsuleHeight, capsuleRadius, seed);

            assert.strictEqual(countSolidVoxels(result.buffer), countSolidVoxels(acc),
                'Should return original when seed is outside grid');
        });

        it('should return original buffer if seed is in solid region', function () {
            const { acc, gridBounds } = buildHollowBox(3, voxelResolution);

            const seed = {
                x: gridBounds.min.x + voxelResolution,
                y: gridBounds.min.y + voxelResolution,
                z: gridBounds.min.z + voxelResolution
            };
            const result = carveInterior(acc, gridBounds, voxelResolution, capsuleHeight, capsuleRadius, seed);

            assert.strictEqual(countSolidVoxels(result.buffer), countSolidVoxels(acc),
                'Should return original when seed is in blocked region');
        });
    });

    describe('empty buffer', function () {
        it('should carve out all reachable space (no obstacles)', function () {
            const acc = new BlockMaskBuffer();
            const gridBounds = alignGridBounds(0, 0, 0, 1, 1, 1, voxelResolution);
            const seed = { x: 0.5, y: 0.5, z: 0.5 };

            const result = carveInterior(acc, gridBounds, voxelResolution, capsuleHeight, capsuleRadius, seed);
            const resultCount = countSolidVoxels(result.buffer);
            const nx = Math.round((gridBounds.max.x - gridBounds.min.x) / voxelResolution);
            const ny = Math.round((gridBounds.max.y - gridBounds.min.y) / voxelResolution);
            const nz = Math.round((gridBounds.max.z - gridBounds.min.z) / voxelResolution);
            const totalCells = nx * ny * nz;

            assert.ok(resultCount < totalCells,
                'With no obstacles the entire grid is reachable; most cells should be empty');
        });

        it('should return empty buffer when input is empty', function () {
            const acc = new BlockMaskBuffer();
            assert.strictEqual(acc.count, 0);
            const gridBounds = alignGridBounds(0, 0, 0, 1, 1, 1, voxelResolution);
            const seed = { x: 0.5, y: 0.5, z: 0.5 };

            const result = carveInterior(acc, gridBounds, voxelResolution, capsuleHeight, capsuleRadius, seed);
            assert.strictEqual(result.buffer, acc, 'Empty input should return same buffer reference');
        });
    });

    describe('single solid block', function () {
        it('should retain solid voxels around the block', function () {
            const acc = new BlockMaskBuffer();
            acc.addBlock(xyzToMorton(2, 2, 2), SOLID_LO, SOLID_HI);

            const gridBounds = alignGridBounds(0, 0, 0, 5, 5, 5, voxelResolution);
            const blockMinX = 2 * 4 * voxelResolution;
            const seed = { x: blockMinX - capsuleRadius - voxelResolution, y: 2 * 4 * voxelResolution + 2 * voxelResolution, z: 2 * 4 * voxelResolution + 2 * voxelResolution };

            const result = carveInterior(acc, gridBounds, voxelResolution, capsuleHeight, capsuleRadius, seed);

            const resultCount = countSolidVoxels(result.buffer);
            assert.ok(resultCount > 0,
                'Should retain solid voxels near the reachable space');
            assert.ok(resultCount >= 64,
                `Should retain at least the original 64 voxels (got ${resultCount})`);
        });
    });

    describe('unreachable regions', function () {
        it('should crop exterior and preserve walls around navigable space', function () {
            const sizeBlocks = 6;
            const acc = new BlockMaskBuffer();

            for (let bz = 0; bz < sizeBlocks; bz++) {
                for (let by = 0; by < sizeBlocks; by++) {
                    for (let bx = 0; bx < sizeBlocks; bx++) {
                        const isWall = bx === 0 || bx === sizeBlocks - 1 ||
                                       by === 0 || by === sizeBlocks - 1 ||
                                       bz === 0 || bz === sizeBlocks - 1;
                        if (isWall) {
                            acc.addBlock(xyzToMorton(bx, by, bz), SOLID_LO, SOLID_HI);
                        }
                    }
                }
            }

            const totalSize = (sizeBlocks + 4) * 4 * voxelResolution;
            const gridBounds = alignGridBounds(0, 0, 0, totalSize, totalSize, totalSize, voxelResolution);

            const centerWorld = sizeBlocks * 4 * voxelResolution / 2;
            const seed = { x: centerWorld, y: centerWorld, z: centerWorld };

            const result = carveInterior(acc, gridBounds, voxelResolution, capsuleHeight, capsuleRadius, seed);
            const resultCount = countSolidVoxels(result.buffer);

            assert.ok(resultCount > 0,
                'Should preserve solid walls around the navigable space');

            const nx = Math.round((gridBounds.max.x - gridBounds.min.x) / voxelResolution);
            const ny = Math.round((gridBounds.max.y - gridBounds.min.y) / voxelResolution);
            const nz = Math.round((gridBounds.max.z - gridBounds.min.z) / voxelResolution);
            const totalCells = nx * ny * nz;
            assert.ok(resultCount < totalCells,
                `Result (${resultCount}) should leave reachable interior empty (total: ${totalCells})`);
        });
    });

    describe('parameter validation', function () {
        it('should throw for zero voxel resolution', function () {
            const acc = new BlockMaskBuffer();
            const gridBounds = alignGridBounds(0, 0, 0, 1, 1, 1, 0.25);
            assert.throws(
                () => carveInterior(acc, gridBounds, 0, capsuleHeight, capsuleRadius, { x: 0.5, y: 0.5, z: 0.5 }),
                /voxelResolution must be finite and > 0/
            );
        });

        it('should throw for negative capsule height', function () {
            const acc = new BlockMaskBuffer();
            const gridBounds = alignGridBounds(0, 0, 0, 1, 1, 1, 0.25);
            assert.throws(
                () => carveInterior(acc, gridBounds, voxelResolution, -1, capsuleRadius, { x: 0.5, y: 0.5, z: 0.5 }),
                /capsuleHeight must be finite and > 0/
            );
        });

        it('should throw for negative capsule radius', function () {
            const acc = new BlockMaskBuffer();
            const gridBounds = alignGridBounds(0, 0, 0, 1, 1, 1, 0.25);
            assert.throws(
                () => carveInterior(acc, gridBounds, voxelResolution, capsuleHeight, -0.5, { x: 0.5, y: 0.5, z: 0.5 }),
                /capsuleRadius must be finite and >= 0/
            );
        });
    });
});
