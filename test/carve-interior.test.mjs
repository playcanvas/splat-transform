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
import { xyzToMorton, popcount } from '../src/lib/voxel/morton.js';
import { alignGridBounds } from '../src/lib/voxel/sparse-octree.js';
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
 * Build a hollow box of solid blocks. The box has solid walls of 1 block thick
 * and an empty interior. Returns the accumulator and grid bounds.
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
            const resultCount = countSolidVoxels(result.accumulator);

            assert.ok(resultCount > 0,
                'Should produce solid voxels around the navigable space');
        });

        it('should not include reachable cells as solid', function () {
            const { acc, gridBounds } = buildHollowBox(6, voxelResolution);

            const centerWorld = (gridBounds.min.x + gridBounds.max.x) / 2;
            const seed = { x: centerWorld, y: centerWorld, z: centerWorld };

            const result = carveInterior(acc, gridBounds, voxelResolution, capsuleHeight, capsuleRadius, seed);

            const resultCount = countSolidVoxels(result.accumulator);
            const nx = Math.round((gridBounds.max.x - gridBounds.min.x) / voxelResolution);
            const totalCells = nx * nx * nx;

            assert.ok(resultCount < totalCells,
                `Result (${resultCount}) must leave reachable cells empty (total grid: ${totalCells})`);
        });
    });

    describe('seed validation', function () {
        it('should return original accumulator if seed is outside grid', function () {
            const { acc, gridBounds } = buildHollowBox(4, voxelResolution);

            const seed = { x: -100, y: -100, z: -100 };
            const result = carveInterior(acc, gridBounds, voxelResolution, capsuleHeight, capsuleRadius, seed);

            assert.strictEqual(countSolidVoxels(result.accumulator), countSolidVoxels(acc),
                'Should return original when seed is outside grid');
        });

        it('should return original accumulator if seed is in solid region', function () {
            // 3-block box: walls at blocks 0 and 2, interior is only block 1
            // (4 voxels per axis). After dilation by yHalfExtent=3 in Y the
            // interior is fully blocked, so no free cell exists within search
            // radius and the function returns the original accumulator.
            const { acc, gridBounds } = buildHollowBox(3, voxelResolution);

            const seed = {
                x: gridBounds.min.x + voxelResolution,
                y: gridBounds.min.y + voxelResolution,
                z: gridBounds.min.z + voxelResolution
            };
            const result = carveInterior(acc, gridBounds, voxelResolution, capsuleHeight, capsuleRadius, seed);

            assert.strictEqual(countSolidVoxels(result.accumulator), countSolidVoxels(acc),
                'Should return original when seed is in blocked region');
        });
    });

    describe('empty accumulator', function () {
        it('should carve out all reachable space (no obstacles)', function () {
            const acc = new BlockMaskBuffer();
            const gridBounds = alignGridBounds(0, 0, 0, 1, 1, 1, voxelResolution);
            const seed = { x: 0.5, y: 0.5, z: 0.5 };

            const result = carveInterior(acc, gridBounds, voxelResolution, capsuleHeight, capsuleRadius, seed);
            const resultCount = countSolidVoxels(result.accumulator);
            const nx = Math.round((gridBounds.max.x - gridBounds.min.x) / voxelResolution);
            const ny = Math.round((gridBounds.max.y - gridBounds.min.y) / voxelResolution);
            const nz = Math.round((gridBounds.max.z - gridBounds.min.z) / voxelResolution);
            const totalCells = nx * ny * nz;

            assert.ok(resultCount < totalCells,
                'With no obstacles the entire grid is reachable; most cells should be empty');
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

            const resultCount = countSolidVoxels(result.accumulator);
            assert.ok(resultCount > 0,
                'Should retain solid voxels near the reachable space');
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
            const resultCount = countSolidVoxels(result.accumulator);

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
});
