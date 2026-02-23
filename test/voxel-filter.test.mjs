/**
 * Tests for voxel filter (remove isolated voxels, fill isolated holes).
 *
 * Bit layout: bitIdx = lx + ly*4 + lz*16
 * lo = bits 0-31 (lz=0: bits 0-15, lz=1: bits 16-31)
 * hi = bits 32-63 (lz=2: bits 0-15, lz=3: bits 16-31)
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';

import {
    BlockAccumulator,
    xyzToMorton,
    popcount
} from '../src/lib/voxel/sparse-octree.js';
import { filterAndFillBlocks } from '../src/lib/voxel/voxel-filter.js';

const SOLID_LO = 0xFFFFFFFF >>> 0;
const SOLID_HI = 0xFFFFFFFF >>> 0;

/**
 * Set a single voxel bit in a 4x4x4 block.
 * Returns [lo, hi] mask pair.
 */
function voxelBit(lx, ly, lz) {
    const bitIdx = lx + ly * 4 + lz * 16;
    if (bitIdx < 32) {
        return [1 << bitIdx, 0];
    }
    return [0, 1 << (bitIdx - 32)];
}

/**
 * Combine multiple voxel positions into a single [lo, hi] mask.
 */
function voxelMask(...positions) {
    let lo = 0, hi = 0;
    for (const [lx, ly, lz] of positions) {
        const [blo, bhi] = voxelBit(lx, ly, lz);
        lo |= blo;
        hi |= bhi;
    }
    return [lo, hi];
}

/**
 * Count total set voxels across lo and hi.
 */
function countVoxels(lo, hi) {
    return popcount(lo) + popcount(hi);
}

/**
 * Check if a specific voxel is set in a [lo, hi] mask.
 */
function isVoxelSet(lo, hi, lx, ly, lz) {
    const bitIdx = lx + ly * 4 + lz * 16;
    if (bitIdx < 32) {
        return (lo & (1 << bitIdx)) !== 0;
    }
    return (hi & (1 << (bitIdx - 32))) !== 0;
}

// ============================================================================
// Isolated Voxel Removal
// ============================================================================

describe('filterAndFillBlocks', function () {
    describe('isolated voxel removal', function () {
        it('should remove a single isolated voxel with no neighbors', function () {
            const acc = new BlockAccumulator();
            // Single voxel at (1,1,1) in block (0,0,0) — no neighbors in any direction
            const [lo, hi] = voxelBit(1, 1, 1);
            acc.addBlock(xyzToMorton(0, 0, 0), lo, hi);

            const result = filterAndFillBlocks(acc);

            // Block should become empty and be discarded
            assert.strictEqual(result.count, 0, 'Isolated voxel should be removed');
        });

        it('should preserve voxels with at least one neighbor', function () {
            const acc = new BlockAccumulator();
            // Two adjacent voxels at (1,1,1) and (2,1,1) — neighbors in +X/-X
            const [lo, hi] = voxelMask([1, 1, 1], [2, 1, 1]);
            acc.addBlock(xyzToMorton(0, 0, 0), lo, hi);

            const result = filterAndFillBlocks(acc);

            assert.strictEqual(result.count, 1, 'Block with connected voxels should remain');
            const mixed = result.getMixedBlocks();
            assert.strictEqual(countVoxels(mixed.masks[0], mixed.masks[1]), 2,
                'Both connected voxels should be preserved');
        });

        it('should remove isolated voxels while preserving connected ones', function () {
            const acc = new BlockAccumulator();
            // (1,1,1) and (2,1,1) are neighbors; (0,0,3) is isolated
            const [lo, hi] = voxelMask([1, 1, 1], [2, 1, 1], [0, 0, 3]);
            acc.addBlock(xyzToMorton(0, 0, 0), lo, hi);

            const result = filterAndFillBlocks(acc);

            const mixed = result.getMixedBlocks();
            assert.strictEqual(mixed.morton.length, 1);
            const rlo = mixed.masks[0];
            const rhi = mixed.masks[1];

            assert.ok(isVoxelSet(rlo, rhi, 1, 1, 1), '(1,1,1) should be preserved');
            assert.ok(isVoxelSet(rlo, rhi, 2, 1, 1), '(2,1,1) should be preserved');
            assert.ok(!isVoxelSet(rlo, rhi, 0, 0, 3), '(0,0,3) should be removed');
        });

        it('should preserve voxels with diagonal neighbors if any axis-aligned neighbor exists', function () {
            const acc = new BlockAccumulator();
            // Line of 3 voxels along X — middle has 2 neighbors, ends have 1 each
            const [lo, hi] = voxelMask([0, 0, 0], [1, 0, 0], [2, 0, 0]);
            acc.addBlock(xyzToMorton(0, 0, 0), lo, hi);

            const result = filterAndFillBlocks(acc);

            const mixed = result.getMixedBlocks();
            assert.strictEqual(countVoxels(mixed.masks[0], mixed.masks[1]), 3,
                'All three connected voxels should be preserved');
        });
    });

    // ============================================================================
    // Hole Filling
    // ============================================================================

    describe('hole filling', function () {
        it('should fill a voxel surrounded by all 6 neighbors', function () {
            const acc = new BlockAccumulator();
            // 6-connected neighbors of (2,2,2) without (2,2,2) itself
            const [lo, hi] = voxelMask(
                [1, 2, 2], [3, 2, 2],  // -X, +X
                [2, 1, 2], [2, 3, 2],  // -Y, +Y
                [2, 2, 1], [2, 2, 3]   // -Z, +Z
            );
            acc.addBlock(xyzToMorton(0, 0, 0), lo, hi);

            const result = filterAndFillBlocks(acc);

            const mixed = result.getMixedBlocks();
            assert.strictEqual(mixed.morton.length, 1);
            const rlo = mixed.masks[0];
            const rhi = mixed.masks[1];

            assert.ok(isVoxelSet(rlo, rhi, 2, 2, 2),
                'Center voxel (2,2,2) should be filled');
        });

        it('should not fill a voxel missing one of 6 neighbors', function () {
            const acc = new BlockAccumulator();
            // Only 5 of 6 neighbors of (2,2,2) — missing +Z
            const [lo, hi] = voxelMask(
                [1, 2, 2], [3, 2, 2],  // -X, +X
                [2, 1, 2], [2, 3, 2],  // -Y, +Y
                [2, 2, 1]              // -Z only
            );
            acc.addBlock(xyzToMorton(0, 0, 0), lo, hi);

            const result = filterAndFillBlocks(acc);

            const mixed = result.getMixedBlocks();
            // (2,2,2) should NOT be filled since +Z neighbor is missing
            // But the 5 neighbor voxels should also be checked for isolation:
            // each has at most 0 axis-aligned neighbors from within this set
            // (they only neighbor the empty center), so they should all be removed.
            // Actually: (1,2,2) neighbors (2,2,2) which is empty, so it has 0 neighbors.
            // All voxels are isolated and should be removed.
            assert.strictEqual(result.count, 0,
                'All voxels are isolated (no mutual neighbors)');
        });
    });

    // ============================================================================
    // Solid Block Handling
    // ============================================================================

    describe('solid blocks', function () {
        it('should pass through solid blocks unchanged', function () {
            const acc = new BlockAccumulator();
            acc.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);

            const result = filterAndFillBlocks(acc);

            assert.strictEqual(result.solidCount, 1);
            assert.strictEqual(result.mixedCount, 0);
        });

        it('should pass through multiple solid blocks', function () {
            const acc = new BlockAccumulator();
            acc.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);
            acc.addBlock(xyzToMorton(1, 0, 0), SOLID_LO, SOLID_HI);
            acc.addBlock(xyzToMorton(0, 1, 0), SOLID_LO, SOLID_HI);

            const result = filterAndFillBlocks(acc);

            assert.strictEqual(result.solidCount, 3);
        });
    });

    // ============================================================================
    // Cross-Block Adjacency
    // ============================================================================

    describe('cross-block adjacency', function () {
        it('should preserve voxels at block boundaries with neighbors in adjacent block', function () {
            const acc = new BlockAccumulator();
            // Block (0,0,0): voxel at lx=3 (right face)
            const [lo0, hi0] = voxelBit(3, 1, 1);
            acc.addBlock(xyzToMorton(0, 0, 0), lo0, hi0);

            // Block (1,0,0): voxel at lx=0 (left face) — neighbor of above
            const [lo1, hi1] = voxelBit(0, 1, 1);
            acc.addBlock(xyzToMorton(1, 0, 0), lo1, hi1);

            const result = filterAndFillBlocks(acc);

            // Both voxels should be preserved since they are cross-block neighbors
            assert.strictEqual(result.count, 2,
                'Both blocks should remain (cross-block neighbors)');
            const mixed = result.getMixedBlocks();
            assert.strictEqual(mixed.morton.length, 2);
        });

        it('should remove boundary voxels without cross-block neighbors', function () {
            const acc = new BlockAccumulator();
            // Block (0,0,0): single voxel at lx=3 (right face)
            const [lo0, hi0] = voxelBit(3, 1, 1);
            acc.addBlock(xyzToMorton(0, 0, 0), lo0, hi0);

            // Block (2,0,0): single voxel — NOT adjacent to block (0,0,0)
            const [lo2, hi2] = voxelBit(0, 1, 1);
            acc.addBlock(xyzToMorton(2, 0, 0), lo2, hi2);

            const result = filterAndFillBlocks(acc);

            // Both voxels are isolated (no neighbors in any direction)
            assert.strictEqual(result.count, 0,
                'Isolated boundary voxels should be removed');
        });

        it('should handle cross-block adjacency in Y direction', function () {
            const acc = new BlockAccumulator();
            // Block (0,0,0): voxel at ly=3 (top face)
            const [lo0, hi0] = voxelBit(1, 3, 1);
            acc.addBlock(xyzToMorton(0, 0, 0), lo0, hi0);

            // Block (0,1,0): voxel at ly=0 (bottom face)
            const [lo1, hi1] = voxelBit(1, 0, 1);
            acc.addBlock(xyzToMorton(0, 1, 0), lo1, hi1);

            const result = filterAndFillBlocks(acc);

            assert.strictEqual(result.count, 2,
                'Y cross-block neighbors should be preserved');
        });

        it('should handle cross-block adjacency in Z direction', function () {
            const acc = new BlockAccumulator();
            // Block (0,0,0): voxel at lz=3 (far face, in hi word bits 16-31)
            const [lo0, hi0] = voxelBit(1, 1, 3);
            acc.addBlock(xyzToMorton(0, 0, 0), lo0, hi0);

            // Block (0,0,1): voxel at lz=0 (near face, in lo word bits 0-15)
            const [lo1, hi1] = voxelBit(1, 1, 0);
            acc.addBlock(xyzToMorton(0, 0, 1), lo1, hi1);

            const result = filterAndFillBlocks(acc);

            assert.strictEqual(result.count, 2,
                'Z cross-block neighbors should be preserved');
        });

        it('should use solid adjacent blocks as neighbor sources', function () {
            const acc = new BlockAccumulator();
            // Block (0,0,0): solid — provides neighbors for block (1,0,0)
            acc.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);

            // Block (1,0,0): single voxel at lx=0 (neighbors solid block's lx=3 face)
            const [lo1, hi1] = voxelBit(0, 2, 2);
            acc.addBlock(xyzToMorton(1, 0, 0), lo1, hi1);

            const result = filterAndFillBlocks(acc);

            // The mixed voxel at (1,0,0) lx=0 has a solid neighbor from block (0,0,0)
            assert.strictEqual(result.solidCount, 1, 'Solid block should remain');
            assert.strictEqual(result.mixedCount, 1,
                'Mixed voxel adjacent to solid block should be preserved');
        });
    });

    // ============================================================================
    // State Transitions
    // ============================================================================

    describe('state transitions', function () {
        it('should transition mixed block to empty when all voxels removed', function () {
            const acc = new BlockAccumulator();
            // Single isolated voxel
            const [lo, hi] = voxelBit(2, 2, 2);
            acc.addBlock(xyzToMorton(0, 0, 0), lo, hi);

            const result = filterAndFillBlocks(acc);

            assert.strictEqual(result.count, 0, 'Block with only isolated voxels should be removed');
            assert.strictEqual(result.mixedCount, 0);
            assert.strictEqual(result.solidCount, 0);
        });

        it('should transition mixed block to solid when all voxels filled', function () {
            const acc = new BlockAccumulator();
            // Start with all voxels set except one at (2,2,2)
            let lo = SOLID_LO;
            let hi = SOLID_HI;
            const bitIdx = 2 + 2 * 4 + 2 * 16;  // = 42, in hi word
            hi &= ~(1 << (bitIdx - 32));

            acc.addBlock(xyzToMorton(0, 0, 0), lo, hi);

            const result = filterAndFillBlocks(acc);

            // (2,2,2) has all 6 neighbors occupied, so it should be filled -> block becomes solid
            assert.strictEqual(result.solidCount, 1, 'Block should transition to solid');
            assert.strictEqual(result.mixedCount, 0);
        });
    });

    // ============================================================================
    // Edge Cases
    // ============================================================================

    describe('edge cases', function () {
        it('should handle empty accumulator', function () {
            const acc = new BlockAccumulator();
            const result = filterAndFillBlocks(acc);
            assert.strictEqual(result.count, 0);
        });

        it('should handle all-solid input', function () {
            const acc = new BlockAccumulator();
            for (let x = 0; x < 4; x++) {
                for (let y = 0; y < 4; y++) {
                    acc.addBlock(xyzToMorton(x, y, 0), SOLID_LO, SOLID_HI);
                }
            }

            const result = filterAndFillBlocks(acc);

            assert.strictEqual(result.solidCount, 16, 'All solid blocks should be preserved');
            assert.strictEqual(result.mixedCount, 0);
        });

        it('should handle voxels at block corners', function () {
            const acc = new BlockAccumulator();
            // Voxel at corner (0,0,0) of block (0,0,0) with neighbor at (3,3,3) of adjacent blocks
            // These are in different blocks and NOT adjacent, so both should be removed
            const [lo0, hi0] = voxelBit(0, 0, 0);
            acc.addBlock(xyzToMorton(0, 0, 0), lo0, hi0);

            const result = filterAndFillBlocks(acc);

            assert.strictEqual(result.count, 0,
                'Isolated corner voxel should be removed');
        });

        it('should preserve a 2x2x2 solid cube within a block', function () {
            const acc = new BlockAccumulator();
            // 2x2x2 cube at positions (1,1,1) through (2,2,2)
            const [lo, hi] = voxelMask(
                [1, 1, 1], [2, 1, 1], [1, 2, 1], [2, 2, 1],
                [1, 1, 2], [2, 1, 2], [1, 2, 2], [2, 2, 2]
            );
            acc.addBlock(xyzToMorton(0, 0, 0), lo, hi);

            const result = filterAndFillBlocks(acc);

            const mixed = result.getMixedBlocks();
            assert.strictEqual(mixed.morton.length, 1);
            assert.strictEqual(countVoxels(mixed.masks[0], mixed.masks[1]), 8,
                'All 8 voxels in the 2x2x2 cube should be preserved');
        });
    });
});
