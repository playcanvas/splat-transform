/**
 * Tests for Sparse Octree (Phase 4 of voxelizer).
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';

import { Vec3 } from 'playcanvas';

import { BlockMaskBuffer } from '../src/lib/voxel/block-mask-buffer.js';
import {
    xyzToMorton,
    mortonToXYZ,
    popcount,
    isSolid,
    isEmpty,
    getChildOffset
} from '../src/lib/voxel/morton.js';
import {
    buildSparseOctree,
    SOLID_LEAF_MARKER
} from '../src/lib/writers/sparse-octree.js';
import { alignGridBounds } from '../src/lib/voxel/voxelize.js';

// ============================================================================
// Morton Code Tests
// ============================================================================

describe('Morton codes', function () {
    describe('xyzToMorton', function () {
        it('should encode origin correctly', function () {
            const morton = xyzToMorton(0, 0, 0);
            assert.strictEqual(morton, 0);
        });

        it('should encode unit positions correctly', function () {
            // x=1: bit 0 set -> morton = 1
            assert.strictEqual(xyzToMorton(1, 0, 0), 1);
            // y=1: bit 1 set -> morton = 2
            assert.strictEqual(xyzToMorton(0, 1, 0), 2);
            // z=1: bit 2 set -> morton = 4
            assert.strictEqual(xyzToMorton(0, 0, 1), 4);
            // x=1, y=1, z=1: bits 0,1,2 set -> morton = 7
            assert.strictEqual(xyzToMorton(1, 1, 1), 7);
        });

        it('should encode larger coordinates', function () {
            // x=2: bit 3 set -> morton = 8
            assert.strictEqual(xyzToMorton(2, 0, 0), 8);
            // x=3: bits 0 and 3 set -> morton = 9
            assert.strictEqual(xyzToMorton(3, 0, 0), 9);
        });
    });

    describe('mortonToXYZ', function () {
        it('should decode origin correctly', function () {
            const [x, y, z] = mortonToXYZ(0);
            assert.deepStrictEqual([x, y, z], [0, 0, 0]);
        });

        it('should decode unit positions correctly', function () {
            assert.deepStrictEqual(mortonToXYZ(1), [1, 0, 0]);
            assert.deepStrictEqual(mortonToXYZ(2), [0, 1, 0]);
            assert.deepStrictEqual(mortonToXYZ(4), [0, 0, 1]);
            assert.deepStrictEqual(mortonToXYZ(7), [1, 1, 1]);
        });
    });

    describe('round-trip', function () {
        it('should round-trip small coordinates', function () {
            const testCases = [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 1],
                [2, 3, 4],
                [7, 7, 7],
                [15, 15, 15]
            ];

            for (const [x, y, z] of testCases) {
                const morton = xyzToMorton(x, y, z);
                const [rx, ry, rz] = mortonToXYZ(morton);
                assert.deepStrictEqual(
                    [rx, ry, rz],
                    [x, y, z],
                    `Failed for (${x}, ${y}, ${z})`
                );
            }
        });

        it('should round-trip larger coordinates', function () {
            const testCases = [
                [100, 50, 75],
                [1000, 500, 750],
                [255, 255, 255],
                [1023, 1023, 1023]
            ];

            for (const [x, y, z] of testCases) {
                const morton = xyzToMorton(x, y, z);
                const [rx, ry, rz] = mortonToXYZ(morton);
                assert.deepStrictEqual(
                    [rx, ry, rz],
                    [x, y, z],
                    `Failed for (${x}, ${y}, ${z})`
                );
            }
        });

        it('should handle maximum supported coordinates', function () {
            // 17 bits per axis = max 131071
            const max = (1 << 17) - 1;
            const morton = xyzToMorton(max, max, max);
            const [rx, ry, rz] = mortonToXYZ(morton);
            assert.deepStrictEqual([rx, ry, rz], [max, max, max]);
        });
    });

    describe('ordering', function () {
        it('should preserve Z-order', function () {
            // In Morton order, (0,0,0) < (1,0,0) < (0,1,0) < (1,1,0) < ...
            const m000 = xyzToMorton(0, 0, 0);
            const m100 = xyzToMorton(1, 0, 0);
            const m010 = xyzToMorton(0, 1, 0);
            const m110 = xyzToMorton(1, 1, 0);
            const m001 = xyzToMorton(0, 0, 1);

            assert.ok(m000 < m100, '(0,0,0) should be < (1,0,0)');
            assert.ok(m100 < m010, '(1,0,0) should be < (0,1,0)');
            assert.ok(m010 < m110, '(0,1,0) should be < (1,1,0)');
            assert.ok(m110 < m001, '(1,1,0) should be < (0,0,1)');
        });

        it('should group siblings together', function () {
            // All 8 children of parent at (0,0,0) should have Morton codes 0-7
            for (let i = 0; i < 8; i++) {
                const x = i & 1;
                const y = (i >> 1) & 1;
                const z = (i >> 2) & 1;
                const morton = xyzToMorton(x, y, z);
                assert.ok(morton >= 0 && morton < 8, `Child ${i} should have Morton < 8`);
            }
        });
    });

    describe('parent-child relationship', function () {
        it('should compute parent Morton code correctly', function () {
            // Parent of any child is child >>> 3
            const child = xyzToMorton(3, 2, 1);
            const parent = Math.floor(child / 8);

            // Parent coordinates should be (1, 1, 0)
            const [px, py, pz] = mortonToXYZ(parent);
            assert.deepStrictEqual([px, py, pz], [1, 1, 0]);
        });

        it('should compute octant correctly', function () {
            // Octant is child % 8
            const coords = [
                [[0, 0, 0], 0],
                [[1, 0, 0], 1],
                [[0, 1, 0], 2],
                [[1, 1, 0], 3],
                [[0, 0, 1], 4],
                [[1, 0, 1], 5],
                [[0, 1, 1], 6],
                [[1, 1, 1], 7]
            ];

            for (const [[x, y, z], expectedOctant] of coords) {
                const morton = xyzToMorton(x, y, z);
                const octant = morton % 8;
                assert.strictEqual(octant, expectedOctant, `Octant for (${x},${y},${z})`);
            }
        });
    });
});

// ============================================================================
// Utility Function Tests
// ============================================================================

describe('Utility functions', function () {
    describe('popcount', function () {
        it('should count zero bits', function () {
            assert.strictEqual(popcount(0), 0);
        });

        it('should count single bits', function () {
            assert.strictEqual(popcount(1), 1);
            assert.strictEqual(popcount(2), 1);
            assert.strictEqual(popcount(4), 1);
            assert.strictEqual(popcount(0x80000000), 1);
        });

        it('should count multiple bits', function () {
            assert.strictEqual(popcount(3), 2);     // 0b11
            assert.strictEqual(popcount(7), 3);     // 0b111
            assert.strictEqual(popcount(0xFF), 8);  // 8 bits
            assert.strictEqual(popcount(0xFFFFFFFF), 32);
        });

        it('should handle arbitrary patterns', function () {
            assert.strictEqual(popcount(0xAAAAAAAA), 16); // Alternating bits
            assert.strictEqual(popcount(0x55555555), 16); // Alternating bits
        });
    });

    describe('isSolid', function () {
        it('should return true for all bits set', function () {
            assert.strictEqual(isSolid(0xFFFFFFFF, 0xFFFFFFFF), true);
        });

        it('should return false for partial fill', function () {
            assert.strictEqual(isSolid(0xFFFFFFFF, 0), false);
            assert.strictEqual(isSolid(0, 0xFFFFFFFF), false);
            assert.strictEqual(isSolid(0xFFFFFFFE, 0xFFFFFFFF), false);
        });

        it('should return false for empty', function () {
            assert.strictEqual(isSolid(0, 0), false);
        });
    });

    describe('isEmpty', function () {
        it('should return true for no bits set', function () {
            assert.strictEqual(isEmpty(0, 0), true);
        });

        it('should return false for any bits set', function () {
            assert.strictEqual(isEmpty(1, 0), false);
            assert.strictEqual(isEmpty(0, 1), false);
            assert.strictEqual(isEmpty(0xFFFFFFFF, 0xFFFFFFFF), false);
        });
    });

    describe('getChildOffset', function () {
        it('should return 0 for first child', function () {
            // Mask with all children: octant 0 is first
            assert.strictEqual(getChildOffset(0xFF, 0), 0);
        });

        it('should count preceding children', function () {
            // Mask 0b11111111 (all children): offset = octant
            for (let octant = 0; octant < 8; octant++) {
                assert.strictEqual(getChildOffset(0xFF, octant), octant);
            }
        });

        it('should skip missing children', function () {
            // Mask 0b10101010 (octants 1, 3, 5, 7 present)
            const mask = 0b10101010;
            assert.strictEqual(getChildOffset(mask, 1), 0); // First present
            assert.strictEqual(getChildOffset(mask, 3), 1); // Second present
            assert.strictEqual(getChildOffset(mask, 5), 2); // Third present
            assert.strictEqual(getChildOffset(mask, 7), 3); // Fourth present
        });

        it('should handle sparse masks', function () {
            // Only octant 7 present
            assert.strictEqual(getChildOffset(0b10000000, 7), 0);

            // Octants 0 and 7 present
            assert.strictEqual(getChildOffset(0b10000001, 0), 0);
            assert.strictEqual(getChildOffset(0b10000001, 7), 1);
        });
    });
});

// ============================================================================
// BlockMaskBuffer Tests
// ============================================================================

describe('BlockMaskBuffer', function () {
    describe('addBlock', function () {
        it('should classify solid blocks', function () {
            const acc = new BlockMaskBuffer();
            acc.addBlock(xyzToMorton(0, 0, 0), 0xFFFFFFFF, 0xFFFFFFFF);

            assert.strictEqual(acc.solidCount, 1);
            assert.strictEqual(acc.mixedCount, 0);
            assert.strictEqual(acc.count, 1);
        });

        it('should classify mixed blocks', function () {
            const acc = new BlockMaskBuffer();
            acc.addBlock(xyzToMorton(0, 0, 0), 0x12345678, 0x9ABCDEF0);

            assert.strictEqual(acc.solidCount, 0);
            assert.strictEqual(acc.mixedCount, 1);
            assert.strictEqual(acc.count, 1);
        });

        it('should discard empty blocks', function () {
            const acc = new BlockMaskBuffer();
            acc.addBlock(xyzToMorton(0, 0, 0), 0, 0);

            assert.strictEqual(acc.count, 0);
        });

        it('should handle multiple blocks', function () {
            const acc = new BlockMaskBuffer();

            // Add 3 solid, 2 mixed, 1 empty
            acc.addBlock(xyzToMorton(0, 0, 0), 0xFFFFFFFF, 0xFFFFFFFF); // solid
            acc.addBlock(xyzToMorton(1, 0, 0), 0xFFFFFFFF, 0xFFFFFFFF); // solid
            acc.addBlock(xyzToMorton(2, 0, 0), 0xFFFFFFFF, 0xFFFFFFFF); // solid
            acc.addBlock(xyzToMorton(3, 0, 0), 0x00000001, 0x00000000); // mixed
            acc.addBlock(xyzToMorton(4, 0, 0), 0xFFFFFFFE, 0xFFFFFFFF); // mixed
            acc.addBlock(xyzToMorton(5, 0, 0), 0, 0);                   // empty

            assert.strictEqual(acc.solidCount, 3);
            assert.strictEqual(acc.mixedCount, 2);
            assert.strictEqual(acc.count, 5);
        });
    });

    describe('getMixedBlocks', function () {
        it('should return morton codes and interleaved masks', function () {
            const acc = new BlockMaskBuffer();
            acc.addBlock(xyzToMorton(0, 0, 0), 0x11111111, 0x22222222);
            acc.addBlock(xyzToMorton(1, 0, 0), 0x33333333, 0x44444444);

            const mixed = acc.getMixedBlocks();

            assert.strictEqual(mixed.morton.length, 2);
            assert.strictEqual(mixed.masks.length, 4); // 2 blocks * 2 values

            // Check first block
            assert.strictEqual(mixed.masks[0], 0x11111111);
            assert.strictEqual(mixed.masks[1], 0x22222222);

            // Check second block
            assert.strictEqual(mixed.masks[2], 0x33333333);
            assert.strictEqual(mixed.masks[3], 0x44444444);
        });
    });

    describe('getSolidBlocks', function () {
        it('should return morton codes only', function () {
            const acc = new BlockMaskBuffer();
            acc.addBlock(xyzToMorton(0, 0, 0), 0xFFFFFFFF, 0xFFFFFFFF);
            acc.addBlock(xyzToMorton(5, 5, 5), 0xFFFFFFFF, 0xFFFFFFFF);

            const solid = acc.getSolidBlocks();

            assert.strictEqual(solid.length, 2);
            assert.ok(solid.includes(xyzToMorton(0, 0, 0)));
            assert.ok(solid.includes(xyzToMorton(5, 5, 5)));
        });
    });

    describe('clear', function () {
        it('should remove all blocks', function () {
            const acc = new BlockMaskBuffer();
            acc.addBlock(xyzToMorton(0, 0, 0), 0xFFFFFFFF, 0xFFFFFFFF);
            acc.addBlock(xyzToMorton(1, 0, 0), 0x12345678, 0x9ABCDEF0);

            assert.strictEqual(acc.count, 2);

            acc.clear();

            assert.strictEqual(acc.count, 0);
            assert.strictEqual(acc.solidCount, 0);
            assert.strictEqual(acc.mixedCount, 0);
        });
    });
});

// ============================================================================
// Octree Construction Tests
// ============================================================================

describe('buildSparseOctree', function () {
    describe('empty octree', function () {
        it('should handle empty buffer', function () {
            const acc = new BlockMaskBuffer();
            const gridBounds = {
                min: new Vec3(0, 0, 0),
                max: new Vec3(1, 1, 1)
            };
            const sceneBounds = {
                min: new Vec3(0, 0, 0),
                max: new Vec3(1, 1, 1)
            };

            const octree = buildSparseOctree(acc, gridBounds, sceneBounds, 0.25);

            assert.strictEqual(octree.nodes.length, 0);
            assert.strictEqual(octree.leafData.length, 0);
        });
    });

    describe('single block', function () {
        it('should create octree with single solid block', function () {
            const acc = new BlockMaskBuffer();
            acc.addBlock(xyzToMorton(0, 0, 0), 0xFFFFFFFF, 0xFFFFFFFF);

            const gridBounds = {
                min: new Vec3(0, 0, 0),
                max: new Vec3(1, 1, 1)
            };
            const sceneBounds = gridBounds;

            const octree = buildSparseOctree(acc, gridBounds, sceneBounds, 0.25);

            assert.ok(octree.nodes.length >= 1);
            assert.strictEqual(octree.numMixedLeaves, 0);
        });

        it('should create octree with single mixed block', function () {
            const acc = new BlockMaskBuffer();
            acc.addBlock(xyzToMorton(0, 0, 0), 0x12345678, 0x9ABCDEF0);

            const gridBounds = {
                min: new Vec3(0, 0, 0),
                max: new Vec3(1, 1, 1)
            };
            const sceneBounds = gridBounds;

            const octree = buildSparseOctree(acc, gridBounds, sceneBounds, 0.25);

            assert.ok(octree.nodes.length >= 1);
            assert.strictEqual(octree.numMixedLeaves, 1);
            assert.strictEqual(octree.leafData.length, 2); // lo + hi
            assert.strictEqual(octree.leafData[0], 0x12345678);
            assert.strictEqual(octree.leafData[1], 0x9ABCDEF0);
        });
    });

    describe('solid region merging', function () {
        it('should collapse 8 solid siblings into solid parent', function () {
            const acc = new BlockMaskBuffer();

            // Add all 8 children of the root (octants 0-7)
            for (let z = 0; z < 2; z++) {
                for (let y = 0; y < 2; y++) {
                    for (let x = 0; x < 2; x++) {
                        acc.addBlock(xyzToMorton(x, y, z), 0xFFFFFFFF, 0xFFFFFFFF);
                    }
                }
            }

            assert.strictEqual(acc.solidCount, 8);

            const gridBounds = {
                min: new Vec3(0, 0, 0),
                max: new Vec3(2, 2, 2)
            };
            const sceneBounds = gridBounds;

            const octree = buildSparseOctree(acc, gridBounds, sceneBounds, 0.25);

            // Should collapse to a single solid node (or at most a few nodes)
            // The exact count depends on tree depth calculation
            assert.ok(octree.nodes.length <= 8, 'Should have merged some nodes');
        });

        it('should not collapse mixed siblings', function () {
            const acc = new BlockMaskBuffer();

            // Add 7 solid + 1 mixed
            for (let i = 0; i < 7; i++) {
                const x = i & 1;
                const y = (i >> 1) & 1;
                const z = (i >> 2) & 1;
                acc.addBlock(xyzToMorton(x, y, z), 0xFFFFFFFF, 0xFFFFFFFF);
            }
            // Last one is mixed
            acc.addBlock(xyzToMorton(1, 1, 1), 0x12345678, 0x9ABCDEF0);

            const gridBounds = {
                min: new Vec3(0, 0, 0),
                max: new Vec3(2, 2, 2)
            };
            const sceneBounds = gridBounds;

            const octree = buildSparseOctree(acc, gridBounds, sceneBounds, 0.25);

            // Should have at least 8 leaf nodes (not collapsed)
            assert.strictEqual(octree.numMixedLeaves, 1);
        });
    });

    describe('node encoding', function () {
        it('should encode solid leaves correctly', function () {
            const acc = new BlockMaskBuffer();
            acc.addBlock(xyzToMorton(0, 0, 0), 0xFFFFFFFF, 0xFFFFFFFF);

            const gridBounds = {
                min: new Vec3(0, 0, 0),
                max: new Vec3(1, 1, 1)
            };

            const octree = buildSparseOctree(acc, gridBounds, gridBounds, 0.25);

            // Check that at least one node has solid marker
            let hasSolidLeaf = false;
            for (let i = 0; i < octree.nodes.length; i++) {
                if (octree.nodes[i] === SOLID_LEAF_MARKER) {
                    hasSolidLeaf = true;
                    break;
                }
            }
            assert.ok(hasSolidLeaf, 'Should have a solid leaf node');
        });

        it('should encode mixed leaves with leafData index', function () {
            const acc = new BlockMaskBuffer();
            acc.addBlock(xyzToMorton(0, 0, 0), 0xAAAAAAAA, 0x55555555);

            const gridBounds = {
                min: new Vec3(0, 0, 0),
                max: new Vec3(1, 1, 1)
            };

            const octree = buildSparseOctree(acc, gridBounds, gridBounds, 0.25);

            // Check that leafData contains the mask
            assert.strictEqual(octree.leafData.length, 2);
            assert.strictEqual(octree.leafData[0], 0xAAAAAAAA);
            assert.strictEqual(octree.leafData[1], 0x55555555);

            // Check that a node has mixed leaf encoding
            // Mixed leaves have highByte=0x00 (lower 24 bits = leafData index)
            let hasMixedLeaf = false;
            for (let i = 0; i < octree.nodes.length; i++) {
                const n = octree.nodes[i] >>> 0;
                if (n !== SOLID_LEAF_MARKER && ((n >> 24) & 0xFF) === 0x00) {
                    hasMixedLeaf = true;
                    break;
                }
            }
            assert.ok(hasMixedLeaf, 'Should have a mixed leaf node');
        });
    });

    describe('metadata', function () {
        it('should preserve bounds and resolution', function () {
            const acc = new BlockMaskBuffer();
            acc.addBlock(xyzToMorton(0, 0, 0), 0xFFFFFFFF, 0xFFFFFFFF);

            const gridBounds = {
                min: new Vec3(-10, -5, -10),
                max: new Vec3(10, 5, 10)
            };
            const sceneBounds = {
                min: new Vec3(-9, -4, -9),
                max: new Vec3(9, 4, 9)
            };

            const octree = buildSparseOctree(acc, gridBounds, sceneBounds, 0.05);

            assert.strictEqual(octree.voxelResolution, 0.05);
            assert.strictEqual(octree.leafSize, 4);
            assert.deepStrictEqual(
                [octree.gridBounds.min.x, octree.gridBounds.min.y, octree.gridBounds.min.z],
                [-10, -5, -10]
            );
            assert.deepStrictEqual(
                [octree.sceneBounds.min.x, octree.sceneBounds.min.y, octree.sceneBounds.min.z],
                [-9, -4, -9]
            );
        });
    });
});

// ============================================================================
// alignGridBounds Tests
// ============================================================================

describe('alignGridBounds', function () {
    it('should align to block boundaries', function () {
        const bounds = alignGridBounds(0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.25);

        // Block size = 4 * 0.25 = 1.0
        // Min should floor to 0, max should ceil to 1
        assert.strictEqual(bounds.min.x, 0);
        assert.strictEqual(bounds.min.y, 0);
        assert.strictEqual(bounds.min.z, 0);
        assert.strictEqual(bounds.max.x, 1);
        assert.strictEqual(bounds.max.y, 1);
        assert.strictEqual(bounds.max.z, 1);
    });

    it('should handle negative coordinates', function () {
        const bounds = alignGridBounds(-5.5, -3.2, -7.8, 5.5, 3.2, 7.8, 0.5);

        // Block size = 4 * 0.5 = 2.0
        // Min should floor, max should ceil
        assert.strictEqual(bounds.min.x, -6);
        assert.strictEqual(bounds.min.y, -4);
        assert.strictEqual(bounds.min.z, -8);
        assert.strictEqual(bounds.max.x, 6);
        assert.strictEqual(bounds.max.y, 4);
        assert.strictEqual(bounds.max.z, 8);
    });

    it('should expand small bounds', function () {
        const bounds = alignGridBounds(0.001, 0.001, 0.001, 0.002, 0.002, 0.002, 0.05);

        // Block size = 4 * 0.05 = 0.2
        // Should expand to at least one block
        assert.strictEqual(bounds.min.x, 0);
        assert.strictEqual(bounds.max.x, 0.2);
    });
});
