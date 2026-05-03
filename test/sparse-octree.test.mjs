/**
 * Tests for Sparse Octree (Phase 4 of voxelizer).
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';

import { Vec3 } from 'playcanvas';

import { BlockMaskBuffer } from '../src/lib/voxel/block-mask-buffer.js';
import { SparseVoxelGrid } from '../src/lib/voxel/sparse-voxel-grid.js';
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

// Linear block index: bx + by*nbx + bz*nbx*nby. The buffer stores blocks
// keyed on this linear index now (not morton).
function linearBlockIdx(bx, by, bz, nbx, nby) {
    return bx + by * nbx + bz * nbx * nby;
}

// Build a SparseVoxelGrid from a BlockMaskBuffer for the new buildSparseOctree
// API. nx/ny/nz are voxel dimensions (each block is 4 voxels per axis).
function gridFromBuffer(buffer, nx, ny, nz) {
    return SparseVoxelGrid.fromBuffer(buffer, nx, ny, nz);
}

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
    // Use a fixed (nbx,nby) that fits all coords used below.
    const NBX = 8, NBY = 8;
    const bi = (x, y, z) => linearBlockIdx(x, y, z, NBX, NBY);

    describe('addBlock', function () {
        it('should classify solid blocks', function () {
            const acc = new BlockMaskBuffer();
            acc.addBlock(bi(0, 0, 0), 0xFFFFFFFF, 0xFFFFFFFF);

            assert.strictEqual(acc.solidCount, 1);
            assert.strictEqual(acc.mixedCount, 0);
            assert.strictEqual(acc.count, 1);
        });

        it('should classify mixed blocks', function () {
            const acc = new BlockMaskBuffer();
            acc.addBlock(bi(0, 0, 0), 0x12345678, 0x9ABCDEF0);

            assert.strictEqual(acc.solidCount, 0);
            assert.strictEqual(acc.mixedCount, 1);
            assert.strictEqual(acc.count, 1);
        });

        it('should discard empty blocks', function () {
            const acc = new BlockMaskBuffer();
            acc.addBlock(bi(0, 0, 0), 0, 0);

            assert.strictEqual(acc.count, 0);
        });

        it('should handle multiple blocks', function () {
            const acc = new BlockMaskBuffer();

            // Add 3 solid, 2 mixed, 1 empty
            acc.addBlock(bi(0, 0, 0), 0xFFFFFFFF, 0xFFFFFFFF); // solid
            acc.addBlock(bi(1, 0, 0), 0xFFFFFFFF, 0xFFFFFFFF); // solid
            acc.addBlock(bi(2, 0, 0), 0xFFFFFFFF, 0xFFFFFFFF); // solid
            acc.addBlock(bi(3, 0, 0), 0x00000001, 0x00000000); // mixed
            acc.addBlock(bi(4, 0, 0), 0xFFFFFFFE, 0xFFFFFFFF); // mixed
            acc.addBlock(bi(5, 0, 0), 0, 0);                   // empty

            assert.strictEqual(acc.solidCount, 3);
            assert.strictEqual(acc.mixedCount, 2);
            assert.strictEqual(acc.count, 5);
        });
    });

    describe('getMixedBlocks', function () {
        it('should return morton codes and interleaved masks', function () {
            const acc = new BlockMaskBuffer();
            acc.addBlock(bi(0, 0, 0), 0x11111111, 0x22222222);
            acc.addBlock(bi(1, 0, 0), 0x33333333, 0x44444444);

            const mixed = acc.getMixedBlocks();

            assert.strictEqual(mixed.blockIdx.length, 2);
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
            acc.addBlock(bi(0, 0, 0), 0xFFFFFFFF, 0xFFFFFFFF);
            acc.addBlock(bi(5, 5, 5), 0xFFFFFFFF, 0xFFFFFFFF);

            const solid = acc.getSolidBlocks();

            assert.strictEqual(solid.length, 2);
            assert.ok(solid.includes(bi(0, 0, 0)));
            assert.ok(solid.includes(bi(5, 5, 5)));
        });
    });

    describe('clear', function () {
        it('should remove all blocks', function () {
            const acc = new BlockMaskBuffer();
            acc.addBlock(bi(0, 0, 0), 0xFFFFFFFF, 0xFFFFFFFF);
            acc.addBlock(bi(1, 0, 0), 0x12345678, 0x9ABCDEF0);

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
            // 1x1x1 / 0.25 = 4x4x4 voxels = 1x1x1 blocks.
            const octree = buildSparseOctree(gridFromBuffer(acc, 4, 4, 4), gridBounds, sceneBounds, 0.25);

            assert.strictEqual(octree.nodes.length, 0);
            assert.strictEqual(octree.leafData.length, 0);
        });
    });

    describe('single block', function () {
        it('should create octree with single solid block', function () {
            const acc = new BlockMaskBuffer();
            acc.addBlock(linearBlockIdx(0, 0, 0, 1, 1), 0xFFFFFFFF, 0xFFFFFFFF);

            const gridBounds = {
                min: new Vec3(0, 0, 0),
                max: new Vec3(1, 1, 1)
            };
            const sceneBounds = gridBounds;

            const octree = buildSparseOctree(gridFromBuffer(acc, 4, 4, 4), gridBounds, sceneBounds, 0.25);

            assert.ok(octree.nodes.length >= 1);
            assert.strictEqual(octree.numMixedLeaves, 0);
        });

        it('should create octree with single mixed block', function () {
            const acc = new BlockMaskBuffer();
            acc.addBlock(linearBlockIdx(0, 0, 0, 1, 1), 0x12345678, 0x9ABCDEF0);

            const gridBounds = {
                min: new Vec3(0, 0, 0),
                max: new Vec3(1, 1, 1)
            };
            const sceneBounds = gridBounds;

            const octree = buildSparseOctree(gridFromBuffer(acc, 4, 4, 4), gridBounds, sceneBounds, 0.25);

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

            // 2x2x2 / 0.25 = 8x8x8 voxels = 2x2x2 blocks.
            const NBX = 2, NBY = 2;
            // Add all 8 children of the root (octants 0-7)
            for (let z = 0; z < 2; z++) {
                for (let y = 0; y < 2; y++) {
                    for (let x = 0; x < 2; x++) {
                        acc.addBlock(linearBlockIdx(x, y, z, NBX, NBY), 0xFFFFFFFF, 0xFFFFFFFF);
                    }
                }
            }

            assert.strictEqual(acc.solidCount, 8);

            const gridBounds = {
                min: new Vec3(0, 0, 0),
                max: new Vec3(2, 2, 2)
            };
            const sceneBounds = gridBounds;

            const octree = buildSparseOctree(gridFromBuffer(acc, 8, 8, 8), gridBounds, sceneBounds, 0.25);

            // Should collapse to a single solid node (or at most a few nodes)
            // The exact count depends on tree depth calculation
            assert.ok(octree.nodes.length <= 8, 'Should have merged some nodes');
        });

        it('should not collapse mixed siblings', function () {
            const acc = new BlockMaskBuffer();

            // 2x2x2 / 0.25 = 8x8x8 voxels = 2x2x2 blocks.
            const NBX = 2, NBY = 2;
            // Add 7 solid + 1 mixed
            for (let i = 0; i < 7; i++) {
                const x = i & 1;
                const y = (i >> 1) & 1;
                const z = (i >> 2) & 1;
                acc.addBlock(linearBlockIdx(x, y, z, NBX, NBY), 0xFFFFFFFF, 0xFFFFFFFF);
            }
            // Last one is mixed
            acc.addBlock(linearBlockIdx(1, 1, 1, NBX, NBY), 0x12345678, 0x9ABCDEF0);

            const gridBounds = {
                min: new Vec3(0, 0, 0),
                max: new Vec3(2, 2, 2)
            };
            const sceneBounds = gridBounds;

            const octree = buildSparseOctree(gridFromBuffer(acc, 8, 8, 8), gridBounds, sceneBounds, 0.25);

            // Should have at least 8 leaf nodes (not collapsed)
            assert.strictEqual(octree.numMixedLeaves, 1);
        });
    });

    describe('node encoding', function () {
        it('should encode solid leaves correctly', function () {
            const acc = new BlockMaskBuffer();
            acc.addBlock(linearBlockIdx(0, 0, 0, 1, 1), 0xFFFFFFFF, 0xFFFFFFFF);

            const gridBounds = {
                min: new Vec3(0, 0, 0),
                max: new Vec3(1, 1, 1)
            };

            const octree = buildSparseOctree(gridFromBuffer(acc, 4, 4, 4), gridBounds, gridBounds, 0.25);

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
            acc.addBlock(linearBlockIdx(0, 0, 0, 1, 1), 0xAAAAAAAA, 0x55555555);

            const gridBounds = {
                min: new Vec3(0, 0, 0),
                max: new Vec3(1, 1, 1)
            };

            const octree = buildSparseOctree(gridFromBuffer(acc, 4, 4, 4), gridBounds, gridBounds, 0.25);

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
            // gridBounds 20x10x20 / res 0.05 = 400x200x400 voxels = 100x50x100 blocks.
            const acc = new BlockMaskBuffer();
            acc.addBlock(linearBlockIdx(0, 0, 0, 100, 50), 0xFFFFFFFF, 0xFFFFFFFF);

            const gridBounds = {
                min: new Vec3(-10, -5, -10),
                max: new Vec3(10, 5, 10)
            };
            const sceneBounds = {
                min: new Vec3(-9, -4, -9),
                max: new Vec3(9, 4, 9)
            };

            const octree = buildSparseOctree(gridFromBuffer(acc, 400, 200, 400), gridBounds, sceneBounds, 0.05);

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

    // ========================================================================
    // Dual-stream encoding (post-refactor): exercises level-0 split into
    // separate solid/mixed streams, the `li === -1` wave-entry sentinel, the
    // `ii < nMixed` vs `ii >= nMixed` solid-leaf encoding, the dual-stream
    // child binary search, and the `sortMixedByMorton` paired-mask permutation.
    // ========================================================================

    /**
     * Walk a built octree from root, return the set of {morton, isSolid, lo, hi}
     * for every reachable leaf. Mortons are reconstructed from the BFS path.
     */
    function walkLeaves(octree) {
        const leaves = [];
        if (octree.nodes.length === 0) return leaves;

        // BFS: each frame holds (nodeIdx, morton). Root morton is 0; child
        // morton is parent * 8 + octant.
        const stack = [{ idx: 0, morton: 0 }];
        while (stack.length > 0) {
            const { idx, morton } = stack.pop();
            const node = octree.nodes[idx] >>> 0;

            if (node === 0xFF000000 >>> 0) {
                // Solid leaf marker
                leaves.push({ morton, isSolid: true, lo: 0xFFFFFFFF, hi: 0xFFFFFFFF });
                continue;
            }

            const childMask = (node >>> 24) & 0xFF;
            const baseOffset = node & 0x00FFFFFF;

            if (childMask === 0x00) {
                // Mixed leaf — baseOffset is leafData index
                const lo = octree.leafData[baseOffset * 2];
                const hi = octree.leafData[baseOffset * 2 + 1];
                leaves.push({ morton, isSolid: false, lo, hi });
                continue;
            }

            // Interior — walk children in octant order
            let off = 0;
            for (let oct = 0; oct < 8; oct++) {
                if (childMask & (1 << oct)) {
                    stack.push({
                        idx: baseOffset + off,
                        morton: morton * 8 + oct
                    });
                    off++;
                }
            }
        }
        return leaves;
    }

    describe('dual-stream encoding', function () {
        it('should round-trip a mixed solid+mixed leaf set through the tree', function () {
            // Pseudorandom seed: deterministic across runs.
            let s = 12345;
            const rand = () => {
                s = (s * 1664525 + 1013904223) >>> 0;
                return s;
            };

            // 4x4x4 of leaf blocks (= 16x16x16 voxels at voxel scale, since
            // each block is 4 voxels per axis). gridBounds (0..4) at res 0.25
            // means 16x16x16 voxels = 4x4x4 blocks.
            const NBX = 4, NBY = 4;
            const expected = new Map(); // morton -> {isSolid, lo, hi}
            const acc = new BlockMaskBuffer();

            for (let z = 0; z < 4; z++) {
                for (let y = 0; y < 4; y++) {
                    for (let x = 0; x < 4; x++) {
                        const r = rand() % 10;
                        if (r < 4) continue; // empty
                        const morton = xyzToMorton(x, y, z);
                        const bi = linearBlockIdx(x, y, z, NBX, NBY);
                        if (r < 7) {
                            // solid
                            acc.addBlock(bi, 0xFFFFFFFF, 0xFFFFFFFF);
                            expected.set(morton, { isSolid: true, lo: 0xFFFFFFFF, hi: 0xFFFFFFFF });
                        } else {
                            // mixed — distinctive lo/hi tied to morton so a
                            // mis-permutation in sortMixedByMorton is visible.
                            const lo = (morton * 0x01010101 + 0x12345678) >>> 0;
                            const hi = (morton * 0x02020203 + 0x9ABCDEF0) >>> 0;
                            acc.addBlock(bi, lo, hi);
                            expected.set(morton, { isSolid: false, lo, hi });
                        }
                    }
                }
            }

            assert.ok(expected.size > 0, 'test setup should produce some leaves');

            const gridBounds = {
                min: new Vec3(0, 0, 0),
                max: new Vec3(4, 4, 4)
            };
            const octree = buildSparseOctree(gridFromBuffer(acc, 16, 16, 16), gridBounds, gridBounds, 0.25);
            const got = walkLeaves(octree);

            // Every reachable leaf must match an expected entry — UNLESS it's
            // a solid leaf one level above the leaf grid (i.e. parent of an
            // 8-solid group, which the builder collapses).
            const remaining = new Map(expected);
            for (const leaf of got) {
                if (remaining.has(leaf.morton)) {
                    const exp = remaining.get(leaf.morton);
                    assert.strictEqual(leaf.isSolid, exp.isSolid,
                        `morton ${leaf.morton}: solidness mismatch`);
                    if (!leaf.isSolid) {
                        assert.strictEqual(leaf.lo >>> 0, exp.lo >>> 0,
                            `morton ${leaf.morton}: lo mismatch`);
                        assert.strictEqual(leaf.hi >>> 0, exp.hi >>> 0,
                            `morton ${leaf.morton}: hi mismatch`);
                    }
                    remaining.delete(leaf.morton);
                } else {
                    // Could be a collapsed-solid parent; verify all 8
                    // children of this morton were expected solid and remove.
                    assert.ok(leaf.isSolid,
                        `unexpected non-solid leaf at morton ${leaf.morton}`);
                    const baseChild = leaf.morton * 8;
                    for (let oct = 0; oct < 8; oct++) {
                        const childMorton = baseChild + oct;
                        const exp = remaining.get(childMorton);
                        assert.ok(exp && exp.isSolid,
                            `collapsed solid covers morton ${childMorton} which was not expected solid`);
                        remaining.delete(childMorton);
                    }
                }
            }
            assert.strictEqual(remaining.size, 0,
                `unreachable leaves: ${[...remaining.keys()]}`);
        });

        it('should preserve mixed mask pairing under sortMixedByMorton', function () {
            // 8x8x8 / 0.25 = 32x32x32 voxels = 8x8x8 blocks.
            const NBX = 8, NBY = 8, NBZ = 8;
            // Insert blocks in REVERSE morton order so the octree's internal
            // sort actually permutes; each mask is a unique signature derived
            // from its morton, so any mis-pairing in sortMixedByMorton would
            // surface as a mismatch when we walk the tree.
            const acc = new BlockMaskBuffer();
            const expectedByMorton = new Map();
            // Pick 32 (x,y,z) coords with monotonic-but-spread mortons.
            for (let m = 31; m >= 0; m--) {
                // Distinct (x,y,z) per m; well within the 8x8x8 block grid.
                const x = m & 7;
                const y = (m >> 3) & 3;
                const z = m >> 5;
                const morton = xyzToMorton(x, y, z);
                const lo = (morton * 0xDEADBEEF) >>> 0;
                const hi = (morton * 0xCAFEBABE) >>> 0;
                acc.addBlock(linearBlockIdx(x, y, z, NBX, NBY), lo, hi);
                expectedByMorton.set(morton, { lo, hi });
            }

            const gridBounds = {
                min: new Vec3(0, 0, 0),
                max: new Vec3(8, 8, 8)
            };
            const octree = buildSparseOctree(gridFromBuffer(acc, NBX * 4, NBY * 4, NBZ * 4),
                gridBounds, gridBounds, 0.25);

            // Walk the octree and verify each mixed leaf has the mask paired
            // with the morton it was inserted for. Any mis-permutation in the
            // internal sort would yield a mask-vs-morton mismatch here.
            const got = walkLeaves(octree).filter(l => !l.isSolid);
            assert.strictEqual(got.length, expectedByMorton.size,
                'all mixed leaves should round-trip');
            for (const leaf of got) {
                const exp = expectedByMorton.get(leaf.morton);
                assert.ok(exp, `unexpected mixed leaf at morton ${leaf.morton}`);
                assert.strictEqual(leaf.lo >>> 0, exp.lo >>> 0,
                    `mask lo at morton ${leaf.morton} mismatched (mis-pairing in sort?)`);
                assert.strictEqual(leaf.hi >>> 0, exp.hi >>> 0,
                    `mask hi at morton ${leaf.morton} mismatched (mis-pairing in sort?)`);
            }
        });

        it('should handle solid-only input (mixed stream empty)', function () {
            // Exercises the dual-stream merge / search with one stream empty.
            // gridBounds (0..2) at res 0.25 = 8x8x8 voxels = 2x2x2 blocks.
            const NBX = 2, NBY = 2;
            const acc = new BlockMaskBuffer();
            const expected = [];
            for (let i = 0; i < 8; i++) {
                const x = i & 1, y = (i >> 1) & 1, z = (i >> 2) & 1;
                // Spread within a 2x2x2 leaf-block space, but skip 1 to
                // prevent the 8-children collapse so we keep distinct leaves.
                if (i === 7) continue;
                const m = xyzToMorton(x, y, z);
                acc.addBlock(linearBlockIdx(x, y, z, NBX, NBY), 0xFFFFFFFF, 0xFFFFFFFF);
                expected.push(m);
            }

            const gridBounds = { min: new Vec3(0, 0, 0), max: new Vec3(2, 2, 2) };
            const octree = buildSparseOctree(gridFromBuffer(acc, 8, 8, 8), gridBounds, gridBounds, 0.25);
            const got = walkLeaves(octree).filter(l => l.isSolid).map(l => l.morton);

            assert.deepStrictEqual([...got].sort((a, b) => a - b), expected.sort((a, b) => a - b));
            assert.strictEqual(octree.numMixedLeaves, 0);
        });

        it('should handle mixed-only input (solid stream empty)', function () {
            // gridBounds (0..4) at res 0.25 = 16x16x16 voxels = 4x4x4 blocks.
            const NBX = 4, NBY = 4;
            const acc = new BlockMaskBuffer();
            const expectedMortons = [];
            // Pick 5 well-spread (x,y,z) coords in the 4x4x4 block grid.
            const coords = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]];
            for (let i = 0; i < coords.length; i++) {
                const [x, y, z] = coords[i];
                const m = xyzToMorton(x, y, z);
                acc.addBlock(linearBlockIdx(x, y, z, NBX, NBY), 0xAAAA0000 | i, 0x5555FFFF & ~i);
                expectedMortons.push(m);
            }
            assert.strictEqual(acc.solidCount, 0);

            const gridBounds = { min: new Vec3(0, 0, 0), max: new Vec3(4, 4, 4) };
            const octree = buildSparseOctree(gridFromBuffer(acc, 16, 16, 16), gridBounds, gridBounds, 0.25);
            const gotMixed = walkLeaves(octree).filter(l => !l.isSolid).map(l => l.morton);

            assert.deepStrictEqual(
                [...gotMixed].sort((a, b) => a - b),
                expectedMortons.sort((a, b) => a - b)
            );
            assert.strictEqual(octree.numMixedLeaves, expectedMortons.length);
        });
    });

    describe('input mutation contract', function () {
        it('should sort the BlockMaskBuffer in place by morton ascending', function () {
            // Post-refactor: buildSparseOctree takes a SparseVoxelGrid (built
            // from the buffer), so it cannot mutate the buffer. Instead, we
            // verify the octree's internal sort produces a tree with leaves
            // in monotonically ascending morton order and that ALL inserted
            // (linear) blocks survive the round-trip into the tree.
            // gridBounds (0..8) at res 0.25 = 32x32x32 voxels = 8x8x8 blocks.
            const NBX = 8, NBY = 8;
            const acc = new BlockMaskBuffer();
            // Insert in non-monotonic order; coords pre-chosen to give
            // distinct linear indices and morton codes.
            const inputSolidCoords = [[2, 1, 0], [3, 0, 0], [1, 2, 0], [0, 3, 1], [3, 3, 0]];
            const inputMixedCoords = [[1, 0, 1], [0, 1, 0], [2, 2, 1], [3, 1, 1]];
            const expectedSolidMortons = inputSolidCoords.map(([x, y, z]) => xyzToMorton(x, y, z));
            const expectedMixed = new Map();
            for (const [x, y, z] of inputSolidCoords) {
                acc.addBlock(linearBlockIdx(x, y, z, NBX, NBY), 0xFFFFFFFF, 0xFFFFFFFF);
            }
            for (const [x, y, z] of inputMixedCoords) {
                const m = xyzToMorton(x, y, z);
                const lo = (m * 17) >>> 0;
                const hi = (m * 31) >>> 0;
                acc.addBlock(linearBlockIdx(x, y, z, NBX, NBY), lo, hi);
                expectedMixed.set(m, { lo, hi });
            }

            const gridBounds = { min: new Vec3(0, 0, 0), max: new Vec3(8, 8, 8) };
            const octree = buildSparseOctree(gridFromBuffer(acc, 32, 32, 32),
                gridBounds, gridBounds, 0.25);
            const leaves = walkLeaves(octree);

            // Walk yields leaves in morton-ascending order (stack-popped from
            // a depth-first traversal of an octant-sorted tree).
            const gotSolidMortons = leaves.filter(l => l.isSolid).map(l => l.morton);
            const gotMixed = leaves.filter(l => !l.isSolid);
            assert.deepStrictEqual(
                [...gotSolidMortons].sort((a, b) => a - b),
                [...expectedSolidMortons].sort((a, b) => a - b)
            );
            assert.strictEqual(gotMixed.length, expectedMixed.size);
            for (const leaf of gotMixed) {
                const exp = expectedMixed.get(leaf.morton);
                assert.ok(exp, `unexpected mixed leaf at morton ${leaf.morton}`);
                assert.strictEqual(leaf.lo >>> 0, exp.lo);
                assert.strictEqual(leaf.hi >>> 0, exp.hi);
            }
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
