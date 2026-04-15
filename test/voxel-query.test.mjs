/**
 * Tests for voxel query functions (buildBlockLookup, isCenterInOccupiedVoxel,
 * gaussianContributesToVoxels) used by filterCluster and filterFloaters.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { BlockMaskBuffer } from '../src/lib/voxel/block-mask-buffer.js';
import { xyzToMorton } from '../src/lib/voxel/morton.js';
import {
    buildBlockLookup,
    isCenterInOccupiedVoxel,
    gaussianContributesToVoxels
} from '../src/lib/voxel/voxel-query.js';

const SOLID_LO = 0xFFFFFFFF >>> 0;
const SOLID_HI = 0xFFFFFFFF >>> 0;

function makeGrid(numBlocksX, numBlocksY, numBlocksZ, voxelResolution) {
    const blockSize = 4 * voxelResolution;
    return {
        gridMinX: 0,
        gridMinY: 0,
        gridMinZ: 0,
        blockSize,
        voxelResolution,
        numBlocksX,
        numBlocksY,
        numBlocksZ,
        strideY: numBlocksX,
        strideZ: numBlocksX * numBlocksY
    };
}

function makeGridWithOrigin(gridMinX, gridMinY, gridMinZ, numBlocksX, numBlocksY, numBlocksZ, voxelResolution) {
    const blockSize = 4 * voxelResolution;
    return {
        gridMinX,
        gridMinY,
        gridMinZ,
        blockSize,
        voxelResolution,
        numBlocksX,
        numBlocksY,
        numBlocksZ,
        strideY: numBlocksX,
        strideZ: numBlocksX * numBlocksY
    };
}

function voxelBit(lx, ly, lz) {
    const bitIdx = lx + ly * 4 + lz * 16;
    if (bitIdx < 32) return [1 << bitIdx, 0];
    return [0, 1 << (bitIdx - 32)];
}

function voxelMask(...positions) {
    let lo = 0, hi = 0;
    for (const [lx, ly, lz] of positions) {
        const [blo, bhi] = voxelBit(lx, ly, lz);
        lo |= blo;
        hi |= bhi;
    }
    return [lo, hi];
}

describe('voxel-query', function () {
    describe('buildBlockLookup', function () {
        it('should index solid blocks correctly', function () {
            const buffer = new BlockMaskBuffer();
            buffer.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);
            buffer.addBlock(xyzToMorton(2, 1, 0), SOLID_LO, SOLID_HI);

            const strideY = 4;
            const strideZ = 4 * 4;
            const lookup = buildBlockLookup(buffer, strideY, strideZ);

            assert.strictEqual(lookup.solidSet.size, 2);
            assert.ok(lookup.solidSet.has(0 + 0 * strideY + 0 * strideZ));
            assert.ok(lookup.solidSet.has(2 + 1 * strideY + 0 * strideZ));
            assert.strictEqual(lookup.mixedMap.size, 0);
        });

        it('should index mixed blocks correctly', function () {
            const buffer = new BlockMaskBuffer();
            const lo = 0x0000000F >>> 0;
            const hi = 0;
            buffer.addBlock(xyzToMorton(1, 1, 1), lo, hi);

            const strideY = 4;
            const strideZ = 16;
            const lookup = buildBlockLookup(buffer, strideY, strideZ);

            assert.strictEqual(lookup.solidSet.size, 0);
            assert.strictEqual(lookup.mixedMap.size, 1);
            const blockIdx = 1 + 1 * strideY + 1 * strideZ;
            assert.ok(lookup.mixedMap.has(blockIdx));
            const maskIdx = lookup.mixedMap.get(blockIdx);
            assert.strictEqual(lookup.masks[maskIdx * 2], lo);
            assert.strictEqual(lookup.masks[maskIdx * 2 + 1], hi);
        });

        it('should handle empty buffer', function () {
            const buffer = new BlockMaskBuffer();
            const lookup = buildBlockLookup(buffer, 4, 16);

            assert.strictEqual(lookup.solidSet.size, 0);
            assert.strictEqual(lookup.mixedMap.size, 0);
        });
    });

    describe('isCenterInOccupiedVoxel', function () {
        it('should return true for point inside solid block', function () {
            const buffer = new BlockMaskBuffer();
            buffer.addBlock(xyzToMorton(1, 1, 1), SOLID_LO, SOLID_HI);

            const voxelRes = 0.25;
            const grid = makeGrid(4, 4, 4, voxelRes);
            const lookup = buildBlockLookup(buffer, grid.strideY, grid.strideZ);

            const result = isCenterInOccupiedVoxel(
                1.0 * grid.blockSize + 0.5 * voxelRes,
                1.0 * grid.blockSize + 0.5 * voxelRes,
                1.0 * grid.blockSize + 0.5 * voxelRes,
                grid, lookup
            );
            assert.strictEqual(result, true);
        });

        it('should return false for point in empty block', function () {
            const buffer = new BlockMaskBuffer();
            buffer.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);

            const voxelRes = 0.25;
            const grid = makeGrid(4, 4, 4, voxelRes);
            const lookup = buildBlockLookup(buffer, grid.strideY, grid.strideZ);

            const result = isCenterInOccupiedVoxel(
                2.0 * grid.blockSize + 0.5 * voxelRes,
                2.0 * grid.blockSize + 0.5 * voxelRes,
                2.0 * grid.blockSize + 0.5 * voxelRes,
                grid, lookup
            );
            assert.strictEqual(result, false);
        });

        it('should return false for point outside grid', function () {
            const buffer = new BlockMaskBuffer();
            buffer.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);

            const voxelRes = 0.25;
            const grid = makeGrid(4, 4, 4, voxelRes);
            const lookup = buildBlockLookup(buffer, grid.strideY, grid.strideZ);

            assert.strictEqual(isCenterInOccupiedVoxel(-1, 0, 0, grid, lookup), false);
            assert.strictEqual(isCenterInOccupiedVoxel(100, 0, 0, grid, lookup), false);
        });

        it('should respect blockFilter when provided', function () {
            const buffer = new BlockMaskBuffer();
            buffer.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);
            buffer.addBlock(xyzToMorton(1, 0, 0), SOLID_LO, SOLID_HI);

            const voxelRes = 0.25;
            const grid = makeGrid(4, 4, 4, voxelRes);
            const lookup = buildBlockLookup(buffer, grid.strideY, grid.strideZ);

            const blockFilter = new Set([0]);

            assert.strictEqual(
                isCenterInOccupiedVoxel(0.5 * voxelRes, 0.5 * voxelRes, 0.5 * voxelRes, grid, lookup, blockFilter),
                true,
                'Block 0 is in the filter'
            );

            assert.strictEqual(
                isCenterInOccupiedVoxel(1.0 * grid.blockSize + 0.5 * voxelRes, 0.5 * voxelRes, 0.5 * voxelRes, grid, lookup, blockFilter),
                false,
                'Block 1 is not in the filter'
            );
        });

        it('should correctly test mixed block voxels', function () {
            const buffer = new BlockMaskBuffer();
            const lo = 1 >>> 0;
            const hi = 0;
            buffer.addBlock(xyzToMorton(0, 0, 0), lo, hi);

            const voxelRes = 1.0;
            const grid = makeGrid(2, 2, 2, voxelRes);
            const lookup = buildBlockLookup(buffer, grid.strideY, grid.strideZ);

            assert.strictEqual(
                isCenterInOccupiedVoxel(0.5, 0.5, 0.5, grid, lookup),
                true,
                'Voxel (0,0,0) has bit 0 set'
            );

            assert.strictEqual(
                isCenterInOccupiedVoxel(1.5, 0.5, 0.5, grid, lookup),
                false,
                'Voxel (1,0,0) has bit 1 unset'
            );
        });
    });

    describe('gaussianContributesToVoxels', function () {
        it('should detect contribution from a Gaussian centered on an occupied voxel', function () {
            const buffer = new BlockMaskBuffer();
            buffer.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);

            const voxelRes = 1.0;
            const grid = makeGrid(2, 2, 2, voxelRes);
            const lookup = buildBlockLookup(buffer, grid.strideY, grid.strideZ);

            const gaussianCols = {
                posX: new Float32Array([2.0]),
                posY: new Float32Array([2.0]),
                posZ: new Float32Array([2.0]),
                rotW: new Float32Array([1.0]),
                rotX: new Float32Array([0.0]),
                rotY: new Float32Array([0.0]),
                rotZ: new Float32Array([0.0]),
                scaleX: new Float32Array([Math.log(2.0)]),
                scaleY: new Float32Array([Math.log(2.0)]),
                scaleZ: new Float32Array([Math.log(2.0)]),
                opacity: new Float32Array([5.0]),
                extentX: new Float32Array([6.0]),
                extentY: new Float32Array([6.0]),
                extentZ: new Float32Array([6.0])
            };

            const result = gaussianContributesToVoxels(0, gaussianCols, grid, lookup, 1e-6);
            assert.strictEqual(result, true, 'Gaussian near solid block should contribute');
        });

        it('should not detect contribution from a distant Gaussian', function () {
            const buffer = new BlockMaskBuffer();
            buffer.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);

            const voxelRes = 1.0;
            const grid = makeGrid(2, 2, 2, voxelRes);
            const lookup = buildBlockLookup(buffer, grid.strideY, grid.strideZ);

            const gaussianCols = {
                posX: new Float32Array([100.0]),
                posY: new Float32Array([100.0]),
                posZ: new Float32Array([100.0]),
                rotW: new Float32Array([1.0]),
                rotX: new Float32Array([0.0]),
                rotY: new Float32Array([0.0]),
                rotZ: new Float32Array([0.0]),
                scaleX: new Float32Array([Math.log(0.1)]),
                scaleY: new Float32Array([Math.log(0.1)]),
                scaleZ: new Float32Array([Math.log(0.1)]),
                opacity: new Float32Array([5.0]),
                extentX: new Float32Array([0.3]),
                extentY: new Float32Array([0.3]),
                extentZ: new Float32Array([0.3])
            };

            const result = gaussianContributesToVoxels(0, gaussianCols, grid, lookup, 1e-6);
            assert.strictEqual(result, false, 'Distant Gaussian should not contribute');
        });

        it('should respect blockFilter', function () {
            const buffer = new BlockMaskBuffer();
            buffer.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);
            buffer.addBlock(xyzToMorton(1, 0, 0), SOLID_LO, SOLID_HI);

            const voxelRes = 1.0;
            const grid = makeGrid(4, 4, 4, voxelRes);
            const lookup = buildBlockLookup(buffer, grid.strideY, grid.strideZ);

            const gaussianCols = {
                posX: new Float32Array([2.0]),
                posY: new Float32Array([2.0]),
                posZ: new Float32Array([2.0]),
                rotW: new Float32Array([1.0]),
                rotX: new Float32Array([0.0]),
                rotY: new Float32Array([0.0]),
                rotZ: new Float32Array([0.0]),
                scaleX: new Float32Array([Math.log(2.0)]),
                scaleY: new Float32Array([Math.log(2.0)]),
                scaleZ: new Float32Array([Math.log(2.0)]),
                opacity: new Float32Array([5.0]),
                extentX: new Float32Array([8.0]),
                extentY: new Float32Array([8.0]),
                extentZ: new Float32Array([8.0])
            };

            const emptyFilter = new Set([999]);

            const withoutFilter = gaussianContributesToVoxels(0, gaussianCols, grid, lookup, 1e-6);
            const withEmptyFilter = gaussianContributesToVoxels(0, gaussianCols, grid, lookup, 1e-6, emptyFilter);

            assert.strictEqual(withoutFilter, true, 'Should contribute without filter');
            assert.strictEqual(withEmptyFilter, false, 'Should not contribute with restrictive filter');
        });
    });

    // ====================================================================
    // Negative grid origin tests
    // ====================================================================

    describe('isCenterInOccupiedVoxel with negative grid origin', function () {
        it('should correctly map world coords to block with negative origin', function () {
            // Grid: origin (-12,-8,-4), 8x6x4 blocks, voxelRes=1 (blockSize=4)
            const grid = makeGridWithOrigin(-12, -8, -4, 8, 6, 4, 1.0);
            const buffer = new BlockMaskBuffer();
            // Block at (2,1,0): world origin = (-12+8, -8+4, -4+0) = (-4, -4, -4)
            buffer.addBlock(xyzToMorton(2, 1, 0), SOLID_LO, SOLID_HI);
            const lookup = buildBlockLookup(buffer, grid.strideY, grid.strideZ);
            const blockIdx = 2 + 1 * grid.strideY + 0 * grid.strideZ;
            const filter = new Set([blockIdx]);

            // Center of voxel (0,0,0) in block (2,1,0) → world (-3.5, -3.5, -3.5)
            assert.strictEqual(
                isCenterInOccupiedVoxel(-3.5, -3.5, -3.5, grid, lookup, filter),
                true,
                'Voxel (0,0,0) of block (2,1,0) with neg origin'
            );

            // Voxel (3,3,3) in block (2,1,0) → world (-0.5, -0.5, -0.5)
            assert.strictEqual(
                isCenterInOccupiedVoxel(-0.5, -0.5, -0.5, grid, lookup, filter),
                true,
                'Voxel (3,3,3) of block (2,1,0) with neg origin'
            );

            // Just past the block boundary into block (3,1,0) → world (0.5, -3.5, -3.5)
            assert.strictEqual(
                isCenterInOccupiedVoxel(0.5, -3.5, -3.5, grid, lookup, filter),
                false,
                'Block (3,1,0) is not in the filter'
            );
        });

        it('should handle negative coords near grid origin boundary', function () {
            const grid = makeGridWithOrigin(-20, -20, -20, 10, 10, 10, 1.0);
            const buffer = new BlockMaskBuffer();
            // Block at (0,0,0): world origin = (-20,-20,-20), covers (-20,-16) in each axis
            buffer.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);
            const lookup = buildBlockLookup(buffer, grid.strideY, grid.strideZ);

            assert.strictEqual(
                isCenterInOccupiedVoxel(-19.5, -19.5, -19.5, grid, lookup),
                true,
                'First voxel of first block'
            );

            assert.strictEqual(
                isCenterInOccupiedVoxel(-20.5, -19.5, -19.5, grid, lookup),
                false,
                'Outside grid (below min)'
            );
        });
    });

    // ====================================================================
    // Separated clusters with blockFilter
    // ====================================================================

    describe('isCenterInOccupiedVoxel with separated clusters', function () {
        it('should exclude Gaussians in non-cluster island blocks', function () {
            // Cluster A: blocks (0,0,0) and (1,0,0)
            // Island B:  blocks (5,0,0) and (6,0,0)
            // Gap of 3 empty blocks between them
            const grid = makeGrid(10, 4, 4, 1.0);
            const buffer = new BlockMaskBuffer();
            buffer.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);
            buffer.addBlock(xyzToMorton(1, 0, 0), SOLID_LO, SOLID_HI);
            buffer.addBlock(xyzToMorton(5, 0, 0), SOLID_LO, SOLID_HI);
            buffer.addBlock(xyzToMorton(6, 0, 0), SOLID_LO, SOLID_HI);
            const lookup = buildBlockLookup(buffer, grid.strideY, grid.strideZ);

            // ccSet: only cluster A
            const ccSet = new Set([
                0 + 0 * grid.strideY + 0 * grid.strideZ,
                1 + 0 * grid.strideY + 0 * grid.strideZ
            ]);

            // Gaussian in cluster A → kept
            assert.strictEqual(
                isCenterInOccupiedVoxel(0.5, 0.5, 0.5, grid, lookup, ccSet),
                true,
                'Cluster A block 0'
            );
            assert.strictEqual(
                isCenterInOccupiedVoxel(4.5, 0.5, 0.5, grid, lookup, ccSet),
                true,
                'Cluster A block 1'
            );

            // Gaussian in island B → excluded
            assert.strictEqual(
                isCenterInOccupiedVoxel(20.5, 0.5, 0.5, grid, lookup, ccSet),
                false,
                'Island B block 5 should be excluded'
            );
            assert.strictEqual(
                isCenterInOccupiedVoxel(24.5, 0.5, 0.5, grid, lookup, ccSet),
                false,
                'Island B block 6 should be excluded'
            );

            // Gaussian at edge of island B closest to cluster A → still excluded
            assert.strictEqual(
                isCenterInOccupiedVoxel(20.1, 0.5, 0.5, grid, lookup, ccSet),
                false,
                'Near edge of island B should be excluded'
            );
        });

        it('should work with negative grid origin and separated clusters', function () {
            // Real-world scenario: grid centered around origin
            const grid = makeGridWithOrigin(-20, -12, -16, 10, 6, 8, 1.0);
            const buffer = new BlockMaskBuffer();
            // Cluster A at blocks (1,1,1) and (2,1,1)
            buffer.addBlock(xyzToMorton(1, 1, 1), SOLID_LO, SOLID_HI);
            buffer.addBlock(xyzToMorton(2, 1, 1), SOLID_LO, SOLID_HI);
            // Island B at blocks (7,1,1) and (8,1,1)
            buffer.addBlock(xyzToMorton(7, 1, 1), SOLID_LO, SOLID_HI);
            buffer.addBlock(xyzToMorton(8, 1, 1), SOLID_LO, SOLID_HI);
            const lookup = buildBlockLookup(buffer, grid.strideY, grid.strideZ);

            const ccSet = new Set([
                1 + 1 * grid.strideY + 1 * grid.strideZ,
                2 + 1 * grid.strideY + 1 * grid.strideZ
            ]);

            // Block (1,1,1): world origin (-20+4, -12+4, -16+4) = (-16, -8, -12)
            assert.strictEqual(
                isCenterInOccupiedVoxel(-15.5, -7.5, -11.5, grid, lookup, ccSet),
                true,
                'Cluster A with negative origin'
            );

            // Block (7,1,1): world origin (-20+28, -12+4, -16+4) = (8, -8, -12)
            assert.strictEqual(
                isCenterInOccupiedVoxel(8.5, -7.5, -11.5, grid, lookup, ccSet),
                false,
                'Island B with negative origin should be excluded'
            );
        });
    });

    // ====================================================================
    // Mixed-block voxel-level precision with blockFilter
    // ====================================================================

    describe('isCenterInOccupiedVoxel mixed-block precision', function () {
        it('should distinguish occupied vs empty voxels within a cluster block', function () {
            // Block has only a few specific voxels occupied
            const [lo, hi] = voxelMask([0, 0, 0], [1, 0, 0], [0, 1, 0]);
            const grid = makeGrid(4, 4, 4, 1.0);
            const buffer = new BlockMaskBuffer();
            buffer.addBlock(xyzToMorton(0, 0, 0), lo, hi);
            const lookup = buildBlockLookup(buffer, grid.strideY, grid.strideZ);
            const ccSet = new Set([0]);

            // Occupied voxel (0,0,0) → true
            assert.strictEqual(
                isCenterInOccupiedVoxel(0.5, 0.5, 0.5, grid, lookup, ccSet),
                true,
                'Occupied voxel (0,0,0)'
            );
            // Occupied voxel (1,0,0) → true
            assert.strictEqual(
                isCenterInOccupiedVoxel(1.5, 0.5, 0.5, grid, lookup, ccSet),
                true,
                'Occupied voxel (1,0,0)'
            );
            // Unoccupied voxel (2,0,0) in same block → false
            assert.strictEqual(
                isCenterInOccupiedVoxel(2.5, 0.5, 0.5, grid, lookup, ccSet),
                false,
                'Empty voxel (2,0,0) in cluster block'
            );
            // Unoccupied voxel (3,3,3) in same block → false
            assert.strictEqual(
                isCenterInOccupiedVoxel(3.5, 3.5, 3.5, grid, lookup, ccSet),
                false,
                'Empty voxel (3,3,3) in cluster block'
            );
        });

        it('should test all 64 voxel positions in a mixed block', function () {
            // Set specific pattern: checkerboard on lx (even lx = occupied)
            let lo = 0, hi = 0;
            for (let lz = 0; lz < 4; lz++) {
                for (let ly = 0; ly < 4; ly++) {
                    for (let lx = 0; lx < 4; lx++) {
                        if (lx % 2 === 0) {
                            const [blo, bhi] = voxelBit(lx, ly, lz);
                            lo |= blo;
                            hi |= bhi;
                        }
                    }
                }
            }
            const grid = makeGrid(2, 2, 2, 1.0);
            const buffer = new BlockMaskBuffer();
            buffer.addBlock(xyzToMorton(0, 0, 0), lo, hi);
            const lookup = buildBlockLookup(buffer, grid.strideY, grid.strideZ);

            for (let lz = 0; lz < 4; lz++) {
                for (let ly = 0; ly < 4; ly++) {
                    for (let lx = 0; lx < 4; lx++) {
                        const expected = lx % 2 === 0;
                        const result = isCenterInOccupiedVoxel(
                            lx + 0.5, ly + 0.5, lz + 0.5,
                            grid, lookup
                        );
                        assert.strictEqual(result, expected,
                            `Voxel (${lx},${ly},${lz}) expected ${expected}`
                        );
                    }
                }
            }
        });

        it('should handle hi-word voxels (lz=2,3) correctly', function () {
            // Only set voxels in lz=2 and lz=3 (hi word)
            const [lo, hi] = voxelMask([0, 0, 2], [3, 3, 3]);
            const grid = makeGrid(2, 2, 2, 1.0);
            const buffer = new BlockMaskBuffer();
            buffer.addBlock(xyzToMorton(0, 0, 0), lo, hi);
            const lookup = buildBlockLookup(buffer, grid.strideY, grid.strideZ);

            assert.strictEqual(
                isCenterInOccupiedVoxel(0.5, 0.5, 2.5, grid, lookup),
                true,
                'Voxel (0,0,2) in hi word'
            );
            assert.strictEqual(
                isCenterInOccupiedVoxel(3.5, 3.5, 3.5, grid, lookup),
                true,
                'Voxel (3,3,3) in hi word'
            );
            assert.strictEqual(
                isCenterInOccupiedVoxel(0.5, 0.5, 0.5, grid, lookup),
                false,
                'Voxel (0,0,0) in lo word should be empty'
            );
            assert.strictEqual(
                isCenterInOccupiedVoxel(0.5, 0.5, 1.5, grid, lookup),
                false,
                'Voxel (0,0,1) in lo word should be empty'
            );
        });
    });

    // ====================================================================
    // Block boundary precision
    // ====================================================================

    describe('isCenterInOccupiedVoxel block boundary', function () {
        it('should map positions exactly at block boundary to correct block', function () {
            const grid = makeGrid(4, 4, 4, 1.0);
            const buffer = new BlockMaskBuffer();
            // Block 0 and block 1 are both solid
            buffer.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);
            buffer.addBlock(xyzToMorton(1, 0, 0), SOLID_LO, SOLID_HI);
            const lookup = buildBlockLookup(buffer, grid.strideY, grid.strideZ);

            const blockFilter0 = new Set([0]);
            const blockFilter1 = new Set([1]);

            // Exactly at the boundary (x = blockSize = 4.0) → floor(4.0/4.0) = 1 → block 1
            assert.strictEqual(
                isCenterInOccupiedVoxel(4.0, 0.5, 0.5, grid, lookup, blockFilter0),
                false,
                'x=4.0 should be in block 1, not block 0'
            );
            assert.strictEqual(
                isCenterInOccupiedVoxel(4.0, 0.5, 0.5, grid, lookup, blockFilter1),
                true,
                'x=4.0 should be in block 1'
            );

            // Just below the boundary (epsilon before 4.0)
            assert.strictEqual(
                isCenterInOccupiedVoxel(3.999, 0.5, 0.5, grid, lookup, blockFilter0),
                true,
                'x=3.999 should be in block 0'
            );
            assert.strictEqual(
                isCenterInOccupiedVoxel(3.999, 0.5, 0.5, grid, lookup, blockFilter1),
                false,
                'x=3.999 should not be in block 1'
            );
        });

        it('should handle negative-origin block boundaries', function () {
            const grid = makeGridWithOrigin(-8, -8, -8, 4, 4, 4, 1.0);
            const buffer = new BlockMaskBuffer();
            buffer.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);
            buffer.addBlock(xyzToMorton(1, 0, 0), SOLID_LO, SOLID_HI);
            const lookup = buildBlockLookup(buffer, grid.strideY, grid.strideZ);

            const blockFilter0 = new Set([0]);
            const blockFilter1 = new Set([1]);

            // Boundary at x = -8 + 4 = -4.0
            assert.strictEqual(
                isCenterInOccupiedVoxel(-4.0, -7.5, -7.5, grid, lookup, blockFilter0),
                false,
                'x=-4.0 at neg boundary should be block 1, not block 0'
            );
            assert.strictEqual(
                isCenterInOccupiedVoxel(-4.0, -7.5, -7.5, grid, lookup, blockFilter1),
                true,
                'x=-4.0 at neg boundary should be block 1'
            );
            assert.strictEqual(
                isCenterInOccupiedVoxel(-4.001, -7.5, -7.5, grid, lookup, blockFilter0),
                true,
                'x=-4.001 should be block 0'
            );
        });
    });

    // ====================================================================
    // gaussianContributesToVoxels with separated clusters
    // ====================================================================

    describe('gaussianContributesToVoxels cluster isolation', function () {
        it('should exclude large Gaussian in island whose extent reaches cluster', function () {
            // Cluster at block (0,0,0), island at block (4,0,0)
            // Gap of 3 empty blocks between
            const grid = makeGrid(8, 4, 4, 1.0);
            const buffer = new BlockMaskBuffer();
            buffer.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);
            buffer.addBlock(xyzToMorton(4, 0, 0), SOLID_LO, SOLID_HI);
            const lookup = buildBlockLookup(buffer, grid.strideY, grid.strideZ);

            const ccSet = new Set([0]); // only cluster block

            // Large Gaussian centered in island block (4,0,0), with huge extent
            const gaussianCols = {
                posX: new Float32Array([18.0]),  // center of block 4 (4*4 + 2 = 18)
                posY: new Float32Array([2.0]),
                posZ: new Float32Array([2.0]),
                rotW: new Float32Array([1.0]),
                rotX: new Float32Array([0.0]),
                rotY: new Float32Array([0.0]),
                rotZ: new Float32Array([0.0]),
                scaleX: new Float32Array([Math.log(5.0)]),
                scaleY: new Float32Array([Math.log(5.0)]),
                scaleZ: new Float32Array([Math.log(5.0)]),
                opacity: new Float32Array([5.0]),
                extentX: new Float32Array([20.0]),
                extentY: new Float32Array([20.0]),
                extentZ: new Float32Array([20.0])
            };

            // With ccSet filter, even though AABB reaches block 0, the Gaussian
            // is far away so contribution should be negligible
            const result = gaussianContributesToVoxels(0, gaussianCols, grid, lookup, 0.1, ccSet);
            assert.strictEqual(result, false,
                'Island Gaussian should not contribute enough to cluster at minContribution=0.1'
            );
        });

        it('should include Gaussian close to cluster boundary with low threshold', function () {
            const grid = makeGrid(4, 4, 4, 1.0);
            const buffer = new BlockMaskBuffer();
            buffer.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);
            const lookup = buildBlockLookup(buffer, grid.strideY, grid.strideZ);

            // Gaussian just outside block 0, with extent reaching into it
            const gaussianCols = {
                posX: new Float32Array([5.0]),
                posY: new Float32Array([2.0]),
                posZ: new Float32Array([2.0]),
                rotW: new Float32Array([1.0]),
                rotX: new Float32Array([0.0]),
                rotY: new Float32Array([0.0]),
                rotZ: new Float32Array([0.0]),
                scaleX: new Float32Array([Math.log(3.0)]),
                scaleY: new Float32Array([Math.log(3.0)]),
                scaleZ: new Float32Array([Math.log(3.0)]),
                opacity: new Float32Array([5.0]),
                extentX: new Float32Array([10.0]),
                extentY: new Float32Array([10.0]),
                extentZ: new Float32Array([10.0])
            };

            const result = gaussianContributesToVoxels(0, gaussianCols, grid, lookup, 1e-6);
            assert.strictEqual(result, true,
                'Nearby Gaussian with low threshold should contribute'
            );
        });

        it('should respect minContribution threshold for nearby Gaussians', function () {
            const grid = makeGrid(4, 4, 4, 1.0);
            const buffer = new BlockMaskBuffer();
            buffer.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);
            const lookup = buildBlockLookup(buffer, grid.strideY, grid.strideZ);

            // Small Gaussian at moderate distance
            const gaussianCols = {
                posX: new Float32Array([8.0]),
                posY: new Float32Array([2.0]),
                posZ: new Float32Array([2.0]),
                rotW: new Float32Array([1.0]),
                rotX: new Float32Array([0.0]),
                rotY: new Float32Array([0.0]),
                rotZ: new Float32Array([0.0]),
                scaleX: new Float32Array([Math.log(1.0)]),
                scaleY: new Float32Array([Math.log(1.0)]),
                scaleZ: new Float32Array([Math.log(1.0)]),
                opacity: new Float32Array([5.0]),
                extentX: new Float32Array([5.0]),
                extentY: new Float32Array([5.0]),
                extentZ: new Float32Array([5.0])
            };

            const lowThreshold = gaussianContributesToVoxels(0, gaussianCols, grid, lookup, 1e-6);
            const highThreshold = gaussianContributesToVoxels(0, gaussianCols, grid, lookup, 0.5);

            assert.strictEqual(lowThreshold, true, 'Should pass with very low threshold');
            assert.strictEqual(highThreshold, false, 'Should fail with high threshold');
        });

        it('should handle gaussianContributesToVoxels with negative grid origin', function () {
            const grid = makeGridWithOrigin(-16, -16, -16, 8, 8, 8, 1.0);
            const buffer = new BlockMaskBuffer();
            // Cluster block at (1,1,1): world origin (-12,-12,-12)
            buffer.addBlock(xyzToMorton(1, 1, 1), SOLID_LO, SOLID_HI);
            const lookup = buildBlockLookup(buffer, grid.strideY, grid.strideZ);
            const blockIdx = 1 + 1 * grid.strideY + 1 * grid.strideZ;
            const ccSet = new Set([blockIdx]);

            // Gaussian far from cluster
            const gaussianCols = {
                posX: new Float32Array([10.0]),
                posY: new Float32Array([10.0]),
                posZ: new Float32Array([10.0]),
                rotW: new Float32Array([1.0]),
                rotX: new Float32Array([0.0]),
                rotY: new Float32Array([0.0]),
                rotZ: new Float32Array([0.0]),
                scaleX: new Float32Array([Math.log(1.0)]),
                scaleY: new Float32Array([Math.log(1.0)]),
                scaleZ: new Float32Array([Math.log(1.0)]),
                opacity: new Float32Array([5.0]),
                extentX: new Float32Array([30.0]),
                extentY: new Float32Array([30.0]),
                extentZ: new Float32Array([30.0])
            };

            const result = gaussianContributesToVoxels(0, gaussianCols, grid, lookup, 0.1, ccSet);
            assert.strictEqual(result, false,
                'Distant Gaussian should not pass with minContribution=0.1 even with large extent'
            );
        });
    });

});
