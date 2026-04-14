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

            const blockIdx0 = 0;
            const emptyFilter = new Set([999]);

            const withoutFilter = gaussianContributesToVoxels(0, gaussianCols, grid, lookup, 1e-6);
            const withEmptyFilter = gaussianContributesToVoxels(0, gaussianCols, grid, lookup, 1e-6, emptyFilter);

            assert.strictEqual(withoutFilter, true, 'Should contribute without filter');
            assert.strictEqual(withEmptyFilter, false, 'Should not contribute with restrictive filter');
        });
    });
});
