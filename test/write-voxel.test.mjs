/**
 * Tests for voxel writer file contracts.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';

import { Vec3 } from 'playcanvas';

import { MemoryFileSystem } from '../src/lib/io/write/index.js';
import { writeOctreeFiles } from '../src/lib/writers/write-voxel.js';

describe('writeOctreeFiles', function () {
    it('writes metadata and little-endian nodes followed by leafData', async function () {
        const fs = new MemoryFileSystem();
        const octree = {
            gridBounds: {
                min: new Vec3(0, 1, 2),
                max: new Vec3(3, 4, 5)
            },
            sceneBounds: {
                min: new Vec3(-1, -2, -3),
                max: new Vec3(6, 7, 8)
            },
            voxelResolution: 0.25,
            leafSize: 4,
            treeDepth: 2,
            numInteriorNodes: 1,
            numMixedLeaves: 1,
            nodes: new Uint32Array([0x11223344, 0xAABBCCDD]),
            leafData: new Uint32Array([0x01020304, 0xFFEEDDCC])
        };

        await writeOctreeFiles(fs, 'scene.voxel.json', octree);

        const jsonBytes = fs.results.get('scene.voxel.json');
        assert.ok(jsonBytes, 'metadata file should be written');
        const metadata = JSON.parse(new TextDecoder().decode(jsonBytes));
        assert.strictEqual(metadata.nodeCount, 2);
        assert.strictEqual(metadata.leafDataCount, 2);
        assert.deepStrictEqual(metadata.gridBounds.min, [0, 1, 2]);
        assert.deepStrictEqual(metadata.sceneBounds.max, [6, 7, 8]);

        const binBytes = fs.results.get('scene.voxel.bin');
        assert.ok(binBytes, 'binary file should be written');
        assert.strictEqual(binBytes.byteLength, 16);
        assert.deepStrictEqual([...binBytes], [
            0x44, 0x33, 0x22, 0x11,
            0xDD, 0xCC, 0xBB, 0xAA,
            0x04, 0x03, 0x02, 0x01,
            0xCC, 0xDD, 0xEE, 0xFF
        ]);
    });
});
