import { describe, it } from 'node:test';
import assert from 'node:assert';

import { ABSENT, BlockIndexMap, SOLID } from '../src/lib/voxel/block-index-map.js';
import { BlockMaskMap } from '../src/lib/voxel/block-mask-map.js';
import { SparseVoxelGrid } from '../src/lib/voxel/sparse-voxel-grid.js';

const LARGE_KEYS = [
    0x7FFFFFFF,
    0x80000000,
    0xFFFFFFFF,
    0x100000000,
    0x100000001,
    0x10000000000 + 17
];

describe('large voxel block maps', function () {
    it('stores block states across the 2^31 and 2^32 boundaries', function () {
        const map = new BlockIndexMap(LARGE_KEYS.length);
        for (let i = 0; i < LARGE_KEYS.length; i++) {
            map.set(LARGE_KEYS[i], i === 0 ? SOLID : i);
        }

        assert.strictEqual(map.get(LARGE_KEYS[0]), SOLID);
        for (let i = 1; i < LARGE_KEYS.length; i++) {
            assert.strictEqual(map.get(LARGE_KEYS[i]), i);
        }
        assert.strictEqual(map.get(0x200000000), ABSENT);
    });

    it('stores, updates, and deletes masks across the 2^31 and 2^32 boundaries', function () {
        const map = new BlockMaskMap(4);
        for (let i = 0; i < LARGE_KEYS.length; i++) {
            map.set(LARGE_KEYS[i], 0x1000 + i, 0x2000 + i);
        }

        for (let i = 0; i < LARGE_KEYS.length; i++) {
            const slot = map.slot(LARGE_KEYS[i]);
            assert.strictEqual(map.keys[slot], LARGE_KEYS[i]);
            assert.strictEqual(map.lo[slot], 0x1000 + i);
            assert.strictEqual(map.hi[slot], 0x2000 + i);
        }

        map.set(0x100000000, 0xAAAA, 0xBBBB);
        let slot = map.slot(0x100000000);
        assert.strictEqual(map.lo[slot], 0xAAAA);
        assert.strictEqual(map.hi[slot], 0xBBBB);

        map.delete(0xFFFFFFFF);
        assert.strictEqual(map.has(0xFFFFFFFF), false);
        for (const key of LARGE_KEYS.filter(key => key !== 0xFFFFFFFF)) {
            assert.strictEqual(map.has(key), true);
        }

        slot = map.slot(0x100000000);
        assert.strictEqual(map.keys[slot], 0x100000000);
    });

    it('rejects mutable grids above the 32-bit block-index range', function () {
        assert.throws(
            () => new SparseVoxelGrid(65536 * 4, 65536 * 4, 8),
            /limit is 2\^32/
        );
    });
});
