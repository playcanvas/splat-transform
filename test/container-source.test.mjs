/**
 * Tests for the lazy multi-LOD container source (the LCC/LCC2 backend): segments
 * are decoded on demand, LRU-cached, and stitched per LOD via concatSource.
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

import { Transform } from '../src/lib/index.js';
import { materializeToDataTable } from '../src/lib/compat/data-table.js';
import { containerSource } from '../src/lib/readers/container-source.js';
import { createChunkDataPool, createInMemoryChunkSource, POSITION_STRIDE } from '../src/lib/source/index.js';

// A position-only resident source whose x values are `xs` (y=z=0), chunked at
// `chunkSize` — stands in for a decoded sub-file.
const makePosSource = (xs, chunkSize) => {
    const n = xs.length;
    const chunks = [];
    for (let base = 0; base < n; base += chunkSize) {
        const c = Math.min(chunkSize, n - base);
        const ab = new ArrayBuffer(c * POSITION_STRIDE);
        const f = new Float32Array(ab);
        for (let i = 0; i < c; i++) f[i * 3] = xs[base + i];
        chunks.push(ab);
    }
    return createInMemoryChunkSource({
        numGaussians: n,
        chunkSize,
        shBands: 0,
        transform: Transform.IDENTITY,
        lodCounts: [n],
        position: [chunks]
    });
};

describe('containerSource', () => {
    it('stitches lazily-decoded segments into a multi-LOD source, decoding each once', async () => {
        const chunkSize = 4;
        // LOD 0: segments of 3 + 5 (straddle the chunk-4 boundary); LOD 1: one of 4.
        const segXs = [[10, 11, 12], [20, 21, 22, 23, 24], [30, 31, 32, 33]];
        const calls = [0, 0, 0];
        const seg = i => ({
            count: segXs[i].length,
            decode: () => {
                calls[i]++;
                return Promise.resolve(makePosSource(segXs[i], chunkSize));
            }
        });
        const segmentsByLod = [[seg(0), seg(1)], [seg(2)]];

        const pool = createChunkDataPool({ chunkSize });
        const src = await containerSource(segmentsByLod, pool, { cacheSize: 3 });

        assert.strictEqual(src.meta.numLods, 2);
        assert.deepStrictEqual([...src.meta.lodCounts], [8, 4]);
        assert.strictEqual(src.meta.numGaussians, 8);

        const dt = await materializeToDataTable(src, pool);

        // Flattened: LOD 0's segments in order, then LOD 1's.
        assert.strictEqual(dt.numRows, 12);
        assert.deepStrictEqual(
            [...dt.getColumnByName('x').data],
            [10, 11, 12, 20, 21, 22, 23, 24, 30, 31, 32, 33]
        );

        // Sequential reads decode each sub-file exactly once (caching works).
        assert.deepStrictEqual(calls, [1, 1, 1]);

        await src.close();
    });

    it('gathers across segments and LODs', async () => {
        const chunkSize = 4;
        const segXs = [[10, 11, 12], [20, 21, 22, 23, 24], [30, 31, 32, 33]];
        const seg = i => ({ count: segXs[i].length, decode: () => Promise.resolve(makePosSource(segXs[i], chunkSize)) });
        const segmentsByLod = [[seg(0), seg(1)], [seg(2)]];

        const pool = createChunkDataPool({ chunkSize });
        const src = await containerSource(segmentsByLod, pool);

        const layout = src.meta.layouts.position;
        const px = (cd, n) => { const f = new Float32Array(cd.data, 0, n * 3); return Array.from({ length: n }, (_, i) => f[i * 3]); };

        // LOD 0 is [10,11,12, 20,21,22,23,24]; gather a shuffled cross-segment subset.
        const order = new Uint32Array([7, 0, 4, 2]); // -> 24, 10, 21, 12
        const cd0 = pool.acquire('position', layout, order.length);
        await src.read({ indices: order, indexOffset: 0, count: order.length, lod: 0, position: cd0 });
        assert.deepStrictEqual(px(cd0, order.length), [24, 10, 21, 12]);
        cd0.release();

        // LOD 1 is [30,31,32,33]; gather from that structural LOD.
        const cd1 = pool.acquire('position', layout, 2);
        await src.read({ indices: new Uint32Array([3, 1]), indexOffset: 0, count: 2, lod: 1, position: cd1 });
        assert.deepStrictEqual(px(cd1, 2), [33, 31]);
        cd1.release();

        await src.close();
    });

    it('reads a single LOD directly by index', async () => {
        const chunkSize = 4;
        const seg = xs => ({ count: xs.length, decode: () => Promise.resolve(makePosSource(xs, chunkSize)) });
        const segmentsByLod = [[seg([1, 2])], [seg([7, 8, 9])]];

        const pool = createChunkDataPool({ chunkSize });
        const src = await containerSource(segmentsByLod, pool);

        // Read LOD 1's only chunk directly.
        const layout = src.meta.layouts.position;
        const cd = pool.acquire('position', layout, 3);
        await src.read({ chunkIndex: 0, lod: 1, position: cd });
        const f = new Float32Array(cd.data, 0, 3 * (layout.stride >> 2));
        assert.deepStrictEqual([f[0], f[3], f[6]], [7, 8, 9]);
        cd.release();

        await src.close();
    });
});
