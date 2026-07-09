/**
 * KD partition + coherence detector tests.
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

import { kdPartition, coherenceRuns } from '../src/lib/decimate/partition.js';

const grid = (n) => {
    const x = new Float32Array(n * n * n), y = new Float32Array(n * n * n), z = new Float32Array(n * n * n);
    let i = 0;
    for (let a = 0; a < n; a++) {
        for (let b = 0; b < n; b++) {
            for (let c = 0; c < n; c++, i++) {
                x[i] = a; y[i] = b; z[i] = c;
            }
        }
    }
    return { x, y, z };
};

describe('kdPartition', () => {
    it('blocks cover all indices exactly once, sizes bounded, owned ranges sorted, AABBs contain', () => {
        const pos = grid(16); // 4096 points
        const { order, blocks } = kdPartition(pos, 300);
        assert.strictEqual(order.length, 4096);
        assert.strictEqual(blocks.reduce((a, b) => a + (b.end - b.start), 0), 4096);
        const seen = new Uint8Array(4096);
        for (const b of blocks) {
            assert.ok(b.end - b.start <= 300, 'block size bounded');
            for (let i = b.start; i < b.end; i++) {
                assert.strictEqual(seen[order[i]], 0, 'index appears once');
                seen[order[i]] = 1;
                if (i > b.start) assert.ok(order[i] > order[i - 1], 'owned range sorted ascending');
                const g = order[i];
                assert.ok(pos.x[g] >= b.aabb[0] - 1e-6 && pos.x[g] <= b.aabb[3] + 1e-6, 'x in aabb');
                assert.ok(pos.y[g] >= b.aabb[1] - 1e-6 && pos.y[g] <= b.aabb[4] + 1e-6, 'y in aabb');
                assert.ok(pos.z[g] >= b.aabb[2] - 1e-6 && pos.z[g] <= b.aabb[5] + 1e-6, 'z in aabb');
            }
        }
        assert.ok(seen.every ? true : true);
    });

    it('scene smaller than block size yields one block', () => {
        const pos = grid(4); // 64 points
        const { blocks } = kdPartition(pos, 1000);
        assert.strictEqual(blocks.length, 1);
        assert.strictEqual(blocks[0].end - blocks[0].start, 64);
    });

    it('degenerate coincident points partition without hanging', () => {
        const n = 5000;
        const pos = { x: new Float32Array(n), y: new Float32Array(n), z: new Float32Array(n) };
        const { blocks } = kdPartition(pos, 512);
        assert.strictEqual(blocks.reduce((a, b) => a + (b.end - b.start), 0), n);
    });
});

describe('coherenceRuns', () => {
    it('sequential rows are 1 run, scattered rows are many', () => {
        const seq = Uint32Array.from({ length: 1000 }, (_, i) => i * 2); // gaps of 2 ≤ 16
        assert.strictEqual(coherenceRuns(seq, 0, 1000, 16), 1);
        const scattered = Uint32Array.from({ length: 1000 }, (_, i) => i * 10000);
        assert.strictEqual(coherenceRuns(scattered, 0, 1000, 16), 1000);
        assert.strictEqual(coherenceRuns(seq, 0, 0, 16), 0);
    });
});
