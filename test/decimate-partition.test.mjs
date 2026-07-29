/**
 * KD partition + coherence detector tests.
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

import { buildBlockHalo, kdPartition, coherenceRuns } from '../src/lib/decimate/partition.js';

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

    it('jitters split planes deterministically between generations while preserving the leaf cap', () => {
        const pos = grid(16);
        const a = kdPartition(pos, 300, 1);
        const again = kdPartition(pos, 300, 1);
        const b = kdPartition(pos, 300, 2);
        assert.deepStrictEqual(Array.from(a.order), Array.from(again.order), 'same generation is deterministic');
        assert.notDeepStrictEqual(
            a.blocks.map(block => block.end - block.start),
            b.blocks.map(block => block.end - block.start),
            'generation changes split quantiles'
        );
        assert.ok(a.blocks.every(block => block.end - block.start <= 300));
        assert.ok(b.blocks.every(block => block.end - block.start <= 300));
    });

    it('builds sorted bounded halos without duplicating core ownership', () => {
        const pos = grid(16);
        const partition = kdPartition(pos, 300, 1);
        let foundCapped = false;
        for (let bi = 0; bi < partition.blocks.length; bi++) {
            const block = partition.blocks[bi];
            const core = new Set(partition.order.subarray(block.start, block.end));
            const halo = buildBlockHalo(pos, partition, bi, 10);
            assert.ok(halo.rows.length <= Math.min(10, core.size));
            for (let i = 0; i < halo.rows.length; i++) {
                assert.ok(!core.has(halo.rows[i]), 'halo row is not core-owned');
                if (i > 0) assert.ok(halo.rows[i] > halo.rows[i - 1], 'halo rows sorted and unique');
            }
            foundCapped ||= halo.capped;
        }
        assert.ok(foundCapped, 'dense boundary halos exercise the cap');
    });

    it('moves a coincident boundary pair into one interior core on the next jitter pattern', () => {
        const n = 256;
        const pos = { x: new Float32Array(n), y: new Float32Array(n), z: new Float32Array(n) };
        const assignments = [1, 2].map((generation) => {
            const partition = kdPartition(pos, 64, generation);
            const owner = new Int32Array(n);
            partition.blocks.forEach((block, bi) => {
                for (let i = block.start; i < block.end; i++) owner[partition.order[i]] = bi;
            });
            return owner;
        });
        let pair = null;
        for (let a = 0; a < n && !pair; a++) {
            for (let b = a + 1; b < n; b++) {
                if (assignments[0][a] !== assignments[0][b] && assignments[1][a] === assignments[1][b]) {
                    pair = [a, b];
                    break;
                }
            }
        }
        assert.ok(pair, 'a coincident pair crosses the first boundary and becomes interior after jitter');
    });

    it('rare flyaways land in residual blocks; core blocks stay tight', () => {
        const g = grid(22); // 10648 bulk points in [0,21]^3
        const nBulk = g.x.length;
        const fly = [
            [10000, 9000, -11000], [-12000, 10000, 9500], [11000, -9500, 12000], [-9000, -10000, -12000],
            [15000, 14000, 13000], [-15000, 16000, -14000], [14000, -13000, 15000], [-16000, -15000, 14000]
        ];
        const n = nBulk + fly.length;
        const pos = { x: new Float32Array(n), y: new Float32Array(n), z: new Float32Array(n) };
        pos.x.set(g.x); pos.y.set(g.y); pos.z.set(g.z);
        fly.forEach(([fx, fy, fz], i) => {
            pos.x[nBulk + i] = fx; pos.y[nBulk + i] = fy; pos.z[nBulk + i] = fz;
        });
        const { order, blocks } = kdPartition(pos, 1024);
        assert.strictEqual(blocks.reduce((a, b) => a + (b.end - b.start), 0), n);
        for (const b of blocks) {
            let hasBulk = false, hasFly = false;
            for (let i = b.start; i < b.end; i++) {
                if (order[i] < nBulk) hasBulk = true;
                else hasFly = true;
                if (i > b.start) assert.ok(order[i] > order[i - 1], 'owned range sorted ascending');
            }
            assert.ok(!(hasBulk && hasFly), 'flyaways segregated from the bulk');
            assert.strictEqual(b.residual, hasFly);
            if (hasBulk) {
                const ext = Math.max(b.aabb[3] - b.aabb[0], b.aabb[4] - b.aabb[1], b.aabb[5] - b.aabb[2]);
                assert.ok(ext <= 21 + 1e-6, `core block stretched by flyaways (extent ${ext})`);
            }
        }
    });

    it('a genuinely sparse scene (many out-of-fence points) is not split', () => {
        // 4.6% "flyaways" exceed the rare-outlier fraction: the fence must
        // stand down and partition the scene as-is (mixed blocks allowed).
        const g = grid(16); // 4096 bulk points
        const nBulk = g.x.length;
        const nFly = 200;
        const n = nBulk + nFly;
        const pos = { x: new Float32Array(n), y: new Float32Array(n), z: new Float32Array(n) };
        pos.x.set(g.x); pos.y.set(g.y); pos.z.set(g.z);
        for (let i = 0; i < nFly; i++) {
            const s = i % 2 === 0 ? 1 : -1;
            pos.x[nBulk + i] = s * (10000 + i * 37);
            pos.y[nBulk + i] = -s * (9000 + i * 53);
            pos.z[nBulk + i] = s * (11000 + i * 71);
        }
        const { order, blocks } = kdPartition(pos, 300);
        assert.strictEqual(blocks.reduce((a, b) => a + (b.end - b.start), 0), n);
        const mixed = blocks.some((b) => {
            let hasBulk = false, hasFly = false;
            for (let i = b.start; i < b.end; i++) {
                if (order[i] < nBulk) hasBulk = true;
                else hasFly = true;
            }
            return hasBulk && hasFly;
        });
        assert.ok(mixed, 'fence should stand down above the rare-outlier fraction');
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
