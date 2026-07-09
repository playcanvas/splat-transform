/**
 * KdTree tests — the engine-free kd-tree over raw column arrays, used by
 * worker tasks (which must not import DataTable/playcanvas), decimation
 * block KNN, and the k-means CPU assignment fallback.
 *
 * Includes regression coverage for the O(N^2) build hang on degenerate
 * inputs where many points share a coordinate (e.g. a splat with every
 * gaussian at the origin): the build must stay O(N log N), and
 * nearest-neighbour queries must remain correct when ties are present.
 */

import assert from 'node:assert';
import { performance } from 'node:perf_hooks';
import { describe, it } from 'node:test';

import { KdTree } from '../src/lib/spatial/kd-tree.js';

const mulberry = (seed) => {
    let t = seed >>> 0;
    return () => {
        t += 0x6d2b79f5;
        let r = Math.imul(t ^ (t >>> 15), t | 1);
        r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
        return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
};

const randomCols = (n, dims, seed) => {
    const rand = mulberry(seed);
    return Array.from({ length: dims }, () => Float32Array.from({ length: n }, () => rand() * 10));
};

describe('KdTree', () => {
    it('findKNearest matches brute force (3D)', () => {
        const n = 2000;
        const [x, y, z] = randomCols(n, 3, 12345);
        const tree = new KdTree([x, y, z]);
        const q = new Float32Array([5, 5, 5]);
        const k = 8;
        const got = tree.findKNearest(q, k);
        const d2 = i => (x[i] - q[0]) ** 2 + (y[i] - q[1]) ** 2 + (z[i] - q[2]) ** 2;
        const brute = Array.from({ length: n }, (_, i) => i).sort((a, b) => d2(a) - d2(b)).slice(0, k);
        assert.deepStrictEqual([...got.indices], brute);
        for (let i = 0; i < k; i++) {
            assert.ok(Math.abs(got.distances[i] - d2(brute[i])) < 1e-5);
        }
    });

    it('findNearest matches brute force and respects filter (5D, k-means shape)', () => {
        const n = 500;
        const cols = randomCols(n, 5, 777);
        const tree = new KdTree(cols);
        const rand = mulberry(42);
        for (let t = 0; t < 50; t++) {
            const q = Float32Array.from({ length: 5 }, () => rand() * 10);
            const d2 = i => cols.reduce((acc, c, j) => acc + (c[i] - q[j]) ** 2, 0);
            const bruteAll = Array.from({ length: n }, (_, i) => i).sort((a, b) => d2(a) - d2(b));
            assert.strictEqual(tree.findNearest(q).index, bruteAll[0]);
            const filter = i => i % 2 === 0;
            assert.strictEqual(tree.findNearest(q, filter).index, bruteAll.find(filter));
        }
    });

    it('flatten produces a traversable tree covering all points exactly once', () => {
        const n = 257;
        const [x, y, z] = randomCols(n, 3, 99);
        const flat = new KdTree([x, y, z]).flatten();
        assert.strictEqual(flat.rootIdx, 0);
        const seen = new Set();
        const stack = [flat.rootIdx];
        while (stack.length) {
            const t = stack.pop();
            const splat = flat.nodeSplatIdx[t];
            assert.ok(!seen.has(splat), 'splat appears once');
            seen.add(splat);
            assert.strictEqual(flat.nodeX[t], x[splat]);
            assert.strictEqual(flat.nodeY[t], y[splat]);
            assert.strictEqual(flat.nodeZ[t], z[splat]);
            if (flat.nodeLeft[t] !== 0xFFFFFFFF) stack.push(flat.nodeLeft[t]);
            if (flat.nodeRight[t] !== 0xFFFFFFFF) stack.push(flat.nodeRight[t]);
        }
        assert.strictEqual(seen.size, n);
    });

    it('builds in sub-quadratic time over a large all-identical point set', { timeout: 5000 }, () => {
        // Pre-fix the 2-way Lomuto partition degenerated to O(N^2) in the KD-tree
        // *build* (~5e9 ops at N=100k → tens of seconds), which is exactly how the
        // prod GPU pipeline pod hung on a splat with every gaussian at the origin.
        // Post-fix the 3-way partition keeps the build O(N log N): milliseconds.
        // Time the build (plus one query) so a regression in build time — the
        // actual fault — trips this assertion explicitly, not just the 5s timeout.
        const N = 100_000;

        const start = performance.now();
        const tree = new KdTree([new Float32Array(N), new Float32Array(N), new Float32Array(N)]);
        const { indices, distances } = tree.findKNearest(new Float32Array([0, 0, 0]), 8);
        const elapsed = performance.now() - start;

        // Generous ceiling, far below the multi-second O(N^2) build but well above
        // the O(N log N) build's few-ms cost — a hang detector, not a microbenchmark.
        assert.ok(elapsed < 2000,
            `KD-tree build + KNN over ${N} coincident points took ${elapsed.toFixed(0)}ms; ` +
            `expected < 2000ms (O(N^2) build regression)`);

        // KNN still returns k valid neighbours, all co-located at distance 0.
        assert.strictEqual(indices.length, 8);
        for (let i = 0; i < 8; i++) {
            assert.ok(indices[i] >= 0 && indices[i] < N, `index ${indices[i]} out of range`);
            assert.strictEqual(distances[i], 0);
        }
    });

    it('returns correct neighbours when the set contains duplicate coordinates', () => {
        // Five duplicates at the origin plus distinct points along +x. Exercises
        // the equal-block handling while keeping a well-defined nearest result.
        const xs = Float32Array.from([0, 0, 0, 0, 0, 1, 2, 5]);
        const ys = new Float32Array(8);
        const zs = new Float32Array(8);
        const tree = new KdTree([xs, ys, zs]);

        // Nearest to (1.9, 0, 0) is index 6 at (2,0,0), dist^2 = 0.01.
        const near = tree.findNearest(new Float32Array([1.9, 0, 0]));
        assert.strictEqual(near.index, 6);
        assert.ok(Math.abs(near.distanceSqr - 0.01) < 1e-5, `distanceSqr=${near.distanceSqr}`);

        // 3 nearest to (5,0,0): idx7 (0), idx6 (9), idx5 (16) — distinct dists.
        const { indices, distances } = tree.findKNearest(new Float32Array([5, 0, 0]), 3);
        assert.deepStrictEqual(Array.from(indices), [7, 6, 5]);
        assert.ok(Math.abs(distances[0] - 0) < 1e-5);
        assert.ok(Math.abs(distances[1] - 9) < 1e-5);
        assert.ok(Math.abs(distances[2] - 16) < 1e-5);
    });

    it('matches brute-force KNN on a mixed set with frequent ties', () => {
        // Deterministic pseudo-random points drawn from a small coordinate set
        // (0 over-represented → frequent duplicates) compared to brute force.
        const N = 500;
        const coords = [-2, -1, 0, 0, 0, 1, 2];
        let seed = 12345;
        const rand = () => {
            seed = (seed * 1103515245 + 12345) & 0x7fffffff;
            return coords[seed % coords.length];
        };
        const xs = [], ys = [], zs = [];
        for (let i = 0; i < N; i++) {
            xs.push(rand());
            ys.push(rand());
            zs.push(rand());
        }
        const tree = new KdTree([xs, ys, zs]);

        const k = 5;
        for (const qi of [0, 17, 199, 333, 499]) {
            const q = new Float32Array([xs[qi], ys[qi], zs[qi]]);
            const { distances } = tree.findKNearest(q, k);

            // Brute-force k smallest squared distances (self included, as the CPU
            // findKNearest does not exclude the query point).
            const all = [];
            for (let i = 0; i < N; i++) {
                const dx = xs[i] - q[0], dy = ys[i] - q[1], dz = zs[i] - q[2];
                all.push(dx * dx + dy * dy + dz * dz);
            }
            all.sort((a, b) => a - b);

            // Distances must equal the true k smallest (indices may differ on ties).
            for (let i = 0; i < k; i++) {
                assert.ok(Math.abs(distances[i] - all[i]) < 1e-5,
                    `query ${qi} neighbour ${i}: got ${distances[i]}, expected ${all[i]}`);
            }
        }
    });

    it('is engine-free (module source imports nothing from data-table or playcanvas)', async () => {
        const { readFile } = await import('node:fs/promises');
        const src = await readFile(new URL('../src/lib/spatial/kd-tree.ts', import.meta.url), 'utf8');
        assert.ok(!/from '.*data-table/.test(src), 'no data-table import');
        assert.ok(!/from 'playcanvas'/.test(src), 'no playcanvas import');
    });
});
