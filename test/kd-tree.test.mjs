/**
 * KdTree tests for splat-transform.
 *
 * Regression coverage for the O(N^2) build hang on degenerate inputs where many
 * points share a coordinate (e.g. a splat with every gaussian at the origin):
 * the build must stay O(N log N), and nearest-neighbour queries must remain
 * correct when ties are present.
 */

import assert from 'node:assert';
import { performance } from 'node:perf_hooks';
import { describe, it } from 'node:test';

import { Column, DataTable } from '../src/lib/index.js';
import { KdTree } from '../src/lib/spatial/kd-tree.js';

const buildTree = (xs, ys, zs) => new KdTree(new DataTable([
    new Column('x', Float32Array.from(xs)),
    new Column('y', Float32Array.from(ys)),
    new Column('z', Float32Array.from(zs))
]));

describe('KdTree', () => {
    it('builds in sub-quadratic time over a large all-identical point set', { timeout: 5000 }, () => {
        // Pre-fix the 2-way Lomuto partition degenerated to O(N^2) here
        // (~5e9 ops at N=100k → tens of seconds), which is exactly how the prod
        // GPU pipeline pod hung on a splat with every gaussian at the origin.
        // Post-fix the 3-way partition keeps the build O(N log N): milliseconds.
        const N = 100_000;
        const tree = new KdTree(new DataTable([
            new Column('x', new Float32Array(N)),
            new Column('y', new Float32Array(N)),
            new Column('z', new Float32Array(N))
        ]));

        const start = performance.now();
        const { indices, distances } = tree.findKNearest(new Float32Array([0, 0, 0]), 8);
        const elapsed = performance.now() - start;

        // A single KNN query must not pay O(N) per node either; generous ceiling
        // so this is a hang detector, not a perf microbenchmark.
        assert.ok(elapsed < 1000, `KNN query took ${elapsed.toFixed(0)}ms; expected < 1000ms`);

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
        const xs = [0, 0, 0, 0, 0, 1, 2, 5];
        const ys = [0, 0, 0, 0, 0, 0, 0, 0];
        const zs = [0, 0, 0, 0, 0, 0, 0, 0];
        const tree = buildTree(xs, ys, zs);

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
        const tree = buildTree(xs, ys, zs);

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
});
