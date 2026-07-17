/**
 * Block KNN tests: exactness of block∪halo KNN + verification against global
 * brute force, on both benign (uniform) and adversarial (clustered) scenes.
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

import { collectBlock, knnBlockCpu, toGlobalNeighbors, verifyAndFixKnn } from '../src/lib/decimate/knn-blocks.js';
import { kdPartition } from '../src/lib/decimate/partition.js';

const mulberry = (seed) => {
    let t = seed >>> 0;
    return () => {
        t += 0x6d2b79f5;
        let r = Math.imul(t ^ (t >>> 15), t | 1);
        r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
        return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
};

const scenes = {
    uniform: (n, r) => Float32Array.from({ length: n }, () => r() * 10),
    clustered: (n, r) => Float32Array.from({ length: n }, (_, i) => (i % 7) + r() * 0.01),
    // Bulk cluster + rare extreme flyaways: stretched block AABBs defeat the
    // density-based halo estimate — the requery-heavy regime.
    flyaway: (n, r) => Float32Array.from({ length: n }, () => (r() < 0.01 ? (r() - 0.5) * 5000 : r() * 10)),
    // Integer-grid coordinates: many coincident points and exact distance ties.
    coincident: (n, r) => Float32Array.from({ length: n }, () => Math.floor(r() * 12))
};

// Adversarial scenes are ALLOWED to requery heavily; the benign ones must not.
const requeryCapped = new Set(['uniform', 'clustered']);

describe('block KNN with halo + verification', () => {
    for (const [name, gen] of Object.entries(scenes)) {
        it(`matches global brute force (${name})`, () => {
            const n = 3000, k = 8, r = mulberry(3);
            const pos = { x: gen(n, r), y: gen(n, r), z: gen(n, r) };
            const { order, blocks } = kdPartition(pos, 500);
            const d2 = (a, b) => (pos.x[a] - pos.x[b]) ** 2 + (pos.y[a] - pos.y[b]) ** 2 + (pos.z[a] - pos.z[b]) ** 2;
            let totalFixed = 0;
            for (let bi = 0; bi < blocks.length; bi++) {
                const locals = collectBlock(pos, order, blocks, bi, k, 2.5);
                // Buffer-sizing contract: locals never exceed owned × (1 + haloCap).
                assert.ok(
                    locals.ids.length <= locals.ownedCount + Math.ceil(locals.ownedCount * 1),
                    `${name} block ${bi}: halo exceeds the cap (${locals.ids.length} locals for ${locals.ownedCount} owned)`
                );
                const nbLocal = knnBlockCpu(locals, k);
                const nbGlobal = toGlobalNeighbors(locals, nbLocal);
                totalFixed += verifyAndFixKnn(pos, order, blocks, bi, locals, k, nbGlobal, nbLocal);
                for (let q = 0; q < locals.ownedCount; q++) {
                    const g = locals.ids[q];
                    const dists = [];
                    for (let i = 0; i < n; i++) {
                        if (i !== g) dists.push(d2(g, i));
                    }
                    dists.sort((a, b) => a - b);
                    const got = [];
                    const seen = new Set();
                    for (let s = 0; s < k; s++) {
                        const nb = nbGlobal[q * k + s];
                        assert.notStrictEqual(nb, 0xFFFFFFFF, `${name} block ${bi} q ${q} slot ${s} sentinel`);
                        assert.notStrictEqual(nb, g, 'self excluded');
                        assert.ok(!seen.has(nb), `${name} block ${bi} q ${q}: duplicate neighbour`);
                        seen.add(nb);
                        got.push(d2(g, nb));
                    }
                    // Exact k-NN: the neighbour distance multiset must equal the
                    // true k smallest (valid under ties, where ids may differ).
                    got.sort((a, b) => a - b);
                    assert.deepStrictEqual(got, dists.slice(0, k), `${name} block ${bi} q ${q}: not the exact k nearest`);
                }
            }
            // sanity: verification exists but is not doing all the work
            if (requeryCapped.has(name)) {
                assert.ok(totalFixed < n * 0.5, `${name}: too many requeries (${totalFixed})`);
            }
        });
    }

    it('enveloping residual block (no covering halo) stays exact via forced requery', () => {
        // Bulk grid + rare extreme flyaways: the residual block's AABB
        // envelops the core, so no halo radius can satisfy the size cap —
        // collectBlock must fall back to an empty halo (h = -Infinity) and
        // verification must recover exactness for every residual query.
        const k = 8;
        const side = 22, nBulk = side * side * side; // 10648
        const fly = [
            [10000, 9000, -11000], [-12000, 10000, 9500], [11000, -9500, 12000], [-9000, -10000, -12000],
            [15000, 14000, 13000], [-15000, 16000, -14000], [14000, -13000, 15000], [-16000, -15000, 14000]
        ];
        const n = nBulk + fly.length;
        const pos = { x: new Float32Array(n), y: new Float32Array(n), z: new Float32Array(n) };
        let i = 0;
        for (let a = 0; a < side; a++) {
            for (let b = 0; b < side; b++) {
                for (let c = 0; c < side; c++, i++) {
                    pos.x[i] = a; pos.y[i] = b; pos.z[i] = c;
                }
            }
        }
        fly.forEach(([fx, fy, fz], j) => {
            pos.x[nBulk + j] = fx; pos.y[nBulk + j] = fy; pos.z[nBulk + j] = fz;
        });
        const { order, blocks } = kdPartition(pos, 1024);
        const d2 = (a, b) => (pos.x[a] - pos.x[b]) ** 2 + (pos.y[a] - pos.y[b]) ** 2 + (pos.z[a] - pos.z[b]) ** 2;
        let sawFallback = false;
        for (let bi = 0; bi < blocks.length; bi++) {
            const locals = collectBlock(pos, order, blocks, bi, k, 2.5);
            assert.ok(
                locals.ids.length <= locals.ownedCount + Math.ceil(locals.ownedCount * 1),
                `block ${bi}: halo exceeds the cap`
            );
            if (locals.h === -Infinity) sawFallback = true;
            const nbLocal = knnBlockCpu(locals, k);
            const nbGlobal = toGlobalNeighbors(locals, nbLocal);
            verifyAndFixKnn(pos, order, blocks, bi, locals, k, nbGlobal, nbLocal);
            for (let q = 0; q < locals.ownedCount; q++) {
                const g = locals.ids[q];
                const dists = [];
                for (let j = 0; j < n; j++) {
                    if (j !== g) dists.push(d2(g, j));
                }
                dists.sort((a, b) => a - b);
                const got = [];
                for (let s = 0; s < k; s++) {
                    const nb = nbGlobal[q * k + s];
                    assert.notStrictEqual(nb, 0xFFFFFFFF, `block ${bi} q ${q} sentinel`);
                    got.push(d2(g, nb));
                }
                got.sort((a, b) => a - b);
                assert.deepStrictEqual(got, dists.slice(0, k), `block ${bi} q ${q}: not the exact k nearest`);
            }
        }
        assert.ok(sawFallback, 'expected at least one no-covering-halo fallback block');
    });

    it('tiny scene (n <= k) keeps sentinels', () => {
        const n = 4, k = 8;
        const pos = {
            x: Float32Array.from([0, 1, 2, 3]),
            y: new Float32Array(4),
            z: new Float32Array(4)
        };
        const { order, blocks } = kdPartition(pos, 100);
        const locals = collectBlock(pos, order, blocks, 0, k, 2.5);
        const nbLocal = knnBlockCpu(locals, k);
        const nbGlobal = toGlobalNeighbors(locals, nbLocal);
        verifyAndFixKnn(pos, order, blocks, 0, locals, k, nbGlobal, nbLocal);
        for (let q = 0; q < 4; q++) {
            const filled = [];
            for (let s = 0; s < k; s++) {
                if (nbGlobal[q * k + s] !== 0xFFFFFFFF) filled.push(nbGlobal[q * k + s]);
            }
            assert.strictEqual(filled.length, 3, 'exactly n-1 real neighbours');
            assert.ok(!filled.includes(locals.ids[q]), 'self excluded');
        }
    });
});
