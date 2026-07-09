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
    clustered: (n, r) => Float32Array.from({ length: n }, (_, i) => (i % 7) + r() * 0.01)
};

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
                    const bruteMax = dists[k - 1];
                    for (let s = 0; s < k; s++) {
                        const nb = nbGlobal[q * k + s];
                        assert.notStrictEqual(nb, 0xFFFFFFFF, `${name} block ${bi} q ${q} slot ${s} sentinel`);
                        assert.notStrictEqual(nb, g, 'self excluded');
                        assert.ok(d2(g, nb) <= bruteMax + 1e-6, `${name} block ${bi} q ${q}: neighbor farther than true k-th`);
                    }
                }
            }
            // sanity: verification exists but is not doing all the work
            assert.ok(totalFixed < n * 0.5, `${name}: too many requeries (${totalFixed})`);
        });
    }

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
