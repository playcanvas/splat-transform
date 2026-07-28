/**
 * Forest KNN tests: exactness of the multi-part carried-bound query against
 * global brute force on benign and adversarial scenes — no halos, no
 * verification pass, exact by construction — plus the canonical neighbour
 * ordering all downstream tie-breaking depends on.
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

import { knnForestQuery, KNN_SENTINEL } from '../src/lib/decimate/knn-core.js';
import { kdPartition } from '../src/lib/decimate/partition.js';
import { sortNeighborRows } from '../src/lib/decimate/priority.js';
import { buildFlatKdTree } from '../src/lib/spatial/kd-tree.js';

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
    // Bulk cluster + rare extreme flyaways: the stretched-AABB regime that
    // defeated the old halo estimate.
    flyaway: (n, r) => Float32Array.from({ length: n }, () => (r() < 0.01 ? (r() - 0.5) * 5000 : r() * 10)),
    // Integer-grid coordinates: many coincident points and exact distance ties.
    coincident: (n, r) => Float32Array.from({ length: n }, () => Math.floor(r() * 12))
};

// Build a forest exactly as the buildKdForestPart task does: part-local
// column slices from partition ranges, node splat ids remapped to global,
// point AABB alongside (the part-entry cull).
const buildForest = (pos, order, ranges) => {
    return ranges.map(([start, end]) => {
        const cnt = end - start;
        const x = new Float32Array(cnt);
        const y = new Float32Array(cnt);
        const z = new Float32Array(cnt);
        const ids = new Uint32Array(cnt);
        const aabb = new Float32Array([Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity]);
        for (let i = 0; i < cnt; i++) {
            const g = order[start + i];
            ids[i] = g;
            x[i] = pos.x[g];
            y[i] = pos.y[g];
            z[i] = pos.z[g];
            aabb[0] = Math.min(aabb[0], x[i]);
            aabb[1] = Math.min(aabb[1], y[i]);
            aabb[2] = Math.min(aabb[2], z[i]);
            aabb[3] = Math.max(aabb[3], x[i]);
            aabb[4] = Math.max(aabb[4], y[i]);
            aabb[5] = Math.max(aabb[5], z[i]);
        }
        const flat = buildFlatKdTree(x, y, z);
        for (let t = 0; t < flat.nodeSplatIdx.length; t++) flat.nodeSplatIdx[t] = ids[flat.nodeSplatIdx[t]];
        return { ...flat, aabb };
    });
};

// Group partition blocks into `partCount`-ish contiguous ranges.
const partRanges = (blocks, partCount) => {
    const per = Math.ceil(blocks.length / partCount);
    const ranges = [];
    for (let b = 0; b < blocks.length; b += per) {
        const last = Math.min(b + per, blocks.length) - 1;
        ranges.push([blocks[b].start, blocks[last].end]);
    }
    return ranges;
};

const queryAll = (pos, n, forest, k) => {
    const queryPos = new Float32Array(n * 3);
    const ids = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
        ids[i] = i;
        queryPos[i * 3] = pos.x[i];
        queryPos[i * 3 + 1] = pos.y[i];
        queryPos[i * 3 + 2] = pos.z[i];
    }
    const out = new Uint32Array(n * k);
    knnForestQuery(forest, queryPos, ids, n, k, out);
    return out;
};

describe('forest KNN', () => {
    for (const [name, gen] of Object.entries(scenes)) {
        it(`matches global brute force across parts (${name})`, () => {
            const n = 3000, k = 8, r = mulberry(3);
            const pos = { x: gen(n, r), y: gen(n, r), z: gen(n, r) };
            const { order, blocks } = kdPartition(pos, 500);
            const forest = buildForest(pos, order, partRanges(blocks, 3));
            assert.ok(forest.length >= 2, 'multi-part coverage');
            const out = queryAll(pos, n, forest, k);

            const d2 = (a, b) => (pos.x[a] - pos.x[b]) ** 2 + (pos.y[a] - pos.y[b]) ** 2 + (pos.z[a] - pos.z[b]) ** 2;
            for (let q = 0; q < n; q += 7) {
                const dists = [];
                for (let i = 0; i < n; i++) {
                    if (i !== q) dists.push(d2(q, i));
                }
                dists.sort((a, b) => a - b);
                const got = [];
                const seen = new Set();
                for (let s = 0; s < k; s++) {
                    const nb = out[q * k + s];
                    assert.notStrictEqual(nb, KNN_SENTINEL, `${name} q ${q} slot ${s} sentinel`);
                    assert.notStrictEqual(nb, q, 'self excluded');
                    assert.ok(!seen.has(nb), `${name} q ${q}: duplicate neighbour`);
                    seen.add(nb);
                    got.push(d2(q, nb));
                }
                // Exact k-NN: the neighbour distance multiset must equal the
                // true k smallest (valid under ties, where ids may differ).
                got.sort((a, b) => a - b);
                assert.deepStrictEqual(got, dists.slice(0, k), `${name} q ${q}: not the exact k nearest`);
            }
        });
    }

    it('enveloping flyaways stay exact (the old no-covering-halo regime)', () => {
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
        const forest = buildForest(pos, order, partRanges(blocks, 4));
        const out = queryAll(pos, n, forest, k);

        const d2 = (a, b) => (pos.x[a] - pos.x[b]) ** 2 + (pos.y[a] - pos.y[b]) ** 2 + (pos.z[a] - pos.z[b]) ** 2;
        for (let q = 0; q < n; q += 97) {
            const dists = [];
            for (let j = 0; j < n; j++) {
                if (j !== q) dists.push(d2(q, j));
            }
            dists.sort((a, b) => a - b);
            const got = [];
            for (let s = 0; s < k; s++) {
                const nb = out[q * k + s];
                assert.notStrictEqual(nb, KNN_SENTINEL, `q ${q} sentinel`);
                got.push(d2(q, nb));
            }
            got.sort((a, b) => a - b);
            assert.deepStrictEqual(got, dists.slice(0, k), `q ${q}: not the exact k nearest`);
        }
    });

    it('tiny scene (n <= k) keeps sentinels', () => {
        const n = 4, k = 8;
        const pos = {
            x: Float32Array.from([0, 1, 2, 3]),
            y: new Float32Array(4),
            z: new Float32Array(4)
        };
        const { order, blocks } = kdPartition(pos, 100);
        const forest = buildForest(pos, order, partRanges(blocks, 1));
        const out = queryAll(pos, n, forest, k);
        for (let q = 0; q < 4; q++) {
            const filled = [];
            for (let s = 0; s < k; s++) {
                if (out[q * k + s] !== KNN_SENTINEL) filled.push(out[q * k + s]);
            }
            assert.strictEqual(filled.length, 3, 'exactly n-1 real neighbours');
            assert.ok(!filled.includes(q), 'self excluded');
        }
    });

    it('canonical row order: (d², id) ascending, sentinels last', () => {
        const n = 500, k = 8, r = mulberry(11);
        const pos = {
            x: scenes.coincident(n, r),
            y: scenes.coincident(n, r),
            z: scenes.coincident(n, r)
        };
        const { order, blocks } = kdPartition(pos, 128);
        const forest = buildForest(pos, order, partRanges(blocks, 2));
        const nb = queryAll(pos, n, forest, k);
        const owned = new Uint32Array(n);
        for (let i = 0; i < n; i++) owned[i] = i;
        sortNeighborRows(pos, owned, nb, k);

        const d2 = (a, b) => (pos.x[a] - pos.x[b]) ** 2 + (pos.y[a] - pos.y[b]) ** 2 + (pos.z[a] - pos.z[b]) ** 2;
        for (let q = 0; q < n; q++) {
            let prevD = -1, prevId = -1, sawSentinel = false;
            for (let s = 0; s < k; s++) {
                const j = nb[q * k + s];
                if (j === KNN_SENTINEL) {
                    sawSentinel = true;
                    continue;
                }
                assert.ok(!sawSentinel, `q ${q}: id after sentinel`);
                const dist = d2(q, j);
                assert.ok(
                    dist > prevD || (dist === prevD && j > prevId),
                    `q ${q} slot ${s}: (${dist}, ${j}) not after (${prevD}, ${prevId})`
                );
                prevD = dist;
                prevId = j;
            }
        }
    });
});
