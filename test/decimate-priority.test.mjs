/**
 * Priority pass tests (CPU path): the resident best-K candidates must match
 * brute-force legacy edge costs computed over exact global KNN.
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

import { legacyEdgeCost } from './fixtures/legacy-decimate-math.mjs';
import { makeSyntheticSource } from './helpers/synthetic-source.mjs';
import { makeGaussianSamples } from '../src/lib/decimate/moment-match.js';
import { kdPartition } from '../src/lib/decimate/partition.js';
import { runPriorityPass } from '../src/lib/decimate/priority.js';

describe('priority pass (CPU)', () => {
    it('best-K candidates match brute-force legacy costs over exact KNN', async () => {
        const n = 1500, k = 16, K = 4;
        const { source, pool, view, pos } = await makeSyntheticSource(n, 1, 5, { chunkSize: 256 });
        const { order, blocks } = kdPartition(pos, 400);
        const cand = {
            idx: new Uint32Array(n * K).fill(0xFFFFFFFF),
            cost: new Float32Array(n * K).fill(Infinity)
        };
        await runPriorityPass({ source, pool, pos, order, blocks, K, k }, cand);

        const Z = makeGaussianSamples(1, 0);
        const d2 = (a, b) => (pos.x[a] - pos.x[b]) ** 2 + (pos.y[a] - pos.y[b]) ** 2 + (pos.z[a] - pos.z[b]) ** 2;
        for (let i = 0; i < n; i += 97) {
            const knn = Array.from({ length: n }, (_, j) => j)
                .filter(j => j !== i)
                .sort((a, b) => d2(i, a) - d2(i, b))
                .slice(0, k);
            const refCosts = knn.map(j => legacyEdgeCost(view, i, j, Z)).sort((a, b) => a - b);
            const got = [];
            for (let s = 0; s < K; s++) {
                if (cand.idx[i * K + s] !== 0xFFFFFFFF) got.push(cand.cost[i * K + s]);
            }
            assert.strictEqual(got.length, K, `query ${i} has K candidates`);
            for (let s = 1; s < got.length; s++) assert.ok(got[s] >= got[s - 1], `query ${i} costs ascending`);
            for (let s = 0; s < got.length; s++) {
                assert.ok(
                    Math.abs(got[s] - refCosts[s]) < Math.max(1e-3, Math.abs(refCosts[s]) * 1e-4),
                    `query ${i} slot ${s}: ${got[s]} vs ${refCosts[s]}`
                );
            }
        }
    });

    it('candidate ids are real neighbours (no self, no sentinels leaking as ids)', async () => {
        const n = 600, k = 16, K = 2;
        const { source, pool, pos } = await makeSyntheticSource(n, 0, 9, { chunkSize: 128 });
        const { order, blocks } = kdPartition(pos, 200);
        const cand = {
            idx: new Uint32Array(n * K).fill(0xFFFFFFFF),
            cost: new Float32Array(n * K).fill(Infinity)
        };
        await runPriorityPass({ source, pool, pos, order, blocks, K, k }, cand);
        for (let g = 0; g < n; g++) {
            for (let s = 0; s < K; s++) {
                const j = cand.idx[g * K + s];
                assert.notStrictEqual(j, g, 'no self candidate');
                assert.ok(j < n, `candidate id in range (${j})`);
                assert.ok(Number.isFinite(cand.cost[g * K + s]), 'finite cost');
            }
        }
    });
});
