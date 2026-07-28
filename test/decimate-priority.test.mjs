/**
 * Priority pass tests (CPU path): the resident best-K candidates must match
 * brute-force edge costs computed over exact global KNN.
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

import { makeSyntheticSource } from './helpers/synthetic-source.mjs';
import { buildSplatCache, computeEdgeCost, CACHE_STRIDE } from '../src/lib/decimate/edge-cost-cpu.js';
import { kdPartition } from '../src/lib/decimate/partition.js';
import { runPriorityPass } from '../src/lib/decimate/priority.js';

describe('priority pass (CPU)', () => {
    it('best-K candidates match brute-force costs over exact KNN', async () => {
        const n = 1500, k = 16, K = 4;
        const { source, pool, view, pos } = await makeSyntheticSource(n, 1, 5, { chunkSize: 256 });
        const { order, blocks } = kdPartition(pos, 400);
        const cand = {
            idx: new Uint32Array(n * K).fill(0xFFFFFFFF),
            cost: new Float32Array(n * K).fill(Infinity)
        };
        await runPriorityPass({ source, pool, pos, order, blocks, K, k }, cand);

        const cache = new Float32Array(n * CACHE_STRIDE);
        buildSplatCache(view, cache);
        const d2 = (a, b) => (pos.x[a] - pos.x[b]) ** 2 + (pos.y[a] - pos.y[b]) ** 2 + (pos.z[a] - pos.z[b]) ** 2;
        for (let i = 0; i < n; i += 97) {
            const knn = Array.from({ length: n }, (_, j) => j)
                .filter(j => j !== i)
                .sort((a, b) => d2(i, a) - d2(i, b))
                .slice(0, k);
            const refCosts = knn.map(j => computeEdgeCost(cache, i, j)).sort((a, b) => a - b);
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

    it('persist-only mode (no cand) fills cache + neighbours without cost work', async () => {
        const n = 900, k = 16, K = 4;
        const { source, pool, view, pos } = await makeSyntheticSource(n, 1, 21, { chunkSize: 256 });
        const { order, blocks } = kdPartition(pos, 300);
        const cacheOut = new Float32Array(n * CACHE_STRIDE);
        const neighborsOut = new Uint32Array(n * k);
        await runPriorityPass(
            { source, pool, pos, order, blocks, K, k, cacheOut, neighborsOut },
            undefined
        );

        // Cache rows must equal buildSplatCache over the same splats.
        const ref = new Float32Array(n * CACHE_STRIDE);
        buildSplatCache(view, ref);
        for (let i = 0; i < n; i += 53) {
            for (let c = 0; c < CACHE_STRIDE; c++) {
                const got = cacheOut[i * CACHE_STRIDE + c];
                const want = ref[i * CACHE_STRIDE + c];
                assert.ok(
                    Math.abs(got - want) <= Math.max(1e-6, Math.abs(want) * 1e-6),
                    `cache row ${i} field ${c}: ${got} vs ${want}`
                );
            }
        }

        // Neighbour rows must be the exact global KNN as a set.
        const d2 = (a, b) => (pos.x[a] - pos.x[b]) ** 2 + (pos.y[a] - pos.y[b]) ** 2 + (pos.z[a] - pos.z[b]) ** 2;
        for (let i = 0; i < n; i += 97) {
            const brute = new Set(Array.from({ length: n }, (_, j) => j)
                .filter(j => j !== i)
                .sort((a, b) => d2(i, a) - d2(i, b))
                .slice(0, k));
            for (let s = 0; s < k; s++) {
                const g = neighborsOut[i * k + s];
                assert.ok(brute.has(g), `query ${i} slot ${s}: ${g} not in brute-force KNN`);
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
