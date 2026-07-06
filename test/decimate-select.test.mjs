/**
 * Global selection tests: disjointness, exact targets, closure behaviour,
 * bucketed-vs-exact-sorted greedy equivalence, non-finite exclusion.
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

import { selectMerges } from '../src/lib/decimate/select.js';

const mulberry = (seed) => {
    let t = seed >>> 0;
    return () => {
        t += 0x6d2b79f5;
        let r = Math.imul(t ^ (t >>> 15), t | 1);
        r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
        return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
};

// Ring candidates: each i lists i±1, i±2 — a perfect matching always exists.
const ringCandidates = (N, K, r) => {
    const idx = new Uint32Array(N * K);
    const cost = new Float32Array(N * K);
    for (let i = 0; i < N; i++) {
        const nb = [(i + 1) % N, (i + N - 1) % N, (i + 2) % N, (i + N - 2) % N];
        for (let s = 0; s < K; s++) {
            idx[i * K + s] = nb[s];
            cost[i * K + s] = r() + (s >= 2 ? 1 : 0);
        }
    }
    return { idx, cost };
};

describe('selectMerges', () => {
    it('selection is disjoint, hits exact target, CSR consistent', () => {
        const N = 10000, K = 4, r = mulberry(9);
        const cand = ringCandidates(N, K, r);
        const needed = N / 2;
        const sel = selectMerges(cand, N, K, needed);
        assert.strictEqual(sel.removed, needed);
        const seen = new Int32Array(N).fill(-1);
        for (let g = 0; g < sel.mergedGroups; g++) {
            const size = sel.groupOffsets[g + 1] - sel.groupOffsets[g];
            assert.ok(size >= 2 && size <= 4, `group ${g} size ${size}`);
            let min = Infinity;
            for (let m = sel.groupOffsets[g]; m < sel.groupOffsets[g + 1]; m++) {
                const id = sel.groupMembers[m];
                assert.strictEqual(seen[id], -1, `gaussian ${id} in one group`);
                seen[id] = g;
                assert.strictEqual(sel.memberGroup[id], g);
                min = Math.min(min, id);
            }
            assert.strictEqual(sel.groupMin[g], min);
        }
        let survivors = 0;
        for (let i = 0; i < N; i++) {
            if (sel.memberGroup[i] === -1) survivors++;
        }
        assert.strictEqual(survivors + sel.mergedGroups, N - needed);
    });

    it('bucketed greedy tracks exact-sorted greedy selection cost within quantization', () => {
        const N = 4000, K = 4, r = mulberry(21);
        const cand = ringCandidates(N, K, r);
        const needed = Math.floor(N * 0.3); // selective regime
        const sel = selectMerges(cand, N, K, needed);
        assert.strictEqual(sel.removed, needed);

        // reference: exact-sort greedy over the same candidate edges
        const entries = [];
        for (let e = 0; e < N * K; e++) {
            if (Number.isFinite(cand.cost[e])) entries.push(e);
        }
        entries.sort((a, b) => cand.cost[a] - cand.cost[b]);
        const used = new Uint8Array(N);
        let refCost = 0, taken = 0;
        for (const e of entries) {
            const i = Math.floor(e / K), j = cand.idx[e];
            if (used[i] || used[j]) continue;
            used[i] = 1;
            used[j] = 1;
            refCost += cand.cost[e];
            if (++taken >= needed) break;
        }

        let gotCost = 0;
        for (let g = 0; g < sel.mergedGroups; g++) {
            const a = sel.groupMembers[sel.groupOffsets[g]];
            const b = sel.groupMembers[sel.groupOffsets[g] + 1];
            let c = Infinity;
            for (let s = 0; s < K; s++) {
                if (cand.idx[a * K + s] === b) c = Math.min(c, cand.cost[a * K + s]);
                if (cand.idx[b * K + s] === a) c = Math.min(c, cand.cost[b * K + s]);
            }
            gotCost += c;
        }
        assert.ok(gotCost <= refCost * 1.02, `bucketed ${gotCost} vs sorted ${refCost}`);
    });

    it('closure attaches unmatched gaussians when pairs alone cannot reach the target', () => {
        // Star topology: everyone's candidates point at a tiny hub set, so
        // primary pairing exhausts quickly and closure must attach the rest.
        const N = 100, K = 2;
        const idx = new Uint32Array(N * K);
        const cost = new Float32Array(N * K);
        const r = mulberry(5);
        for (let i = 0; i < N; i++) {
            idx[i * K] = i < 8 ? (i + 1) % 8 : i % 8;      // hub 0..7
            idx[i * K + 1] = (i % 8 === 0) ? 1 : 0;
            cost[i * K] = r();
            cost[i * K + 1] = r() + 0.5;
        }
        const needed = 12;
        const sel = selectMerges({ idx, cost }, N, K, needed);
        assert.ok(sel.removed > 4, `closure extended removal (${sel.removed})`);
        for (let g = 0; g < sel.mergedGroups; g++) {
            const size = sel.groupOffsets[g + 1] - sel.groupOffsets[g];
            assert.ok(size >= 2 && size <= 4);
        }
    });

    it('non-finite costs are never selected', () => {
        const N = 100, K = 2;
        const idx = new Uint32Array(N * K).fill(0xFFFFFFFF);
        const cost = new Float32Array(N * K).fill(Infinity);
        idx[0] = 1;
        cost[0] = NaN;
        idx[2 * K] = 3;
        cost[2 * K] = 0.5;
        const sel = selectMerges({ idx, cost }, N, K, 2);
        assert.strictEqual(sel.removed, 1);
        assert.strictEqual(sel.memberGroup[0], -1);
        assert.strictEqual(sel.memberGroup[2], 0);
        assert.strictEqual(sel.memberGroup[3], 0);
    });

    it('zero merges needed selects nothing', () => {
        const N = 10, K = 2, r = mulberry(1);
        const cand = ringCandidates(N, K, r);
        const sel = selectMerges(cand, N, K, 0);
        assert.strictEqual(sel.removed, 0);
        assert.strictEqual(sel.mergedGroups, 0);
    });
});
