/**
 * Global selection tests (cost-ordered agglomeration): disjoint clustering,
 * exact targets, CSR consistency, cheap-region concentration (expensive
 * gaussians spared), multi-member clusters when pairing alone can't reach the
 * target, non-finite exclusion.
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

import { selectMerges, MAX_GROUP } from '../src/lib/decimate/select.js';

const mulberry = (seed) => {
    let t = seed >>> 0;
    return () => {
        t += 0x6d2b79f5;
        let r = Math.imul(t ^ (t >>> 15), t | 1);
        r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
        return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
};

// Ring candidates: each i lists i±1, i±2 — a dense graph that clusters freely.
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

describe('selectMerges (agglomeration)', () => {
    it('clusters are disjoint, hit exact target, CSR consistent', () => {
        const N = 10000, K = 4, r = mulberry(9);
        const cand = ringCandidates(N, K, r);
        const needed = N / 2;
        const sel = selectMerges(cand, N, K, needed);
        assert.strictEqual(sel.removed, needed);
        const seen = new Int32Array(N).fill(-1);
        for (let g = 0; g < sel.mergedGroups; g++) {
            const size = sel.groupOffsets[g + 1] - sel.groupOffsets[g];
            assert.ok(size >= 2 && size <= MAX_GROUP, `group ${g} size ${size}`);
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
        // Each cluster collapses to one survivor: survivors + groups = N - removed.
        let survivors = 0;
        for (let i = 0; i < N; i++) {
            if (sel.memberGroup[i] === -1) survivors++;
        }
        assert.strictEqual(survivors + sel.mergedGroups, N - needed);
    });

    it('removal concentrates in the low-cost region, sparing expensive gaussians', () => {
        // Two disjoint rings: a cheap region [0, H) and an expensive one [H, N).
        // Cost-ordered agglomeration must exhaust the cheap region before ever
        // touching an expensive edge, so with a target the cheap side can cover
        // on its own, every expensive gaussian survives untouched.
        const N = 2000, K = 4, H = 1000, r = mulberry(7);
        const idx = new Uint32Array(N * K);
        const cost = new Float32Array(N * K);
        for (let i = 0; i < N; i++) {
            const base = i < H ? 0 : H;
            const local = i - base;
            const nb = [
                base + (local + 1) % H, base + (local + H - 1) % H,
                base + (local + 2) % H, base + (local + H - 2) % H
            ];
            const cheap = i < H;
            for (let s = 0; s < K; s++) {
                idx[i * K + s] = nb[s];
                cost[i * K + s] = (cheap ? 0.01 * r() : 10 + r()) + (s >= 2 ? (cheap ? 0.02 : 1) : 0);
            }
        }
        const needed = 400; // < cheap-region capacity
        const sel = selectMerges({ idx, cost }, N, K, needed);
        assert.strictEqual(sel.removed, needed);
        for (let i = H; i < N; i++) {
            assert.strictEqual(sel.memberGroup[i], -1, `expensive gaussian ${i} must be spared`);
        }
    });

    it('reaches the target via multi-member clusters when pairing alone cannot', () => {
        // Star topology: everyone points at a tiny hub set, so a disjoint
        // matching tops out at ~8 removals. Agglomeration must grow hub-centred
        // clusters (size > 2) to reach a target beyond that.
        const N = 100, K = 2, r = mulberry(5);
        const idx = new Uint32Array(N * K);
        const cost = new Float32Array(N * K);
        for (let i = 0; i < N; i++) {
            idx[i * K] = i < 8 ? (i + 1) % 8 : i % 8;      // hub 0..7
            idx[i * K + 1] = (i % 8 === 0) ? 1 : 0;
            cost[i * K] = r();
            cost[i * K + 1] = r() + 0.5;
        }
        const needed = 12;
        const sel = selectMerges({ idx, cost }, N, K, needed);
        assert.strictEqual(sel.removed, needed);
        let maxSize = 0;
        for (let g = 0; g < sel.mergedGroups; g++) {
            const size = sel.groupOffsets[g + 1] - sel.groupOffsets[g];
            assert.ok(size >= 2 && size <= MAX_GROUP, `group ${g} size ${size}`);
            maxSize = Math.max(maxSize, size);
        }
        assert.ok(maxSize >= 3, `expected a multi-member cluster (max size ${maxSize})`);
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
