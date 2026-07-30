/**
 * Stateless re-costed selection kernel:
 *
 * 1. Singleton-pair parity — the stateless cancelled-form eval must agree
 *    with the pairwise kernel (they are the same formula, differently
 *    associated).
 * 2. Marginal-cost decomposition — accumulated ΔE (minus the ordering-only
 *    colour terms) telescopes to E(final cluster), so it must be independent
 *    of the merge path taken.
 * 3. selectMergesRecosted behaviour — target respected, CSR well-formed,
 *    group cap enforced, and coincident inputs concentrate into full groups
 *    (continuation merges via refresh — the concentration behaviour the
 *    quality study validated).
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

import { buildSplatCache, computeEdgeCost, CACHE_STRIDE, COLOR_WEIGHT } from '../src/lib/decimate/edge-cost-cpu.js';
import { evalMergeCore } from '../src/lib/decimate/recost-core.js';
import { MAX_GROUP } from '../src/lib/decimate/select.js';
import { selectMergesRecosted } from '../src/lib/decimate/select-recost.js';

const NIL = 0xFFFFFFFF;

const mulberry = (seed) => {
    let t = seed >>> 0;
    return () => {
        t += 0x6d2b79f5;
        let r = Math.imul(t ^ (t >>> 15), t | 1);
        r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
        return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
};

// Random splats in a unit-ish box; identity quats, DC-only colour.
const makeCache = (n, seed) => {
    const rand = mulberry(seed);
    const pos = new Float32Array(n * 3);
    const geo = new Float32Array(n * 8);
    const color = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
        pos[i * 3] = rand() * 2;
        pos[i * 3 + 1] = rand() * 2;
        pos[i * 3 + 2] = rand() * 2;
        geo[i * 8] = 1;
        geo[i * 8 + 4] = Math.log(0.05 + rand() * 0.1);
        geo[i * 8 + 5] = Math.log(0.05 + rand() * 0.1);
        geo[i * 8 + 6] = Math.log(0.05 + rand() * 0.1);
        geo[i * 8 + 7] = rand() * 4 - 2;
        color[i * 3] = rand() - 0.5;
        color[i * 3 + 1] = rand() - 0.5;
        color[i * 3 + 2] = rand() - 0.5;
    }
    const cache = new Float32Array(n * CACHE_STRIDE);
    buildSplatCache({ pos, geo, color, colorDim: 3 }, cache);
    return cache;
};

// Brute-force k nearest neighbour ids per splat (sentinel padded).
const bruteNeighbors = (cache, n, k) => {
    const nb = new Uint32Array(n * k).fill(NIL);
    const d2 = (a, b) => {
        const oa = a * CACHE_STRIDE, ob = b * CACHE_STRIDE;
        return (cache[oa] - cache[ob]) ** 2 + (cache[oa + 1] - cache[ob + 1]) ** 2 + (cache[oa + 2] - cache[ob + 2]) ** 2;
    };
    for (let i = 0; i < n; i++) {
        const ids = Array.from({ length: n }, (_, j) => j)
            .filter(j => j !== i)
            .sort((a, b) => d2(i, a) - d2(i, b))
            .slice(0, k);
        for (let s = 0; s < ids.length; s++) nb[i * k + s] = ids[s];
    }
    return nb;
};

// A hand-driven selection state (mirrors select-recost's structure).
const makeState = (cache, neighbors, D, n) => {
    const parent = new Uint32Array(n);
    const size = new Uint32Array(n).fill(1);
    const version = new Uint32Array(n).fill(1);
    const mHead = new Uint32Array(n);
    const mNext = new Uint32Array(n).fill(NIL);
    const mTail = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
        parent[i] = i; mHead[i] = i; mTail[i] = i;
    }
    return {
        st: { SC: cache, cands: neighbors, D, N: n, maxGroup: MAX_GROUP, parent, size, version, mHead, mNext },
        mTail
    };
};

// Commit a merge exactly as select-recost does (keep = larger side).
const commit = ({ st, mTail }, a, b) => {
    const keep = st.size[a] >= st.size[b] ? a : b;
    const lose = keep === a ? b : a;
    st.mNext[mTail[keep]] = st.mHead[lose];
    mTail[keep] = mTail[lose];
    st.parent[lose] = keep;
    st.size[keep] += st.size[lose];
    st.version[keep]++;
    return keep;
};

// The ordering-only colour term between two clusters (mass-weighted mean
// base colours) — subtracted so accumulated costs telescope to pure E.
const colorTerm = (cache, membersA, membersB) => {
    const mean = (members) => {
        let w = 0, b0 = 0, b1 = 0, b2 = 0;
        for (const m of members) {
            const o = m * CACHE_STRIDE;
            const mass = cache[o + 11];
            w += mass;
            b0 += mass * cache[o + 12];
            b1 += mass * cache[o + 13];
            b2 += mass * cache[o + 14];
        }
        return [b0 / w, b1 / w, b2 / w];
    };
    const a = mean(membersA), b = mean(membersB);
    const d0 = a[0] - b[0], d1 = a[1] - b[1], d2 = a[2] - b[2];
    return COLOR_WEIGHT * (d0 * d0 + d1 * d1 + d2 * d2);
};

describe('stateless re-costed eval', () => {
    it('singleton pair cost matches the pairwise kernel', () => {
        const n = 200, k = 8;
        const cache = makeCache(n, 42);
        const neighbors = bruteNeighbors(cache, n, k);
        const { st } = makeState(cache, neighbors, k, n);
        // Same gate as the reference tool's selftest (rel 1e-3): near-zero
        // costs are differences of large self terms, so association noise
        // amplifies relatively — the accepted near-tie class.
        let worst = 0;
        for (let i = 0; i < n; i += 3) {
            const j = neighbors[i * k];
            const mine = evalMergeCore(st, i, j);
            const lib = computeEdgeCost(cache, i, j);
            const rel = Math.abs(mine - lib) / Math.max(1e-12, Math.abs(lib));
            worst = Math.max(worst, rel);
        }
        assert.ok(worst < 1e-3, `worst singleton parity rel diff ${worst.toExponential(2)}`);
    });

    it('accumulated marginal cost is merge-path independent (telescopes to E)', () => {
        const n = 60, k = 8;
        const cache = makeCache(n, 7);
        const neighbors = bruteNeighbors(cache, n, k);

        for (let a = 0; a < n; a += 7) {
            const b = neighbors[a * k];
            const c = neighbors[a * k + 1];

            // Path 1: (a+b) then (ab+c).
            const s1 = makeState(cache, neighbors, k, n);
            const e1ab = evalMergeCore(s1.st, a, b) - colorTerm(cache, [a], [b]);
            const ab = commit(s1, a, b);
            const e1c = evalMergeCore(s1.st, ab, c) - colorTerm(cache, [a, b], [c]);

            // Path 2: (a+c) then (ac+b).
            const s2 = makeState(cache, neighbors, k, n);
            const e2ac = evalMergeCore(s2.st, a, c) - colorTerm(cache, [a], [c]);
            const ac = commit(s2, a, c);
            const e2b = evalMergeCore(s2.st, ac, b) - colorTerm(cache, [a, c], [b]);

            const sum1 = e1ab + e1c;
            const sum2 = e2ac + e2b;
            assert.ok(
                Math.abs(sum1 - sum2) <= Math.max(1e-12, 1e-8 * Math.abs(sum1)),
                `triple (${a},${b},${c}): path sums ${sum1} vs ${sum2}`
            );
        }
    });
});

describe('selectMergesRecosted', () => {
    it('hits the target with a well-formed capped CSR', async () => {
        const n = 256, k = 8;
        const cache = makeCache(n, 99);
        const neighbors = bruteNeighbors(cache, n, k);
        const needed = n >> 1;
        const sel = await selectMergesRecosted({ splatCache: cache, neighbors, D: k, N: n, mergesNeeded: needed });

        assert.strictEqual(sel.removed, needed);
        assert.strictEqual(sel.groupOffsets[sel.mergedGroups], sel.groupMembers.length);
        const seen = new Set();
        let members = 0;
        for (let g = 0; g < sel.mergedGroups; g++) {
            const sz = sel.groupOffsets[g + 1] - sel.groupOffsets[g];
            assert.ok(sz >= 2 && sz <= MAX_GROUP, `group ${g} size ${sz}`);
            members += sz;
            for (let s = sel.groupOffsets[g]; s < sel.groupOffsets[g + 1]; s++) {
                const m = sel.groupMembers[s];
                assert.ok(!seen.has(m), 'member in one group only');
                seen.add(m);
                assert.strictEqual(sel.memberGroup[m], g);
            }
            assert.strictEqual(sel.groupMin[g], sel.groupMembers[sel.groupOffsets[g]]);
        }
        assert.strictEqual(members - sel.mergedGroups, needed);
    });

    it('concentrates coincident splats into full groups (refresh continuation)', async () => {
        const n = 64, k = 16;
        // All splats identical and coincident.
        const pos = new Float32Array(n * 3).fill(0.5);
        const geo = new Float32Array(n * 8);
        const color = new Float32Array(n * 3).fill(0.25);
        for (let i = 0; i < n; i++) {
            geo[i * 8] = 1;
            geo[i * 8 + 4] = geo[i * 8 + 5] = geo[i * 8 + 6] = Math.log(0.1);
            geo[i * 8 + 7] = 0;
        }
        const cache = new Float32Array(n * CACHE_STRIDE);
        buildSplatCache({ pos, geo, color, colorDim: 3 }, cache);
        // Ring neighbour graph (i ± 1..8 mod n): coincident points have
        // arbitrary-but-diverse KNN in the real pipeline; a hub-shaped graph
        // (everyone pointing at the same few ids) would starve pools by
        // construction, which is a fixture artifact, not pipeline behaviour.
        const neighbors = new Uint32Array(n * k);
        for (let i = 0; i < n; i++) {
            for (let s = 0; s < k; s++) {
                const off = (s >> 1) + 1;
                neighbors[i * k + s] = (i + (s & 1 ? n - off : off)) % n;
            }
        }
        const needed = n - (n / MAX_GROUP);   // 48: reachable only by filling groups to the cap
        const sel = await selectMergesRecosted({ splatCache: cache, neighbors, D: k, N: n, mergesNeeded: needed });

        // Disjoint pairing alone tops out at n/2 = 32 removals; anything well
        // beyond that requires continuation merges via refresh re-entry (the
        // concentration behaviour). The ring graph's bounded reach (±8) can
        // strand a final under-full pair, so demand near-full rather than
        // perfect packing.
        assert.ok(sel.removed >= needed - MAX_GROUP, `removed ${sel.removed} of ${needed}`);
        let full = 0;
        for (let g = 0; g < sel.mergedGroups; g++) {
            const sz = sel.groupOffsets[g + 1] - sel.groupOffsets[g];
            assert.ok(sz <= MAX_GROUP, `group ${g} size ${sz} over cap`);
            if (sz === MAX_GROUP) full++;
        }
        assert.ok(full >= (n / MAX_GROUP) - 2, `only ${full} full groups`);
    });
});
