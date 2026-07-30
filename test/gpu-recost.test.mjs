/**
 * GPU wave-engine acceptance (GPU required; suites skip without a WebGPU
 * adapter):
 *
 * 1. Refresh-kernel parity — wave-0 (singleton) and post-commit
 *    (multi-member, replayed through the commit log) best-edge costs must
 *    match the CPU f64 stateless eval within the study's 1e-3 relative gate
 *    for ≥99% of roots.
 * 2. Merge-set equality — on a fixture with well-separated costs (no
 *    near-ties), the GPU and inline paths must produce identical selections.
 */

import assert from 'node:assert';
import { after, before, describe, it } from 'node:test';

import { buildSplatCache, CACHE_STRIDE } from '../src/lib/decimate/edge-cost-cpu.js';
import {
    bestEdgeFor,
    bestEdgesForPartition,
    bestOut,
    partitionBestOut
} from '../src/lib/decimate/recost-core.js';
import { planBlockMerges, replayBlockPlan } from '../src/lib/decimate/block-plan.js';
import { MAX_GROUP } from '../src/lib/decimate/select.js';
import { selectMergesRecosted } from '../src/lib/decimate/select-recost.js';
import { GpuRecost, COMMIT_LOG_STRIDE } from '../src/lib/gpu/gpu-recost.js';

const NIL = 0xFFFFFFFF;
const K = 16;

let device = null;

before(async () => {
    try {
        const { createDevice } = await import('../src/cli/node-device.js');
        device = await createDevice();
    } catch {
        device = null;
    }
});

after(() => {
    device?.destroy?.();
});

const mulberry = (seed) => {
    let t = seed >>> 0;
    return () => {
        t += 0x6d2b79f5;
        let r = Math.imul(t ^ (t >>> 15), t | 1);
        r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
        return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
};

// Random splats; size spread makes costs well-separated (few near-ties).
const makeCache = (n, seed) => {
    const rand = mulberry(seed);
    const pos = new Float32Array(n * 3);
    const geo = new Float32Array(n * 8);
    const color = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
        pos[i * 3] = rand() * 4;
        pos[i * 3 + 1] = rand() * 4;
        pos[i * 3 + 2] = rand() * 4;
        geo[i * 8] = 1;
        geo[i * 8 + 4] = Math.log(0.02 + rand() * 0.3);
        geo[i * 8 + 5] = Math.log(0.02 + rand() * 0.3);
        geo[i * 8 + 6] = Math.log(0.02 + rand() * 0.3);
        geo[i * 8 + 7] = rand() * 4 - 2;
        color[i * 3] = rand() - 0.5;
        color[i * 3 + 1] = rand() - 0.5;
        color[i * 3 + 2] = rand() - 0.5;
    }
    const cache = new Float32Array(n * CACHE_STRIDE);
    buildSplatCache({ pos, geo, color, colorDim: 3 }, cache);
    return cache;
};

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

const makeState = (cache, neighbors, n) => {
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
        st: { SC: cache, cands: neighbors, D: K, N: n, maxGroup: MAX_GROUP, parent, size, version, mHead, mNext },
        mTail
    };
};

// Compare GPU (partner, cost f32) rows against CPU bestEdgeFor over `roots`.
const compareRefresh = (state, roots, out) => {
    const outCost = new Float32Array(out.buffer);
    let compared = 0, ok = 0, disagreePartner = 0;
    for (let p = 0; p < roots.length; p++) {
        const root = roots[p];
        const gPartner = out[p * 2];
        const has = bestEdgeFor(state.st, root);
        if (!has || gPartner === NIL) {
            // Both must agree there is no legal candidate.
            assert.strictEqual(has, gPartner !== NIL, `root ${root}: candidate existence disagrees`);
            continue;
        }
        compared++;
        const gCost = outCost[p * 2 + 1];
        const rel = Math.abs(gCost - bestOut.cost) / Math.max(1e-12, Math.abs(bestOut.cost));
        if (rel < 1e-3) ok++;
        if (gPartner !== bestOut.partner) disagreePartner++;
    }
    return { compared, ok, disagreePartner };
};

describe('GpuRecost refresh parity', () => {
    it('accepts a tiny residual core-plus-halo block in required-GPU mode', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');

        const n = 42;
        const coreCount = 21;
        const cache = makeCache(n, 2468);
        const neighbors = bruteNeighbors(cache, n, K);
        const plan = await planBlockMerges({
            splatCache: cache,
            neighbors,
            D: K,
            coreCount,
            totalCount: n,
            device,
            requireGpu: true
        });
        assert.ok(Array.from(plan.pairs).every(row => row < coreCount));
    });

    it('plans a core-only GPU block with the compact output stride', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');

        const n = 512;
        const cache = makeCache(n, 777);
        const neighbors = bruteNeighbors(cache, n, K);
        const inputs = {
            splatCache: cache,
            neighbors,
            D: K,
            coreCount: n,
            totalCount: n
        };
        const inline = await planBlockMerges(inputs);
        const gpu = await planBlockMerges({ ...inputs, device, requireGpu: true });

        assert.strictEqual(gpu.costs.length, inline.costs.length);
        assert.deepStrictEqual(
            Array.from(replayBlockPlan(n, gpu).memberGroup),
            Array.from(replayBlockPlan(n, inline).memberGroup)
        );
    });

    it('wave-0 singleton and post-commit cluster costs match CPU within 1e-3', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');

        const n = 2048;
        const cache = makeCache(n, 1234);
        const neighbors = bruteNeighbors(cache, n, K);
        const state = makeState(cache, neighbors, n);

        const gpu = new GpuRecost(device, n, K, MAX_GROUP, 4096);
        try {
            gpu.init(cache, neighbors);

            // Wave 0: all singletons.
            const pending = new Uint32Array(n);
            for (let i = 0; i < n; i++) pending[i] = i;
            const out = new Uint32Array(n * 2);
            await gpu.wave(new Uint32Array(0), 0, pending, n, out);
            const w0 = compareRefresh(state, pending, out);
            assert.ok(w0.compared > n * 0.9, `wave-0 compared ${w0.compared}`);
            assert.ok(w0.ok / w0.compared >= 0.99, `wave-0 parity ${w0.ok}/${w0.compared}`);

            // Commit ~300 merges on the CPU mirrors, replay the same log on
            // the GPU, then compare multi-member refreshes. Each root may be
            // touched at most once per wave — the production drain guarantees
            // this via its version checks, and the parallel replay's
            // race-freedom depends on it.
            const { st, mTail } = state;
            const log = new Uint32Array(4096 * COMMIT_LOG_STRIDE);
            let commits = 0;
            const touched = [];
            const touchedSet = new Set();
            for (let a = 0; a < n && commits < 300; a += 3) {
                if (st.parent[a] !== a || touchedSet.has(a)) continue;
                if (!bestEdgeFor(st, a)) continue;
                const b = bestOut.partner;
                if (st.parent[b] !== b || touchedSet.has(b) || st.size[a] + st.size[b] > MAX_GROUP) continue;
                touchedSet.add(a);
                touchedSet.add(b);
                const keep = st.size[a] >= st.size[b] ? a : b;
                const lose = keep === a ? b : a;
                const o = commits * COMMIT_LOG_STRIDE;
                log[o] = lose;
                log[o + 1] = keep;
                log[o + 2] = mTail[keep];
                log[o + 3] = st.mHead[lose];
                log[o + 4] = st.size[keep] + st.size[lose];
                st.mNext[mTail[keep]] = st.mHead[lose];
                mTail[keep] = mTail[lose];
                st.parent[lose] = keep;
                st.size[keep] += st.size[lose];
                st.version[keep]++;
                touched.push(keep);
                commits++;
            }
            assert.ok(commits >= 200, `committed ${commits}`);

            const roots = Uint32Array.from(touched);
            const out2 = new Uint32Array(roots.length * 2);
            await gpu.wave(log, commits, roots, roots.length, out2);
            const w1 = compareRefresh(state, roots, out2);
            assert.ok(w1.compared > roots.length * 0.5, `post-commit compared ${w1.compared}`);
            assert.ok(w1.ok / w1.compared >= 0.99, `post-commit parity ${w1.ok}/${w1.compared}`);
        } finally {
            gpu.destroy();
        }
    });

    it('returns independent best core and immutable-halo candidates', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');

        const n = 2048;
        const coreCount = 2000;
        const cache = makeCache(n, 4321);
        const neighbors = bruteNeighbors(cache, n, K);
        for (let i = 0; i < coreCount; i++) neighbors[i * K + K - 1] = coreCount + (i % (n - coreCount));
        neighbors.fill(NIL, coreCount * K);
        const state = makeState(cache, neighbors, n);
        const pending = Uint32Array.from({ length: 128 }, (_, i) => i * 7);
        const out = new Uint32Array(pending.length * 4);
        const costs = new Float32Array(out.buffer);

        const gpu = new GpuRecost(device, n, K, MAX_GROUP, 4096, coreCount);
        try {
            gpu.init(cache, neighbors);
            await gpu.wave(new Uint32Array(0), 0, pending, pending.length, out);
            for (let p = 0; p < pending.length; p++) {
                const root = pending[p];
                assert.ok(bestEdgesForPartition(state.st, root, coreCount));
                assert.strictEqual(out[p * 4], partitionBestOut.corePartner);
                assert.strictEqual(out[p * 4 + 2], partitionBestOut.haloPartner);
                for (const [gpuCost, cpuCost] of [
                    [costs[p * 4 + 1], partitionBestOut.coreCost],
                    [costs[p * 4 + 3], partitionBestOut.haloCost]
                ]) {
                    const rel = Math.abs(gpuCost - cpuCost) / Math.max(1e-12, Math.abs(cpuCost));
                    assert.ok(rel < 1e-3, `root ${root} partitioned cost parity ${rel}`);
                }
            }
        } finally {
            gpu.destroy();
        }
    });
});

describe('GpuRecost selection equality', () => {
    it('GPU and inline paths produce identical selections on distinct costs', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');

        const n = 2048;
        const cache = makeCache(n, 777);
        const neighbors = bruteNeighbors(cache, n, K);
        const needed = n >> 1;

        const inline = await selectMergesRecosted({ splatCache: cache, neighbors, D: K, N: n, mergesNeeded: needed });
        const gpu = await selectMergesRecosted({ splatCache: cache, neighbors, D: K, N: n, mergesNeeded: needed, device });

        assert.strictEqual(gpu.removed, inline.removed);
        assert.strictEqual(gpu.mergedGroups, inline.mergedGroups);
        assert.deepStrictEqual(Array.from(gpu.memberGroup), Array.from(inline.memberGroup));
    });
});
