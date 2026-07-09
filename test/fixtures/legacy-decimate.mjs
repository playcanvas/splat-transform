/**
 * End-to-end reference copy of the legacy `simplifyGaussians` loop (CPU
 * branches only), over SplatView arrays: per iteration — exact 16-NN,
 * dedup i<j edges, legacy edge costs, cost-sorted greedy disjoint pairing,
 * legacy pairwise momentMatch — until the target count. Small scenes only.
 */

import { legacyEdgeCost, legacyMomentMatch, makeGaussianSamples, logit } from './legacy-decimate-math.mjs';
import { KdTree } from '../../src/lib/spatial/kd-tree.js';

const KNN_K = 16;

export const legacySimplify = (view0, targetCount) => {
    let view = {
        pos: view0.pos.slice(),
        geo: view0.geo.slice(),
        color: view0.color.slice(),
        colorDim: view0.colorDim
    };
    const Z = makeGaussianSamples(1, 0);

    while (view.pos.length / 3 > targetCount) {
        const n = view.pos.length / 3;
        const kEff = Math.min(KNN_K, n - 1);

        // exact KNN
        const x = Float32Array.from({ length: n }, (_, i) => view.pos[i * 3]);
        const y = Float32Array.from({ length: n }, (_, i) => view.pos[i * 3 + 1]);
        const z = Float32Array.from({ length: n }, (_, i) => view.pos[i * 3 + 2]);
        const tree = new KdTree([x, y, z]);
        const q = new Float32Array(3);

        // dedup edges i<j
        const edgeU = [];
        const edgeV = [];
        for (let i = 0; i < n; i++) {
            q[0] = x[i]; q[1] = y[i]; q[2] = z[i];
            const res = tree.findKNearest(q, kEff + 1);
            let outPos = 0;
            for (let m = 0; m < res.indices.length && outPos < kEff; m++) {
                const j = res.indices[m];
                if (j === i) continue;
                outPos++;
                if (j > i) {
                    edgeU.push(i);
                    edgeV.push(j);
                }
            }
        }

        const costs = new Float64Array(edgeU.length);
        for (let e = 0; e < edgeU.length; e++) {
            costs[e] = legacyEdgeCost(view, edgeU[e], edgeV[e], Z);
        }

        const orderE = Array.from({ length: edgeU.length }, (_, e) => e)
            .filter(e => Number.isFinite(costs[e]))
            .sort((a, b) => costs[a] - costs[b]);

        const mergesNeeded = n - targetCount;
        const used = new Uint8Array(n);
        const pairs = [];
        for (const e of orderE) {
            const u = edgeU[e], v = edgeV[e];
            if (used[u] || used[v]) continue;
            used[u] = 1;
            used[v] = 1;
            pairs.push([u, v]);
            if (pairs.length >= mergesNeeded) break;
        }
        if (pairs.length === 0) throw new Error('legacySimplify: no valid pairs');

        // output: kept splats in original order, then merged results (legacy layout)
        const outCount = n - pairs.length;
        const next = {
            pos: new Float32Array(outCount * 3),
            geo: new Float32Array(outCount * 8),
            color: new Float32Array(outCount * view.colorDim),
            colorDim: view.colorDim
        };
        let dst = 0;
        for (let i = 0; i < n; i++) {
            if (used[i]) continue;
            next.pos.set(view.pos.subarray(i * 3, i * 3 + 3), dst * 3);
            next.geo.set(view.geo.subarray(i * 8, i * 8 + 8), dst * 8);
            next.color.set(view.color.subarray(i * view.colorDim, (i + 1) * view.colorDim), dst * view.colorDim);
            dst++;
        }
        for (const [u, v] of pairs) {
            const m = legacyMomentMatch(view, u, v);
            next.pos.set(m.mu, dst * 3);
            next.geo[dst * 8] = m.q[0];
            next.geo[dst * 8 + 1] = m.q[1];
            next.geo[dst * 8 + 2] = m.q[2];
            next.geo[dst * 8 + 3] = m.q[3];
            next.geo[dst * 8 + 4] = m.sc[0];
            next.geo[dst * 8 + 5] = m.sc[1];
            next.geo[dst * 8 + 6] = m.sc[2];
            next.geo[dst * 8 + 7] = logit(Math.max(0, Math.min(1, m.op)));
            next.color.set(m.sh, dst * view.colorDim);
            dst++;
        }
        view = next;
    }
    return view;
};
