/**
 * Field-L2 edge-cost formula properties (CPU): merging identical coincident
 * splats is (near-)lossless, and — unlike the old scale-invariant KL cost —
 * the cost scales with absolute Gaussian size, so a geometrically-similar merge
 * of large Gaussians costs far more than one of tiny Gaussians.
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

import { buildSplatCache, computeEdgeCost, CACHE_STRIDE } from '../src/lib/decimate/edge-cost-cpu.js';

// Build a minimal SplatView from per-splat specs (identity quaternion; only the
// DC colour is set, higher SH left zero). colorDim = 3 (DC only).
const makeView = (splats) => {
    const n = splats.length;
    const colorDim = 3;
    const pos = new Float32Array(n * 3);
    const geo = new Float32Array(n * 8);
    const color = new Float32Array(n * colorDim);
    for (let i = 0; i < n; i++) {
        pos[i * 3] = splats[i].pos[0];
        pos[i * 3 + 1] = splats[i].pos[1];
        pos[i * 3 + 2] = splats[i].pos[2];
        geo[i * 8] = 1; // identity quat (w, x, y, z)
        geo[i * 8 + 4] = splats[i].ls[0];
        geo[i * 8 + 5] = splats[i].ls[1];
        geo[i * 8 + 6] = splats[i].ls[2];
        geo[i * 8 + 7] = splats[i].op;
        color[i * 3] = splats[i].dc[0];
        color[i * 3 + 1] = splats[i].dc[1];
        color[i * 3 + 2] = splats[i].dc[2];
    }
    return { pos, geo, color, colorDim };
};

const cost = (view, i, j) => {
    const cache = new Float32Array((view.geo.length / 8) * CACHE_STRIDE);
    buildSplatCache(view, cache);
    return computeEdgeCost(cache, i, j);
};

describe('field-L2 edge cost', () => {
    it('merging identical coincident splats is (near-)lossless', () => {
        const l = Math.log(0.1);
        const s = { pos: [0.3, -0.2, 0.5], ls: [l, l, l], op: -2, dc: [0.1, 0.2, 0.3] };
        const view = makeView([s, { ...s }]);
        assert.ok(cost(view, 0, 1) < 1e-3, `expected ~0, got ${cost(view, 0, 1)}`);
    });

    it('cost grows with absolute Gaussian size (not scale-invariant)', () => {
        const small = makeView([
            { pos: [0, 0, 0], ls: [Math.log(0.1), Math.log(0.1), Math.log(0.1)], op: -4, dc: [1, 0, 0] },
            { pos: [0.2, 0, 0], ls: [Math.log(0.1), Math.log(0.1), Math.log(0.1)], op: -4, dc: [0, 0, 1] }
        ]);
        // Same geometry scaled ×10 (positions and scales).
        const large = makeView([
            { pos: [0, 0, 0], ls: [Math.log(1.0), Math.log(1.0), Math.log(1.0)], op: -4, dc: [1, 0, 0] },
            { pos: [2, 0, 0], ls: [Math.log(1.0), Math.log(1.0), Math.log(1.0)], op: -4, dc: [0, 0, 1] }
        ]);
        const cSmall = cost(small, 0, 1);
        const cLarge = cost(large, 0, 1);
        assert.ok(cSmall > 0, `distinct merge should cost > 0 (got ${cSmall})`);
        // Field L2 scales with Gaussian volume (~s³); a scale-invariant cost
        // would give a ratio near 1. Require a large margin.
        assert.ok(cLarge > cSmall * 100, `large/small ratio ${(cLarge / cSmall).toFixed(1)} (want >> 1)`);
    });
});
