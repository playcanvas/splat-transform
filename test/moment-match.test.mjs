/**
 * n-ary moment match tests.
 *
 * The n = 2 path must be arithmetic-identical to the legacy pairwise
 * momentMatch (verbatim reference in fixtures/legacy-decimate-math.mjs) —
 * this is the math-preservation guarantee of the chunk-native decimation.
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

import { legacyMomentMatch } from './fixtures/legacy-decimate-math.mjs';
import { mergeGroup, createMergeScratch, sigmoid } from '../src/lib/decimate/moment-match.js';

const mulberry = (seed) => {
    let t = seed >>> 0;
    return () => {
        t += 0x6d2b79f5;
        let r = Math.imul(t ^ (t >>> 15), t | 1);
        r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
        return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
};

export const randomSplats = (n, colorDim, seed) => {
    const r = mulberry(seed);
    const pos = new Float32Array(n * 3);
    const geo = new Float32Array(n * 8);
    const color = new Float32Array(n * colorDim);
    for (let i = 0; i < n; i++) {
        pos.set([r() * 10 - 5, r() * 10 - 5, r() * 10 - 5], i * 3);
        const q = [r() - 0.5, r() - 0.5, r() - 0.5, r() - 0.5];
        const qn = Math.hypot(...q) || 1;
        geo.set([
            q[0] / qn, q[1] / qn, q[2] / qn, q[3] / qn,
            Math.log(r() * 0.5 + 0.01), Math.log(r() * 0.5 + 0.01), Math.log(r() * 0.5 + 0.01),
            r() * 8 - 4
        ], i * 8);
        for (let c = 0; c < colorDim; c++) color[i * colorDim + c] = r() * 2 - 1;
    }
    return { pos, geo, color, colorDim };
};

describe('mergeGroup', () => {
    it('n=2 matches legacy momentMatch exactly', () => {
        const view = randomSplats(2000, 48, 7);
        const out = { pos: new Float64Array(3), geo: new Float64Array(8), color: new Float64Array(48) };
        const scratch = createMergeScratch();
        for (let p = 0; p < 1000; p++) {
            const a = p * 2, b = p * 2 + 1;
            mergeGroup(view, [a, b], 2, out, scratch);
            const ref = legacyMomentMatch(view, a, b);
            for (let c = 0; c < 3; c++) assert.ok(Math.abs(out.pos[c] - ref.mu[c]) < 1e-9, `pair ${p} mu[${c}]`);
            for (let c = 0; c < 3; c++) assert.ok(Math.abs(out.geo[4 + c] - ref.sc[c]) < 1e-9, `pair ${p} sc[${c}]: ${out.geo[4 + c]} vs ${ref.sc[c]}`);
            const dot = out.geo[0] * ref.q[0] + out.geo[1] * ref.q[1] + out.geo[2] * ref.q[2] + out.geo[3] * ref.q[3];
            assert.ok(Math.abs(Math.abs(dot) - 1) < 1e-9, `pair ${p} quat dot ${dot}`);
            const refOp = Math.max(1e-7, Math.min(1 - 1e-7, ref.op));
            assert.ok(Math.abs(sigmoid(out.geo[7]) - refOp) < 1e-6, `pair ${p} opacity`);
            for (let c = 0; c < 48; c++) assert.ok(Math.abs(out.color[c] - ref.sh[c]) < 1e-9, `pair ${p} sh[${c}]`);
        }
    });

    it('n=3 invariants: mean in bbox, alpha<=1, finite encodings, unit quaternion', () => {
        const view = randomSplats(300, 3, 11);
        const out = { pos: new Float64Array(3), geo: new Float64Array(8), color: new Float64Array(3) };
        const scratch = createMergeScratch();
        for (let g = 0; g < 100; g++) {
            const m = [g * 3, g * 3 + 1, g * 3 + 2];
            mergeGroup(view, m, 3, out, scratch);
            for (let c = 0; c < 8; c++) assert.ok(Number.isFinite(out.geo[c]), `group ${g} geo[${c}] finite`);
            assert.ok(sigmoid(out.geo[7]) <= 1 + 1e-12, `group ${g} alpha`);
            assert.ok(Math.abs(Math.hypot(out.geo[0], out.geo[1], out.geo[2], out.geo[3]) - 1) < 1e-9, `group ${g} quat`);
            for (let c = 0; c < 3; c++) {
                const vals = m.map(i => view.pos[i * 3 + c]);
                assert.ok(
                    out.pos[c] >= Math.min(...vals) - 1e-6 && out.pos[c] <= Math.max(...vals) + 1e-6,
                    `group ${g} mean[${c}] in member bbox`
                );
            }
            // scales sorted descending (eigenvalues sorted)
            assert.ok(out.geo[4] >= out.geo[5] - 1e-12 && out.geo[5] >= out.geo[6] - 1e-12, `group ${g} scale order`);
        }
    });

    it('is engine-free (module imports nothing from data-table or playcanvas)', async () => {
        const { readFile } = await import('node:fs/promises');
        const src = await readFile(new URL('../src/lib/decimate/moment-match.ts', import.meta.url), 'utf8');
        assert.ok(!/from '.*data-table/.test(src), 'no data-table import');
        assert.ok(!/from 'playcanvas'/.test(src), 'no playcanvas import');
    });
});
