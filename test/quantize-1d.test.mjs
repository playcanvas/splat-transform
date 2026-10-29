/**
 * quantize1dColumns: the histogram/DP codebook quantizer.
 *
 * Focus: non-finite input robustness. `scale_* = -Infinity` (flat splats) and
 * `opacity = +Infinity` are valid pipeline values (filterNaN deliberately
 * keeps them), and a single one pooled into the histogram used to NaN-poison
 * the entire codebook — serialized as JSON nulls, which SOG readers decode as
 * log-scale 0 (unit-sized splats). Infinities must land on dedicated finite
 * end centroids; finite values must quantize as before.
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

import { quantize1dColumns } from '../src/lib/spatial/quantize-1d-core.js';

// deterministic rng for reproducible data
const mulberry32 = (seed) => {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6D2B79F5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
};

const assertAllFinite = (centroids) => {
    for (let i = 0; i < centroids.length; i++) {
        assert.ok(isFinite(centroids[i]), `centroid ${i} is ${centroids[i]}`);
    }
};

const assertAscending = (centroids) => {
    for (let i = 1; i < centroids.length; i++) {
        assert.ok(centroids[i] >= centroids[i - 1], `centroids not ascending at ${i}`);
    }
};

// decoded value of row i of column c
const decode = (result, c, i) => result.centroids[result.labels[c].data[i]];

describe('quantize1dColumns', () => {

    it('quantizes finite columns accurately', () => {
        const rand = mulberry32(1);
        const n = 1000;
        const cols = ['scale_0', 'scale_1', 'scale_2'].map(name => ({
            name,
            data: Float32Array.from({ length: n }, () => -12 + rand() * 11)
        }));

        const result = quantize1dColumns(cols);

        assert.strictEqual(result.centroids.length, 256);
        assertAllFinite(result.centroids);
        assertAscending(result.centroids);
        for (let c = 0; c < 3; c++) {
            assert.strictEqual(result.labels[c].name, cols[c].name);
            for (let i = 0; i < n; i++) {
                assert.ok(Math.abs(decode(result, c, i) - cols[c].data[i]) < 11 / 64,
                    `column ${c} row ${i} decodes too far from source`);
            }
        }
    });

    it('a single -Infinity does not poison the codebook', () => {
        const rand = mulberry32(2);
        const n = 1000;
        const data = Float32Array.from({ length: n }, () => -12 + rand() * 11);
        data[500] = -Infinity;

        const result = quantize1dColumns([{ name: 'scale_0', data }]);

        assertAllFinite(result.centroids);
        assertAscending(result.centroids);
        // the -Infinity row decodes to a sentinel below every finite value
        assert.ok(decode(result, 0, 500) <= -12 - 10, 'flat sentinel not far below finite range');
        // finite rows are unaffected
        for (let i = 0; i < n; i++) {
            if (i === 500) continue;
            assert.ok(Math.abs(decode(result, 0, i) - data[i]) < 11 / 64);
        }
    });

    it('flat splats (-Infinity scales pooled with other columns) keep a dedicated flat centroid', () => {
        // the corrupt-publish shape: scale_2 mostly one constant with a band of
        // -Infinity (flat) splats, scale_0/1 varied
        const rand = mulberry32(3);
        const n = 2000;
        const s0 = Float32Array.from({ length: n }, () => -11.6 + rand() * 10.3);
        const s1 = Float32Array.from({ length: n }, () => -13.5 + rand() * 12.1);
        const s2 = new Float32Array(n).fill(-10.6);
        for (let i = 0; i < n; i += 10) s2[i] = -Infinity;

        const result = quantize1dColumns([
            { name: 'scale_0', data: s0 }, { name: 'scale_1', data: s1 }, { name: 'scale_2', data: s2 }
        ]);

        assertAllFinite(result.centroids);
        for (let i = 0; i < n; i++) {
            const d = decode(result, 2, i);
            if (i % 10 === 0) {
                assert.ok(d <= -13.5 - 10, `flat splat ${i} decoded to ${d}`);
            } else {
                assert.ok(Math.abs(d - -10.6) < 0.1, `constant scale ${i} decoded to ${d}`);
            }
        }
    });

    it('+Infinity gets a dedicated top centroid', () => {
        const rand = mulberry32(4);
        const n = 1000;
        const data = Float32Array.from({ length: n }, () => -5 + rand() * 11);
        data[0] = Infinity;
        data[999] = Infinity;

        const result = quantize1dColumns([{ name: 'opacity', data }]);

        assertAllFinite(result.centroids);
        assertAscending(result.centroids);
        assert.ok(decode(result, 0, 0) >= 6 + 10, 'top sentinel not far above finite range');
        assert.ok(decode(result, 0, 999) >= 6 + 10);
        for (let i = 1; i < 999; i++) {
            assert.ok(Math.abs(decode(result, 0, i) - data[i]) < 11 / 64);
        }
    });

    it('all -Infinity input yields a finite codebook', () => {
        const data = new Float32Array(100).fill(-Infinity);

        const result = quantize1dColumns([{ name: 'scale_0', data }]);

        assertAllFinite(result.centroids);
        for (let i = 0; i < 100; i++) {
            assert.ok(decode(result, 0, i) <= -10);
        }
    });

    it('NaN values do not poison the codebook', () => {
        const rand = mulberry32(5);
        const n = 1000;
        const data = Float32Array.from({ length: n }, () => rand() * 4);
        data[10] = NaN;
        data[20] = NaN;

        const result = quantize1dColumns([{ name: 'scale_0', data }]);

        assertAllFinite(result.centroids);
        for (let i = 0; i < n; i++) {
            const label = result.labels[0].data[i];
            assert.ok(label >= 0 && label < 256);
            if (i !== 10 && i !== 20) {
                assert.ok(Math.abs(decode(result, 0, i) - data[i]) < 4 / 64);
            }
        }
    });

    it('identical finite values keep the legacy all-zero labels', () => {
        const data = new Float32Array(100).fill(-7.25);

        const result = quantize1dColumns([{ name: 'scale_0', data }]);

        for (let i = 0; i < 256; i++) {
            assert.strictEqual(result.centroids[i], -7.25);
        }
        for (let i = 0; i < 100; i++) {
            assert.strictEqual(result.labels[0].data[i], 0);
        }
    });

    it('identical finite values alongside -Infinity keep both representable', () => {
        const data = new Float32Array(100).fill(-10.6);
        data[0] = -Infinity;

        const result = quantize1dColumns([{ name: 'scale_2', data }]);

        assertAllFinite(result.centroids);
        assert.ok(decode(result, 0, 0) <= -10.6 - 10);
        for (let i = 1; i < 100; i++) {
            assert.strictEqual(decode(result, 0, i), Math.fround(-10.6));
        }
    });

    it('importance (gamma) shifts resolution toward a sparse tail of large values', () => {
        // log-scale-like: a wide core of small splats and a narrow 1% tail of large ones
        const rnd = mulberry32(7);
        const core = 300000, tail = 3000;
        const data = new Float32Array(core + tail);
        for (let i = 0; i < core; i++) {
            const u = rnd() || 1e-9, v = rnd();
            data[i] = -8 + 2 * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
        }
        for (let i = 0; i < tail; i++) {
            data[core + i] = 1 + 0.9 * rnd();
        }

        const rms = (result, from, to) => {
            let se = 0;
            for (let i = from; i < to; i++) {
                se += (data[i] - decode(result, 0, i)) ** 2;
            }
            return Math.sqrt(se / (to - from));
        };

        const plain = quantize1dColumns([{ name: 'scale_0', data }]);
        const weighted = quantize1dColumns([{ name: 'scale_0', data }], 256, 0.5, 0.25);

        // gamma 0 is the default
        assert.deepStrictEqual(quantize1dColumns([{ name: 'scale_0', data }], 256, 0.5, 0).centroids, plain.centroids);

        assertAscending(weighted.centroids);
        assert.ok(rms(weighted, core, core + tail) < 0.7 * rms(plain, core, core + tail), 'tail error should drop');
        assert.ok(rms(weighted, 0, core) < 1.25 * rms(plain, 0, core), 'core error should stay close');
    });

    it('empty input returns zeroed shapes', () => {
        const result = quantize1dColumns([{ name: 'scale_0', data: new Float32Array(0) }]);

        assert.strictEqual(result.centroids.length, 256);
        assert.strictEqual(result.labels[0].data.length, 0);
    });
});
