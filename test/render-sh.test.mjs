/**
 * View-dependent colour: the renderer's SH evaluation, bands 1 to 4, against
 * an independent CPU evaluation of the real spherical harmonics (associated
 * Legendre polynomials with the Condon-Shortley phase, coefficients ordered
 * m = -l..l, channel-major), which is the 3DGS convention. One large opaque
 * gaussian is viewed from many directions and its centre pixel compared, on
 * every scene tier.
 */

import assert from 'node:assert';
import { after, before, describe, it } from 'node:test';

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

const TIERS = ['resident', 'streamed-gpu', 'streamed-cpu'];
const W = 33;
const OPACITY = 12;

const fact = n => (n <= 1 ? 1 : n * fact(n - 1));

// Associated Legendre P_l^m(x), m >= 0, with the Condon-Shortley phase.
const legendre = (l, m, x) => {
    let pmm = 1;
    const s = Math.sqrt(Math.max(0, 1 - x * x));
    for (let i = 1; i <= m; i++) pmm *= -(2 * i - 1) * s;
    if (l === m) return pmm;
    let pm1 = x * (2 * m + 1) * pmm;
    for (let ll = m + 2; ll <= l; ll++) {
        const pll = ((2 * ll - 1) * x * pm1 - (ll + m - 1) * pmm) / (ll - m);
        pmm = pm1;
        pm1 = pll;
    }
    return pm1;
};

// Real SH Y_l^m for a unit direction d.
const realSH = (l, m, d) => {
    const am = Math.abs(m);
    const k = Math.sqrt(((2 * l + 1) / (4 * Math.PI)) * (fact(l - am) / fact(l + am)));
    const p = legendre(l, am, d[2]);
    if (m === 0) return k * p;
    const phi = Math.atan2(d[1], d[0]);
    return Math.SQRT2 * k * p * (m > 0 ? Math.cos(am * phi) : Math.sin(am * phi));
};

// Deterministic coefficients, distinct per channel and coefficient.
const makeCoeffs = (bands) => {
    let s = 7 + bands;
    const rnd = () => (s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32;
    const n = (bands + 1) * (bands + 1) - 1;
    return [0, 1, 2].map(() => Array.from({ length: n }, () => (rnd() * 2 - 1) * 0.2));
};

const makeScene = async (bands, coeffs) => {
    const { Column, DataTable, dataTableToChunkSource } = await import('../src/lib/index.js');
    const { createChunkDataPool } = await import('../src/lib/chunk/index.js');
    const one = v => new Float32Array([v]);
    const values = {
        x: 0, y: 0, z: 0, rot_0: 1, rot_1: 0, rot_2: 0, rot_3: 0,
        scale_0: 0, scale_1: 0, scale_2: 0, opacity: OPACITY, f_dc_0: 0, f_dc_1: 0, f_dc_2: 0
    };
    const columns = Object.entries(values).map(([name, v]) => new Column(name, one(v)));
    const n = coeffs[0].length;
    for (let c = 0; c < 3; c++) {
        for (let k = 0; k < n; k++) columns.push(new Column(`f_rest_${c * n + k}`, one(coeffs[c][k])));
    }
    return { source: dataTableToChunkSource(new DataTable(columns)), pool: createChunkDataPool() };
};

// View directions spread over the sphere (Fibonacci lattice).
const directions = (count) => {
    const out = [];
    const golden = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < count; i++) {
        const z = 1 - (2 * (i + 0.5)) / count;
        const r = Math.sqrt(1 - z * z);
        out.push([r * Math.cos(golden * i), z, r * Math.sin(golden * i)]);
    }
    return out;
};

// Expected centre byte: alpha-composited over black, as the rasterizer does.
const expected = (coeffs, bands, dir) => {
    const alpha = Math.min(0.99, (1 / (1 + Math.exp(-OPACITY))) * (1 - Math.exp(-4.5)));
    const basis = [];
    for (let l = 1; l <= bands; l++) for (let m = -l; m <= l; m++) basis.push(realSH(l, m, dir));
    return coeffs.map((cc) => {
        const col = Math.max(0, cc.reduce((sum, v, k) => sum + v * basis[k], 0) + 0.5);
        return Math.round(Math.min(1, alpha * col) * 255);
    });
};

describe('SH evaluation', () => {
    for (const tier of TIERS) {
        for (const bands of [1, 2, 3, 4]) {
            it(`${tier}: bands 1-${bands} match the textbook real SH from every direction`, async (t) => {
                if (!device) return t.skip('no WebGPU adapter available');
                const { Vec3 } = await import('playcanvas');
                const { SceneRenderer } = await import('../src/lib/render/index.js');
                const coeffs = makeCoeffs(bands);
                const scene = await makeScene(bands, coeffs);
                const renderer = new SceneRenderer(device, scene.source, scene.pool, {
                    projection: 'pinhole', width: W, height: W, background: { r: 0, g: 0, b: 0, a: 1 }, tier
                });
                assert.strictEqual(await renderer.upload(), tier, 'forced tier honoured');
                try {
                    const centre = ((W >> 1) * W + (W >> 1)) * 4;
                    let worst = 0, spread = 0;
                    for (const d of directions(32)) {
                        const eye = d.map(v => v * 12);
                        const up = Math.abs(d[1]) > 0.9 ? new Vec3(1, 0, 0) : new Vec3(0, 1, 0);
                        const image = await renderer.render({
                            position: new Vec3(...eye), target: new Vec3(0, 0, 0), up,
                            fovY: Math.PI / 6, width: W, height: W, near: 0.1
                        });
                        // the colour is evaluated along eye -> splat
                        const want = expected(coeffs, bands, d.map(v => -v));
                        for (let c = 0; c < 3; c++) {
                            worst = Math.max(worst, Math.abs(image[centre + c] - want[c]));
                            spread = Math.max(spread, Math.abs(want[c] - 126));
                        }
                    }
                    // f32 on the GPU against f64 here: allow one step of rounding
                    assert.ok(worst <= 1, `centre colour off by up to ${worst}/255`);
                    assert.ok(spread > 20, `coefficients should move the colour well off grey (${spread})`);
                } finally {
                    renderer.destroy();
                    await scene.source.close();
                }
            });
        }
    }
});
