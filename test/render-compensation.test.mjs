/**
 * Rendering excess opacity (`opacity_excess`): a gaussian whose true opacity
 * exceeds 1 is drawn with an opaque core and the gaussian's falloff stretched
 * over the rest of its footprint, carrying exactly its mass, with the opacity capped at
 * OPACITY_MAX — on every scene tier, and `compensation: false` draws it as if
 * the column were absent.
 */

import assert from 'node:assert';
import { after, before, describe, it } from 'node:test';

import { SIGMA_CUTOFF } from '../src/lib/render/config.js';
import { OPACITY_MAX, plateauCore, plateauCoreConstants } from '../src/lib/utils/opacity.js';

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
const W = 64;
const SIGMA = 0.05;

// One white isotropic gaussian at the origin of std dev `sigma`, saturated
// unless `opacity` (a logit) says otherwise, with excess `e` (null = no column
// at all), tagged with splat model `model`.
const makeScene = async (e, { sigma = SIGMA, opacity = 20, model = 'default' } = {}) => {
    const { Column, DataTable, dataTableToChunkSource } = await import('../src/lib/index.js');
    const { createChunkDataPool } = await import('../src/lib/chunk/index.js');
    const one = (v) => new Float32Array([v]);
    const columns = [
        new Column('x', one(0)), new Column('y', one(0)), new Column('z', one(0)),
        new Column('rot_0', one(1)), new Column('rot_1', one(0)), new Column('rot_2', one(0)), new Column('rot_3', one(0)),
        new Column('scale_0', one(Math.log(sigma))), new Column('scale_1', one(Math.log(sigma))),
        new Column('scale_2', one(Math.log(sigma))),
        new Column('opacity', one(opacity)),
        new Column('f_dc_0', one(0.5 / 0.28209479177387814)), new Column('f_dc_1', one(0.5 / 0.28209479177387814)),
        new Column('f_dc_2', one(0.5 / 0.28209479177387814))
    ];
    if (e !== null) columns.push(new Column('opacity_excess', one(e)));
    return { source: dataTableToChunkSource(new DataTable(columns), undefined, undefined, model), pool: createChunkDataPool() };
};

const render = async (scene, tier, compensation) => {
    const { Vec3 } = await import('playcanvas');
    const { SceneRenderer } = await import('../src/lib/render/index.js');
    const renderer = new SceneRenderer(device, scene.source, scene.pool, {
        projection: 'pinhole', width: W, height: W, background: { r: 0, g: 0, b: 0, a: 1 }, tier, compensation
    });
    await renderer.upload();
    try {
        // 1 world unit spans the image width at distance 1 with a 90° fov:
        // each pixel is 2/64 world units, so sigma = 0.05 is 1.6 px.
        return await renderer.render({
            position: new Vec3(0, 0, 1), target: new Vec3(0, 0, 0), up: new Vec3(0, 1, 0),
            fovY: Math.PI / 2, width: W, height: W, near: 0.125
        });
    } finally {
        renderer.destroy?.();
    }
};

// Summed red channel: the drawn mass of a white splat on black.
const mass = (image) => {
    let m = 0;
    for (let i = 0; i < image.length; i += 4) m += image[i];
    return m;
};

// Lit pixels and the farthest lit pixel's distance from the centre, in sigma.
const footprint = (image) => {
    let lit = 0, reach = 0;
    const pxPerSigma = SIGMA / (2 / W);
    for (let y = 0; y < W; y++) {
        for (let x = 0; x < W; x++) {
            if (image[(y * W + x) * 4] > 0) {
                lit++;
                reach = Math.max(reach, Math.hypot(x + 0.5 - W / 2, y + 0.5 - W / 2) / pxPerSigma);
            }
        }
    }
    return { lit, reach };
};

describe('rendering excess opacity', () => {
    for (const tier of TIERS) {
        it(`${tier}: excess lights a heavier footprint, capped at OPACITY_MAX; --no-compensation and a zero excess do not`, async (t) => {
            if (!device) return t.skip('no WebGPU adapter available');
            const plainImg = await render(await makeScene(null), tier, true);
            const plain = footprint(plainImg);
            const zero = footprint(await render(await makeScene(0), tier, true));
            const atMax = await render(await makeScene(0.5625), tier, true);
            const heavyImg = await render(await makeScene(15), tier, true);
            const off = footprint(await render(await makeScene(15), tier, false));

            assert.ok(plain.reach <= 3.3, `plain gaussian stays within 3σ (${plain.reach})`);
            assert.deepStrictEqual(zero, plain, 'a zero excess renders as no excess');
            assert.deepStrictEqual(off, plain, 'compensation off ignores the excess');
            const heavy = footprint(heavyImg);
            // A = 1.5625: an opaque core of 0.24 of the 3σ footprint, ending at 3σ like the plain gaussian
            assert.ok(mass(heavyImg) > 1.3 * mass(plainImg), `excess draws more mass (${mass(heavyImg)} vs ${mass(plainImg)})`);
            assert.ok(heavy.reach <= plain.reach + 0.1, `the over-unity profile stays within the plain footprint (${heavy.reach} vs ${plain.reach})`);
            assert.deepStrictEqual(Array.from(heavyImg), Array.from(atMax), 'A = 16 renders as A = OPACITY_MAX (1.5625)');
        });
    }

    it('the core of an over-unity gaussian is opaque', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');
        const plain = await render(await makeScene(null), 'resident', true);
        const heavy = await render(await makeScene(15), 'resident', true);
        // pixel centres at 0.31σ and 0.94σ from the gaussian: the core
        // (radius 0.24 × 3σ ≈ 0.72σ) is opaque, and its edge outweighs the gaussian's
        const at = (image, x) => image[((W / 2) * W + x) * 4];
        assert.ok(at(heavy, W / 2) >= 250, `plateau core is opaque: ${at(heavy, W / 2)}`);
        assert.ok(at(heavy, W / 2 + 1) > at(plain, W / 2 + 1) + 40, `plateau edge at 0.94σ: ${at(heavy, W / 2 + 1)} vs ${at(plain, W / 2 + 1)}`);
    });
});

describe('rendering antialiased scenes', () => {
    // A scene trained with anti-aliasing pays for the 0.3 px² dilation with opacity, as the engine's
    // aaFactor does: it draws sqrt(det Σ / det(Σ + 0.3)) = σ² / (σ² + 0.3) of the plain splat's mass
    // (σ in pixels), applied before the over-unity profile, which conserves mass.
    const pxPerSigma = (sigma) => sigma / (2 / W);
    const aaFactor = (sigma) => pxPerSigma(sigma) ** 2 / (pxPerSigma(sigma) ** 2 + 0.3);
    const cases = [
        ['a 1.6 px splat', { sigma: SIGMA }, null],
        ['a 0.5 px splat', { sigma: 0.016 }, null],
        ['a half-opaque splat', { sigma: SIGMA, opacity: 0 }, null],
        ['an over-unity splat (A = 1.5625)', { sigma: SIGMA }, 0.5625]
    ];
    for (const tier of TIERS) {
        it(`${tier}: draws the undilated mass`, async (t) => {
            if (!device) return t.skip('no WebGPU adapter available');
            for (const [label, opts, e] of cases) {
                const plain = mass(await render(await makeScene(e, opts), tier, true));
                const aa = mass(await render(await makeScene(e, { ...opts, model: 'antialiased' }), tier, true));
                const expected = aaFactor(opts.sigma);
                assert.ok(Math.abs(aa / plain / expected - 1) < 0.03, `${label}: ratio ${(aa / plain).toFixed(4)}, expected ${expected.toFixed(4)}`);
            }
        });
    }
});

describe('plateau core', () => {
    // The drawn mass of a unit-peak splat as rasterize-binned.ts draws it: the distance in sigma remapped
    // past the core, the gaussian minus GAUSSIAN_FLOOR, out to SIGMA_CUTOFF, integrated numerically.
    const drawn = (core) => {
        const floor = Math.exp(-0.5 * SIGMA_CUTOFF * SIGMA_CUTOFF);
        const steps = 20000;
        let mass = 0;
        for (let i = 0; i < steps; i++) {
            const r = ((i + 0.5) / steps) * SIGMA_CUTOFF;
            const s = Math.max(0, r - core * SIGMA_CUTOFF) / (1 - core);
            mass += Math.max(0, Math.exp(-0.5 * s * s) - floor) * 2 * Math.PI * r * (SIGMA_CUTOFF / steps);
        }
        return mass;
    };

    it('draws each opacity in (1, OPACITY_MAX] with its mass, relative to a plain gaussian of opacity 1', () => {
        const plain = drawn(0);
        assert.strictEqual(plateauCore(1, SIGMA_CUTOFF), 0);
        for (let A = 1.025; A <= OPACITY_MAX; A += 0.025) {
            const ratio = drawn(plateauCore(A, SIGMA_CUTOFF)) / plain;
            assert.ok(Math.abs(ratio / A - 1) < 1e-5, `A = ${A.toFixed(3)} draws ${ratio.toFixed(6)}`);
        }
    });

    it('gives the engine constants at its 2√2σ cutoff (gsplatPlateauCore in gsplatHelpersVS)', () => {
        const { r0, q } = plateauCoreConstants(2 * Math.SQRT2);
        assert.ok(Math.abs(r0 * r0 - 0.28986718) < 1e-6 && Math.abs(q - 0.62504992) < 1e-6 && Math.abs(r0 - 0.53839315) < 1e-6, `${r0 * r0} ${q} ${r0}`);
    });
});
