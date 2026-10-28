/**
 * Rendering excess opacity (`opacity_excess`): a gaussian whose true opacity
 * exceeds 1 is drawn with the over-unity plateau profile — an opaque core and
 * a footprint wider than 3σ — on every scene tier, and `compensation: false`
 * draws it as if the column were absent.
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
const W = 64;
const SIGMA = 0.05;

// One white isotropic gaussian at the origin, saturated, with excess `e`
// (null = no column at all).
const makeScene = async (e) => {
    const { Column, DataTable, dataTableToChunkSource } = await import('../src/lib/index.js');
    const { createChunkDataPool } = await import('../src/lib/chunk/index.js');
    const one = (v) => new Float32Array([v]);
    const columns = [
        new Column('x', one(0)), new Column('y', one(0)), new Column('z', one(0)),
        new Column('rot_0', one(1)), new Column('rot_1', one(0)), new Column('rot_2', one(0)), new Column('rot_3', one(0)),
        new Column('scale_0', one(Math.log(SIGMA))), new Column('scale_1', one(Math.log(SIGMA))),
        new Column('scale_2', one(Math.log(SIGMA))),
        new Column('opacity', one(20)),
        new Column('f_dc_0', one(0.5 / 0.28209479177387814)), new Column('f_dc_1', one(0.5 / 0.28209479177387814)),
        new Column('f_dc_2', one(0.5 / 0.28209479177387814))
    ];
    if (e !== null) columns.push(new Column('opacity_excess', one(e)));
    return { source: dataTableToChunkSource(new DataTable(columns)), pool: createChunkDataPool() };
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
        it(`${tier}: excess widens the footprint beyond 3σ; --no-compensation and a zero excess do not`, async (t) => {
            if (!device) return t.skip('no WebGPU adapter available');
            const plain = footprint(await render(await makeScene(null), tier, true));
            const zero = footprint(await render(await makeScene(0), tier, true));
            const heavy = footprint(await render(await makeScene(15), tier, true));
            const off = footprint(await render(await makeScene(15), tier, false));

            assert.ok(plain.reach <= 3.3, `plain gaussian stays within 3σ (${plain.reach})`);
            assert.deepStrictEqual(zero, plain, 'a zero excess renders as no excess');
            assert.deepStrictEqual(off, plain, 'compensation off ignores the excess');
            // A = 16: D ≈ 5.0, D' = 1 + 0.4·4 ≈ 2.6, extent ≈ 1.6 + 3/1.6 ≈ 3.5σ
            assert.ok(heavy.reach > 3.3, `excess reaches past 3σ (${heavy.reach})`);
            assert.ok(heavy.lit > plain.lit, `excess lights more pixels (${heavy.lit} vs ${plain.lit})`);
        });
    }

    it('the core of an over-unity gaussian is opaque', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');
        const plain = await render(await makeScene(null), 'resident', true);
        const heavy = await render(await makeScene(15), 'resident', true);
        // one sigma off centre: the plain gaussian has fallen to exp(-1/2),
        // the plateau (radius D'-1 ≈ 1.6σ) is still at full opacity
        const px = Math.round(W / 2 + SIGMA / (2 / W));
        const at = (image) => image[((W / 2) * W + px) * 4];
        assert.ok(at(heavy) > at(plain) + 40, `plateau at 1σ: ${at(heavy)} vs ${at(plain)}`);
        assert.ok(at(heavy) >= 250, `plateau is opaque: ${at(heavy)}`);
    });
});
