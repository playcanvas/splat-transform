/**
 * Resident-path parity.
 *
 * `SceneRenderer` keeps the scene on the GPU and culls, depth-sorts and
 * gathers there; `renderSplats` (the chunked path) culls and sorts on the
 * CPU and streams the sorted list. Both drive the same shaders and must
 * produce the same pixels.
 *
 * The comparison is byte-exact. Scene positions sit on a 1/64 grid and the
 * cameras have exactly representable bases, so both paths compute
 * bit-identical depth keys (f32 on the GPU, f64 rounded to f32 on the CPU)
 * and break ties the same way (stable sort, ascending row). Scenes include
 * gaussians behind the camera, one exactly on the near plane and one with a
 * NaN position, which the CPU cull drops before the GPU sees them and the
 * resident path must invalidate on the GPU instead.
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

/**
 * Deterministic scene on a 1/64 grid: `n` gaussians in [-4, 4]³ plus a
 * group behind a camera at z = 10 (z in [10.5, 14]), one gaussian whose
 * camera depth equals the near plane exactly for that camera (z = 9.875
 * with near 0.125) and one with a NaN x.
 *
 * @param {number} n - Gaussians in the main group.
 * @param {number} seed - PRNG seed.
 * @returns {Promise<import('../src/lib/index.js').DataTable>} The scene.
 */
const makeScene = async (n, seed) => {
    const { Column, DataTable } = await import('../src/lib/index.js');
    let s = seed >>> 0;
    const rnd = () => {
        s = (s * 1664525 + 1013904223) >>> 0;
        return s / 2 ** 32;
    };
    const grid = (lo, hi) => Math.round((lo + (hi - lo) * rnd()) * 64) / 64;

    const behind = Math.floor(n / 20);
    const total = n + behind + 2;
    const f = () => new Float32Array(total);
    const x = f(), y = f(), z = f();
    const rot = [f(), f(), f(), f()];
    const scale = [f(), f(), f()];
    const opacity = f();
    const fdc = [f(), f(), f()];

    for (let i = 0; i < total; i++) {
        x[i] = grid(-4, 4);
        y[i] = grid(-4, 4);
        z[i] = i < n ? grid(-4, 4) : grid(10.5, 14);
        // Random unit quaternion (exactness irrelevant to the sort keys).
        const u1 = rnd(), u2 = rnd(), u3 = rnd();
        const a = Math.sqrt(1 - u1), b = Math.sqrt(u1);
        rot[0][i] = a * Math.sin(2 * Math.PI * u2);
        rot[1][i] = a * Math.cos(2 * Math.PI * u2);
        rot[2][i] = b * Math.sin(2 * Math.PI * u3);
        rot[3][i] = b * Math.cos(2 * Math.PI * u3);
        for (let k = 0; k < 3; k++) {
            scale[k][i] = -3 + 2.5 * rnd();
            fdc[k][i] = -1.5 + 3 * rnd();
        }
        opacity[i] = -2 + 6 * rnd();
    }
    // Exactly on the near plane of the z = 10 camera (cz = 0.125 = near).
    z[total - 2] = 9.875;
    // Undefined position.
    x[total - 1] = NaN;

    return new DataTable([
        new Column('x', x), new Column('y', y), new Column('z', z),
        new Column('rot_0', rot[0]), new Column('rot_1', rot[1]), new Column('rot_2', rot[2]), new Column('rot_3', rot[3]),
        new Column('scale_0', scale[0]), new Column('scale_1', scale[1]), new Column('scale_2', scale[2]),
        new Column('opacity', opacity),
        new Column('f_dc_0', fdc[0]), new Column('f_dc_1', fdc[1]), new Column('f_dc_2', fdc[2])
    ]);
};

const background = { r: 0.1, g: 0.2, b: 0.3, a: 1 };

const compare = (resident, chunked, label) => {
    let differing = 0, max = 0;
    for (let i = 0; i < chunked.length; i++) {
        const d = Math.abs(resident[i] - chunked[i]);
        if (d !== 0) {
            differing++;
            if (d > max) max = d;
        }
    }
    assert.strictEqual(resident.length, chunked.length, `${label}: image size`);
    assert.strictEqual(differing, 0, `${label}: ${differing} bytes differ (max ${max})`);
};

const countForeground = (image) => {
    const bg = [background.r, background.g, background.b].map(v => Math.round(v * 255));
    let n = 0;
    for (let i = 0; i < image.length; i += 4) {
        if (image[i] !== bg[0] || image[i + 1] !== bg[1] || image[i + 2] !== bg[2]) n++;
    }
    return n;
};

const renderBoth = async (dataTable, camera) => {
    const { SceneRenderer, renderSplats } = await import('../src/lib/render/index.js');
    const scene = new SceneRenderer(device, dataTable, {
        projection: camera.projection ?? 'pinhole',
        width: camera.width,
        height: camera.height,
        motionBlur: camera.shutterClose !== undefined,
        background
    });
    try {
        await scene.upload();
        const resident = await scene.render(camera);
        const chunked = await renderSplats(device, dataTable, camera, background);
        return { resident, chunked };
    } finally {
        scene.destroy();
    }
};

describe('resident renderer matches the chunked path', () => {
    it('static pinhole, defocus, motion blur and equirect are byte-identical', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');
        const { Vec3 } = await import('playcanvas');
        const dataTable = await makeScene(20000, 7);

        const pinhole = {
            position: new Vec3(0, 0, 10),
            target: new Vec3(0, 0, 0),
            up: new Vec3(0, 1, 0),
            fovY: Math.PI / 3,
            width: 256,
            height: 192,
            near: 0.125
        };
        const cases = [
            ['static', pinhole],
            ['defocus', { ...pinhole, focusDistance: 10, apertureScale: 2 }],
            // Shutter-close eye moves 0.5 units along +z: depths stay on the grid.
            ['motion blur', {
                ...pinhole,
                shutterClose: { position: new Vec3(0.5, 0.25, 10.5), target: new Vec3(0.5, 0.25, 0.5), up: new Vec3(0, 1, 0) }
            }],
            ['equirect', {
                projection: 'equirect',
                position: new Vec3(0.5, 0.25, 0.75),
                target: new Vec3(0.5, 0.25, -1),
                up: new Vec3(0, 1, 0),
                fovY: 0,
                width: 256,
                height: 128,
                near: 0.125
            }]
        ];

        for (const [label, camera] of cases) {
            const { resident, chunked } = await renderBoth(dataTable, camera);
            assert.ok(countForeground(chunked) > 1000, `${label}: scene should cover the frame`);
            compare(resident, chunked, label);
        }
    });

    it('a view with nothing in front of the camera renders the background', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');
        const { Vec3 } = await import('playcanvas');
        const dataTable = await makeScene(500, 11);
        // Looking along +z from z = 20: every gaussian is behind the camera.
        const camera = {
            position: new Vec3(0, 0, 20),
            target: new Vec3(0, 0, 30),
            up: new Vec3(0, 1, 0),
            fovY: Math.PI / 3,
            width: 64,
            height: 64,
            near: 0.125
        };
        const { resident, chunked } = await renderBoth(dataTable, camera);
        assert.strictEqual(countForeground(resident), 0, 'resident: only background');
        compare(resident, chunked, 'empty view');
    });
});
