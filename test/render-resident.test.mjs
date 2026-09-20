/**
 * Scene renderer parity.
 *
 * `SceneRenderer` renders a `ChunkSource` in one of three tiers: resident
 * (every layer on the GPU; GPU cull, depth sort and gather), streamed with
 * the GPU sort (positions resident, attributes gathered from the source per
 * pass in depth order) and streamed with the CPU sort. `renderSplats` (the
 * chunked reference path) culls and sorts on the CPU from a `DataTable` and
 * streams the sorted list. All four drive the same shaders and must produce
 * the same pixels.
 *
 * The comparison is byte-exact. Scene positions sit on a 1/64 grid and the
 * cameras have exactly representable bases, so every path computes
 * bit-identical depth keys (f32 on the GPU, f64 rounded to f32 on the CPU)
 * and breaks ties the same way (stable sort, ascending row). Scenes include
 * gaussians behind the camera, one exactly on the near plane and one with a
 * NaN position, which the CPU cull drops before the GPU sees them and the
 * GPU key pass must sort to the tail instead.
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

/**
 * Deterministic scene on a 1/64 grid: `n` gaussians in [-4, 4]³ plus a
 * group behind a camera at z = 10 (z in [10.5, 14]), one gaussian whose
 * camera depth equals the near plane exactly for that camera (z = 9.875
 * with near 0.125) and one with a NaN x.
 *
 * @param {number} n - Gaussians in the main group.
 * @param {number} seed - PRNG seed.
 * @param {number} [scaleBase] - Smallest log scale; the default gives small gaussians, 0 frame-filling ones.
 * @returns {Promise<{ dataTable: object, source: object, pool: object }>} The scene as a table and as a source.
 */
const makeScene = async (n, seed, scaleBase = -3) => {
    const { Column, DataTable, dataTableToChunkSource } = await import('../src/lib/index.js');
    const { createChunkDataPool } = await import('../src/lib/chunk/index.js');
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
            scale[k][i] = scaleBase + 2.5 * rnd();
            fdc[k][i] = -1.5 + 3 * rnd();
        }
        opacity[i] = -2 + 6 * rnd();
    }
    // Exactly on the near plane of the z = 10 camera (cz = 0.125 = near).
    z[total - 2] = 9.875;
    // Undefined position.
    x[total - 1] = NaN;

    const dataTable = new DataTable([
        new Column('x', x), new Column('y', y), new Column('z', z),
        new Column('rot_0', rot[0]), new Column('rot_1', rot[1]), new Column('rot_2', rot[2]), new Column('rot_3', rot[3]),
        new Column('scale_0', scale[0]), new Column('scale_1', scale[1]), new Column('scale_2', scale[2]),
        new Column('opacity', opacity),
        new Column('f_dc_0', fdc[0]), new Column('f_dc_1', fdc[1]), new Column('f_dc_2', fdc[2])
    ]);
    // Small chunks so the streamed tiers cross several range boundaries.
    const chunkSize = 4096;
    const source = dataTableToChunkSource(dataTable, chunkSize);
    const pool = createChunkDataPool({ chunkSize });
    return { dataTable, source, pool };
};

const background = { r: 0.1, g: 0.2, b: 0.3, a: 1 };

const compare = (candidate, reference, label) => {
    let differing = 0, max = 0;
    for (let i = 0; i < reference.length; i++) {
        const d = Math.abs(candidate[i] - reference[i]);
        if (d !== 0) {
            differing++;
            if (d > max) max = d;
        }
    }
    assert.strictEqual(candidate.length, reference.length, `${label}: image size`);
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

const makeRenderer = async (scene, camera, tier) => {
    const { SceneRenderer } = await import('../src/lib/render/index.js');
    const renderer = new SceneRenderer(device, scene.source, scene.pool, {
        projection: camera.projection ?? 'pinhole',
        width: camera.width,
        height: camera.height,
        background,
        tier
    });
    const chosen = await renderer.upload();
    assert.strictEqual(chosen, tier, 'forced tier honoured');
    return renderer;
};

/**
 * Run `fn` with the device reporting a smaller storage-binding limit. The
 * renderer sizes its groups and pair sorts from it; the engine reads only
 * the dispatch limit at render time.
 *
 * @param {number} bytes - The binding limit to report.
 * @param {() => Promise<void>} fn - The work.
 */
const withBindingLimit = async (bytes, fn) => {
    const real = device.limits;
    device.limits = { maxStorageBufferBindingSize: bytes, maxComputeWorkgroupsPerDimension: real.maxComputeWorkgroupsPerDimension };
    try {
        await fn();
    } finally {
        device.limits = real;
    }
};

describe('scene renderer matches the chunked path', () => {
    it('static pinhole, defocus and equirect are byte-identical in every tier', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');
        const { Vec3 } = await import('playcanvas');
        const { renderSplats } = await import('../src/lib/render/index.js');
        const scene = await makeScene(20000, 7);

        const up = new Vec3(0, 1, 0);
        const pinhole = {
            position: new Vec3(0, 0, 10),
            target: new Vec3(0, 0, 0),
            up,
            fovY: Math.PI / 3,
            width: 256,
            height: 192,
            near: 0.125
        };
        const equirect = {
            projection: 'equirect',
            position: new Vec3(0.5, 0.25, 0.75),
            target: new Vec3(0.5, 0.25, -1),
            up,
            fovY: 0,
            width: 256,
            height: 128,
            near: 0.125
        };
        const cases = [
            ['static', pinhole],
            ['defocus', { ...pinhole, focusDistance: 10, apertureScale: 2 }],
            ['equirect', equirect]
        ];

        for (const [label, camera] of cases) {
            const reference = await renderSplats(device, scene.dataTable, camera, background);
            assert.ok(countForeground(reference) > 1000, `${label}: scene should cover the frame`);
            for (const tier of TIERS) {
                const renderer = await makeRenderer(scene, camera, tier);
                try {
                    compare(await renderer.render(camera), reference, `${label} (${tier})`);
                } finally {
                    renderer.destroy();
                }
            }
        }
    });

    it('renders in groups and cuts ranges within a small storage binding', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');
        const { Vec3 } = await import('playcanvas');
        const { renderSplats } = await import('../src/lib/render/index.js');
        // Frame-filling gaussians: every scan block's pairs exceed the cap below.
        const scene = await makeScene(6000, 11, 0);
        const camera = { position: new Vec3(0, 0, 10), target: new Vec3(0, 0, 0), up: new Vec3(0, 1, 0), fovY: Math.PI / 3, width: 256, height: 192, near: 0.125 };
        const reference = await renderSplats(device, scene.dataTable, camera, background);
        assert.ok(countForeground(reference) > 1000, 'scene should cover the frame');

        // A 64 KiB binding: groups of one tile row (twelve per frame) and
        // sorts of at most 16K pairs, so blocks are cut inside.
        await withBindingLimit(64 * 1024, async () => {
            for (const tier of TIERS) {
                const renderer = await makeRenderer(scene, camera, tier);
                try {
                    compare(await renderer.render(camera), reference, `small binding (${tier})`);
                } finally {
                    renderer.destroy();
                }
            }
        });
    });

    it('GPU slice accumulation matches the float mean of the slices', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');
        const { Vec3 } = await import('playcanvas');
        const scene = await makeScene(20000, 7);
        const up = new Vec3(0, 1, 0);
        const slice = z => ({
            position: new Vec3(0, 0, z),
            target: new Vec3(0, 0, 0),
            up,
            fovY: Math.PI / 3,
            width: 256,
            height: 192,
            near: 0.125
        });
        const slices = [slice(10), slice(10.25), slice(10.5), slice(10.75)];
        for (const tier of ['resident', 'streamed-gpu']) {
            const renderer = await makeRenderer(scene, slices[0], tier);
            try {
                const gpu = await renderer.renderSlices(slices);
                assert.ok(countForeground(gpu) > 1000, `${tier}: scene should cover the frame`);

                // Reference: each slice rendered alone and the 8-bit results
                // averaged in float. The GPU averages before quantizing, so the
                // two may differ by the per-slice rounding: at most one level.
                const accum = new Float32Array(gpu.length);
                for (const s of slices) {
                    const img = await renderer.render(s);
                    for (let p = 0; p < img.length; p++) accum[p] += img[p];
                }
                let maxDiff = 0;
                for (let p = 0; p < gpu.length; p++) {
                    const d = Math.abs(gpu[p] - Math.round(accum[p] / slices.length));
                    if (d > maxDiff) maxDiff = d;
                }
                assert.ok(maxDiff <= 1, `${tier}: accumulated frame differs from the slice mean by ${maxDiff} levels`);

                // Two identical slices: their float mean is exact, so the result
                // must equal the single render byte for byte.
                const twice = await renderer.renderSlices([slices[0], slices[0]]);
                compare(twice, await renderer.render(slices[0]), `${tier}: identical slices`);
            } finally {
                renderer.destroy();
            }
        }
    });

    it('a view with nothing in front of the camera renders the background', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');
        const { Vec3 } = await import('playcanvas');
        const scene = await makeScene(500, 11);
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
        for (const tier of TIERS) {
            const renderer = await makeRenderer(scene, camera, tier);
            try {
                const image = await renderer.render(camera);
                assert.strictEqual(countForeground(image), 0, `${tier}: only background`);
            } finally {
                renderer.destroy();
            }
        }
    });
});
