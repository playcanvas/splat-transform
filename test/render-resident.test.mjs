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
    it('aperture averaging softens an opaque edge and keeps the focus plane sharp', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');
        const { Vec3 } = await import('playcanvas');
        const { Column, DataTable, dataTableToChunkSource } = await import('../src/lib/index.js');
        const { createChunkDataPool } = await import('../src/lib/chunk/index.js');
        const { SceneRenderer } = await import('../src/lib/render/index.js');
        const { buildApertureCameras } = await import('../src/lib/render/camera.js');
        const n = 40 * 80;
        const values = {
            x: 0, y: 0, z: 0,
            rot_0: 1, rot_1: 0, rot_2: 0, rot_3: 0,
            scale_0: Math.log(0.04), scale_1: Math.log(0.04), scale_2: Math.log(1e-8),
            opacity: 10, f_dc_0: 0.5 / 0.28209479177387814,
            f_dc_1: 0.5 / 0.28209479177387814, f_dc_2: 0.5 / 0.28209479177387814
        };
        const table = new DataTable(Object.entries(values).map(([name, value]) => new Column(name, new Float32Array(n).fill(value))));
        for (let i = 0; i < n; i++) {
            table.getColumnByName('x').data[i] = -2 + (i % 40 + 0.5) * 0.05;
            table.getColumnByName('y').data[i] = -2 + (Math.floor(i / 40) + 0.5) * 0.05;
        }
        const source = dataTableToChunkSource(table);
        const pool = createChunkDataPool();
        const camera = {
            position: new Vec3(0, 0, 2), target: new Vec3(0, 0, 0), up: new Vec3(0, 1, 0),
            fovY: Math.PI / 2, width: 64, height: 64, near: 0.125, focusDistance: 4
        };
        for (const tier of TIERS) {
            const renderer = new SceneRenderer(device, source, pool, {
                projection: 'pinhole', width: 64, height: 64, background: { r: 0, g: 0, b: 0, a: 1 }, tier
            });
            await renderer.upload();
            try {
                const sharp = await renderer.render(camera);
                const blurred = await renderer.renderSlices(buildApertureCameras(camera, 1, 64));
                const red = (image, x) => image[(32 * 64 + x) * 4];
                assert.equal(red(sharp, 36), 0, `${tier}: outside the sharp edge`);
                assert.ok(red(blurred, 36) > 10, `${tier}: blur extends into the background`);
                assert.ok(red(sharp, 28) > 240, `${tier}: opaque surface`);
                assert.ok(red(blurred, 28) < 240, `${tier}: background blends through the edge`);
                const focused = await renderer.renderSlices(buildApertureCameras({ ...camera, focusDistance: 2 }, 1, 32));
                for (let i = 0; i < sharp.length; i++) {
                    assert.ok(Math.abs(focused[i] - sharp[i]) <= 1, `${tier}: focus plane changed at byte ${i}`);
                }
            } finally {
                renderer.destroy();
            }
        }
        await source.close();
    });

    it('accumulates defocused splats whose individual opacity is below 1/255', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');
        const { Vec3 } = await import('playcanvas');
        const { Column, DataTable, dataTableToChunkSource } = await import('../src/lib/index.js');
        const { createChunkDataPool } = await import('../src/lib/chunk/index.js');
        const values = {
            x: 0, y: 0, z: 0,
            rot_0: 1, rot_1: 0, rot_2: 0, rot_3: 0,
            scale_0: Math.log(0.01), scale_1: Math.log(0.01), scale_2: Math.log(0.01),
            opacity: 0, f_dc_0: 0, f_dc_1: 0, f_dc_2: 0
        };
        const dataTable = new DataTable(Object.entries(values).map(([name, value]) =>
            new Column(name, new Float32Array(256).fill(value))
        ));
        const source = dataTableToChunkSource(dataTable);
        const scene = { source, pool: createChunkDataPool() };
        // A 10-pixel defocus sigma reduces every splat's peak alpha below
        // 1/255, but their combined contribution should remain visible.
        const camera = {
            position: new Vec3(0, 0, 2), target: new Vec3(0, 0, 0), up: new Vec3(0, 1, 0),
            fovY: Math.PI / 3, width: 64, height: 64, near: 0.125,
            focusDistance: 4, apertureScale: 10
        };
        const renderer = await makeRenderer(scene, camera, 'resident');
        try {
            const image = await renderer.render(camera);
            const red = x => image[(32 * 64 + x) * 4];
            assert.ok(red(32) > 50, 'faint splats must accumulate at the center');
            assert.ok(red(47) > 30, 'faint splat tails must accumulate away from the center');
            assert.strictEqual(red(63), Math.round(background.r * 255), 'outside the footprint stays background');
        } finally {
            renderer.destroy();
            await source.close();
        }
    });

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
            ['off-axis', { ...pinhole, offsetX: 17, offsetY: -9 }],
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

    it('integrates exposure without dark transparent fringes or per-sample highlight clipping', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');
        const { Vec3 } = await import('playcanvas');
        const { SceneRenderer } = await import('../src/lib/render/index.js');
        const { dataTableToChunkSource } = await import('../src/lib/index.js');
        const scene = await makeScene(8, 1);
        for (const column of scene.dataTable.columns) column.data.fill(0);
        scene.dataTable.getColumnByName('rot_0').data.fill(1);
        scene.dataTable.getColumnByName('opacity').data.fill(20);
        const camera = {
            position: new Vec3(0, 0, 2), target: new Vec3(0, 0, 0), up: new Vec3(0, 1, 0),
            fovY: Math.PI / 2, width: 16, height: 16, near: 0.125
        };
        const empty = { ...camera, target: new Vec3(0, 0, 4) };
        const encode = c => c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055;
        for (const tier of TIERS) {
            for (const [color, bgAlpha] of [[1, 1], [1, 0], [1, 0.5], [1.2, 1]]) {
                for (let c = 0; c < 3; c++) scene.dataTable.getColumnByName(`f_dc_${c}`).data.fill((color - 0.5) / 0.28209479177387814);
                const source = dataTableToChunkSource(scene.dataTable);
                const renderer = new SceneRenderer(device, source, scene.pool, {
                    projection: 'pinhole', width: 16, height: 16,
                    background: { r: bgAlpha === 0 ? 0.75 : 0, g: 0, b: 0, a: bgAlpha }, tier
                });
                await renderer.upload();
                try {
                    const image = await renderer.renderSlices([camera, empty]);
                    const pixel = Array.from(image.subarray((8 * 16 + 8) * 4, (8 * 16 + 8) * 4 + 4));
                    const alpha = (1 + bgAlpha) / 2;
                    const linear = ((color + 0.055) / 1.055) ** 2.4;
                    const rgb = Math.round(255 * Math.min(1, encode(linear / (2 * alpha))));
                    const expected = [rgb, rgb, rgb, Math.round(alpha * 255)];
                    for (let c = 0; c < 4; c++) assert.ok(Math.abs(pixel[c] - expected[c]) <= 1, `${tier}, color ${color}, bg alpha ${bgAlpha}: ${pixel} != ${expected}`);
                    if (bgAlpha === 0) {
                        const single = await renderer.render(camera);
                        // Straight RGBA retains white even where surface coverage is partial.
                        for (let p = 0; p < single.length; p += 4) {
                            if (single[p + 3] > 0) assert.equal(single[p], 255);
                        }
                    }
                } finally {
                    renderer.destroy();
                    await source.close();
                }
            }
        }
        await scene.source.close();
    });

    it('GPU slice accumulation matches the linear-light mean of the slices', async (t) => {
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
        for (const tier of TIERS) {
            const renderer = await makeRenderer(scene, slices[0], tier);
            try {
                const gpu = await renderer.renderSlices(slices);
                assert.ok(countForeground(gpu) > 1000, `${tier}: scene should cover the frame`);

                // Reference: each slice rendered alone and the 8-bit results
                // decoded to linear light and averaged. The GPU averages before quantizing, so the
                // two may differ by the per-slice rounding: at most one level.
                const decode = c => c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
                const encode = c => c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055;
                const accum = new Float32Array(gpu.length);
                for (const s of slices) {
                    const img = await renderer.render(s);
                    for (let p = 0; p < img.length; p++) accum[p] += p % 4 === 3 ? img[p] / 255 : decode(img[p] / 255);
                }
                let maxDiff = 0;
                for (let p = 0; p < gpu.length; p++) {
                    const mean = accum[p] / slices.length;
                    const d = Math.abs(gpu[p] - Math.round(255 * (p % 4 === 3 ? mean : encode(mean))));
                    if (d > maxDiff) maxDiff = d;
                }
                assert.ok(maxDiff <= 1, `${tier}: accumulated frame differs from the slice mean by ${maxDiff} levels`);

                // The colour-space round trip can move a value across an
                // 8-bit rounding boundary, but must not visibly change it.
                const twice = await renderer.renderSlices([slices[0], slices[0]]);
                const single = await renderer.render(slices[0]);
                for (let p = 0; p < single.length; p++) {
                    assert.ok(Math.abs(twice[p] - single[p]) <= 1, `${tier}: identical slices at ${p}`);
                }
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

describe('pair prefix unwrap', () => {
    it('recovers prefixes past 2^32 from the u32 scan', async () => {
        const { unwrapPrefixes } = await import('../src/lib/gpu/gpu-scene-rasterizer.js');
        // Four blocks of 1.5 billion pairs: the true prefixes 0, 1.5e9, 3e9,
        // 4.5e9 and total 6e9 wrap twice in u32.
        const block = 1.5e9;
        const truth = [0, block, 2 * block, 3 * block, 4 * block];
        const wrapped = truth.map(v => v % 2 ** 32);
        const out = new Float64Array(5);
        unwrapPrefixes(Uint32Array.from(wrapped.slice(0, 4)), 4, wrapped[4], out);
        assert.deepStrictEqual(Array.from(out), truth);

        // Exactly 2^32 pairs: the total reads back as 0.
        unwrapPrefixes(Uint32Array.from([0, 2 ** 31]), 2, 0, out);
        assert.deepStrictEqual(Array.from(out.subarray(0, 3)), [0, 2 ** 31, 2 ** 32]);

        // Empty blocks repeat a prefix without wrapping.
        unwrapPrefixes(Uint32Array.from([0, 7, 7, 9]), 4, 9, out);
        assert.deepStrictEqual(Array.from(out.subarray(0, 5)), [0, 7, 7, 9, 9]);
    });
});

describe('pair-sort capacity', () => {
    it('reserves complete sorter workgroups when pair counts grow between views', async () => {
        const { GpuSceneRasterizer } = await import('../src/lib/gpu/gpu-scene-rasterizer.js');
        const stop = new Error('stop after checking sort allocation');
        for (const granularity of [2048, 3840]) {
            let allocatedGroups = 0;
            let allocatedElements = 0;
            const sorter = {
                capacity: 0,
                prepareIndirect: () => new Uint32Array([1, granularity, 0, 0]),
                sort(_keys, count) {
                    // Model the engine's allocation contract: changing capacity
                    // within the same workgroup count does not resize buffers.
                    const effective = Math.max(count, this.capacity);
                    const groups = Math.ceil(effective / granularity);
                    if (groups !== allocatedGroups) {
                        allocatedGroups = groups;
                        allocatedElements = effective;
                    }
                    assert.ok(count <= allocatedElements, `${granularity}: ${count} elements exceed ${allocatedElements} allocated`);
                    throw stop;
                }
            };
            const rasterizer = {
                device: {}, radixSort: sorter,
                ensurePairCapacity() {}, dispatch2D() {},
                totalReadback: new Uint32Array(1),
                totalPairsBuffer: { write() {} },
                emitCompute: { setParameter() {} }
            };
            for (const count of [3800, 4097, 6000, 7681, 8000, 4000]) {
                assert.throws(() => GpuSceneRasterizer.prototype.rasterizeCut.call(rasterizer, 0, 1, 0, count, 1, 1), err => err === stop);
            }
        }
    });
});
