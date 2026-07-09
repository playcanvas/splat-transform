/**
 * Decimation acceptance gates (GPU required; suites skip without a WebGPU
 * adapter):
 *
 * 1. GPU vs CPU priority-pass parity — same block pipeline with and without
 *    a device must produce near-identical candidate costs.
 * 2. Rendered-quality acceptance — the spec's gate: rendering the
 *    new-decimated scene must not regress vs the legacy-decimated scene
 *    (PSNR against the undecimated original, 3 viewpoints, ≥ -0.5 dB).
 */

import assert from 'node:assert';
import { after, before, describe, it } from 'node:test';

import { legacySimplify } from './fixtures/legacy-decimate.mjs';
import { makeSyntheticSource } from './helpers/synthetic-source.mjs';
import { kdPartition } from '../src/lib/decimate/partition.js';
import { runPriorityPass } from '../src/lib/decimate/priority.js';

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

describe('GPU vs CPU priority parity', () => {
    it('candidate costs agree within float tolerance', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');

        const n = 5000, k = 16, K = 4;
        const { source, pool, pos } = await makeSyntheticSource(n, 1, 47, { chunkSize: 1024 });
        const { order, blocks } = kdPartition(pos, 1200);

        const run = async (dev) => {
            const cand = {
                idx: new Uint32Array(n * K).fill(0xFFFFFFFF),
                cost: new Float32Array(n * K).fill(Infinity)
            };
            await runPriorityPass({ source, pool, pos, order, blocks, device: dev, K, k }, cand);
            return cand;
        };

        const gpu = await run(device);
        const cpu = await run(undefined);

        let costAgree = 0, idSetAgree = 0;
        for (let g = 0; g < n; g++) {
            const gpuIds = new Set(), cpuIds = new Set();
            let rowCostsAgree = true;
            for (let s = 0; s < K; s++) {
                const cg = gpu.cost[g * K + s], cc = cpu.cost[g * K + s];
                if (Math.abs(cg - cc) > Math.max(1e-3, Math.abs(cc) * 1e-3)) rowCostsAgree = false;
                gpuIds.add(gpu.idx[g * K + s]);
                cpuIds.add(cpu.idx[g * K + s]);
            }
            if (rowCostsAgree) costAgree++;
            const inter = [...gpuIds].filter(x => cpuIds.has(x)).length;
            if (inter >= K - 1) idSetAgree++;   // allow one float-order swap at the K boundary
        }
        assert.ok(costAgree / n >= 0.99, `cost agreement ${(costAgree / n * 100).toFixed(2)}% (want >= 99%)`);
        assert.ok(idSetAgree / n >= 0.95, `candidate-id agreement ${(idSetAgree / n * 100).toFixed(2)}% (want >= 95%)`);
    });
});

describe('rendered-quality acceptance', () => {
    it('new decimation renders no worse than legacy (PSNR gate)', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');

        const { Vec3 } = await import('playcanvas');
        const { Column, DataTable, processDataTable } = await import('../src/lib/index.js');
        const { renderSplats } = await import('../src/lib/render/index.js');

        const n = 4000, target = 2000;
        const { view } = await makeSyntheticSource(n, 1, 53, { chunkSize: 1024 });

        const toTable = (v) => {
            const count = v.pos.length / 3;
            const col = (f) => Float32Array.from({ length: count }, (_, i) => f(i));
            const cols = [
                new Column('x', col(i => v.pos[i * 3])),
                new Column('y', col(i => v.pos[i * 3 + 1])),
                new Column('z', col(i => v.pos[i * 3 + 2])),
                new Column('rot_0', col(i => v.geo[i * 8])),
                new Column('rot_1', col(i => v.geo[i * 8 + 1])),
                new Column('rot_2', col(i => v.geo[i * 8 + 2])),
                new Column('rot_3', col(i => v.geo[i * 8 + 3])),
                new Column('scale_0', col(i => v.geo[i * 8 + 4])),
                new Column('scale_1', col(i => v.geo[i * 8 + 5])),
                new Column('scale_2', col(i => v.geo[i * 8 + 6])),
                new Column('opacity', col(i => v.geo[i * 8 + 7])),
                new Column('f_dc_0', col(i => v.color[i * v.colorDim])),
                new Column('f_dc_1', col(i => v.color[i * v.colorDim + 1])),
                new Column('f_dc_2', col(i => v.color[i * v.colorDim + 2]))
            ];
            for (let r = 0; r < v.colorDim - 3; r++) {
                cols.push(new Column(`f_rest_${r}`, col(i => v.color[i * v.colorDim + 3 + r])));
            }
            return new DataTable(cols);
        };

        const original = toTable(view);
        const legacy = toTable(legacySimplify(view, target));
        const ours = await processDataTable(toTable(view), [{ kind: 'decimate', count: target, percent: null }]);
        assert.strictEqual(ours.numRows, target);

        const cameras = [
            { position: new Vec3(5, 5, 25), target: new Vec3(5, 5, 5) },
            { position: new Vec3(25, 8, 5), target: new Vec3(5, 5, 5) },
            { position: new Vec3(-10, 18, -10), target: new Vec3(5, 5, 5) }
        ].map(c => ({
            ...c,
            up: new Vec3(0, 1, 0),
            fovY: Math.PI / 3,
            width: 256,
            height: 256,
            near: 0.1
        }));
        const background = { r: 0, g: 0, b: 0, a: 1 };

        const psnr = (a, b) => {
            let se = 0;
            for (let i = 0; i < a.length; i++) {
                const d = (a[i] - b[i]) / 255;
                se += d * d;
            }
            const mse = se / a.length;
            return mse === 0 ? Infinity : -10 * Math.log10(mse);
        };

        for (const camera of cameras) {
            const ref = await renderSplats(device, original, camera, background);
            const legacyImg = await renderSplats(device, legacy, camera, background);
            const oursImg = await renderSplats(device, ours, camera, background);
            const psnrLegacy = psnr(legacyImg, ref);
            const psnrOurs = psnr(oursImg, ref);
            assert.ok(
                psnrOurs >= psnrLegacy - 0.5,
                `PSNR regression: ours ${psnrOurs.toFixed(2)} dB vs legacy ${psnrLegacy.toFixed(2)} dB (camera at ${camera.position.x},${camera.position.y},${camera.position.z})`
            );
        }
    });
});
