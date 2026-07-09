/**
 * decimateSource orchestrator tests: exact counts, value domains, deep
 * targets (multi-generation with RAM intermediates), spill path with temp
 * cleanup, statistical parity vs the legacy reference, and input validation.
 */

import assert from 'node:assert';
import { mkdtemp, readdir, rm, unlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, it } from 'node:test';

import { legacySimplify } from './fixtures/legacy-decimate.mjs';
import { makeSyntheticSource } from './helpers/synthetic-source.mjs';

import { decimateSource } from '../src/lib/decimate/index.js';

// Read the requested layers in ONE sequential pass (the producer contract:
// each chunk is served exactly once, all layers together).
const LAYER_DIMS = { position: 3, geometric: 8 };
const readLayers = async (src, pool, wanted) => {
    const { meta } = src;
    const out = {};
    const dims = {};
    for (const layer of wanted) {
        dims[layer] = layer === 'color' ?
            meta.layouts.color.stride >> 2 :
            (layer === 'other' ? meta.layouts.other.stride >> 2 : LAYER_DIMS[layer]);
        const Ctor = layer === 'other' ? Uint32Array : Float32Array;
        out[layer] = new Ctor(meta.numGaussians * dims[layer]);
    }
    let base = 0;
    for (let k = 0; k < meta.numChunks[0]; k++) {
        const count = Math.min(meta.chunkSize, meta.numGaussians - k * meta.chunkSize);
        const acquired = {};
        for (const layer of wanted) acquired[layer] = pool.acquire(layer, meta.layouts[layer], count);
        await src.read({ chunkIndex: k, ...acquired });
        for (const layer of wanted) {
            const Ctor = layer === 'other' ? Uint32Array : Float32Array;
            out[layer].set(new Ctor(acquired[layer].data, 0, count * dims[layer]), base * dims[layer]);
            acquired[layer].release();
        }
        base += count;
    }
    return out;
};

const sigmoid = x => 1 / (1 + Math.exp(-x));

const stats = (geo, n) => {
    let opSum = 0, opSq = 0, scSum = 0;
    for (let i = 0; i < n; i++) {
        const op = sigmoid(geo[i * 8 + 7]);
        opSum += op;
        opSq += op * op;
        scSum += (geo[i * 8 + 4] + geo[i * 8 + 5] + geo[i * 8 + 6]) / 3;
    }
    return {
        opMean: opSum / n,
        opStd: Math.sqrt(Math.max(0, opSq / n - (opSum / n) ** 2)),
        scaleMean: scSum / n
    };
};

describe('decimateSource', () => {
    it('50% decimation: exact count, single generation, values in-domain', async () => {
        const n = 4000;
        const { source, pool } = await makeSyntheticSource(n, 1, 17, { chunkSize: 512 });
        const out = await decimateSource(source, pool, { targetCount: 2000 });
        assert.strictEqual(out.meta.numGaussians, 2000);
        const { position: pos, geometric: geo } = await readLayers(out, pool, ["position", "geometric"]);
        await out.close();
        for (let i = 0; i < 2000; i++) {
            const qn = Math.hypot(geo[i * 8], geo[i * 8 + 1], geo[i * 8 + 2], geo[i * 8 + 3]);
            assert.ok(Math.abs(qn - 1) < 1e-3, `row ${i} quat unit (${qn})`);
            for (let c = 0; c < 8; c++) assert.ok(Number.isFinite(geo[i * 8 + c]), `row ${i} geo[${c}] finite`);
            for (let c = 0; c < 3; c++) assert.ok(Number.isFinite(pos[i * 3 + c]), `row ${i} pos[${c}] finite`);
            assert.ok(sigmoid(geo[i * 8 + 7]) <= 1 + 1e-9, `row ${i} opacity <= 1`);
        }
    });

    it('deep target runs multiple generations (RAM intermediates) to the exact count', async () => {
        const n = 4000;
        const { source, pool } = await makeSyntheticSource(n, 0, 19, { chunkSize: 512 });
        const out = await decimateSource(source, pool, { targetCount: 500 }); // 12.5% → 3 generations
        assert.strictEqual(out.meta.numGaussians, 500);
        const { geometric: geo } = await readLayers(out, pool, ["geometric"]);
        await out.close();
        assert.strictEqual(geo.length, 500 * 8);
    });

    it('statistical parity with the legacy algorithm on the same scene', async () => {
        const n = 3000;
        const { source, pool, view } = await makeSyntheticSource(n, 1, 23, { chunkSize: 512 });
        const out = await decimateSource(source, pool, { targetCount: 1500 });
        const { geometric: oursGeo } = await readLayers(out, pool, ["geometric"]);
        await out.close();

        const legacy = legacySimplify(view, 1500);
        const a = stats(oursGeo, 1500);
        const b = stats(legacy.geo, 1500);
        assert.ok(Math.abs(a.opMean - b.opMean) / Math.abs(b.opMean) < 0.05, `opMean ${a.opMean} vs ${b.opMean}`);
        assert.ok(Math.abs(a.scaleMean - b.scaleMean) / Math.abs(b.scaleMean) < 0.05, `scaleMean ${a.scaleMean} vs ${b.scaleMean}`);
        assert.ok(Math.abs(a.opStd - b.opStd) / Math.abs(b.opStd) < 0.10, `opStd ${a.opStd} vs ${b.opStd}`);
    });

    it('spill path: intermediate generations write temp PLYs and clean them up', async () => {
        const n = 3000;
        const { source, pool } = await makeSyntheticSource(n, 0, 29, { chunkSize: 512 });
        const scratchDir = await mkdtemp(join(tmpdir(), 'decimate-spill-'));
        const { NodeFileSystem, NodeReadFileSystem } = await import('../src/cli/node-file-system.js');
        let sawSpill = false;
        const out = await decimateSource(source, pool, {
            targetCount: 400,                      // deep target → intermediates
            memoryBudgetBytes: 1,                  // force the spill path
            spill: {
                writeFs: new NodeFileSystem(),
                readFs: new NodeReadFileSystem(),
                scratchDir,
                remove: async (path) => {
                    sawSpill = true;
                    await unlink(path);
                }
            }
        });
        assert.strictEqual(out.meta.numGaussians, 400);
        const { geometric: geo } = await readLayers(out, pool, ["geometric"]);
        assert.strictEqual(geo.length, 400 * 8);
        await out.close();
        assert.ok(sawSpill, 'spill files were created and removed');
        const leftovers = await readdir(scratchDir);
        assert.deepStrictEqual(leftovers, [], `scratch dir clean (${leftovers})`);
        await rm(scratchDir, { recursive: true, force: true });
    });

    it('extra columns survive decimation', async () => {
        const n = 1000;
        const { source, pool } = await makeSyntheticSource(n, 0, 37, {
            chunkSize: 256,
            extraColumns: [{ name: 'tag', type: 'uint32' }]
        });
        const out = await decimateSource(source, pool, { targetCount: 600 });
        assert.deepStrictEqual([...out.meta.extraColumns.map(e => e.name)], ['tag']);
        const { other } = await readLayers(out, pool, ["other"]);
        await out.close();
        assert.strictEqual(other.length, 600);
    });

    it('target >= N passes the source through; invalid inputs throw', async () => {
        const { source, pool } = await makeSyntheticSource(100, 0, 41, { chunkSize: 64 });
        const out = await decimateSource(source, pool, { targetCount: 100 });
        assert.strictEqual(out, source);

        await assert.rejects(() => decimateSource(source, pool, { targetCount: 0 }), /at least 1/);

        // multi-LOD source rejected
        const { buffers } = await makeSyntheticSource(50, 0, 43, { chunkSize: 1024 });
        const { createInMemoryChunkSource } = await import('../src/lib/chunk/index.js');
        const { Transform } = await import('../src/lib/utils/index.js');
        const slice = (layer, rows, stride) => [layer[0][0].slice(0, rows * stride)];
        const multiLod = createInMemoryChunkSource({
            numGaussians: 50,
            chunkSize: 1024,
            shBands: 0,
            transform: new Transform(),
            lodCounts: [50, 25],
            position: [buffers.position[0], slice(buffers.position, 25, 12)],
            geometric: [buffers.geometric[0], slice(buffers.geometric, 25, 32)],
            color: [buffers.color[0], slice(buffers.color, 25, 12)]
        });
        await assert.rejects(() => decimateSource(multiLod, pool, { targetCount: 10 }), /single-LOD/);
        await multiLod.close();
    });
});
