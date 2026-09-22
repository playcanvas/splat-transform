import assert from 'node:assert';
import { after, before, describe, it } from 'node:test';

import { makeSyntheticSource } from './helpers/synthetic-source.mjs';

import { decimateSourceAdaptive } from '../src/lib/decimate/index.js';

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

describe('decimateSourceAdaptive multi-block adaptive path', () => {
    it('fails clearly when the memory budget requires multiple blocks without WebGPU', async () => {
        const { source, pool } = await makeSyntheticSource(65540, 0, 123, { chunkSize: 1024 });
        await assert.rejects(
            decimateSourceAdaptive(source, pool, { targetCount: 65000, memoryBudgetBytes: 1 }),
            /multi-block adaptive decimation requires WebGPU/
        );
    });

    it('hits the exact quota through in-memory block plans with no scratch storage', { timeout: 120000 }, async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');

        // Dynamic sizing bottoms out at 65,536 rows. Four extra rows force
        // two jittered cores without making this acceptance fixture huge.
        const n = 65540;
        const targetCount = 65000;
        const { source, pool } = await makeSyntheticSource(n, 0, 9876, { chunkSize: 1024 });

        // No spill: a single generation must never need scratch storage.
        const out = await decimateSourceAdaptive(source, pool, {
            targetCount,
            createDevice: async () => device,
            memoryBudgetBytes: 1
        });
        assert.strictEqual(out.meta.numGaussians, targetCount);
        let rows = 0;
        for (let c = 0; c < out.meta.numChunks[0]; c++) {
            const count = Math.min(out.meta.chunkSize, targetCount - rows);
            const position = pool.acquire('position', out.meta.layouts.position, count);
            await out.read({ chunkIndex: c, position });
            for (const value of new Float32Array(position.data, 0, count * 3)) {
                assert.ok(Number.isFinite(value));
            }
            position.release();
            rows += count;
        }
        assert.strictEqual(rows, targetCount);
        await out.close();
    });
});
