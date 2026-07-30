import assert from 'node:assert';
import { after, before, describe, it } from 'node:test';

import { makeSyntheticSource } from './helpers/synthetic-source.mjs';

import { decimateSource } from '../src/lib/decimate/index.js';
import { MemoryReadSource } from '../src/lib/io/read/memory-file-system.js';
import { MemoryFileSystem } from '../src/lib/io/write/memory-file-system.js';

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

describe('decimateSource multi-block adaptive path', () => {
    it('fails clearly when the memory budget requires multiple blocks without WebGPU', async () => {
        const { source, pool } = await makeSyntheticSource(65540, 0, 123, { chunkSize: 1024 });
        await assert.rejects(
            decimateSource(source, pool, { targetCount: 65000, memoryBudgetBytes: 1 }),
            /multi-block adaptive decimation requires WebGPU/
        );
    });

    it('hits the exact quota through scratch plans and removes every plan on close', { timeout: 120000 }, async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');

        // Dynamic sizing bottoms out at 65,536 rows. Four extra rows force
        // two jittered cores without making this acceptance fixture huge.
        const n = 65540;
        const targetCount = 65000;
        const { source, pool } = await makeSyntheticSource(n, 0, 9876, { chunkSize: 1024 });
        const writeFs = new MemoryFileSystem();
        const spill = {
            writeFs,
            readFs: {
                async createSource(path) {
                    const bytes = writeFs.results.get(path);
                    if (!bytes) throw new Error(`missing scratch file ${path}`);
                    return new MemoryReadSource(bytes);
                }
            },
            scratchDir: 'scratch',
            async remove(path) {
                writeFs.results.delete(path);
            }
        };

        const out = await decimateSource(source, pool, {
            targetCount,
            createDevice: async () => device,
            memoryBudgetBytes: 1,
            spill
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
        assert.strictEqual(writeFs.results.size, 0, 'all block plans cleaned');
    });
});
