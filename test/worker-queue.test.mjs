/**
 * Tests for the cross-platform worker queue.
 *
 * Running from source via tsx leaves the worker-source placeholder null, so
 * the queue runs every task inline on the calling thread - which is exactly
 * the transport these tests exercise. The worker transport runs the same
 * handler code and is covered by the built CLI.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { Column, DataTable, WorkerQueue } from '../src/lib/index.js';
import { quantize1d } from '../src/lib/spatial/quantize-1d.js';
import { runQuantize1d, runEncodeWebp } from '../src/lib/workers/index.js';
import { WebPCodec } from '../src/lib/utils/webp-codec.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

// running from source via tsx: point the codec at the repo's wasm binary
WebPCodec.wasmUrl = join(__dirname, '..', 'lib', 'webp.wasm');

describe('worker queue', () => {
    it('runs inline when running from source', () => {
        assert.strictEqual(WorkerQueue.isInline, true);
    });

    it('runQuantize1d matches direct quantize1d output', async () => {
        const data = new Float32Array(10000);
        for (let i = 0; i < data.length; ++i) {
            data[i] = Math.sin(i * 0.37) * 5;
        }
        const makeTable = () => new DataTable([new Column('a', data.slice())]);

        const direct = quantize1d(makeTable());
        const viaQueue = await runQuantize1d(makeTable());

        assert.deepStrictEqual(
            Array.from(viaQueue.centroids.getColumn(0).data),
            Array.from(direct.centroids.getColumn(0).data)
        );
        assert.deepStrictEqual(
            Array.from(viaQueue.labels.getColumn(0).data),
            Array.from(direct.labels.getColumn(0).data)
        );
    });

    it('runEncodeWebp produces a decodable lossless webp', async () => {
        const width = 32;
        const height = 16;
        const rgba = new Uint8Array(width * height * 4);
        for (let i = 0; i < rgba.length; ++i) {
            rgba[i] = (i * 7) & 0xff;
        }
        const original = rgba.slice();

        const webp = await runEncodeWebp(rgba, width, height);
        assert.ok(webp.byteLength > 0);

        const codec = await WebPCodec.create();
        const decoded = codec.decodeRGBA(webp);
        assert.strictEqual(decoded.width, width);
        assert.strictEqual(decoded.height, height);
        assert.deepStrictEqual(Array.from(decoded.rgba), Array.from(original));
    });

    it('maxWorkers = 0 forces inline mode', async () => {
        const previous = WorkerQueue.maxWorkers;
        WorkerQueue.maxWorkers = 0;
        try {
            assert.strictEqual(WorkerQueue.isInline, true);
            const result = await WorkerQueue.run('quantize1d', {
                columns: [{ name: 'a', data: new Float32Array([1, 2, 3, 4]) }]
            });
            assert.strictEqual(result.centroids.length, 256);
            assert.strictEqual(result.labels[0].data.length, 4);
        } finally {
            WorkerQueue.maxWorkers = previous;
        }
    });

    it('rejects with the handler error for a failing task', async () => {
        // invalid dimensions make the wasm encoder fail
        await assert.rejects(
            () => WorkerQueue.run('encodeWebp', { rgba: new Uint8Array(4), width: 0, height: 0 }),
            /encode failed|WebP/i
        );
    });

    it('destroy() resolves with no workers running', async () => {
        await WorkerQueue.destroy();
    });
});
