/**
 * Unit tests for WebPCodec.
 *
 * Verifies that the wasm module is compiled/instantiated once and shared
 * across instances (per-chunk readers like readLcc2 call create() per chunk),
 * including under concurrent first calls, and that codec instances still
 * round-trip data correctly. Node's test runner isolates files into separate
 * processes, so the static module cache cannot leak into other test files.
 */

import assert from 'node:assert';
import { dirname, join } from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';

import { WebPCodec } from '../src/lib/index.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
WebPCodec.wasmUrl = join(__dirname, '..', 'lib', 'webp.wasm');

describe('WebPCodec', () => {
    it('shares one wasm module across sequential create() calls', async () => {
        const a = await WebPCodec.create();
        const b = await WebPCodec.create();
        assert.ok(a !== b, 'create() returns distinct instances');
        assert.strictEqual(a.Module, b.Module, 'instances share the same wasm module');
    });

    it('shares one wasm module across concurrent create() calls', async () => {
        const [a, b] = await Promise.all([WebPCodec.create(), WebPCodec.create()]);
        assert.strictEqual(a.Module, b.Module, 'concurrent first calls share one instantiation');
    });

    it('round-trips RGBA data losslessly', async () => {
        const codec = await WebPCodec.create();
        const width = 4;
        const height = 4;
        const rgba = new Uint8Array(width * height * 4);
        for (let i = 0; i < rgba.length; i++) {
            rgba[i] = (i * 37) & 0xff;
        }
        const webp = codec.encodeLosslessRGBA(rgba, width, height);
        const decoded = codec.decodeRGBA(webp);
        assert.strictEqual(decoded.width, width);
        assert.strictEqual(decoded.height, height);
        assert.deepStrictEqual(Array.from(decoded.rgba), Array.from(rgba));
    });
});
