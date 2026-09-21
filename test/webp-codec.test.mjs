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

    it('uses the default WebP encoder unless an effort level is explicitly provided', async (t) => {
        const codec = await WebPCodec.create();
        const defaultEncoder = t.mock.method(codec.Module, '_webp_encode_lossless_rgba');
        const preset = t.mock.method(codec.Module, '_webp_encode_lossless_rgba_level');
        const rgba = new Uint8Array([37, 74, 111, 255]);

        codec.encodeLosslessRGBA(rgba, 1, 1);
        codec.encodeLosslessRGBA(rgba, 1, 1, 4, undefined);
        assert.strictEqual(defaultEncoder.mock.callCount(), 2);
        assert.strictEqual(preset.mock.callCount(), 0);

        for (const effort of [0, 9]) {
            rgba[3] = 0;
            const webp = codec.encodeLosslessRGBA(rgba, 1, 1, 4, effort);
            assert.deepStrictEqual(codec.decodeRGBA(webp).rgba, rgba);
            assert.strictEqual(preset.mock.calls.at(-1).arguments[4], effort);
        }
        assert.strictEqual(defaultEncoder.mock.callCount(), 2);
        assert.strictEqual(preset.mock.callCount(), 2);
    });
});
