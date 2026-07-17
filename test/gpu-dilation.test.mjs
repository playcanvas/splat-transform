import { describe, it } from 'node:test';
import assert from 'node:assert';

import { GpuDilation } from '../src/lib/gpu/gpu-dilation.js';
import { extractWgsl } from '../src/lib/gpu/shaders/dilation.js';

describe('GpuDilation', function () {
    it('packs Float64 block-mask keys for the u32 shader table', function () {
        let uploadedKeys;
        const gpu = Object.create(GpuDilation.prototype);
        gpu.srcTypesBuffer = { write() {} };
        gpu.srcKeysBuffer = {
            write(_offset, data, dataOffset, count) {
                uploadedKeys = data.slice(dataOffset, dataOffset + count);
            }
        };
        gpu.srcLoBuffer = { write() {} };
        gpu.srcHiBuffer = { write() {} };
        gpu.srcTypesCapacity = Number.MAX_SAFE_INTEGER;
        gpu.srcMasksCapacity = Number.MAX_SAFE_INTEGER;

        const keys = new Float64Array([
            -1,
            0,
            0x7FFFFFFF,
            0x80000000,
            0xFFFFFFFE,
            0xFFFFFFFF,
            -1,
            -1
        ]);
        gpu.uploadSrc({
            types: new Uint32Array(1),
            masks: {
                keys,
                lo: new Uint32Array(keys.length),
                hi: new Uint32Array(keys.length)
            },
            nbx: 1,
            nby: 1,
            nbz: 1,
            bStride: 1
        });

        assert.ok(uploadedKeys instanceof Uint32Array);
        assert.deepStrictEqual([...uploadedKeys], [
            0xFFFFFFFF,
            0,
            0x7FFFFFFF,
            0x80000000,
            0xFFFFFFFE,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF
        ]);
    });

    it('bounds shader hash probes to the uploaded table capacity', function () {
        const shader = extractWgsl();
        assert.match(shader, /var probes: u32 = 0u;/);
        assert.match(shader, /probes == u\.srcCapMinusOne/);
    });
});
