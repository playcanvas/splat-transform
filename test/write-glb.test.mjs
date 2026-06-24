/**
 * Tests for the GLB writer's conformance to the KHR_gaussian_splatting spec.
 *
 * The spec requires SCALE to be linear and non-negative and OPACITY to be a
 * normalized linear value in [0, 1]. splat-transform stores scale in log-space
 * and opacity as a logit internally (PLY convention), so the writer must apply
 * exp() to scale and sigmoid() to opacity on export.
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

import { Column, DataTable, writeGlb } from '../src/lib/index.js';
import { MemoryFileSystem } from '../src/lib/io/write/index.js';

const sigmoid = v => 1 / (1 + Math.exp(-v));

// glTF constants
const FLOAT = 5126;
const GLB_MAGIC = 0x46546C67;
const JSON_CHUNK_TYPE = 0x4E4F534A;
const BIN_CHUNK_TYPE = 0x004E4942;
const COMPONENTS = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };

// Per-splat inputs in the internal (PLY) representation: scale in log-space,
// opacity as a logit. Chosen to exercise negative, zero and positive logs.
const SPLATS = [
    { scale: [-2.0, 0.0, 1.5], opacity: 0.0 },
    { scale: [3.0, -1.0, 0.25], opacity: -2.0 },
    { scale: [-5.0, 4.0, 0.0], opacity: 3.0 }
];

const makeTable = () => {
    const n = SPLATS.length;
    const col = pick => new Float32Array(SPLATS.map(pick));
    return new DataTable([
        new Column('x', new Float32Array(n)),
        new Column('y', new Float32Array(n)),
        new Column('z', new Float32Array(n)),
        new Column('rot_0', new Float32Array(n).fill(1)),
        new Column('rot_1', new Float32Array(n)),
        new Column('rot_2', new Float32Array(n)),
        new Column('rot_3', new Float32Array(n)),
        new Column('scale_0', col(s => s.scale[0])),
        new Column('scale_1', col(s => s.scale[1])),
        new Column('scale_2', col(s => s.scale[2])),
        new Column('f_dc_0', new Float32Array(n)),
        new Column('f_dc_1', new Float32Array(n)),
        new Column('f_dc_2', new Float32Array(n)),
        new Column('opacity', col(s => s.opacity))
    ]);
};

// Parse a GLB blob into its glTF JSON and BIN chunk.
const parseGlb = (bytes) => {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    assert.strictEqual(view.getUint32(0, true), GLB_MAGIC, 'GLB magic');
    const length = view.getUint32(8, true);

    let json = null;
    let bin = null;
    let offset = 12;
    while (offset < length) {
        const chunkLength = view.getUint32(offset, true);
        const chunkType = view.getUint32(offset + 4, true);
        const data = bytes.subarray(offset + 8, offset + 8 + chunkLength);
        if (chunkType === JSON_CHUNK_TYPE) {
            json = JSON.parse(new TextDecoder().decode(data));
        } else if (chunkType === BIN_CHUNK_TYPE) {
            bin = data;
        }
        offset += 8 + chunkLength;
    }
    assert.ok(json, 'JSON chunk present');
    assert.ok(bin, 'BIN chunk present');
    return { json, bin };
};

// Read a float attribute back out of the BIN chunk via its accessor/bufferView.
const readAttribute = (json, bin, name) => {
    const accessorIndex = json.meshes[0].primitives[0].attributes[name];
    assert.notStrictEqual(accessorIndex, undefined, `attribute ${name} present`);
    const accessor = json.accessors[accessorIndex];
    const bufferView = json.bufferViews[accessor.bufferView];
    const numComponents = COMPONENTS[accessor.type];
    const base = (bufferView.byteOffset ?? 0) + (accessor.byteOffset ?? 0);

    const view = new DataView(bin.buffer, bin.byteOffset + base, accessor.count * numComponents * 4);
    const values = new Float32Array(accessor.count * numComponents);
    for (let i = 0; i < values.length; i++) {
        values[i] = view.getFloat32(i * 4, true);
    }
    return { accessor, numComponents, values };
};

const writeAndParse = async () => {
    const fs = new MemoryFileSystem();
    await writeGlb({ filename: '/out.glb', dataTable: makeTable() }, fs);
    const bytes = fs.results.get('/out.glb');
    assert.ok(bytes && bytes.byteLength > 0, 'GLB written');
    return parseGlb(bytes);
};

// Float32-friendly relative+absolute tolerance.
const approx = (actual, expected) => Math.abs(actual - expected) <= 1e-4 * Math.max(1, Math.abs(expected));

describe('writeGlb KHR_gaussian_splatting conformance', function () {
    it('declares the KHR_gaussian_splatting extension', async function () {
        const { json } = await writeAndParse();
        assert.ok(json.extensionsUsed.includes('KHR_gaussian_splatting'));
    });

    it('writes SCALE in linear space (exp of the log-space input)', async function () {
        const { json, bin } = await writeAndParse();
        const { accessor, values } = readAttribute(json, bin, 'KHR_gaussian_splatting:SCALE');

        assert.strictEqual(accessor.type, 'VEC3');
        assert.strictEqual(accessor.componentType, FLOAT);
        assert.strictEqual(accessor.count, SPLATS.length);

        SPLATS.forEach((splat, i) => {
            splat.scale.forEach((logScale, c) => {
                const got = values[i * 3 + c];
                assert.ok(approx(got, Math.exp(logScale)),
                    `splat ${i} scale[${c}]: expected exp(${logScale})=${Math.exp(logScale)}, got ${got}`);
            });
        });
    });

    it('writes strictly non-negative SCALE values', async function () {
        const { json, bin } = await writeAndParse();
        const { values } = readAttribute(json, bin, 'KHR_gaussian_splatting:SCALE');
        for (const v of values) {
            assert.ok(v >= 0, `scale value must be non-negative, got ${v}`);
        }
    });

    it('writes OPACITY as sigmoid-activated linear values in [0, 1]', async function () {
        const { json, bin } = await writeAndParse();
        const { accessor, values } = readAttribute(json, bin, 'KHR_gaussian_splatting:OPACITY');

        assert.strictEqual(accessor.type, 'SCALAR');
        assert.strictEqual(accessor.count, SPLATS.length);

        SPLATS.forEach((splat, i) => {
            const got = values[i];
            assert.ok(approx(got, sigmoid(splat.opacity)),
                `splat ${i} opacity: expected sigmoid(${splat.opacity})=${sigmoid(splat.opacity)}, got ${got}`);
            assert.ok(got >= 0 && got <= 1, `opacity must be in [0, 1], got ${got}`);
        });
    });
});
