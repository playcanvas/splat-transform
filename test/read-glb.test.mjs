import assert from 'node:assert';
import { describe, it } from 'node:test';

import {
    Column,
    DataTable,
    MemoryReadFileSystem,
    Transform,
    createChunkDataPool,
    getInputFormat,
    materializeToDataTable,
    readFile,
    readFileInfo,
    readGlb,
    writeGlb
} from '../src/lib/index.js';
import { MemoryFileSystem } from '../src/lib/io/write/index.js';

const SH_REST_COUNTS = [0, 9, 24, 45];
const ROWS = 5;

const makeTable = (shBands) => {
    const values = (fn) => new Float32Array(Array.from({ length: ROWS }, (_, i) => fn(i)));
    const columns = [
        new Column(
            'x',
            values((i) => i + 0.125)
        ),
        new Column(
            'y',
            values((i) => -i * 2 - 0.25)
        ),
        new Column(
            'z',
            values((i) => i * i + 0.5)
        ),
        new Column(
            'rot_0',
            values((i) => Math.cos((i + 1) * 0.05))
        ),
        new Column(
            'rot_1',
            values((i) => Math.sin((i + 1) * 0.05))
        ),
        new Column(
            'rot_2',
            values(() => 0)
        ),
        new Column(
            'rot_3',
            values(() => 0)
        ),
        new Column(
            'scale_0',
            values((i) => -1 + i * 0.2)
        ),
        new Column(
            'scale_1',
            values((i) => 0.1 + i * 0.15)
        ),
        new Column(
            'scale_2',
            values((i) => 1 - i * 0.1)
        ),
        new Column(
            'opacity',
            values((i) => -2 + i)
        ),
        new Column(
            'f_dc_0',
            values((i) => -0.2 + i * 0.03)
        ),
        new Column(
            'f_dc_1',
            values((i) => 0.1 - i * 0.02)
        ),
        new Column(
            'f_dc_2',
            values((i) => i * 0.04)
        )
    ];
    for (let coefficient = 0; coefficient < SH_REST_COUNTS[shBands]; coefficient++) {
        columns.push(
            new Column(
                `f_rest_${coefficient}`,
                values((i) => coefficient * 0.01 + i * 0.001)
            )
        );
    }
    return new DataTable(columns, Transform.IDENTITY);
};

const writeBytes = async (table) => {
    const fs = new MemoryFileSystem();
    await writeGlb({ filename: 'scene.glb', dataTable: table }, fs);
    return fs.results.get('scene.glb');
};

const encodeGlb = (gltf, bin) => {
    const encodedJson = new TextEncoder().encode(JSON.stringify(gltf));
    const jsonLength = (encodedJson.length + 3) & ~3;
    const binLength = (bin.length + 3) & ~3;
    const result = new Uint8Array(12 + 8 + jsonLength + 8 + binLength);
    const resultView = new DataView(result.buffer);
    resultView.setUint32(0, 0x46546c67, true);
    resultView.setUint32(4, 2, true);
    resultView.setUint32(8, result.length, true);
    resultView.setUint32(12, jsonLength, true);
    resultView.setUint32(16, 0x4e4f534a, true);
    result.fill(0x20, 20, 20 + jsonLength);
    result.set(encodedJson, 20);
    const binHeader = 20 + jsonLength;
    resultView.setUint32(binHeader, binLength, true);
    resultView.setUint32(binHeader + 4, 0x004e4942, true);
    result.set(bin, binHeader + 8);
    return result;
};

const makeInterleavedNormalizedGlb = ({ normalizedScale = true, opacityValues = [64, 192], indices } = {}) => {
    const rows = 2;
    const rowStride = 28;
    const dcOffset = rows * rowStride;
    const indexOffset = dcOffset + rows * 12;
    const bin = new Uint8Array(indexOffset + (indices?.length ?? 0) * 2);
    const view = new DataView(bin.buffer);
    const positions = [
        [1, 2, 3],
        [-4, 5, -6]
    ];
    const rotations = [
        [0, 0, 0, 32767],
        [32767, 0, 0, 0]
    ];
    const scales = normalizedScale
        ? [
              [16384, 32768, 49151],
              [6554, 13107, 19661]
          ]
        : [
              [1, 2, 3],
              [4, 5, 6]
          ];
    const opacities = opacityValues;
    const dc = [
        [0.1, 0.2, 0.3],
        [-0.1, -0.2, -0.3]
    ];
    for (let row = 0; row < rows; row++) {
        const base = row * rowStride;
        for (let c = 0; c < 3; c++) view.setFloat32(base + c * 4, positions[row][c], true);
        for (let c = 0; c < 4; c++) view.setInt16(base + 12 + c * 2, rotations[row][c], true);
        for (let c = 0; c < 3; c++) view.setUint16(base + 20 + c * 2, scales[row][c], true);
        view.setUint8(base + 26, opacities[row]);
        for (let c = 0; c < 3; c++) view.setFloat32(dcOffset + row * 12 + c * 4, dc[row][c], true);
    }
    for (let i = 0; i < (indices?.length ?? 0); i++) view.setUint16(indexOffset + i * 2, indices[i], true);

    const gltf = {
        asset: { version: '2.0' },
        extensionsUsed: ['KHR_gaussian_splatting'],
        scene: 0,
        scenes: [{ nodes: [0] }],
        buffers: [{ byteLength: bin.length }],
        bufferViews: [
            { buffer: 0, byteOffset: 0, byteLength: dcOffset, byteStride: rowStride },
            { buffer: 0, byteOffset: dcOffset, byteLength: rows * 12 },
            ...(indices ? [{ buffer: 0, byteOffset: indexOffset, byteLength: indices.length * 2 }] : [])
        ],
        accessors: [
            { bufferView: 0, byteOffset: 0, componentType: 5126, count: rows, type: 'VEC3' },
            { bufferView: 0, byteOffset: 12, componentType: 5122, normalized: true, count: rows, type: 'VEC4' },
            {
                bufferView: 0,
                byteOffset: 20,
                componentType: 5123,
                ...(normalizedScale ? { normalized: true } : {}),
                count: rows,
                type: 'VEC3'
            },
            { bufferView: 0, byteOffset: 26, componentType: 5121, normalized: true, count: rows, type: 'SCALAR' },
            { bufferView: 1, byteOffset: 0, componentType: 5126, count: rows, type: 'VEC3' },
            ...(indices
                ? [{ bufferView: 2, byteOffset: 0, componentType: 5123, count: indices.length, type: 'SCALAR' }]
                : [])
        ],
        meshes: [
            {
                primitives: [
                    {
                        mode: 0,
                        attributes: {
                            POSITION: 0,
                            'KHR_gaussian_splatting:ROTATION': 1,
                            'KHR_gaussian_splatting:SCALE': 2,
                            'KHR_gaussian_splatting:OPACITY': 3,
                            'KHR_gaussian_splatting:SH_DEGREE_0_COEF_0': 4
                        },
                        ...(indices ? { indices: 5 } : {}),
                        extensions: {
                            KHR_gaussian_splatting: {
                                kernel: 'ellipse',
                                colorSpace: 'srgb_rec709_display'
                            }
                        }
                    }
                ]
            }
        ],
        nodes: [{ mesh: 0 }]
    };
    return { bytes: encodeGlb(gltf, bin), gltf, bin, positions, rotations, scales, opacities, dc };
};

const assertTablesClose = (actual, expected, tolerance = 2e-5) => {
    assert.strictEqual(actual.numRows, expected.numRows);
    assert.deepStrictEqual(
        actual.columns.map((column) => column.name),
        expected.columns.map((column) => column.name)
    );
    for (const expectedColumn of expected.columns) {
        const actualData = actual.getColumnByName(expectedColumn.name).data;
        for (let i = 0; i < expectedColumn.data.length; i++) {
            assert.ok(
                Math.abs(actualData[i] - expectedColumn.data[i]) <= tolerance,
                `${expectedColumn.name}[${i}]: expected ${expectedColumn.data[i]}, got ${actualData[i]}`
            );
        }
    }
};

describe('GLB reader', () => {
    it('reads a GLB written by writeGlb back as the same DataTable', async () => {
        const expected = makeTable(3);
        const fs = new MemoryReadFileSystem();
        fs.set('round-trip.glb', await writeBytes(expected));

        const [source] = await readFile({
            filename: 'round-trip.glb',
            inputFormat: getInputFormat('round-trip.glb'),
            fileSystem: fs
        });
        try {
            const actual = await materializeToDataTable(source, createChunkDataPool());
            assertTablesClose(actual, expected);
        } finally {
            await source.close();
        }
    });

    for (let shBands = 0; shBands <= 3; shBands++) {
        it(`round-trips writer output with ${shBands} SH bands`, async () => {
            const expected = makeTable(shBands);
            const bytes = await writeBytes(expected);
            const fs = new MemoryReadFileSystem();
            fs.set('scene.glb', bytes);
            const pool = createChunkDataPool({ chunkSize: 2 });
            const source = await readGlb(await fs.createSource('scene.glb'), pool);
            try {
                assert.strictEqual(source.meta.shBands, shBands);
                assert.ok(source.meta.transform.equals(Transform.IDENTITY));
                assert.strictEqual(source.meta.numChunks[0], 3);
                assertTablesClose(await materializeToDataTable(source, pool), expected);
            } finally {
                await source.close();
            }
        });
    }

    it('supports shuffled and duplicate gather reads', async () => {
        const expected = makeTable(1);
        const bytes = await writeBytes(expected);
        const fs = new MemoryReadFileSystem();
        fs.set('scene.glb', bytes);
        const pool = createChunkDataPool({ chunkSize: 8 });
        const source = await readGlb(await fs.createSource('scene.glb'), pool);
        const indices = new Uint32Array([4, 1, 4, 0]);
        const count = indices.length;
        const position = pool.acquire('position', source.meta.layouts.position, count);
        const geometric = pool.acquire('geometric', source.meta.layouts.geometric, count);
        const color = pool.acquire('color', source.meta.layouts.color, count);
        try {
            await source.read({ indices, indexOffset: 0, count, position, geometric, color });
            const positions = new Float32Array(position.data);
            const scalesAndOpacity = new Float32Array(geometric.data);
            const colors = new Float32Array(color.data);
            for (let slot = 0; slot < count; slot++) {
                const row = indices[slot];
                assert.strictEqual(positions[slot * 3], expected.getColumnByName('x').data[row]);
                assert.ok(
                    Math.abs(scalesAndOpacity[slot * 8 + 4] - expected.getColumnByName('scale_0').data[row]) < 2e-5
                );
                assert.strictEqual(colors[slot * 12], expected.getColumnByName('f_dc_0').data[row]);
                assert.strictEqual(colors[slot * 12 + 3], expected.getColumnByName('f_rest_0').data[row]);
            }
        } finally {
            position.release();
            geometric.release();
            color.release();
            await source.close();
        }
    });

    it('decodes interleaved normalized KHR accessors', async () => {
        const fixture = makeInterleavedNormalizedGlb();
        const fs = new MemoryReadFileSystem();
        fs.set('normalized.glb', fixture.bytes);
        const pool = createChunkDataPool({ chunkSize: 2 });
        const source = await readGlb(await fs.createSource('normalized.glb'), pool);
        try {
            const table = await materializeToDataTable(source, pool);
            for (let row = 0; row < 2; row++) {
                assert.strictEqual(table.getColumnByName('x').data[row], fixture.positions[row][0]);
                assert.ok(
                    Math.abs(table.getColumnByName('rot_0').data[row] - fixture.rotations[row][3] / 32767) < 1e-6
                );
                assert.ok(
                    Math.abs(table.getColumnByName('rot_1').data[row] - fixture.rotations[row][0] / 32767) < 1e-6
                );
                assert.ok(
                    Math.abs(table.getColumnByName('scale_0').data[row] - Math.log(fixture.scales[row][0] / 65535)) <
                        1e-6
                );
                const alpha = fixture.opacities[row] / 255;
                assert.ok(Math.abs(table.getColumnByName('opacity').data[row] - Math.log(alpha / (1 - alpha))) < 1e-5);
                assert.ok(Math.abs(table.getColumnByName('f_dc_2').data[row] - fixture.dc[row][2]) < 1e-6);
            }
        } finally {
            await source.close();
        }
    });

    it('decodes unnormalized integer SCALE accessors', async () => {
        const fixture = makeInterleavedNormalizedGlb({ normalizedScale: false });
        const fs = new MemoryReadFileSystem();
        fs.set('integer-scale.glb', fixture.bytes);
        const pool = createChunkDataPool({ chunkSize: 2 });
        const source = await readGlb(await fs.createSource('integer-scale.glb'), pool);
        try {
            const table = await materializeToDataTable(source, pool);
            for (let row = 0; row < 2; row++) {
                for (let component = 0; component < 3; component++) {
                    assert.ok(
                        Math.abs(
                            table.getColumnByName(`scale_${component}`).data[row] -
                                Math.log(fixture.scales[row][component])
                        ) < 1e-6
                    );
                }
            }
        } finally {
            await source.close();
        }
    });

    it('preserves fully transparent and opaque opacity endpoints', async () => {
        const fixture = makeInterleavedNormalizedGlb({ opacityValues: [0, 255] });
        const fs = new MemoryReadFileSystem();
        fs.set('opacity-endpoints.glb', fixture.bytes);
        const pool = createChunkDataPool({ chunkSize: 2 });
        const source = await readGlb(await fs.createSource('opacity-endpoints.glb'), pool);
        try {
            const opacity = (await materializeToDataTable(source, pool)).getColumnByName('opacity').data;
            assert.strictEqual(opacity[0], -Infinity);
            assert.strictEqual(opacity[1], Infinity);
        } finally {
            await source.close();
        }
    });

    it('honors POINTS indices and their order', async () => {
        const fixture = makeInterleavedNormalizedGlb({ indices: [1, 0, 1] });
        const fs = new MemoryReadFileSystem();
        fs.set('indexed.glb', fixture.bytes);
        const pool = createChunkDataPool({ chunkSize: 2 });
        const source = await readGlb(await fs.createSource('indexed.glb'), pool);
        try {
            assert.strictEqual(source.meta.numGaussians, 3);
            const table = await materializeToDataTable(source, pool);
            assert.deepStrictEqual(Array.from(table.getColumnByName('x').data), [-4, 1, -4]);
            const dc = table.getColumnByName('f_dc_0').data;
            assert.ok(Math.abs(dc[0] + 0.1) < 1e-6);
            assert.ok(Math.abs(dc[1] - 0.1) < 1e-6);
            assert.ok(Math.abs(dc[2] + 0.1) < 1e-6);
        } finally {
            await source.close();
        }
    });

    it('composes shuffled gathers with POINTS indices', async () => {
        const fixture = makeInterleavedNormalizedGlb({ indices: [1, 0, 1] });
        const fs = new MemoryReadFileSystem();
        fs.set('indexed-gather.glb', fixture.bytes);
        const pool = createChunkDataPool({ chunkSize: 8 });
        const source = await readGlb(await fs.createSource('indexed-gather.glb'), pool);
        const logicalRows = new Uint32Array([2, 0, 2, 1]);
        const position = pool.acquire('position', source.meta.layouts.position, logicalRows.length);
        try {
            await source.read({
                indices: logicalRows,
                indexOffset: 0,
                count: logicalRows.length,
                position
            });
            const values = new Float32Array(position.data).subarray(0, logicalRows.length * 3);
            assert.deepStrictEqual(Array.from(values.filter((_, i) => i % 3 === 0)), [-4, -4, -4, 1]);
        } finally {
            position.release();
            await source.close();
        }
    });

    it('rejects a POINTS index outside the attribute range', async () => {
        const fixture = makeInterleavedNormalizedGlb({ indices: [2] });
        const fs = new MemoryReadFileSystem();
        fs.set('bad-index.glb', fixture.bytes);
        const pool = createChunkDataPool({ chunkSize: 2 });
        const source = await readGlb(await fs.createSource('bad-index.glb'), pool);
        try {
            await assert.rejects(materializeToDataTable(source, pool), /primitive index 2 is out of range/);
        } finally {
            await source.close();
        }
    });

    it('requires the ellipse kernel and a supported color space', async () => {
        for (const missing of ['kernel', 'colorSpace']) {
            const fixture = makeInterleavedNormalizedGlb();
            delete fixture.gltf.meshes[0].primitives[0].extensions.KHR_gaussian_splatting[missing];
            const fs = new MemoryReadFileSystem();
            fs.set('invalid-extension.glb', encodeGlb(fixture.gltf, fixture.bin));
            await assert.rejects(
                readGlb(await fs.createSource('invalid-extension.glb'), createChunkDataPool()),
                missing === 'colorSpace' ? /color.?space/i : /kernel/i
            );
        }

        for (const [property, value] of [
            ['kernel', 'box'],
            ['colorSpace', 'acescg']
        ]) {
            const fixture = makeInterleavedNormalizedGlb();
            fixture.gltf.meshes[0].primitives[0].extensions.KHR_gaussian_splatting[property] = value;
            const fs = new MemoryReadFileSystem();
            fs.set('unsupported-extension.glb', encodeGlb(fixture.gltf, fixture.bin));
            await assert.rejects(
                readGlb(await fs.createSource('unsupported-extension.glb'), createChunkDataPool()),
                property === 'colorSpace' ? /color.?space/i : /kernel/i
            );
        }
    });

    it('rejects normalized float accessors', async () => {
        const fixture = makeInterleavedNormalizedGlb();
        fixture.gltf.accessors[0].normalized = true;
        const fs = new MemoryReadFileSystem();
        fs.set('normalized-float.glb', encodeGlb(fixture.gltf, fixture.bin));
        await assert.rejects(
            readGlb(await fs.createSource('normalized-float.glb'), createChunkDataPool()),
            /float accessor 'POSITION' must not be normalized/
        );
    });

    it('ignores unrelated node transforms but rejects a transformed Gaussian node', async () => {
        const unrelated = makeInterleavedNormalizedGlb();
        unrelated.gltf.nodes.push({ translation: [1, 2, 3] });
        unrelated.gltf.scenes[0].nodes.push(1);
        const unrelatedFs = new MemoryReadFileSystem();
        unrelatedFs.set('unrelated-transform.glb', encodeGlb(unrelated.gltf, unrelated.bin));
        const source = await readGlb(
            await unrelatedFs.createSource('unrelated-transform.glb'),
            createChunkDataPool({ chunkSize: 2 })
        );
        await source.close();

        const transformed = makeInterleavedNormalizedGlb();
        transformed.gltf.nodes[0].translation = [1, 2, 3];
        const transformedFs = new MemoryReadFileSystem();
        transformedFs.set('transformed.glb', encodeGlb(transformed.gltf, transformed.bin));
        await assert.rejects(
            readGlb(await transformedFs.createSource('transformed.glb'), createChunkDataPool()),
            /node transform/
        );

        const transformedParent = makeInterleavedNormalizedGlb();
        transformedParent.gltf.nodes.push({ children: [0], translation: [1, 2, 3] });
        transformedParent.gltf.scenes[0].nodes = [1];
        const transformedParentFs = new MemoryReadFileSystem();
        transformedParentFs.set('transformed-parent.glb', encodeGlb(transformedParent.gltf, transformedParent.bin));
        await assert.rejects(
            readGlb(await transformedParentFs.createSource('transformed-parent.glb'), createChunkDataPool()),
            /node transform/
        );
    });

    it('integrates with format detection and file info', async () => {
        assert.strictEqual(getInputFormat('https://example.com/scene.GLB?signature=abc'), 'glb');
        const fs = new MemoryReadFileSystem();
        fs.set('scene.glb', await writeBytes(makeTable(2)));
        const info = await readFileInfo({ filename: 'scene.glb', inputFormat: 'glb', fileSystem: fs });
        assert.deepStrictEqual(
            { format: info.format, gaussian: info.gaussian, numGaussians: info.numGaussians, shBands: info.shBands },
            { format: 'glb', gaussian: true, numGaussians: ROWS, shBands: 2 }
        );
    });

    it('rejects an invalid declared length', async () => {
        const bytes = (await writeBytes(makeTable(0))).slice();
        new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).setUint32(8, bytes.length - 4, true);
        const fs = new MemoryReadFileSystem();
        fs.set('bad.glb', bytes);
        await assert.rejects(readFile({ filename: 'bad.glb', inputFormat: 'glb', fileSystem: fs }), /declared length/);
    });
});
