/**
 * Format conversion tests for splat-transform.
 * Tests round-trip conversions and input-only format reading.
 */

import { describe, it, before } from 'node:test';
import assert from 'node:assert';
import { readFile as fsReadFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
    Column,
    DataTable,
    Transform,
    computeSummary,
    getInputFormat,
    readFile,
    readPly,
    readSplat,
    readKsplat,
    readSpz,
    readSog,
    writePly,
    writeCompressedPly,
    writeSog,
    writeSpz,
    writeCsv,
    MemoryReadFileSystem,
    MemoryFileSystem,
    ZipReadFileSystem,
    WebPCodec
} from '../src/lib/index.js';

import { compareSummaries, compareDataTables } from './helpers/summary-compare.mjs';
import { createMinimalTestData, createTestDataTable, encodePlyBinary } from './helpers/test-utils.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const fixturesDir = join(__dirname, 'fixtures', 'splat');
WebPCodec.wasmUrl = join(__dirname, '..', 'lib', 'webp.wasm');

const SH_C0 = 0.28209479177387814;
const logit = value => Math.log(value / (1 - value));

/**
 * Creates a ReadSource from a Uint8Array for testing readers.
 */
class BufferReadSource {
    constructor(data) {
        this.data = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
        this.size = this.data.length;
        this.seekable = true;
        this.closed = false;
    }

    read(start = 0, end = this.size) {
        if (this.closed) throw new Error('Source is closed');
        const clampedStart = Math.max(0, Math.min(start, this.size));
        const clampedEnd = Math.max(clampedStart, Math.min(end, this.size));
        return new BufferReadStream(this.data, clampedStart, clampedEnd);
    }

    close() {
        this.closed = true;
    }
}

class BufferReadStream {
    constructor(data, start, end) {
        this.data = data;
        this.offset = start;
        this.end = end;
        this.size = end - start;
        this.bytesRead = 0;
    }

    async pull(target) {
        const remaining = this.end - this.offset;
        if (remaining <= 0) return 0;

        const bytesToCopy = Math.min(target.length, remaining);
        target.set(this.data.subarray(this.offset, this.offset + bytesToCopy));
        this.offset += bytesToCopy;
        this.bytesRead += bytesToCopy;
        return bytesToCopy;
    }

    async readAll() {
        const result = this.data.subarray(this.offset, this.end);
        this.bytesRead += result.length;
        this.offset = this.end;
        return result;
    }
}

const createSpzTestData = ({ shDegree = 0 } = {}) => {
    const dataTable = createTestDataTable(2, {
        includeSH: shDegree > 0 && shDegree <= 3,
        shBands: Math.min(shDegree, 3)
    });
    dataTable.transform = Transform.PLY.clone();

    const rot0 = dataTable.getColumnByName('rot_0').data;
    const rot1 = dataTable.getColumnByName('rot_1').data;
    const rot2 = dataTable.getColumnByName('rot_2').data;
    const rot3 = dataTable.getColumnByName('rot_3').data;

    rot0[1] = Math.cos(Math.PI / 4);
    rot1[1] = 0;
    rot2[1] = 0;
    rot3[1] = Math.sin(Math.PI / 4);

    if (shDegree === 4) {
        for (let i = 0; i < 72; i++) {
            dataTable.columns.push(new Column(`f_rest_${i}`, new Float32Array(dataTable.numRows)));
        }
    }

    return dataTable;
};

const addSpzV4HeaderExtensions = (bytes, extensionBytes) => {
    const headerSize = 32;
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const originalTocByteOffset = view.getUint32(16, true);
    const result = new Uint8Array(bytes.length + extensionBytes.length);

    result.set(bytes.subarray(0, headerSize), 0);
    result.set(extensionBytes, headerSize);
    result.set(bytes.subarray(originalTocByteOffset), originalTocByteOffset + extensionBytes.length);

    const resultView = new DataView(result.buffer);
    resultView.setUint8(14, resultView.getUint8(14) | 0x2);
    resultView.setUint32(16, originalTocByteOffset + extensionBytes.length, true);

    return result;
};

const createSpzFixture = async ({ version = 4, shDegree = 0, extensionBytes = null } = {}) => {
    const writeFs = new MemoryFileSystem();
    await writeSpz({
        filename: 'fixture.spz',
        dataTable: createSpzTestData({ shDegree }),
        version
    }, writeFs);

    const bytes = writeFs.results.get('fixture.spz');
    return extensionBytes ? addSpzV4HeaderExtensions(bytes, extensionBytes) : bytes;
};

describe('PLY Format', () => {
    let testData;
    let plyBytes;
    let expectedSummary;

    before(() => {
        testData = createMinimalTestData();
        testData.transform = Transform.PLY.clone();
        plyBytes = encodePlyBinary(testData);
        expectedSummary = computeSummary(testData);
    });

    it('should read PLY binary data', async () => {
        const source = new BufferReadSource(plyBytes);
        const dataTable = await readPly(source);

        assert.strictEqual(dataTable.numRows, 16);
        assert.strictEqual(dataTable.numColumns, 14);

        const actualSummary = computeSummary(dataTable);
        compareSummaries(actualSummary, expectedSummary, { tolerance: 1e-5 });
    });

    it('should round-trip PLY losslessly', async () => {
        // Write PLY
        const writeFs = new MemoryFileSystem();
        await writePly({
            filename: 'test.ply',
            plyData: {
                comments: [],
                elements: [{ name: 'vertex', dataTable: testData }]
            }
        }, writeFs);

        const writtenPly = writeFs.results.get('test.ply');
        assert(writtenPly, 'PLY file should be written');
        assert(writtenPly.length > 0, 'PLY file should not be empty');

        // Read back
        const source = new BufferReadSource(writtenPly);
        const readBack = await readPly(source);

        // Compare data tables (should be identical)
        compareDataTables(readBack, testData, 0);
    });

    it('should tolerate trailing/duplicate whitespace in the header', async () => {
        // reuse the binary body from a clean encode, but prepend a deliberately "messy" header:
        // a space-padded vertex count and duplicate spaces between tokens — both of which
        // previously caused 'invalid ply header'.
        const clean = encodePlyBinary(testData);
        const term = 'end_header\n';
        const termIdx = new TextDecoder('ascii').decode(clean).indexOf(term);
        assert(termIdx !== -1, 'end_header terminator not found in encoded PLY');
        const body = clean.subarray(termIdx + term.length);

        const plyType = { float32: 'float', float64: 'double', int8: 'char', uint8: 'uchar', int16: 'short', uint16: 'ushort', int32: 'int', uint32: 'uint' };
        const headerLines = [
            'ply',
            'format binary_little_endian 1.0',
            `element vertex ${testData.numRows}            `,        // trailing padding spaces
            ...testData.columns.map(c => `property  ${plyType[c.dataType]}   ${c.name}`), // duplicate spaces
            'end_header'
        ];
        const header = new TextEncoder().encode(`${headerLines.join('\n')}\n`);
        const messy = new Uint8Array(header.length + body.length);
        messy.set(header, 0);
        messy.set(body, header.length);

        const dataTable = await readPly(new BufferReadSource(messy));
        assert.strictEqual(dataTable.numRows, testData.numRows);
        assert.strictEqual(dataTable.numColumns, testData.columns.length);
    });
});

describe('Compressed PLY Format', () => {
    let testData;
    let expectedSummary;

    before(() => {
        testData = createMinimalTestData();
        testData.transform = Transform.PLY.clone();
        expectedSummary = computeSummary(testData);
    });

    it('should round-trip Compressed PLY with acceptable loss', async () => {
        // Write Compressed PLY
        const writeFs = new MemoryFileSystem();
        await writeCompressedPly({
            filename: 'test.compressed.ply',
            dataTable: testData
        }, writeFs);

        const writtenPly = writeFs.results.get('test.compressed.ply');
        assert(writtenPly, 'Compressed PLY file should be written');
        assert(writtenPly.length > 0, 'Compressed PLY file should not be empty');

        // Read back (readPly auto-detects compressed format)
        const source = new BufferReadSource(writtenPly);
        const readBack = await readPly(source);

        assert.strictEqual(readBack.numRows, testData.numRows);

        // Compare with tolerance (lossy compression)
        const actualSummary = computeSummary(readBack);
        compareSummaries(actualSummary, expectedSummary, { tolerance: 0.1 });
    });
});

describe('SOG Format (Bundled)', () => {
    let testData;
    let expectedSummary;

    before(() => {
        testData = createMinimalTestData();
        testData.transform = Transform.PLY.clone();
        expectedSummary = computeSummary(testData);
    });

    it('should round-trip SOG bundled format with acceptable loss', async () => {
        // Write SOG (bundled = zip file)
        const writeFs = new MemoryFileSystem();
        await writeSog({
            filename: 'test.sog',
            dataTable: testData,
            bundle: true,
            iterations: 5 // Fewer iterations for faster test
            // No GPU device - will use CPU fallback
        }, writeFs);

        const writtenSog = writeFs.results.get('test.sog');
        assert(writtenSog, 'SOG file should be written');
        assert(writtenSog.length > 0, 'SOG file should not be empty');

        // Read back using ZipReadFileSystem
        const readFs = new MemoryReadFileSystem();
        readFs.set('test.sog', writtenSog);
        const source = await readFs.createSource('test.sog');
        const zipFs = new ZipReadFileSystem(source);

        try {
            const readBack = await readSog(zipFs, 'meta.json');

            assert.strictEqual(readBack.numRows, testData.numRows);

            // Compare with higher tolerance (lossy compression)
            const actualSummary = computeSummary(readBack);
            compareSummaries(actualSummary, expectedSummary, {
                tolerance: 0.5,
                allowExtraColumns: true
            });
        } finally {
            zipFs.close();
        }
    });
});

describe('SOG Format (Unbundled)', () => {
    let testData;
    let expectedSummary;

    before(() => {
        testData = createMinimalTestData();
        testData.transform = Transform.PLY.clone();
        expectedSummary = computeSummary(testData);
    });

    it('should round-trip SOG unbundled format with acceptable loss', async () => {
        // Write SOG (unbundled = separate files)
        const writeFs = new MemoryFileSystem();
        await writeSog({
            filename: 'meta.json',
            dataTable: testData,
            bundle: false,
            iterations: 5
        }, writeFs);

        // Create a read filesystem with all written files
        // Note: writeSog uses pathe.resolve which creates absolute paths for texture files
        // We need to map these back to relative paths for reading
        const readFs = new MemoryReadFileSystem();
        for (const [name, data] of writeFs.results.entries()) {
            // Extract just the filename for texture files (they're stored with absolute paths)
            const baseName = name.includes('/') ? name.split('/').pop() : name;
            readFs.set(baseName, data);
        }

        // Verify meta.json is present
        assert(readFs.get('meta.json'), 'meta.json should be written');

        // Read back
        const readBack = await readSog(readFs, 'meta.json');

        assert.strictEqual(readBack.numRows, testData.numRows);

        // Compare with higher tolerance (lossy compression)
        const actualSummary = computeSummary(readBack);
        compareSummaries(actualSummary, expectedSummary, {
            tolerance: 0.5,
            allowExtraColumns: true
        });
    });
});

describe('SOG Format (V1 Legacy)', () => {
    // Build a tiny synthetic V1 SOG fixture (2 splats, no SH) by encoding the
    // textures with the V1 quantization scheme: linear lerp between per-axis
    // mins/maxs. This mirrors the layout published by older PlayCanvas tools
    // (e.g. https://d8dooaaq2ugo3.cloudfront.net/4e05290d/v1/meta.json) and
    // verifies that readSog still decodes it correctly.
    const buildV1Fixture = async () => {
        const codec = await WebPCodec.create();
        const count = 2;

        // Pick simple ground-truth values that survive quantization.
        const positions = [
            [0.5, -0.25, 1.0],
            [-0.5, 0.25, -1.0]
        ];
        const scales = [
            [Math.log(0.1), Math.log(0.2), Math.log(0.05)],
            [Math.log(0.05), Math.log(0.1), Math.log(0.2)]
        ];
        const colors = [
            [0.2, 0.4, 0.6],
            [-0.2, -0.4, -0.6]
        ];
        const opacities = [1.0, -1.5]; // logit values stored directly in V1

        // Identity quaternion (w largest, mode 0): a=b=c=0 -> bytes 127 with
        // tag 252 (= mode 0). The decoder's unpackQuat call then produces
        // a quat with w ~= 1 and x/y/z ~= 0 (column convention: rot_0 = w).
        const quatBytes = new Uint8Array(2 * 4);
        for (let i = 0; i < count; i++) {
            quatBytes[i * 4 + 0] = 127;
            quatBytes[i * 4 + 1] = 127;
            quatBytes[i * 4 + 2] = 127;
            quatBytes[i * 4 + 3] = 252;
        }

        // Means: per-axis mins/maxs span the actual value range; bytes encode
        // a logTransform'd position (sign(x) * ln(|x|+1)) packed into 16 bits
        // across two textures.
        const logT = v => Math.sign(v) * Math.log(Math.abs(v) + 1);
        const means = {
            mins: [0, 0, 0].map((_, k) => Math.min(logT(positions[0][k]), logT(positions[1][k]))),
            maxs: [0, 0, 0].map((_, k) => Math.max(logT(positions[0][k]), logT(positions[1][k])))
        };
        const meansLo = new Uint8Array(count * 4);
        const meansHi = new Uint8Array(count * 4);
        for (let i = 0; i < count; i++) {
            for (let k = 0; k < 3; k++) {
                const lv = logT(positions[i][k]);
                const span = means.maxs[k] - means.mins[k] || 1;
                const u16 = Math.round(((lv - means.mins[k]) / span) * 65535);
                meansLo[i * 4 + k] = u16 & 0xff;
                meansHi[i * 4 + k] = (u16 >> 8) & 0xff;
            }
            meansLo[i * 4 + 3] = 255;
            meansHi[i * 4 + 3] = 255;
        }

        // Scales: per-axis mins/maxs, single 8-bit lerp.
        const scaleMeta = {
            mins: [0, 0, 0].map((_, k) => Math.min(scales[0][k], scales[1][k])),
            maxs: [0, 0, 0].map((_, k) => Math.max(scales[0][k], scales[1][k]))
        };
        const scaleBytes = new Uint8Array(count * 4);
        for (let i = 0; i < count; i++) {
            for (let k = 0; k < 3; k++) {
                const span = scaleMeta.maxs[k] - scaleMeta.mins[k] || 1;
                scaleBytes[i * 4 + k] = Math.round(((scales[i][k] - scaleMeta.mins[k]) / span) * 255);
            }
            scaleBytes[i * 4 + 3] = 255;
        }

        // sh0: 4-element mins/maxs (rgb + opacity logit), 8-bit lerp per channel.
        const sh0Meta = {
            mins: [
                Math.min(colors[0][0], colors[1][0]),
                Math.min(colors[0][1], colors[1][1]),
                Math.min(colors[0][2], colors[1][2]),
                Math.min(opacities[0], opacities[1])
            ],
            maxs: [
                Math.max(colors[0][0], colors[1][0]),
                Math.max(colors[0][1], colors[1][1]),
                Math.max(colors[0][2], colors[1][2]),
                Math.max(opacities[0], opacities[1])
            ]
        };
        const sh0Bytes = new Uint8Array(count * 4);
        for (let i = 0; i < count; i++) {
            for (let k = 0; k < 3; k++) {
                const span = sh0Meta.maxs[k] - sh0Meta.mins[k] || 1;
                sh0Bytes[i * 4 + k] = Math.round(((colors[i][k] - sh0Meta.mins[k]) / span) * 255);
            }
            const opSpan = sh0Meta.maxs[3] - sh0Meta.mins[3] || 1;
            sh0Bytes[i * 4 + 3] = Math.round(((opacities[i] - sh0Meta.mins[3]) / opSpan) * 255);
        }

        // Encode each texture as a 1-row WebP. Encoder requires a non-zero
        // height; using width=count, height=1 packs all entries on one row.
        const meansLoWebp = codec.encodeLosslessRGBA(meansLo, count, 1);
        const meansHiWebp = codec.encodeLosslessRGBA(meansHi, count, 1);
        const quatsWebp = codec.encodeLosslessRGBA(quatBytes, count, 1);
        const scalesWebp = codec.encodeLosslessRGBA(scaleBytes, count, 1);
        const sh0Webp = codec.encodeLosslessRGBA(sh0Bytes, count, 1);

        const meta = {
            // No `version` field => V1
            means: {
                shape: [count, 3],
                mins: means.mins,
                maxs: means.maxs,
                files: ['means_l.webp', 'means_u.webp']
            },
            scales: {
                shape: [count, 3],
                mins: scaleMeta.mins,
                maxs: scaleMeta.maxs,
                files: ['scales.webp']
            },
            quats: {
                shape: [count, 4],
                files: ['quats.webp']
            },
            sh0: {
                shape: [count, 1, 4],
                mins: sh0Meta.mins,
                maxs: sh0Meta.maxs,
                files: ['sh0.webp']
            }
        };

        const fs = new MemoryReadFileSystem();
        fs.set('meta.json', new TextEncoder().encode(JSON.stringify(meta)));
        fs.set('means_l.webp', meansLoWebp);
        fs.set('means_u.webp', meansHiWebp);
        fs.set('quats.webp', quatsWebp);
        fs.set('scales.webp', scalesWebp);
        fs.set('sh0.webp', sh0Webp);

        return { fs, count, positions, scales, colors, opacities };
    };

    it('should decode V1 meta.json (no version field, mins/maxs lerp)', async () => {
        const { fs, count, positions, scales, colors, opacities } = await buildV1Fixture();

        const dataTable = await readSog(fs, 'meta.json');

        assert.strictEqual(dataTable.numRows, count);
        assert(dataTable.hasColumn('x'));
        assert(dataTable.hasColumn('opacity'));

        const get = name => dataTable.getColumnByName(name).data;
        for (let i = 0; i < count; i++) {
            // Position uses a 16-bit lerp + invLogTransform; tolerance accounts
            // for that quantization round-trip.
            assert(Math.abs(get('x')[i] - positions[i][0]) < 1e-3, `x[${i}]`);
            assert(Math.abs(get('y')[i] - positions[i][1]) < 1e-3, `y[${i}]`);
            assert(Math.abs(get('z')[i] - positions[i][2]) < 1e-3, `z[${i}]`);

            // Scales / colors / opacity use 8-bit lerp; ~1/255 of the span.
            for (let k = 0; k < 3; k++) {
                const sSpan = Math.abs(scales[0][k] - scales[1][k]) + 1e-6;
                assert(Math.abs(get(`scale_${k}`)[i] - scales[i][k]) < sSpan / 200, `scale_${k}[${i}]`);

                const cSpan = Math.abs(colors[0][k] - colors[1][k]) + 1e-6;
                assert(Math.abs(get(`f_dc_${k}`)[i] - colors[i][k]) < cSpan / 200, `f_dc_${k}[${i}]`);
            }
            const oSpan = Math.abs(opacities[0] - opacities[1]) + 1e-6;
            assert(Math.abs(get('opacity')[i] - opacities[i]) < oSpan / 200, `opacity[${i}]`);

            // Quat encoded with mode 0 (w is the max component) and byte 127
            // for x/y/z; byte 127 maps to ~-0.0039 not exactly 0, so the
            // reconstructed w is ~1 and the smallest components are ~-0.003.
            // Engine convention: rot_0 = w, rot_1..3 = x,y,z.
            assert(Math.abs(get('rot_0')[i] - 1) < 5e-3, `rot_0[${i}] = ${get('rot_0')[i]}`);
            assert(Math.abs(get('rot_1')[i]) < 5e-3, `rot_1[${i}]`);
            assert(Math.abs(get('rot_2')[i]) < 5e-3, `rot_2[${i}]`);
            assert(Math.abs(get('rot_3')[i]) < 5e-3, `rot_3[${i}]`);
        }
    });
});

describe('SPLAT Format (Input Only)', () => {
    it('should read .splat file', async () => {
        const splatData = await fsReadFile(join(fixturesDir, 'minimal.splat'));
        const source = new BufferReadSource(splatData);
        const dataTable = await readSplat(source);

        assert.strictEqual(dataTable.numRows, 4);
        assert.strictEqual(dataTable.numColumns, 14);

        // Verify required columns exist
        const requiredColumns = [
            'x', 'y', 'z',
            'scale_0', 'scale_1', 'scale_2',
            'f_dc_0', 'f_dc_1', 'f_dc_2',
            'opacity',
            'rot_0', 'rot_1', 'rot_2', 'rot_3'
        ];
        for (const col of requiredColumns) {
            assert(dataTable.hasColumn(col), `Missing column: ${col}`);
        }

        // Verify no NaN or Inf values
        const summary = computeSummary(dataTable);
        for (const [name, stats] of Object.entries(summary.columns)) {
            assert.strictEqual(stats.nanCount, 0, `${name} has NaN values`);
            // Allow opacity to have Inf (for fully transparent/opaque)
            if (name !== 'opacity') {
                assert.strictEqual(stats.infCount, 0, `${name} has Inf values`);
            }
        }
    });
});

describe('KSPLAT Format (Input Only)', () => {
    it('should read .ksplat file (compression mode 0)', async () => {
        const ksplatData = await fsReadFile(join(fixturesDir, 'minimal.ksplat'));
        const source = new BufferReadSource(ksplatData);
        const dataTable = await readKsplat(source);

        assert.strictEqual(dataTable.numRows, 4);

        // Verify required columns exist
        const requiredColumns = [
            'x', 'y', 'z',
            'scale_0', 'scale_1', 'scale_2',
            'f_dc_0', 'f_dc_1', 'f_dc_2',
            'opacity',
            'rot_0', 'rot_1', 'rot_2', 'rot_3'
        ];
        for (const col of requiredColumns) {
            assert(dataTable.hasColumn(col), `Missing column: ${col}`);
        }

        // Verify no NaN values
        const summary = computeSummary(dataTable);
        for (const [name, stats] of Object.entries(summary.columns)) {
            assert.strictEqual(stats.nanCount, 0, `${name} has NaN values`);
        }
    });
});

describe('SPZ Format (Input Only)', () => {
    it('should read .spz v2 fixture file', async () => {
        const spzData = await fsReadFile(join(fixturesDir, 'minimal-v2.spz'));
        const source = new BufferReadSource(spzData);
        const dataTable = await readSpz(source);

        assert.strictEqual(dataTable.numRows, 4);

        // Verify required columns exist
        const requiredColumns = [
            'x', 'y', 'z',
            'scale_0', 'scale_1', 'scale_2',
            'f_dc_0', 'f_dc_1', 'f_dc_2',
            'opacity',
            'rot_0', 'rot_1', 'rot_2', 'rot_3'
        ];
        for (const col of requiredColumns) {
            assert(dataTable.hasColumn(col), `Missing column: ${col}`);
        }

        // Verify no NaN values
        const summary = computeSummary(dataTable);
        for (const [name, stats] of Object.entries(summary.columns)) {
            assert.strictEqual(stats.nanCount, 0, `${name} has NaN values`);
        }
    });

    it('should read .spz v4 fixture file', async () => {
        const spzData = await fsReadFile(join(fixturesDir, 'minimal-v4.spz'));
        const source = new BufferReadSource(spzData);
        const dataTable = await readSpz(source);

        assert.strictEqual(dataTable.numRows, 4);

        const summary = computeSummary(dataTable);
        for (const [name, stats] of Object.entries(summary.columns)) {
            assert.strictEqual(stats.nanCount, 0, `${name} has NaN values`);
        }

        const scale0 = dataTable.getColumnByName('scale_0').data;
        const opacity = dataTable.getColumnByName('opacity').data;
        const color0 = dataTable.getColumnByName('f_dc_0').data;
        const color1 = dataTable.getColumnByName('f_dc_1').data;
        const color2 = dataTable.getColumnByName('f_dc_2').data;

        assert(Math.abs(scale0[0] - Math.log(0.1)) < 0.05);
        assert(Math.abs(opacity[0] - logit(0.9)) < 0.08);
        assert(Math.abs(color0[0] - ((1 / 3 - 0.5) / SH_C0)) < 0.04);
        assert(Math.abs(color1[0] - ((1 / 3 - 0.5) / SH_C0)) < 0.04);
        assert(Math.abs(color2[0]) < 0.04);
    });

    it('should read .spz v4 data from a non-zero byteOffset view', async () => {
        const spzData = await fsReadFile(join(fixturesDir, 'minimal-v4.spz'));
        const wrapped = new Uint8Array(spzData.length + 17);
        wrapped.set(spzData, 17);
        const source = new BufferReadSource(wrapped.subarray(17));
        const dataTable = await readSpz(source);

        assert.strictEqual(dataTable.numRows, 4);
    });

    it('should read .spz v3 fixture file', async () => {
        const spzData = await fsReadFile(join(fixturesDir, 'minimal-v3.spz'));
        const source = new BufferReadSource(spzData);
        const dataTable = await readSpz(source);

        assert.strictEqual(dataTable.numRows, 4);

        const summary = computeSummary(dataTable);
        for (const [name, stats] of Object.entries(summary.columns)) {
            assert.strictEqual(stats.nanCount, 0, `${name} has NaN values`);
        }
    });

    it('should read .spz v3 file with correct quaternion decoding', async () => {
        const spzData = await createSpzFixture({ version: 3 });
        const source = new BufferReadSource(spzData);
        const dataTable = await readSpz(source);

        assert.strictEqual(dataTable.numRows, 2);

        const requiredColumns = [
            'x', 'y', 'z',
            'scale_0', 'scale_1', 'scale_2',
            'f_dc_0', 'f_dc_1', 'f_dc_2',
            'opacity',
            'rot_0', 'rot_1', 'rot_2', 'rot_3'
        ];
        for (const col of requiredColumns) {
            assert(dataTable.hasColumn(col), `Missing column: ${col}`);
        }

        const summary = computeSummary(dataTable);
        for (const [name, stats] of Object.entries(summary.columns)) {
            assert.strictEqual(stats.nanCount, 0, `${name} has NaN values`);
        }

        const rot0 = dataTable.getColumnByName('rot_0').data;
        const rot1 = dataTable.getColumnByName('rot_1').data;
        const rot2 = dataTable.getColumnByName('rot_2').data;
        const rot3 = dataTable.getColumnByName('rot_3').data;

        assert(Math.abs(rot0[0] - 1.0) < 1e-6);
        assert(Math.abs(rot1[0]) < 1e-6);
        assert(Math.abs(rot2[0]) < 1e-6);
        assert(Math.abs(rot3[0]) < 1e-6);

        assert(Math.abs(Math.abs(rot0[1]) - Math.cos(Math.PI / 4)) < 0.01);
        assert(Math.abs(rot1[1]) < 0.01);
        assert(Math.abs(rot2[1]) < 0.01);
        assert(Math.abs(Math.abs(rot3[1]) - Math.sin(Math.PI / 4)) < 0.01);
    });

    it('should read .spz v4 file with shDegree=0', async () => {
        const spzData = await createSpzFixture({ version: 4, shDegree: 0 });
        const source = new BufferReadSource(spzData);
        const dataTable = await readSpz(source);

        assert.strictEqual(dataTable.numRows, 2);

        const summary = computeSummary(dataTable);
        for (const [name, stats] of Object.entries(summary.columns)) {
            assert.strictEqual(stats.nanCount, 0, `${name} has NaN values`);
        }

        const rot0 = dataTable.getColumnByName('rot_0').data;
        const rot1 = dataTable.getColumnByName('rot_1').data;
        const rot2 = dataTable.getColumnByName('rot_2').data;
        const rot3 = dataTable.getColumnByName('rot_3').data;

        assert(Math.abs(rot0[0] - 1.0) < 1e-6);
        assert(Math.abs(rot1[0]) < 1e-6);
        assert(Math.abs(rot2[0]) < 1e-6);
        assert(Math.abs(rot3[0]) < 1e-6);

        assert(Math.abs(Math.abs(rot0[1]) - Math.cos(Math.PI / 4)) < 0.01);
        assert(Math.abs(rot1[1]) < 0.01);
        assert(Math.abs(rot2[1]) < 0.01);
        assert(Math.abs(Math.abs(rot3[1]) - Math.sin(Math.PI / 4)) < 0.01);
    });

    it('should read .spz v4 file with shDegree=4', async () => {
        const spzData = await createSpzFixture({ version: 4, shDegree: 4 });
        const source = new BufferReadSource(spzData);
        const dataTable = await readSpz(source);

        assert.strictEqual(dataTable.numRows, 2);
        for (let i = 0; i < 72; i++) {
            assert(dataTable.hasColumn(`f_rest_${i}`));
        }

        for (let i = 0; i < 72; i++) {
            const col = dataTable.getColumnByName(`f_rest_${i}`).data;
            assert(Math.abs(col[0]) < 1e-6);
            assert(Math.abs(col[1]) < 1e-6);
        }
    });

    it('should return an empty table for v4 files with mismatched stream count', async () => {
        const spzData = await createSpzFixture({ version: 4, shDegree: 0 });
        spzData[15] = 4;
        const source = new BufferReadSource(spzData);
        const dataTable = await readSpz(source);
        assert.strictEqual(dataTable.numRows, 0);
    });

    it('should return an empty table for v4 files with TOC past end of file', async () => {
        const spzData = await createSpzFixture({ version: 4, shDegree: 0 });
        const view = new DataView(spzData.buffer, spzData.byteOffset, spzData.byteLength);
        view.setUint32(16, spzData.length + 100, true);
        const source = new BufferReadSource(spzData);
        const dataTable = await readSpz(source);
        assert.strictEqual(dataTable.numRows, 0);
    });

    it('should read .spz v4 file with header extensions', async () => {
        const spzData = await createSpzFixture({
            version: 4,
            shDegree: 0,
            extensionBytes: new Uint8Array([
                0x02, 0x00, 0xBE, 0xAD,
                0x04, 0x00, 0x00, 0x00,
                0xDE, 0xAD, 0xFA, 0xCE
            ])
        });
        const source = new BufferReadSource(spzData);
        const dataTable = await readSpz(source);

        assert.strictEqual(dataTable.numRows, 2);

        const summary = computeSummary(dataTable);
        for (const [name, stats] of Object.entries(summary.columns)) {
            assert.strictEqual(stats.nanCount, 0, `${name} has NaN values`);
        }
    });
});

describe('SPZ Format (Output)', () => {
    let testData;

    before(() => {
        testData = createMinimalTestData();
        testData.transform = Transform.PLY.clone();
    });

    it('should write .spz and read it back', async () => {
        const writeFs = new MemoryFileSystem();
        await writeSpz({
            filename: 'test.spz',
            dataTable: testData
        }, writeFs);

        const writtenSpz = writeFs.results.get('test.spz');
        assert(writtenSpz, 'SPZ file should be written');
        assert(writtenSpz.length > 0, 'SPZ file should not be empty');

        const header = new DataView(writtenSpz.buffer, writtenSpz.byteOffset, 16);
        assert.strictEqual(header.getUint32(0, true), 0x5053474E);
        assert.strictEqual(header.getUint32(4, true), 4);

        const source = new BufferReadSource(writtenSpz);
        const readBack = await readSpz(source);

        assert.strictEqual(readBack.numRows, testData.numRows);
        assert.strictEqual(readBack.transform.equals(Transform.PLY), true);

        const actualSummary = computeSummary(readBack);
        const expectedSummary = computeSummary(testData);
        compareSummaries(actualSummary, expectedSummary, {
            tolerance: 0.25,
            allowExtraColumns: true
        });
    });

    it('should write legacy SPZ3 when requested', async () => {
        const writeFs = new MemoryFileSystem();
        await writeSpz({
            filename: 'test-v3.spz',
            dataTable: testData,
            version: 3
        }, writeFs);

        const writtenSpz = writeFs.results.get('test-v3.spz');
        assert(writtenSpz, 'SPZ file should be written');
        assert.strictEqual(writtenSpz[0], 0x1f);
        assert.strictEqual(writtenSpz[1], 0x8b);

        const readBack = await readSpz(new BufferReadSource(writtenSpz));
        assert.strictEqual(readBack.numRows, testData.numRows);
        assert.strictEqual(readBack.transform.equals(Transform.PLY), true);
    });
});

describe('CSV Format (Output Only)', () => {
    let testData;

    before(() => {
        testData = createMinimalTestData();
    });

    it('should write valid CSV with header and data rows', async () => {
        const writeFs = new MemoryFileSystem();
        await writeCsv({
            filename: 'test.csv',
            dataTable: testData
        }, writeFs);

        const csvData = writeFs.results.get('test.csv');
        assert(csvData, 'CSV file should be written');

        const csvText = new TextDecoder().decode(csvData);
        const lines = csvText.trim().split('\n');

        // Should have header + data rows
        assert.strictEqual(lines.length, testData.numRows + 1);

        // Check header
        const header = lines[0].split(',');
        assert.deepStrictEqual(header, testData.columnNames);

        // Check first data row has correct number of values
        const firstRow = lines[1].split(',');
        assert.strictEqual(firstRow.length, testData.numColumns);

        // Verify values are valid numbers
        for (const value of firstRow) {
            const num = parseFloat(value);
            assert(!Number.isNaN(num), `Invalid number in CSV: ${value}`);
        }
    });
});

describe('MJS Generator Format (Input Only)', () => {
    it('should read generator script and produce DataTable', async () => {
        // Test the generator directly (can't easily test dynamic import in test)
        const { Generator } = await import('./fixtures/generator.mjs');

        const generator = Generator.create([
            { name: 'width', value: '4' },
            { name: 'height', value: '4' }
        ]);

        assert.strictEqual(generator.count, 16);
        assert.strictEqual(generator.columnNames.length, 14);

        // Create DataTable from generator
        const columns = generator.columnNames.map(name => new Column(name, new Float32Array(generator.count)));
        const row = {};
        for (let i = 0; i < generator.count; i++) {
            generator.getRow(i, row);
            for (const col of columns) {
                col.data[i] = row[col.name];
            }
        }
        const dataTable = new DataTable(columns);

        assert.strictEqual(dataTable.numRows, 16);

        // Verify summary is reasonable
        const summary = computeSummary(dataTable);
        assert.strictEqual(summary.rowCount, 16);

        // Positions should span expected range
        assert(summary.columns.x.min < 0, 'x.min should be negative');
        assert(summary.columns.x.max > 0, 'x.max should be positive');
    });
});
