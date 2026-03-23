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
    computeSummary,
    getInputFormat,
    readFile,
    readPly,
    readSplat,
    readKsplat,
    readSpz,
    readSog,
    readVoxel,
    writePly,
    writeCompressedPly,
    writeSog,
    writeCsv,
    MemoryReadFileSystem,
    MemoryFileSystem,
    ZipReadFileSystem,
    WebPCodec
} from '../src/lib/index.js';

import { compareSummaries, compareDataTables } from './helpers/summary-compare.mjs';
import { createMinimalTestData, encodePlyBinary, createVoxelFixture } from './helpers/test-utils.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const fixturesDir = join(__dirname, 'fixtures', 'splat');
WebPCodec.wasmUrl = join(__dirname, '..', 'lib', 'webp.wasm');

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

// Hardcoded packed uint32 values for known quaternions, verified against the
// reference C++ smallest-three encoder in nianticlabs/spz.
// Using hardcoded values avoids re-implementing the packing logic in tests,
// which could mask spec/bit-layout bugs if both encoder and decoder share
// the same mistake.
const SPZ_V3_PACKED_QUATERNIONS = {
    // Identity [x=0, y=0, z=0, w=1]: largestIndex=3 (w), all others zero
    identity: 0xC0000000,
    // 45-deg Z rotation [x=0, y=0, z=sin(pi/4), w=cos(pi/4)]: largestIndex=2 (z), w encodes as mag=511
    rotate45Z: 0x800001FF
};

const createSpzV3Fixture = () => {
    const count = 2;
    const HEADER_SIZE = 16;
    const positionsByteSize = count * 3 * 3;
    const alphasByteSize = count;
    const colorsByteSize = count * 3;
    const scalesByteSize = count * 3;
    const rotationsByteSize = count * 4;
    const totalSize = HEADER_SIZE + positionsByteSize + alphasByteSize + colorsByteSize + scalesByteSize + rotationsByteSize;

    const buffer = new ArrayBuffer(totalSize);
    const view = new DataView(buffer);
    const bytes = new Uint8Array(buffer);

    view.setUint32(0, 0x5053474E, true);
    view.setUint32(4, 3, true);
    view.setUint32(8, count, true);
    view.setUint8(12, 0);
    view.setUint8(13, 8);
    view.setUint8(14, 0);
    view.setUint8(15, 0);

    let offset = HEADER_SIZE;
    const positions = [
        [0, 0, 0],
        [1, 0, 0]
    ];
    for (let i = 0; i < count; i++) {
        const values = positions[i].map(value => (Math.round(value * 256) & 0xFFFFFF) >>> 0);
        const posOffset = offset + i * 9;
        for (let j = 0; j < 3; j++) {
            bytes[posOffset + j * 3 + 0] = values[j] & 0xFF;
            bytes[posOffset + j * 3 + 1] = (values[j] >> 8) & 0xFF;
            bytes[posOffset + j * 3 + 2] = (values[j] >> 16) & 0xFF;
        }
    }
    offset += positionsByteSize;

    bytes[offset + 0] = 230;
    bytes[offset + 1] = 230;
    offset += alphasByteSize;

    bytes[offset + 0] = 180;
    bytes[offset + 1] = 90;
    bytes[offset + 2] = 128;
    bytes[offset + 3] = 90;
    bytes[offset + 4] = 180;
    bytes[offset + 5] = 128;
    offset += colorsByteSize;

    const encodedScale = Math.round((Math.log(0.1) + 10) * 16);
    for (let i = 0; i < count; i++) {
        bytes[offset + i * 3 + 0] = encodedScale;
        bytes[offset + i * 3 + 1] = encodedScale;
        bytes[offset + i * 3 + 2] = encodedScale;
    }
    offset += scalesByteSize;

    view.setUint32(offset, SPZ_V3_PACKED_QUATERNIONS.identity, true);
    view.setUint32(offset + 4, SPZ_V3_PACKED_QUATERNIONS.rotate45Z, true);

    return new Uint8Array(buffer);
};

describe('PLY Format', () => {
    let testData;
    let plyBytes;
    let expectedSummary;

    before(() => {
        testData = createMinimalTestData();
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
});

describe('Compressed PLY Format', () => {
    let testData;
    let expectedSummary;

    before(() => {
        testData = createMinimalTestData();
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
    it('should read .spz file', async () => {
        const spzData = await fsReadFile(join(fixturesDir, 'minimal.spz'));
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

    it('should read .spz v3 file with correct quaternion decoding', async () => {
        const spzData = createSpzV3Fixture();
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

        assert(Math.abs(rot0[1] - Math.cos(Math.PI / 4)) < 0.01);
        assert(Math.abs(rot1[1]) < 0.01);
        assert(Math.abs(rot2[1]) < 0.01);
        assert(Math.abs(rot3[1] - Math.sin(Math.PI / 4)) < 0.01);
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

describe('Voxel Format (Input Only)', () => {
    let voxelFs;
    let fixtureMetadata;

    before(async () => {
        // Generate valid voxel fixture data using the library's own octree
        // building functions (BlockAccumulator + buildSparseOctree).
        // The fixture contains 4 solid + 2 mixed leaf blocks.
        const { jsonBytes, binBytes, metadata } = await createVoxelFixture();
        fixtureMetadata = metadata;

        voxelFs = new MemoryReadFileSystem();
        voxelFs.set('test.voxel.json', jsonBytes);
        voxelFs.set('test.voxel.bin', binBytes);
    });

    it('should detect voxel format from filename', () => {
        assert.strictEqual(getInputFormat('scene.voxel.json'), 'voxel');
    });

    it('should read voxel file and produce DataTable with leaf blocks', async () => {
        const dataTable = await readVoxel(voxelFs, 'test.voxel.json');

        assert(dataTable.numRows > 0, 'Should have at least one leaf block');
        assert.strictEqual(dataTable.numColumns, 14);

        const requiredColumns = [
            'x', 'y', 'z',
            'scale_0', 'scale_1', 'scale_2',
            'rot_0', 'rot_1', 'rot_2', 'rot_3',
            'f_dc_0', 'f_dc_1', 'f_dc_2',
            'opacity'
        ];
        for (const col of requiredColumns) {
            assert(dataTable.hasColumn(col), `Should have column ${col}`);
        }

        // All leaf blocks should have uniform scale and opacity
        const scale0 = dataTable.getColumnByName('scale_0').data;
        const opacities = dataTable.getColumnByName('opacity').data;
        const blockSize = 4 * fixtureMetadata.voxelResolution;
        const expectedScale = Math.log(blockSize * 0.4);

        for (let i = 0; i < dataTable.numRows; i++) {
            assert.ok(
                Math.abs(scale0[i] - expectedScale) < 1e-5,
                `scale_0[${i}] should be log(blockSize*0.4)`
            );
            assert.strictEqual(opacities[i], 5.0, `opacity[${i}] should be 5.0`);
        }
    });

    it('should read voxel via readFile and write to PLY', async () => {
        const tables = await readFile({
            filename: 'test.voxel.json',
            inputFormat: getInputFormat('test.voxel.json'),
            options: {},
            params: [],
            fileSystem: voxelFs
        });

        assert.strictEqual(tables.length, 1);
        const dataTable = tables[0];
        assert(dataTable.numRows > 0);

        const writeFs = new MemoryFileSystem();
        await writePly({
            filename: 'voxel-out.ply',
            plyData: {
                comments: ['Voxel leaf visualization'],
                elements: [{ name: 'vertex', dataTable }]
            }
        }, writeFs);

        const plyData = writeFs.results.get('voxel-out.ply');
        assert(plyData, 'PLY should be written');
        assert(plyData.length > 0, 'PLY should not be empty');
    });
});
