/**
 * Tests for readFileInfo (header-only structural metadata), columnNamesFromMeta,
 * and the `info` process action.
 *
 *  - columnNamesFromMeta reproduces materializeToDataTable's canonical column
 *    order from `meta` alone (per SH-band count, extras, and partial layer sets).
 *  - readFileInfo reports counts/columns/shBands without decoding gaussian data;
 *    a truncated file is rejected by the reader's size guard (throws).
 *  - the `info` action passes the source through unchanged (meta-only).
 */

import assert from 'node:assert';
import { readFile as fsReadFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';

import { createTestDataTable, encodePlyBinary } from './helpers/test-utils.mjs';
import { Column, DataTable, MemoryReadFileSystem, readFile, readFileInfo } from '../src/lib/index.js';
import { columnNamesFromMeta, dataTableToChunkSource } from '../src/lib/compat/data-table.js';
import { processSource } from '../src/lib/process-source.js';
import { createChunkDataPool } from '../src/lib/chunk/index.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const fixturesDir = join(__dirname, 'fixtures', 'splat');

// Canonical non-SH columns in the order columnNamesFromMeta emits them.
const STANDARD = [
    'x', 'y', 'z',
    'rot_0', 'rot_1', 'rot_2', 'rot_3', 'scale_0', 'scale_1', 'scale_2', 'opacity',
    'f_dc_0', 'f_dc_1', 'f_dc_2'
];

const memFs = (name, bytes) => {
    const fs = new MemoryReadFileSystem();
    fs.set(name, bytes);
    return fs;
};

describe('columnNamesFromMeta', () => {
    it('lists the canonical columns for a 0-band source', () => {
        const src = dataTableToChunkSource(createTestDataTable(10));
        assert.deepStrictEqual(columnNamesFromMeta(src.meta), STANDARD);
    });

    it('appends f_rest_0..N for each SH band count', () => {
        for (const [bands, rest] of [[1, 9], [2, 24], [3, 45]]) {
            const src = dataTableToChunkSource(createTestDataTable(10, { includeSH: true, shBands: bands }));
            assert.strictEqual(src.meta.shBands, bands);
            const cols = columnNamesFromMeta(src.meta);
            assert.deepStrictEqual(cols.slice(0, STANDARD.length), STANDARD);
            const restCols = cols.slice(STANDARD.length);
            assert.strictEqual(restCols.length, rest);
            assert.strictEqual(restCols[0], 'f_rest_0');
            assert.strictEqual(restCols[rest - 1], `f_rest_${rest - 1}`);
        }
    });

    it('appends extra (other-layer) columns last', () => {
        const base = createTestDataTable(6);
        const dt = new DataTable([...base.columns, new Column('my_extra', new Float32Array(6))]);
        const src = dataTableToChunkSource(dt);
        assert.deepStrictEqual(columnNamesFromMeta(src.meta), [...STANDARD, 'my_extra']);
    });

    it('reports only the available layers (position-only)', () => {
        const n = 6;
        const dt = new DataTable([
            new Column('x', new Float32Array(n)),
            new Column('y', new Float32Array(n)),
            new Column('z', new Float32Array(n))
        ]);
        const src = dataTableToChunkSource(dt);
        assert.deepStrictEqual([...src.meta.availableLayers], ['position']);
        assert.deepStrictEqual(columnNamesFromMeta(src.meta), ['x', 'y', 'z']);
    });
});

describe('readFileInfo', () => {
    const options = { lodSelect: [] };

    it('reports PLY structural metadata', async () => {
        const dt = createTestDataTable(50, { includeSH: true, shBands: 1 });
        const bytes = encodePlyBinary(dt);
        const info = await readFileInfo({
            filename: 'scene.ply', inputFormat: 'ply', options, params: [], fileSystem: memFs('scene.ply', bytes)
        });
        assert.strictEqual(info.format, 'ply');
        assert.strictEqual(info.numGaussians, 50);
        assert.strictEqual(info.numLods, 1);
        assert.deepStrictEqual(info.lodCounts, [50]);
        assert.strictEqual(info.shBands, 1);
        assert.deepStrictEqual(info.columns.slice(0, STANDARD.length), STANDARD);
        assert.strictEqual(info.columns.length, STANDARD.length + 9); // + 9 f_rest for 1 band
    });

    it('rejects a truncated PLY via the reader size guard', async () => {
        const bytes = encodePlyBinary(createTestDataTable(50));
        const truncated = bytes.subarray(0, bytes.length - 100);
        await assert.rejects(
            () => readFileInfo({
                filename: 'scene.ply', inputFormat: 'ply', options, params: [], fileSystem: memFs('scene.ply', truncated)
            }),
            /does not match header-implied size/
        );
    });

    it('reports .splat metadata and agrees with a full read', async () => {
        const bytes = await fsReadFile(join(fixturesDir, 'minimal.splat'));
        const fileSystem = memFs('minimal.splat', bytes);
        const info = await readFileInfo({ filename: 'minimal.splat', inputFormat: 'splat', options, params: [], fileSystem });
        assert.strictEqual(info.format, 'splat');
        assert.strictEqual(info.numGaussians, 4);
        assert.strictEqual(info.shBands, 0);
        assert.deepStrictEqual(info.columns, STANDARD);

        const [full] = await readFile({ filename: 'minimal.splat', inputFormat: 'splat', options, params: [], fileSystem });
        assert.strictEqual(info.numGaussians, full.meta.numGaussians);
        assert.deepStrictEqual(info.columns, columnNamesFromMeta(full.meta));
        await full.close();
    });

    it('reports .spz metadata', async () => {
        const bytes = await fsReadFile(join(fixturesDir, 'minimal-v4.spz'));
        const fileSystem = memFs('minimal.spz', bytes);
        const info = await readFileInfo({ filename: 'minimal.spz', inputFormat: 'spz', options, params: [], fileSystem });
        assert.strictEqual(info.format, 'spz');
        assert.ok(info.numGaussians > 0);
        assert.ok(info.columns.includes('x') && info.columns.includes('opacity'));
    });
});

describe('info process action', () => {
    it('passes the source through unchanged (meta-only)', async () => {
        const pool = createChunkDataPool();
        const src = dataTableToChunkSource(createTestDataTable(20, { includeSH: true, shBands: 1 }));
        const out = await processSource(src, [{ kind: 'info' }], pool);
        assert.strictEqual(out, src); // no-op pass-through
        assert.strictEqual(out.meta.numGaussians, 20);
        assert.strictEqual(out.meta.shBands, 1);
    });
});
