/**
 * Tests for the lod-meta.json file contract emitted by writeLod.
 */

import assert from 'node:assert';
import { dirname, join } from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';

import { Column, DataTable, Transform, WebPCodec } from '../src/lib/index.js';
import { dataTableToChunkSource, materializeToDataTable } from '../src/lib/compat/data-table.js';
import { MemoryFileSystem } from '../src/lib/io/write/index.js';
import { concatSource } from '../src/lib/ops/index.js';
import { readPly } from '../src/lib/readers/read-ply.js';
import { createChunkDataPool } from '../src/lib/source/index.js';
import { writeLod, writeLodSource } from '../src/lib/writers/write-lod.js';
import { version } from '../src/lib/version.js';

import { encodePlyBinary } from './helpers/test-utils.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
WebPCodec.wasmUrl = join(__dirname, '..', 'lib', 'webp.wasm');

// Minimal seekable ReadSource over a buffer, for the disk-PLY writeLodSource path.
class BufferReadSource {
    constructor(data) {
        this.data = data;
        this.size = data.length;
        this.seekable = true;
    }

    read(start = 0, end = this.size) {
        let offset = Math.max(0, Math.min(start, this.size));
        const limit = Math.max(offset, Math.min(end, this.size));
        const data = this.data;
        return {
            async pull(target) {
                const remaining = limit - offset;
                if (remaining <= 0) return 0;
                const n = Math.min(target.length, remaining);
                target.set(data.subarray(offset, offset + n));
                offset += n;
                return n;
            }
        };
    }

    close() {}
}

// Build a minimal splat table (no SH, so no GPU device is needed) with one
// row per entry of `lods`. Constructed directly in PLY space so writeLod's
// convertToSpace is a no-op.
const makeTable = (lods) => {
    const n = lods.length;
    const fill = (value) => new Float32Array(n).fill(value);
    const ramp = (scale) => new Float32Array(Array.from({ length: n }, (_, i) => i * scale));

    return new DataTable([
        new Column('x', ramp(1)),
        new Column('y', ramp(0.5)),
        new Column('z', ramp(0.25)),
        new Column('rot_0', fill(1)),
        new Column('rot_1', fill(0)),
        new Column('rot_2', fill(0)),
        new Column('rot_3', fill(0)),
        new Column('scale_0', fill(-3)),
        new Column('scale_1', fill(-3)),
        new Column('scale_2', fill(-3)),
        new Column('f_dc_0', fill(0)),
        new Column('f_dc_1', fill(0)),
        new Column('f_dc_2', fill(0)),
        new Column('opacity', fill(0)),
        new Column('lod', new Float32Array(lods))
    ], Transform.PLY);
};

const writeScene = async (lods, envRows) => {
    const fs = new MemoryFileSystem();
    await writeLod({
        filename: '/scene/lod-meta.json',
        dataTable: makeTable(lods),
        envDataTable: envRows > 0 ? makeTable(new Array(envRows).fill(0)) : null,
        iterations: 1,
        chunkCount: 1,
        chunkExtent: 16
    }, fs);
    const meta = JSON.parse(new TextDecoder().decode(fs.results.get('/scene/lod-meta.json')));
    return { fs, meta };
};

describe('writeLod', function () {
    it('writes lod-meta.json header fields and chunk references', async function () {
        const { fs, meta } = await writeScene([0, 0, 0, 1, 1], 0);

        assert.strictEqual(meta.version, 1);
        assert.strictEqual(meta.asset.generator, `splat-transform v${version}`);
        assert.strictEqual(meta.count, 5);
        assert.deepStrictEqual(meta.counts, [3, 2]);
        assert.strictEqual(meta.lodLevels, 2);
        assert.ok(!('environment' in meta), 'environment should be omitted when there are no environment splats');
        assert.deepStrictEqual([...meta.filenames].sort(), ['0_0/meta.json', '1_0/meta.json']);

        // single small chunk: the tree is one leaf referencing both lod levels
        assert.strictEqual(meta.filenames[meta.tree.lods['0'].file], '0_0/meta.json');
        assert.strictEqual(meta.filenames[meta.tree.lods['1'].file], '1_0/meta.json');
        assert.deepStrictEqual(
            { offset: meta.tree.lods['0'].offset, count: meta.tree.lods['0'].count },
            { offset: 0, count: 3 }
        );
        assert.deepStrictEqual(
            { offset: meta.tree.lods['1'].offset, count: meta.tree.lods['1'].count },
            { offset: 0, count: 2 }
        );

        // referenced chunk SOGs are written
        assert.ok(fs.results.has('/scene/0_0/meta.json'));
        assert.ok(fs.results.has('/scene/1_0/meta.json'));
    });

    it('references the environment SOG when environment splats are present', async function () {
        const { fs, meta } = await writeScene([0, 0, 0], 2);

        assert.strictEqual(meta.environment, 'env/meta.json');
        assert.ok(fs.results.has('/scene/env/meta.json'));
    });

    // The disk path: writeLodSource fed a fixed-stride PLY source (positions
    // streamed resident, heavy data gathered per output chunk via readRows) must
    // produce byte-identical output to the DataTable wrapper. No SH → encoding is
    // deterministic, so a byte-for-byte A/B is the right gate.
    it('writeLodSource over a disk-backed PLY matches the DataTable path byte-for-byte', async function () {
        const lods = [0, 0, 0, 1, 1, 1, 1];

        // A: DataTable wrapper.
        const fsA = new MemoryFileSystem();
        await writeLod({
            filename: '/a/lod-meta.json',
            dataTable: makeTable(lods),
            envDataTable: null,
            iterations: 1,
            chunkCount: 1,
            chunkExtent: 16
        }, fsA);

        // B: disk-backed PLY source + the lod array supplied separately.
        const plyBytes = encodePlyBinary(makeTable(lods));
        const diskSrc = await readPly(new BufferReadSource(plyBytes), createChunkDataPool());
        const fsB = new MemoryFileSystem();
        await writeLodSource({
            filename: '/b/lod-meta.json',
            mainSource: diskSrc,
            envSource: null,
            lod: new Float32Array(lods),
            iterations: 1,
            chunkCount: 1,
            chunkExtent: 16
        }, fsB);
        await diskSrc.close();

        // Same set of output files (modulo the /a vs /b root).
        const relA = [...fsA.results.keys()].map(k => k.replace('/a/', '')).sort();
        const relB = [...fsB.results.keys()].map(k => k.replace('/b/', '')).sort();
        assert.deepStrictEqual(relB, relA, 'disk path should write the same files');

        // Byte-identical contents for every file (lod-meta.json + each unit SOG).
        for (const rel of relA) {
            assert.deepStrictEqual(
                fsB.results.get(`/b/${rel}`),
                fsA.results.get(`/a/${rel}`),
                `bytes differ for ${rel}`
            );
        }
    });

    // Multi-input --lod: the CLI stitches per-LOD PLYs with concatSource and
    // supplies the tagged-lod array. That path (writeLodSource over a concat of
    // disk PLYs) must match the combined-DataTable wrapper byte-for-byte.
    it('writeLodSource over a multi-input concatSource matches the combined DataTable path', async function () {
        const lods = [0, 0, 0, 1, 1, 1, 1]; // rows 0-2 → LOD 0, rows 3-6 → LOD 1

        // A: one combined DataTable through the wrapper.
        const fsA = new MemoryFileSystem();
        await writeLod({
            filename: '/a/lod-meta.json',
            dataTable: makeTable(lods),
            envDataTable: null,
            iterations: 1,
            chunkCount: 1,
            chunkExtent: 16
        }, fsA);

        // B: split the same scene into two PLYs by LOD, concat, supply the lod array.
        const full = makeTable(lods);
        const subset = async (indices) => materializeToDataTable(
            dataTableToChunkSource(full, 1 << 20, Uint32Array.from(indices)),
            createChunkDataPool()
        );
        const dt0 = await subset([0, 1, 2]);
        const dt1 = await subset([3, 4, 5, 6]);
        const src0 = await readPly(new BufferReadSource(encodePlyBinary(dt0)), createChunkDataPool());
        const src1 = await readPly(new BufferReadSource(encodePlyBinary(dt1)), createChunkDataPool());
        const mainSource = concatSource([src0, src1], createChunkDataPool());

        const fsB = new MemoryFileSystem();
        await writeLodSource({
            filename: '/b/lod-meta.json',
            mainSource,
            envSource: null,
            lod: new Float32Array(lods),
            iterations: 1,
            chunkCount: 1,
            chunkExtent: 16
        }, fsB);
        await mainSource.close();

        const relA = [...fsA.results.keys()].map(k => k.replace('/a/', '')).sort();
        const relB = [...fsB.results.keys()].map(k => k.replace('/b/', '')).sort();
        assert.deepStrictEqual(relB, relA, 'multi-input concat path should write the same files');
        for (const rel of relA) {
            assert.deepStrictEqual(
                fsB.results.get(`/b/${rel}`),
                fsA.results.get(`/a/${rel}`),
                `bytes differ for ${rel}`
            );
        }
    });
});
