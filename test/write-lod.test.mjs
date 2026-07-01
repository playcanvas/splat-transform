/**
 * Tests for the lod-meta.json contract emitted by writeLodSource, and that it is
 * source-type-agnostic: a lazy disk-backed PLY (positions streamed, heavy data
 * gathered per output chunk) and a resident bridged DataTable produce
 * byte-identical output. LOD is structural — a multi-LOD source is built by
 * stacking single-LOD sources (stackLods); there is no per-gaussian lod column.
 */

import assert from 'node:assert';
import { dirname, join } from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';

import { Column, DataTable, Transform, WebPCodec } from '../src/lib/index.js';
import { dataTableToChunkSource } from '../src/lib/compat/data-table.js';
import { MemoryFileSystem } from '../src/lib/io/write/index.js';
import { stackLods } from '../src/lib/ops/index.js';
import { readPly } from '../src/lib/readers/read-ply.js';
import { createChunkDataPool } from '../src/lib/chunk/index.js';
import { writeLodSource } from '../src/lib/writers/write-lod.js';
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

// A minimal n-row splat table (no SH, so no GPU device is needed), in PLY space
// (so writeSogSource's convert-to-PLY is a no-op and the encode is deterministic).
const makeTable = (n) => {
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
        new Column('opacity', fill(0))
    ], Transform.PLY);
};

// Build a structural multi-LOD source from per-level row counts (each level a
// single-LOD resident source; stacked when there is more than one level).
const makeSource = (levelCounts) => {
    const perLevel = levelCounts.map(c => dataTableToChunkSource(makeTable(c), 1 << 20));
    return perLevel.length === 1 ? perLevel[0] : stackLods(perLevel);
};

// Compare two MemoryFileSystems (modulo the /a vs /b root): same file set + bytes.
const assertSameFiles = (fsA, fsB) => {
    const relA = [...fsA.results.keys()].map(k => k.replace('/a/', '')).sort();
    const relB = [...fsB.results.keys()].map(k => k.replace('/b/', '')).sort();
    assert.deepStrictEqual(relB, relA, 'same output files');
    for (const rel of relA) {
        assert.deepStrictEqual(fsB.results.get(`/b/${rel}`), fsA.results.get(`/a/${rel}`), `bytes differ for ${rel}`);
    }
};

const writeScene = async (levelCounts, envRows) => {
    const fs = new MemoryFileSystem();
    await writeLodSource({
        filename: '/scene/lod-meta.json',
        mainSource: makeSource(levelCounts),
        envSource: envRows > 0 ? dataTableToChunkSource(makeTable(envRows), 1 << 20) : null,
        iterations: 1,
        chunkCount: 1,
        chunkExtent: 16
    }, fs);
    const meta = JSON.parse(new TextDecoder().decode(fs.results.get('/scene/lod-meta.json')));
    return { fs, meta };
};

describe('writeLodSource: lod-meta.json contract', function () {
    it('writes header fields and chunk references', async function () {
        const { fs, meta } = await writeScene([3, 2], 0);

        assert.strictEqual(meta.version, 1);
        assert.strictEqual(meta.asset.generator, `splat-transform v${version}`);
        assert.strictEqual(meta.count, 5);
        assert.deepStrictEqual(meta.counts, [3, 2]);
        assert.strictEqual(meta.lodLevels, 2);
        assert.ok(!('environment' in meta), 'environment omitted when there are no environment splats');
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

        assert.ok(fs.results.has('/scene/0_0/meta.json'));
        assert.ok(fs.results.has('/scene/1_0/meta.json'));
    });

    it('references the environment SOG when environment splats are present', async function () {
        const { fs, meta } = await writeScene([3], 2);
        assert.strictEqual(meta.environment, 'env/meta.json');
        assert.ok(fs.results.has('/scene/env/meta.json'));
    });

    it('disk PLY source == resident source, single LOD, byte-for-byte', async function () {
        const table = makeTable(5);

        const fsA = new MemoryFileSystem();
        await writeLodSource({
            filename: '/a/lod-meta.json',
            mainSource: dataTableToChunkSource(table, 1 << 20),
            envSource: null, iterations: 1, chunkCount: 1, chunkExtent: 16
        }, fsA);

        const diskSrc = await readPly(new BufferReadSource(encodePlyBinary(table)), createChunkDataPool());
        const fsB = new MemoryFileSystem();
        await writeLodSource({
            filename: '/b/lod-meta.json',
            mainSource: diskSrc,
            envSource: null, iterations: 1, chunkCount: 1, chunkExtent: 16
        }, fsB);
        await diskSrc.close();

        assertSameFiles(fsA, fsB);
    });

    it('disk PLY source == resident source, multi-LOD (stackLods), byte-for-byte', async function () {
        const t0 = makeTable(3), t1 = makeTable(4);

        const fsA = new MemoryFileSystem();
        await writeLodSource({
            filename: '/a/lod-meta.json',
            mainSource: stackLods([dataTableToChunkSource(t0, 1 << 20), dataTableToChunkSource(t1, 1 << 20)]),
            envSource: null, iterations: 1, chunkCount: 1, chunkExtent: 16
        }, fsA);

        const s0 = await readPly(new BufferReadSource(encodePlyBinary(t0)), createChunkDataPool());
        const s1 = await readPly(new BufferReadSource(encodePlyBinary(t1)), createChunkDataPool());
        const fsB = new MemoryFileSystem();
        await writeLodSource({
            filename: '/b/lod-meta.json',
            mainSource: stackLods([s0, s1]),
            envSource: null, iterations: 1, chunkCount: 1, chunkExtent: 16
        }, fsB);
        await s0.close();
        await s1.close();

        assertSameFiles(fsA, fsB);
    });
});
