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

import {
    Column, DataTable, MemoryReadFileSystem, Transform, WebPCodec,
    getInputFormat, readFile
} from '../src/lib/index.js';
import { dataTableToChunkSource, materializeToDataTable } from '../src/lib/compat/data-table.js';
import { MemoryFileSystem } from '../src/lib/io/write/index.js';
import { bakeTransform, mapSource, stackLods } from '../src/lib/ops/index.js';
import { readPly } from '../src/lib/readers/read-ply.js';
import { readLodEnvironmentSource } from '../src/lib/readers/read-lod.js';
import { createChunkDataPool } from '../src/lib/chunk/index.js';
import { positionsFromSlim, writeLodSource } from '../src/lib/writers/write-lod.js';
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

const asReadFileSystem = (writeFs) => {
    const readFs = new MemoryReadFileSystem();
    for (const [name, data] of writeFs.results) readFs.set(name, data);
    return readFs;
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

    it('bakes a pending transform before tree construction (deferred == pre-baked, byte-for-byte)', async function () {
        // A carries the transform *deferred* (mapSource only composes
        // meta.transform; the data stays raw); B is the same scene with the
        // transform already baked into the data. The writer must bake before the
        // partition/bounds passes, so both must produce identical output —
        // including lod-meta.json's tree bounds (the payload space).
        const T = new Transform().fromEulers(90, 0, 180);
        const pendingLevel = (n) => mapSource(dataTableToChunkSource(makeTable(n), 1 << 20), T);
        const bakedLevel = async (n) => {
            const dt = await materializeToDataTable(bakeTransform(pendingLevel(n), Transform.PLY), createChunkDataPool());
            return dataTableToChunkSource(dt, 1 << 20);
        };

        const fsA = new MemoryFileSystem();
        await writeLodSource({
            filename: '/a/lod-meta.json',
            mainSource: stackLods([pendingLevel(5), pendingLevel(3)]),
            envSource: null, iterations: 1, chunkCount: 1, chunkExtent: 16
        }, fsA);

        const fsB = new MemoryFileSystem();
        await writeLodSource({
            filename: '/b/lod-meta.json',
            mainSource: stackLods([await bakedLevel(5), await bakedLevel(3)]),
            envSource: null, iterations: 1, chunkCount: 1, chunkExtent: 16
        }, fsB);

        assertSameFiles(fsA, fsB);

        // Bounds are in payload (baked / PLY) space: the single-leaf root bound
        // contains every baked position (across both levels) and hugs their union
        // range to within the splat extents (isotropic exp(-3) half-extents,
        // ≤ ~0.09 after rotation).
        const meta = JSON.parse(new TextDecoder().decode(fsA.results.get('/a/lod-meta.json')));
        const axes = ['x', 'y', 'z'];
        const lo = [Infinity, Infinity, Infinity];
        const hi = [-Infinity, -Infinity, -Infinity];
        for (const n of [5, 3]) {
            const dt = await materializeToDataTable(bakeTransform(pendingLevel(n), Transform.PLY), createChunkDataPool());
            axes.forEach((name, axis) => {
                for (const v of dt.getColumnByName(name).data) {
                    lo[axis] = Math.min(lo[axis], v);
                    hi[axis] = Math.max(hi[axis], v);
                }
            });
        }
        axes.forEach((name, axis) => {
            assert.ok(meta.tree.bound.min[axis] <= lo[axis] + 1e-5, `root bound min[${name}] contains baked positions`);
            assert.ok(meta.tree.bound.max[axis] >= hi[axis] - 1e-5, `root bound max[${name}] contains baked positions`);
            assert.ok(meta.tree.bound.min[axis] >= lo[axis] - 0.2, `root bound min[${name}] within splat extents of baked positions`);
            assert.ok(meta.tree.bound.max[axis] <= hi[axis] + 0.2, `root bound max[${name}] within splat extents of baked positions`);
        });
    });
});

describe('readLodSource: streamed SOG input', function () {
    it('detects lod-meta.json before a regular SOG meta.json', function () {
        assert.strictEqual(getInputFormat('/scene/lod-meta.json'), 'lod');
        assert.strictEqual(getInputFormat('/scene/meta.json'), 'sog');
    });

    it('reads all levels as a structural multi-LOD ChunkSource', async function () {
        const { fs } = await writeScene([3, 2], 0);
        const [source] = await readFile({
            filename: '/scene/lod-meta.json',
            inputFormat: 'lod',
            options: { lodSelect: [] },
            fileSystem: asReadFileSystem(fs)
        });

        assert.strictEqual(source.meta.numLods, 2);
        assert.strictEqual(source.meta.numGaussians, 3);
        assert.deepStrictEqual(source.meta.lodCounts, [3, 2]);
        const table = await materializeToDataTable(source, createChunkDataPool());
        assert.strictEqual(table.numRows, 5);
        await source.close();
    });

    it('honors LOD selection', async function () {
        const { fs } = await writeScene([3, 2], 0);
        const [source] = await readFile({
            filename: '/scene/lod-meta.json',
            inputFormat: 'lod',
            options: { lodSelect: [1] },
            fileSystem: asReadFileSystem(fs)
        });

        assert.strictEqual(source.meta.numLods, 1);
        assert.strictEqual(source.meta.numGaussians, 2);
        assert.deepStrictEqual(source.meta.lodCounts, [2]);
        await source.close();
    });

    it('opens the optional environment SOG', async function () {
        const { fs } = await writeScene([3], 2);
        const source = await readLodEnvironmentSource(
            asReadFileSystem(fs),
            '/scene/lod-meta.json',
            createChunkDataPool()
        );

        assert.ok(source);
        assert.strictEqual(source.meta.numGaussians, 2);
        await source.close();
    });
});

describe('positionsFromSlim', () => {
    const CHUNK = 4;
    const makeParent = (n, calls) => ({
        meta: { chunkSize: CHUNK, numGaussians: n, numLods: 1, lodCounts: [n], numChunks: [Math.ceil(n / CHUNK)] },
        read: async (request) => {
            calls.push(request);
            if (request.geometric) {
                // marker fill so forwarding is observable
                new Float32Array(request.geometric.data).fill(99);
            }
        },
        close: async () => {}
    });
    const slim = {
        x: Float32Array.from({ length: 32 }, (_, i) => i + 0.25),
        y: Float32Array.from({ length: 32 }, (_, i) => i + 0.5),
        z: Float32Array.from({ length: 32 }, (_, i) => i + 0.75)
    };
    // unit of 6 rows mapping to scattered flat indices
    const flat = Uint32Array.from([20, 3, 17, 8, 30, 11]);

    it('serves chunk position reads from slim without touching the parent', async () => {
        const calls = [];
        const src = positionsFromSlim(makeParent(6, calls), slim, flat);
        const pos = { data: new ArrayBuffer(CHUNK * 12) };
        await src.read({ chunkIndex: 1, position: pos }); // rows 4..5
        const f = new Float32Array(pos.data);
        assert.strictEqual(f[0], slim.x[30]);
        assert.strictEqual(f[1], slim.y[30]);
        assert.strictEqual(f[2], slim.z[30]);
        assert.strictEqual(f[3], slim.x[11]);
        assert.strictEqual(calls.length, 0);
    });

    it('serves gather position reads from slim by output row', async () => {
        const calls = [];
        const src = positionsFromSlim(makeParent(6, calls), slim, flat);
        const pos = { data: new ArrayBuffer(3 * 12) };
        await src.read({ indices: Uint32Array.from([5, 0, 3]), indexOffset: 0, count: 3, position: pos });
        const f = new Float32Array(pos.data);
        assert.strictEqual(f[0], slim.x[11]); // unit row 5 -> flat 11
        assert.strictEqual(f[3], slim.x[20]); // unit row 0 -> flat 20
        assert.strictEqual(f[6], slim.x[8]);  // unit row 3 -> flat 8
        assert.strictEqual(calls.length, 0);
    });

    it('forwards non-position layers with position stripped', async () => {
        const calls = [];
        const src = positionsFromSlim(makeParent(6, calls), slim, flat);
        const pos = { data: new ArrayBuffer(CHUNK * 12) };
        const geo = { data: new ArrayBuffer(CHUNK * 32) };
        await src.read({ chunkIndex: 0, position: pos, geometric: geo });
        assert.strictEqual(calls.length, 1);
        assert.strictEqual(calls[0].position, undefined);
        assert.strictEqual(calls[0].geometric, geo);
        assert.strictEqual(calls[0].chunkIndex, 0);
        assert.strictEqual(new Float32Array(geo.data)[0], 99); // parent filled it
        assert.strictEqual(new Float32Array(pos.data)[0], slim.x[20]); // slim filled it
    });

    it('forwards position-free requests untouched', async () => {
        const calls = [];
        const src = positionsFromSlim(makeParent(6, calls), slim, flat);
        const geo = { data: new ArrayBuffer(CHUNK * 32) };
        const request = { chunkIndex: 0, geometric: geo };
        await src.read(request);
        assert.strictEqual(calls.length, 1);
        assert.strictEqual(calls[0], request);
    });
});
