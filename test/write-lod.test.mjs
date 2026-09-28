/**
 * Tests for the lod-meta.json contract emitted by writeLodSource, and that it is
 * source-type-agnostic: a lazy disk-backed PLY (positions streamed, heavy data
 * gathered per output chunk) and a resident bridged DataTable produce
 * byte-identical output. LOD is structural — a multi-LOD source is built by
 * stacking single-LOD sources (stackLods); there is no per-gaussian lod column.
 */

import assert from 'node:assert';
import { dirname, join } from 'node:path';
import { after, before, describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';

import {
    Column, DataTable, MemoryReadFileSystem, Transform, WebPCodec,
    getInputFormat, logger, readFile
} from '../src/lib/index.js';
import { dataTableToChunkSource, materializeToDataTable } from '../src/lib/compat/data-table.js';
import { MemoryFileSystem } from '../src/lib/io/write/index.js';
import { bakeTransform, mapSource, stackLods } from '../src/lib/ops/index.js';
import { readPly } from '../src/lib/readers/read-ply.js';
import { collectFilesByLod, readLodEnvironmentSource } from '../src/lib/readers/read-lod.js';
import { createChunkDataPool } from '../src/lib/chunk/index.js';
import { chooseChunkExtent, countLeaves, positionsFromSlim, writeLodSource } from '../src/lib/writers/write-lod.js';
import { BTree } from '../src/lib/spatial/index.js';
import { version } from '../src/lib/version.js';

import { encodePlyBinary } from './helpers/test-utils.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
WebPCodec.wasmUrl = join(__dirname, '..', 'lib', 'webp.wasm');

// The per-leaf error tables are opt-in and rendered on the GPU; tests that read
// them request them and skip when no adapter is available, and the header tests
// assert the declaration matches whether a device was supplied.
let device = null;

before(async () => {
    try {
        const { createDevice } = await import('../src/cli/node-device.js');
        device = await createDevice();
    } catch {
        device = null;
    }
});

after(() => {
    device?.destroy?.();
});

const deviceOptions = () => (device ? { createDevice: async () => device, lodErrors: true } : { lodErrors: true });

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

// A table from explicit per-splat values, for exercising the LOD error metric.
// Defaults put every splat at the origin with log-scale -3 (sigma ~0.0498) and
// opacity logit 0 (alpha 0.5).
const makeSplatTable = (splats) => {
    const col = (key, fallback) => new Float32Array(splats.map(s => s[key] ?? fallback));
    return new DataTable([
        new Column('x', col('x', 0)),
        new Column('y', col('y', 0)),
        new Column('z', col('z', 0)),
        new Column('rot_0', col('rot_0', 1)),
        new Column('rot_1', col('rot_1', 0)),
        new Column('rot_2', col('rot_2', 0)),
        new Column('rot_3', col('rot_3', 0)),
        new Column('scale_0', col('scale', -3)),
        new Column('scale_1', col('scale', -3)),
        new Column('scale_2', col('scale', -3)),
        new Column('f_dc_0', col('f_dc', 0)),
        new Column('f_dc_1', col('f_dc', 0)),
        new Column('f_dc_2', col('f_dc', 0)),
        new Column('opacity', col('opacity', 0))
    ], Transform.PLY);
};

// The error table of a scene whose levels are given splat-by-splat.
const writeErrors = async (levels) => {
    const fs = new MemoryFileSystem();
    const sources = levels.map(splats => dataTableToChunkSource(makeSplatTable(splats), 1 << 20));
    await writeLodSource({
        filename: '/scene/lod-meta.json',
        mainSource: sources.length === 1 ? sources[0] : stackLods(sources),
        envSource: null,
        iterations: 1,
        chunkCount: 1,
        chunkExtent: 16,
        ...deviceOptions()
    }, fs);
    const meta = JSON.parse(new TextDecoder().decode(fs.results.get('/scene/lod-meta.json')));
    return meta.tree.errors;
};

// A one-splat table with `count` SH coefficients (9, 24 or 45 for bands 1 to 3),
// all `restValue` except those in `overrides`.
const makeShTable = (restValue, count = 9, overrides = {}) => {
    const table = makeTable(1);
    for (let i = 0; i < count; i++) {
        table.addColumn(new Column(`f_rest_${i}`, new Float32Array([overrides[i] ?? restValue])));
    }
    return table;
};

// The error table of a two-level scene whose levels are the given one-splat tables.
const writeTableErrors = async (tables) => {
    const fs = new MemoryFileSystem();
    await writeLodSource({
        filename: '/scene/lod-meta.json',
        mainSource: stackLods(tables.map(table => dataTableToChunkSource(table, 1 << 20))),
        envSource: null,
        iterations: 1,
        chunkCount: 1,
        chunkExtent: 16,
        createDevice: async () => device,
        lodErrors: true
    }, fs);
    return JSON.parse(new TextDecoder().decode(fs.results.get('/scene/lod-meta.json'))).tree.errors;
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
        chunkExtent: 16,
        ...deviceOptions()
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
        // the partition parameters the caller asked for: chunkCount is in units of
        // 1024 gaussians, chunkExtent in world units
        assert.strictEqual(meta.asset.chunkGaussians, 1024);
        assert.strictEqual(meta.asset.chunkExtent, 16);
        assert.strictEqual(meta.asset.maxChunks, 5000, 'the default leaf target is recorded');
        assert.strictEqual(meta.count, 5);
        assert.deepStrictEqual(meta.counts, [3, 2]);
        assert.strictEqual(meta.lodLevels, 2);
        assert.strictEqual(meta.lodErrors, device !== null, 'error tables are declared exactly when a GPU rendered them');
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
        if (device) {
            assert.strictEqual(meta.tree.errors.length, 2);
            assert.strictEqual(meta.tree.errors[0], 0);
            assert.ok(Number.isFinite(meta.tree.errors[1]) && meta.tree.errors[1] >= 0);
        } else {
            assert.ok(!('errors' in meta.tree), 'no per-leaf error table without a GPU');
        }

        assert.ok(fs.results.has('/scene/0_0/meta.json'));
        assert.ok(fs.results.has('/scene/1_0/meta.json'));
    });

    const wideSplats = Array.from({ length: 300 }, (_, i) => ({ x: i * 0.5 }));

    it('records the chunk minimum and does not split a sparse node for extent below it', async function () {
        // 300 splats over 150 m: wider than the 16 m extent limit (and enough for the
        // spatial tree to have interior nodes), but below the default minimum of 1K
        // gaussians, so the region stays one leaf
        const source = dataTableToChunkSource(makeSplatTable(wideSplats), 1 << 20);
        const fs = new MemoryFileSystem();
        await writeLodSource({
            filename: '/scene/lod-meta.json',
            mainSource: source,
            envSource: null,
            iterations: 1,
            chunkCount: 1,
            chunkExtent: 16
        }, fs);
        const meta = JSON.parse(new TextDecoder().decode(fs.results.get('/scene/lod-meta.json')));
        assert.strictEqual(meta.asset.chunkMinGaussians, 1024);
        assert.ok(!('children' in meta.tree), 'a sparse wide node is one leaf');
        assert.strictEqual(meta.tree.lods['0'].count, wideSplats.length);
    });

    it('splits a wide node for extent once it holds more than the chunk minimum', async function () {
        const source = dataTableToChunkSource(makeSplatTable(wideSplats), 1 << 20);
        const fs = new MemoryFileSystem();
        await writeLodSource({
            filename: '/scene/lod-meta.json',
            mainSource: source,
            envSource: null,
            iterations: 1,
            chunkCount: 1,
            chunkExtent: 16,
            chunkMin: 0
        }, fs);
        const meta = JSON.parse(new TextDecoder().decode(fs.results.get('/scene/lod-meta.json')));
        assert.strictEqual(meta.asset.chunkMinGaussians, 0);
        assert.strictEqual(meta.tree.children?.length, 2, 'the extent limit splits a node above the minimum');
        const leaves = [];
        const walk = (n) => { if (n.children) n.children.forEach(walk); else leaves.push(n); };
        walk(meta.tree);
        assert.ok(leaves.length >= 2, `expected the extent limit to split the node, got ${leaves.length} leaf`);
        assert.strictEqual(leaves.reduce((sum, l) => sum + l.lods['0'].count, 0), wideSplats.length);
    });

    // 4096 splats along 2 km: at a 16 m extent with no minimum the tree splits down to its
    // 256-gaussian leaves
    const lineSplats = Array.from({ length: 4096 }, (_, i) => ({ x: i * 0.5 }));

    it('raises the chunk extent until the tree fits the chunk target', async function () {
        const source = dataTableToChunkSource(makeSplatTable(lineSplats), 1 << 20);
        const fs = new MemoryFileSystem();
        await writeLodSource({
            filename: '/scene/lod-meta.json',
            mainSource: source,
            envSource: null,
            iterations: 1,
            chunkCount: 1024,
            chunkExtent: 16,
            chunkMin: 0,
            maxChunks: 4
        }, fs);
        const meta = JSON.parse(new TextDecoder().decode(fs.results.get('/scene/lod-meta.json')));
        const leaves = [];
        const walk = (n) => { if (n.children) n.children.forEach(walk); else leaves.push(n); };
        walk(meta.tree);
        assert.ok(leaves.length <= 4, `expected at most 4 leaves, got ${leaves.length}`);
        assert.ok(leaves.length > 1, 'the fitted extent still splits the line');
        assert.ok(meta.asset.chunkExtent > 16, `the extent limit is raised, got ${meta.asset.chunkExtent}`);
        assert.strictEqual(meta.asset.maxChunks, 4);
        assert.strictEqual(leaves.reduce((sum, l) => sum + l.lods['0'].count, 0), lineSplats.length);
    });

    it('keeps the requested chunk extent when the chunk target is off or already met', async function () {
        for (const maxChunks of [0, 100000]) {
            const source = dataTableToChunkSource(makeSplatTable(lineSplats), 1 << 20);
            const fs = new MemoryFileSystem();
            await writeLodSource({
                filename: '/scene/lod-meta.json',
                mainSource: source,
                envSource: null,
                iterations: 1,
                chunkCount: 1024,
                chunkExtent: 16,
                chunkMin: 0,
                maxChunks
            }, fs);
            const meta = JSON.parse(new TextDecoder().decode(fs.results.get('/scene/lod-meta.json')));
            assert.strictEqual(meta.asset.chunkExtent, 16);
            assert.strictEqual(meta.asset.maxChunks, maxChunks);
        }
    });

    describe('chunk target reporting', function () {
        // writes the splats and returns the leaf count, the recorded asset fields and the LOD
        // chunk messages logged
        const writeLogged = async (splats, options) => {
            const messages = [];
            logger.setRenderer({ handle: (e) => { if (e.text?.includes('LOD chunk')) messages.push(e.text); } });
            try {
                const fs = new MemoryFileSystem();
                await writeLodSource({
                    filename: '/scene/lod-meta.json',
                    mainSource: dataTableToChunkSource(makeSplatTable(splats), 1 << 20),
                    envSource: null,
                    iterations: 1,
                    chunkMin: 0,
                    ...options
                }, fs);
                const meta = JSON.parse(new TextDecoder().decode(fs.results.get('/scene/lod-meta.json')));
                const leaves = [];
                const walk = (n) => { if (n.children) n.children.forEach(walk); else leaves.push(n); };
                walk(meta.tree);
                return { leaves: leaves.length, asset: meta.asset, messages };
            } finally {
                logger.setRenderer({ handle: () => {} });
            }
        };

        // 4096 gaussians inside 1 m, and 4096 spread along 2 km
        const compact = Array.from({ length: 4096 }, (_, i) => ({ x: (i % 16) / 16, y: ((i >> 4) % 16) / 16, z: (i >> 8) / 16 }));
        const spread = Array.from({ length: 4096 }, (_, i) => ({ x: i * 0.5 }));

        it('reports a missed target when the scene already fits within the extent', async function () {
            const { leaves, messages } = await writeLogged(compact, { chunkCount: 1, chunkExtent: 16, maxChunks: 2 });
            assert.strictEqual(leaves, 4);
            assert.deepStrictEqual(messages, ['LOD chunks: 4 chunks, above the --lod-max-chunks target of 2 (--lod-chunk-count splits require more)']);
        });

        it('reports a missed target for gaussians at a single point', async function () {
            const point = Array.from({ length: 4096 }, () => ({ x: 3 }));
            const { leaves, messages } = await writeLogged(point, { chunkCount: 1, chunkExtent: 0, maxChunks: 2 });
            assert.strictEqual(leaves, 4);
            assert.strictEqual(messages.length, 1);
            assert.ok(messages[0].includes('4 chunks, above the --lod-max-chunks target of 2'), messages[0]);
        });

        it('records the smallest extent reaching the fewest chunks when the target is missed', async function () {
            const { leaves, asset, messages } = await writeLogged(spread, { chunkCount: 1, chunkExtent: 16, maxChunks: 2 });
            assert.strictEqual(leaves, 4);
            assert.ok(asset.chunkExtent > 16 && asset.chunkExtent < 1000, `expected an extent just above a file unit's width, got ${asset.chunkExtent}`);
            assert.strictEqual(messages.length, 1);
            assert.ok(messages[0].startsWith('LOD chunk extent raised to') && messages[0].includes('4 chunks, above the --lod-max-chunks target of 2'), messages[0]);
        });

        it('reports a met target after raising the extent, with singular wording for one chunk', async function () {
            const { leaves, messages } = await writeLogged(spread, { chunkCount: 1024, chunkExtent: 16, maxChunks: 1 });
            assert.strictEqual(leaves, 1);
            assert.strictEqual(messages.length, 1);
            assert.ok(messages[0].endsWith('to fit the --lod-max-chunks target of 1: 1 chunk'), messages[0]);
        });

        it('says nothing when the requested extent already fits the target', async function () {
            const { messages } = await writeLogged(spread, { chunkCount: 1024, chunkExtent: 16, maxChunks: 100000 });
            assert.deepStrictEqual(messages, []);
        });

        it('says nothing when the target is off, including for a negative extent', async function () {
            for (const chunkExtent of [16, 0, -5]) {
                const { asset, messages } = await writeLogged(spread, { chunkCount: 1, chunkExtent, maxChunks: 0 });
                assert.deepStrictEqual(messages, [], `extent ${chunkExtent}`);
                assert.strictEqual(asset.maxChunks, 0);
            }
        });
    });

    describe('chooseChunkExtent', function () {
        const lineTree = (count, spacing) => {
            const x = new Float32Array(count).map((_, i) => i * spacing);
            return new BTree(new DataTable([
                new Column('x', x), new Column('y', new Float32Array(count)), new Column('z', new Float32Array(count))
            ])).root;
        };

        it('picks the smallest extent at which the leaf count fits', function () {
            const root = lineTree(1 << 16, 0.25);
            for (const target of [8, 32, 100]) {
                const extent = chooseChunkExtent(root, 1 << 30, 0, 16, target);
                assert.ok(countLeaves(root, 1 << 30, extent, 0) <= target, `target ${target} is met`);
                assert.ok(countLeaves(root, 1 << 30, extent / 1.02, 0) > target, `target ${target}: a smaller extent would not fit`);
            }
        });

        it('terminates and meets the target from a requested extent of 0', function () {
            // an extent of 0 splits down to the spatial tree's leaves; the fit must still converge
            const root = lineTree(4096, 0.5);
            const extent = chooseChunkExtent(root, 1 << 30, 0, 0, 4);
            assert.ok(extent > 0);
            assert.ok(countLeaves(root, 1 << 30, extent, 0) <= 4);
            assert.ok(countLeaves(root, 1 << 30, extent / 1.02, 0) > 4, 'a smaller extent would not fit');
        });

        it('treats a negative requested extent as 0', function () {
            const root = lineTree(4096, 0.5);
            assert.strictEqual(chooseChunkExtent(root, 1 << 30, 0, -5, 4), chooseChunkExtent(root, 1 << 30, 0, 0, 4));
            assert.strictEqual(chooseChunkExtent(root, 1 << 30, 0, -5, 0), 0);
        });

        it('never goes below the requested extent', function () {
            const root = lineTree(1024, 0.01);
            assert.strictEqual(chooseChunkExtent(root, 1 << 30, 0, 16, 1), 16);
        });

        it('aims for the fewest reachable leaves when file units alone exceed the target', function () {
            // 512-gaussian file units over 4096 gaussians force 8 leaves: the fit returns the
            // smallest extent reaching those 8, not the full width, which splits the same way
            const root = lineTree(4096, 1);
            const extent = chooseChunkExtent(root, 512, 0, 16, 2);
            assert.strictEqual(countLeaves(root, 512, extent, 0), 8);
            assert.ok(extent < root.aabb.largestDim(), `expected less than the full width, got ${extent}`);
            assert.ok(countLeaves(root, 512, extent / 1.02, 0) > 8, 'a smaller extent would not reach the fewest leaves');
        });

        it('keeps the requested extent when file units alone exceed the target and it already reaches the fewest leaves', function () {
            const root = lineTree(4096, 0.001);
            assert.strictEqual(chooseChunkExtent(root, 512, 0, 16, 2), 16);
        });
    });

    it('matches errors to lodLevels when trailing structural LODs are empty', async function () {
        const { meta } = await writeScene([1, 0], 0);
        assert.strictEqual(meta.lodLevels, 1);
        if (device) assert.deepStrictEqual(meta.tree.errors, [0]);
        assert.doesNotThrow(() => collectFilesByLod(meta, '/scene/lod-meta.json'));
    });

    it('omits error tables when no GPU device is supplied', async function () {
        const fs = new MemoryFileSystem();
        await writeLodSource({
            filename: '/scene/lod-meta.json',
            mainSource: makeSource([3, 2]),
            envSource: null,
            iterations: 1,
            chunkCount: 1,
            chunkExtent: 16,
            lodErrors: true
        }, fs);
        const meta = JSON.parse(new TextDecoder().decode(fs.results.get('/scene/lod-meta.json')));
        assert.strictEqual(meta.lodErrors, false);
        assert.ok(!('errors' in meta.tree), 'no per-leaf error table without a GPU');
    });

    it('omits error tables by default', async function () {
        const fs = new MemoryFileSystem();
        await writeLodSource({
            filename: '/scene/lod-meta.json',
            mainSource: makeSource([3, 2]),
            envSource: null,
            iterations: 1,
            chunkCount: 1,
            chunkExtent: 16,
            ...(device ? { createDevice: async () => device } : {})
        }, fs);
        const meta = JSON.parse(new TextDecoder().decode(fs.results.get('/scene/lod-meta.json')));
        assert.strictEqual(meta.lodErrors, false);
        assert.ok(!('errors' in meta.tree), 'no per-leaf error table unless requested');
    });

    it('includes stored spherical harmonics in the LOD error', async function (t) {
        if (!device) return t.skip('no WebGPU adapter available');
        const errors = await writeTableErrors([makeShTable(0), makeShTable(2)]);
        assert.ok(errors[1] > 0);
    });

    it('sees colour held in SH coefficients that vanish on the coordinate axes', async function (t) {
        if (!device) return t.skip('no WebGPU adapter available');
        // f_rest_3 is the red channel's xy term, zero in any view along an axis
        const errors = await writeTableErrors([makeShTable(0, 45), makeShTable(0, 45, { 3: 2 })]);
        assert.ok(errors[1] > 0, `expected a non-zero error, got ${errors[1]}`);
    });

    it('reports no error for a level identical to the finest', async function (t) {
        if (!device) return t.skip('no WebGPU adapter available');
        assert.deepStrictEqual(await writeErrors([[{}], [{}]]), [0, 0]);
    });

    it('grows with displacement', async function (t) {
        if (!device) return t.skip('no WebGPU adapter available');
        const sigma = Math.exp(-3);
        const small = (await writeErrors([[{}], [{ x: 0.5 * sigma }]]))[1];
        const large = (await writeErrors([[{}], [{ x: 4 * sigma }]]))[1];
        assert.ok(small > 0, `expected a non-zero error for a half-sigma shift, got ${small}`);
        assert.ok(large > small, `expected a larger error for a larger shift, got ${small} then ${large}`);
    });

    it('penalises an opacity drop at identical geometry', async function (t) {
        if (!device) return t.skip('no WebGPU adapter available');
        const errors = await writeErrors([[{ opacity: 2 }], [{ opacity: -2 }]]);
        assert.ok(errors[1] > 0, `expected a non-zero error, got ${errors[1]}`);
    });

    it('scores a level that draws where the reference renders nothing', async function (t) {
        if (!device) return t.skip('no WebGPU adapter available');
        // opacity logit -10 is under the rasterizer's alpha floor, so the reference is transparent
        const errors = await writeErrors([[{ opacity: -10 }], [{ opacity: 5 }]]);
        assert.ok(errors[1] > 0 && Number.isFinite(errors[1]), `expected a positive finite error, got ${errors[1]}`);
    });

    it('penalises thinning even when the survivors are identical', async function (t) {
        if (!device) return t.skip('no WebGPU adapter available');
        // two coincident splats decimated to one: the survivor is exact, but the
        // level paints less coverage
        const errors = await writeErrors([[{}, {}], [{}]]);
        assert.ok(errors[1] > 0, `expected a non-zero error, got ${errors[1]}`);
    });

    // Non-finite input is rejected up front rather than tolerated: a NaN anywhere
    // in a gaussian would paint NaN into the error renders, and the comparison
    // would quietly absorb it. The check runs on the bounds pass, so it holds with
    // or without a GPU.
    const rejects = [
        ['a NaN scale', { scale: NaN }, /non-finite scale/],
        ['a NaN opacity', { opacity: NaN }, /non-finite opacity/],
        ['a NaN position', { x: NaN }, /non-finite position/],
        ['a NaN rotation', { rot_0: NaN }, /non-finite rotation/],
        ['a zero-norm rotation', { rot_0: 0 }, /zero-norm rotation/],
        ['a NaN color', { f_dc: NaN }, /non-finite color or SH/]
    ];

    for (const [label, splat, expected] of rejects) {
        it(`refuses to write LODs for input with ${label}`, async function () {
            await assert.rejects(() => writeErrors([[{}, splat], [{}]]), (err) => {
                assert.match(err.message, expected);
                assert.match(err.message, /--filter-nan/);
                return true;
            });
        });
    }

    it('accepts the non-finite values --filter-nan deliberately keeps', async function () {
        // a flat splat (scale -Inf) and a fully opaque one (opacity +Inf) survive
        // filterNaN, so the writer must not reject them
        const errors = await writeErrors([[{}, { scale: -Infinity }, { opacity: Infinity }], [{}]]);
        if (device) {
            assert.ok(
                errors.every(error => Number.isFinite(error) && error >= 0),
                `expected finite non-negative errors, got ${errors}`
            );
        }
    });

    it('measures small leaves through the atlas', async function (t) {
        if (!device) return t.skip('no WebGPU adapter available');
        // a grid of small splats: every splat is far below half the leaf's bounding
        // radius, so the leaf is batched into an atlas rather than rendered alone.
        // Three levels, so the atlas renderer's second slot is reused within a view.
        const grid = Array.from({ length: 64 }, (_, i) => ({ x: (i % 8) * 0.25, y: Math.floor(i / 8) * 0.25 }));
        const errors = await writeErrors([grid, grid, grid.filter((_, i) => i % 2 === 0)]);
        assert.strictEqual(errors[1], 0, `expected no error for an identical level, got ${errors[1]}`);
        assert.ok(errors[2] > 0 && Number.isFinite(errors[2]), `expected a positive finite error for thinning, got ${errors[2]}`);
    });

    it('keeps the error table monotone across levels', async function (t) {
        if (!device) return t.skip('no WebGPU adapter available');
        // level 2 matches a level-0 splat exactly while level 1 sits between
        // both, so the raw errors would rank the coarser level as the better one
        const errors = await writeErrors([[{ x: 0 }, { x: 1 }], [{ x: 0.5 }], [{ x: 0 }]]);
        assert.ok(errors[1] > 0, `expected a non-zero error, got ${errors[1]}`);
        assert.ok(errors[2] >= errors[1], `expected monotone errors, got ${errors}`);
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

    it('rejects malformed tree children with a controlled error', function () {
        for (const child of [null, 1]) {
            const meta = { lodLevels: 1, counts: [0], filenames: [], tree: { children: [child] } };
            assert.throws(
                () => collectFilesByLod(meta, '/scene/lod-meta.json'),
                { message: 'Invalid lod-meta.json tree: /scene/lod-meta.json' }
            );
        }
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

    it('reads a pre-v1 manifest (no version/count/counts), deriving counts from the tree', async function () {
        const { fs, meta } = await writeScene([3, 2], 0);

        // Simulate an older published manifest: drop the fields #261 introduced.
        delete meta.version;
        delete meta.count;
        delete meta.counts;
        fs.results.set('/scene/lod-meta.json', new TextEncoder().encode(JSON.stringify(meta)));

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

    it('still rejects an unsupported (non-1) version', async function () {
        const { fs, meta } = await writeScene([3], 0);
        meta.version = 2;
        fs.results.set('/scene/lod-meta.json', new TextEncoder().encode(JSON.stringify(meta)));

        await assert.rejects(
            readFile({
                filename: '/scene/lod-meta.json',
                inputFormat: 'lod',
                options: { lodSelect: [] },
                fileSystem: asReadFileSystem(fs)
            }),
            { message: 'Unsupported lod-meta.json version: 2' }
        );
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
