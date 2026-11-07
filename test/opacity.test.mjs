/**
 * Opacity at the file boundary. Inside the pipeline a source's opacity slot holds
 * true (linear) opacity, which may exceed 1; files carry it as one property: the
 * standard `opacity` logit, or `opacity_linear` when the scene may exceed 1.
 */

import assert from 'node:assert';
import { dirname, join } from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';

import { createTestDataTable, encodePlyBinary } from './helpers/test-utils.mjs';
import { createChunkDataPool, createInMemoryChunkSource } from '../src/lib/chunk/index.js';
import { dataTableToChunkSource, materializeToDataTable } from '../src/lib/compat/data-table.js';
import { combine } from '../src/lib/data-table/combine.js';
import {
    Column, DataTable, getInputFormat, logger, MemoryFileSystem, MemoryReadFileSystem, readFile, WebPCodec, writeSource
} from '../src/lib/index.js';
import { filterByValueRows } from '../src/lib/ops/filter-mask.js';
import { decodePlyToDataTable, readPly } from '../src/lib/readers/read-ply.js';
import { Transform } from '../src/lib/utils/index.js';
import { opacityToLogit } from '../src/lib/utils/opacity.js';
import { writePlyStreaming } from '../src/lib/writers/write-ply-streaming.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
WebPCodec.wasmUrl = join(__dirname, '..', 'lib', 'webp.wasm');

const sigmoid = (v) => 1 / (1 + Math.exp(-v));

// A source of `opacities.length` gaussians with the given true opacities.
const sourceWith = (opacities, opacityAboveOne = false) => {
    const n = opacities.length;
    const pos = new Float32Array(n * 3).map((_, i) => i);
    const geo = new Float32Array(n * 8);
    for (let i = 0; i < n; i++) geo.set([1, 0, 0, 0, -3, -3, -3, opacities[i]], i * 8);
    return createInMemoryChunkSource({
        numGaussians: n,
        chunkSize: 1024,
        shBands: 0,
        opacityAboveOne,
        transform: Transform.PLY.clone(),
        lodCounts: [n],
        position: [[pos.buffer]],
        geometric: [[geo.buffer]],
        color: [[new Float32Array(n * 3).buffer]]
    });
};

// The opacity slot of every gaussian of a (single-chunk) source.
const opacitySlot = async (src, pool) => {
    const n = src.meta.numGaussians;
    const g = pool.acquire('geometric', src.meta.layouts.geometric, n);
    await src.read({ chunkIndex: 0, geometric: g });
    const values = Array.from({ length: n }, (_, i) => new Float32Array(g.data)[i * 8 + 7]);
    g.release();
    return values;
};

// Write a source to PLY: the header text and the file read back as a source.
const writeAndRead = async (src, pool, options = {}) => {
    const fs = new MemoryFileSystem();
    await writePlyStreaming(src, pool, { filename: 'out.ply', ...options }, fs);
    const bytes = fs.results.get('out.ply');
    const rfs = new MemoryReadFileSystem();
    rfs.set('out.ply', bytes);
    const header = new TextDecoder().decode(bytes.subarray(0, 1024)).split('end_header')[0];
    // `stored`: the file's own columns, as written
    const stored = await decodePlyToDataTable(await rfs.createSource('out.ply'));
    return { header, stored, back: await readPly(await rfs.createSource('out.ply'), pool) };
};

const readPlyBytes = async (bytes, pool) => {
    const rfs = new MemoryReadFileSystem();
    rfs.set('in.ply', bytes);
    return readPly(await rfs.createSource('in.ply'), pool);
};

const captureWarnings = async (fn) => {
    const warnings = [];
    logger.setRenderer({ handle: (e) => e.kind === 'message' && e.level === 'warn' && warnings.push(e.text) });
    try {
        await fn();
    } finally {
        logger.setRenderer({ handle: () => {} });
    }
    return warnings;
};

describe('PLY opacity', () => {
    it('writes a scene that may exceed 1 as opacity_linear alone, and reads it back exactly', async () => {
        const pool = createChunkDataPool();
        const opacities = [0, 0.25, 1, 1.25, 3];
        const { header, back } = await writeAndRead(sourceWith(opacities, true), pool);
        assert.match(header, /property float opacity_linear\n/);
        assert.doesNotMatch(header, /property float opacity\n/);
        assert.strictEqual(back.meta.opacityAboveOne, true);
        assert.deepStrictEqual(back.meta.extraColumns, [], 'opacity_linear fills the opacity slot, not an extra column');
        assert.deepStrictEqual(await opacitySlot(back, pool), opacities.map(Math.fround));
    });

    it('writes any other scene as the standard opacity logit, round-tripping infinite logits', async () => {
        const pool = createChunkDataPool();
        const logits = [-Infinity, -20, -1, 0, 3, Infinity];
        const opacities = logits.map(sigmoid);
        const { header, back } = await writeAndRead(sourceWith(opacities), pool);
        assert.match(header, /property float opacity\n/);
        assert.doesNotMatch(header, /opacity_linear/);
        assert.strictEqual(back.meta.opacityAboveOne, undefined);
        const read = await opacitySlot(back, pool);
        for (let i = 0; i < logits.length; i++) {
            assert.ok(Math.abs(read[i] - Math.fround(opacities[i])) < 1e-7, `logit ${logits[i]}: ${read[i]}`);
        }
        const stored = (await writeAndRead(back, pool)).stored.getColumnByName('opacity').data;
        assert.strictEqual(stored[0], -Infinity, '-Infinity round-trips');
        assert.strictEqual(stored[5], Infinity, '+Infinity round-trips');
        assert.ok(Math.abs(stored[1] + 20) < 1e-4, `logit -20 round-trips (${stored[1]})`);
    });

    it('--ply-logit-opacity writes the standard logit, clamping opacity above 1 with a warning', async () => {
        const pool = createChunkDataPool();
        let result;
        const warnings = await captureWarnings(async () => {
            result = await writeAndRead(sourceWith([0.25, 1, 1.25, 3], true), pool, { logitOpacity: true });
        });
        assert.match(result.header, /property float opacity\n/);
        assert.doesNotMatch(result.header, /opacity_linear/);
        assert.ok(warnings.some((w) => /^2 gaussians had opacity above 1, clamped/.test(w)), warnings.join('\n'));
        const stored = Array.from(result.stored.getColumnByName('opacity').data);
        assert.ok(Math.abs(sigmoid(stored[0]) - 0.25) < 1e-6);
        assert.strictEqual(stored[1], Infinity, 'opacity 1 is the +Infinity logit');
        assert.strictEqual(stored[2], Math.fround(opacityToLogit(1.25)), 'opacity above 1 clamps to a finite logit');
        assert.ok(Number.isFinite(stored[2]) && stored[2] > 16);
    });

    it('reads opacity_linear from a file that carries both properties', async () => {
        const pool = createChunkDataPool();
        const dt = createTestDataTable(4);
        dt.getColumnByName('opacity').data.fill(20); // the logit says opaque
        dt.addColumn(new Column('opacity_linear', Float32Array.from([0, 0.5, 1.5, 0.25])));
        const src = await readPlyBytes(encodePlyBinary(dt), pool);
        assert.deepStrictEqual(await opacitySlot(src, pool), [0, 0.5, 1.5, 0.25]);
        assert.strictEqual(src.meta.opacityAboveOne, true);
        assert.deepStrictEqual(src.meta.extraColumns, []);
    });
});

describe('opacity in every reader', () => {
    it('fills the opacity slot with true opacity, 0 to 1', async () => {
        const dt = createTestDataTable(64);
        for (const [filename, outputFormat] of [
            ['a.ply', 'ply'], ['a.compressed.ply', 'compressed-ply'], ['a.splat', 'splat'], ['a.spz', 'spz'], ['a.sog', 'sog-bundle']
        ]) {
            const pool = createChunkDataPool();
            const fs = new MemoryFileSystem();
            await writeSource({ filename, outputFormat, source: dataTableToChunkSource(dt, pool.chunkSize), pool, options: { iterations: 2 } }, fs);
            const rfs = new MemoryReadFileSystem();
            rfs.set(filename, fs.results.get(filename));
            const [src] = await readFile({ filename, inputFormat: getInputFormat(filename), fileSystem: rfs, pool });
            const read = await opacitySlot(src, pool);
            for (let i = 0; i < read.length; i++) {
                assert.ok(read[i] >= 0 && read[i] <= 1, `${outputFormat} row ${i}: ${read[i]}`);
            }
            if (outputFormat === 'ply') {
                // PLY is lossless: the opacity the logit encodes
                const expected = Array.from(dt.getColumnByName('opacity').data, sigmoid);
                for (let i = 0; i < read.length; i++) assert.ok(Math.abs(read[i] - expected[i]) < 1e-6, `ply row ${i}`);
            }
            await src.close();
        }
    });
});

describe('DataTable bridge', () => {
    it('materializes true opacity as the opacity logit, plus opacity_linear only when some exceeds 1', async () => {
        const pool = createChunkDataPool();
        const plain = await materializeToDataTable(sourceWith([0.25, 1]), pool);
        assert.ok(!plain.hasColumn('opacity_linear'));
        assert.ok(Math.abs(sigmoid(plain.getColumnByName('opacity').data[0]) - 0.25) < 1e-6);
        const heavy = await materializeToDataTable(sourceWith([0.25, 1.5], true), pool);
        assert.deepStrictEqual(Array.from(heavy.getColumnByName('opacity_linear').data), [0.25, 1.5]);
        assert.ok(Number.isFinite(heavy.getColumnByName('opacity').data[1]), 'the logit of opacity above 1 is finite');
    });

    it('reads a table carrying only opacity_linear, and flags it', async () => {
        const pool = createChunkDataPool();
        const dt = createTestDataTable(3);
        dt.removeColumn('opacity');
        dt.addColumn(new Column('opacity_linear', Float32Array.from([0.1, 1.2, 0.7])));
        const src = dataTableToChunkSource(dt, pool.chunkSize);
        assert.strictEqual(src.meta.opacityAboveOne, true);
        assert.deepStrictEqual(await opacitySlot(src, pool), [0.1, 1.2, 0.7].map(Math.fround));
    });

    it('combine() fills opacity_linear for tables without it from their opacity', () => {
        const a = createTestDataTable(2);
        a.getColumnByName('opacity').data.set([0, 2]);
        const b = createTestDataTable(2);
        b.addColumn(new Column('opacity_linear', Float32Array.from([1.5, 0.25])));
        const linear = Array.from(combine([a, b]).getColumnByName('opacity_linear').data);
        assert.deepStrictEqual(linear, [0.5, sigmoid(2), 1.5, 0.25].map(Math.fround));
    });
});

describe('opacity filters', () => {
    it('compare true opacity (opacity) and the PLY logit (opacity_raw)', async () => {
        const pool = createChunkDataPool();
        const src = sourceWith([0.1, 0.25, 0.75, 1.5], true);
        const rows = async (columnName, comparator, value) =>
            Array.from(await filterByValueRows(src, pool, { columnName, comparator, value }));
        assert.deepStrictEqual(await rows('opacity', 'lt', 0.5), [0, 1]);
        assert.deepStrictEqual(await rows('opacity', 'gt', 1), [3], 'opacity above 1 is comparable');
        assert.deepStrictEqual(await rows('opacity_raw', 'gt', 0), [2, 3], 'opacity_raw compares the logit');
    });
});

describe('opacity_raw', () => {
    it('compares the logit a standard PLY stores: opacity above 1 as its clamped logit', async () => {
        const pool = createChunkDataPool();
        const src = sourceWith([1.5], true);
        const rows = async (comparator, value) =>
            Array.from(await filterByValueRows(src, pool, { columnName: 'opacity_raw', comparator, value }));
        const stored = Math.fround(opacityToLogit(1.5)); // 16.635532...
        assert.deepStrictEqual(await rows('gt', 20), [], 'its stored logit is ~16.64, not above 20');
        assert.deepStrictEqual(await rows('gt', 16.7), [], 'nor above 16.7');
        assert.deepStrictEqual(await rows('gt', 16.6), [0]);
        assert.deepStrictEqual(await rows('eq', stored), [0], 'equal to its exact stored logit');
        assert.deepStrictEqual(await rows('gte', stored), [0]);
        assert.deepStrictEqual(await rows('gt', stored), []);
    });

    it('compares the same way on a DataTable carrying opacity_linear', async () => {
        const { processDataTable } = await import('../src/lib/index.js');
        const table = await materializeToDataTable(sourceWith([1.5], true), createChunkDataPool());
        const keep = async (comparator, value) =>
            (await processDataTable(table, [{ kind: 'filterByValue', columnName: 'opacity_raw', comparator, value }])).numRows;
        const stored = Math.fround(opacityToLogit(1.5));
        assert.strictEqual(await keep('gt', 16.7), 0);
        assert.strictEqual(await keep('gt', 16.6), 1);
        assert.strictEqual(await keep('eq', stored), 1);
    });

    it('compares a read logit equal to itself', async () => {
        const pool = createChunkDataPool();
        const dt = createTestDataTable(2);
        dt.getColumnByName('opacity').data.set([1, -2]);
        const src = await readPlyBytes(encodePlyBinary(dt), pool);
        const rows = async (comparator, value) =>
            Array.from(await filterByValueRows(src, pool, { columnName: 'opacity_raw', comparator, value }));
        assert.deepStrictEqual(await rows('eq', 1), [0]);
        assert.deepStrictEqual(await rows('gt', 1), []);
        assert.deepStrictEqual(await rows('lt', 1), [1]);
    });
});

describe('opacity just below 1', () => {
    it('keeps its own logit: compared by opacity_raw and written back unclamped', async () => {
        const pool = createChunkDataPool();
        const dt = createTestDataTable(1);
        dt.getColumnByName('opacity').data[0] = 16.5; // reads as f32 opacity 0.99999994
        const src = await readPlyBytes(encodePlyBinary(dt), pool);
        const [a] = await opacitySlot(src, pool);
        assert.ok(a < 1, `${a}`);
        assert.deepStrictEqual(Array.from(await filterByValueRows(src, pool, { columnName: 'opacity_raw', comparator: 'gt', value: 16.2 })), [0]);
        const logit = (await writeAndRead(src, pool)).stored.getColumnByName('opacity').data[0];
        assert.ok(logit > 16.2, `written back as ${logit}`);
        assert.strictEqual(Math.fround(sigmoid(logit)), a, 'reads back as the same opacity');
        assert.ok(opacityToLogit(1.5) >= opacityToLogit(a), 'opacity above 1 never logs lower than opacity below it');
    });
});

describe('processDataTable opacity filters', () => {
    it('compare true opacity, with opacity_linear winning, as the source filters do', async () => {
        const { processDataTable } = await import('../src/lib/index.js');
        const pool = createChunkDataPool();
        const table = await materializeToDataTable(sourceWith([0.25, 0.75, 1.5], true), pool);
        const keep = async (columnName, comparator, value) =>
            (await processDataTable(table, [{ kind: 'filterByValue', columnName, comparator, value }])).numRows;
        assert.strictEqual(await keep('opacity', 'gt', 1), 1, 'opacity above 1 is comparable');
        assert.strictEqual(await keep('opacity', 'lt', 0.5), 1);
        assert.strictEqual(await keep('opacity_raw', 'gt', 20), 0, 'opacity above 1 as its clamped logit');
        // a stale logit loses to opacity_linear
        const stale = createTestDataTable(2);
        stale.getColumnByName('opacity').data.fill(20);
        stale.addColumn(new Column('opacity_linear', Float32Array.from([0.25, 0.75])));
        const out = await processDataTable(stale, [{ kind: 'filterByValue', columnName: 'opacity', comparator: 'lt', value: 0.5 }]);
        assert.deepStrictEqual(Array.from(out.getColumnByName('opacity_linear').data), [0.25]);
    });
});

describe('a DataTable with an inconsistent logit', () => {
    it('is not a case the bridge can be wrong about: opacity_linear wins', async () => {
        const pool = createChunkDataPool();
        const dt = new DataTable([
            new Column('x', new Float32Array(1)), new Column('y', new Float32Array(1)), new Column('z', new Float32Array(1)),
            new Column('rot_0', Float32Array.from([1])), new Column('rot_1', new Float32Array(1)),
            new Column('rot_2', new Float32Array(1)), new Column('rot_3', new Float32Array(1)),
            new Column('scale_0', new Float32Array(1)), new Column('scale_1', new Float32Array(1)), new Column('scale_2', new Float32Array(1)),
            new Column('f_dc_0', new Float32Array(1)), new Column('f_dc_1', new Float32Array(1)), new Column('f_dc_2', new Float32Array(1)),
            new Column('opacity', Float32Array.from([20])),
            new Column('opacity_linear', Float32Array.from([0.25]))
        ]);
        assert.deepStrictEqual(await opacitySlot(dataTableToChunkSource(dt, pool.chunkSize), pool), [0.25]);
    });
});

describe('DataTable writers called directly', () => {
    it('write opacity_linear, which wins over a stale opacity logit', async () => {
        const { writeCompressedPly, writeSpz } = await import('../src/lib/index.js');
        for (const [filename, write] of [['a.compressed.ply', writeCompressedPly], ['a.spz', writeSpz]]) {
            const pool = createChunkDataPool();
            const dt = createTestDataTable(8);
            dt.getColumnByName('opacity').data.fill(20); // the logit says opaque
            dt.addColumn(new Column('opacity_linear', new Float32Array(8).fill(0.25)));
            const fs = new MemoryFileSystem();
            await write({ filename, dataTable: dt }, fs);
            const rfs = new MemoryReadFileSystem();
            rfs.set(filename, fs.results.get(filename));
            const [src] = await readFile({ filename, inputFormat: getInputFormat(filename), fileSystem: rfs, pool });
            for (const v of await opacitySlot(src, pool)) assert.ok(Math.abs(v - 0.25) <= 1 / 255, `${filename}: ${v}`);
        }
    });
});
