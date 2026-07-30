/**
 * Splat model (default / antialiased / 2dgs) detection and propagation.
 *
 *  - comment parsing: Brush's `SplatRenderMode:` and Postshot's `antialiased N`;
 *  - PLY / compressed-PLY / SOG / SPZ outputs carry the tag (PLY always in
 *    Brush's spelling, whichever form it was read from);
 *  - a 2DGS PLY (no scale_2) reads with a full geometric layer and writes back
 *    without the column, and is only inferred from an otherwise-complete record;
 *  - mixed inputs collapse to `default` rather than mistagging;
 *  - untagged scenes are unchanged (no comment, no meta key).
 */

import assert from 'node:assert';
import { dirname, join } from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';

import { createTestDataTable, encodePlyBinary } from './helpers/test-utils.mjs';
import { createChunkDataPool } from '../src/lib/chunk/index.js';
import { materializeToDataTable } from '../src/lib/compat/data-table.js';
import {
    DataTable, WebPCodec,
    MemoryFileSystem, MemoryReadFileSystem,
    resolveSplatModel, writeFile, writeSource
} from '../src/lib/index.js';
import { concatSource } from '../src/lib/ops/index.js';
import { readPly, splatModelFromComments } from '../src/lib/readers/read-ply.js';
import { splatModelComment } from '../src/lib/writers/utils.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
WebPCodec.wasmUrl = join(__dirname, '..', 'lib', 'webp.wasm');

const sourceFromBytes = (bytes) => {
    const rfs = new MemoryReadFileSystem();
    rfs.set('in.ply', bytes);
    return rfs.createSource('in.ply');
};

// Open a PLY built from `dataTable` (+ header comments) as a ChunkSource.
const openPly = async (dataTable, comments = []) => {
    const pool = createChunkDataPool();
    const source = await readPly(await sourceFromBytes(encodePlyBinary(dataTable, comments)), pool);
    return { source, pool };
};

// A 2DGS table: the standard columns minus scale_2.
const make2dgsTable = (count = 16) => {
    const table = createTestDataTable(count);
    return new DataTable(table.columns.filter(c => c.name !== 'scale_2'), table.transform);
};

// The header text of a PLY, up to and including end_header.
const plyHeaderText = (bytes) => {
    const text = Buffer.from(bytes.buffer, bytes.byteOffset, Math.min(bytes.byteLength, 4096)).toString('latin1');
    return text.substring(0, text.indexOf('end_header') + 'end_header'.length);
};

// Write chunk-natively — the path the CLI takes (streaming PLY, native SOG).
const writeChunked = async (source, pool, filename, outputFormat) => {
    const fs = new MemoryFileSystem();
    await writeSource({ filename, outputFormat, source, pool, options: {} }, fs);
    return fs.results;
};

// Write through the DataTable writers (the compat API surface).
const writeTabular = async (source, pool, filename, outputFormat) => {
    const fs = new MemoryFileSystem();
    const dataTable = await materializeToDataTable(source, pool);
    await writeFile({ filename, outputFormat, dataTable, model: source.meta.model, options: {} }, fs);
    return fs.results;
};

describe('splatModelFromComments', () => {
    it('reads Brush\'s SplatRenderMode', () => {
        assert.strictEqual(splatModelFromComments(['SplatRenderMode: mip']), 'antialiased');
        assert.strictEqual(splatModelFromComments(['SplatRenderMode: default']), 'default');
        assert.strictEqual(splatModelFromComments(['SplatRenderMode: 2dgs']), '2dgs');
    });

    it('reads Postshot\'s antialiased flag', () => {
        assert.strictEqual(splatModelFromComments(['antialiased 1']), 'antialiased');
        assert.strictEqual(splatModelFromComments(['antialiased 0']), 'default');
    });

    it('ignores case, surrounding whitespace and unrelated comments', () => {
        assert.strictEqual(splatModelFromComments(['  SPLATRENDERMODE:  MIP  ']), 'antialiased');
        assert.strictEqual(splatModelFromComments(['Exported from Brush', 'SH degree: 3']), 'default');
        assert.strictEqual(splatModelFromComments([]), 'default');
    });

    it('falls back to default for an unknown mode', () => {
        assert.strictEqual(splatModelFromComments(['SplatRenderMode: banana']), 'default');
    });

    it('takes the last match across both forms', () => {
        assert.strictEqual(splatModelFromComments(['antialiased 1', 'SplatRenderMode: default']), 'default');
        assert.strictEqual(splatModelFromComments(['SplatRenderMode: default', 'antialiased 1']), 'antialiased');
    });

    it('emits Brush\'s spelling, and nothing for default', () => {
        assert.strictEqual(splatModelComment('antialiased'), 'SplatRenderMode: mip');
        assert.strictEqual(splatModelComment('2dgs'), 'SplatRenderMode: 2dgs');
        assert.strictEqual(splatModelComment('default'), null);
    });
});

describe('antialiased scenes', () => {
    it('round-trips a Brush-tagged PLY', async () => {
        const { source, pool } = await openPly(createTestDataTable(16), [
            'Exported from Brush', 'SH degree: 0', 'SplatRenderMode: mip'
        ]);
        assert.strictEqual(source.meta.model, 'antialiased');

        const out = await writeChunked(source, pool, 'out.ply', 'ply');
        assert.match(plyHeaderText(out.get('out.ply')), /comment SplatRenderMode: mip/);
    });

    it('retags a Postshot-tagged PLY in Brush\'s spelling', async () => {
        const { source, pool } = await openPly(createTestDataTable(16), ['antialiased 1']);
        assert.strictEqual(source.meta.model, 'antialiased');

        const header = plyHeaderText((await writeChunked(source, pool, 'out.ply', 'ply')).get('out.ply'));
        assert.match(header, /comment SplatRenderMode: mip/);
        assert.doesNotMatch(header, /antialiased/);
    });

    it('tags compressed PLY output', async () => {
        const { source, pool } = await openPly(createTestDataTable(16), ['antialiased 1']);
        const out = await writeChunked(source, pool, 'out.compressed.ply', 'compressed-ply');
        assert.match(plyHeaderText(out.get('out.compressed.ply')), /comment SplatRenderMode: mip/);
    });

    it('tags SOG meta.json', async () => {
        const { source, pool } = await openPly(createTestDataTable(16), ['SplatRenderMode: mip']);
        const out = await writeChunked(source, pool, 'meta.json', 'sog');
        const meta = JSON.parse(Buffer.from(out.get('meta.json')).toString());
        assert.strictEqual(meta.model, 'antialiased');
    });

    it('sets the SPZ antialiased header bit', async () => {
        const { source, pool } = await openPly(createTestDataTable(16), ['SplatRenderMode: mip']);
        const out = await writeChunked(source, pool, 'out.spz', 'spz');
        const bytes = out.get('out.spz');
        assert.strictEqual(bytes[14] & 0x1, 0x1);

        // and comes back as antialiased on read
        const rfs = new MemoryReadFileSystem();
        rfs.set('in.spz', bytes);
        const { readSpz } = await import('../src/lib/readers/read-spz.js');
        const pool2 = createChunkDataPool();
        const spzSource = await readSpz(await rfs.createSource('in.spz'), pool2);
        assert.strictEqual(spzSource.meta.model, 'antialiased');
    });
});

describe('2dgs scenes', () => {
    it('reads a PLY with no scale_2 as a full geometric layer', async () => {
        const { source, pool } = await openPly(make2dgsTable(16));
        assert.strictEqual(source.meta.model, '2dgs');
        assert.ok(source.meta.availableLayers.has('geometric'), 'geometric layer present');
        assert.strictEqual(source.meta.extraColumns.length, 0, 'scale_2 is absent, not extra');

        const table = await materializeToDataTable(source, pool);
        const scale2 = table.getColumnByName('scale_2').data;
        const scale1 = table.getColumnByName('scale_1').data;
        const opacity = table.getColumnByName('opacity').data;
        const reference = createTestDataTable(16);
        for (let i = 0; i < 16; i++) {
            assert.strictEqual(scale2[i], -Infinity, `row ${i} scale_2`);
            // the columns either side of the gap still land in their own slots
            assert.strictEqual(scale1[i], reference.getColumnByName('scale_1').data[i], `row ${i} scale_1`);
            assert.strictEqual(opacity[i], reference.getColumnByName('opacity').data[i], `row ${i} opacity`);
        }
    });

    it('structural evidence outranks a contradicting comment', async () => {
        const { source } = await openPly(make2dgsTable(8), ['SplatRenderMode: mip']);
        assert.strictEqual(source.meta.model, '2dgs');
    });

    // Two scales alone don't make a 2DGS scene — a point cloud missing rotation
    // or opacity has no geometric layer to tag, so it must stay untagged rather
    // than claiming a model it can't honour.
    it('does not infer 2dgs from an incomplete geometric record', async () => {
        const table = createTestDataTable(8);
        const keep = ['x', 'y', 'z', 'scale_0', 'scale_1', 'f_dc_0', 'f_dc_1', 'f_dc_2'];
        const partial = new DataTable(table.columns.filter(c => keep.includes(c.name)), table.transform);

        const { source } = await openPly(partial);
        assert.strictEqual(source.meta.model, 'default');
        assert.ok(!source.meta.availableLayers.has('geometric'), 'no geometric layer');
    });

    it('drops scale_2 again on PLY output, keeping the tag', async () => {
        const { source, pool } = await openPly(make2dgsTable(16));
        const header = plyHeaderText((await writeChunked(source, pool, 'out.ply', 'ply')).get('out.ply'));
        assert.match(header, /comment SplatRenderMode: 2dgs/);
        assert.match(header, /property float scale_1/);
        assert.doesNotMatch(header, /scale_2/);
    });

    // The streaming writer copies each layer as contiguous 32-bit word runs;
    // omitting a column mid-layer splits the geometric run in two, so check the
    // values on the far side of the gap still land in the right output column.
    it('writes correct values around the dropped column (both writers)', async () => {
        const reference = createTestDataTable(16);
        for (const write of [writeChunked, writeTabular]) {
            const { source, pool } = await openPly(make2dgsTable(16));
            const bytes = (await write(source, pool, 'out.ply', 'ply')).get('out.ply');

            const pool2 = createChunkDataPool();
            const table = await materializeToDataTable(
                await readPly(await sourceFromBytes(bytes), pool2), pool2
            );
            for (const name of ['rot_3', 'scale_0', 'scale_1', 'opacity', 'f_dc_0']) {
                assert.deepStrictEqual(
                    Array.from(table.getColumnByName(name).data),
                    Array.from(reference.getColumnByName(name).data),
                    `${name} via ${write === writeChunked ? 'writeSource' : 'writeFile'}`
                );
            }
            // re-reading the output re-materializes the column
            assert.strictEqual(table.getColumnByName('scale_2').data[0], -Infinity);
        }
    });

    // SPZ can't hold the tag, so the flat axis has to survive as data. Its
    // quantized log-scale range saturates, which is what turns the synthesized
    // -Infinity into an encodable value — the writer does no clamping of its own,
    // so this guards against a future encoder emitting garbage for it.
    it('encodes the flat axis to SPZ as a finite minimal scale', async () => {
        const { source, pool } = await openPly(make2dgsTable(16));
        const bytes = (await writeChunked(source, pool, 'out.spz', 'spz')).get('out.spz');

        const rfs = new MemoryReadFileSystem();
        rfs.set('in.spz', bytes);
        const { readSpz } = await import('../src/lib/readers/read-spz.js');
        const pool2 = createChunkDataPool();
        const table = await materializeToDataTable(await readSpz(await rfs.createSource('in.spz'), pool2), pool2);

        const scale2 = table.getColumnByName('scale_2').data;
        const scale1 = table.getColumnByName('scale_1').data;
        for (let i = 0; i < 16; i++) {
            assert.ok(Number.isFinite(scale2[i]), `row ${i} scale_2 is finite (got ${scale2[i]})`);
            assert.ok(scale2[i] < scale1[i], `row ${i} scale_2 is the flattest axis`);
        }
    });

    it('tags SOG meta.json and keeps three scale channels', async () => {
        const { source, pool } = await openPly(make2dgsTable(16));
        const out = await writeChunked(source, pool, 'meta.json', 'sog');
        const meta = JSON.parse(Buffer.from(out.get('meta.json')).toString());
        assert.strictEqual(meta.model, '2dgs');
        assert.ok(meta.scales.codebook.length > 0, 'scales are still encoded');
    });
});

describe('combining sources', () => {
    it('keeps a shared model', async () => {
        const pool = createChunkDataPool();
        const a = await readPly(await sourceFromBytes(encodePlyBinary(createTestDataTable(16), ['antialiased 1'])), pool);
        const b = await readPly(await sourceFromBytes(encodePlyBinary(createTestDataTable(16), ['SplatRenderMode: mip'])), pool);
        assert.strictEqual(concatSource([a, b], pool).meta.model, 'antialiased');
    });

    it('falls back to default when models disagree', async () => {
        const pool = createChunkDataPool();
        const aa = await readPly(await sourceFromBytes(encodePlyBinary(createTestDataTable(16), ['antialiased 1'])), pool);
        const plain = await readPly(await sourceFromBytes(encodePlyBinary(createTestDataTable(16))), pool);
        assert.strictEqual(concatSource([aa, plain], pool).meta.model, 'default');

        assert.strictEqual(resolveSplatModel(['2dgs', 'default']), 'default');
        assert.strictEqual(resolveSplatModel(['antialiased', '2dgs']), 'default');
        assert.strictEqual(resolveSplatModel([]), 'default');
    });
});

describe('untagged scenes', () => {
    it('adds no comment and no meta key', async () => {
        const { source, pool } = await openPly(createTestDataTable(16));
        assert.strictEqual(source.meta.model, 'default');

        const ply = await writeChunked(source, pool, 'out.ply', 'ply');
        assert.doesNotMatch(plyHeaderText(ply.get('out.ply')), /comment/);

        const { source: s2, pool: p2 } = await openPly(createTestDataTable(16));
        const sog = await writeChunked(s2, p2, 'meta.json', 'sog');
        const meta = JSON.parse(Buffer.from(sog.get('meta.json')).toString());
        assert.ok(!('model' in meta), 'no model key for an untagged scene');
    });
});
