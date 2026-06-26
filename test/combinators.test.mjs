/**
 * Tier-1 combinator tests: filterSource + concatSource.
 *
 *  - filterSource selects an ascending subset of a parent's rows, gathering them
 *    across chunk boundaries; the output rows must equal the parent rows at the
 *    selected indices, for every layer.
 *  - concatSource joins sources end-to-end; materializing it must match the
 *    legacy `combine()` of the same tables (the A/B oracle), including when
 *    per-source counts don't align to the chunk size. It must refuse sources in
 *    mismatched coordinate spaces.
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

import { createTestDataTable } from './helpers/test-utils.mjs';
import { combine, Transform } from '../src/lib/index.js';
import { dataTableToChunkSource, materializeToDataTable } from '../src/lib/compat/data-table.js';
import { concatSource, filterSource, permuteSource } from '../src/lib/ops/index.js';
import { createChunkDataPool } from '../src/lib/source/index.js';

// Compare two DataTables column-by-column (by name), exact.
const assertSameRows = (out, expected) => {
    assert.strictEqual(out.numRows, expected.numRows, 'row count');
    assert.deepStrictEqual([...out.columnNames].sort(), [...expected.columnNames].sort(), 'columns');
    for (const name of expected.columnNames) {
        const o = out.getColumnByName(name).data;
        const e = expected.getColumnByName(name).data;
        for (let i = 0; i < expected.numRows; i++) {
            assert.strictEqual(o[i], e[i], `column '${name}' row ${i}`);
        }
    }
};

describe('combinators: filterSource', () => {
    it('selects an ascending subset, preserving values across chunks', async () => {
        const dt = createTestDataTable(300, { includeSH: true, shBands: 1 });
        const chunkSize = 64;
        const parent = dataTableToChunkSource(dt, chunkSize);
        const pool = createChunkDataPool({ chunkSize });

        // every 3rd row, spanning all chunks
        const idx = [];
        for (let i = 0; i < dt.numRows; i += 3) idx.push(i);
        const indices = new Uint32Array(idx);

        const out = await materializeToDataTable(filterSource(parent, indices, pool), pool);

        assert.strictEqual(out.numRows, indices.length);
        assert.deepStrictEqual([...out.columnNames].sort(), [...dt.columnNames].sort());
        for (const name of dt.columnNames) {
            const inCol = dt.getColumnByName(name).data;
            const outCol = out.getColumnByName(name).data;
            for (let i = 0; i < indices.length; i++) {
                assert.strictEqual(outCol[i], inCol[indices[i]], `column '${name}' row ${i}`);
            }
        }
    });

    it('full selection reproduces the parent; empty selection yields no rows', async () => {
        const dt = createTestDataTable(130);
        const chunkSize = 64;
        const parent = dataTableToChunkSource(dt, chunkSize);
        const pool = createChunkDataPool({ chunkSize });

        const all = new Uint32Array(dt.numRows);
        for (let i = 0; i < all.length; i++) all[i] = i;
        const full = await materializeToDataTable(filterSource(parent, all, pool), pool);
        assertSameRows(full, dt);

        const empty = await materializeToDataTable(filterSource(parent, new Uint32Array(0), pool), pool);
        assert.strictEqual(empty.numRows, 0);
    });
});

describe('combinators: concatSource', () => {
    it('matches combine() across mismatched chunk boundaries', async () => {
        const chunkSize = 64;
        // Counts chosen so source boundaries fall mid-output-chunk.
        const dts = [
            createTestDataTable(100, { includeSH: true, shBands: 1 }),
            createTestDataTable(50, { includeSH: true, shBands: 1 }),
            createTestDataTable(70, { includeSH: true, shBands: 1 })
        ];
        // Tag each source so an ordering bug would surface (f_dc_2 is otherwise
        // constant across these tables).
        dts.forEach((dt, s) => dt.getColumnByName('f_dc_2').data.fill(s + 1));

        const pool = createChunkDataPool({ chunkSize });
        const sources = dts.map(dt => dataTableToChunkSource(dt, chunkSize));
        const out = await materializeToDataTable(concatSource(sources, pool), pool);

        const expected = combine(dts);
        assertSameRows(out, expected);
    });

    it('throws on a transform mismatch', () => {
        const chunkSize = 16;
        const pool = createChunkDataPool({ chunkSize });
        const a = createTestDataTable(20);
        const b = createTestDataTable(20);
        b.transform = new Transform().fromEulers(0, 0, 90);
        const sa = dataTableToChunkSource(a, chunkSize);
        const sb = dataTableToChunkSource(b, chunkSize);
        assert.throws(() => concatSource([sa, sb], pool), /transform mismatch/);
    });

    it('throws on an SH-band mismatch', () => {
        const chunkSize = 16;
        const pool = createChunkDataPool({ chunkSize });
        const sa = dataTableToChunkSource(createTestDataTable(20, { includeSH: true, shBands: 1 }), chunkSize);
        const sb = dataTableToChunkSource(createTestDataTable(20, { includeSH: true, shBands: 2 }), chunkSize);
        assert.throws(() => concatSource([sa, sb], pool), /SH band mismatch/);
    });

    it('forwards readRows (random-access gather across inputs)', async () => {
        const chunkSize = 64;
        const dts = [
            createTestDataTable(100, { includeSH: true, shBands: 1 }),
            createTestDataTable(50, { includeSH: true, shBands: 1 }),
            createTestDataTable(70, { includeSH: true, shBands: 1 })
        ];
        dts.forEach((dt, s) => dt.getColumnByName('f_dc_2').data.fill(s + 1));

        const pool = createChunkDataPool({ chunkSize });
        const concat = concatSource(dts.map(dt => dataTableToChunkSource(dt, chunkSize)), pool);
        assert.strictEqual(typeof concat.readRows, 'function', 'concat of gatherable sources should forward readRows');

        // Scattered order crossing input boundaries (inputs 0/1/2), with a repeat.
        const order = new Uint32Array([219, 0, 100, 99, 150, 149, 100, 75]);
        const gathered = await materializeToDataTable(permuteSource(concat, order), pool);

        const expected = combine(dts);
        assert.strictEqual(gathered.numRows, order.length);
        for (const name of expected.columnNames) {
            const g = gathered.getColumnByName(name).data;
            const e = expected.getColumnByName(name).data;
            for (let i = 0; i < order.length; i++) {
                assert.strictEqual(g[i], e[order[i]], `column '${name}' row ${i} (src ${order[i]})`);
            }
        }
    });
});
