/**
 * Tier-1 op tests: mortonOrder pass + permuteSource combinator.
 *
 *  - mortonOrder produces the same permutation as the legacy sortMortonOrder
 *    (deterministic, so byte-exact equality is the right gate here).
 *  - permuteSource reorders rows by a permutation, preserving every value,
 *    including across chunk boundaries (random-access into a resident parent).
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

import { createTestDataTable } from './helpers/test-utils.mjs';
import { dataTableToChunkSource, materializeToDataTable } from '../src/lib/compat/data-table.js';
import { sortMortonOrder } from '../src/lib/data-table/morton-order.js';
import { mortonOrder, permuteSource } from '../src/lib/ops/index.js';
import { createChunkDataPool } from '../src/lib/source/index.js';

describe('ops: mortonOrder + permuteSource', () => {
    it('mortonOrder matches legacy sortMortonOrder (recursive-refined)', async () => {
        const dt = createTestDataTable(500);
        const chunkSize = 64; // multi-chunk + short final chunk
        const src = dataTableToChunkSource(dt, chunkSize);
        const pool = createChunkDataPool({ chunkSize });

        const order = await mortonOrder(src, pool);

        const legacy = new Uint32Array(dt.numRows);
        for (let i = 0; i < legacy.length; i++) legacy[i] = i;
        sortMortonOrder(dt, legacy);

        assert.deepStrictEqual(order, legacy, 'mortonOrder permutation must equal legacy sortMortonOrder');
    });

    it('permuteSource reorders rows by the permutation, preserving values across chunks', async () => {
        const dt = createTestDataTable(300, { includeSH: true, shBands: 1 });
        const chunkSize = 64;
        const resident = dataTableToChunkSource(dt, chunkSize); // already an InMemoryChunkSource
        const pool = createChunkDataPool({ chunkSize });

        const n = dt.numRows;
        const order = new Uint32Array(n);
        for (let i = 0; i < n; i++) order[i] = n - 1 - i; // reverse: mixes rows across chunk boundaries

        const permuted = permuteSource(resident, order);
        const out = await materializeToDataTable(permuted, pool);

        assert.strictEqual(out.numRows, n);
        assert.deepStrictEqual([...out.columnNames].sort(), [...dt.columnNames].sort());
        for (const name of dt.columnNames) {
            const inCol = dt.getColumnByName(name).data;
            const outCol = out.getColumnByName(name).data;
            for (let i = 0; i < n; i++) {
                assert.strictEqual(outCol[i], inCol[order[i]], `column '${name}' row ${i}`);
            }
        }
    });

    it('permuteSource rejects a non-resident or wrong-length input', () => {
        const dt = createTestDataTable(40);
        const resident = dataTableToChunkSource(dt, 16);
        assert.throws(() => permuteSource(resident, new Uint32Array(39)), /order length/);
        assert.throws(() => permuteSource({ meta: {} }, new Uint32Array(40)), /resident InMemoryChunkSource/);
    });
});
