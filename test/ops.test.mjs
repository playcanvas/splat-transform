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

import { createTestDataTable, encodePlyBinary } from './helpers/test-utils.mjs';
import { dataTableToChunkSource, materializeToDataTable } from '../src/lib/compat/data-table.js';
import { sortMortonOrder } from '../src/lib/data-table/morton-order.js';
import { readPly } from '../src/lib/readers/read-ply.js';
import { mortonOrder, permuteSource, selectLod, stackLods } from '../src/lib/ops/index.js';
import { createChunkDataPool } from '../src/lib/chunk/index.js';

// Minimal seekable ReadSource over a buffer (range reads), for exercising the
// PLY reader's random-access gather path.
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

    it('permuteSource gathers an ordered subset (shorter than the parent)', async () => {
        const dt = createTestDataTable(300, { includeSH: true, shBands: 1 });
        const chunkSize = 64;
        const resident = dataTableToChunkSource(dt, chunkSize);
        const pool = createChunkDataPool({ chunkSize });

        // Pick a non-ascending subset that straddles chunk boundaries.
        const order = new Uint32Array([299, 0, 150, 64, 63, 200]);
        const out = await materializeToDataTable(permuteSource(resident, order), pool);

        assert.strictEqual(out.numRows, order.length);
        for (const name of dt.columnNames) {
            const inCol = dt.getColumnByName(name).data;
            const outCol = out.getColumnByName(name).data;
            for (let i = 0; i < order.length; i++) {
                assert.strictEqual(outCol[i], inCol[order[i]], `column '${name}' row ${i}`);
            }
        }
    });

    it('permuteSource rejects an over-length order', () => {
        const dt = createTestDataTable(40);
        const resident = dataTableToChunkSource(dt, 16);
        assert.throws(() => permuteSource(resident, new Uint32Array(41)), /exceeds lod 0 count/);
    });

    it('gathers identically from a disk-backed PLY and a resident source', async () => {
        // The LOD writer's "positions resident, heavy data fetched per output
        // chunk" model: a scattered subset gathered from a fixed-stride PLY via
        // readRows must equal the same gather from a resident InMemory source.
        const dt = createTestDataTable(300, { includeSH: true, shBands: 1 });
        const chunkSize = 64;
        const pool = createChunkDataPool({ chunkSize });

        const resident = dataTableToChunkSource(dt, chunkSize);
        const diskSrc = await readPly(new BufferReadSource(encodePlyBinary(dt)), pool);

        // Non-ascending, chunk-straddling, with a repeated index.
        const order = new Uint32Array([299, 0, 150, 64, 63, 200, 7, 7, 256]);

        const fromResident = await materializeToDataTable(permuteSource(resident, order), pool);
        const fromDisk = await materializeToDataTable(permuteSource(diskSrc, order), pool);

        assert.deepStrictEqual([...fromDisk.columnNames].sort(), [...fromResident.columnNames].sort());
        assert.strictEqual(fromDisk.numRows, order.length);
        for (const name of fromResident.columnNames) {
            const d = fromDisk.getColumnByName(name).data;
            const r = fromResident.getColumnByName(name).data;
            for (let i = 0; i < order.length; i++) {
                assert.strictEqual(d[i], r[i], `column '${name}' row ${i} (src ${order[i]})`);
            }
        }
        await diskSrc.close();
    });

    it('permuteSource serves a gather request (gather-of-gather)', async () => {
        const dt = createTestDataTable(120, { includeSH: true, shBands: 1 });
        const chunkSize = 32;
        const pool = createChunkDataPool({ chunkSize });
        const resident = dataTableToChunkSource(dt, chunkSize);
        const order = new Uint32Array([100, 5, 60, 7, 119, 0, 33]); // the permutation
        const perm = permuteSource(resident, order);

        // Read the permuted view by explicit indices (not chunkIndex): output row
        // j is parent row order[sel[j]] (the gather composes through `order`).
        const sel = new Uint32Array([4, 1, 6, 0]);
        const count = sel.length;
        const acq = {};
        for (const layer of ['position', 'geometric', 'color']) {
            acq[layer] = pool.acquire(layer, perm.meta.layouts[layer], count);
        }
        await perm.read({ indices: sel, indexOffset: 0, count, position: acq.position, geometric: acq.geometric, color: acq.color });

        const px = acq.position.field('position');
        const op = acq.geometric.field('opacity');
        const x = dt.getColumnByName('x').data;
        const opF = dt.getColumnByName('opacity').data;
        for (let j = 0; j < count; j++) {
            const e = order[sel[j]];
            assert.strictEqual(px[j * 3], x[e], `x out-row ${j}`);
            assert.strictEqual(op[j], opF[e], `opacity out-row ${j}`);
        }
        for (const layer of ['position', 'geometric', 'color']) acq[layer].release();
    });
});

describe('ops: selectLod', () => {
    it('views each level of a stacked multi-LOD source', async () => {
        const chunkSize = 64;
        const pool = createChunkDataPool({ chunkSize });
        const dt0 = createTestDataTable(100, { includeSH: true, shBands: 1 });
        const dt1 = createTestDataTable(60, { includeSH: true, shBands: 1 });
        dt0.getColumnByName('f_dc_2').data.fill(7); // tag the levels so a wrong pick shows
        dt1.getColumnByName('f_dc_2').data.fill(9);

        const multi = stackLods([dataTableToChunkSource(dt0, chunkSize), dataTableToChunkSource(dt1, chunkSize)]);
        assert.strictEqual(multi.meta.numLods, 2);

        const lvl0 = selectLod(multi, 0);
        const lvl1 = selectLod(multi, 1);
        assert.strictEqual(lvl0.meta.numLods, 1);
        assert.strictEqual(lvl0.meta.numGaussians, 100);
        assert.strictEqual(lvl1.meta.numGaussians, 60);

        const out0 = await materializeToDataTable(lvl0, pool);
        const out1 = await materializeToDataTable(lvl1, pool);
        assert.strictEqual(out0.numRows, 100);
        assert.strictEqual(out1.numRows, 60);
        for (const name of dt0.columnNames) {
            const a = out0.getColumnByName(name).data, e = dt0.getColumnByName(name).data;
            for (let i = 0; i < 100; i++) assert.strictEqual(a[i], e[i], `lvl0 '${name}' row ${i}`);
        }
        for (const name of dt1.columnNames) {
            const a = out1.getColumnByName(name).data, e = dt1.getColumnByName(name).data;
            for (let i = 0; i < 60; i++) assert.strictEqual(a[i], e[i], `lvl1 '${name}' row ${i}`);
        }
        assert.throws(() => selectLod(multi, 2), /out of range/);
        await multi.close();
    });
});
