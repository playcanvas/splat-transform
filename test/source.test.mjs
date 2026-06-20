/**
 * M0 tests for the CPU-primary ChunkSource data model:
 *  - DataTable <-> source round-trip byte-equality (via the migration shims)
 *  - ChunkDataPool pooling / reuse / trim
 *  - ChunkData strided field extraction
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

import { createTestDataTable } from './helpers/test-utils.mjs';
import { dataTableToChunkSource, materializeToDataTable } from '../src/lib/compat/data-table.js';
import { Column, DataTable } from '../src/lib/index.js';
import {
    createChunkDataPool,
    compact,
    POSITION_STRIDE,
    GEOMETRIC_STRIDE,
    positionFields,
    geometricFields
} from '../src/lib/source/index.js';

function assertTablesEqual(actual, expected, msg) {
    const aNames = [...actual.columnNames].sort();
    const eNames = [...expected.columnNames].sort();
    assert.deepStrictEqual(aNames, eNames, `${msg}: column names differ`);
    assert.strictEqual(actual.numRows, expected.numRows, `${msg}: row count differs`);
    for (const name of eNames) {
        const a = actual.getColumnByName(name).data;
        const e = expected.getColumnByName(name).data;
        assert.deepStrictEqual(a, e, `${msg}: column '${name}' differs`);
    }
}

describe('ChunkSource data model (M0)', () => {
    describe('DataTable <-> source round-trip', () => {
        it('round-trips byte-equal across multiple chunks with a short final chunk', async () => {
            const dt = createTestDataTable(10, { includeSH: true, shBands: 2 });
            const chunkSize = 4; // -> chunks of 4, 4, 2

            const src = dataTableToChunkSource(dt, chunkSize);
            assert.strictEqual(src.meta.numGaussians, 10);
            assert.strictEqual(src.meta.shBands, 2);
            assert.deepStrictEqual([...src.meta.numChunks], [3]);

            const pool = createChunkDataPool({ chunkSize });
            const out = await materializeToDataTable(src, pool);
            assertTablesEqual(out, dt, 'round-trip');
        });

        it('round-trips extra columns and partial layers (position + other only)', async () => {
            const dt = new DataTable([
                new Column('x', new Float32Array([0, 1, 2, 3, 4])),
                new Column('y', new Float32Array([10, 11, 12, 13, 14])),
                new Column('z', new Float32Array([20, 21, 22, 23, 24])),
                new Column('weird', new Float32Array([0.5, 1.5, 2.5, 3.5, 4.5]))
            ]);
            const chunkSize = 2; // -> chunks of 2, 2, 1

            const src = dataTableToChunkSource(dt, chunkSize);
            assert.ok(src.meta.availableLayers.has('position'));
            assert.ok(src.meta.availableLayers.has('other'));
            assert.ok(!src.meta.availableLayers.has('geometric'));
            assert.ok(!src.meta.availableLayers.has('color'));
            assert.strictEqual(src.meta.shBands, 0);
            assert.deepStrictEqual(src.meta.extraColumns, [{ name: 'weird', type: 'float32' }]);

            const pool = createChunkDataPool({ chunkSize });
            const out = await materializeToDataTable(src, pool);
            assertTablesEqual(out, dt, 'partial round-trip');
        });

        it('compact() materializes a source identically', async () => {
            const dt = createTestDataTable(7, { includeSH: true, shBands: 1 });
            const chunkSize = 3; // -> chunks of 3, 3, 1
            const pool = createChunkDataPool({ chunkSize });

            const src = dataTableToChunkSource(dt, chunkSize);
            const compacted = await compact(src, pool);

            assert.strictEqual(compacted.meta.numGaussians, 7);
            assert.deepStrictEqual([...compacted.meta.numChunks], [3]);

            const out = await materializeToDataTable(compacted, pool);
            assertTablesEqual(out, dt, 'compact round-trip');
        });
    });

    describe('ChunkDataPool', () => {
        it('needs no GraphicsDevice and reports a chunkSize', () => {
            const m = createChunkDataPool({ chunkSize: 8 });
            assert.strictEqual(m.chunkSize, 8);
            assert.strictEqual(m.bytesInUse, 0);
            assert.strictEqual(m.bytesPooled, 0);
        });

        it('pools and reuses released buffers', () => {
            const chunkSize = 4;
            const m = createChunkDataPool({ chunkSize });
            const layout = { stride: POSITION_STRIDE, fields: positionFields() };
            const cap = chunkSize * POSITION_STRIDE;

            const c1 = m.acquire('position', layout, 4);
            assert.strictEqual(m.bytesInUse, cap);
            assert.strictEqual(m.bytesPooled, 0);
            const buf1 = c1.data;

            c1.release();
            assert.strictEqual(m.bytesInUse, 0);
            assert.strictEqual(m.bytesPooled, cap);

            const c2 = m.acquire('position', layout, 4);
            assert.strictEqual(c2.data, buf1, 'should reuse the pooled buffer');
            assert.strictEqual(m.bytesInUse, cap);
            assert.strictEqual(m.bytesPooled, 0);
            c2.release();
        });

        it('reuses a full chunk slot for a short final chunk (no fragmentation)', () => {
            const chunkSize = 4;
            const m = createChunkDataPool({ chunkSize });
            const layout = { stride: POSITION_STRIDE, fields: positionFields() };

            const full = m.acquire('position', layout, 4);
            const buf = full.data;
            full.release();

            const short = m.acquire('position', layout, 2);
            assert.strictEqual(short.data, buf, 'short chunk should reuse the full-capacity buffer');
            assert.strictEqual(short.data.byteLength, chunkSize * POSITION_STRIDE);
            assert.strictEqual(short.count, 2);
            short.release();
        });

        it('rejects counts larger than the chunk size', () => {
            const m = createChunkDataPool({ chunkSize: 4 });
            const layout = { stride: POSITION_STRIDE, fields: positionFields() };
            assert.throws(() => m.acquire('position', layout, 5), /exceeds chunkSize/);
        });

        it('trim() and destroy() free pooled buffers', () => {
            const chunkSize = 4;
            const m = createChunkDataPool({ chunkSize });
            const layout = { stride: GEOMETRIC_STRIDE, fields: geometricFields() };

            m.acquire('geometric', layout, 4).release();
            m.acquire('geometric', layout, 4).release();
            assert.ok(m.bytesPooled > 0);

            m.trim(0);
            assert.strictEqual(m.bytesPooled, 0);

            m.acquire('geometric', layout, 4).release();
            assert.ok(m.bytesPooled > 0);
            m.destroy();
            assert.strictEqual(m.bytesPooled, 0);
        });
    });

    describe('ChunkData.field()', () => {
        it('extracts strided fields from an interleaved layer record', () => {
            const m = createChunkDataPool({ chunkSize: 4 });
            const layout = { stride: GEOMETRIC_STRIDE, fields: geometricFields() };
            const c = m.acquire('geometric', layout, 2);

            // two gaussians: [rot0..3, scale0..2, opacity]
            new Float32Array(c.data, 0, 16).set([
                1, 2, 3, 4, 5, 6, 7, 8,
                9, 10, 11, 12, 13, 14, 15, 16
            ]);

            assert.deepStrictEqual(c.field('rotation'), new Float32Array([1, 2, 3, 4, 9, 10, 11, 12]));
            assert.deepStrictEqual(c.field('scale'), new Float32Array([5, 6, 7, 13, 14, 15]));
            assert.deepStrictEqual(c.field('opacity'), new Float32Array([8, 16]));
            c.release();
        });
    });
});
