/**
 * A/B tests: processSource (chunk path) vs processDataTable (DataTable oracle).
 *
 * For each supported action (and combinations), the chunk path must produce the
 * same scene as the DataTable path. Equivalence is checked after baking BOTH
 * results into engine space (Transform.IDENTITY) — the writers bake on output,
 * and a transform-baking filterByValue bakes the DataTable early, so comparing
 * baked-to-identity is the meaningful, ordering-independent gate. The same
 * convertToSpace runs on both sides, so the comparison is exact.
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';
import { Vec3 } from 'playcanvas';

import { createTestDataTable } from './helpers/test-utils.mjs';
import { convertToSpace, processDataTable, Transform } from '../src/lib/index.js';
import { dataTableToChunkSource, materializeToDataTable } from '../src/lib/compat/data-table.js';
import { processSource } from '../src/lib/process-source.js';
import { createChunkDataPool } from '../src/lib/chunk/index.js';

const approxEqual = (a, b) => {
    if (!Number.isFinite(a) || !Number.isFinite(b)) return Object.is(a, b);
    return Math.abs(a - b) <= 1e-4 + 1e-4 * Math.abs(a);
};

// Bake both results into engine space, then compare every column by name.
const assertEquivalent = (tableA, tableB) => {
    const a = convertToSpace(tableA, Transform.IDENTITY);
    const b = convertToSpace(tableB, Transform.IDENTITY);
    assert.strictEqual(b.numRows, a.numRows, 'row count');
    assert.deepStrictEqual([...b.columnNames].sort(), [...a.columnNames].sort(), 'columns');
    for (const name of a.columnNames) {
        const ad = a.getColumnByName(name).data;
        const bd = b.getColumnByName(name).data;
        for (let i = 0; i < a.numRows; i++) {
            assert.ok(approxEqual(ad[i], bd[i]), `column '${name}' row ${i}: ${ad[i]} != ${bd[i]}`);
        }
    }
};

// Run the same actions through both paths and compare. `inject` (optional) is
// applied identically to each path's own fresh DataTable.
const runAB = async (count, opts, actions, inject) => {
    const chunkSize = 64;
    const make = () => {
        const dt = createTestDataTable(count, opts);
        if (inject) inject(dt);
        return dt;
    };
    const pool = createChunkDataPool({ chunkSize });
    const source = dataTableToChunkSource(make(), chunkSize);
    const tableA = await processDataTable(make(), actions);
    const tableB = await materializeToDataTable(await processSource(source, actions, pool), pool);
    return { tableA, tableB };
};

describe('processSource A/B vs processDataTable', () => {
    it('composed translate/rotate/scale (with SH rotation)', async () => {
        const { tableA, tableB } = await runAB(300, { includeSH: true, shBands: 1 }, [
            { kind: 'scale', value: 0.5 },
            { kind: 'translate', value: new Vec3(1, 2, 3) },
            { kind: 'rotate', value: new Vec3(0, 0, 90) }
        ]);
        assertEquivalent(tableA, tableB);
    });

    it('filterNaN drops the same rows (with +Inf opacity / -Inf scale kept)', async () => {
        const inject = (dt) => {
            const col = n => dt.getColumnByName(n).data;
            col('x')[5] = NaN;            // drop
            col('opacity')[10] = Infinity;  // keep (+Inf ok for opacity)
            col('scale_0')[15] = -Infinity; // keep (-Inf ok for scale)
            col('z')[20] = Infinity;        // drop
            col('f_dc_0')[25] = NaN;        // drop
            col('opacity')[30] = -Infinity; // drop (-Inf not ok for opacity)
            col('scale_1')[35] = Infinity;  // drop (+Inf not ok for scale)
        };
        const { tableA, tableB } = await runAB(300, {}, [{ kind: 'filterNaN' }], inject);
        assert.strictEqual(tableA.numRows, 295);
        assertEquivalent(tableA, tableB);
    });

    it('filterByValue on a non-transform column (f_dc_0)', async () => {
        const { tableA, tableB } = await runAB(300, {}, [
            { kind: 'filterByValue', columnName: 'f_dc_0', comparator: 'gt', value: 0.5 }
        ]);
        assert.ok(tableA.numRows > 0 && tableA.numRows < 300, 'filter should keep a strict subset');
        assertEquivalent(tableA, tableB);
    });

    it('filterByValue on a transform column (x) after a translate — bakes the comparison', async () => {
        const { tableA, tableB } = await runAB(300, {}, [
            { kind: 'translate', value: new Vec3(5, 0, 0) },
            { kind: 'filterByValue', columnName: 'x', comparator: 'gt', value: 5.5 }
        ]);
        assert.ok(tableA.numRows > 0 && tableA.numRows < 300, 'filter should keep a strict subset');
        assertEquivalent(tableA, tableB);
    });

    it('filterBox under a pending translate', async () => {
        const { tableA, tableB } = await runAB(300, {}, [
            { kind: 'translate', value: new Vec3(2, 0, 0) },
            { kind: 'filterBox', min: new Vec3(-3, -1, -3), max: new Vec3(3, 1, 3) }
        ]);
        assert.ok(tableA.numRows > 0 && tableA.numRows < 300, 'filter should keep a strict subset');
        assertEquivalent(tableA, tableB);
    });

    it('filterSphere under a pending translate', async () => {
        const { tableA, tableB } = await runAB(300, {}, [
            { kind: 'translate', value: new Vec3(0, 0, 2) },
            { kind: 'filterSphere', center: new Vec3(0, 0, 0), radius: 4 }
        ]);
        assert.ok(tableA.numRows > 0 && tableA.numRows < 300, 'filter should keep a strict subset');
        assertEquivalent(tableA, tableB);
    });

    it('transform + chained filters + summary', async () => {
        const inject = (dt) => {
            dt.getColumnByName('x')[7] = NaN;
            dt.getColumnByName('y')[42] = NaN;
        };
        const { tableA, tableB } = await runAB(300, { includeSH: true, shBands: 1 }, [
            { kind: 'scale', value: 0.5 },
            { kind: 'filterNaN' },
            { kind: 'filterByValue', columnName: 'f_dc_0', comparator: 'gt', value: 0.5 },
            { kind: 'summary' }
        ], inject);
        assertEquivalent(tableA, tableB);
    });

    it('summary alone leaves the scene unchanged', async () => {
        const { tableA, tableB } = await runAB(120, {}, [{ kind: 'summary' }]);
        assert.strictEqual(tableA.numRows, 120);
        assertEquivalent(tableA, tableB);
    });
});
