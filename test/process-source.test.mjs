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
import { processDataTable, Transform } from '../src/lib/index.js';
import { convertToSpace } from '../src/lib/data-table/index.js';
import { dataTableToChunkSource, materializeToDataTable } from '../src/lib/compat/data-table.js';
import { permuteSource, reduceBandsSource } from '../src/lib/ops/index.js';
import { processSource, processSourceBridged } from '../src/lib/process-source.js';
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

    it('transform + chained filters + stats', async () => {
        const inject = (dt) => {
            dt.getColumnByName('x')[7] = NaN;
            dt.getColumnByName('y')[42] = NaN;
        };
        const { tableA, tableB } = await runAB(300, { includeSH: true, shBands: 1 }, [
            { kind: 'scale', value: 0.5 },
            { kind: 'filterNaN' },
            { kind: 'filterByValue', columnName: 'f_dc_0', comparator: 'gt', value: 0.5 },
            { kind: 'stats' }
        ], inject);
        assertEquivalent(tableA, tableB);
    });

    it('stats alone leaves the scene unchanged', async () => {
        const { tableA, tableB } = await runAB(120, {}, [{ kind: 'stats' }]);
        assert.strictEqual(tableA.numRows, 120);
        assertEquivalent(tableA, tableB);
    });
});

// processSourceBridged applies chunk-native runs via processSource and DataTable-
// only runs (mortonOrder here) via a materialize -> processDataTable -> re-bridge
// island. It must equal the pure DataTable path, including ordering and the
// pending transform carried across the island.
const runBridged = async (count, opts, actions) => {
    const chunkSize = 64;
    const make = () => createTestDataTable(count, opts);
    const pool = createChunkDataPool({ chunkSize });
    const source = dataTableToChunkSource(make(), chunkSize);
    const tableA = await processDataTable(make(), actions);
    const tableB = await materializeToDataTable(await processSourceBridged(source, actions, pool), pool);
    return { tableA, tableB };
};

describe('processSourceBridged (chunk-native runs + DataTable islands)', () => {
    it('bridges a DataTable-only op between chunk-native transforms ([translate, mortonOrder, scale])', async () => {
        const { tableA, tableB } = await runBridged(300, { includeSH: true, shBands: 1 }, [
            { kind: 'translate', value: new Vec3(1, 2, 3) },
            { kind: 'mortonOrder' },
            { kind: 'scale', value: 0.5 }
        ]);
        assertEquivalent(tableA, tableB);
    });
});

describe('reduceBandsSource A/B vs processDataTable filterBands', () => {
    it('band 3 -> 1 sequential read matches the DataTable band drop', async () => {
        const { tableA, tableB } = await runAB(300, { includeSH: true, shBands: 3 }, [
            { kind: 'filterBands', value: 1 }
        ]);
        assert.ok(tableA.hasColumn('f_rest_8') && !tableA.hasColumn('f_rest_9'), 'dropped to band 1 (9 rest coeffs)');
        assertEquivalent(tableA, tableB);
    });

    it('gather (permuteSource) over a band-reduced source matches the DataTable drop', async () => {
        const chunkSize = 64;
        const pool = createChunkDataPool({ chunkSize });
        const order = new Uint32Array([250, 0, 128, 63, 299, 17]); // shuffled, chunk-straddling

        const oracle = await processDataTable(
            createTestDataTable(300, { includeSH: true, shBands: 3 }),
            [{ kind: 'filterBands', value: 1 }]
        );
        const reduced = reduceBandsSource(
            dataTableToChunkSource(createTestDataTable(300, { includeSH: true, shBands: 3 }), chunkSize),
            1,
            pool
        );
        const gathered = await materializeToDataTable(permuteSource(reduced, order), pool);

        assert.strictEqual(gathered.numRows, order.length);
        assert.deepStrictEqual([...gathered.columnNames].sort(), [...oracle.columnNames].sort());
        for (const name of oracle.columnNames) {
            const e = oracle.getColumnByName(name).data;
            const g = gathered.getColumnByName(name).data;
            for (let j = 0; j < order.length; j++) {
                assert.ok(approxEqual(g[j], e[order[j]]), `column '${name}' out-row ${j}`);
            }
        }
    });
});
