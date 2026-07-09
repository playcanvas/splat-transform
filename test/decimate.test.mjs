/**
 * Decimate action tests (processDataTable bridge over the chunk-native
 * decimator) + DataTable.clone coverage that historically lived here.
 *
 * Behavior changes vs the legacy simplifyGaussians (deliberate):
 * - missing geometric/color columns throw instead of silently falling back
 *   to visibility pruning (sortByVisibility is gone);
 * - transforms are baked at decimation rather than kept pending (world-space
 *   result is identical — decimation is TRS-covariant).
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

import { Vec3 } from 'playcanvas';

import { assertClose } from './helpers/summary-compare.mjs';
import { createMinimalTestData } from './helpers/test-utils.mjs';
import { Column, DataTable, processDataTable } from '../src/lib/index.js';

function createGaussianTestData(options = {}) {
    const count = options.count ?? 4;
    const defaults = {
        x: new Float32Array(count).fill(0),
        y: new Float32Array(count).fill(0),
        z: new Float32Array(count).fill(0),
        opacity: new Float32Array(count).fill(0),
        scale_0: new Float32Array(count).fill(0),
        scale_1: new Float32Array(count).fill(0),
        scale_2: new Float32Array(count).fill(0),
        rot_0: new Float32Array(count).fill(0),
        rot_1: new Float32Array(count).fill(0),
        rot_2: new Float32Array(count).fill(0),
        rot_3: new Float32Array(count).fill(1),
        f_dc_0: new Float32Array(count).fill(0),
        f_dc_1: new Float32Array(count).fill(0),
        f_dc_2: new Float32Array(count).fill(0)
    };

    const data = { ...defaults, ...options };

    return new DataTable([
        new Column('x', data.x),
        new Column('y', data.y),
        new Column('z', data.z),
        new Column('opacity', data.opacity),
        new Column('scale_0', data.scale_0),
        new Column('scale_1', data.scale_1),
        new Column('scale_2', data.scale_2),
        new Column('rot_0', data.rot_0),
        new Column('rot_1', data.rot_1),
        new Column('rot_2', data.rot_2),
        new Column('rot_3', data.rot_3),
        new Column('f_dc_0', data.f_dc_0),
        new Column('f_dc_1', data.f_dc_1),
        new Column('f_dc_2', data.f_dc_2)
    ]);
}

const decimate = (count, percent = null) => [{ kind: 'decimate', count, percent }];

describe('decimate - Count Mode', () => {
    it('should produce exactly N splats in count mode', async () => {
        const testData = createMinimalTestData();
        const originalRows = testData.numRows;
        const result = await processDataTable(testData, decimate(5));
        assert.strictEqual(result.numRows, 5, 'Should have exactly 5 rows');
        assert(result.numRows < originalRows, 'Should have fewer rows than original');
    });

    it('should keep all splats when count exceeds numRows', async () => {
        const testData = createMinimalTestData();
        const originalRows = testData.numRows;
        const result = await processDataTable(testData, decimate(1000));
        assert.strictEqual(result.numRows, originalRows, 'Should keep all rows when count > numRows');
    });

    it('should handle count of 0', async () => {
        const testData = createMinimalTestData();
        const result = await processDataTable(testData, decimate(0));
        assert.strictEqual(result.numRows, 0, 'Should have 0 rows when count is 0');
    });

    it('should not invoke createDevice on the early-return paths', async () => {
        const testData = createMinimalTestData();
        let invoked = false;
        const createDevice = async () => {
            invoked = true;
            throw new Error('should not be called');
        };
        await processDataTable(testData, decimate(1000), { createDevice });
        assert.strictEqual(invoked, false, 'createDevice should be lazy when target >= numRows');
        await processDataTable(testData, decimate(0), { createDevice });
        assert.strictEqual(invoked, false, 'createDevice should be lazy on the empty target too');
    });

    it('should produce merged splats within original bounds', async () => {
        const testData = createGaussianTestData({
            count: 4,
            x: new Float32Array([0, 1, 2, 3]),
            opacity: new Float32Array([0, 2.197, -2.197, 0]),
            scale_0: new Float32Array([0, 0, 0, Math.log(2)]),
            scale_1: new Float32Array([0, 0, 0, Math.log(2)]),
            scale_2: new Float32Array([0, 0, 0, Math.log(2)])
        });
        const result = await processDataTable(testData, decimate(2));
        assert.strictEqual(result.numRows, 2, 'Should have exactly 2 rows');
        for (const x of result.getColumnByName('x').data) {
            assert(x >= 0 && x <= 3, `merged x=${x} should be within original bounds [0, 3]`);
        }
    });
});

describe('decimate - Percent Mode', () => {
    it('should keep approximately X% of splats', async () => {
        const result = await processDataTable(createMinimalTestData(), decimate(null, 50));
        assert.strictEqual(result.numRows, 8, 'Should have 50% of rows (8)');
    });

    it('should keep all splats at 100%', async () => {
        const testData = createMinimalTestData();
        const originalRows = testData.numRows;
        const result = await processDataTable(testData, decimate(null, 100));
        assert.strictEqual(result.numRows, originalRows, 'Should keep all rows at 100%');
    });

    it('should remove all splats at 0%', async () => {
        const result = await processDataTable(createMinimalTestData(), decimate(null, 0));
        assert.strictEqual(result.numRows, 0, 'Should have 0 rows at 0%');
    });

    it('should handle 25%', async () => {
        const result = await processDataTable(createMinimalTestData(), decimate(null, 25));
        assert.strictEqual(result.numRows, 4, 'Should have 25% of rows (4)');
    });
});

describe('decimate - merge quality invariants', () => {
    it('should produce valid opacity, finite scales and unit quaternions', async () => {
        const result = await processDataTable(createMinimalTestData(), decimate(8));
        assert.strictEqual(result.numRows, 8);

        const op = result.getColumnByName('opacity').data;
        const r0 = result.getColumnByName('rot_0').data;
        const r1 = result.getColumnByName('rot_1').data;
        const r2 = result.getColumnByName('rot_2').data;
        const r3 = result.getColumnByName('rot_3').data;
        for (let i = 0; i < result.numRows; i++) {
            const linear = 1 / (1 + Math.exp(-op[i]));
            assert(linear > 0 && linear <= 1, `opacity[${i}] sigmoid=${linear} should be in (0, 1]`);
            const len = Math.hypot(r0[i], r1[i], r2[i], r3[i]);
            assertClose(len, 1.0, 0.01, `quaternion at row ${i} should be normalized`);
            for (const col of ['scale_0', 'scale_1', 'scale_2']) {
                assert(isFinite(result.getColumnByName(col).data[i]), `${col}[${i}] finite`);
            }
        }
    });

    it('should mass-conserve opacity and area-weight color when merging two equal overlapping splats', async () => {
        // Two unit-sphere splats co-located at the origin, both with logit
        // opacity 0 (sigmoid → 0.5), identity rotation, opposite DC colors.
        // Mass conservation gives α_m = (0.5+0.5)·A / A = 1.0 (capped at 1);
        // color is the area·α-weighted average → (0.5, 0.5, 0). Locks in the
        // spark-derived merge behavior (not Porter-Duff's 0.75).
        const testData = new DataTable([
            new Column('x', new Float32Array([0, 0])),
            new Column('y', new Float32Array([0, 0])),
            new Column('z', new Float32Array([0, 0])),
            new Column('opacity', new Float32Array([0, 0])),
            new Column('scale_0', new Float32Array([0, 0])),
            new Column('scale_1', new Float32Array([0, 0])),
            new Column('scale_2', new Float32Array([0, 0])),
            new Column('rot_0', new Float32Array([1, 1])),
            new Column('rot_1', new Float32Array([0, 0])),
            new Column('rot_2', new Float32Array([0, 0])),
            new Column('rot_3', new Float32Array([0, 0])),
            new Column('f_dc_0', new Float32Array([1, 0])),
            new Column('f_dc_1', new Float32Array([0, 1])),
            new Column('f_dc_2', new Float32Array([0, 0]))
        ]);

        const result = await processDataTable(testData, decimate(1));
        assert.strictEqual(result.numRows, 1, 'Should merge to a single splat');
        assertClose(result.getColumnByName('x').data[0], 0, 1e-5, 'merged x');
        const linearOp = 1 / (1 + Math.exp(-result.getColumnByName('opacity').data[0]));
        assert(linearOp >= 0.99, `merged opacity sigmoid=${linearOp} should be ≈ 1 (mass-conserving)`);
        assertClose(result.getColumnByName('f_dc_0').data[0], 0.5, 1e-5, 'merged f_dc_0 (red avg)');
        assertClose(result.getColumnByName('f_dc_1').data[0], 0.5, 1e-5, 'merged f_dc_1 (green avg)');
        assertClose(result.getColumnByName('f_dc_2').data[0], 0, 1e-5, 'merged f_dc_2');
    });

    it('should fail loud (throw) when every edge cost is non-finite', async () => {
        const inf = Infinity;
        const testData = createGaussianTestData({
            count: 4,
            x: new Float32Array([0, 1, 2, 3]),
            f_dc_0: new Float32Array([inf, inf, inf, inf])
        });
        await assert.rejects(
            () => processDataTable(testData, decimate(2)),
            /no valid merges/,
            'should throw when every edge cost is non-finite'
        );
    });

    it('should fail loud (throw) when the scene is too degenerate to decimate', async () => {
        // Every splat coincident at the origin: identical queries tie-break to
        // the same KNN hub set, so candidate lists collapse and the matching
        // starves — same pathology and same fail-loud stall guard as legacy.
        const testData = createGaussianTestData({ count: 600 });
        await assert.rejects(
            () => processDataTable(testData, decimate(300)),
            /too degenerate to merge further/,
            'should throw when coincident splats starve the matching'
        );
    });

    it('should throw when gaussian columns are missing (legacy silently pruned)', async () => {
        const testData = new DataTable([
            new Column('x', new Float32Array([0, 1, 2, 3])),
            new Column('y', new Float32Array([0, 0, 0, 0])),
            new Column('z', new Float32Array([0, 0, 0, 0])),
            new Column('opacity', new Float32Array([0, 2.197, -2.197, 0])),
            new Column('scale_0', new Float32Array([0, 0, 0, Math.log(2)])),
            new Column('scale_1', new Float32Array([0, 0, 0, Math.log(2)])),
            new Column('scale_2', new Float32Array([0, 0, 0, Math.log(2)]))
        ]);
        await assert.rejects(
            () => processDataTable(testData, decimate(2)),
            /gaussian splat data|missing/i,
            'should throw on non-gaussian input'
        );
    });
});

describe('clone with row selection', () => {
    it('should create smaller DataTable when indices.length < numRows', async () => {
        const testData = new DataTable([
            new Column('a', new Float32Array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])),
            new Column('b', new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        ]);

        const indices = new Uint32Array([0, 2, 4, 6, 8]);
        const result = testData.clone({ rows: indices });

        assert.strictEqual(result.numRows, 5, 'Should have exactly 5 rows');
        assert.deepStrictEqual(
            Array.from(result.getColumnByName('a').data),
            [10, 30, 50, 70, 90],
            'Should have correct values from selected indices'
        );
        assert.deepStrictEqual(
            Array.from(result.getColumnByName('b').data),
            [1, 3, 5, 7, 9],
            'Should have correct values for all columns'
        );
    });

    it('should handle selecting just one row', async () => {
        const testData = new DataTable([
            new Column('a', new Float32Array([10, 20, 30, 40, 50]))
        ]);

        const indices = new Uint32Array([2]);
        const result = testData.clone({ rows: indices });

        assert.strictEqual(result.numRows, 1, 'Should have exactly 1 row');
        assert.strictEqual(result.getColumnByName('a').data[0], 30, 'Should have value from index 2');
    });

    it('should handle empty indices', async () => {
        const testData = new DataTable([
            new Column('a', new Float32Array([10, 20, 30]))
        ]);

        const indices = new Uint32Array(0);
        const result = testData.clone({ rows: indices });

        assert.strictEqual(result.numRows, 0, 'Should have 0 rows');
    });
});

describe('clone with column selection', () => {
    it('should return only the requested columns', async () => {
        const testData = new DataTable([
            new Column('x', new Float32Array([1, 2, 3])),
            new Column('y', new Float32Array([4, 5, 6])),
            new Column('z', new Float32Array([7, 8, 9]))
        ]);

        const result = testData.clone({ columns: ['x', 'z'] });

        assert.strictEqual(result.numColumns, 2, 'Should have 2 columns');
        assert.deepStrictEqual(result.columnNames, ['x', 'z'], 'Should preserve column order');
        assert.deepStrictEqual(Array.from(result.getColumnByName('x').data), [1, 2, 3]);
        assert.deepStrictEqual(Array.from(result.getColumnByName('z').data), [7, 8, 9]);
        assert.strictEqual(result.getColumnByName('y'), undefined, 'Should not include y');
    });

    it('should preserve typed array types', async () => {
        const testData = new DataTable([
            new Column('a', new Uint8Array([1, 2, 3])),
            new Column('b', new Int32Array([4, 5, 6])),
            new Column('c', new Float64Array([7, 8, 9]))
        ]);

        const result = testData.clone({ columns: ['a', 'c'] });

        assert(result.getColumnByName('a').data instanceof Uint8Array, 'Should preserve Uint8Array');
        assert(result.getColumnByName('c').data instanceof Float64Array, 'Should preserve Float64Array');
    });

    it('should produce an independent copy', async () => {
        const testData = new DataTable([
            new Column('x', new Float32Array([1, 2, 3])),
            new Column('y', new Float32Array([4, 5, 6]))
        ]);

        const result = testData.clone({ columns: ['x'] });
        result.getColumnByName('x').data[0] = 999;

        assert.strictEqual(testData.getColumnByName('x').data[0], 1, 'Source should be unmodified');
    });

    it('should throw on unknown column names', async () => {
        const testData = new DataTable([
            new Column('x', new Float32Array([1, 2, 3]))
        ]);

        assert.throws(
            () => testData.clone({ columns: ['x', 'missing'] }),
            /unknown column name\(s\): missing/
        );
    });

    it('should throw on empty columns array', async () => {
        const testData = new DataTable([
            new Column('x', new Float32Array([1, 2, 3]))
        ]);

        assert.throws(
            () => testData.clone({ columns: [] }),
            /must contain at least one column name/
        );
    });
});

describe('clone with rows and columns combined', () => {
    it('should select specific rows and columns', async () => {
        const testData = new DataTable([
            new Column('x', new Float32Array([10, 20, 30, 40])),
            new Column('y', new Float32Array([1, 2, 3, 4])),
            new Column('z', new Float32Array([100, 200, 300, 400]))
        ]);

        const result = testData.clone({ rows: [1, 3], columns: ['x', 'z'] });

        assert.strictEqual(result.numRows, 2, 'Should have 2 rows');
        assert.strictEqual(result.numColumns, 2, 'Should have 2 columns');
        assert.deepStrictEqual(Array.from(result.getColumnByName('x').data), [20, 40]);
        assert.deepStrictEqual(Array.from(result.getColumnByName('z').data), [200, 400]);
    });

    it('should handle rows reordering with column filter', async () => {
        const testData = new DataTable([
            new Column('a', new Float32Array([10, 20, 30])),
            new Column('b', new Float32Array([1, 2, 3]))
        ]);

        const result = testData.clone({ rows: [2, 0, 1], columns: ['a'] });

        assert.strictEqual(result.numColumns, 1, 'Should have 1 column');
        assert.deepStrictEqual(Array.from(result.getColumnByName('a').data), [30, 10, 20]);
    });
});

describe('decimate Integration', () => {
    it('should chain with other transforms (bake-at-decimate semantics)', async () => {
        const testData = createMinimalTestData();
        const origX = Array.from(testData.getColumnByName('x').data);

        const result = await processDataTable(testData, [
            { kind: 'translate', value: new Vec3(10, 0, 0) },
            { kind: 'decimate', count: 8, percent: null },
            { kind: 'scale', value: 2.0 }
        ]);

        assert.strictEqual(result.numRows, 8, 'Should have 8 rows after decimation');

        // Decimation bakes the pending translate into the values (TRS-covariant,
        // so the world-space result is unchanged); only the post-decimate scale
        // remains pending.
        assertClose(result.transform.translation.x, 0, 1e-5, 'translate baked at decimation');
        assertClose(result.transform.scale, 2.0, 1e-5, 'scale still pending');
        const minOrig = Math.min(...origX) + 10;
        const maxOrig = Math.max(...origX) + 10;
        for (const x of result.getColumnByName('x').data) {
            assert(x >= minOrig - 0.01 && x <= maxOrig + 0.01, `baked x=${x} within translated bounds`);
        }
    });

    it('should preserve all columns after merging', async () => {
        const testData = createGaussianTestData({
            count: 4,
            x: new Float32Array([100, 200, 300, 400]),
            y: new Float32Array([1, 2, 3, 4]),
            z: new Float32Array([10, 20, 30, 40]),
            opacity: new Float32Array([0, 2.197, -2.197, 0]),
            scale_0: new Float32Array([0, 0, 0, Math.log(2)]),
            scale_1: new Float32Array([0, 0, 0, Math.log(2)]),
            scale_2: new Float32Array([0, 0, 0, Math.log(2)])
        });

        const result = await processDataTable(testData, decimate(2));
        assert.strictEqual(result.numRows, 2, 'Should have 2 rows');
        for (const col of ['x', 'y', 'z', 'opacity', 'scale_0', 'rot_0', 'f_dc_0']) {
            assert(result.hasColumn(col), `Should have ${col} column`);
        }
    });

    it('should work with Morton ordering after filtering', async () => {
        const result = await processDataTable(createMinimalTestData(), [
            { kind: 'decimate', count: 8, percent: null },
            { kind: 'mortonOrder' }
        ]);
        assert.strictEqual(result.numRows, 8, 'Should have 8 rows');
        for (const col of ['x', 'y', 'z']) assert(result.hasColumn(col));
    });

    it('should produce finite values in all columns', async () => {
        const result = await processDataTable(createMinimalTestData().clone(), decimate(8));
        for (const col of result.columns) {
            for (let i = 0; i < result.numRows; i++) {
                assert(isFinite(col.data[i]), `${col.name}[${i}]=${col.data[i]} should be finite`);
            }
        }
    });
});
