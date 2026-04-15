/**
 * Decimate tests for splat-transform.
 * Tests sortByVisibility function, simplifyGaussians function,
 * and decimate action (NanoGS progressive pairwise merging).
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';

import {
    Column,
    DataTable,
    processDataTable,
    sortByVisibility,
    simplifyGaussians
} from '../src/lib/index.js';

import { createMinimalTestData } from './helpers/test-utils.mjs';
import { assertClose } from './helpers/summary-compare.mjs';

import { Vec3 } from 'playcanvas';

/**
 * Creates a minimal valid DataTable with required columns for visibility testing.
 * @param {object} options - Column data overrides
 * @returns {DataTable}
 */
function createVisibilityTestData(options = {}) {
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

describe('sortByVisibility', () => {
    it('should sort indices by visibility score (descending)', async () => {
        const testData = createVisibilityTestData({
            count: 4,
            x: new Float32Array([0, 1, 2, 3]),
            opacity: new Float32Array([0, 2.197, -2.197, 0]),
            scale_0: new Float32Array([0, 0, 0, Math.log(2)]),
            scale_1: new Float32Array([0, 0, 0, Math.log(2)]),
            scale_2: new Float32Array([0, 0, 0, Math.log(2)])
        });

        const indices = new Uint32Array([0, 1, 2, 3]);
        sortByVisibility(testData, indices);

        assert.strictEqual(indices[0], 3, 'Highest visibility should be first');
        assert.strictEqual(indices[1], 1, 'Second highest should be second');
        assert.strictEqual(indices[2], 0, 'Third highest should be third');
        assert.strictEqual(indices[3], 2, 'Lowest visibility should be last');
    });

    it('should handle empty indices', async () => {
        const testData = createVisibilityTestData({ count: 4 });
        const indices = new Uint32Array(0);

        sortByVisibility(testData, indices);

        assert.strictEqual(indices.length, 0, 'Empty indices should remain empty');
    });

    it('should handle missing columns gracefully', async () => {
        const testData = new DataTable([
            new Column('x', new Float32Array([0, 1, 2])),
            new Column('y', new Float32Array([0, 0, 0])),
            new Column('z', new Float32Array([0, 0, 0]))
        ]);

        const indices = new Uint32Array([0, 1, 2]);
        const originalIndices = indices.slice();

        sortByVisibility(testData, indices);

        assert.deepStrictEqual(Array.from(indices), Array.from(originalIndices),
            'Indices should be unchanged when columns are missing');
    });

    it('should handle single element', async () => {
        const testData = createVisibilityTestData({
            count: 1,
            x: new Float32Array([5]),
            opacity: new Float32Array([0]),
            scale_0: new Float32Array([0]),
            scale_1: new Float32Array([0]),
            scale_2: new Float32Array([0])
        });

        const indices = new Uint32Array([0]);
        sortByVisibility(testData, indices);

        assert.strictEqual(indices[0], 0, 'Single index should remain 0');
    });
});

describe('simplifyGaussians', () => {
    it('should return all splats when targetCount >= numRows', async () => {
        const testData = createMinimalTestData();
        const result = simplifyGaussians(testData, 1000);
        assert.strictEqual(result.numRows, testData.numRows, 'Should keep all rows');
    });

    it('should return empty DataTable when targetCount is 0', async () => {
        const testData = createMinimalTestData();
        const result = simplifyGaussians(testData, 0);
        assert.strictEqual(result.numRows, 0, 'Should have 0 rows');
    });

    it('should reduce to target count', async () => {
        const testData = createMinimalTestData();
        const result = simplifyGaussians(testData, 8);
        assert.strictEqual(result.numRows, 8, 'Should have exactly 8 rows');
    });

    it('should preserve all columns', async () => {
        const testData = createMinimalTestData();
        const originalCols = testData.columnNames.sort();
        const result = simplifyGaussians(testData, 8);
        const resultCols = result.columnNames.sort();
        assert.deepStrictEqual(resultCols, originalCols, 'Should have same columns');
    });

    it('should produce merged positions within the bounding box of originals', async () => {
        const testData = createMinimalTestData();
        const origX = testData.getColumnByName('x').data;
        const origZ = testData.getColumnByName('z').data;

        const minX = Math.min(...origX);
        const maxX = Math.max(...origX);
        const minZ = Math.min(...origZ);
        const maxZ = Math.max(...origZ);

        const result = simplifyGaussians(testData, 4);

        const resX = result.getColumnByName('x').data;
        const resZ = result.getColumnByName('z').data;
        for (let i = 0; i < result.numRows; i++) {
            assert(resX[i] >= minX - 0.01 && resX[i] <= maxX + 0.01,
                `merged x[${i}]=${resX[i]} should be within original bounds [${minX}, ${maxX}]`);
            assert(resZ[i] >= minZ - 0.01 && resZ[i] <= maxZ + 0.01,
                `merged z[${i}]=${resZ[i]} should be within original bounds [${minZ}, ${maxZ}]`);
        }
    });

    it('should produce valid opacity values', async () => {
        const testData = createMinimalTestData();
        const result = simplifyGaussians(testData, 8);

        const opacityData = result.getColumnByName('opacity').data;
        for (let i = 0; i < result.numRows; i++) {
            const linearOpacity = 1 / (1 + Math.exp(-opacityData[i]));
            assert(linearOpacity > 0 && linearOpacity <= 1,
                `opacity[${i}] sigmoid=${linearOpacity} should be in (0, 1]`);
        }
    });

    it('should produce finite scale values', async () => {
        const testData = createMinimalTestData();
        const result = simplifyGaussians(testData, 8);

        for (const col of ['scale_0', 'scale_1', 'scale_2']) {
            const data = result.getColumnByName(col).data;
            for (let i = 0; i < result.numRows; i++) {
                assert(isFinite(data[i]), `${col}[${i}]=${data[i]} should be finite`);
            }
        }
    });

    it('should produce normalized quaternion rotations', async () => {
        const testData = createMinimalTestData();
        const result = simplifyGaussians(testData, 8);

        const r0 = result.getColumnByName('rot_0').data;
        const r1 = result.getColumnByName('rot_1').data;
        const r2 = result.getColumnByName('rot_2').data;
        const r3 = result.getColumnByName('rot_3').data;

        for (let i = 0; i < result.numRows; i++) {
            const len = Math.sqrt(r0[i] * r0[i] + r1[i] * r1[i] + r2[i] * r2[i] + r3[i] * r3[i]);
            assertClose(len, 1.0, 0.01, `quaternion at row ${i} should be normalized`);
        }
    });

    it('should fall back to visibility pruning when rotation columns are missing', async () => {
        const testData = new DataTable([
            new Column('x', new Float32Array([0, 1, 2, 3])),
            new Column('y', new Float32Array([0, 0, 0, 0])),
            new Column('z', new Float32Array([0, 0, 0, 0])),
            new Column('opacity', new Float32Array([0, 2.197, -2.197, 0])),
            new Column('scale_0', new Float32Array([0, 0, 0, Math.log(2)])),
            new Column('scale_1', new Float32Array([0, 0, 0, Math.log(2)])),
            new Column('scale_2', new Float32Array([0, 0, 0, Math.log(2)]))
        ]);

        const result = simplifyGaussians(testData, 2);
        assert.strictEqual(result.numRows, 2, 'Should produce 2 rows via fallback');
    });
});

describe('decimate - Count Mode', () => {
    it('should produce exactly N splats in count mode', async () => {
        const testData = createMinimalTestData();
        const originalRows = testData.numRows;

        const result = await processDataTable(testData, [{
            kind: 'decimate',
            count: 5,
            percent: null
        }]);

        assert.strictEqual(result.numRows, 5, 'Should have exactly 5 rows');
        assert(result.numRows < originalRows, 'Should have fewer rows than original');
    });

    it('should keep all splats when count exceeds numRows', async () => {
        const testData = createMinimalTestData();
        const originalRows = testData.numRows;

        const result = await processDataTable(testData, [{
            kind: 'decimate',
            count: 1000,
            percent: null
        }]);

        assert.strictEqual(result.numRows, originalRows, 'Should keep all rows when count > numRows');
    });

    it('should handle count of 0', async () => {
        const testData = createMinimalTestData();

        const result = await processDataTable(testData, [{
            kind: 'decimate',
            count: 0,
            percent: null
        }]);

        assert.strictEqual(result.numRows, 0, 'Should have 0 rows when count is 0');
    });

    it('should produce merged splats with reasonable positions', async () => {
        const testData = createVisibilityTestData({
            count: 4,
            x: new Float32Array([0, 1, 2, 3]),
            opacity: new Float32Array([0, 2.197, -2.197, 0]),
            scale_0: new Float32Array([0, 0, 0, Math.log(2)]),
            scale_1: new Float32Array([0, 0, 0, Math.log(2)]),
            scale_2: new Float32Array([0, 0, 0, Math.log(2)])
        });

        const result = await processDataTable(testData, [{
            kind: 'decimate',
            count: 2,
            percent: null
        }]);

        assert.strictEqual(result.numRows, 2, 'Should have exactly 2 rows');

        const xValues = Array.from(result.getColumnByName('x').data);
        for (const x of xValues) {
            assert(x >= 0 && x <= 3,
                `merged x=${x} should be within original bounds [0, 3]`);
        }
    });
});

describe('decimate - Percent Mode', () => {
    it('should keep approximately X% of splats', async () => {
        const testData = createMinimalTestData();

        const result = await processDataTable(testData, [{
            kind: 'decimate',
            count: null,
            percent: 50
        }]);

        assert.strictEqual(result.numRows, 8, 'Should have 50% of rows (8)');
    });

    it('should keep all splats at 100%', async () => {
        const testData = createMinimalTestData();
        const originalRows = testData.numRows;

        const result = await processDataTable(testData, [{
            kind: 'decimate',
            count: null,
            percent: 100
        }]);

        assert.strictEqual(result.numRows, originalRows, 'Should keep all rows at 100%');
    });

    it('should remove all splats at 0%', async () => {
        const testData = createMinimalTestData();

        const result = await processDataTable(testData, [{
            kind: 'decimate',
            count: null,
            percent: 0
        }]);

        assert.strictEqual(result.numRows, 0, 'Should have 0 rows at 0%');
    });

    it('should handle 25%', async () => {
        const testData = createMinimalTestData();

        const result = await processDataTable(testData, [{
            kind: 'decimate',
            count: null,
            percent: 25
        }]);

        assert.strictEqual(result.numRows, 4, 'Should have 25% of rows (4)');
    });
});

describe('Visibility Score Calculation', () => {
    it('should correctly compute visibility from logit opacity and log scales', async () => {
        const testData = createVisibilityTestData({
            count: 2,
            x: new Float32Array([0, 1]),
            opacity: new Float32Array([0, 2.197]),
            scale_0: new Float32Array([0, 0]),
            scale_1: new Float32Array([0, 0]),
            scale_2: new Float32Array([0, 0])
        });

        const indices = new Uint32Array([0, 1]);
        sortByVisibility(testData, indices);

        assert.strictEqual(indices[0], 1, 'Higher opacity splat should be first');
        assert.strictEqual(indices[1], 0, 'Lower opacity splat should be second');
    });

    it('should correctly incorporate scale into visibility', async () => {
        const testData = createVisibilityTestData({
            count: 2,
            x: new Float32Array([0, 1]),
            opacity: new Float32Array([0, 0]),
            scale_0: new Float32Array([0, Math.log(10)]),
            scale_1: new Float32Array([0, 0]),
            scale_2: new Float32Array([0, 0])
        });

        const indices = new Uint32Array([0, 1]);
        sortByVisibility(testData, indices);

        assert.strictEqual(indices[0], 1, 'Larger scale splat should be first');
        assert.strictEqual(indices[1], 0, 'Smaller scale splat should be second');
    });

    it('should handle negative log scales (small splats)', async () => {
        const testData = createVisibilityTestData({
            count: 2,
            x: new Float32Array([0, 1]),
            opacity: new Float32Array([0, 0]),
            scale_0: new Float32Array([-2, 0]),
            scale_1: new Float32Array([-2, 0]),
            scale_2: new Float32Array([-2, 0])
        });

        const indices = new Uint32Array([0, 1]);
        sortByVisibility(testData, indices);

        assert.strictEqual(indices[0], 1, 'Normal scale splat should be first');
        assert.strictEqual(indices[1], 0, 'Small scale splat should be second');
    });

    it('should handle very low opacity', async () => {
        const testData = createVisibilityTestData({
            count: 2,
            x: new Float32Array([0, 1]),
            opacity: new Float32Array([-10, 0]),
            scale_0: new Float32Array([0, 0]),
            scale_1: new Float32Array([0, 0]),
            scale_2: new Float32Array([0, 0])
        });

        const indices = new Uint32Array([0, 1]);
        sortByVisibility(testData, indices);

        assert.strictEqual(indices[0], 1, 'Higher opacity splat should be first');
        assert.strictEqual(indices[1], 0, 'Very low opacity splat should be second');
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
    it('should chain with other transforms', async () => {
        const testData = createMinimalTestData();

        const result = await processDataTable(testData, [
            { kind: 'translate', value: new Vec3(10, 0, 0) },
            { kind: 'decimate', count: 8, percent: null },
            { kind: 'scale', value: 2.0 }
        ]);

        assert.strictEqual(result.numRows, 8, 'Should have 8 rows after filtering');

        // Raw data is unchanged by translate/scale (they compose into transform)
        // scale(2) from origin after translate(10) doubles the translation too
        assertClose(result.transform.translation.x, 20, 1e-5, 'tx');
        assertClose(result.transform.scale, 2.0, 1e-5, 'scale');
    });

    it('should preserve all columns after merging', async () => {
        const testData = createVisibilityTestData({
            count: 4,
            x: new Float32Array([100, 200, 300, 400]),
            y: new Float32Array([1, 2, 3, 4]),
            z: new Float32Array([10, 20, 30, 40]),
            opacity: new Float32Array([0, 2.197, -2.197, 0]),
            scale_0: new Float32Array([0, 0, 0, Math.log(2)]),
            scale_1: new Float32Array([0, 0, 0, Math.log(2)]),
            scale_2: new Float32Array([0, 0, 0, Math.log(2)])
        });

        const result = await processDataTable(testData, [{
            kind: 'decimate',
            count: 2,
            percent: null
        }]);

        assert.strictEqual(result.numRows, 2, 'Should have 2 rows');

        assert(result.hasColumn('x'), 'Should have x column');
        assert(result.hasColumn('y'), 'Should have y column');
        assert(result.hasColumn('z'), 'Should have z column');
        assert(result.hasColumn('opacity'), 'Should have opacity column');
        assert(result.hasColumn('scale_0'), 'Should have scale_0 column');
        assert(result.hasColumn('rot_0'), 'Should have rot_0 column');
        assert(result.hasColumn('f_dc_0'), 'Should have f_dc_0 column');
    });

    it('should work with Morton ordering after filtering', async () => {
        const testData = createMinimalTestData();

        const result = await processDataTable(testData, [
            { kind: 'decimate', count: 8, percent: null },
            { kind: 'mortonOrder' }
        ]);

        assert.strictEqual(result.numRows, 8, 'Should have 8 rows');

        assert(result.hasColumn('x'), 'Should have x column');
        assert(result.hasColumn('y'), 'Should have y column');
        assert(result.hasColumn('z'), 'Should have z column');
    });

    it('should produce finite values in all columns', async () => {
        const testData = createMinimalTestData();

        const result = await processDataTable(testData.clone(), [{
            kind: 'decimate',
            count: 8,
            percent: null
        }]);

        for (const col of result.columns) {
            for (let i = 0; i < result.numRows; i++) {
                assert(isFinite(col.data[i]),
                    `${col.name}[${i}]=${col.data[i]} should be finite`);
            }
        }
    });
});
