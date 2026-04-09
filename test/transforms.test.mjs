/**
 * Transformation tests for splat-transform.
 * Tests translate, rotate, scale, and filter operations.
 */

import { describe, it, before } from 'node:test';
import assert from 'node:assert';

import {
    computeSummary,
    processDataTable,
    Column,
    DataTable,
    Transform
} from '../src/lib/index.js';

import {
    transformColumns,
    computeWriteTransform
} from '../src/lib/data-table/transform.js';

import { createMinimalTestData } from './helpers/test-utils.mjs';
import { assertClose } from './helpers/summary-compare.mjs';

import { Mat4, Quat, Vec3 } from 'playcanvas';

describe('Translate Transform', () => {
    let testData;

    before(() => {
        testData = createMinimalTestData();
    });

    it('should compose translation into transform without modifying raw data', () => {
        const originalSummary = computeSummary(testData);
        const clonedData = testData.clone();

        const result = processDataTable(clonedData, [{
            kind: 'translate',
            value: new Vec3(10, 20, 30)
        }]);

        const newSummary = computeSummary(result);

        // Raw data should be unchanged
        assertClose(newSummary.columns.x.mean, originalSummary.columns.x.mean, 1e-5, 'raw x mean');
        assertClose(newSummary.columns.y.mean, originalSummary.columns.y.mean, 1e-5, 'raw y mean');
        assertClose(newSummary.columns.z.mean, originalSummary.columns.z.mean, 1e-5, 'raw z mean');
        assertClose(newSummary.columns.scale_0.mean, originalSummary.columns.scale_0.mean, 1e-5, 'scale_0');
        assertClose(newSummary.columns.opacity.mean, originalSummary.columns.opacity.mean, 1e-5, 'opacity');

        // transform should have the translation composed in
        const t = result.transform;
        assertClose(t.translation.x, 10, 1e-5, 'transform tx');
        assertClose(t.translation.y, 20, 1e-5, 'transform ty');
        assertClose(t.translation.z, 30, 1e-5, 'transform tz');

        // transformColumns should produce shifted engine-space positions
        const cols = transformColumns(result, ['x', 'y', 'z'], result.transform);
        const engineX = cols.get('x');
        const engineY = cols.get('y');
        const rawX = result.getColumnByName('x').data;
        const rawY = result.getColumnByName('y').data;
        for (let i = 0; i < result.numRows; i++) {
            assertClose(engineX[i], rawX[i] + 10, 1e-4, `engine x[${i}]`);
            assertClose(engineY[i], rawY[i] + 20, 1e-4, `engine y[${i}]`);
        }
    });

    it('should handle zero translation', () => {
        const originalSummary = computeSummary(testData);
        const clonedData = testData.clone();

        const result = processDataTable(clonedData, [{
            kind: 'translate',
            value: new Vec3(0, 0, 0)
        }]);

        const newSummary = computeSummary(result);

        // Positions should be unchanged
        assertClose(newSummary.columns.x.mean, originalSummary.columns.x.mean, 1e-5, 'x mean');
        assertClose(newSummary.columns.y.mean, originalSummary.columns.y.mean, 1e-5, 'y mean');
        assertClose(newSummary.columns.z.mean, originalSummary.columns.z.mean, 1e-5, 'z mean');
    });
});

describe('Scale Transform', () => {
    let testData;

    before(() => {
        testData = createMinimalTestData();
    });

    it('should compose scale into transform without modifying raw data', () => {
        const originalSummary = computeSummary(testData);
        const clonedData = testData.clone();

        const scaleFactor = 2.0;
        const result = processDataTable(clonedData, [{
            kind: 'scale',
            value: scaleFactor
        }]);

        const newSummary = computeSummary(result);

        // Raw data should be unchanged
        assertClose(newSummary.columns.x.min, originalSummary.columns.x.min, 1e-5, 'raw x.min');
        assertClose(newSummary.columns.x.max, originalSummary.columns.x.max, 1e-5, 'raw x.max');
        assertClose(newSummary.columns.scale_0.mean, originalSummary.columns.scale_0.mean, 1e-5, 'raw scale_0');

        // transform should have scale = 2
        assertClose(result.transform.scale, 2.0, 1e-5, 'transform scale');

        // transformColumns should produce scaled engine-space positions
        const cols = transformColumns(result, ['x', 'y', 'z', 'scale_0'], result.transform);
        const rawX = result.getColumnByName('x').data;
        for (let i = 0; i < result.numRows; i++) {
            assertClose(cols.get('x')[i], rawX[i] * scaleFactor, 1e-4, `engine x[${i}]`);
        }

        // Scale columns should be shifted by log(factor)
        const logFactor = Math.log(scaleFactor);
        const rawScale = result.getColumnByName('scale_0').data;
        for (let i = 0; i < result.numRows; i++) {
            assertClose(cols.get('scale_0')[i], rawScale[i] + logFactor, 1e-4, `engine scale_0[${i}]`);
        }
    });

    it('should handle scale factor of 1 (no change)', () => {
        const originalSummary = computeSummary(testData);
        const clonedData = testData.clone();

        const result = processDataTable(clonedData, [{
            kind: 'scale',
            value: 1.0
        }]);

        const newSummary = computeSummary(result);

        assertClose(newSummary.columns.x.mean, originalSummary.columns.x.mean, 1e-5, 'x mean');
        assertClose(newSummary.columns.scale_0.mean, originalSummary.columns.scale_0.mean, 1e-5, 'scale_0');
    });

    it('should handle fractional scale factor', () => {
        const clonedData = testData.clone();

        const scaleFactor = 0.5;
        const result = processDataTable(clonedData, [{
            kind: 'scale',
            value: scaleFactor
        }]);

        assertClose(result.transform.scale, 0.5, 1e-5, 'transform scale');
    });
});

describe('Rotate Transform', () => {
    let testData;

    before(() => {
        testData = createMinimalTestData();
    });

    it('should compose rotation into transform without modifying raw data', () => {
        const originalSummary = computeSummary(testData);
        const clonedData = testData.clone();

        // 90 degree rotation around Y
        const result = processDataTable(clonedData, [{
            kind: 'rotate',
            value: new Vec3(0, 90, 0)
        }]);

        const newSummary = computeSummary(result);

        // Raw data should be unchanged
        assertClose(newSummary.columns.x.min, originalSummary.columns.x.min, 1e-5, 'raw x.min');
        assertClose(newSummary.columns.x.max, originalSummary.columns.x.max, 1e-5, 'raw x.max');
        assert.strictEqual(newSummary.rowCount, originalSummary.rowCount);

        // transform should have a rotation
        assert.ok(!result.transform.isIdentity(), 'transform should not be identity');

        // transformColumns should produce rotated engine-space positions
        // After 90° Y rotation: x' = z, z' = -x
        const cols = transformColumns(result, ['x', 'y', 'z'], result.transform);
        const rawX = result.getColumnByName('x').data;
        const rawZ = result.getColumnByName('z').data;
        for (let i = 0; i < result.numRows; i++) {
            assertClose(cols.get('x')[i], rawZ[i], 1e-4, `engine x[${i}]`);
            assertClose(cols.get('z')[i], -rawX[i], 1e-4, `engine z[${i}]`);
        }
    });

    it('should handle zero rotation', () => {
        const originalSummary = computeSummary(testData);
        const clonedData = testData.clone();

        const result = processDataTable(clonedData, [{
            kind: 'rotate',
            value: new Vec3(0, 0, 0)
        }]);

        const newSummary = computeSummary(result);

        // Positions should be unchanged
        assertClose(newSummary.columns.x.mean, originalSummary.columns.x.mean, 1e-5, 'x mean');
        assertClose(newSummary.columns.y.mean, originalSummary.columns.y.mean, 1e-5, 'y mean');
        assertClose(newSummary.columns.z.mean, originalSummary.columns.z.mean, 1e-5, 'z mean');
    });
});

describe('Filter Box', () => {
    let testData;

    before(() => {
        testData = createMinimalTestData();
    });

    it('should filter out splats outside bounding box', () => {
        const clonedData = testData.clone();

        // Filter to only include splats with x >= 0
        const result = processDataTable(clonedData, [{
            kind: 'filterBox',
            min: new Vec3(0, -Infinity, -Infinity),
            max: new Vec3(Infinity, Infinity, Infinity)
        }]);

        // Should have fewer splats
        assert(result.numRows < testData.numRows, 'Should have fewer rows after filtering');
        assert(result.numRows > 0, 'Should have at least some rows');

        // All remaining x values should be >= 0
        const summary = computeSummary(result);
        assert(summary.columns.x.min >= 0, 'All x values should be >= 0');
    });

    it('should keep all splats when box contains everything', () => {
        const clonedData = testData.clone();

        const result = processDataTable(clonedData, [{
            kind: 'filterBox',
            min: new Vec3(-Infinity, -Infinity, -Infinity),
            max: new Vec3(Infinity, Infinity, Infinity)
        }]);

        assert.strictEqual(result.numRows, testData.numRows, 'Should keep all rows');
    });

    it('should return empty when box excludes everything', () => {
        const clonedData = testData.clone();

        const result = processDataTable(clonedData, [{
            kind: 'filterBox',
            min: new Vec3(1000, 1000, 1000),
            max: new Vec3(1001, 1001, 1001)
        }]);

        assert.strictEqual(result.numRows, 0, 'Should have no rows');
    });

    it('should use exact oriented box test with non-axis-aligned rotation', () => {
        const dt = new DataTable([
            new Column('x', new Float32Array([0, 1, 0, 0.9, -0.9])),
            new Column('y', new Float32Array([0, 0, 0, 0, 0])),
            new Column('z', new Float32Array([0, 0, 1, 0.9, 0.9]))
        ]);

        dt.transform = new Transform().fromEulers(0, 45, 0);

        // Engine-space box [-0.8, 0.8] on x and z.
        // Points 0,1,2 map inside; points 3,4 map outside (engine x=1.27 and z=1.27).
        // A conservative AABB approach would incorrectly include points 3 and 4.
        const result = processDataTable(dt, [{
            kind: 'filterBox',
            min: new Vec3(-0.8, -Infinity, -0.8),
            max: new Vec3(0.8, Infinity, 0.8)
        }]);

        assert.strictEqual(result.numRows, 3, 'Should keep exactly 3 points (exact OBB, not conservative AABB)');
    });
});

describe('Filter Sphere', () => {
    let testData;

    before(() => {
        testData = createMinimalTestData();
    });

    it('should filter out splats outside sphere', () => {
        const clonedData = testData.clone();

        // Filter to splats within radius 1 of origin
        const result = processDataTable(clonedData, [{
            kind: 'filterSphere',
            center: new Vec3(0, 0, 0),
            radius: 1.0
        }]);

        // Should have fewer splats
        assert(result.numRows < testData.numRows, 'Should have fewer rows after filtering');

        // All remaining splats should be within radius
        const xCol = result.getColumnByName('x').data;
        const yCol = result.getColumnByName('y').data;
        const zCol = result.getColumnByName('z').data;

        for (let i = 0; i < result.numRows; i++) {
            const distSq = xCol[i] * xCol[i] + yCol[i] * yCol[i] + zCol[i] * zCol[i];
            assert(distSq < 1.0, `Splat ${i} should be within radius`);
        }
    });

    it('should keep all splats when sphere contains everything', () => {
        const clonedData = testData.clone();

        const result = processDataTable(clonedData, [{
            kind: 'filterSphere',
            center: new Vec3(0, 0, 0),
            radius: 1000
        }]);

        assert.strictEqual(result.numRows, testData.numRows, 'Should keep all rows');
    });
});

describe('Filter By Value', () => {
    let testData;

    before(() => {
        testData = createMinimalTestData();
    });

    it('should filter by greater than comparison', () => {
        const clonedData = testData.clone();
        const originalSummary = computeSummary(testData);
        const threshold = originalSummary.columns.x.median;

        const result = processDataTable(clonedData, [{
            kind: 'filterByValue',
            columnName: 'x',
            comparator: 'gt',
            value: threshold
        }]);

        // Should have fewer rows
        assert(result.numRows < testData.numRows, 'Should have fewer rows');

        // All remaining x values should be > threshold
        const xCol = result.getColumnByName('x').data;
        for (let i = 0; i < result.numRows; i++) {
            assert(xCol[i] > threshold, `x[${i}] should be > ${threshold}`);
        }
    });

    it('should filter by less than or equal comparison', () => {
        const clonedData = testData.clone();
        const originalSummary = computeSummary(testData);
        const threshold = originalSummary.columns.x.median;

        const result = processDataTable(clonedData, [{
            kind: 'filterByValue',
            columnName: 'x',
            comparator: 'lte',
            value: threshold
        }]);

        // All remaining x values should be <= threshold
        const xCol = result.getColumnByName('x').data;
        for (let i = 0; i < result.numRows; i++) {
            assert(xCol[i] <= threshold, `x[${i}] should be <= ${threshold}`);
        }
    });

    it('should filter by equality', () => {
        const clonedData = testData.clone();

        // All y values are 0 in our test data
        const result = processDataTable(clonedData, [{
            kind: 'filterByValue',
            columnName: 'y',
            comparator: 'eq',
            value: 0
        }]);

        // Should keep all rows since all y=0
        assert.strictEqual(result.numRows, testData.numRows, 'Should keep all rows');
    });

    it('should filter by not equal', () => {
        const clonedData = testData.clone();

        // All y values are 0, so filtering neq 0 should give empty
        const result = processDataTable(clonedData, [{
            kind: 'filterByValue',
            columnName: 'y',
            comparator: 'neq',
            value: 0
        }]);

        assert.strictEqual(result.numRows, 0, 'Should have no rows');
    });

    it('should apply transform before filtering transform-sensitive columns', () => {
        const data = new DataTable([
            new Column('x', new Float32Array([0, 1, 2])),
            new Column('y', new Float32Array([0, 0, 0])),
            new Column('z', new Float32Array([0, 0, 0]))
        ]);

        const result = processDataTable(data, [
            { kind: 'translate', value: new Vec3(10, 0, 0) },
            { kind: 'filterByValue', columnName: 'x', comparator: 'gt', value: 11 }
        ]);

        assert.strictEqual(result.numRows, 1, 'Should keep one row after transformed-space filtering');
        assert.strictEqual(result.getColumnByName('x').data[0], 12, 'Transform should be baked into column data');
        assert.ok(result.transform.isIdentity(), 'Transform should be identity after baking');
    });

    it('should apply spatial transform but skip inverse transform for _raw suffix', () => {
        const logVal = Math.log(2);
        const data = new DataTable([
            new Column('x', new Float32Array([0, 0, 0])),
            new Column('y', new Float32Array([0, 0, 0])),
            new Column('z', new Float32Array([0, 0, 0])),
            new Column('scale_0', new Float32Array([0, logVal, logVal * 2])),
            new Column('scale_1', new Float32Array([0, 0, 0])),
            new Column('scale_2', new Float32Array([0, 0, 0]))
        ]);

        const result = processDataTable(data, [
            { kind: 'scale', value: 2 },
            { kind: 'filterByValue', columnName: 'scale_0_raw', comparator: 'gt', value: logVal * 1.5 }
        ]);

        assert.strictEqual(result.numRows, 2, 'Spatial transform should be applied before raw comparison');
    });
});

describe('Filter NaN', () => {
    it('should remove rows with NaN values', () => {
        const testData = createMinimalTestData();

        // Inject some NaN values
        testData.getColumnByName('x').data[0] = NaN;
        testData.getColumnByName('y').data[5] = NaN;

        const result = processDataTable(testData, [{
            kind: 'filterNaN'
        }]);

        // Should have fewer rows
        assert(result.numRows < 16, 'Should have fewer rows after filtering NaN');

        // No NaN values should remain
        const summary = computeSummary(result);
        for (const [name, stats] of Object.entries(summary.columns)) {
            assert.strictEqual(stats.nanCount, 0, `${name} should have no NaN values`);
        }
    });

    it('should keep all rows when no NaN values exist', () => {
        const testData = createMinimalTestData();

        const result = processDataTable(testData, [{
            kind: 'filterNaN'
        }]);

        assert.strictEqual(result.numRows, testData.numRows, 'Should keep all rows');
    });
});

describe('Filter SH Bands', () => {
    it('should remove higher SH bands', () => {
        // Create data with SH coefficients (band 1 = 9 coeffs per channel = 27 total)
        const testData = createMinimalTestData({ includeSH: true, shBands: 2 });

        // Verify we have SH columns
        assert(testData.hasColumn('f_rest_0'), 'Should have f_rest_0');
        assert(testData.hasColumn('f_rest_23'), 'Should have f_rest_23 for band 2');

        const result = processDataTable(testData, [{
            kind: 'filterBands',
            value: 1 // Keep only band 0 and 1 (9 coeffs per channel)
        }]);

        // Should still have band 1 columns
        assert(result.hasColumn('f_rest_0'), 'Should still have f_rest_0');
        assert(result.hasColumn('f_rest_8'), 'Should still have f_rest_8');

        // Should NOT have band 2 columns
        assert(!result.hasColumn('f_rest_9'), 'Should not have f_rest_9');
        assert(!result.hasColumn('f_rest_23'), 'Should not have f_rest_23');
    });

    it('should remove all SH bands when filtering to 0', () => {
        const testData = createMinimalTestData({ includeSH: true, shBands: 1 });

        const result = processDataTable(testData, [{
            kind: 'filterBands',
            value: 0
        }]);

        // Should not have any f_rest columns
        assert(!result.hasColumn('f_rest_0'), 'Should not have f_rest_0');
    });
});

describe('Chained Transforms', () => {
    it('should compose multiple transforms into transform', () => {
        const testData = createMinimalTestData();

        const result = processDataTable(testData, [
            { kind: 'scale', value: 2.0 },
            { kind: 'translate', value: new Vec3(100, 0, 0) }
        ]);

        // After scale(2) + translate(100,0,0):
        // engine_x = raw_x * 2 + 100
        const cols = transformColumns(result, ['x', 'y', 'z'], result.transform);
        const rawX = result.getColumnByName('x').data;
        for (let i = 0; i < result.numRows; i++) {
            assertClose(cols.get('x')[i], rawX[i] * 2 + 100, 1e-4, `engine x[${i}]`);
        }
    });

    it('should handle filter followed by transform', () => {
        const testData = createMinimalTestData();

        const result = processDataTable(testData, [
            {
                kind: 'filterBox',
                min: new Vec3(0, -Infinity, -Infinity),
                max: new Vec3(Infinity, Infinity, Infinity)
            },
            { kind: 'scale', value: 2.0 }
        ]);

        // Should have fewer rows after filter
        assert(result.numRows < testData.numRows, 'Should have fewer rows');

        // All x values should be positive and scaled
        const summary = computeSummary(result);
        assert(summary.columns.x.min >= 0, 'All x should be >= 0');
    });
});

describe('Summary Action', () => {
    it('should not modify data when computing summary', () => {
        const testData = createMinimalTestData();
        const originalRows = testData.numRows;

        // Summary action should just log, not modify data
        const result = processDataTable(testData, [{
            kind: 'summary'
        }]);

        assert.strictEqual(result.numRows, originalRows, 'Row count should be unchanged');
    });
});

describe('LOD Action', () => {
    it('should add LOD column with specified value', () => {
        const testData = createMinimalTestData();
        assert(!testData.hasColumn('lod'), 'Should not have lod column initially');

        const result = processDataTable(testData, [{
            kind: 'lod',
            value: 2
        }]);

        assert(result.hasColumn('lod'), 'Should have lod column after action');

        const lodCol = result.getColumnByName('lod').data;
        for (let i = 0; i < result.numRows; i++) {
            assert.strictEqual(lodCol[i], 2, `lod[${i}] should be 2`);
        }
    });
});

describe('Morton Order', () => {
    let testData;

    before(() => {
        testData = createMinimalTestData();
    });

    it('should preserve row count and summary statistics', () => {
        const originalSummary = computeSummary(testData);
        const clonedData = testData.clone();

        const result = processDataTable(clonedData, [{ kind: 'mortonOrder' }]);

        const newSummary = computeSummary(result);

        // Row count must be unchanged (it's a permutation)
        assert.strictEqual(result.numRows, testData.numRows, 'Row count should be unchanged');

        // All column statistics should be preserved since it's just a reordering
        for (const colName of Object.keys(originalSummary.columns)) {
            const orig = originalSummary.columns[colName];
            const curr = newSummary.columns[colName];
            assertClose(curr.min, orig.min, 1e-10, `${colName}.min`);
            assertClose(curr.max, orig.max, 1e-10, `${colName}.max`);
            assertClose(curr.mean, orig.mean, 1e-10, `${colName}.mean`);
            assertClose(curr.sum, orig.sum, 1e-10, `${colName}.sum`);
        }
    });

    it('should be idempotent (applying twice gives same result)', () => {
        const clonedData = testData.clone();

        // Apply mortonOrder once
        const result1 = processDataTable(clonedData, [{ kind: 'mortonOrder' }]);

        // Capture the order after first application
        const xAfterFirst = Array.from(result1.getColumnByName('x').data);
        const zAfterFirst = Array.from(result1.getColumnByName('z').data);

        // Apply mortonOrder again
        const result2 = processDataTable(result1, [{ kind: 'mortonOrder' }]);

        // Order should be unchanged
        const xAfterSecond = result2.getColumnByName('x').data;
        const zAfterSecond = result2.getColumnByName('z').data;

        for (let i = 0; i < result2.numRows; i++) {
            assert.strictEqual(xAfterSecond[i], xAfterFirst[i], `x[${i}] should be unchanged after second mortonOrder`);
            assert.strictEqual(zAfterSecond[i], zAfterFirst[i], `z[${i}] should be unchanged after second mortonOrder`);
        }
    });

    it('should preserve all data values (permutation correctness)', () => {
        const clonedData = testData.clone();

        // Collect all values before (as sorted arrays for comparison)
        const originalValues = {};
        for (const col of testData.columns) {
            originalValues[col.name] = Array.from(col.data).sort((a, b) => a - b);
        }

        const result = processDataTable(clonedData, [{ kind: 'mortonOrder' }]);

        // After mortonOrder, all values should still be present (just reordered)
        for (const col of result.columns) {
            const resultValues = Array.from(col.data).sort((a, b) => a - b);
            assert.strictEqual(resultValues.length, originalValues[col.name].length, `${col.name} should have same number of values`);
            for (let i = 0; i < resultValues.length; i++) {
                assertClose(resultValues[i], originalValues[col.name][i], 1e-10, `${col.name}[${i}] value should be preserved`);
            }
        }
    });

    it('should order by Morton code (spatial locality)', () => {
        const clonedData = testData.clone();

        const result = processDataTable(clonedData, [{ kind: 'mortonOrder' }]);

        // Compute Morton codes for the resulting order
        const xCol = result.getColumnByName('x').data;
        const yCol = result.getColumnByName('y').data;
        const zCol = result.getColumnByName('z').data;

        // Find extents
        let mx = Infinity, Mx = -Infinity;
        let my = Infinity, My = -Infinity;
        let mz = Infinity, Mz = -Infinity;
        for (let i = 0; i < result.numRows; i++) {
            if (xCol[i] < mx) mx = xCol[i];
            if (xCol[i] > Mx) Mx = xCol[i];
            if (yCol[i] < my) my = yCol[i];
            if (yCol[i] > My) My = yCol[i];
            if (zCol[i] < mz) mz = zCol[i];
            if (zCol[i] > Mz) Mz = zCol[i];
        }

        const xlen = Mx - mx;
        const ylen = My - my;
        const zlen = Mz - mz;

        const xmul = (xlen === 0) ? 0 : 1024 / xlen;
        const ymul = (ylen === 0) ? 0 : 1024 / ylen;
        const zmul = (zlen === 0) ? 0 : 1024 / zlen;

        // Morton encoding helper
        const Part1By2 = (x) => {
            x &= 0x000003ff;
            x = (x ^ (x << 16)) & 0xff0000ff;
            x = (x ^ (x << 8)) & 0x0300f00f;
            x = (x ^ (x << 4)) & 0x030c30c3;
            x = (x ^ (x << 2)) & 0x09249249;
            return x;
        };
        const encodeMorton3 = (x, y, z) => (Part1By2(z) << 2) + (Part1By2(y) << 1) + Part1By2(x);

        // Compute Morton codes
        const mortonCodes = [];
        for (let i = 0; i < result.numRows; i++) {
            const ix = Math.min(1023, (xCol[i] - mx) * xmul) >>> 0;
            const iy = Math.min(1023, (yCol[i] - my) * ymul) >>> 0;
            const iz = Math.min(1023, (zCol[i] - mz) * zmul) >>> 0;
            mortonCodes.push(encodeMorton3(ix, iy, iz));
        }

        // Morton codes should be non-decreasing
        for (let i = 1; i < mortonCodes.length; i++) {
            assert(mortonCodes[i] >= mortonCodes[i - 1],
                `Morton code at index ${i} (${mortonCodes[i]}) should be >= previous (${mortonCodes[i - 1]})`);
        }
    });

    it('should handle empty table', () => {
        const emptyData = new DataTable([
            new Column('x', new Float32Array(0)),
            new Column('y', new Float32Array(0)),
            new Column('z', new Float32Array(0))
        ]);

        const result = processDataTable(emptyData, [{ kind: 'mortonOrder' }]);

        assert.strictEqual(result.numRows, 0, 'Should still have 0 rows');
    });

    it('should handle single row', () => {
        const singleRowData = new DataTable([
            new Column('x', new Float32Array([1.0])),
            new Column('y', new Float32Array([2.0])),
            new Column('z', new Float32Array([3.0]))
        ]);

        const result = processDataTable(singleRowData, [{ kind: 'mortonOrder' }]);

        assert.strictEqual(result.numRows, 1, 'Should have 1 row');
        assertClose(result.getColumnByName('x').data[0], 1.0, 1e-10, 'x value');
        assertClose(result.getColumnByName('y').data[0], 2.0, 1e-10, 'y value');
        assertClose(result.getColumnByName('z').data[0], 3.0, 1e-10, 'z value');
    });

    it('should handle identical points', () => {
        // All points at the same location
        const identicalData = new DataTable([
            new Column('x', new Float32Array([5.0, 5.0, 5.0, 5.0])),
            new Column('y', new Float32Array([5.0, 5.0, 5.0, 5.0])),
            new Column('z', new Float32Array([5.0, 5.0, 5.0, 5.0])),
            new Column('id', new Float32Array([0, 1, 2, 3])) // Unique identifier
        ]);

        const result = processDataTable(identicalData, [{ kind: 'mortonOrder' }]);

        assert.strictEqual(result.numRows, 4, 'Should have 4 rows');

        // All values should still be 5.0
        for (let i = 0; i < 4; i++) {
            assertClose(result.getColumnByName('x').data[i], 5.0, 1e-10, `x[${i}]`);
        }

        // All ids should still be present
        const ids = Array.from(result.getColumnByName('id').data).sort((a, b) => a - b);
        assert.deepStrictEqual(ids, [0, 1, 2, 3], 'All ids should be preserved');
    });

    it('should handle zero-extent on one axis', () => {
        // All points on a plane (y = 0)
        const planarData = new DataTable([
            new Column('x', new Float32Array([0, 1, 2, 3])),
            new Column('y', new Float32Array([0, 0, 0, 0])), // Zero extent
            new Column('z', new Float32Array([0, 1, 2, 3]))
        ]);

        const result = processDataTable(planarData, [{ kind: 'mortonOrder' }]);

        assert.strictEqual(result.numRows, 4, 'Should have 4 rows');

        // All y values should still be 0
        for (let i = 0; i < 4; i++) {
            assertClose(result.getColumnByName('y').data[i], 0, 1e-10, `y[${i}]`);
        }

        // Check all x values are preserved
        const xValues = Array.from(result.getColumnByName('x').data).sort((a, b) => a - b);
        assert.deepStrictEqual(xValues, [0, 1, 2, 3], 'All x values should be preserved');
    });
});

describe('permuteRowsInPlace', () => {
    it('should handle identity permutation', () => {
        const data = new DataTable([
            new Column('a', new Float32Array([1, 2, 3, 4])),
            new Column('b', new Float32Array([10, 20, 30, 40]))
        ]);

        const indices = new Uint32Array([0, 1, 2, 3]);
        data.permuteRowsInPlace(indices);

        // Data should be unchanged
        assert.deepStrictEqual(Array.from(data.getColumnByName('a').data), [1, 2, 3, 4]);
        assert.deepStrictEqual(Array.from(data.getColumnByName('b').data), [10, 20, 30, 40]);
    });

    it('should handle reverse permutation', () => {
        const data = new DataTable([
            new Column('a', new Float32Array([1, 2, 3, 4])),
            new Column('b', new Float32Array([10, 20, 30, 40]))
        ]);

        const indices = new Uint32Array([3, 2, 1, 0]);
        data.permuteRowsInPlace(indices);

        // Data should be reversed
        assert.deepStrictEqual(Array.from(data.getColumnByName('a').data), [4, 3, 2, 1]);
        assert.deepStrictEqual(Array.from(data.getColumnByName('b').data), [40, 30, 20, 10]);
    });

    it('should handle simple swap', () => {
        const data = new DataTable([
            new Column('a', new Float32Array([1, 2, 3, 4])),
            new Column('b', new Float32Array([10, 20, 30, 40]))
        ]);

        // Swap first two elements only
        const indices = new Uint32Array([1, 0, 2, 3]);
        data.permuteRowsInPlace(indices);

        assert.deepStrictEqual(Array.from(data.getColumnByName('a').data), [2, 1, 3, 4]);
        assert.deepStrictEqual(Array.from(data.getColumnByName('b').data), [20, 10, 30, 40]);
    });

    it('should handle multi-element cycle', () => {
        const data = new DataTable([
            new Column('a', new Float32Array([1, 2, 3, 4, 5])),
            new Column('b', new Float32Array([10, 20, 30, 40, 50]))
        ]);

        // 3-cycle: 0 -> 1 -> 2 -> 0, rest unchanged
        // indices[i] = source index for position i
        // indices = [1, 2, 0, 3, 4] means:
        //   position 0 gets value from index 1
        //   position 1 gets value from index 2
        //   position 2 gets value from index 0
        const indices = new Uint32Array([1, 2, 0, 3, 4]);
        data.permuteRowsInPlace(indices);

        // After cycle: [2, 3, 1, 4, 5]
        assert.deepStrictEqual(Array.from(data.getColumnByName('a').data), [2, 3, 1, 4, 5]);
        assert.deepStrictEqual(Array.from(data.getColumnByName('b').data), [20, 30, 10, 40, 50]);
    });

    it('should handle multiple independent cycles', () => {
        const data = new DataTable([
            new Column('a', new Float32Array([1, 2, 3, 4, 5, 6]))
        ]);

        // Two independent 3-cycles: (0,1,2) and (3,4,5)
        const indices = new Uint32Array([1, 2, 0, 4, 5, 3]);
        data.permuteRowsInPlace(indices);

        assert.deepStrictEqual(Array.from(data.getColumnByName('a').data), [2, 3, 1, 5, 6, 4]);
    });

    it('should handle empty data', () => {
        const data = new DataTable([
            new Column('a', new Float32Array(0))
        ]);

        const indices = new Uint32Array(0);
        data.permuteRowsInPlace(indices);

        assert.strictEqual(data.numRows, 0);
    });
});
