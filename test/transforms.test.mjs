/**
 * Transformation tests for splat-transform.
 * Tests translate, rotate, scale, and filter operations.
 */

import { describe, it, before } from 'node:test';
import assert from 'node:assert';

import {
    computeSummary,
    processDataTable
} from '../dist/index.mjs';

import { createMinimalTestData } from './helpers/test-utils.mjs';
import { assertClose } from './helpers/summary-compare.mjs';

// Import Vec3 from playcanvas (used in actions)
import { Vec3 } from 'playcanvas';

describe('Translate Transform', () => {
    let testData;

    before(() => {
        testData = createMinimalTestData();
    });

    it('should translate positions by specified offset', () => {
        const originalSummary = computeSummary(testData);
        const clonedData = testData.clone();

        const result = processDataTable(clonedData, [{
            kind: 'translate',
            value: new Vec3(10, 20, 30)
        }]);

        const newSummary = computeSummary(result);

        // Positions should be shifted
        assertClose(newSummary.columns.x.mean, originalSummary.columns.x.mean + 10, 1e-5, 'x mean');
        assertClose(newSummary.columns.y.mean, originalSummary.columns.y.mean + 20, 1e-5, 'y mean');
        assertClose(newSummary.columns.z.mean, originalSummary.columns.z.mean + 30, 1e-5, 'z mean');

        // Other properties should be unchanged
        assertClose(newSummary.columns.scale_0.mean, originalSummary.columns.scale_0.mean, 1e-5, 'scale_0');
        assertClose(newSummary.columns.opacity.mean, originalSummary.columns.opacity.mean, 1e-5, 'opacity');
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

    it('should scale positions and scales by factor', () => {
        const originalSummary = computeSummary(testData);
        const clonedData = testData.clone();

        const scaleFactor = 2.0;
        const result = processDataTable(clonedData, [{
            kind: 'scale',
            value: scaleFactor
        }]);

        const newSummary = computeSummary(result);

        // Positions should be scaled
        assertClose(newSummary.columns.x.min, originalSummary.columns.x.min * scaleFactor, 1e-5, 'x.min');
        assertClose(newSummary.columns.x.max, originalSummary.columns.x.max * scaleFactor, 1e-5, 'x.max');
        assertClose(newSummary.columns.z.min, originalSummary.columns.z.min * scaleFactor, 1e-5, 'z.min');
        assertClose(newSummary.columns.z.max, originalSummary.columns.z.max * scaleFactor, 1e-5, 'z.max');

        // Log-encoded scales should shift by log(factor)
        const logFactor = Math.log(scaleFactor);
        assertClose(newSummary.columns.scale_0.mean, originalSummary.columns.scale_0.mean + logFactor, 1e-5, 'scale_0');
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
        const originalSummary = computeSummary(testData);
        const clonedData = testData.clone();

        const scaleFactor = 0.5;
        const result = processDataTable(clonedData, [{
            kind: 'scale',
            value: scaleFactor
        }]);

        const newSummary = computeSummary(result);

        // Positions should be scaled down
        assertClose(newSummary.columns.x.max, originalSummary.columns.x.max * scaleFactor, 1e-5, 'x.max');
    });
});

describe('Rotate Transform', () => {
    let testData;

    before(() => {
        testData = createMinimalTestData();
    });

    it('should rotate positions around Y axis', () => {
        const originalSummary = computeSummary(testData);
        const clonedData = testData.clone();

        // 90 degree rotation around Y
        const result = processDataTable(clonedData, [{
            kind: 'rotate',
            value: new Vec3(0, 90, 0)
        }]);

        const newSummary = computeSummary(result);

        // After 90 degree Y rotation (counter-clockwise when looking down Y):
        // x' = z
        // z' = -x
        // So new z range = -old_x_range (reversed)
        assertClose(newSummary.columns.z.min, -originalSummary.columns.x.max, 1e-4, 'z.min after rotation');
        assertClose(newSummary.columns.z.max, -originalSummary.columns.x.min, 1e-4, 'z.max after rotation');

        // Row count should be unchanged
        assert.strictEqual(newSummary.rowCount, originalSummary.rowCount);
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
    it('should apply multiple transforms in order', () => {
        const testData = createMinimalTestData();
        const originalSummary = computeSummary(testData);

        const result = processDataTable(testData, [
            { kind: 'scale', value: 2.0 },
            { kind: 'translate', value: new Vec3(100, 0, 0) }
        ]);

        const newSummary = computeSummary(result);

        // After scale(2) + translate(100,0,0):
        // x_new = x_old * 2 + 100
        const expectedXMean = originalSummary.columns.x.mean * 2 + 100;
        assertClose(newSummary.columns.x.mean, expectedXMean, 1e-4, 'x mean after transforms');
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
