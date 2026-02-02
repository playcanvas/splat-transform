/**
 * Visibility filter tests for splat-transform.
 * Tests sortByVisibility function and filterVisibility action.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';

import {
    Column,
    DataTable,
    processDataTable,
    sortByVisibility
} from '../dist/index.mjs';

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
    it('should sort indices by visibility score (descending)', () => {
        // Create data with known visibility scores:
        // Splat 0: opacity=0 (sigmoid=0.5), scales=0,0,0 (volume=1) -> score=0.5
        // Splat 1: opacity=2.197 (sigmoid≈0.9), scales=0,0,0 (volume=1) -> score≈0.9
        // Splat 2: opacity=-2.197 (sigmoid≈0.1), scales=0,0,0 (volume=1) -> score≈0.1
        // Splat 3: opacity=0 (sigmoid=0.5), scales=ln(2),ln(2),ln(2) (volume=8) -> score=4.0
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

        // Expected order by visibility (highest first): 3, 1, 0, 2
        // Scores: splat3=4.0, splat1≈0.9, splat0=0.5, splat2≈0.1
        assert.strictEqual(indices[0], 3, 'Highest visibility should be first');
        assert.strictEqual(indices[1], 1, 'Second highest should be second');
        assert.strictEqual(indices[2], 0, 'Third highest should be third');
        assert.strictEqual(indices[3], 2, 'Lowest visibility should be last');
    });

    it('should handle empty indices', () => {
        const testData = createVisibilityTestData({ count: 4 });
        const indices = new Uint32Array(0);

        // Should not throw
        sortByVisibility(testData, indices);

        assert.strictEqual(indices.length, 0, 'Empty indices should remain empty');
    });

    it('should handle missing columns gracefully', () => {
        // Create DataTable without opacity column
        const testData = new DataTable([
            new Column('x', new Float32Array([0, 1, 2])),
            new Column('y', new Float32Array([0, 0, 0])),
            new Column('z', new Float32Array([0, 0, 0]))
        ]);

        const indices = new Uint32Array([0, 1, 2]);
        const originalIndices = indices.slice();

        // Should not throw, should leave indices unchanged
        sortByVisibility(testData, indices);

        assert.deepStrictEqual(Array.from(indices), Array.from(originalIndices),
            'Indices should be unchanged when columns are missing');
    });

    it('should handle single element', () => {
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

describe('filterVisibility - Count Mode', () => {
    it('should keep exactly N splats in count mode', () => {
        const testData = createMinimalTestData();
        const originalRows = testData.numRows;

        const result = processDataTable(testData, [{
            kind: 'filterVisibility',
            count: 5,
            percent: null
        }]);

        assert.strictEqual(result.numRows, 5, 'Should have exactly 5 rows');
        assert(result.numRows < originalRows, 'Should have fewer rows than original');
    });

    it('should keep the most visible splats', () => {
        // Create data with distinct visibility scores
        const testData = createVisibilityTestData({
            count: 4,
            x: new Float32Array([0, 1, 2, 3]),
            opacity: new Float32Array([0, 2.197, -2.197, 0]),
            scale_0: new Float32Array([0, 0, 0, Math.log(2)]),
            scale_1: new Float32Array([0, 0, 0, Math.log(2)]),
            scale_2: new Float32Array([0, 0, 0, Math.log(2)])
        });

        // Keep top 2 most visible (should be splats 3 and 1)
        const result = processDataTable(testData, [{
            kind: 'filterVisibility',
            count: 2,
            percent: null
        }]);

        assert.strictEqual(result.numRows, 2, 'Should have exactly 2 rows');

        // Check that we kept splats 3 and 1 (the ones with x=3 and x=1)
        const xValues = Array.from(result.getColumnByName('x').data).sort((a, b) => a - b);
        assert.deepStrictEqual(xValues, [1, 3], 'Should keep splats with x=1 and x=3');
    });

    it('should keep all splats when count exceeds numRows', () => {
        const testData = createMinimalTestData();
        const originalRows = testData.numRows;

        const result = processDataTable(testData, [{
            kind: 'filterVisibility',
            count: 1000,
            percent: null
        }]);

        assert.strictEqual(result.numRows, originalRows, 'Should keep all rows when count > numRows');
    });

    it('should handle count of 0', () => {
        const testData = createMinimalTestData();

        const result = processDataTable(testData, [{
            kind: 'filterVisibility',
            count: 0,
            percent: null
        }]);

        assert.strictEqual(result.numRows, 0, 'Should have 0 rows when count is 0');
    });
});

describe('filterVisibility - Percent Mode', () => {
    it('should keep approximately X% of splats', () => {
        const testData = createMinimalTestData();
        const originalRows = testData.numRows; // 16

        const result = processDataTable(testData, [{
            kind: 'filterVisibility',
            count: null,
            percent: 50
        }]);

        // 50% of 16 = 8
        assert.strictEqual(result.numRows, 8, 'Should have 50% of rows (8)');
    });

    it('should keep all splats at 100%', () => {
        const testData = createMinimalTestData();
        const originalRows = testData.numRows;

        const result = processDataTable(testData, [{
            kind: 'filterVisibility',
            count: null,
            percent: 100
        }]);

        assert.strictEqual(result.numRows, originalRows, 'Should keep all rows at 100%');
    });

    it('should remove all splats at 0%', () => {
        const testData = createMinimalTestData();

        const result = processDataTable(testData, [{
            kind: 'filterVisibility',
            count: null,
            percent: 0
        }]);

        assert.strictEqual(result.numRows, 0, 'Should have 0 rows at 0%');
    });

    it('should handle 25%', () => {
        const testData = createMinimalTestData();
        const originalRows = testData.numRows; // 16

        const result = processDataTable(testData, [{
            kind: 'filterVisibility',
            count: null,
            percent: 25
        }]);

        // 25% of 16 = 4
        assert.strictEqual(result.numRows, 4, 'Should have 25% of rows (4)');
    });
});

describe('Visibility Score Calculation', () => {
    it('should correctly compute visibility from logit opacity and log scales', () => {
        // Splat A: opacity=0 (logit) -> sigmoid=0.5, scales=0,0,0 -> volume=1, visibility=0.5
        // Splat B: opacity=2.197 (logit) -> sigmoid≈0.9, scales=0,0,0 -> volume=1, visibility≈0.9
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

        // Higher opacity splat (index 1) should come first
        assert.strictEqual(indices[0], 1, 'Higher opacity splat should be first');
        assert.strictEqual(indices[1], 0, 'Lower opacity splat should be second');
    });

    it('should correctly incorporate scale into visibility', () => {
        // Both splats have same opacity (0 logit = 0.5 linear)
        // Splat A: scales=0,0,0 -> volume=1, visibility=0.5
        // Splat B: scales=ln(10),0,0 -> volume=10, visibility=5.0
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

        // Larger scale splat (index 1) should come first
        assert.strictEqual(indices[0], 1, 'Larger scale splat should be first');
        assert.strictEqual(indices[1], 0, 'Smaller scale splat should be second');
    });

    it('should handle negative log scales (small splats)', () => {
        // Small splats should have lower visibility
        // Splat A: opacity=0, scales=-2,-2,-2 -> volume=exp(-6)≈0.0025, visibility≈0.00125
        // Splat B: opacity=0, scales=0,0,0 -> volume=1, visibility=0.5
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

        // Normal scale splat (index 1) should come first
        assert.strictEqual(indices[0], 1, 'Normal scale splat should be first');
        assert.strictEqual(indices[1], 0, 'Small scale splat should be second');
    });

    it('should handle very low opacity', () => {
        // Splat A: opacity=-10 (logit) -> sigmoid≈0.00005
        // Splat B: opacity=0 (logit) -> sigmoid=0.5
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

        // Higher opacity splat (index 1) should come first
        assert.strictEqual(indices[0], 1, 'Higher opacity splat should be first');
        assert.strictEqual(indices[1], 0, 'Very low opacity splat should be second');
    });
});

describe('permuteRows with Smaller Indices', () => {
    it('should create smaller DataTable when indices.length < numRows', () => {
        const testData = new DataTable([
            new Column('a', new Float32Array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])),
            new Column('b', new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        ]);

        // Select only 5 rows
        const indices = new Uint32Array([0, 2, 4, 6, 8]);
        const result = testData.permuteRows(indices);

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

    it('should handle selecting just one row', () => {
        const testData = new DataTable([
            new Column('a', new Float32Array([10, 20, 30, 40, 50]))
        ]);

        const indices = new Uint32Array([2]);
        const result = testData.permuteRows(indices);

        assert.strictEqual(result.numRows, 1, 'Should have exactly 1 row');
        assert.strictEqual(result.getColumnByName('a').data[0], 30, 'Should have value from index 2');
    });

    it('should handle empty indices', () => {
        const testData = new DataTable([
            new Column('a', new Float32Array([10, 20, 30]))
        ]);

        const indices = new Uint32Array(0);
        const result = testData.permuteRows(indices);

        assert.strictEqual(result.numRows, 0, 'Should have 0 rows');
    });
});

describe('filterVisibility Integration', () => {
    it('should chain with other transforms', () => {
        const testData = createMinimalTestData();

        const result = processDataTable(testData, [
            { kind: 'translate', value: new Vec3(10, 0, 0) },
            { kind: 'filterVisibility', count: 8, percent: null },
            { kind: 'scale', value: 2.0 }
        ]);

        assert.strictEqual(result.numRows, 8, 'Should have 8 rows after filtering');

        // All x values should be shifted and scaled
        const xCol = result.getColumnByName('x').data;
        for (let i = 0; i < result.numRows; i++) {
            // Original x was in range roughly -1.5 to 1.5, after translate +10 and scale *2
            // x values should be > 10 (since they were > -1.5, after +10 = > 8.5, *2 = > 17)
            assert(xCol[i] > 10, `x[${i}] should be > 10 after transforms`);
        }
    });

    it('should preserve column data integrity after filtering', () => {
        // Create data with unique x values to track rows
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

        const result = processDataTable(testData, [{
            kind: 'filterVisibility',
            count: 2,
            percent: null
        }]);

        assert.strictEqual(result.numRows, 2, 'Should have 2 rows');

        // Check that all columns are present
        assert(result.hasColumn('x'), 'Should have x column');
        assert(result.hasColumn('y'), 'Should have y column');
        assert(result.hasColumn('z'), 'Should have z column');
        assert(result.hasColumn('opacity'), 'Should have opacity column');

        // Check row integrity - y and z values should match x values
        const xCol = result.getColumnByName('x').data;
        const yCol = result.getColumnByName('y').data;
        const zCol = result.getColumnByName('z').data;

        for (let i = 0; i < result.numRows; i++) {
            const x = xCol[i];
            // Based on our test data: y = x/100, z = x/10
            assertClose(yCol[i], x / 100, 1e-5, `y[${i}] should match x[${i}]`);
            assertClose(zCol[i], x / 10, 1e-5, `z[${i}] should match x[${i}]`);
        }
    });

    it('should work with Morton ordering after filtering', () => {
        const testData = createMinimalTestData();

        const result = processDataTable(testData, [
            { kind: 'filterVisibility', count: 8, percent: null },
            { kind: 'mortonOrder' }
        ]);

        assert.strictEqual(result.numRows, 8, 'Should have 8 rows');

        // Morton ordering should have reordered the data - just verify no errors
        assert(result.hasColumn('x'), 'Should have x column');
        assert(result.hasColumn('y'), 'Should have y column');
        assert(result.hasColumn('z'), 'Should have z column');
    });

    it('should preserve all column values (just reordered and filtered)', () => {
        const testData = createMinimalTestData();
        const originalRows = testData.numRows;

        // Get all original x values
        const originalXValues = Array.from(testData.getColumnByName('x').data);

        const result = processDataTable(testData.clone(), [{
            kind: 'filterVisibility',
            count: 8,
            percent: null
        }]);

        // All result x values should be a subset of original x values
        const resultXValues = Array.from(result.getColumnByName('x').data);
        for (const x of resultXValues) {
            assert(originalXValues.includes(x), `x value ${x} should be from original data`);
        }
    });
});
