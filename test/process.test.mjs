/**
 * Tests for processDataTable validation and edge cases.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { processDataTable, Column, DataTable } from '../src/lib/index.js';
import { createTestDataTable } from './helpers/test-utils.mjs';

describe('processDataTable', function () {
    describe('filterNaN', function () {
        it('should use current result columns, not original dataTable columns', async function () {
            const dataTable = createTestDataTable(10, { includeSH: true, shBands: 1 });

            const result = await processDataTable(dataTable, [
                { kind: 'filterBands', value: 0 },
                { kind: 'filterNaN' }
            ]);

            assert.ok(result.numRows > 0, 'Should not drop valid rows');
            assert.ok(!result.hasColumn('f_rest_0'),
                'SH columns should have been removed by filterBands');
        });

        it('should allow +Infinity on opacity column', async function () {
            const dataTable = new DataTable([
                new Column('x', new Float32Array([0, 1])),
                new Column('y', new Float32Array([0, 0])),
                new Column('z', new Float32Array([0, 0])),
                new Column('opacity', new Float32Array([Infinity, 1.0]))
            ]);

            const result = await processDataTable(dataTable, [{ kind: 'filterNaN' }]);
            assert.strictEqual(result.numRows, 2, '+Infinity opacity should be allowed');
        });

        it('should remove rows with NaN values', async function () {
            const dataTable = new DataTable([
                new Column('x', new Float32Array([0, NaN, 2])),
                new Column('y', new Float32Array([0, 0, 0])),
                new Column('z', new Float32Array([0, 0, 0]))
            ]);

            const result = await processDataTable(dataTable, [{ kind: 'filterNaN' }]);
            assert.strictEqual(result.numRows, 2, 'Should remove NaN row');
        });
    });

    describe('filterByValue', function () {
        it('should throw for non-existent column', async function () {
            const dataTable = createTestDataTable(4);

            await assert.rejects(
                processDataTable(dataTable, [{
                    kind: 'filterByValue',
                    columnName: 'nonexistent_column',
                    comparator: 'gt',
                    value: 0
                }]),
                /column 'nonexistent_column' not found/
            );
        });

        it('should throw for opacity value of 0', async function () {
            const dataTable = createTestDataTable(4);

            await assert.rejects(
                processDataTable(dataTable, [{
                    kind: 'filterByValue',
                    columnName: 'opacity',
                    comparator: 'gt',
                    value: 0
                }]),
                /opacity value must be between 0 and 1/
            );
        });

        it('should throw for opacity value of 1', async function () {
            const dataTable = createTestDataTable(4);

            await assert.rejects(
                processDataTable(dataTable, [{
                    kind: 'filterByValue',
                    columnName: 'opacity',
                    comparator: 'gt',
                    value: 1
                }]),
                /opacity value must be between 0 and 1/
            );
        });

        it('should accept valid opacity values', async function () {
            const dataTable = createTestDataTable(10);

            const result = await processDataTable(dataTable, [{
                kind: 'filterByValue',
                columnName: 'opacity',
                comparator: 'gt',
                value: 0.5
            }]);

            assert.ok(result.numRows >= 0, 'Should not throw for valid opacity');
        });

        it('should accept raw column names without opacity validation', async function () {
            const dataTable = createTestDataTable(10);

            const result = await processDataTable(dataTable, [{
                kind: 'filterByValue',
                columnName: 'opacity_raw',
                comparator: 'gt',
                value: 0
            }]);

            assert.ok(result.numRows >= 0, 'Raw column should bypass inverse transform');
        });

        it('should filter correctly with lt comparator', async function () {
            const dataTable = new DataTable([
                new Column('x', new Float32Array([1, 2, 3, 4, 5])),
                new Column('y', new Float32Array(5)),
                new Column('z', new Float32Array(5))
            ]);

            const result = await processDataTable(dataTable, [{
                kind: 'filterByValue',
                columnName: 'x',
                comparator: 'lt',
                value: 3
            }]);

            assert.strictEqual(result.numRows, 2, 'Should keep rows with x < 3');
        });
    });

    describe('filterFloaters', function () {
        it('should throw without createDevice', async function () {
            const dataTable = createTestDataTable(4);

            await assert.rejects(
                processDataTable(dataTable, [{ kind: 'filterFloaters' }]),
                /filterFloaters requires a createDevice function/
            );
        });
    });

    describe('filterCluster', function () {
        it('should throw without createDevice', async function () {
            const dataTable = createTestDataTable(4);

            await assert.rejects(
                processDataTable(dataTable, [{ kind: 'filterCluster' }]),
                /filterCluster requires a createDevice function/
            );
        });
    });
});
