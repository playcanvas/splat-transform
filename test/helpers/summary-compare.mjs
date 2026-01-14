/**
 * Utility functions for comparing DataTable summaries in tests.
 */

import assert from 'node:assert';

/**
 * Asserts that two numbers are approximately equal within a tolerance.
 * @param {number} actual - The actual value
 * @param {number} expected - The expected value
 * @param {number} tolerance - Absolute tolerance for comparison
 * @param {string} [message] - Optional error message
 */
function assertClose(actual, expected, tolerance, message = '') {
    if (Number.isNaN(expected)) {
        assert(Number.isNaN(actual), `${message}: expected NaN, got ${actual}`);
        return;
    }
    if (!Number.isFinite(expected)) {
        assert.strictEqual(actual, expected, `${message}: expected ${expected}, got ${actual}`);
        return;
    }
    const diff = Math.abs(actual - expected);
    assert(
        diff <= tolerance,
        `${message}: expected ${expected} ± ${tolerance}, got ${actual} (diff: ${diff})`
    );
}

/**
 * Compares two column statistics objects.
 * @param {object} actual - Actual column stats
 * @param {object} expected - Expected column stats
 * @param {number} tolerance - Tolerance for numeric comparisons
 * @param {string} columnName - Name of the column (for error messages)
 */
function compareColumnStats(actual, expected, tolerance, columnName) {
    assertClose(actual.min, expected.min, tolerance, `${columnName}.min`);
    assertClose(actual.max, expected.max, tolerance, `${columnName}.max`);
    assertClose(actual.median, expected.median, tolerance, `${columnName}.median`);
    assertClose(actual.mean, expected.mean, tolerance, `${columnName}.mean`);
    assertClose(actual.stdDev, expected.stdDev, tolerance, `${columnName}.stdDev`);
    assert.strictEqual(actual.nanCount, expected.nanCount, `${columnName}.nanCount mismatch`);
    assert.strictEqual(actual.infCount, expected.infCount, `${columnName}.infCount mismatch`);
}

/**
 * Compares two SummaryData objects.
 * @param {object} actual - Actual summary data
 * @param {object} expected - Expected summary data
 * @param {object} [options] - Comparison options
 * @param {number} [options.tolerance=0] - Tolerance for numeric comparisons
 * @param {boolean} [options.allowExtraColumns=false] - Allow actual to have extra columns
 * @param {string[]} [options.ignoreColumns=[]] - Columns to skip in comparison
 */
function compareSummaries(actual, expected, options = {}) {
    const { tolerance = 0, allowExtraColumns = false, ignoreColumns = [] } = options;

    // Compare row counts
    assert.strictEqual(
        actual.rowCount,
        expected.rowCount,
        `Row count mismatch: expected ${expected.rowCount}, got ${actual.rowCount}`
    );

    // Compare columns
    for (const [name, expectedStats] of Object.entries(expected.columns)) {
        if (ignoreColumns.includes(name)) continue;

        const actualStats = actual.columns[name];
        assert(actualStats, `Missing column in actual: ${name}`);

        compareColumnStats(actualStats, expectedStats, tolerance, name);
    }

    // Check for unexpected columns (if not allowed)
    if (!allowExtraColumns) {
        const expectedNames = new Set(Object.keys(expected.columns));
        const ignoredNames = new Set(ignoreColumns);
        for (const name of Object.keys(actual.columns)) {
            if (!ignoredNames.has(name)) {
                assert(
                    expectedNames.has(name),
                    `Unexpected column in actual: ${name}`
                );
            }
        }
    }
}

/**
 * Compares two DataTables row by row (for lossless round-trip tests).
 * @param {object} actual - Actual DataTable
 * @param {object} expected - Expected DataTable
 * @param {number} [tolerance=0] - Tolerance for numeric comparisons
 */
function compareDataTables(actual, expected, tolerance = 0) {
    assert.strictEqual(
        actual.numRows,
        expected.numRows,
        `Row count mismatch: expected ${expected.numRows}, got ${actual.numRows}`
    );

    assert.strictEqual(
        actual.numColumns,
        expected.numColumns,
        `Column count mismatch: expected ${expected.numColumns}, got ${actual.numColumns}`
    );

    // Compare column names
    const actualNames = actual.columnNames;
    const expectedNames = expected.columnNames;
    assert.deepStrictEqual(actualNames, expectedNames, 'Column names mismatch');

    // Compare data row by row
    for (let col = 0; col < expected.numColumns; col++) {
        const colName = expectedNames[col];
        const actualData = actual.getColumn(col).data;
        const expectedData = expected.getColumn(col).data;

        for (let row = 0; row < expected.numRows; row++) {
            assertClose(
                actualData[row],
                expectedData[row],
                tolerance,
                `Data mismatch at column '${colName}', row ${row}`
            );
        }
    }
}

export { assertClose, compareColumnStats, compareSummaries, compareDataTables };
