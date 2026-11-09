import { Column, DataTable } from './data-table';

const shRestNames = new Array(72).fill('').map((_, i) => `f_rest_${i}`);

/**
 * Determines how many SH bands (0-4) the DataTable contains beyond the DC
 * term. Detection assumes the channel-major layout used throughout the
 * codebase: N coefficients per channel, 3 channels, stored as
 * `f_rest_0`..`f_rest_(3N-1)`.
 *
 * - 0 bands: no `f_rest_*` columns
 * - 1 band : `f_rest_0`..`f_rest_8`   (9 coeffs)
 * - 2 bands: `f_rest_0`..`f_rest_23`  (24 coeffs)
 * - 3 bands: `f_rest_0`..`f_rest_44`  (45 coeffs)
 * - 4 bands: `f_rest_0`..`f_rest_71`  (72 coeffs)
 *
 * @param dataTable - The DataTable to inspect.
 * @returns The number of SH bands (0-4).
 */
const getSHBands = (dataTable: DataTable): number => {
    const idx = shRestNames.findIndex((v) => !dataTable.hasColumn(v));
    return ({ '9': 1, '24': 2, '45': 3, '-1': 4 } as Record<string, number>)[String(idx)] ?? 0;
};

/**
 * The table with at most `bands` SH bands: higher-band coefficients are dropped
 * and the kept ones renamed to their channel-major positions for the smaller
 * band count. Returns the table itself when it has no more than `bands`.
 *
 * @param dataTable - The DataTable to reduce.
 * @param bands - The most SH bands to keep (0-3).
 * @returns A DataTable sharing the kept columns' data.
 */
const keepSHBands = (dataTable: DataTable, bands: number): DataTable => {
    const inputBands = getSHBands(dataTable);
    if (inputBands <= bands) return dataTable;
    const inputCoeffs = [0, 3, 8, 15, 24][inputBands];
    const outputCoeffs = [0, 3, 8, 15, 24][bands];
    const rename = new Map<string, string | null>();
    for (let i = 0; i < inputCoeffs; ++i) {
        for (let j = 0; j < 3; ++j) {
            rename.set(`f_rest_${i + j * inputCoeffs}`, i < outputCoeffs ? `f_rest_${i + j * outputCoeffs}` : null);
        }
    }
    const columns: Column[] = [];
    for (const column of dataTable.columns) {
        if (!rename.has(column.name)) columns.push(column);
        else if (rename.get(column.name)) columns.push(new Column(rename.get(column.name)!, column.data));
    }
    return new DataTable(columns, dataTable.transform);
};

export { getSHBands, keepSHBands, shRestNames };
