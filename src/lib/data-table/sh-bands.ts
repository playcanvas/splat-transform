import type { DataTable } from './data-table';

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

export { getSHBands, shRestNames };
