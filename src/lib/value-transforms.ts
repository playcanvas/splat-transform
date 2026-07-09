/**
 * Conversions between user-friendly ("transformed") column values and raw PLY
 * storage values, shared by the `DataTable` process path and the `ChunkSource`
 * filter passes so the two stay byte-for-byte in agreement.
 *
 * - `opacity`, `scale_*`, `f_dc_*` are exposed to users in linear / 0-1 space
 *   but stored in logit / log / SH-DC space. `inverseTransforms` maps a
 *   user-friendly value to raw; `forwardTransforms` maps raw back for display.
 * - `rawColumnMap` resolves a `*_raw` column name to its underlying column
 *   (compare the stored value directly, skipping the user-friendly conversion).
 * - `isTransformColumn` reports whether a column's stored value is affected by a
 *   pending spatial transform (and so must be baked before comparison).
 */

const SH_C0 = 0.28209479177387814;

// Inverse transforms: convert user-friendly values to raw PLY space.
// All transforms are monotonic increasing, so comparison direction is preserved.
const inverseTransforms: Record<string, (v: number) => number> = {
    'opacity': v => Math.log(v / (1 - v)),
    'scale_0': Math.log,
    'scale_1': Math.log,
    'scale_2': Math.log,
    'f_dc_0': v => (v - 0.5) / SH_C0,
    'f_dc_1': v => (v - 0.5) / SH_C0,
    'f_dc_2': v => (v - 0.5) / SH_C0
};

// Forward transforms: convert raw PLY values to user-friendly space (for summary display).
const forwardTransforms: Record<string, (v: number) => number> = {
    'opacity': v => 1 / (1 + Math.exp(-v)),
    'scale_0': Math.exp,
    'scale_1': Math.exp,
    'scale_2': Math.exp,
    'f_dc_0': v => 0.5 + v * SH_C0,
    'f_dc_1': v => 0.5 + v * SH_C0,
    'f_dc_2': v => 0.5 + v * SH_C0
};

// Maps `_raw` suffixed column names to their underlying PLY column.
const rawColumnMap: Record<string, string> = {
    'opacity_raw': 'opacity',
    'scale_0_raw': 'scale_0',
    'scale_1_raw': 'scale_1',
    'scale_2_raw': 'scale_2',
    'f_dc_0_raw': 'f_dc_0',
    'f_dc_1_raw': 'f_dc_1',
    'f_dc_2_raw': 'f_dc_2'
};

const transformColumnNames = new Set([
    'x', 'y', 'z',
    'rot_0', 'rot_1', 'rot_2', 'rot_3',
    'scale_0', 'scale_1', 'scale_2'
]);

/**
 * Whether a column's stored value is affected by a pending spatial transform.
 * @param name - The column name.
 * @returns `true` for position / rotation / scale / `f_rest_*` columns.
 */
const isTransformColumn = (name: string): boolean => transformColumnNames.has(name) || /^f_rest_\d+$/.test(name);

export { SH_C0, inverseTransforms, forwardTransforms, rawColumnMap, isTransformColumn };
