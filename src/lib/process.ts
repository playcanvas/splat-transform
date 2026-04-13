import { Vec3 } from 'playcanvas';

import { Column, DataTable } from './data-table/data-table';
import { simplifyGaussians } from './data-table/decimate';
import { sortMortonOrder } from './data-table/morton-order';
import { computeSummary, type SummaryData } from './data-table/summary';
import { convertToSpace } from './data-table/transform';
import type { DeviceCreator } from './types';
import { logger } from './utils/logger';
import { Transform } from './utils/math';

/**
 * Translate splats by a 3D vector offset.
 */
type Translate = {
    /** Action type identifier. */
    kind: 'translate';
    /** Translation offset. */
    value: Vec3;
};

/**
 * Rotate splats by Euler angles.
 */
type Rotate = {
    /** Action type identifier. */
    kind: 'rotate';
    /** Euler angles in degrees (x, y, z). */
    value: Vec3;
};

/**
 * Uniformly scale all splats.
 */
type Scale = {
    /** Action type identifier. */
    kind: 'scale';
    /** Scale factor. */
    value: number;
};

/**
 * Remove splats containing NaN or Infinity values.
 */
type FilterNaN = {
    /** Action type identifier. */
    kind: 'filterNaN';
};

/**
 * Filter splats by comparing a column value.
 *
 * For `opacity`, `scale_0/1/2`, and `f_dc_0/1/2`, the value is specified in user-friendly
 * (transformed) space: linear opacity (0-1), linear scale, and linear color (0-1).
 * The value is automatically converted to raw PLY space before comparison.
 *
 * To compare against raw PLY values directly (without the user-friendly conversion),
 * use the `_raw` suffix (e.g. `opacity_raw`, `scale_0_raw`, `f_dc_0_raw`).
 *
 * If the DataTable has a pending spatial transform and the column is affected by it
 * (position, rotation, scale, or SH columns), the transform is applied (baked in)
 * before comparison. This applies to both regular and `_raw` columns.
 */
type FilterByValue = {
    /** Action type identifier. */
    kind: 'filterByValue';
    /** Name of the column to compare. */
    columnName: string;
    /** Comparison operator. */
    comparator: 'lt' | 'lte' | 'gt' | 'gte' | 'eq' | 'neq';
    /** Value to compare against. */
    value: number;
};

/**
 * Remove spherical harmonic bands above a threshold.
 */
type FilterBands = {
    /** Action type identifier. */
    kind: 'filterBands';
    /** Maximum SH band to keep (0-3). */
    value: 0 | 1 | 2 | 3;
};

/**
 * Keep only splats within a bounding box.
 */
type FilterBox = {
    /** Action type identifier. */
    kind: 'filterBox';
    /** Minimum corner of the box. */
    min: Vec3;
    /** Maximum corner of the box. */
    max: Vec3;
};

/**
 * Keep only splats within a sphere.
 */
type FilterSphere = {
    /** Action type identifier. */
    kind: 'filterSphere';
    /** Center of the sphere. */
    center: Vec3;
    /** Radius of the sphere. */
    radius: number;
};

/**
 * Parameter for .mjs generator modules.
 */
type Param = {
    /** Action type identifier. */
    kind: 'param';
    /** Parameter name. */
    name: string;
    /** Parameter value. */
    value: string;
};

/**
 * Assign a LOD level to all splats.
 */
type Lod = {
    /** Action type identifier. */
    kind: 'lod';
    /** LOD level to assign. */
    value: number;
};

/**
 * Print a statistical summary to the logger.
 */
type Summary = {
    /** Action type identifier. */
    kind: 'summary';
};

/**
 * Reorder splats by Morton code (Z-order curve) for improved spatial locality.
 */
type MortonOrder = {
    /** Action type identifier. */
    kind: 'mortonOrder';
};

/**
 * Simplify splats to a target count using NanoGS progressive pairwise merging.
 *
 * Instead of discarding low-visibility splats, this iteratively merges nearby
 * similar splats into single approximating Gaussians using Mass-Preserving
 * Moment Matching (MPMM), preserving scene structure and appearance.
 */
type Decimate = {
    /** Action type identifier. */
    kind: 'decimate';
    /** Target number of splats to keep, or null for percentage mode. */
    count: number | null;
    /** Percentage of splats to keep (0-100), or null for count mode. */
    percent: number | null;
};

/**
 * Remove Gaussians that don't meaningfully contribute to any solid voxel.
 *
 * GPU-voxelizes the scene at a given resolution, then evaluates each Gaussian's
 * opacity contribution at occupied voxel centers. Discards Gaussians whose
 * contribution is below a minimum threshold at every solid voxel.
 */
type FilterFloaters = {
    /** Action type identifier. */
    kind: 'filterFloaters';
    /** Voxel size in world units. Default: 0.05 */
    voxelResolution?: number;
    /** Opacity threshold for solid voxels. Default: 0.1 */
    opacityCutoff?: number;
    /** Minimum Gaussian contribution at a voxel center to be kept. Default: 1/255 */
    minContribution?: number;
};

/**
 * Filter Gaussians to keep only those in the connected cluster at a seed position.
 *
 * GPU-voxelizes the scene at a coarse resolution, finds the connected component
 * of occupied blocks containing the seed, and keeps only Gaussians whose AABB
 * overlaps that cluster.
 */
type FilterCluster = {
    /** Action type identifier. */
    kind: 'filterCluster';
    /** Voxel size in world units for coarse voxelization. Default: 1.0 */
    voxelResolution?: number;
    /** Seed position for finding the connected component. Default: Vec3(0,0,0) */
    seed?: Vec3;
    /** Opacity threshold for solid voxels. Default: 0.99 */
    opacityCutoff?: number;
};

/**
 * Options for processing actions that require external resources.
 */
type ProcessOptions = {
    /** Function to create a GPU device (required for filterFloaters, filterCluster). */
    createDevice?: DeviceCreator;
};

/**
 * A processing action to apply to splat data.
 *
 * Actions can transform, filter, or analyze the data:
 * - `translate` - Move splats by a Vec3 offset
 * - `rotate` - Rotate splats by Euler angles (degrees)
 * - `scale` - Uniformly scale splats
 * - `filterNaN` - Remove splats with NaN/Inf values
 * - `filterByValue` - Keep splats matching a column condition
 * - `filterBands` - Remove spherical harmonic bands above a threshold
 * - `filterBox` - Keep splats within a bounding box
 * - `filterSphere` - Keep splats within a sphere
 * - `filterFloaters` - Remove splats not contributing to any occupied voxel (GPU)
 * - `filterCluster` - Keep splats in the connected cluster at a seed position (GPU)
 * - `lod` - Assign LOD level to all splats
 * - `summary` - Print statistical summary to logger
 * - `mortonOrder` - Reorder splats by Morton code for spatial locality
 * - `decimate` - Simplify to target count via progressive pairwise merging
 */
type ProcessAction = Translate | Rotate | Scale | FilterNaN | FilterByValue | FilterBands | FilterBox | FilterSphere | FilterFloaters | FilterCluster | Param | Lod | Summary | MortonOrder | Decimate;

const shNames = new Array(45).fill('').map((_, i) => `f_rest_${i}`);

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

const isTransformColumn = (name: string): boolean => transformColumnNames.has(name) || /^f_rest_\d+$/.test(name);

const formatMarkdown = (summary: SummaryData): string => {
    const lines: string[] = [];

    lines.push('# Summary');
    lines.push('');
    lines.push(`**Row Count:** ${summary.rowCount}`);
    lines.push('');

    // Build header and data rows as string arrays
    const headers = ['Column', 'min', 'max', 'median', 'mean', 'stdDev', 'nans', 'infs', 'histogram'];
    const rows: string[][] = [];

    for (const [name, stats] of Object.entries(summary.columns)) {
        const fn = forwardTransforms[name];
        const fmt = (v: number) => String(fn ? +(fn(v).toPrecision(6)) : v);
        rows.push([
            name,
            fmt(stats.min),
            fmt(stats.max),
            fmt(stats.median),
            fmt(stats.mean),
            fmt(stats.stdDev),
            String(stats.nanCount),
            String(stats.infCount),
            stats.histogram
        ]);
    }

    // Calculate max width for each column
    const colWidths = headers.map((header, colIndex) => {
        const dataWidths = rows.map(row => row[colIndex].length);
        return Math.max(header.length, ...dataWidths);
    });

    // Build aligned table
    const padRow = (cells: string[]) => `| ${cells.map((cell, i) => cell.padEnd(colWidths[i])).join(' | ')} |`;

    const separator = `|${colWidths.map(w => '-'.repeat(w + 2)).join('|')}|`;

    lines.push(padRow(headers));
    lines.push(separator);
    for (const row of rows) {
        lines.push(padRow(row));
    }

    return lines.join('\n');
};

const filter = (dataTable: DataTable, predicate: (row: any, rowIndex: number) => boolean): DataTable => {
    const indices = new Uint32Array(dataTable.numRows);
    let index = 0;
    const row = {};

    for (let i = 0; i < dataTable.numRows; i++) {
        dataTable.getRow(i, row);

        if (predicate(row, i)) {
            indices[index++] = i;
        }
    }

    return dataTable.clone({ rows: indices.subarray(0, index) });
};

/**
 * Applies a sequence of processing actions to splat data.
 *
 * Actions are applied in order and can include transformations (translate, rotate, scale),
 * filters (NaN, value, box, sphere, bands), and analysis (summary).
 *
 * @param dataTable - The input splat data.
 * @param processActions - Array of actions to apply in sequence.
 * @param options - Optional resources for GPU-dependent actions (e.g. filterCluster).
 * @returns The processed DataTable (may be a new instance if filtered).
 *
 * @example
 * ```ts
 * import { Vec3 } from 'playcanvas';
 *
 * const processed = await processDataTable(dataTable, [
 *     { kind: 'scale', value: 0.5 },
 *     { kind: 'translate', value: new Vec3(0, 1, 0) },
 *     { kind: 'filterNaN' },
 *     // opacity value is in linear space (0-1), automatically converted to logit for comparison
 *     { kind: 'filterByValue', columnName: 'opacity', comparator: 'gt', value: 0.1 }
 * ]);
 * ```
 */
const processDataTable = async (dataTable: DataTable, processActions: ProcessAction[], options?: ProcessOptions): Promise<DataTable> => {
    let result = dataTable;

    for (let i = 0; i < processActions.length; i++) {
        const processAction = processActions[i];

        switch (processAction.kind) {
            case 'translate':
                result.transform = new Transform(processAction.value).mul(result.transform);
                break;
            case 'rotate':
                result.transform = new Transform().fromEulers(
                    processAction.value.x,
                    processAction.value.y,
                    processAction.value.z
                ).mul(result.transform);
                break;
            case 'scale':
                result.transform = new Transform(undefined, undefined, processAction.value).mul(result.transform);
                break;
            case 'filterNaN': {
                const infOk = new Set(['opacity']);
                const negInfOk = new Set(['scale_0', 'scale_1', 'scale_2']);
                const columnNames = dataTable.columnNames;

                const predicate = (row: any, rowIndex: number) => {
                    for (const key of columnNames) {
                        const value = row[key];
                        if (!isFinite(value)) {
                            if (value === -Infinity && (infOk.has(key) || negInfOk.has(key))) continue;
                            if (value === Infinity && infOk.has(key)) continue;
                            return false;
                        }
                    }
                    return true;
                };
                result = filter(result, predicate);
                break;
            }
            case 'filterByValue': {
                const { comparator } = processAction;
                let { columnName, value } = processAction;

                if (rawColumnMap[columnName]) {
                    columnName = rawColumnMap[columnName];
                } else if (inverseTransforms[columnName]) {
                    value = inverseTransforms[columnName](value);
                }

                if (!result.transform.isIdentity() && isTransformColumn(columnName)) {
                    result = convertToSpace(result, Transform.IDENTITY);
                }

                const Predicates = {
                    'lt': (row: any) => row[columnName] < value,
                    'lte': (row: any) => row[columnName] <= value,
                    'gt': (row: any) => row[columnName] > value,
                    'gte': (row: any) => row[columnName] >= value,
                    'eq': (row: any) => row[columnName] === value,
                    'neq': (row: any) => row[columnName] !== value
                };
                const predicate = Predicates[comparator] ?? ((row: any) => true);
                result = filter(result, predicate);
                break;
            }
            case 'filterBands': {
                const inputBands = { '9': 1, '24': 2, '-1': 3 }[shNames.findIndex(v => !dataTable.hasColumn(v))] ?? 0;
                const outputBands = processAction.value;

                if (outputBands < inputBands) {
                    const inputCoeffs = [0, 3, 8, 15][inputBands];
                    const outputCoeffs = [0, 3, 8, 15][outputBands];

                    const map: any = {};
                    for (let i = 0; i < inputCoeffs; ++i) {
                        for (let j = 0; j < 3; ++j) {
                            const inputName = `f_rest_${i + j * inputCoeffs}`;
                            map[inputName] = i < outputCoeffs ? `f_rest_${i + j * outputCoeffs}` : null;
                        }
                    }

                    result = new DataTable(result.columns.map((column) => {
                        if (map.hasOwnProperty(column.name)) {
                            const name = map[column.name];
                            return name ? new Column(name, column.data) : null;
                        }
                        return column;

                    }).filter(c => c !== null), result.transform);
                }
                break;
            }
            case 'filterBox': {
                const { min, max } = processAction;

                if (result.transform.isIdentity()) {
                    const predicate = (row: any) => {
                        const { x, y, z } = row;
                        return x >= min.x && x <= max.x &&
                               y >= min.y && y <= max.y &&
                               z >= min.z && z <= max.z;
                    };
                    result = filter(result, predicate);
                } else {
                    const { translation, scale } = result.transform;
                    if (scale === 0) {
                        throw new Error('Cannot apply filterBox with scale 0');
                    }
                    const invRot = result.transform.rotation.clone().invert();

                    const axes = [new Vec3(1, 0, 0), new Vec3(0, 1, 0), new Vec3(0, 0, 1)];
                    const rawAxes = axes.map((a) => {
                        const r = new Vec3();
                        invRot.transformVector(a, r);
                        return r;
                    });

                    const minArr = [min.x, min.y, min.z];
                    const maxArr = [max.x, max.y, max.z];
                    const rawMin = new Array(3);
                    const rawMax = new Array(3);
                    for (let j = 0; j < 3; j++) {
                        const dot = axes[j].dot(translation);
                        rawMin[j] = (minArr[j] - dot) / scale;
                        rawMax[j] = (maxArr[j] - dot) / scale;
                    }

                    const predicate = (row: any) => {
                        const { x, y, z } = row;
                        for (let j = 0; j < 3; j++) {
                            const proj = rawAxes[j].x * x + rawAxes[j].y * y + rawAxes[j].z * z;
                            if (proj < rawMin[j] || proj > rawMax[j]) return false;
                        }
                        return true;
                    };
                    result = filter(result, predicate);
                }
                break;
            }
            case 'filterSphere': {
                const rawCenter = processAction.center.clone();
                let rawRadius = processAction.radius;
                if (!result.transform.isIdentity()) {
                    result.transform.clone().invert().transformPoint(rawCenter, rawCenter);
                    rawRadius /= result.transform.scale;
                }
                const radiusSq = rawRadius * rawRadius;
                const predicate = (row: any, rowIndex: number) => {
                    const { x, y, z } = row;
                    return (x - rawCenter.x) ** 2 + (y - rawCenter.y) ** 2 + (z - rawCenter.z) ** 2 < radiusSq;
                };
                result = filter(result, predicate);
                break;
            }
            case 'param': {
                // skip params
                break;
            }
            case 'lod': {
                if (!result.getColumnByName('lod')) {
                    result.addColumn(new Column('lod', new Float32Array(result.numRows)));
                }
                result.getColumnByName('lod').data.fill(processAction.value);
                break;
            }
            case 'summary': {
                const summary = computeSummary(result);
                const markdown = formatMarkdown(summary);
                logger.output(markdown);
                break;
            }
            case 'mortonOrder': {
                const indices = new Uint32Array(result.numRows);
                for (let i = 0; i < indices.length; i++) {
                    indices[i] = i;
                }
                sortMortonOrder(result, indices);
                result.permuteRowsInPlace(indices);
                break;
            }
            case 'decimate': {
                let keepCount: number;
                if (processAction.count !== null) {
                    keepCount = Math.min(processAction.count, result.numRows);
                } else {
                    keepCount = Math.round(result.numRows * (processAction.percent ?? 100) / 100);
                }
                keepCount = Math.max(0, keepCount);

                result = simplifyGaussians(result, keepCount);
                break;
            }
            case 'filterFloaters': {
                if (!options?.createDevice) {
                    throw new Error('filterFloaters requires a createDevice function (GPU voxelization)');
                }
                const { filterFloaters } = await import('./voxel/filter-floaters');
                result = await filterFloaters(
                    result,
                    options.createDevice,
                    processAction.voxelResolution,
                    processAction.opacityCutoff,
                    processAction.minContribution
                );
                break;
            }
            case 'filterCluster': {
                if (!options?.createDevice) {
                    throw new Error('filterCluster requires a createDevice function (GPU voxelization)');
                }
                const { filterCluster } = await import('./voxel/filter-cluster');
                result = await filterCluster(
                    result,
                    options.createDevice,
                    processAction.voxelResolution,
                    processAction.seed,
                    processAction.opacityCutoff
                );
                break;
            }
        }
    }

    return result;
};

export {
    processDataTable,
    type ProcessAction,
    type ProcessOptions,
    type Translate,
    type Rotate,
    type Scale,
    type FilterNaN,
    type FilterByValue,
    type FilterBands,
    type FilterBox,
    type FilterSphere,
    type FilterFloaters,
    type FilterCluster,
    type Param,
    type Lod,
    type Summary,
    type MortonOrder,
    type Decimate
};
