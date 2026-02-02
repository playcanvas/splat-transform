import { Quat, Vec3 } from 'playcanvas';

import { Column, DataTable } from './data-table/data-table';
import { sortByVisibility } from './data-table/filter-visibility';
import { sortMortonOrder } from './data-table/morton-order';
import { computeSummary, type SummaryData } from './data-table/summary';
import { transform } from './data-table/transform';
import { logger } from './utils/logger';

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
 * Filter splats by visibility score, keeping only the most visible ones.
 *
 * Visibility is computed as: linear_opacity * volume
 * where opacity is converted from logit and scales from log space.
 */
type FilterVisibility = {
    /** Action type identifier. */
    kind: 'filterVisibility';
    /** Target number of splats to keep, or null for percentage mode. */
    count: number | null;
    /** Percentage of splats to keep (0-100), or null for count mode. */
    percent: number | null;
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
 * - `lod` - Assign LOD level to all splats
 * - `summary` - Print statistical summary to logger
 * - `mortonOrder` - Reorder splats by Morton code for spatial locality
 * - `filterVisibility` - Keep only the most visible splats by opacity * volume
 */
type ProcessAction = Translate | Rotate | Scale | FilterNaN | FilterByValue | FilterBands | FilterBox | FilterSphere | Param | Lod | Summary | MortonOrder | FilterVisibility;

const shNames = new Array(45).fill('').map((_, i) => `f_rest_${i}`);

const formatMarkdown = (summary: SummaryData): string => {
    const lines: string[] = [];

    lines.push('# Summary');
    lines.push('');
    lines.push(`**Row Count:** ${summary.rowCount}`);
    lines.push('');

    // Build header and data rows as string arrays
    const headers = ['Column', 'min', 'max', 'median', 'mean', 'stdDev', 'nanCount', 'infCount'];
    const rows: string[][] = [];

    for (const [name, stats] of Object.entries(summary.columns)) {
        rows.push([
            name,
            String(stats.min),
            String(stats.max),
            String(stats.median),
            String(stats.mean),
            String(stats.stdDev),
            String(stats.nanCount),
            String(stats.infCount)
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

    return dataTable.permuteRows(indices.subarray(0, index));
};

/**
 * Applies a sequence of processing actions to splat data.
 *
 * Actions are applied in order and can include transformations (translate, rotate, scale),
 * filters (NaN, value, box, sphere, bands), and analysis (summary).
 *
 * @param dataTable - The input splat data.
 * @param processActions - Array of actions to apply in sequence.
 * @returns The processed DataTable (may be a new instance if filtered).
 *
 * @example
 * ```ts
 * import { Vec3 } from 'playcanvas';
 *
 * const processed = processDataTable(dataTable, [
 *     { kind: 'scale', value: 0.5 },
 *     { kind: 'translate', value: new Vec3(0, 1, 0) },
 *     { kind: 'filterNaN' },
 *     { kind: 'filterByValue', columnName: 'opacity', comparator: 'gt', value: 0.1 }
 * ]);
 * ```
 */
const processDataTable = (dataTable: DataTable, processActions: ProcessAction[]) => {
    let result = dataTable;

    for (let i = 0; i < processActions.length; i++) {
        const processAction = processActions[i];

        switch (processAction.kind) {
            case 'translate':
                transform(result, processAction.value, Quat.IDENTITY, 1);
                break;
            case 'rotate':
                transform(result, Vec3.ZERO, new Quat().setFromEulerAngles(
                    processAction.value.x,
                    processAction.value.y,
                    processAction.value.z
                ), 1);
                break;
            case 'scale':
                transform(result, Vec3.ZERO, Quat.IDENTITY, processAction.value);
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
                const { columnName, comparator, value } = processAction;
                const Predicates = {
                    'lt': (row: any, rowIndex: number) => row[columnName] < value,
                    'lte': (row: any, rowIndex: number) => row[columnName] <= value,
                    'gt': (row: any, rowIndex: number) => row[columnName] > value,
                    'gte': (row: any, rowIndex: number) => row[columnName] >= value,
                    'eq': (row: any, rowIndex: number) => row[columnName] === value,
                    'neq': (row: any, rowIndex: number) => row[columnName] !== value
                };
                const predicate = Predicates[comparator] ?? ((row: any, rowIndex: number) => true);
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

                    }).filter(c => c !== null));
                }
                break;
            }
            case 'filterBox': {
                const { min, max } = processAction;
                const predicate = (row: any, rowIndex: number) => {
                    const { x, y, z } = row;
                    return x >= min.x && x <= max.x && y >= min.y && y <= max.y && z >= min.z && z <= max.z;
                };
                result = filter(result, predicate);
                break;
            }
            case 'filterSphere': {
                const { center, radius } = processAction;
                const radiusSq = radius * radius;
                const predicate = (row: any, rowIndex: number) => {
                    const { x, y, z } = row;
                    return (x - center.x) ** 2 + (y - center.y) ** 2 + (z - center.z) ** 2 < radiusSq;
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
            case 'filterVisibility': {
                const indices = new Uint32Array(result.numRows);
                for (let i = 0; i < indices.length; i++) {
                    indices[i] = i;
                }
                sortByVisibility(result, indices);

                // Determine how many to keep
                let keepCount: number;
                if (processAction.count !== null) {
                    keepCount = Math.min(processAction.count, result.numRows);
                } else {
                    keepCount = Math.round(result.numRows * (processAction.percent ?? 100) / 100);
                }
                keepCount = Math.max(0, keepCount);

                result = result.permuteRows(indices.subarray(0, keepCount));
                break;
            }
        }
    }

    return result;
};

export {
    processDataTable,
    type ProcessAction,
    type Translate,
    type Rotate,
    type Scale,
    type FilterNaN,
    type FilterByValue,
    type FilterBands,
    type FilterBox,
    type FilterSphere,
    type Param,
    type Lod,
    type Summary,
    type MortonOrder,
    type FilterVisibility
};
