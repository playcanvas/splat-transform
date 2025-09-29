import { Quat, Vec3 } from 'playcanvas';

import { Column, DataTable } from './data-table';
import { transform } from './transform';

type Translate = {
    kind: 'translate';
    value: Vec3;
};

type Rotate = {
    kind: 'rotate';
    value: Vec3;        // euler angles in degrees
};

type Scale = {
    kind: 'scale';
    value: number;
};

type FilterNaN = {
    kind: 'filterNaN';
};

type FilterByValue = {
    kind: 'filterByValue';
    columnName: string;
    comparator: 'lt' | 'lte' | 'gt' | 'gte' | 'eq' | 'neq';
    value: number;
};

type FilterBands = {
    kind: 'filterBands';
    value: 0 | 1 | 2 | 3;
};

type FilterBox = {
    kind: 'filterBox';
    min: Vec3;
    max: Vec3;
};

type FilterSphere = {
    kind: 'filterSphere';
    center: Vec3;
    radius: number;
};

type Param = {
    kind: 'param';
    name: string;
    value: string;
};

type ProcessAction = Translate | Rotate | Scale | FilterNaN | FilterByValue | FilterBands | FilterBox | FilterSphere | Param;

const shNames = new Array(45).fill('').map((_, i) => `f_rest_${i}`);

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

// process a data table with standard options
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
        }
    }

    return result;
};

export { ProcessAction, processDataTable };
