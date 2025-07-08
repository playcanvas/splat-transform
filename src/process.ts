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

type ProcessAction = Translate | Rotate | Scale | FilterNaN | FilterByValue | FilterBands;

const shNames = new Array(45).fill('').map((_, i) => `f_rest_${i}`);

// process a data table with standard options
const process = (dataTable: DataTable, processActions: ProcessAction[]) => {
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
                const predicate = (rowIndex: number, row: any) => {
                    for (const key in row) {
                        if (!isFinite(row[key])) {
                            return false;
                        }
                    }
                    return true;
                };
                result = result.filter(predicate);
                break;
            }
            case 'filterByValue': {
                const { columnName, comparator, value } = processAction;
                const Predicates = {
                    'lt': (rowIndex: number, row: any) => row[columnName] < value,
                    'lte': (rowIndex: number, row: any) => row[columnName] <= value,
                    'gt': (rowIndex: number, row: any) => row[columnName] > value,
                    'gte': (rowIndex: number, row: any) => row[columnName] >= value,
                    'eq': (rowIndex: number, row: any) => row[columnName] === value,
                    'neq': (rowIndex: number, row: any) => row[columnName] !== value
                };
                const predicate = Predicates[comparator] ?? ((rowIndex: number, row: any) => true);
                result = result.filter(predicate);
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
        }
    }

    return result;
};

export { ProcessAction, process };
