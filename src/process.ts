import { Quat, Vec3 } from 'playcanvas';

import { DataTable } from './data-table';
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

type FilterColumn = {
    kind: 'filterColumn';
    columnName: string;
    comparator: 'lt' | 'lte' | 'gt' | 'gte' | 'eq' | 'neq';
    value: number;
};

type CustomFilter = {
    kind: 'customFilter';
    jsCallback: (rowIndex: number, row: any) => boolean;
};

type ProcessAction = Translate | Rotate | Scale | FilterNaN | FilterColumn | CustomFilter;

// process a data table with standard options
const process = (dataTable: DataTable, processActions: ProcessAction[]) => {
    let result = dataTable;

    for (let i = 0; i < processActions.length; i++) {
        const ProcessAction = processActions[i];

        switch (ProcessAction.kind) {
            case 'translate':
                transform(result, ProcessAction.value, Quat.IDENTITY, 1);
                break;
            case 'rotate':
                transform(result, Vec3.ZERO, new Quat().setFromEulerAngles(
                    ProcessAction.value.x,
                    ProcessAction.value.y,
                    ProcessAction.value.z
                ), 1);
                break;
            case 'scale':
                transform(result, Vec3.ZERO, Quat.IDENTITY, ProcessAction.value);
                break;
            case 'filterNaN':
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
            case 'filterColumn': {
                const { columnName, comparator, value } = ProcessAction;
                const Predicates = {
                    'lt': (rowIndex: number, row: any) => row[columnName] < value,
                    'lte': (rowIndex: number, row: any) => row[columnName] <= value,
                    'gt':  (rowIndex: number, row: any) => row[columnName] > value,
                    'gte': (rowIndex: number, row: any) => row[columnName] >= value,
                    'eq': (rowIndex: number, row: any) => row[columnName] === value,
                    'neq': (rowIndex: number, row: any) => row[columnName] !== value,
                };
                const predicate = Predicates[comparator] ?? ((rowIndex: number, row: any) => true);
                result = result.filter(predicate);
                break;
            }
            case 'customFilter':
                result = result.filter(ProcessAction.jsCallback);
                break;
        }
    }

    return result;
};

export { ProcessAction, process };
