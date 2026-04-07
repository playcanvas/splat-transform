import { Column, DataTable, TypedArray } from './data-table';
import { convertToSpace, Transform } from './transform';
import { logger } from '../utils/logger';

/**
 * Combines multiple DataTables into a single DataTable.
 *
 * Merges rows from all input tables. Columns are matched by name and type;
 * columns that don't exist in all tables will have undefined values for
 * rows from tables lacking that column.
 *
 * If tables have differing source transforms, all data is first converted
 * to engine coordinate space (identity transform) before combining.
 *
 * @param dataTables - Array of DataTables to combine.
 * @returns A new DataTable containing all rows from all input tables.
 *
 * @example
 * ```ts
 * const combined = combine([tableA, tableB, tableC]);
 * console.log(combined.numRows); // tableA.numRows + tableB.numRows + tableC.numRows
 * ```
 */
const combine = (dataTables: DataTable[]) : DataTable => {
    if (dataTables.length === 1) {
        return dataTables[0];
    }

    // Check if all transforms match the first table's transform
    const refTransform = dataTables[0].transform;
    const allMatch = dataTables.every(dt => dt.transform.equals(refTransform));

    let tables = dataTables;
    let resultTransform = refTransform;

    if (!allMatch) {
        logger.warn('Combining DataTables with different source transforms; converting to engine space.');
        tables = dataTables.map(dt => convertToSpace(dt, Transform.IDENTITY));
        resultTransform = Transform.IDENTITY;
    }

    const findMatchingColumn = (columns: Column[], column: Column) => {
        for (let i = 0; i < columns.length; ++i) {
            if (columns[i].name === column.name &&
                columns[i].dataType === column.dataType) {
                return columns[i];
            }
        }
        return null;
    };

    // make unique list of columns where name and type must match
    const columns = tables[0].columns.slice();
    for (let i = 1; i < tables.length; ++i) {
        const dataTable = tables[i];
        for (let j = 0; j < dataTable.columns.length; ++j) {
            if (!findMatchingColumn(columns, dataTable.columns[j])) {
                columns.push(dataTable.columns[j]);
            }
        }
    }

    // count total number of rows
    const totalRows = tables.reduce((sum, dataTable) => sum + dataTable.numRows, 0);

    // construct output dataTable
    const resultColumns = columns.map((column) => {
        const constructor = column.data.constructor as new (length: number) => TypedArray;
        return new Column(column.name, new constructor(totalRows));
    });
    const result = new DataTable(resultColumns, resultTransform);

    // copy data
    let rowOffset = 0;
    for (let i = 0; i < tables.length; ++i) {
        const dataTable = tables[i];

        for (let j = 0; j < dataTable.columns.length; ++j) {
            const column = dataTable.columns[j];
            const targetColumn = findMatchingColumn(result.columns, column);
            targetColumn.data.set(column.data, rowOffset);
        }

        rowOffset += dataTable.numRows;
    }

    return result;
};

export { combine };
