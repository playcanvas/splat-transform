import { Column, DataTable, TypedArray } from './data-table';

// combine multiple DataTables into a single DataTable instance
const combine = (dataTables: DataTable[]) : DataTable => {
    if (dataTables.length === 1) {
        // nothing to combine
        return dataTables[0];
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

    // make unique list of columns where name and type much match
    const columns = dataTables[0].columns.slice();
    for (let i = 1; i < dataTables.length; ++i) {
        const dataTable = dataTables[i];
        for (let j = 0; j < dataTable.columns.length; ++j) {
            if (!findMatchingColumn(columns, dataTable.columns[j])) {
                columns.push(dataTable.columns[j]);
            }
        }
    }

    // count total number of rows
    const totalRows = dataTables.reduce((sum, dataTable) => sum + dataTable.numRows, 0);

    // construct output dataTable
    const resultColumns = columns.map((column) => {
        const constructor = column.data.constructor as new (length: number) => TypedArray;
        return new Column(column.name, new constructor(totalRows));
    });
    const result = new DataTable(resultColumns);

    // copy data
    let rowOffset = 0;
    for (let i = 0; i < dataTables.length; ++i) {
        const dataTable = dataTables[i];

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
