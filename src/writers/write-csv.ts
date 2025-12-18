import { FileHandle } from 'node:fs/promises';

import { DataTable } from '../data-table/data-table';

const writeCsv = async (fileHandle: FileHandle, dataTable: DataTable) => {

    const len = dataTable.numRows;

    // write header
    await fileHandle.write(`${dataTable.columnNames.join(',')}\n`);

    const columns = dataTable.columns.map(c => c.data);

    // write rows
    for (let i = 0; i < len; ++i) {
        let row = '';
        for (let c = 0; c < dataTable.columns.length; ++c) {
            if (c) row += ',';
            row += columns[c][i];
        }
        await fileHandle.write(`${row}\n`);
    }
};

export { writeCsv };
