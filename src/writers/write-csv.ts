import { DataTable } from '../data-table/data-table';

import { Platform } from '../serialize/platform';

type WriteCSVOptions = {
    filename: string;
    dataTable: DataTable;
};

const writeCsv = async (options: WriteCSVOptions, platform: Platform) => {
    const { filename, dataTable } = options;

    const len = dataTable.numRows;

    const textEncoder = new TextEncoder();

    const writer = await platform.createWriter(filename);

    // write header
    await writer.write(textEncoder.encode(`${dataTable.columnNames.join(',')}\n`));

    const columns = dataTable.columns.map(c => c.data);

    // write rows
    for (let i = 0; i < len; ++i) {
        let row = '';
        for (let c = 0; c < dataTable.columns.length; ++c) {
            if (c) row += ',';
            row += columns[c][i];
        }
        await writer.write(textEncoder.encode(`${row}\n`));
    }

    await writer.close();
};

export { writeCsv };
