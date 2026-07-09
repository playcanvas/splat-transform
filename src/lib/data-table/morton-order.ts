import { DataTable } from './data-table';
import { sortMortonColumns } from '../ops/morton-order';
import { logger } from '../utils';

// Sort the provided indices into morton order using the table's x/y/z columns.
// A thin DataTable adapter over the shared columnar core in ops/morton-order.
const sortMortonOrder = (dataTable: DataTable, indices: Uint32Array): void => {
    const xCol = dataTable.getColumnByName('x');
    const yCol = dataTable.getColumnByName('y');
    const zCol = dataTable.getColumnByName('z');

    if (!xCol || !yCol || !zCol) {
        logger.debug('missing required position columns');
        return;
    }

    sortMortonColumns(xCol.data, yCol.data, zCol.data, indices);
};

export { sortMortonOrder };
