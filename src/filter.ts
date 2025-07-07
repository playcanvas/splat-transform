import { DataTable } from './data-table';
import { sigmoid } from './utils/math';

const filter = (dataTable: DataTable, removeInvalid: boolean, removeInvisible: boolean) => {

    const predicate = (rowIndex: number, row: any) => {
        // remove any rows containing NaN or Inf
        if (removeInvalid) {
            for (const key in row) {
                if (!isFinite(row[key])) {
                    return false;
                }
            }
        }

        // filter out very small opacities
        if (removeInvisible && sigmoid(row.opacity) < 1 / 255) {
            return false;
        }

        return true;
    };

    return dataTable.filter(predicate);
};

export { filter };
