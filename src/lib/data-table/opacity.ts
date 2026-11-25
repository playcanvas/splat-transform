import { OPACITY_LINEAR_COLUMN } from '../chunk';
import { opacityToLogit } from '../utils/opacity';

import { Column, DataTable } from './data-table';

/**
 * The table with its `opacity` logit taken from `opacity_linear`, which wins over
 * it, when it has that column (see `opacityToLogit`: opacity above 1 clamps), for
 * writers whose format stores only the logit. Returns the table itself otherwise.
 *
 * @param dataTable - The table to write.
 * @returns A table whose `opacity` agrees with its true opacity, without `opacity_linear`.
 */
const withOpacityLogit = (dataTable: DataTable): DataTable => {
    const linear = dataTable.getColumnByName(OPACITY_LINEAR_COLUMN)?.data;
    if (!linear) return dataTable;
    const opacity = new Float32Array(linear.length);
    for (let i = 0; i < linear.length; i++) opacity[i] = opacityToLogit(linear[i] as number);
    const columns = dataTable.columns.filter((c) => c.name !== 'opacity' && c.name !== OPACITY_LINEAR_COLUMN);
    return new DataTable([...columns, new Column('opacity', opacity)], dataTable.transform);
};

export { withOpacityLogit };
