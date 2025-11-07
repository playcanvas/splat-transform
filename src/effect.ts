import { DataTable } from './data-table';

// This density function multiplies only the smallest and the greatest dimension to stay related to the
// quadratic screen-area which is what is later blended on screen. Calculating the cubic volume did not work well.
const density = (a: number, b: number, c: number) => (a * b + a * c + b * c) / 3;

const sigmoid = (v: number) => 1 / (1 + Math.exp(-v));
const invSigmoid = (v: number) => -1 * Math.log(1 / v - 1);

const blur = (dataTable: DataTable, radius: number, cutOff: number = 0.01) => {
    const hasData = ['scale_0', 'scale_1', 'scale_2', 'opacity'].every(c => dataTable.hasColumn(c));
    if (!hasData) throw new Error('Required fields for blurring missing');

    const row: any = {};
    const indices = new Uint32Array(dataTable.numRows);
    let scale_0, scale_1, scale_2, oldDensity, newOpacity: number;
    let index = 0;

    for (let i = 0; i < dataTable.numRows; ++i) {
        dataTable.getRow(i, row);

        scale_0 = Math.exp(row.scale_0);
        scale_1 = Math.exp(row.scale_1);
        scale_2 = Math.exp(row.scale_2);

        if ((scale_0 + scale_1 + scale_2) > 3 * radius) {
            oldDensity = density(scale_0, scale_1, scale_2);

            scale_0 += radius;
            scale_1 += radius;
            scale_2 += radius;

            newOpacity = sigmoid(row.opacity) * oldDensity / density(scale_0, scale_1, scale_2);

            if (newOpacity >= cutOff) {
                indices[index++] = i;

                row.scale_0 = Math.log(scale_0);
                row.scale_1 = Math.log(scale_1);
                row.scale_2 = Math.log(scale_2);
                row.opacity = invSigmoid(newOpacity);

                dataTable.setRow(i, row);
            }
        }
    }

    return dataTable.permuteRows(indices.subarray(0, index));
};

export { blur };