import { Mat3, Mat4, Quat, Vec3 } from 'playcanvas';

import { DataTable } from './data-table';
import { RotateSH } from './utils/rotate-sh';

const shNames = new Array(45).fill('').map((_, i) => `f_rest_${i}`);

const v = new Vec3();
const q = new Quat();

// apply translation, rotation and scale to a data table
const transform = (dataTable: DataTable, t: Vec3, r: Quat, s: number) => {
    const mat = new Mat4().setTRS(t, r, new Vec3(s, s, s));
    const mat3 = new Mat3().setFromQuat(r);
    const rotateSH = new RotateSH(mat3);

    const hasTranslation = ['x', 'y', 'z'].every(c => dataTable.hasColumn(c));
    const hasRotation = ['rot_0', 'rot_1', 'rot_2', 'rot_3'].every(c => dataTable.hasColumn(c));
    const hasScale = ['scale_0', 'scale_1', 'scale_2'].every(c => dataTable.hasColumn(c));
    const shBands = { '9': 1, '24': 2, '-1': 3 }[shNames.findIndex(v => !dataTable.hasColumn(v))] ?? 0;
    const shCoeffs = new Float32Array([0, 3, 8, 15][shBands]);

    const row: any = {};
    for (let i = 0; i < dataTable.numRows; ++i) {
        dataTable.getRow(i, row);

        if (hasTranslation) {
            v.set(row.x, row.y, row.z);
            mat.transformPoint(v, v);
            row.x = v.x;
            row.y = v.y;
            row.z = v.z;
        }

        if (hasRotation) {
            q.set(row.rot_1, row.rot_2, row.rot_3, row.rot_0).mul2(r, q);
            row.rot_0 = q.w;
            row.rot_1 = q.x;
            row.rot_2 = q.y;
            row.rot_3 = q.z;
        }

        if (hasScale) {
            row.scale_0 = Math.log(Math.exp(row.scale_0) * s);
            row.scale_1 = Math.log(Math.exp(row.scale_1) * s);
            row.scale_2 = Math.log(Math.exp(row.scale_2) * s);
        }

        if (shBands > 0) {
            for (let j = 0; j < 3; ++j) {
                for (let k = 0; k < shCoeffs.length; ++k) {
                    shCoeffs[k] = row[shNames[k + j * shCoeffs.length]];
                }

                rotateSH.apply(shCoeffs);

                for (let k = 0; k < shCoeffs.length; ++k) {
                    row[shNames[k + j * shCoeffs.length]] = shCoeffs[k];
                }
            }
        }

        dataTable.setRow(i, row);
    }
};


export { transform };
