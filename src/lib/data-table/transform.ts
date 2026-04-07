import { Mat3, Mat4, Quat, Vec3 } from 'playcanvas';

import { Column, DataTable, TypedArray } from './data-table';
import { Transform } from '../utils/math';
import { RotateSH } from '../utils/rotate-sh';

const shNames = new Array(45).fill('').map((_, i) => `f_rest_${i}`);

const _v = new Vec3();
const _q = new Quat();

// -- Helpers for on-demand column generation --

/**
 * Computes the delta transform needed to convert raw data from its current
 * coordinate system into the output format's coordinate system.
 *
 * @param transform - The DataTable's current source transform.
 * @param outputFormatTransform - The output format's expected transform.
 * @returns The delta transform to apply to raw data, or null if it is identity.
 */
const computeWriteTransform = (transform: Transform, outputFormatTransform: Transform): Transform | null => {
    const delta = new Transform().invert(outputFormatTransform).mul(transform);
    return delta.isIdentity() ? null : delta;
};

/**
 * Detects how many SH bands (0-3) the DataTable has.
 * @ignore
 */
const detectSHBands = (dataTable: DataTable): number => {
    return ({ '9': 1, '24': 2, '-1': 3 } as Record<string, number>)[String(shNames.findIndex(n => !dataTable.hasColumn(n)))] ?? 0;
};

/**
 * Generates transformed typed arrays for requested columns, applying the given
 * transform. Columns unaffected by the transform return references to the
 * original arrays (zero copy).
 *
 * @param dataTable - The source DataTable.
 * @param columnNames - Which columns to produce.
 * @param delta - The transform to apply. If identity or null, original arrays are returned.
 * @returns A map of column name to typed array.
 */
const transformColumns = (dataTable: DataTable, columnNames: string[], delta: Transform | null): Map<string, TypedArray> => {
    const result = new Map<string, TypedArray>();

    if (!delta || delta.isIdentity()) {
        for (const name of columnNames) {
            const col = dataTable.getColumnByName(name);
            if (col) result.set(name, col.data);
        }
        return result;
    }

    const numRows = dataTable.numRows;
    const mat = new Mat4();
    delta.getMatrix(mat);

    const r = delta.rotation;
    const s = delta.scale;

    // Categorize requested columns
    const posNames = ['x', 'y', 'z'];
    const rotNames = ['rot_0', 'rot_1', 'rot_2', 'rot_3'];
    const scaleNames = ['scale_0', 'scale_1', 'scale_2'];

    const needPos = posNames.every(n => columnNames.includes(n) && dataTable.hasColumn(n));
    const needRot = rotNames.every(n => columnNames.includes(n) && dataTable.hasColumn(n));
    const needScale = scaleNames.some(n => columnNames.includes(n) && dataTable.hasColumn(n)) && s !== 1;

    const shBands = detectSHBands(dataTable);
    const shCoeffsPerChannel = [0, 3, 8, 15][shBands];
    const requestedSH = shBands > 0 && shNames.slice(0, shCoeffsPerChannel * 3).some(n => columnNames.includes(n));

    // Position columns
    if (needPos) {
        const srcX = dataTable.getColumnByName('x')!.data;
        const srcY = dataTable.getColumnByName('y')!.data;
        const srcZ = dataTable.getColumnByName('z')!.data;
        const dstX = new Float32Array(numRows);
        const dstY = new Float32Array(numRows);
        const dstZ = new Float32Array(numRows);

        for (let i = 0; i < numRows; ++i) {
            _v.set(srcX[i], srcY[i], srcZ[i]);
            mat.transformPoint(_v, _v);
            dstX[i] = _v.x;
            dstY[i] = _v.y;
            dstZ[i] = _v.z;
        }

        result.set('x', dstX);
        result.set('y', dstY);
        result.set('z', dstZ);
    }

    // Rotation columns
    if (needRot) {
        const src0 = dataTable.getColumnByName('rot_0')!.data;
        const src1 = dataTable.getColumnByName('rot_1')!.data;
        const src2 = dataTable.getColumnByName('rot_2')!.data;
        const src3 = dataTable.getColumnByName('rot_3')!.data;
        const dst0 = new Float32Array(numRows);
        const dst1 = new Float32Array(numRows);
        const dst2 = new Float32Array(numRows);
        const dst3 = new Float32Array(numRows);

        for (let i = 0; i < numRows; ++i) {
            _q.set(src1[i], src2[i], src3[i], src0[i]).mul2(r, _q);
            dst0[i] = _q.w;
            dst1[i] = _q.x;
            dst2[i] = _q.y;
            dst3[i] = _q.z;
        }

        result.set('rot_0', dst0);
        result.set('rot_1', dst1);
        result.set('rot_2', dst2);
        result.set('rot_3', dst3);
    }

    // Scale columns (only affected when uniform scale != 1)
    if (needScale) {
        const logS = Math.log(s);
        for (const name of scaleNames) {
            if (!columnNames.includes(name) || !dataTable.hasColumn(name)) continue;
            const src = dataTable.getColumnByName(name)!.data;
            const dst = new Float32Array(numRows);
            for (let i = 0; i < numRows; ++i) {
                dst[i] = src[i] + logS;
            }
            result.set(name, dst);
        }
    }

    // SH columns
    if (requestedSH) {
        const mat3 = new Mat3().setFromQuat(r);
        const rotateSH = new RotateSH(mat3);
        const shCoeffs = new Float32Array(shCoeffsPerChannel);

        const shSrc: Float32Array[][] = [];
        const shDst: Float32Array[][] = [];
        for (let j = 0; j < 3; ++j) {
            const src: Float32Array[] = [];
            const dst: Float32Array[] = [];
            for (let k = 0; k < shCoeffsPerChannel; ++k) {
                const name = shNames[k + j * shCoeffsPerChannel];
                src.push(dataTable.getColumnByName(name)!.data as Float32Array);
                dst.push(new Float32Array(numRows));
            }
            shSrc.push(src);
            shDst.push(dst);
        }

        for (let i = 0; i < numRows; ++i) {
            for (let j = 0; j < 3; ++j) {
                for (let k = 0; k < shCoeffsPerChannel; ++k) {
                    shCoeffs[k] = shSrc[j][k][i];
                }
                rotateSH.apply(shCoeffs);
                for (let k = 0; k < shCoeffsPerChannel; ++k) {
                    shDst[j][k][i] = shCoeffs[k];
                }
            }
        }

        for (let j = 0; j < 3; ++j) {
            for (let k = 0; k < shCoeffsPerChannel; ++k) {
                const name = shNames[k + j * shCoeffsPerChannel];
                if (columnNames.includes(name)) {
                    result.set(name, shDst[j][k]);
                }
            }
        }
    }

    // All remaining requested columns: return original array references
    for (const name of columnNames) {
        if (!result.has(name)) {
            const col = dataTable.getColumnByName(name);
            if (col) result.set(name, col.data);
        }
    }

    return result;
};

/**
 * Transforms a point from engine space to raw data space using the inverse
 * of the given source transform.
 *
 * @param t - The source transform (source -> engine).
 * @param point - The point in engine space (modified in-place).
 * @returns The point in raw data space.
 */
const inverseTransformPoint = (t: Transform, point: Vec3): Vec3 => {
    const inv = new Transform().invert(t);
    const mat = new Mat4();
    inv.getMatrix(mat);
    mat.transformPoint(point, point);
    return point;
};

/**
 * Transforms an AABB from engine space to raw data space. The result is
 * a (possibly conservative) AABB that contains the transformed box.
 *
 * @param t - The source transform (source -> engine).
 * @param min - The AABB minimum in engine space (modified in-place to output min).
 * @param max - The AABB maximum in engine space (modified in-place to output max).
 */
const inverseTransformAABB = (t: Transform, min: Vec3, max: Vec3): void => {
    const inv = new Transform().invert(t);
    const mat = new Mat4();
    inv.getMatrix(mat);

    const corners = [
        new Vec3(min.x, min.y, min.z),
        new Vec3(max.x, min.y, min.z),
        new Vec3(min.x, max.y, min.z),
        new Vec3(max.x, max.y, min.z),
        new Vec3(min.x, min.y, max.z),
        new Vec3(max.x, min.y, max.z),
        new Vec3(min.x, max.y, max.z),
        new Vec3(max.x, max.y, max.z)
    ];

    mat.transformPoint(corners[0], corners[0]);
    min.copy(corners[0]);
    max.copy(corners[0]);

    for (let i = 1; i < 8; ++i) {
        mat.transformPoint(corners[i], corners[i]);
        min.min(corners[i]);
        max.max(corners[i]);
    }
};

/**
 * Returns a new DataTable with column data converted to the target coordinate
 * space. If the DataTable is already in that space, returns it unchanged.
 *
 * @param dataTable - The source DataTable.
 * @param targetTransform - The desired coordinate-space transform.
 * @returns A DataTable whose raw data is in the target coordinate space.
 */
const convertToSpace = (dataTable: DataTable, targetTransform: Transform): DataTable => {
    const delta = computeWriteTransform(dataTable.transform, targetTransform);
    if (!delta) return dataTable;
    const allNames = dataTable.columnNames;
    const cols = transformColumns(dataTable, allNames, delta);
    return new DataTable(allNames.map(name => new Column(name, cols.get(name)!)), targetTransform);
};

// -- Legacy in-place transform function --

/**
 * Applies a spatial transformation to splat data in-place.
 *
 * Transforms position, rotation, scale, and spherical harmonics data.
 * The transformation is applied as: scale first, then rotation, then translation.
 *
 * @param dataTable - The DataTable to transform (modified in-place).
 * @param t - Translation vector.
 * @param r - Rotation quaternion.
 * @param s - Uniform scale factor.
 *
 * @example
 * ```ts
 * import { Vec3, Quat } from 'playcanvas';
 *
 * // Scale by 2x, rotate 90° around Y, translate up
 * transform(dataTable, new Vec3(0, 5, 0), new Quat().setFromEulerAngles(0, 90, 0), 2.0);
 * ```
 */
const transform = (dataTable: DataTable, t: Vec3, r: Quat, s: number): void => {
    const mat = new Mat4().setTRS(t, r, new Vec3(s, s, s));
    const mat3 = new Mat3().setFromQuat(r);
    const rotateSH = new RotateSH(mat3);

    const hasTranslation = ['x', 'y', 'z'].every(c => dataTable.hasColumn(c));
    const hasRotation = ['rot_0', 'rot_1', 'rot_2', 'rot_3'].every(c => dataTable.hasColumn(c));
    const hasScale = ['scale_0', 'scale_1', 'scale_2'].every(c => dataTable.hasColumn(c));
    const shBands = detectSHBands(dataTable);
    const shCoeffs = new Float32Array([0, 3, 8, 15][shBands]);

    const row: any = {};
    for (let i = 0; i < dataTable.numRows; ++i) {
        dataTable.getRow(i, row);

        if (hasTranslation) {
            _v.set(row.x, row.y, row.z);
            mat.transformPoint(_v, _v);
            row.x = _v.x;
            row.y = _v.y;
            row.z = _v.z;
        }

        if (hasRotation) {
            _q.set(row.rot_1, row.rot_2, row.rot_3, row.rot_0).mul2(r, _q);
            row.rot_0 = _q.w;
            row.rot_1 = _q.x;
            row.rot_2 = _q.y;
            row.rot_3 = _q.z;
        }

        if (hasScale && s !== 1) {
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

export {
    Transform,
    transform,
    transformColumns,
    computeWriteTransform,
    convertToSpace,
    inverseTransformPoint,
    inverseTransformAABB
};
