import { Mat3, Quat, Vec3 } from 'playcanvas';

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
    const delta = outputFormatTransform.clone().invert().mul(transform);
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
    const r = delta.rotation;
    const s = delta.scale;

    // Categorize requested columns
    const posNames = ['x', 'y', 'z'];
    const rotNames = ['rot_0', 'rot_1', 'rot_2', 'rot_3'];
    const scaleNames = ['scale_0', 'scale_1', 'scale_2'];

    const hasPos = posNames.every(n => dataTable.hasColumn(n));
    const needPos = hasPos && posNames.some(n => columnNames.includes(n));
    const needRot = rotNames.every(n => columnNames.includes(n) && dataTable.hasColumn(n));
    const needScale = scaleNames.some(n => columnNames.includes(n) && dataTable.hasColumn(n)) && s !== 1;

    const shBands = detectSHBands(dataTable);
    const shCoeffsPerChannel = [0, 3, 8, 15][shBands];
    const rotIsIdentity = Math.abs(Math.abs(r.w) - 1) < 1e-6;
    const requestedSH = shBands > 0 && !rotIsIdentity && shNames.slice(0, shCoeffsPerChannel * 3).some(n => columnNames.includes(n));

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
            delta.transformPoint(_v, _v);
            dstX[i] = _v.x;
            dstY[i] = _v.y;
            dstZ[i] = _v.z;
        }

        if (columnNames.includes('x')) result.set('x', dstX);
        if (columnNames.includes('y')) result.set('y', dstY);
        if (columnNames.includes('z')) result.set('z', dstZ);
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
 * Transforms an AABB by the given transform. The result is a
 * (possibly conservative) axis-aligned bounding box that contains
 * the transformed box.
 *
 * @param t - The transform to apply.
 * @param min - The AABB minimum (modified in-place to output min).
 * @param max - The AABB maximum (modified in-place to output max).
 */
const _aabbCorner = new Vec3();
const _aabbResult = new Vec3();

const transformAABB = (t: Transform, min: Vec3, max: Vec3): void => {
    const extents = [min.x, max.x, min.y, max.y, min.z, max.z];

    _aabbCorner.set(extents[0], extents[2], extents[4]);
    t.transformPoint(_aabbCorner, _aabbResult);
    min.copy(_aabbResult);
    max.copy(_aabbResult);

    for (let i = 1; i < 8; ++i) {
        _aabbCorner.set(extents[i & 1], extents[2 + ((i >> 1) & 1)], extents[4 + ((i >> 2) & 1)]);
        t.transformPoint(_aabbCorner, _aabbResult);
        min.min(_aabbResult);
        max.max(_aabbResult);
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

export {
    transformColumns,
    computeWriteTransform,
    convertToSpace,
    transformAABB
};
