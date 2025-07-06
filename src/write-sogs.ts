import { FileHandle } from 'node:fs/promises';

import { DataTable } from './data-table';
import { generateOrdering } from './ordering';
import sharp from 'sharp';

const shNames = new Array(45).fill('').map((_, i) => `f_rest_${i}`);

const calcMinMax = (dataTable: DataTable, columnNames: string[]) => {
    const columns = columnNames.map(name => dataTable.getColumnByName(name));
    const minMax = columnNames.map(() => [Infinity, -Infinity]);
    const row = {};

    for (let i = 0; i < dataTable.numRows; ++i) {
        const r = dataTable.getRow(i, row, columns);

        for (let j = 0; j < columnNames.length; ++j) {
            const value = r[columnNames[j]];
            if (value < minMax[j][0]) minMax[j][0] = value;
            if (value > minMax[j][1]) minMax[j][1] = value;
        }
    }

    return minMax;
};

const logTransform = (value: number) => {
    return Math.sign(value) * Math.log(Math.abs(value) + 1);
};

const writeSogs = async (fileHandle: FileHandle, dataTable: DataTable) => {
    const shBands = { '9': 1, '24': 2, '-1': 3 }[shNames.findIndex(v => !dataTable.hasColumn(v))] ?? 0;
    const outputSHCoeffs = [0, 3, 8, 15][shBands];

    const sortIndices = generateOrdering(dataTable);

    const numRows = dataTable.numRows;
    const width = Math.ceil(Math.sqrt(numRows));
    const height = width;
    const channels = 4;

    const write = (filename: string, data: Uint8Array) => {
        return sharp(data, { raw: { width, height, channels } })
            .webp({ lossless: true })
            .toFile(filename);
    };

    const row: any = {};

    // convert position/means
    const meansL = new Uint8Array(width * height * channels);
    const meansU = new Uint8Array(width * height * channels);
    const meansNames = ['x', 'y', 'z'];
    const meansMinMax = calcMinMax(dataTable, meansNames).map(v => v.map(logTransform));
    const meansColumns = meansNames.map(name => dataTable.getColumnByName(name));
    for (let i = 0; i < dataTable.numRows; ++i) {
        dataTable.getRow(sortIndices[i], row, meansColumns);

        const x = 65535 * (logTransform(row.x) - meansMinMax[0][0]) / (meansMinMax[0][1] - meansMinMax[0][0]);
        const y = 65535 * (logTransform(row.y) - meansMinMax[1][0]) / (meansMinMax[1][1] - meansMinMax[1][0]);
        const z = 65535 * (logTransform(row.z) - meansMinMax[2][0]) / (meansMinMax[2][1] - meansMinMax[2][0]);

        meansL[i * 4] = x & 0xff;
        meansL[i * 4 + 1] = y & 0xff;
        meansL[i * 4 + 2] = z & 0xff;
        meansL[i * 4 + 3] = 0xff;

        meansU[i * 4] = (x >> 8) & 0xff;
        meansU[i * 4 + 1] = (y >> 8) & 0xff;
        meansU[i * 4 + 2] = (z >> 8) & 0xff;
        meansU[i * 4 + 3] = 0xff;
    }
    write('means_l.webp', meansL);
    write('means_u.webp', meansU);

    // convert quaternions
    const quats = new Uint8Array(width * height * channels);
    const quatNames = ['rot_0', 'rot_1', 'rot_2', 'rot_3'];
    const quatColumns = quatNames.map(name => dataTable.getColumnByName(name));
    const q = [0, 0, 0, 0];
    for (let i = 0; i < dataTable.numRows; ++i) {
        dataTable.getRow(sortIndices[i], row, quatColumns);

        q[0] = row.rot_0;
        q[1] = row.rot_1;
        q[2] = row.rot_2;
        q[3] = row.rot_3;

        const l = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);

        // normalize
        q.forEach((v, j) => q[j] = v / l);

        // find max component
        const maxComp = q.reduce((v, _, i) => Math.abs(q[i]) > Math.abs(q[v]) ? i : v, 0);

        // invert if max component is negative
        if (q[maxComp] < 0) {
            q.forEach((v, j) => q[j] *= -1);
        }

        // scale by sqrt(2) to fit in [-1, 1] range
        const sqrt2 = Math.sqrt(2);
        q.forEach((v, j) => q[j] *= sqrt2);

        const idx = [
            [1, 2, 3],
            [0, 2, 3],
            [0, 1, 3],
            [0, 1, 2]
        ][maxComp];

        quats[i * 4]     = 255 * (q[idx[0]] * 0.5 + 0.5);
        quats[i * 4 + 1] = 255 * (q[idx[1]] * 0.5 + 0.5);
        quats[i * 4 + 2] = 255 * (q[idx[2]] * 0.5 + 0.5);
        quats[i * 4 + 3] = 252 + maxComp;
    }
    write('quats.webp', quats);

    // scales
    const scales = new Uint8Array(width * height * channels);
    const scaleNames = ['scale_0', 'scale_1', 'scale_2'];
    const scaleColumns = scaleNames.map(name => dataTable.getColumnByName(name));
    const scaleMinMax = calcMinMax(dataTable, scaleNames);
    for (let i = 0; i < dataTable.numRows; ++i) {
        dataTable.getRow(sortIndices[i], row, scaleColumns);

        scales[i * 4]     = 255 * (row.scale_0 - scaleMinMax[0][0]) / (scaleMinMax[0][1] - scaleMinMax[0][0]);
        scales[i * 4 + 1] = 255 * (row.scale_1 - scaleMinMax[1][0]) / (scaleMinMax[1][1] - scaleMinMax[1][0]);
        scales[i * 4 + 2] = 255 * (row.scale_2 - scaleMinMax[2][0]) / (scaleMinMax[2][1] - scaleMinMax[2][0]);
        scales[i * 4 + 3] = 0xff;
    }
    write('scales.webp', scales);

    // colors
    const sh0 = new Uint8Array(width * height * channels);
    const sh0Names = ['f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity'];
    const sh0Columns = sh0Names.map(name => dataTable.getColumnByName(name));
    const sh0MinMax = calcMinMax(dataTable, sh0Names);
    for (let i = 0; i < dataTable.numRows; ++i) {
        dataTable.getRow(sortIndices[i], row, sh0Columns);

        sh0[i * 4]     = 255 * (row.f_dc_0 - sh0MinMax[0][0]) / (sh0MinMax[0][1] - sh0MinMax[0][0]);
        sh0[i * 4 + 1] = 255 * (row.f_dc_1 - sh0MinMax[1][0]) / (sh0MinMax[1][1] - sh0MinMax[1][0]);
        sh0[i * 4 + 2] = 255 * (row.f_dc_2 - sh0MinMax[2][0]) / (sh0MinMax[2][1] - sh0MinMax[2][0]);
        sh0[i * 4 + 3] = 255 * (row.opacity - sh0MinMax[3][0]) / (sh0MinMax[3][1] - sh0MinMax[3][0]);
    }
    write('sh0.webp', sh0);

    // write meta.json
    const meta = {
        means: {
            shape: [numRows, 3],
            dtype: 'float32',
            mins: meansMinMax.map(v => v[0]),
            maxs: meansMinMax.map(v => v[1]),
            files: [
                'means_l.webp',
                'means_u.webp'
            ],
        },
        scales: {
            shape: [numRows, 3],
            dtype: 'float32',
            mins: scaleMinMax.map(v => v[0]),
            maxs: scaleMinMax.map(v => v[1]),
            files: ['scales.webp']
        },
        quats: {
            shape: [numRows, 4],
            dtype: 'uint8',
            encoding: 'quaternion_packed',
            files: ['quats.webp']
        },
        sh0: {
            shape: [numRows, 1, 4],
            dtype: 'float32',
            mins: sh0MinMax.map(v => v[0]),
            maxs: sh0MinMax.map(v => v[1]),
            files: ['sh0.webp']
        }
    };

    fileHandle.write((new TextEncoder()).encode(JSON.stringify(meta, null, 4)));
};

export { writeSogs };
