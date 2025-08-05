import { FileHandle } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';

import sharp from 'sharp';
import * as archiver from 'archiver';

import { DataTable } from '../data-table';
import { createDevice } from '../gpu/gpu-device';
import { generateOrdering } from '../ordering';
import { kmeans } from '../utils/k-means';

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

// pack every 256 indices into a grid of 16x16 chunks
const rectChunks = (index: number, width: number) => {
    const chunkWidth = width / 16;
    const chunkIndex = Math.floor(index / 256);
    const chunkX = chunkIndex % chunkWidth;
    const chunkY = Math.floor(chunkIndex / chunkWidth);

    const x = chunkX * 16 + (index % 16);
    const y = chunkY * 16 + Math.floor((index % 256) / 16);

    return x + y * width;
};

// no packing
const identity = (index: number, width: number) => {
    return index;
};

const writeSog = async (fileHandle: FileHandle, dataTable: DataTable, outputFilename: string, shIterations = 10, shMethod: 'cpu' | 'gpu') => {
    // initialize the zip archive in bundle mode
    const zip = outputFilename.toLocaleLowerCase().endsWith('.sog') && archiver.create('zip', { zlib: { level: 9 } });
    const out = zip && fileHandle.createWriteStream();
    const zipDone = zip && new Promise<void>((resolve, reject) => {
        out.on("close", resolve);
        zip.on("error", reject);
    });
    if (zip) {
        zip.pipe(out);
    }

    // generate an optimal ordering
    const sortIndices = generateOrdering(dataTable);

    const numRows = dataTable.numRows;
    const width = Math.ceil(Math.sqrt(numRows) / 16) * 16;
    const height = Math.ceil(numRows / width / 16) * 16;
    const channels = 4;

    const write = async (filename: string, data: Uint8Array, w = width, h = height) => {
        const pathname = resolve(dirname(outputFilename), filename);
        console.log(`writing '${pathname}'...`);
        const webp = sharp(data, { raw: { width: w, height: h, channels } }).webp({ lossless: true });
        if (zip) {
            zip.append(await webp.toBuffer(), { name: filename });
        } else {
            await webp.toFile(pathname);
        }
    };

    // the layout function determines how the data is packed into the output texture.
    const layout = identity; // rectChunks;

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

        const ti = layout(i, width);

        meansL[ti * 4] = x & 0xff;
        meansL[ti * 4 + 1] = y & 0xff;
        meansL[ti * 4 + 2] = z & 0xff;
        meansL[ti * 4 + 3] = 0xff;

        meansU[ti * 4] = (x >> 8) & 0xff;
        meansU[ti * 4 + 1] = (y >> 8) & 0xff;
        meansU[ti * 4 + 2] = (z >> 8) & 0xff;
        meansU[ti * 4 + 3] = 0xff;
    }
    await write('means_l.webp', meansL);
    await write('means_u.webp', meansU);

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
        q.forEach((v, j) => {
            q[j] = v / l;
        });

        // find max component
        const maxComp = q.reduce((v, _, i) => (Math.abs(q[i]) > Math.abs(q[v]) ? i : v), 0);

        // invert if max component is negative
        if (q[maxComp] < 0) {
            q.forEach((v, j) => {
                q[j] *= -1;
            });
        }

        // scale by sqrt(2) to fit in [-1, 1] range
        const sqrt2 = Math.sqrt(2);
        q.forEach((v, j) => {
            q[j] *= sqrt2;
        });

        const idx = [
            [1, 2, 3],
            [0, 2, 3],
            [0, 1, 3],
            [0, 1, 2]
        ][maxComp];

        const ti = layout(i, width);

        quats[ti * 4]     = 255 * (q[idx[0]] * 0.5 + 0.5);
        quats[ti * 4 + 1] = 255 * (q[idx[1]] * 0.5 + 0.5);
        quats[ti * 4 + 2] = 255 * (q[idx[2]] * 0.5 + 0.5);
        quats[ti * 4 + 3] = 252 + maxComp;
    }
    await write('quats.webp', quats);

    // scales
    const scales = new Uint8Array(width * height * channels);
    const scaleNames = ['scale_0', 'scale_1', 'scale_2'];
    const scaleColumns = scaleNames.map(name => dataTable.getColumnByName(name));
    const scaleMinMax = calcMinMax(dataTable, scaleNames);
    for (let i = 0; i < dataTable.numRows; ++i) {
        dataTable.getRow(sortIndices[i], row, scaleColumns);

        const ti = layout(i, width);

        scales[ti * 4]     = 255 * (row.scale_0 - scaleMinMax[0][0]) / (scaleMinMax[0][1] - scaleMinMax[0][0]);
        scales[ti * 4 + 1] = 255 * (row.scale_1 - scaleMinMax[1][0]) / (scaleMinMax[1][1] - scaleMinMax[1][0]);
        scales[ti * 4 + 2] = 255 * (row.scale_2 - scaleMinMax[2][0]) / (scaleMinMax[2][1] - scaleMinMax[2][0]);
        scales[ti * 4 + 3] = 0xff;
    }
    await write('scales.webp', scales);

    // colors
    const sh0 = new Uint8Array(width * height * channels);
    const sh0Names = ['f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity'];
    const sh0Columns = sh0Names.map(name => dataTable.getColumnByName(name));
    const sh0MinMax = calcMinMax(dataTable, sh0Names);
    for (let i = 0; i < dataTable.numRows; ++i) {
        dataTable.getRow(sortIndices[i], row, sh0Columns);

        const ti = layout(i, width);

        sh0[ti * 4]     = 255 * (row.f_dc_0 - sh0MinMax[0][0]) / (sh0MinMax[0][1] - sh0MinMax[0][0]);
        sh0[ti * 4 + 1] = 255 * (row.f_dc_1 - sh0MinMax[1][0]) / (sh0MinMax[1][1] - sh0MinMax[1][0]);
        sh0[ti * 4 + 2] = 255 * (row.f_dc_2 - sh0MinMax[2][0]) / (sh0MinMax[2][1] - sh0MinMax[2][0]);
        sh0[ti * 4 + 3] = 255 * (row.opacity - sh0MinMax[3][0]) / (sh0MinMax[3][1] - sh0MinMax[3][0]);
    }
    await write('sh0.webp', sh0);

    // write meta.json
    const meta: any = {
        means: {
            shape: [numRows, 3],
            dtype: 'float32',
            mins: meansMinMax.map(v => v[0]),
            maxs: meansMinMax.map(v => v[1]),
            files: [
                'means_l.webp',
                'means_u.webp'
            ]
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

    // spherical harmonics
    const shBands = { '9': 1, '24': 2, '-1': 3 }[shNames.findIndex(v => !dataTable.hasColumn(v))] ?? 0;

    // @ts-ignore
    if (shBands > 0) {
        const shCoeffs = [0, 3, 8, 15][shBands];
        const shColumnNames = shNames.slice(0, shCoeffs * 3);
        const shColumns = shColumnNames.map(name => dataTable.getColumnByName(name));

        // create a table with just spherical harmonics data
        const shDataTable = new DataTable(shColumns);

        const paletteSize = Math.min(64, 2 ** Math.floor(Math.log2(dataTable.numRows / 1024))) * 1024;

        // calculate kmeans
        const gpuDevice = shMethod === 'gpu' ? await createDevice() : null;
        const { centroids, labels } = await kmeans(shDataTable, paletteSize, shIterations, gpuDevice);

        // write centroids
        const centroidsBuf = new Uint8Array(64 * shCoeffs * Math.ceil(centroids.numRows / 64) * channels);
        const centroidsMinMax = calcMinMax(shDataTable, shColumnNames);
        const centroidsMin = centroidsMinMax.map(v => v[0]).reduce((a, b) => Math.min(a, b));
        const centroidsMax = centroidsMinMax.map(v => v[1]).reduce((a, b) => Math.max(a, b));
        const centroidsRow: any = {};
        for (let i = 0; i < centroids.numRows; ++i) {
            centroids.getRow(i, centroidsRow);

            for (let j = 0; j < shCoeffs; ++j) {
                const x = centroidsRow[shColumnNames[shCoeffs * 0 + j]];
                const y = centroidsRow[shColumnNames[shCoeffs * 1 + j]];
                const z = centroidsRow[shColumnNames[shCoeffs * 2 + j]];

                centroidsBuf[i * shCoeffs * 4 + j * 4 + 0] = 255 * ((x - centroidsMin) / (centroidsMax - centroidsMin));
                centroidsBuf[i * shCoeffs * 4 + j * 4 + 1] = 255 * ((y - centroidsMin) / (centroidsMax - centroidsMin));
                centroidsBuf[i * shCoeffs * 4 + j * 4 + 2] = 255 * ((z - centroidsMin) / (centroidsMax - centroidsMin));
                centroidsBuf[i * shCoeffs * 4 + j * 4 + 3] = 0xff;
            }
        }
        await write('shN_centroids.webp', centroidsBuf, 64 * shCoeffs, Math.ceil(centroids.numRows / 64));

        // write labels
        const labelsBuf = new Uint8Array(width * height * channels);
        for (let i = 0; i < dataTable.numRows; ++i) {
            const label = labels[sortIndices[i]];

            const ti = layout(i, width);
            labelsBuf[ti * 4] = label & 0xff;
            labelsBuf[ti * 4 + 1] = (label >> 8) & 0xff;
            labelsBuf[ti * 4 + 2] = 0;
            labelsBuf[ti * 4 + 3] = 0xff;
        }
        await write('shN_labels.webp', labelsBuf);

        meta.shN = {
            shape: [dataTable.numRows, shCoeffs],
            dtype: 'float32',
            mins: centroidsMin,
            maxs: centroidsMax,
            quantization: 8,
            files: [
                'shN_centroids.webp',
                'shN_labels.webp'
            ]
        };
    }

    if (zip) {
        // write the zip bundle to zip
        zip.append(JSON.stringify(meta, null, 4), { name: 'meta.json' });
        await zip.finalize();
        await zipDone;
    } else {
        // write meta.json
        await fileHandle.write((new TextEncoder()).encode(JSON.stringify(meta, null, 4)));
    }
};

export { writeSog };
