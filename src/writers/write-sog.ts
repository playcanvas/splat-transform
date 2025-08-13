import { FileHandle } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';

import sharp from 'sharp';

import { Column, DataTable } from '../data-table';
import { createDevice, GpuDevice } from '../gpu/gpu-device';
import { generateOrdering } from '../ordering';
import { kmeans } from '../utils/k-means';

const shNames = new Array(45).fill('').map((_, i) => `f_rest_${i}`);

const shNamesForBand = (numBands: 1 | 2 | 3, band: 1 | 2 | 3) => {
    const step = [3, 8, 15][numBands - 1];
    const base = [0, 3, 8][band - 1];
    const size = [3, 5, 7][band - 1];
    const result = [];
    for (let i = 0; i < 3; ++i) {
        for (let j = 0; j < size; ++j) {
            result.push(shNames[j + base + i * step]);
        }
    }
    return result;
};

const calcMinMax = (dataTable: DataTable, columnNames: string[], indices: Uint32Array) => {
    const columns = columnNames.map(name => dataTable.getColumnByName(name));
    const minMax = columnNames.map(() => [Infinity, -Infinity]);
    const row = {};

    for (let i = 0; i < indices.length; ++i) {
        const r = dataTable.getRow(indices[i], row, columns);

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

const generateIndices = (dataTable: DataTable) => {
    const result = new Uint32Array(dataTable.numRows);
    for (let i = 0; i < result.length; ++i) {
        result[i] = i;
    }
    generateOrdering(dataTable, result);
    return result;
};

// convert a dataTable with multiple columns into a single column 
// calculate 256 clusters using kmeans
// return
//      - the resulting labels in a new datatable having same shape as the input
//      - array of 256 centroids
const codify1d = async (dataTable: DataTable, iterations: number, device?: GpuDevice) => {
    const { numColumns, numRows } = dataTable;

    // construct 1d points from the columns of data
    const data = new Float32Array(numRows * numColumns);
    for (let i = 0; i < numColumns; ++i) {
        data.set(dataTable.getColumn(i).data, i * numRows);
    }

    const src = new DataTable([new Column('data', data)]);

    const { centroids, labels } = await kmeans(src, 256, iterations, device);

    // order centroids smallest to largest
    const centroidsData = centroids.getColumn(0).data;
    const order = centroidsData.map((_, i) => i);
    order.sort((a, b) => centroidsData[a] - centroidsData[b]);

    // reorder centroids
    const tmp = centroidsData.slice();
    for (let i = 0; i < order.length; ++i) {
        centroidsData[i] = tmp[order[i]];
    }

    const invOrder = [];
    for (let i = 0; i < order.length; ++i) {
        invOrder[order[i]] = i;
    }

    // reorder labels
    for (let i = 0; i < labels.length; i++) {
        labels[i] = invOrder[labels[i]];
    }

    const result = new DataTable(dataTable.columnNames.map(name => new Column(name, new Uint8Array(numRows))));
    for (let i = 0; i < numColumns; ++i) {
        result.getColumn(i).data.set((labels as Uint32Array).subarray(i * numRows, (i + 1) * numRows));
    }

    return {
        centroids,
        labels: result
    };
};

const writeSog = async (fileHandle: FileHandle, dataTable: DataTable, outputFilename: string, shIterations = 10, shMethod: 'cpu' | 'gpu', indices = generateIndices(dataTable)) => {
    const numRows = indices.length;
    const width = Math.ceil(Math.sqrt(numRows) / 16) * 16;
    const height = Math.ceil(numRows / width / 16) * 16;
    const channels = 4;

    const write = (filename: string, data: Uint8Array, w = width, h = height) => {
        const pathname = resolve(dirname(outputFilename), filename);
        console.log(`writing '${pathname}'...`);
        return sharp(data, { raw: { width: w, height: h, channels } })
        .webp({ lossless: true })
        .toFile(pathname);
    };

    const writeData = (filename: string, dataTable: DataTable, w = width, h = height) => {
        const data = new Uint8Array(w * h * channels);
        const columns = dataTable.columns.map(c => c.data);
        const numColumns = columns.length;

        for (let i = 0; i < indices.length; ++i) {
            const idx = indices[i];
            data[i * channels + 0] = columns[0][idx];
            data[i * channels + 1] = numColumns > 1 ? columns[1][idx] : 0;
            data[i * channels + 2] = numColumns > 2 ? columns[2][idx] : 0;
            data[i * channels + 3] = numColumns > 3 ? columns[3][idx] : 255;
        }

        return write(filename, data, w, h);
    };

    // the layout function determines how the data is packed into the output texture.
    const layout = identity; // rectChunks;

    const row: any = {};

    // convert position/means
    const meansL = new Uint8Array(width * height * channels);
    const meansU = new Uint8Array(width * height * channels);
    const meansNames = ['x', 'y', 'z'];
    const meansMinMax = calcMinMax(dataTable, meansNames, indices).map(v => v.map(logTransform));
    const meansColumns = meansNames.map(name => dataTable.getColumnByName(name));
    for (let i = 0; i < indices.length; ++i) {
        dataTable.getRow(indices[i], row, meansColumns);

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
    for (let i = 0; i < indices.length; ++i) {
        dataTable.getRow(indices[i], row, quatColumns);

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

    const gpuDevice = shMethod === 'gpu' ? await createDevice() : null;

    // scales
    const scaleData = await codify1d(
        new DataTable(['scale_0', 'scale_1', 'scale_2'].map(name => dataTable.getColumnByName(name))),
        shIterations,
        gpuDevice
    );
    await writeData('scales.webp', scaleData.labels);

    const scaleNames = ['scale_0', 'scale_1', 'scale_2'];
    const scaleMinMax = calcMinMax(dataTable, scaleNames, indices);

    // colors
    const colorData = await codify1d(
        new DataTable(['f_dc_0', 'f_dc_1', 'f_dc_2'].map(name => dataTable.getColumnByName(name))),
        shIterations,
        gpuDevice
    );
    const opacityData = await codify1d(
        new DataTable(['opacity'].map(name => dataTable.getColumnByName(name))),
        shIterations,
        gpuDevice
    );
    colorData.labels.addColumn(opacityData.labels.getColumn(0));
    await writeData('sh0.webp', colorData.labels);

    const sh0Names = ['f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity'];
    const sh0MinMax = calcMinMax(dataTable, sh0Names, indices);

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
            codebook: Array.from(scaleData.centroids.getColumn(0).data),
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
            codebook: Array.from(colorData.centroids.getColumn(0).data),
            opacityCodebook: Array.from(opacityData.centroids.getColumn(0).data),
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
        // NOTE: this step should also copy the rows referenced in indices, but that's a
        // lot of duplicate data when it's unneeded (which is currently never). so that
        // means k-means is clustering the full dataset, instead of the rows referenced in
        // indices.
        const shDataTable = new DataTable(shColumns);

        const paletteSize = Math.min(64, 2 ** Math.floor(Math.log2(indices.length / 1024))) * 1024;

        // calculate kmeans
        const { centroids, labels } = await kmeans(shDataTable, paletteSize, shIterations, gpuDevice);

        // calculate centroid codebook for each band seperately
        const codebook1Cols = shNamesForBand(shBands as 1 | 2 | 3, 1).map(name => centroids.getColumnByName(name));
        const codebook1 = await codify1d(new DataTable(codebook1Cols), shIterations, gpuDevice);

        const codebook2Cols = shNamesForBand(shBands as 1 | 2 | 3, 2).map(name => centroids.getColumnByName(name));
        const codebook2 = shBands > 1 && await codify1d(new DataTable(codebook2Cols), shIterations,gpuDevice);

        const codebook3Cols = shNamesForBand(shBands as 1 | 2 | 3, 3).map(name => centroids.getColumnByName(name));
        const codebook3 = shBands > 2 && await codify1d(new DataTable(codebook3Cols), shIterations, gpuDevice);

        const codebookCols = [
            codebook1.labels.columns,
            codebook2?.labels?.columns ?? [],
            codebook3?.labels?.columns ?? []
        ].filter(a => !!a).flat();
        const codebookCentroids = new DataTable(codebookCols);

        // write centroids
        const centroidsBuf = new Uint8Array(64 * shCoeffs * Math.ceil(centroids.numRows / 64) * channels);
        const centroidsMinMax = calcMinMax(shDataTable, shColumnNames, indices);
        const centroidsMin = centroidsMinMax.map(v => v[0]).reduce((a, b) => Math.min(a, b));
        const centroidsMax = centroidsMinMax.map(v => v[1]).reduce((a, b) => Math.max(a, b));
        const centroidsRow: any = {};
        for (let i = 0; i < centroids.numRows; ++i) {
            codebookCentroids.getRow(i, centroidsRow);

            for (let j = 0; j < shCoeffs; ++j) {
                const x = centroidsRow[shColumnNames[shCoeffs * 0 + j]];
                const y = centroidsRow[shColumnNames[shCoeffs * 1 + j]];
                const z = centroidsRow[shColumnNames[shCoeffs * 2 + j]];

                centroidsBuf[i * shCoeffs * 4 + j * 4 + 0] = x;
                centroidsBuf[i * shCoeffs * 4 + j * 4 + 1] = y;
                centroidsBuf[i * shCoeffs * 4 + j * 4 + 2] = z;
                centroidsBuf[i * shCoeffs * 4 + j * 4 + 3] = 0xff;
            }
        }
        await write('shN_centroids.webp', centroidsBuf, 64 * shCoeffs, Math.ceil(centroids.numRows / 64));

        // write labels
        const labelsBuf = new Uint8Array(width * height * channels);
        for (let i = 0; i < indices.length; ++i) {
            const label = labels[indices[i]];

            const ti = layout(i, width);
            labelsBuf[ti * 4] = label & 0xff;
            labelsBuf[ti * 4 + 1] = (label >> 8) & 0xff;
            labelsBuf[ti * 4 + 2] = 0;
            labelsBuf[ti * 4 + 3] = 0xff;
        }
        await write('shN_labels.webp', labelsBuf);

        meta.shN = {
            shape: [indices.length, shCoeffs],
            dtype: 'float32',
            mins: centroidsMin,
            maxs: centroidsMax,
            quantization: 8,
            codebook1: Array.from(codebook1.centroids.getColumn(0).data),
            codebook2: Array.from(codebook2.centroids.getColumn(0).data),
            codebook3: Array.from(codebook3.centroids.getColumn(0).data),
            files: [
                'shN_centroids.webp',
                'shN_labels.webp'
            ]
        };
    }

    await fileHandle.write((new TextEncoder()).encode(JSON.stringify(meta, null, 4)));
};

export { writeSog };
