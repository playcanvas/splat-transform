import { FileHandle, open } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';

import { version } from '../../package.json';
import { Column, DataTable } from '../data-table';
import { createDevice, enumerateAdapters, GpuDevice } from '../gpu/gpu-device';
import { logger } from '../logger';
import { generateOrdering } from '../ordering';
import { FileWriter } from '../serialize/writer';
import { ZipWriter } from '../serialize/zip-writer';
import { Options } from '../types';
import { kmeans } from '../utils/k-means';
import { sigmoid } from '../utils/math';
import { WebPCodec } from '../utils/webp-codec';

const shNames = new Array(45).fill('').map((_, i) => `f_rest_${i}`);

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
const cluster1d = async (dataTable: DataTable, iterations: number, device?: GpuDevice) => {
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

const writeFile = async (filename: string, data: Uint8Array) => {
    const outputFile = await open(filename, 'w');
    outputFile.write(data);
    await outputFile.close();
};

let webPCodec: WebPCodec;
let gpuDevice: GpuDevice;

const writeSog = async (fileHandle: FileHandle, dataTable: DataTable, outputFilename: string, options: Options, indices = generateIndices(dataTable)) => {
    // initialize output stream
    const isBundle = outputFilename.toLowerCase().endsWith('.sog');
    const fileWriter = isBundle && new FileWriter(fileHandle);
    const zipWriter = fileWriter && new ZipWriter(fileWriter);

    const numRows = indices.length;
    const width = Math.ceil(Math.sqrt(numRows) / 4) * 4;
    const height = Math.ceil(numRows / width / 4) * 4;
    const channels = 4;

    // the layout function determines how the data is packed into the output texture.
    const layout = identity; // rectChunks;

    const writeWebp = async (filename: string, data: Uint8Array, w = width, h = height) => {
        const pathname = resolve(dirname(outputFilename), filename);
        logger.info(`writing '${pathname}'...`);

        // construct the encoder on first use
        if (!webPCodec) {
            webPCodec = await WebPCodec.create();
        }

        const webp = await webPCodec.encodeLosslessRGBA(data, w, h);

        if (zipWriter) {
            await zipWriter.file(filename, webp);
        } else {
            await writeFile(pathname, webp);
        }
    };

    const writeTableData = (filename: string, dataTable: DataTable, w = width, h = height) => {
        const data = new Uint8Array(w * h * channels);
        const columns = dataTable.columns.map(c => c.data);
        const numColumns = columns.length;

        for (let i = 0; i < indices.length; ++i) {
            const idx = indices[i];
            const ti = layout(i, width);
            data[ti * channels + 0] = columns[0][idx];
            data[ti * channels + 1] = numColumns > 1 ? columns[1][idx] : 0;
            data[ti * channels + 2] = numColumns > 2 ? columns[2][idx] : 0;
            data[ti * channels + 3] = numColumns > 3 ? columns[3][idx] : 255;
        }

        return writeWebp(filename, data, w, h);
    };

    const row: any = {};

    const writeMeans = async () => {
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
        await writeWebp('means_l.webp', meansL);
        await writeWebp('means_u.webp', meansU);

        return {
            mins: meansMinMax.map(v => v[0]),
            maxs: meansMinMax.map(v => v[1])
        };
    };

    const writeQuaternions = async () => {
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
        await writeWebp('quats.webp', quats);
    };

    const writeScales = async () => {
        const scaleData = await cluster1d(
            new DataTable(['scale_0', 'scale_1', 'scale_2'].map(name => dataTable.getColumnByName(name))),
            options.iterations,
            gpuDevice
        );

        await writeTableData('scales.webp', scaleData.labels);

        return Array.from(scaleData.centroids.getColumn(0).data);
    };

    const writeColors = async () => {
        const colorData = await cluster1d(
            new DataTable(['f_dc_0', 'f_dc_1', 'f_dc_2'].map(name => dataTable.getColumnByName(name))),
            options.iterations,
            gpuDevice
        );

        // generate and store sigmoid(opacity) [0..1]
        const opacity = dataTable.getColumnByName('opacity').data;
        const opacityData = new Uint8Array(opacity.length);
        for (let i = 0; i < numRows; ++i) {
            opacityData[i] = Math.max(0, Math.min(255, sigmoid(opacity[i]) * 255));
        }
        colorData.labels.addColumn(new Column('opacity', opacityData));

        await writeTableData('sh0.webp', colorData.labels);

        return Array.from(colorData.centroids.getColumn(0).data);
    };

    const writeSH = async (shBands: number) => {
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
        const { centroids, labels } = await kmeans(shDataTable, paletteSize, options.iterations, gpuDevice);

        // construct a codebook for all spherical harmonic coefficients
        const codebook = await cluster1d(centroids, options.iterations, gpuDevice);

        // write centroids
        const centroidsBuf = new Uint8Array(64 * shCoeffs * Math.ceil(centroids.numRows / 64) * channels);
        const centroidsRow: any = {};
        for (let i = 0; i < centroids.numRows; ++i) {
            codebook.labels.getRow(i, centroidsRow);

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
        await writeWebp('shN_centroids.webp', centroidsBuf, 64 * shCoeffs, Math.ceil(centroids.numRows / 64));

        // write labels
        const labelsBuf = new Uint8Array(width * height * channels);
        for (let i = 0; i < indices.length; ++i) {
            const label = labels[indices[i]];
            const ti = layout(i, width);

            labelsBuf[ti * 4 + 0] = 0xff & label;
            labelsBuf[ti * 4 + 1] = 0xff & (label >> 8);
            labelsBuf[ti * 4 + 2] = 0;
            labelsBuf[ti * 4 + 3] = 0xff;
        }
        await writeWebp('shN_labels.webp', labelsBuf);

        return {
            count: paletteSize,
            bands: shBands,
            codebook: Array.from(codebook.centroids.getColumn(0).data),
            files: [
                'shN_centroids.webp',
                'shN_labels.webp'
            ]
        };
    };

    const shBands = { '9': 1, '24': 2, '-1': 3 }[shNames.findIndex(v => !dataTable.hasColumn(v))] ?? 0;

    // convert and write attributes
    const meansMinMax = await writeMeans();
    await writeQuaternions();

    // Initialize GPU device if not using CPU mode
    // device: -1 = auto, -2 = CPU, 0+ = specific GPU index
    if (options.device !== -2 && !gpuDevice) {
        let adapterName: string | undefined;

        if (options.device >= 0) {
            const adapters = await enumerateAdapters();
            const adapter = adapters[options.device];
            if (adapter) {
                adapterName = adapter.name;
            } else {
                logger.warn(`GPU adapter index ${options.device} not found, using default`);
            }
        }

        gpuDevice = await createDevice(adapterName);
    }

    const scalesCodebook = await writeScales();
    const colorsCodebook = await writeColors();
    const shN = shBands > 0 ? await writeSH(shBands) : null;

    // construct meta.json
    const meta: any = {
        version: 2,
        asset: {
            generator: `splat-transform v${version}`
        },
        count: numRows,
        means: {
            mins: meansMinMax.mins,
            maxs: meansMinMax.maxs,
            files: [
                'means_l.webp',
                'means_u.webp'
            ]
        },
        scales: {
            codebook: scalesCodebook,
            files: ['scales.webp']
        },
        quats: {
            files: ['quats.webp']
        },
        sh0: {
            codebook: colorsCodebook,
            files: ['sh0.webp']
        },
        ...(shN ? { shN } : {})
    };

    const metaJson = (new TextEncoder()).encode(JSON.stringify(meta));

    if (zipWriter) {
        await zipWriter.file('meta.json', metaJson);
        await zipWriter.close();
        await fileWriter.close();
    } else {
        await fileHandle.write(metaJson);
    }
};

export { writeSog };
