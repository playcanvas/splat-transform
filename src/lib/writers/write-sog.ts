import { basename, dirname, resolve } from 'pathe';

import { logWrittenFile } from './utils';
import { Column, DataTable, sortMortonOrder, convertToSpace, getSHBands, shRestNames } from '../data-table';
import { type FileSystem, writeFile, ZipFileSystem } from '../io/write';
import { kmeans } from '../spatial';
import type { DeviceCreator } from '../types';
import { logger, sigmoid, Transform } from '../utils';
import { version } from '../version';
import { runEncodeWebp, runQuantize1d } from '../workers';

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
const identity = (index: number) => {
    return index;
};

const generateIndices = (dataTable: DataTable) => {
    const result = new Uint32Array(dataTable.numRows);
    for (let i = 0; i < result.length; ++i) {
        result[i] = i;
    }
    sortMortonOrder(dataTable, result);
    return result;
};

type WriteSogOptions = {
    filename: string;
    dataTable: DataTable;
    indices?: Uint32Array;
    bundle: boolean;
    iterations: number;
    createDevice?: DeviceCreator;
    // Controls how writeSog reports its own progress. This only affects
    // writeSog's `Writing` group and per-file size lines; nested algorithm
    // logging (e.g. `kmeans()` SH compression progress bars) still goes to
    // the global logger and is not suppressed by 'silent'.
    //   'own'    — open a `Writing` group and emit per-file info lines (default)
    //   'flat'   — no group; emit per-file info lines into caller's scope
    //   'silent' — suppress writeSog's own group/per-file info lines (use
    //              when writing to an in-memory fs and the caller will log
    //              the final bundled size itself)
    logging?: 'own' | 'flat' | 'silent';
};

/**
 * Writes Gaussian splat data to the PlayCanvas SOG format.
 *
 * SOG (Splat Optimized Graphics) uses WebP lossless compression and k-means
 * clustering to achieve high compression ratios. Data is stored in textures
 * for efficient GPU loading.
 *
 * @param options - Options including filename, data, and compression settings.
 * @param fs - File system for writing output files.
 * @ignore
 */
const writeSog = async (options: WriteSogOptions, fs: FileSystem) => {
    const { filename: outputFilename, bundle, iterations, createDevice } = options;
    const logging = options.logging ?? 'own';
    const emitInfo = logging !== 'silent';
    const openGroup = logging === 'own';
    const dataTable = convertToSpace(options.dataTable, Transform.PLY);

    // initialize output stream - use ZipFileSystem for bundled output. The
    // underlying writer's `bytesWritten` is read after close to report the
    // final archive size as a single log entry instead of per-internal-file
    // lines.
    const bundleWriter = bundle ? await fs.createWriter(outputFilename) : null;
    const zipFs = bundleWriter ? new ZipFileSystem(bundleWriter) : null;
    const outputFs = zipFs || fs;

    const indices = options.indices || generateIndices(dataTable);
    const numRows = indices.length;
    const width = Math.ceil(Math.sqrt(numRows) / 4) * 4;
    const height = Math.ceil(numRows / width / 4) * 4;
    const channels = 4;

    // Texture texels follow the provided or generated index order.
    const layout = identity;

    // Encodes run concurrently, but writes are committed in call order: the
    // write step is enqueued onto this chain synchronously (so archive entry
    // order follows call order rather than encode-completion order) and awaits
    // its own encode inside the chained section. ZipFileSystem emits entries
    // contiguously into a single stream, so writes must run one at a time.
    let writeChain: Promise<void> = Promise.resolve();

    const writeWebp = (filename: string, data: Uint8Array, w = width, h = height) => {
        const pathname = zipFs ? filename : resolve(dirname(outputFilename), filename);

        // start the encode now, in parallel; NOTE: data may be transferred to a
        // worker and is unusable afterwards
        const encoded = runEncodeWebp(data, w, h);

        const write = writeChain.then(async () => {
            const webp = await encoded;
            await writeFile(outputFs, pathname, webp);

            // For bundled output the per-file sizes are an internal detail; we
            // report a single bundle size after the archive closes.
            if (emitInfo && !zipFs) {
                logWrittenFile(filename, webp.byteLength);
            }
        });

        // keep the chain usable if this write fails; the failure itself is
        // propagated to the caller below
        writeChain = write.catch(() => {});

        return write;
    };

    const writeTableData = (filename: string, dataTable: DataTable, w = width, h = height) => {
        const data = new Uint8Array(w * h * channels);
        const columns = dataTable.columns.map(c => c.data);
        const numColumns = columns.length;

        for (let i = 0; i < indices.length; ++i) {
            const idx = indices[i];
            const ti = layout(i);
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

            const ti = layout(i);

            meansL[ti * 4] = x & 0xff;
            meansL[ti * 4 + 1] = y & 0xff;
            meansL[ti * 4 + 2] = z & 0xff;
            meansL[ti * 4 + 3] = 0xff;

            meansU[ti * 4] = (x >> 8) & 0xff;
            meansU[ti * 4 + 1] = (y >> 8) & 0xff;
            meansU[ti * 4 + 2] = (z >> 8) & 0xff;
            meansU[ti * 4 + 3] = 0xff;
        }
        await Promise.all([
            writeWebp('means_l.webp', meansL),
            writeWebp('means_u.webp', meansU)
        ]);

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
                q.forEach((_v, j) => {
                    q[j] *= -1;
                });
            }

            // scale by sqrt(2) to fit in [-1, 1] range
            const sqrt2 = Math.sqrt(2);
            q.forEach((_v, j) => {
                q[j] *= sqrt2;
            });

            const idx = [
                [1, 2, 3],
                [0, 2, 3],
                [0, 1, 3],
                [0, 1, 2]
            ][maxComp];

            const ti = layout(i);

            quats[ti * 4]     = 255 * (q[idx[0]] * 0.5 + 0.5);
            quats[ti * 4 + 1] = 255 * (q[idx[1]] * 0.5 + 0.5);
            quats[ti * 4 + 2] = 255 * (q[idx[2]] * 0.5 + 0.5);
            quats[ti * 4 + 3] = 252 + maxComp;
        }
        await writeWebp('quats.webp', quats);
    };

    const writeScales = async () => {
        const scaleData = await runQuantize1d(
            new DataTable(['scale_0', 'scale_1', 'scale_2'].map(name => dataTable.getColumnByName(name)))
        );

        await writeTableData('scales.webp', scaleData.labels);

        return Array.from(scaleData.centroids.getColumn(0).data);
    };

    const writeColors = async () => {
        const colorData = await runQuantize1d(
            new DataTable(['f_dc_0', 'f_dc_1', 'f_dc_2'].map(name => dataTable.getColumnByName(name)))
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
        const shColumnNames = shRestNames.slice(0, shCoeffs * 3);
        const shColumns = shColumnNames.map(name => dataTable.getColumnByName(name));

        // create a table with just spherical harmonics data
        // NOTE: this step should also copy the rows referenced in indices, but that's a
        // lot of duplicate data when it's unneeded (which is currently never). so that
        // means k-means is clustering the full dataset, instead of the rows referenced in
        // indices.
        const shDataTable = new DataTable(shColumns);

        const paletteSize = Math.min(64, 2 ** Math.floor(Math.log2(indices.length / 1024))) * 1024;

        // Create GPU device lazily — only needed for SH k-means clustering
        const gpuDevice = createDevice ? await createDevice() : undefined;

        const { centroids, labels } = await kmeans(shDataTable, paletteSize, iterations, gpuDevice);

        // quantize the centroids while the labels buffer is built below
        const codebookPromise = runQuantize1d(centroids);

        // build labels
        const labelsBuf = new Uint8Array(width * height * channels);
        for (let i = 0; i < indices.length; ++i) {
            const label = labels[indices[i]];
            const ti = layout(i);

            labelsBuf[ti * 4 + 0] = 0xff & label;
            labelsBuf[ti * 4 + 1] = 0xff & (label >> 8);
            labelsBuf[ti * 4 + 2] = 0;
            labelsBuf[ti * 4 + 3] = 0xff;
        }

        const codebook = await codebookPromise;

        // build centroids
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

        await Promise.all([
            writeWebp('shN_centroids.webp', centroidsBuf, 64 * shCoeffs, Math.ceil(centroids.numRows / 64)),
            writeWebp('shN_labels.webp', labelsBuf)
        ]);

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

    const shBands = getSHBands(dataTable);

    const writingGroup = openGroup ? logger.group('Writing') : null;

    try {
        // The texture pipelines are independent, so run them concurrently:
        // quantization and encoding overlap across worker threads (or run
        // inline and serial when workers are unavailable). Quantize-bearing
        // pipelines are started first so workers receive work immediately.
        const pipelines = [
            writeScales(),
            writeColors(),
            shBands > 0 ? writeSH(shBands) : Promise.resolve(null),
            writeMeans(),
            writeQuaternions()
        ] as const;

        // when one pipeline fails, Promise.all rejects immediately while the
        // others are still in flight; mark their later rejections as handled
        // so the original error propagates instead of an unhandled rejection
        pipelines.forEach(p => p.catch(() => {}));

        const [scalesCodebook, colorsCodebook, shN, meansMinMax] = await Promise.all(pipelines);

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

        const metaFilename = zipFs ? 'meta.json' : outputFilename;

        await writeFile(outputFs, metaFilename, metaJson);

        if (emitInfo && !zipFs) {
            logWrittenFile(basename(outputFilename), metaJson.byteLength);
        }

        // Close zip archive if bundling
        if (zipFs) {
            await zipFs.close();
        }

        if (emitInfo && bundleWriter) {
            logWrittenFile(basename(outputFilename), bundleWriter.bytesWritten);
        }

        writingGroup?.end();
    } catch (err) {
        // Best-effort close of the underlying bundle writer to avoid a file
        // handle leak / partially-written zip on failure. Leave `writingGroup`
        // open so the caller's `logger.error()` -> `unwindAll(true)` marks it
        // as failed.
        if (bundleWriter) {
            try {
                await bundleWriter.close();
            } catch {
                // already failing — swallow secondary close errors
            }
        }
        throw err;
    }
};

export { writeSog };
