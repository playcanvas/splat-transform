import { basename, dirname, resolve } from 'pathe';

import { logWrittenFile } from './utils';
import {
    createChunkDataPool,
    type ChunkDataPool,
    type ChunkSource
} from '../chunk';
import { dataTableToChunkSource } from '../compat/data-table';
import { type DataTable, shRestNames } from '../data-table';
import { type FileSystem, writeFile, ZipFileSystem } from '../io/write';
import { bakeTransform } from '../ops';
import { sortMortonInterleaved } from '../ops/morton-order';
import { kmeansInterleaved } from '../spatial';
import { type SplatModel } from '../splat-model';
import type { DeviceCreator } from '../types';
import { logger, sigmoid, Transform } from '../utils';
import { version } from '../version';
import { runEncodeWebp, runQuantize1dColumns } from '../workers';

const GEOMETRIC_COLS = ['rot_0', 'rot_1', 'rot_2', 'rot_3', 'scale_0', 'scale_1', 'scale_2', 'opacity'];

const logTransform = (value: number): number => {
    return Math.sign(value) * Math.log(Math.abs(value) + 1);
};

type SogLayers = {
    /** Interleaved xyz, for the Morton sort and the means textures. */
    position: Float32Array;
    /** One column per `GEOMETRIC_COLS` entry, for quantize1d and the quaternion pack. */
    geometric: Float32Array[];
    /** `f_dc_0..2` columns, for quantize1d. */
    colorDc: Float32Array[];
    /** Interleaved `f_rest_*`, for k-means (empty at 0 bands). */
    shRest: Float32Array;
};

// Gather every SOG input layer with one source.read() per chunk, so an
// interleaved file source fetches and scans each record once rather than once
// per layer. The whole scene is resident afterwards: fine for single SOG
// output, whose practical ceiling is ~1-2M gaussians (larger scenes are written
// as many small units by the LOD writer), and it lets every texture pipeline
// start immediately instead of waiting on a per-layer read.
const gatherSogLayers = async (source: ChunkSource, pool: ChunkDataPool): Promise<SogLayers> => {
    const { meta } = source;
    const n = meta.numGaussians;
    const positionLayout = meta.layouts.position!;
    const geometricLayout = meta.layouts.geometric!;
    const colorLayout = meta.layouts.color!;
    const positionStride = positionLayout.stride >>> 2;
    const geometricStride = geometricLayout.stride >>> 2;
    const colorStride = colorLayout.stride >>> 2;
    const restCount = [0, 9, 24, 45][meta.shBands];

    const position = new Float32Array(n * positionStride);
    const geometric = GEOMETRIC_COLS.map(() => new Float32Array(n));
    const colorDc = Array.from({ length: 3 }, () => new Float32Array(n));
    const shRest = new Float32Array(n * restCount);

    let base = 0;
    const numChunks = meta.numChunks[0] ?? 0;
    for (let k = 0; k < numChunks; k++) {
        const count = Math.min(meta.chunkSize, n - base);
        const positionData = pool.acquire('position', positionLayout, count);
        const geometricData = pool.acquire('geometric', geometricLayout, count);
        const colorData = pool.acquire('color', colorLayout, count);
        try {
            await source.read({
                chunkIndex: k,
                position: positionData,
                geometric: geometricData,
                color: colorData
            });

            const pos = new Float32Array(positionData.data, 0, count * positionStride);
            position.set(pos, base * positionStride);

            const geo = new Float32Array(geometricData.data, 0, count * geometricStride);
            for (let c = 0; c < geometric.length; c++) {
                const column = geometric[c];
                for (let i = 0; i < count; i++) column[base + i] = geo[i * geometricStride + c];
            }

            // f_rest occupies words [3, 3+restCount) of each record, so it copies
            // out as one block per gaussian - no de/re-interleave.
            const color = new Float32Array(colorData.data, 0, count * colorStride);
            for (let i = 0; i < count; i++) {
                const offset = i * colorStride;
                colorDc[0][base + i] = color[offset];
                colorDc[1][base + i] = color[offset + 1];
                colorDc[2][base + i] = color[offset + 2];
                if (restCount > 0) {
                    shRest.set(color.subarray(offset + 3, offset + 3 + restCount), (base + i) * restCount);
                }
            }
        } finally {
            positionData.release();
            geometricData.release();
            colorData.release();
        }
        base += count;
    }

    return { position, geometric, colorDc, shRest };
};

type WriteSogSourceOptions = {
    filename: string;
    bundle: boolean;
    iterations: number;
    /** Lossless WebP compression effort, 0–9. Omit to use the default WebP encoder. */
    webpEffort?: number;
    createDevice?: DeviceCreator;
    logging?: 'own' | 'flat' | 'silent';
    // Optional pre-computed gaussian ordering (texel placement order): a
    // **full-length permutation** of `[0, meta.numGaussians)`. When supplied the
    // internal Morton sort is skipped and this order is used as-is. It is an
    // ordering, NOT a subset filter — to write a subset, filter the source
    // upstream (`filterSource` / `gatherRows`) so the source *is* the subset,
    // then pass its within-source ordering here.
    indices?: Uint32Array;
};

type ShNMeta = { count: number; bands: number; codebook: number[]; files: string[] };

/**
 * Native SOG writer: encodes a {@link ChunkSource} to the PlayCanvas SOG format.
 *
 * The source is gathered once, every layer in one read per chunk, so an
 * interleaved file input is scanned a single time. The gathered scene stays
 * resident for the duration of the write — position, geometric and color
 * together — which is the right trade for single SOG output (bounded by the
 * ~1-2M gaussian practical ceiling below; larger scenes go through the LOD
 * writer as many small units).
 *
 * The texture pipelines are independent, so the worker-side jobs (scale and
 * color quantization, then SH k-means) are started before the main thread does
 * its own Morton sort and texel packing, and WebP encodes are queued as each
 * texture is ready. The pool is busy from the outset and the wall-clock is the
 * longest pipeline rather than the sum.
 *
 * Output is equivalent to the legacy DataTable `writeSog` (same Morton order,
 * quantization/clustering, texel encoding), and byte-identical for the per-file
 * (non-bundled) outputs. Everything works on raw typed-array columns / interleaved
 * buffers (no DataTable): `runQuantize1dColumns` / `kmeansInterleaved` /
 * `runEncodeWebp` consume the gathered layers directly.
 *
 * @param source - The source to encode (its pending transform is baked to PLY space).
 * @param pool - Pool for the temporary per-chunk read buffers.
 * @param options - Output options.
 * @param fs - File system to write through.
 * @ignore
 */
const writeSogSource = async (
    source: ChunkSource,
    pool: ChunkDataPool,
    options: WriteSogSourceOptions,
    fs: FileSystem
): Promise<void> => {
    const { filename: outputFilename, bundle, iterations, webpEffort, createDevice } = options;
    const logging = options.logging ?? 'own';
    const emitInfo = logging !== 'silent';
    const openGroup = logging === 'own';

    const baked = bakeTransform(source, Transform.PLY);
    const { meta } = baked;
    const numRows = meta.numGaussians;
    const shBands = meta.shBands;

    // `indices`, when supplied, is a full-length ordering (permutation) of
    // `[0, numRows)`, not a subset filter — a short array would mis-size the
    // textures / `meta.count` and read past the order in the per-texel loops.
    // Validate before any writer is opened. To write a subset, filter the source
    // upstream (`filterSource`) so the source itself is the subset.
    if (options.indices && options.indices.length !== numRows) {
        throw new Error(
            `writeSogSource: indices length ${options.indices.length} must equal the source's gaussian count ${numRows} ` +
            '(indices is a full-length ordering, not a subset filter — filter the source upstream with filterSource)'
        );
    }

    const width = Math.ceil(Math.sqrt(numRows) / 4) * 4;
    const height = Math.ceil(numRows / width / 4) * 4;
    const channels = 4;

    // Hard failure point only: WebP's 16383-texel dimension ceiling. The
    // practical threshold is far lower — beyond ~1-2M gaussians a scene should
    // be written as streamed SOG (lod-meta.json output), not because of size
    // but because the runtime then gets chunked frustum culling, much faster
    // startup, and LOD rendering. Fail before any output is opened.
    if (width > 16383 || height > 16383) {
        throw new Error(
            `SOG output is capped at 16383x16383 WebP texels (~268M gaussians); got ${numRows}. ` +
            'Write streamed SOG (lod-meta.json output) instead — recommended for any scene beyond ~1-2M gaussians.'
        );
    }

    const layers = await gatherSogLayers(baked, pool);

    const bundleWriter = bundle ? await fs.createWriter(outputFilename) : null;
    const zipFs = bundleWriter ? new ZipFileSystem(bundleWriter) : null;
    const outputFs = zipFs || fs;

    // Writes are committed in call order (zip entries must be contiguous);
    // encodes run concurrently and each awaits inside its chained section.
    let writeChain: Promise<void> = Promise.resolve();

    const writeWebp = (filename: string, data: Uint8Array, w = width, h = height): Promise<void> => {
        const pathname = zipFs ? filename : resolve(dirname(outputFilename), filename);
        const encoded = runEncodeWebp(data, w, h, webpEffort);
        const write = writeChain.then(async () => {
            const webp = await encoded;
            await writeFile(outputFs, pathname, webp);
            if (emitInfo && !zipFs) {
                logWrittenFile(filename, webp.byteLength);
            }
        });
        writeChain = write.catch(() => {});
        return write;
    };

    // Scatter quantize1d label columns (per-gaussian codebook indices) to a webp:
    // texel i receives cols[*][indices[i]].
    const writeLabels = (filename: string, cols: Uint8Array[], indices: Uint32Array): Promise<void> => {
        const data = new Uint8Array(width * height * channels);
        const nc = cols.length;
        for (let i = 0; i < indices.length; ++i) {
            const idx = indices[i];
            const ti = i;
            data[ti * channels + 0] = cols[0][idx];
            data[ti * channels + 1] = nc > 1 ? cols[1][idx] : 0;
            data[ti * channels + 2] = nc > 2 ? cols[2][idx] : 0;
            data[ti * channels + 3] = nc > 3 ? cols[3][idx] : 255;
        }
        return writeWebp(filename, data);
    };

    const writingGroup = openGroup ? logger.group('Writing') : null;
    const pending: Promise<void>[] = [];
    const externalOrder = options.indices;
    const indices = externalOrder ?? new Uint32Array(numRows);
    if (!externalOrder) for (let i = 0; i < numRows; i++) indices[i] = i;

    try {
        // ---- Worker-side jobs first, so the pool is busy while the main thread
        // runs its own passes below. Quantize-bearing textures go first (each
        // is a full-column pass); SH k-means mostly waits on the GPU.
        const [r0, r1, r2, r3, s0, s1, s2, op] = layers.geometric;
        const [fdc0, fdc1, fdc2] = layers.colorDc;
        const scalesQuant = runQuantize1dColumns([
            { name: 'scale_0', data: s0 }, { name: 'scale_1', data: s1 }, { name: 'scale_2', data: s2 }
        ]);
        const colorsQuant = runQuantize1dColumns([
            { name: 'f_dc_0', data: fdc0 }, { name: 'f_dc_1', data: fdc1 }, { name: 'f_dc_2', data: fdc2 }
        ]);
        const restCount = [0, 9, 24, 45][shBands];
        const paletteSize = Math.min(64, 2 ** Math.floor(Math.log2(numRows / 1024))) * 1024;
        const shCluster = shBands > 0 ? (async () => {
            const gpuDevice = createDevice ? await createDevice() : undefined;
            return kmeansInterleaved(layers.shRest, numRows, restCount, paletteSize, iterations, gpuDevice);
        })() : null;
        // If the main thread throws below, these settle later; mark their
        // rejections handled so the original error propagates instead of an
        // unhandled rejection.
        [scalesQuant, colorsQuant, shCluster].forEach(p => p?.catch(() => {}));

        // ---- means: Morton order (unless a caller-supplied order is used) +
        // log-encoded positions split into low/high bytes.
        const meansMeta = (() => {
            const pos = layers.position;
            if (!externalOrder) sortMortonInterleaved(pos, indices);

            const mm = [[Infinity, -Infinity], [Infinity, -Infinity], [Infinity, -Infinity]];
            for (let g = 0; g < numRows; g++) {
                for (let a = 0; a < 3; a++) {
                    const v = pos[g * 3 + a];
                    if (v < mm[a][0]) mm[a][0] = v;
                    if (v > mm[a][1]) mm[a][1] = v;
                }
            }
            const minMax = mm.map(v => v.map(logTransform));
            const meansL = new Uint8Array(width * height * channels);
            const meansU = new Uint8Array(width * height * channels);
            for (let i = 0; i < numRows; ++i) {
                const g = indices[i];
                const x = 65535 * (logTransform(pos[g * 3 + 0]) - minMax[0][0]) / (minMax[0][1] - minMax[0][0]);
                const y = 65535 * (logTransform(pos[g * 3 + 1]) - minMax[1][0]) / (minMax[1][1] - minMax[1][0]);
                const z = 65535 * (logTransform(pos[g * 3 + 2]) - minMax[2][0]) / (minMax[2][1] - minMax[2][0]);
                const ti = i;
                meansL[ti * 4] = x & 0xff;
                meansL[ti * 4 + 1] = y & 0xff;
                meansL[ti * 4 + 2] = z & 0xff;
                meansL[ti * 4 + 3] = 0xff;
                meansU[ti * 4] = (x >> 8) & 0xff;
                meansU[ti * 4 + 1] = (y >> 8) & 0xff;
                meansU[ti * 4 + 2] = (z >> 8) & 0xff;
                meansU[ti * 4 + 3] = 0xff;
            }
            pending.push(writeWebp('means_l.webp', meansL), writeWebp('means_u.webp', meansU));
            return { mins: minMax.map(v => v[0]), maxs: minMax.map(v => v[1]) };
        })();

        // ---- quats: largest-3 packed quaternions.
        {
            const quats = new Uint8Array(width * height * channels);
            const q = [0, 0, 0, 0];
            const sqrt2 = Math.sqrt(2);
            // Largest-3 component orders, indexed by the dropped component.
            const quatIdx = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];
            for (let i = 0; i < numRows; ++i) {
                const g = indices[i];
                q[0] = r0[g]; q[1] = r1[g]; q[2] = r2[g]; q[3] = r3[g];
                const l = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
                q[0] /= l; q[1] /= l; q[2] /= l; q[3] /= l;
                let maxComp = 0;
                if (Math.abs(q[1]) > Math.abs(q[maxComp])) maxComp = 1;
                if (Math.abs(q[2]) > Math.abs(q[maxComp])) maxComp = 2;
                if (Math.abs(q[3]) > Math.abs(q[maxComp])) maxComp = 3;
                const s = (q[maxComp] < 0 ? -1 : 1) * sqrt2;
                q[0] *= s; q[1] *= s; q[2] *= s; q[3] *= s;
                const idx = quatIdx[maxComp];
                const ti = i;
                quats[ti * 4]     = 255 * (q[idx[0]] * 0.5 + 0.5);
                quats[ti * 4 + 1] = 255 * (q[idx[1]] * 0.5 + 0.5);
                quats[ti * 4 + 2] = 255 * (q[idx[2]] * 0.5 + 0.5);
                quats[ti * 4 + 3] = 252 + maxComp;
            }
            pending.push(writeWebp('quats.webp', quats));
        }

        // ---- scales: quantized log-scales.
        const sd = await scalesQuant;
        pending.push(writeLabels('scales.webp', sd.labels.map(c => c.data), indices));
        const scalesCodebook = Array.from(sd.centroids);

        // ---- sh0: quantized DC + sigmoid(opacity).
        const opacityData = new Uint8Array(numRows);
        for (let i = 0; i < numRows; ++i) {
            opacityData[i] = Math.max(0, Math.min(255, sigmoid(op[i]) * 255));
        }
        const cd = await colorsQuant;
        pending.push(writeLabels('sh0.webp', [...cd.labels.map(c => c.data), opacityData], indices));
        const colorsCodebook = Array.from(cd.centroids);

        // ---- shN: k-means palette + per-gaussian labels.
        let shN: ShNMeta | null = null;
        if (shCluster) {
            const shCoeffs = [0, 3, 8, 15][shBands];
            const { centroids, labels } = await shCluster;
            const numCentroids = centroids.length / restCount;

            // quantize the centroid palette to a uint8 codebook. De-interleave
            // the (small) centroids into restCount columns for the quantizer.
            const cbCols: { name: string, data: Float32Array }[] = [];
            for (let j = 0; j < restCount; ++j) {
                const col = new Float32Array(numCentroids);
                for (let i = 0; i < numCentroids; ++i) col[i] = centroids[i * restCount + j];
                cbCols.push({ name: shRestNames[j], data: col });
            }
            const codebookPromise = runQuantize1dColumns(cbCols);

            const labelsBuf = new Uint8Array(width * height * channels);
            for (let i = 0; i < numRows; ++i) {
                const label = labels[indices[i]];
                const ti = i;
                labelsBuf[ti * 4 + 0] = 0xff & label;
                labelsBuf[ti * 4 + 1] = 0xff & (label >> 8);
                labelsBuf[ti * 4 + 2] = 0;
                labelsBuf[ti * 4 + 3] = 0xff;
            }

            const cb = await codebookPromise;
            const cbLabels = cb.labels.map(c => c.data); // restCount columns, length numCentroids
            const centroidsBuf = new Uint8Array(64 * shCoeffs * Math.ceil(numCentroids / 64) * channels);
            for (let i = 0; i < numCentroids; ++i) {
                for (let j = 0; j < shCoeffs; ++j) {
                    centroidsBuf[i * shCoeffs * 4 + j * 4 + 0] = cbLabels[shCoeffs * 0 + j][i];
                    centroidsBuf[i * shCoeffs * 4 + j * 4 + 1] = cbLabels[shCoeffs * 1 + j][i];
                    centroidsBuf[i * shCoeffs * 4 + j * 4 + 2] = cbLabels[shCoeffs * 2 + j][i];
                    centroidsBuf[i * shCoeffs * 4 + j * 4 + 3] = 0xff;
                }
            }
            pending.push(
                writeWebp('shN_centroids.webp', centroidsBuf, 64 * shCoeffs, Math.ceil(numCentroids / 64)),
                writeWebp('shN_labels.webp', labelsBuf)
            );
            shN = {
                count: paletteSize,
                bands: shBands,
                codebook: Array.from(cb.centroids),
                files: ['shN_centroids.webp', 'shN_labels.webp']
            };
        }

        await Promise.all(pending);

        // ---- meta.json --------------------------------------------------
        const metaObj: any = {
            version: 2,
            asset: { generator: `splat-transform v${version}` },
            count: numRows,
            // untagged scenes stay byte-identical to pre-3.2 output
            ...(meta.model === 'default' ? {} : { model: meta.model }),
            means: { mins: meansMeta.mins, maxs: meansMeta.maxs, files: ['means_l.webp', 'means_u.webp'] },
            scales: { codebook: scalesCodebook, files: ['scales.webp'] },
            quats: { files: ['quats.webp'] },
            sh0: { codebook: colorsCodebook, files: ['sh0.webp'] },
            ...(shN ? { shN } : {}),
            ...(meta.camera ? { camera: meta.camera } : {})
        };
        const metaJson = (new TextEncoder()).encode(JSON.stringify(metaObj));
        const metaFilename = zipFs ? 'meta.json' : outputFilename;
        await writeFile(outputFs, metaFilename, metaJson);
        if (emitInfo && !zipFs) {
            logWrittenFile(basename(outputFilename), metaJson.byteLength);
        }

        if (zipFs) {
            await zipFs.close();
        }
        if (emitInfo && bundleWriter) {
            logWrittenFile(basename(outputFilename), bundleWriter.bytesWritten);
        }
        writingGroup?.end();
    } catch (err) {
        if (bundleWriter) {
            try {
                // discard rather than close: close() commits the temp file to
                // the destination, publishing a truncated bundle
                await bundleWriter.abort();
            } catch {
                // already failing — swallow secondary abort errors
            }
        }
        throw err;
    }
};

type WriteSogOptions = WriteSogSourceOptions & { dataTable: DataTable; model?: SplatModel };

/**
 * DataTable-input adapter over {@link writeSogSource}, for callers that still
 * hold a whole-scene DataTable (the legacy writers/tests). Wraps the table as a
 * resident {@link ChunkSource} via the migration shim and encodes it through the
 * same path. The chunk-native `writeSogSource` is preferred for new code.
 *
 * @param options - Output options plus the `dataTable` to encode.
 * @param fs - File system to write through.
 * @ignore
 */
const writeSog = async (options: WriteSogOptions, fs: FileSystem): Promise<void> => {
    const { dataTable, model, ...rest } = options;
    const pool = createChunkDataPool();
    const source = dataTableToChunkSource(dataTable, pool.chunkSize, undefined, model);
    await writeSogSource(source, pool, rest, fs);
};

export { writeSog, writeSogSource };
