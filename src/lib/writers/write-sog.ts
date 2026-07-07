import { basename, dirname, resolve } from 'pathe';

import { logWrittenFile } from './utils';
import {
    createChunkDataPool,
    type ChunkDataPool,
    type ChunkLayer,
    type ReadRequest,
    type ChunkSource
} from '../chunk';
import { dataTableToChunkSource } from '../compat/data-table';
import { type DataTable, shRestNames } from '../data-table';
import { type FileSystem, writeFile, ZipFileSystem } from '../io/write';
import { bakeTransform } from '../ops';
import { sortMortonInterleaved } from '../ops/morton-order';
import { kmeansInterleaved } from '../spatial';
import type { DeviceCreator } from '../types';
import { logger, sigmoid, Transform } from '../utils';
import { version } from '../version';
import { runEncodeWebp, runQuantize1dColumns } from '../workers';

const GEOMETRIC_COLS = ['rot_0', 'rot_1', 'rot_2', 'rot_3', 'scale_0', 'scale_1', 'scale_2', 'opacity'];

const logTransform = (value: number): number => {
    return Math.sign(value) * Math.log(Math.abs(value) + 1);
};

// Gather a layer's data as one interleaved Float32Array `[g0 words, g1 words, ...]`
// (a contiguous copy per chunk — the layer is already packed).
const gatherInterleaved = async (source: ChunkSource, pool: ChunkDataPool, layer: ChunkLayer): Promise<Float32Array> => {
    const { meta } = source;
    const n = meta.numGaussians;
    const layout = meta.layouts[layer]!;
    const sw = layout.stride >>> 2;
    const out = new Float32Array(n * sw);
    const numChunks = meta.numChunks[0] ?? 0;
    let base = 0;
    for (let k = 0; k < numChunks; k++) {
        const count = Math.min(meta.chunkSize, n - base);
        const cd = pool.acquire(layer, layout, count);
        await source.read({ chunkIndex: k, [layer]: cd } as ReadRequest);
        out.set(new Float32Array(cd.data, 0, count * sw), base * sw);
        cd.release();
        base += count;
    }
    return out;
};

// Gather a layer into per-column Float32Arrays, in the layer's canonical word
// order (so `names[c]` maps to source word `c`). Used where downstream helpers
// (quantize1d / kmeans) want individual columns.
const gatherColumns = async (source: ChunkSource, pool: ChunkDataPool, layer: ChunkLayer, names: string[]): Promise<Float32Array[]> => {
    const { meta } = source;
    const n = meta.numGaussians;
    const layout = meta.layouts[layer]!;
    const sw = layout.stride >>> 2;
    const cols = names.map(() => new Float32Array(n));
    const numChunks = meta.numChunks[0] ?? 0;
    let base = 0;
    for (let k = 0; k < numChunks; k++) {
        const count = Math.min(meta.chunkSize, n - base);
        const cd = pool.acquire(layer, layout, count);
        await source.read({ chunkIndex: k, [layer]: cd } as ReadRequest);
        const f = new Float32Array(cd.data, 0, count * sw);
        for (let c = 0; c < names.length; c++) {
            const col = cols[c];
            for (let i = 0; i < count; i++) col[base + i] = f[i * sw + c];
        }
        cd.release();
        base += count;
    }
    return cols;
};

type WriteSogSourceOptions = {
    filename: string;
    bundle: boolean;
    iterations: number;
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
 * Native SOG writer: encodes a {@link ChunkSource} to the PlayCanvas SOG format,
 * reading the source one layer at a time. Each layer is gathered, consumed, and
 * **released before the next is loaded** (each phase below is its own scope), so
 * peak resident scene data is the largest single layer — not the whole scene.
 *
 * Output is equivalent to the legacy DataTable `writeSog` (same Morton order,
 * quantization/clustering, texel encoding), and byte-identical for the per-file
 * (non-bundled) outputs. Everything works on raw typed-array columns / interleaved
 * buffers (no DataTable): `runQuantize1dColumns` / `kmeansInterleaved` /
 * `runEncodeWebp` consume the gathered layers directly.
 *
 * @param source - The source to encode (its pending transform is baked to PLY space).
 * @param pool - Pool for the temporary per-layer read buffers.
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
    const { filename: outputFilename, bundle, iterations, createDevice } = options;
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

    const bundleWriter = bundle ? await fs.createWriter(outputFilename) : null;
    const zipFs = bundleWriter ? new ZipFileSystem(bundleWriter) : null;
    const outputFs = zipFs || fs;

    // Writes are committed in call order (zip entries must be contiguous);
    // encodes run concurrently and each awaits inside its chained section.
    let writeChain: Promise<void> = Promise.resolve();

    const writeWebp = (filename: string, data: Uint8Array, w = width, h = height): Promise<void> => {
        const pathname = zipFs ? filename : resolve(dirname(outputFilename), filename);
        const encoded = runEncodeWebp(data, w, h);
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
        // ---- Phase 1: positions — Morton order (unless a caller-supplied order
        // is used) + means. `pos` is released when this scope returns (only
        // `indices` + the small means meta escape).
        const meansMeta = await (async () => {
            const pos = await gatherInterleaved(baked, pool, 'position');
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

        // ---- Phase 2: geometric — quaternions + scales. The 32 B/gaussian layer
        // is released on return; only the 1 B/gaussian `opacityData` escapes.
        const { scalesCodebook, opacityData } = await (async () => {
            const geom = await gatherColumns(baked, pool, 'geometric', GEOMETRIC_COLS);
            const [r0, r1, r2, r3, s0, s1, s2, op] = geom;

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

            const sd = await runQuantize1dColumns([
                { name: 'scale_0', data: s0 }, { name: 'scale_1', data: s1 }, { name: 'scale_2', data: s2 }
            ]);
            pending.push(writeLabels('scales.webp', sd.labels.map(c => c.data), indices));

            const od = new Uint8Array(numRows);
            for (let i = 0; i < numRows; ++i) {
                od[i] = Math.max(0, Math.min(255, sigmoid(op[i]) * 255));
            }
            return { scalesCodebook: Array.from(sd.centroids), opacityData: od };
        })();

        // ---- Phase 3: color — sh0 (DC + opacity) and SH rest. The color layer
        // is released on return.
        const { colorsCodebook, shN } = await (async (): Promise<{ colorsCodebook: number[]; shN: ShNMeta | null }> => {
            const restCount = [0, 9, 24, 45][shBands];

            // Gather the color layer once, splitting it into the 3 DC columns
            // (for sh0 quantization) and the SH-rest as a contiguous interleaved
            // buffer (for k-means). f_rest occupies words [3, 3+restCount) of each
            // record, so it copies out as one block per gaussian — no de/re-interleave.
            const layout = meta.layouts.color!;
            const sw = layout.stride >>> 2;
            const fdc0 = new Float32Array(numRows);
            const fdc1 = new Float32Array(numRows);
            const fdc2 = new Float32Array(numRows);
            const shRest = restCount > 0 ? new Float32Array(numRows * restCount) : new Float32Array(0);
            {
                const numChunks = meta.numChunks[0] ?? 0;
                let base = 0;
                for (let c = 0; c < numChunks; c++) {
                    const count = Math.min(meta.chunkSize, numRows - base);
                    const cdBuf = pool.acquire('color', layout, count);
                    await baked.read({ chunkIndex: c, color: cdBuf } as ReadRequest);
                    const f = new Float32Array(cdBuf.data, 0, count * sw);
                    for (let i = 0; i < count; i++) {
                        const o = i * sw;
                        fdc0[base + i] = f[o];
                        fdc1[base + i] = f[o + 1];
                        fdc2[base + i] = f[o + 2];
                    }
                    if (restCount > 0) {
                        for (let i = 0; i < count; i++) {
                            shRest.set(f.subarray(i * sw + 3, i * sw + 3 + restCount), (base + i) * restCount);
                        }
                    }
                    cdBuf.release();
                    base += count;
                }
            }

            const cd = await runQuantize1dColumns([
                { name: 'f_dc_0', data: fdc0 }, { name: 'f_dc_1', data: fdc1 }, { name: 'f_dc_2', data: fdc2 }
            ]);
            pending.push(writeLabels('sh0.webp', [...cd.labels.map(c => c.data), opacityData], indices));
            const codebook = Array.from(cd.centroids);

            let shNLocal: ShNMeta | null = null;
            if (shBands > 0) {
                const shCoeffs = [0, 3, 8, 15][shBands];
                const paletteSize = Math.min(64, 2 ** Math.floor(Math.log2(numRows / 1024))) * 1024;
                const gpuDevice = createDevice ? await createDevice() : undefined;
                const { centroids, labels } = await kmeansInterleaved(shRest, numRows, restCount, paletteSize, iterations, gpuDevice);
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
                shNLocal = {
                    count: paletteSize,
                    bands: shBands,
                    codebook: Array.from(cb.centroids),
                    files: ['shN_centroids.webp', 'shN_labels.webp']
                };
            }
            return { colorsCodebook: codebook, shN: shNLocal };
        })();

        await Promise.all(pending);

        // ---- meta.json --------------------------------------------------
        const metaObj: any = {
            version: 2,
            asset: { generator: `splat-transform v${version}` },
            count: numRows,
            means: { mins: meansMeta.mins, maxs: meansMeta.maxs, files: ['means_l.webp', 'means_u.webp'] },
            scales: { codebook: scalesCodebook, files: ['scales.webp'] },
            quats: { files: ['quats.webp'] },
            sh0: { codebook: colorsCodebook, files: ['sh0.webp'] },
            ...(shN ? { shN } : {})
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

type WriteSogOptions = WriteSogSourceOptions & { dataTable: DataTable };

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
    const { dataTable, ...rest } = options;
    const pool = createChunkDataPool();
    const source = dataTableToChunkSource(dataTable, pool.chunkSize);
    await writeSogSource(source, pool, rest, fs);
};

export { writeSog, writeSogSource };
