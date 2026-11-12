import { basename, dirname, resolve } from 'pathe';

import { createChunkDataPool, SH_REST_COUNTS } from '../chunk';
import type { ChunkDataPool, ChunkSource } from '../chunk';
import { dataTableToChunkSource } from '../compat/data-table';
import type { DataTable } from '../data-table';
import { writeFile, ZipFileSystem } from '../io/write';
import type { FileSystem } from '../io/write';
import { bakeTransform, reduceBandsSource } from '../ops';
import { sortMortonInterleaved } from '../ops/morton-order';
import { kmeansInterleaved } from '../spatial';
import type { SplatModel } from '../splat-model';
import type { DeviceCreator } from '../types';
import { logger, Transform } from '../utils';
import { decodeOpacityByte, encodeOpacityByte, OPACITY_MAX } from '../utils/opacity';
import { encodeQuatExpMap } from '../utils/quat-expmap';
import {
    BLIND_ENCODING_BYTES,
    blindFieldId,
    blindTexturesOf,
    encodeBlindValue,
    integerEncodingFor,
    isBlindEncoding
} from '../utils/sog-blind';
import type { SogBlindEncoding, SogBlindTexture } from '../utils/sog-blind';
import { version as version_ } from '../version';
import { runEncodeWebp, runQuantize1dColumns } from '../workers';

import { logWrittenFile } from './utils';

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
    /** The requested blind-data columns, in request order (f64 holds uint32 and float32 values exactly). */
    blind: Float64Array[];
};

// Gather every SOG input layer with one source.read() per chunk, so an
// interleaved file source fetches and scans each record once rather than once
// per layer. The whole scene is resident afterwards: fine for single SOG
// output, whose practical ceiling is ~1-2M gaussians (larger scenes are written
// as many small units by the LOD writer), and it lets every texture pipeline
// start immediately instead of waiting on a per-layer read.
const gatherSogLayers = async (
    source: ChunkSource,
    pool: ChunkDataPool,
    blindColumns: string[] = []
): Promise<SogLayers> => {
    const { meta } = source;
    const n = meta.numGaussians;
    const positionLayout = meta.layouts.position!;
    const geometricLayout = meta.layouts.geometric!;
    const colorLayout = meta.layouts.color!;
    const positionStride = positionLayout.stride >>> 2;
    const geometricStride = geometricLayout.stride >>> 2;
    const colorStride = colorLayout.stride >>> 2;
    const restCount = SH_REST_COUNTS[meta.shBands];

    // `other`-layer columns SOG understands: the user-named blind columns (only when asked for).
    const otherLayout = meta.availableLayers.has('other') ? meta.layouts.other : undefined;
    const blindFields = blindColumns.map((column) => {
        const field = otherLayout?.fields[column];
        if (!field) {
            const available = meta.extraColumns.map((c) => c.name).join(', ') || 'none';
            throw new Error(`SOG blind data: column '${column}' not found in the source (extra columns: ${available})`);
        }
        return field;
    });
    const otherStride = otherLayout ? otherLayout.stride >>> 2 : 0;
    const readOther = blindFields.length > 0;

    const position = new Float32Array(n * positionStride);
    const geometric = GEOMETRIC_COLS.map(() => new Float32Array(n));
    const colorDc = Array.from({ length: 3 }, () => new Float32Array(n));
    const shRest = new Float32Array(n * restCount);
    const blind = blindFields.map(() => new Float64Array(n));

    let base = 0;
    const numChunks = meta.numChunks[0] ?? 0;
    for (let k = 0; k < numChunks; k++) {
        const count = Math.min(meta.chunkSize, n - base);
        const positionData = pool.acquire('position', positionLayout, count);
        const geometricData = pool.acquire('geometric', geometricLayout, count);
        const colorData = pool.acquire('color', colorLayout, count);
        const otherData = readOther ? pool.acquire('other', otherLayout!, count) : null;
        try {
            await source.read({
                chunkIndex: k,
                position: positionData,
                geometric: geometricData,
                color: colorData,
                ...(otherData ? { other: otherData } : {})
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

            if (otherData) {
                const f32 = new Float32Array(otherData.data, 0, count * otherStride);
                const u32 = new Uint32Array(otherData.data, 0, count * otherStride);
                for (let b = 0; b < blind.length; b++) {
                    const w = blindFields[b].byteOffset >>> 2;
                    const words = blindFields[b].type === 'uint32' ? u32 : f32;
                    const column = blind[b];
                    for (let i = 0; i < count; i++) column[base + i] = words[i * otherStride + w];
                }
            }
        } finally {
            positionData.release();
            geometricData.release();
            colorData.release();
            otherData?.release();
        }
        base += count;
    }

    return { position, geometric, colorDc, shRest, blind };
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
    /**
     * SOG format version to write. Default 3. Version 2 is for viewers that
     * predate v3: it cannot carry opacity above 1, blind data or a 4th SH band,
     * and drops them with a warning.
     */
    version?: SogVersion;
    /**
     * Per-gaussian blind data (v3 only): source columns packed into blind
     * textures, one entry per texture. Each texture's fields fill its 4 bytes
     * (R, G, B, A) in order; see {@link SogBlindEncoding} for the encodings.
     * Undefined stores the columns the source read from SOG blind data as
     * they were stored; an empty list stores none.
     */
    blind?: SogBlindTexture[];
    /**
     * Application-defined JSON for meta.json's `extras` (as in glTF). Undefined
     * writes the source's extras, if any; null writes none. v3 only.
     */
    extras?: unknown;
    /**
     * Producer comments for meta.json's `comments` (e.g. a PLY's header comments).
     * Undefined writes the source's comments, if any; null writes none. v3 only.
     */
    comments?: readonly string[] | null;
    /**
     * SH palette entries. Default: the gaussian count rounded down to a power of 2, at most 65536.
     * Above 65536 needs v3 (24-bit labels), up to {@link MAX_SH_PALETTE}.
     */
    paletteSize?: number;
};

/** SOG format version. */
type SogVersion = 2 | 3;

/**
 * WebGL2 guarantees 16 textures per shader: SOG v3 with SH uses 7 and the PlayCanvas engine adds a
 * codebook lookup, so more blind textures than this may not render on the weakest devices.
 */
const MAX_PORTABLE_BLIND_TEXTURES = 8;

/**
 * Importance exponent of the scale codebook: entries are weighted by exp(0.25 * log scale), shifting
 * resolution toward the large splats that cover the most pixels. Without it the few hundred thousand
 * large splats of a scene can share 3-4 entries, and which of them land where depends on the single
 * largest scale, so render quality varies by ±1.5 dB with the splat set.
 */
const SCALE_IMPORTANCE = 0.25;

/**
 * Largest SH palette written: the palette image is 64 entries per row, so this keeps it within the
 * 4096 rows WebGL2 guarantees. v3 labels hold 24 bits; v2 labels 16 (65536 entries).
 */
const MAX_SH_PALETTE = 64 * 4096;

/**
 * Encode the v3 opacity byte of every gaussian from its true opacity (see
 * {@link encodeOpacityByte}; opacity above {@link OPACITY_MAX} is clamped).
 *
 * @param opacity - True opacity column.
 * @returns One byte per gaussian and the largest opacity stored.
 */
const encodeOpacityBytes = (opacity: Float32Array): { bytes: Uint8Array; max: number } => {
    const n = opacity.length;
    const bytes = new Uint8Array(n);
    let max = 0;
    for (let i = 0; i < n; i++) {
        const b = encodeOpacityByte(opacity[i]);
        bytes[i] = b;
        if (b > max) max = b;
    }
    return { bytes, max: decodeOpacityByte(max) };
};

/**
 * Check the requested blind textures' shape: known encodings, ranges only on
 * unorm fields, at most 4 bytes per texture and each column used once under
 * a distinct, non-empty shader identifier.
 *
 * @param textures - The requested textures.
 */
const validateBlindTextures = (textures: SogBlindTexture[]): void => {
    const seen = new Map<string, string>();
    for (const fields of textures) {
        if (fields.length === 0) throw new Error('SOG blind data: a blind texture needs at least one field');
        let bytes = 0;
        for (const f of fields) {
            // the engine reads each field through getBlind_<id>(), id derived from the name
            const id = blindFieldId(f.column);
            if (!id) {
                throw new Error(`SOG blind data: column '${f.column}' has no letters, digits or '_'`);
            }
            const other = seen.get(id);
            if (other === f.column) throw new Error(`SOG blind data: column '${f.column}' is stored more than once`);
            if (other !== undefined) {
                throw new Error(`SOG blind data: columns '${other}' and '${f.column}' share the shader name '${id}'`);
            }
            seen.set(id, f.column);
            if (f.encoding !== undefined && !isBlindEncoding(f.encoding)) {
                throw new Error(`SOG blind data: unknown encoding '${f.encoding as string}' for column '${f.column}'`);
            }
            if (f.range !== undefined) {
                if (!f.encoding?.startsWith('unorm')) {
                    throw new Error(`SOG blind data: a range applies only to unorm8 / unorm16 (column '${f.column}')`);
                }
                const [min, max] = f.range;
                if (!(Number.isFinite(min) && Number.isFinite(max) && min < max)) {
                    throw new Error(`SOG blind data: invalid range [${min}, ${max}] for column '${f.column}'`);
                }
            }
            // an unspecified encoding resolves to at most 4 bytes; count it as 1 here
            bytes += f.encoding ? BLIND_ENCODING_BYTES[f.encoding] : 1;
        }
        if (bytes > 4) {
            const names = fields.map((f) => f.column).join(', ');
            throw new Error(`SOG blind data: fields ${names} need ${bytes} bytes; a blind texture holds 4`);
        }
    }
};

/**
 * Resolve each blind texture's fields against the gathered columns: default
 * encodings (the fewest-byte unsigned integer that holds the column), byte
 * offsets, and the 4-byte limit again now that defaults are known.
 *
 * @param textures - The requested textures (validated).
 * @param columns - The gathered columns, in request order.
 * @returns The resolved textures.
 */
const resolveBlindTextures = (textures: SogBlindTexture[], columns: Float64Array[]): ResolvedBlindTexture[] => {
    let next = 0;
    return textures.map((fields) => {
        let offset = 0;
        const resolved = fields.map((f) => {
            const values = columns[next++];
            let encoding = f.encoding;
            if (!encoding) {
                let max = 0;
                for (let i = 0; i < values.length; i++) if (values[i] > max) max = values[i];
                encoding = integerEncodingFor(max);
            }
            const field = { name: f.column, encoding, offset, range: f.range, values };
            offset += BLIND_ENCODING_BYTES[encoding];
            return field;
        });
        if (offset > 4) {
            const names = fields.map((f) => f.column).join(', ');
            throw new Error(`SOG blind data: fields ${names} need ${offset} bytes; a blind texture holds 4`);
        }
        return { fields: resolved, bytes: offset };
    });
};

type ShNMeta = { count: number; bands: number; codebook: number[]; files: string[] };

type BlindFieldMeta = { name: string; type: SogBlindEncoding; range?: [number, number] };

type BlindMeta = { files: string[]; fields: BlindFieldMeta[] };

type ResolvedBlindTexture = {
    fields: {
        name: string;
        encoding: SogBlindEncoding;
        offset: number;
        range: [number, number] | undefined;
        values: Float64Array;
    }[];
    bytes: number;
};

type Clustering = { centroids: Float32Array; labels: Uint32Array };

// One SOG write between its two phases: the gathered layers plus the worker
// and GPU jobs already running on them.
type SogWriteState = {
    version: SogVersion;
    numRows: number;
    shBands: number;
    model: SplatModel;
    width: number;
    height: number;
    layers: SogLayers;
    iterations: number;
    restCount: number;
    paletteSize: number;
    scalesQuant: ReturnType<typeof runQuantize1dColumns>;
    colorsQuant: ReturnType<typeof runQuantize1dColumns>;
    shCluster: Promise<Clustering> | null;
    /** v3 only: blind textures to write, with resolved encodings and offsets. */
    blind: ResolvedBlindTexture[];
    /** meta.json `extras`, or undefined for none. */
    extras: unknown;
    /** meta.json `comments`, or undefined for none. */
    comments: readonly string[] | undefined;
    // k-means iterations completed so far, and where to report further ones
    progress: { done: number; report: ((done: number) => void) | null };
};

/**
 * First phase of a SOG write: validate, gather the source and start the
 * worker-side and GPU jobs (scale and color quantization, SH k-means). Emits
 * no log output and opens no output files, so a caller can start the next
 * write while a previous one is still finishing (see the LOD writer).
 *
 * @param source - The source to encode (its pending transform is baked to PLY space).
 * @param pool - Pool for the temporary per-chunk read buffers.
 * @param options - Output options.
 * @returns The state to hand to {@link finishSogWrite}.
 * @ignore
 */
const startSogWrite = async (
    source: ChunkSource,
    pool: ChunkDataPool,
    options: Pick<
        WriteSogSourceOptions,
        'iterations' | 'createDevice' | 'indices' | 'version' | 'blind' | 'extras' | 'comments' | 'paletteSize'
    >
): Promise<SogWriteState> => {
    const { iterations, createDevice } = options;
    const version = options.version ?? 3;
    if (version !== 2 && version !== 3) {
        throw new Error(`writeSogSource: unsupported SOG version ${version as number} (must be 2 or 3)`);
    }
    if (options.paletteSize !== undefined) {
        const maxPalette = version === 3 ? MAX_SH_PALETTE : 65536;
        if (!Number.isInteger(options.paletteSize) || options.paletteSize < 1 || options.paletteSize > maxPalette) {
            throw new Error(
                `SOG v${version} SH palette size must be an integer from 1 to ${maxPalette} (got ${options.paletteSize})`
            );
        }
    }
    // by default, store the columns read from SOG blind data as they were stored
    const blindTextures = options.blind ?? blindTexturesOf(source.meta.extraColumns);
    validateBlindTextures(blindTextures);

    let baked = bakeTransform(source, Transform.PLY);
    // v2 readers infer bands from the palette width and stop at 3
    if (version === 2 && baked.meta.shBands > 3) {
        logger.warn('SOG v2 cannot store a 4th SH band; writing 3 bands (use SOG v3)');
        baked = reduceBandsSource(baked, 3, pool);
    }
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

    const blindColumns = blindTextures.flat().map((f) => f.column);
    if (version === 2 && blindColumns.length > 0) {
        logger.warn(`SOG v2 cannot store blind data; ${blindColumns.join(', ')} not written (use SOG v3)`);
    }
    if (version === 3 && blindTextures.length > MAX_PORTABLE_BLIND_TEXTURES) {
        logger.warn(
            `${blindTextures.length} SOG blind textures may exceed the texture units of some WebGL2 devices ` +
                `(${MAX_PORTABLE_BLIND_TEXTURES} fit everywhere); pack narrow columns into shared textures`
        );
    }
    const layers = await gatherSogLayers(baked, pool, version === 3 ? blindColumns : []);

    if (version === 2 && layers.geometric[7].some((a) => a > 1)) {
        logger.warn('SOG v2 cannot store opacity above 1; it is clamped to 1 (use SOG v3)');
    }

    const blind = version === 3 ? resolveBlindTextures(blindTextures, layers.blind) : [];

    // ---- Worker-side jobs first, so the pool is busy while the main thread
    // runs its own passes in the finish phase. Quantize-bearing textures go
    // first (each is a full-column pass); SH k-means mostly waits on the GPU.
    const [, , , , s0, s1, s2] = layers.geometric;
    const [fdc0, fdc1, fdc2] = layers.colorDc;
    const scalesQuant = runQuantize1dColumns(
        [
            { name: 'scale_0', data: s0 },
            { name: 'scale_1', data: s1 },
            { name: 'scale_2', data: s2 }
        ],
        256,
        0.5,
        SCALE_IMPORTANCE
    );
    const colorsQuant = runQuantize1dColumns([
        { name: 'f_dc_0', data: fdc0 },
        { name: 'f_dc_1', data: fdc1 },
        { name: 'f_dc_2', data: fdc2 }
    ]);
    const restCount = SH_REST_COUNTS[shBands];
    const paletteSize = options.paletteSize ?? Math.min(64, 2 ** Math.floor(Math.log2(numRows / 1024))) * 1024;
    const progress: SogWriteState['progress'] = { done: 0, report: null };
    const shCluster =
        shBands > 0
            ? (async () => {
                  const gpuDevice = createDevice ? await createDevice() : undefined;
                  return kmeansInterleaved(
                      layers.shRest,
                      numRows,
                      restCount,
                      paletteSize,
                      iterations,
                      gpuDevice,
                      () => {
                          progress.done++;
                          progress.report?.(progress.done);
                      }
                  );
              })()
            : null;
    // If the finish phase throws (or never runs), these settle later; mark
    // their rejections handled so the original error propagates instead of an
    // unhandled rejection.
    // eslint-disable-next-line @typescript-eslint/no-empty-function -- preserve rejection-only control flow
    [scalesQuant, colorsQuant, shCluster].forEach((p) => p?.catch(() => {}));

    return {
        version,
        numRows,
        shBands,
        model: meta.model,
        width,
        height,
        layers,
        iterations,
        restCount,
        paletteSize,
        scalesQuant,
        colorsQuant,
        shCluster,
        blind,
        extras: options.extras === null ? undefined : (options.extras ?? meta.extras),
        comments: options.comments === null ? undefined : (options.comments ?? meta.comments),
        progress
    };
};

/**
 * Second phase of a SOG write: pack and encode the textures, await the jobs
 * started by {@link startSogWrite} and write the output files and meta.json.
 * All of the write's log output happens here.
 *
 * @param state - The state returned by {@link startSogWrite}.
 * @param options - Output options (the same `indices` passed to the start phase).
 * @param fs - File system to write through.
 * @ignore
 */
const finishSogWrite = async (
    state: SogWriteState,
    options: Pick<WriteSogSourceOptions, 'filename' | 'bundle' | 'webpEffort' | 'logging' | 'indices'>,
    fs: FileSystem
): Promise<void> => {
    const { filename: outputFilename, bundle, webpEffort } = options;
    const logging = options.logging ?? 'own';
    const emitInfo = logging !== 'silent';
    const openGroup = logging === 'own';

    const { version, numRows, shBands, width, height, layers, iterations, restCount, paletteSize, progress } = state;
    const { scalesQuant, colorsQuant, shCluster, blind } = state;
    const channels = 4;

    const bundleWriter = bundle ? await fs.createWriter(outputFilename) : null;
    const zipFs = bundleWriter ? new ZipFileSystem(bundleWriter) : null;
    const outputFs = zipFs || fs;

    // Writes are committed in call order (zip entries must be contiguous);
    // encodes run concurrently and each awaits inside its chained section.
    let writeChain: Promise<void> = Promise.resolve();

    // `exact` textures carry data in alpha that can be 0, so they must use the
    // level encoder, which keeps the RGB of fully transparent texels; the
    // simple-API default discards it. Level 6 is libwebp's preset for the
    // simple API's own defaults (method 4, quality 75).
    const writeWebp = (filename: string, data: Uint8Array, w = width, h = height, exact = false): Promise<void> => {
        const pathname = zipFs ? filename : resolve(dirname(outputFilename), filename);
        const encoded = runEncodeWebp(data, w, h, webpEffort ?? (exact ? 6 : undefined));
        const write = writeChain.then(async () => {
            const webp = await encoded;
            await writeFile(outputFs, pathname, webp);
            if (emitInfo && !zipFs) {
                logWrittenFile(filename, webp.byteLength);
            }
        });
        writeChain = write.catch<undefined>(() => undefined);
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
        // ---- k-means progress. The clustering was started with the gather;
        // its bar opens here so it nests under this write's scope. For a write
        // started ahead of time (LOD units) the bar times the remaining wait.
        if (shCluster) {
            logger.debug(
                `running k-means clustering: dims=${restCount} points=${numRows} clusters=${paletteSize} iterations=${iterations}`
            );
            const bar = logger.bar('k-means', iterations);
            bar.update(progress.done);
            progress.report = (done) => bar.update(done);
            shCluster.then(
                () => bar.end(),
                // eslint-disable-next-line @typescript-eslint/no-empty-function -- preserve rejection-only control flow
                () => {}
            );
        }

        const [r0, r1, r2, r3, , , , op] = layers.geometric;
        const v3 = version === 3;

        // ---- means: Morton order (unless a caller-supplied order is used) +
        // log-encoded positions split into low/high bytes.
        const meansMeta = (() => {
            const pos = layers.position;
            if (!externalOrder) sortMortonInterleaved(pos, indices);

            const mm = [
                [Infinity, -Infinity],
                [Infinity, -Infinity],
                [Infinity, -Infinity]
            ];
            for (let g = 0; g < numRows; g++) {
                for (let a = 0; a < 3; a++) {
                    const v = pos[g * 3 + a];
                    if (v < mm[a][0]) mm[a][0] = v;
                    if (v > mm[a][1]) mm[a][1] = v;
                }
            }
            const minMax = mm.map((v) => v.map(logTransform));
            const meansL = new Uint8Array(width * height * channels);
            const meansU = new Uint8Array(width * height * channels);
            for (let i = 0; i < numRows; ++i) {
                const g = indices[i];
                const x = (65535 * (logTransform(pos[g * 3 + 0]) - minMax[0][0])) / (minMax[0][1] - minMax[0][0]);
                const y = (65535 * (logTransform(pos[g * 3 + 1]) - minMax[1][0])) / (minMax[1][1] - minMax[1][0]);
                const z = (65535 * (logTransform(pos[g * 3 + 2]) - minMax[2][0])) / (minMax[2][1] - minMax[2][0]);
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
            return { mins: minMax.map((v) => v[0]), maxs: minMax.map((v) => v[1]) };
        })();

        // ---- v3 quats: a 24-bit exp map in RGB. The alpha stays 255: WebKit
        // premultiplies decoded images under WebGPU, which destroys RGB
        // stored under a non-opaque alpha.
        if (v3) {
            const quats = new Uint8Array(width * height * channels);
            for (let i = 0; i < numRows; ++i) {
                const g = indices[i];
                encodeQuatExpMap(r0[g], r1[g], r2[g], r3[g], quats, i * 4);
                quats[i * 4 + 3] = 0xff;
            }
            pending.push(writeWebp('quats.webp', quats));
        } else {
            // ---- v2 quats: largest-3 packed quaternions.
            const quats = new Uint8Array(width * height * channels);
            const q = [0, 0, 0, 0];
            const sqrt2 = Math.sqrt(2);
            // Largest-3 component orders, indexed by the dropped component.
            const quatIdx = [
                [1, 2, 3],
                [0, 2, 3],
                [0, 1, 3],
                [0, 1, 2]
            ];
            for (let i = 0; i < numRows; ++i) {
                const g = indices[i];
                q[0] = r0[g];
                q[1] = r1[g];
                q[2] = r2[g];
                q[3] = r3[g];
                const l = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
                q[0] /= l;
                q[1] /= l;
                q[2] /= l;
                q[3] /= l;
                let maxComp = 0;
                if (Math.abs(q[1]) > Math.abs(q[maxComp])) maxComp = 1;
                if (Math.abs(q[2]) > Math.abs(q[maxComp])) maxComp = 2;
                if (Math.abs(q[3]) > Math.abs(q[maxComp])) maxComp = 3;
                const s = (q[maxComp] < 0 ? -1 : 1) * sqrt2;
                q[0] *= s;
                q[1] *= s;
                q[2] *= s;
                q[3] *= s;
                const idx = quatIdx[maxComp];
                const ti = i;
                // Round, don't let the Uint8Array store truncate: truncation
                // biases every component down half a step and doubles the
                // rotation error the 8-bit lattice allows.
                quats[ti * 4] = Math.round(255 * (q[idx[0]] * 0.5 + 0.5));
                quats[ti * 4 + 1] = Math.round(255 * (q[idx[1]] * 0.5 + 0.5));
                quats[ti * 4 + 2] = Math.round(255 * (q[idx[2]] * 0.5 + 0.5));
                quats[ti * 4 + 3] = 252 + maxComp;
            }
            pending.push(writeWebp('quats.webp', quats));
        }

        // ---- scales: quantized log-scales.
        const sd = await scalesQuant;
        const scaleCols = sd.labels.map((c) => c.data);
        pending.push(writeLabels('scales.webp', scaleCols, indices));
        const scalesCodebook = Array.from(sd.centroids);

        // ---- sh0: quantized DC + opacity (v2: linear 0..1 byte; v3: the
        // fixed sqrt / over-unity byte, see encodeOpacityByte).
        let opacityData: Uint8Array;
        let opacityMax = 1;
        if (v3) {
            const enc = encodeOpacityBytes(op);
            opacityData = enc.bytes;
            opacityMax = enc.max;
        } else {
            // rounded, like the quaternions: a Uint8Array store truncates
            opacityData = new Uint8Array(numRows);
            for (let i = 0; i < numRows; ++i) {
                opacityData[i] = Math.max(0, Math.min(255, Math.round(op[i] * 255)));
            }
        }
        const cd = await colorsQuant;
        pending.push(writeLabels('sh0.webp', [...cd.labels.map((c) => c.data), opacityData], indices));
        const colorsCodebook = Array.from(cd.centroids);

        // ---- shN: k-means palette + per-gaussian labels.
        let shN: ShNMeta | null = null;
        if (shCluster) {
            const shCoeffs = restCount / 3;
            const { centroids, labels } = await shCluster;
            const numCentroids = centroids.length / restCount;

            // quantize the centroid palette to a uint8 codebook. De-interleave
            // the (small) centroids into restCount columns for the quantizer.
            const cbCols: { name: string; data: Float32Array }[] = [];
            for (let j = 0; j < restCount; ++j) {
                const col = new Float32Array(numCentroids);
                for (let i = 0; i < numCentroids; ++i) col[i] = centroids[i * restCount + j];
                cbCols.push({ name: `f_rest_${j}`, data: col });
            }
            const codebookPromise = runQuantize1dColumns(cbCols);

            const labelsBuf = new Uint8Array(width * height * channels);
            for (let i = 0; i < numRows; ++i) {
                const label = labels[indices[i]];
                const ti = i;
                labelsBuf[ti * 4 + 0] = 0xff & label;
                labelsBuf[ti * 4 + 1] = 0xff & (label >> 8);
                // v3 labels are 24-bit; v2 readers expect 0
                labelsBuf[ti * 4 + 2] = version === 3 ? 0xff & (label >> 16) : 0;
                labelsBuf[ti * 4 + 3] = 0xff;
            }

            const cb = await codebookPromise;
            const cbLabels = cb.labels.map((c) => c.data); // restCount columns, length numCentroids
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
                // fewer gaussians than requested entries: each is its own entry
                count: numCentroids,
                bands: shBands,
                codebook: Array.from(cb.centroids),
                files: ['shN_centroids.webp', 'shN_labels.webp']
            };
        }

        // ---- blind (v3): each texture's fields fill its bytes (R, G, B, A) in
        // order, each little-endian. Unused bytes are RGB 0 / A 255, so every
        // texel is opaque unless the data itself uses A.
        const blindMeta: BlindMeta[] = [];
        for (let t = 0; t < blind.length; t++) {
            const { fields, bytes } = blind[t];
            const buf = new Uint8Array(width * height * channels);
            if (bytes < 4) for (let ti = 3; ti < buf.length; ti += 4) buf[ti] = 0xff;
            for (const f of fields) {
                const n = BLIND_ENCODING_BYTES[f.encoding];
                const range = f.range ?? [0, 1];
                for (let i = 0; i < numRows; ++i) {
                    const v = encodeBlindValue(f.values[indices[i]], f.encoding, range, f.name);
                    const ti = i * 4 + f.offset;
                    for (let k = 0; k < n; k++) buf[ti + k] = (v >>> (8 * k)) & 0xff;
                }
            }
            const file = `blind_${t}.webp`;
            pending.push(writeWebp(file, buf, width, height, bytes > 3));
            blindMeta.push({
                files: [file],
                fields: fields.map((f) => ({
                    name: f.name,
                    type: f.encoding,
                    ...(f.range ? { range: f.range } : {})
                }))
            });
        }

        await Promise.all(pending);

        // ---- meta.json --------------------------------------------------
        const metaObj = v3
            ? {
                  version: 3,
                  asset: { generator: `splat-transform v${version_}` },
                  count: numRows,
                  ...(state.model === 'default' ? {} : { model: state.model }),
                  means: { mins: meansMeta.mins, maxs: meansMeta.maxs, files: ['means_l.webp', 'means_u.webp'] },
                  scales: { codebook: scalesCodebook, files: ['scales.webp'] },
                  quats: { files: ['quats.webp'] },
                  sh0: { codebook: colorsCodebook, files: ['sh0.webp'] },
                  ...(opacityMax > 1 ? { opacity: { max: opacityMax } } : {}),
                  ...(shN ? { shN } : {}),
                  ...(blindMeta.length > 0 ? { blind: blindMeta } : {}),
                  ...(state.extras !== undefined ? { extras: state.extras } : {}),
                  ...(state.comments?.length ? { comments: state.comments } : {})
              }
            : {
                  version: 2,
                  asset: { generator: `splat-transform v${version_}` },
                  count: numRows,
                  // untagged scenes stay byte-identical to pre-3.2 output
                  ...(state.model === 'default' ? {} : { model: state.model }),
                  means: { mins: meansMeta.mins, maxs: meansMeta.maxs, files: ['means_l.webp', 'means_u.webp'] },
                  scales: { codebook: scalesCodebook, files: ['scales.webp'] },
                  quats: { files: ['quats.webp'] },
                  sh0: { codebook: colorsCodebook, files: ['sh0.webp'] },
                  // extras and comments are v3 additions: v2 output stays as released
                  ...(shN ? { shN } : {})
              };
        const metaJson = new TextEncoder().encode(JSON.stringify(metaObj));
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
    await finishSogWrite(await startSogWrite(source, pool, options), options, fs);
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

export { finishSogWrite, startSogWrite, writeSog, writeSogSource };
export type { SogVersion };
