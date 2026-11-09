import {
    POSITION_STRIDE,
    GEOMETRIC_STRIDE,
    SH_REST_COUNTS,
    colorStride,
    positionFields,
    geometricFields,
    colorFields,
    otherLayout,
    createChunkDataPool
} from '../chunk';
import type {
    ReadRequest,
    ChunkSource,
    ChunkSourceMetadata,
    ChunkDataPool,
    ChunkLayer,
    ExtraColumn,
    LayerLayout,
    SHBands
} from '../chunk';
import { dataTableToChunkSource, materializeToDataTable } from '../compat/data-table';
import type { DataTable } from '../data-table';
import { basename, dirname, join, readFile } from '../io/read';
import type { ReadFileSystem } from '../io/read';
import { isSplatModel } from '../splat-model';
import type { SplatModel } from '../splat-model';
import { logger, Transform, WebPCodec } from '../utils';
import { decodeOpacityByte } from '../utils/opacity';
import { decodeQuatExpMap } from '../utils/quat-expmap';
import { BLIND_ENCODING_BYTES, decodeBlindValue, isBlindEncoding, isIntegerEncoding } from '../utils/sog-blind';
import type { SogBlindEncoding } from '../utils/sog-blind';

import { readSogV1 } from './read-sog-v1';
import type { MetaV1 } from './read-sog-v1';

// V2/V3 SOG meta layout - codebook-based quantization for scales / sh0 / shN.
// V3 drops `quats` (the rotation is an exp map in the alpha channels of
// means_l / means_u / scales), stores opacity as a fixed byte that may reach
// above 1 (`opacity.max` > 1 says some do, see decodeOpacityByte) and adds
// optional `blind` data. Legacy V1 uses a
// per-channel mins/maxs scheme and is handled separately in read-sog-v1.ts.
type MetaV2 = {
    version: 2 | 3;
    count: number;
    model?: string;
    means: { mins: number[]; maxs: number[]; files: string[] };
    scales: { codebook: number[]; files: string[] };
    quats?: { files: string[] };
    sh0: { codebook: number[]; files: string[] };
    opacity?: { max: number };
    shN?: { count: number; bands: number; codebook: number[]; files: string[] };
    blind?: { files: string[]; fields: { name: string; type: string; range?: [number, number] }[] }[];
    extras?: unknown;
    comments?: string[];
};

// The `other`-layer columns a V3 meta implies: each blind field under its recorded
// name (integer encodings as uint32, the rest as float32), tagged with how it was stored.
const extraColumnsFromMeta = (meta: MetaV2): ExtraColumn[] => {
    const extras: ExtraColumn[] = [];
    if (meta.version === 3 && meta.blind) {
        meta.blind.forEach((texture, t) => {
            for (const f of texture.fields) {
                if (!isBlindEncoding(f.type))
                    throw new Error(`SOG blind field '${f.name}' has unknown type '${f.type}'`);
                extras.push({
                    name: f.name,
                    type: isIntegerEncoding(f.type) ? 'uint32' : 'float32',
                    sogBlind: { texture: t, encoding: f.type, ...(f.range ? { range: f.range } : {}) }
                });
            }
        });
    }
    return extras;
};

// V3 records `opacity.max` exactly when some stored opacity exceeds 1.
const opacityAboveOneFromMeta = (meta: MetaV2): boolean => meta.version === 3 && (meta.opacity?.max ?? 1) > 1;

// SH band count from shN.bands, or 0 when absent / out of range.
const bandsFromMeta = (meta: MetaV2): SHBands => {
    const b = meta.shN?.bands ?? 0;
    return (b in SH_REST_COUNTS ? b : 0) as SHBands;
};

type ReadSogOptions = {
    // Controls how readSog reports its own progress (mirrors writeSog's
    // `logging` option, reduced to the modes a bar-only reader needs):
    //   'own'    — open readSog's internal 'decoding' bar (default)
    //   'silent' — no bar; the caller drives its own progress (e.g. readLcc2
    //              decoding many chunks concurrently — logger bars are strictly
    //              LIFO, so concurrent per-chunk bars would corrupt each other)
    logging?: 'own' | 'silent';
};

// meta.json's `model` entry: absent means an untagged scene, and an unrecognized
// value warns rather than failing the read.
const modelFromMeta = (value: string | undefined): SplatModel => {
    if (value === undefined) return 'default';
    if (isSplatModel(value)) return value;
    logger.warn(`unrecognized splat model '${value}' in meta.json, reading as 'default'`);
    return 'default';
};

// Inverse of logTransform(x) = sign(x) * ln(|x| + 1)
const invLogTransform = (v: number) => {
    const a = Math.abs(v);
    const e = Math.exp(a) - 1; // |x|
    return v < 0 ? -e : e;
};

// Component order of the three packed values for each maxComp (flat, 3 per
// row), plus a shared output scratch — this decode runs once per gaussian per
// read, so it must not allocate (single-threaded, mirroring read-spz's
// tmpQuat pattern).
const QUAT_IDX = [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2];
const SQRT2 = Math.sqrt(2);
const tmpQuat = new Float32Array(4);
const f32View = new Float32Array(1);
const u32View = new Uint32Array(f32View.buffer);

// Decode a smallest-three packed quaternion into `tmpQuat` as [w, x, y, z].
const unpackQuat = (px: number, py: number, pz: number, tag: number): void => {
    const maxComp = tag - 252;
    const a = ((px / 255) * 2 - 1) / SQRT2;
    const b = ((py / 255) * 2 - 1) / SQRT2;
    const c = ((pz / 255) * 2 - 1) / SQRT2;
    tmpQuat[0] = 0;
    tmpQuat[1] = 0;
    tmpQuat[2] = 0;
    tmpQuat[3] = 0;
    const base = maxComp * 3;
    tmpQuat[QUAT_IDX[base]] = a;
    tmpQuat[QUAT_IDX[base + 1]] = b;
    tmpQuat[QUAT_IDX[base + 2]] = c;
    // reconstruct max component to make unit length with positive sign
    const t = 1 - (a * a + b * b + c * c);
    tmpQuat[maxComp] = Math.sqrt(Math.max(0, t));
};

/**
 * Native (V2/V3) SOG `ChunkSource`: the decoded WebP textures (RGBA) + codebooks /
 * SH palette are held resident, and `read`/`readRows` expand only the requested
 * gaussians on demand — the per-row inverse of `writeSog`'s texel encode. Peak
 * resident is the (quantized) textures (~tens of bytes/gaussian), not the f32
 * scene; a position-only pass costs one texel lookup per gaussian. WebP has no
 * sub-image random access, so each texture is decoded once up front.
 *
 * @param meta - The parsed V2/V3 meta.
 * @param fileSystem - File system for the texture files.
 * @param resolve - Resolves a texture filename relative to meta.json's directory.
 * @param pool - Pool whose `chunkSize` sets the chunking granularity.
 * @returns A lazy `ChunkSource` over the SOG's gaussians.
 */
const readSogSourceV2 = async (
    meta: MetaV2,
    fileSystem: ReadFileSystem,
    resolve: (name: string) => string,
    pool: ChunkDataPool
): Promise<ChunkSource> => {
    const decoder = await WebPCodec.create();
    const count = meta.count;
    const chunkSize = pool.chunkSize;

    const load = async (name: string): Promise<Uint8Array> => {
        const src = await fileSystem.createSource(resolve(name));
        try {
            return await src.read().readAll();
        } finally {
            src.close();
        }
    };

    // means: two textures (low + high byte) packed as a 16-bit lerp between the
    // mins/maxs of the logTransform'd positions.
    const meansLo = decoder.decodeRGBA(await load(meta.means.files[0]));
    const lo = meansLo.rgba;
    const hi = decoder.decodeRGBA(await load(meta.means.files[1])).rgba;
    if (meansLo.width * meansLo.height < count) throw new Error('SOG means texture too small for count');
    const { mins, maxs } = meta.means;
    const xMin = mins[0],
        xScale = maxs[0] - mins[0] || 1;
    const yMin = mins[1],
        yScale = maxs[1] - mins[1] || 1;
    const zMin = mins[2],
        zScale = maxs[2] - mins[2] || 1;

    const v3 = meta.version === 3;

    // V2 quats: 4 bytes/splat, last byte is the largest-component tag (252-255).
    // V3 has no quats texture (rotation lives in the alpha channels).
    let qr: Uint8Array | null = null;
    if (!v3) {
        const quats = decoder.decodeRGBA(await load(meta.quats!.files[0]));
        qr = quats.rgba;
        if (quats.width * quats.height < count) throw new Error('SOG quats texture too small for count');
    }

    // scales: 3 bytes index the shared 256-entry codebook.
    const scales = decoder.decodeRGBA(await load(meta.scales.files[0]));
    const sl = scales.rgba;
    if (scales.width * scales.height < count) throw new Error('SOG scales texture too small for count');
    const sCode = new Float32Array(meta.scales.codebook);

    // sh0: 3 color bytes index the codebook; the 4th is the opacity — linear
    // 0..1 in V2, the fixed sqrt / over-unity byte in V3 (may exceed 1).
    const sh0 = decoder.decodeRGBA(await load(meta.sh0.files[0]));
    const c0 = sh0.rgba;
    if (sh0.width * sh0.height < count) throw new Error('SOG sh0 texture too small for count');
    const cCode = new Float32Array(meta.sh0.codebook);
    const oCode = new Float32Array(256);
    for (let i = 0; i < 256; i++) oCode[i] = v3 ? decodeOpacityByte(i) : i / 255;

    // V3 blind data: each texture's fields packed in order into R,G,B,A, little-endian.
    const extras = extraColumnsFromMeta(meta);
    const otherL = extras.length > 0 ? otherLayout(extras) : null;
    const otherSw = otherL ? otherL.stride >>> 2 : 0;
    const blindFields: {
        rgba: Uint8Array;
        offset: number;
        bytes: number;
        encoding: SogBlindEncoding;
        range: [number, number];
        word: number;
        integer: boolean;
    }[] = [];
    if (v3 && meta.blind) {
        for (const texture of meta.blind) {
            const bl = decoder.decodeRGBA(await load(texture.files[0]));
            if (bl.width * bl.height < count) throw new Error('SOG blind texture too small for count');
            let offset = 0;
            for (const f of texture.fields) {
                const encoding = f.type as SogBlindEncoding;
                const bytes = BLIND_ENCODING_BYTES[encoding];
                if (offset + bytes > 4) {
                    throw new Error(`SOG blind field '${f.name}' does not fit its texture (offset ${offset})`);
                }
                blindFields.push({
                    rgba: bl.rgba,
                    offset,
                    bytes,
                    encoding,
                    range: f.range ?? [0, 1],
                    word: otherL!.fields[f.name].byteOffset >>> 2,
                    integer: isIntegerEncoding(encoding)
                });
                offset += bytes;
            }
        }
    }

    // shN (optional): label -> centroid palette -> codebook, channel-major f_rest.
    let shBands: SHBands = 0;
    let shCoeffs = 0;
    let restCount = 0;
    let shCodebook: Float32Array | null = null;
    let centroidsRGBA: Uint8Array | null = null;
    let labelsRGBA: Uint8Array | null = null;
    let cW = 0;
    let cH = 0;
    let paletteCount = 0;
    if (bandsFromMeta(meta) > 0) {
        shBands = bandsFromMeta(meta);
        restCount = SH_REST_COUNTS[shBands];
        shCoeffs = restCount / 3;
        shCodebook = new Float32Array(meta.shN!.codebook);
        const cen = decoder.decodeRGBA(await load(meta.shN!.files[0]));
        centroidsRGBA = cen.rgba;
        cW = cen.width;
        cH = cen.height;
        const lab = decoder.decodeRGBA(await load(meta.shN!.files[1]));
        labelsRGBA = lab.rgba;
        if (lab.width * lab.height < count) throw new Error('SOG shN labels texture too small for count');
        if (cW !== 64 * shCoeffs) {
            throw new Error(
                `SOG shN centroids texture width ${cW} does not match expected ${64 * shCoeffs} for ${shBands}-band palette`
            );
        }
        paletteCount = meta.shN!.count;
    }
    const colorSw = 3 + restCount;

    const layouts: Partial<Record<ChunkLayer, LayerLayout>> = {
        position: { stride: POSITION_STRIDE, fields: positionFields() },
        geometric: { stride: GEOMETRIC_STRIDE, fields: geometricFields() },
        color: { stride: colorStride(shBands), fields: colorFields(shBands) },
        ...(otherL ? { other: otherL } : {})
    };
    const availableLayers = new Set<ChunkLayer>(['position', 'geometric', 'color']);
    if (otherL) availableLayers.add('other');

    const meta_: ChunkSourceMetadata = {
        numGaussians: count,
        numLods: 1,
        lodCounts: [count],
        chunkSize,
        numChunks: [count === 0 ? 0 : Math.ceil(count / chunkSize)],
        shBands,
        model: modelFromMeta(meta.model),
        extraColumns: extras,
        transform: Transform.PLY.clone(),
        availableLayers,
        layouts,
        ...(opacityAboveOneFromMeta(meta) ? { opacityAboveOne: true } : {}),
        ...(meta.extras !== undefined ? { extras: meta.extras } : {}),
        ...(Array.isArray(meta.comments) && meta.comments.length > 0 ? { comments: meta.comments.map(String) } : {})
    };

    // Expand source gaussian `g` into output row `r` of the requested layers.
    // Shared by sequential `read` and random-access `readRows`. Every field of a
    // requested layer is written (pool buffers may hold prior data), so the SH
    // rest is explicitly zeroed for an out-of-range label rather than left stale.
    const decodeInto = (
        posOut: Float32Array | null,
        geoOut: Float32Array | null,
        colOut: Float32Array | null,
        othOut: Uint32Array | null,
        g: number,
        r: number
    ): void => {
        const o4 = g * 4;

        if (posOut) {
            const o = r * 3;
            const xv = lo[o4] | (hi[o4] << 8);
            const yv = lo[o4 + 1] | (hi[o4 + 1] << 8);
            const zv = lo[o4 + 2] | (hi[o4 + 2] << 8);
            posOut[o] = invLogTransform(xMin + xScale * (xv / 65535));
            posOut[o + 1] = invLogTransform(yMin + yScale * (yv / 65535));
            posOut[o + 2] = invLogTransform(zMin + zScale * (zv / 65535));
        }

        if (geoOut) {
            const o = r * 8; // [rot0..3, scale0..2, opacity]
            if (v3) {
                decodeQuatExpMap(lo[o4 + 3], hi[o4 + 3], sl[o4 + 3], geoOut, o);
            } else if (qr![o4 + 3] < 252 || qr![o4 + 3] > 255) {
                geoOut[o] = 1;
                geoOut[o + 1] = 0;
                geoOut[o + 2] = 0;
                geoOut[o + 3] = 0;
            } else {
                unpackQuat(qr![o4], qr![o4 + 1], qr![o4 + 2], qr![o4 + 3]);
                geoOut[o] = tmpQuat[0];
                geoOut[o + 1] = tmpQuat[1];
                geoOut[o + 2] = tmpQuat[2];
                geoOut[o + 3] = tmpQuat[3];
            }
            geoOut[o + 4] = sCode[sl[o4]];
            geoOut[o + 5] = sCode[sl[o4 + 1]];
            geoOut[o + 6] = sCode[sl[o4 + 2]];
            geoOut[o + 7] = oCode[c0[o4 + 3]];
        }

        if (othOut) {
            const o = r * otherSw;
            for (const f of blindFields) {
                const b = f.rgba;
                const t = o4 + f.offset;
                let bits = 0;
                for (let k = 0; k < f.bytes; k++) bits |= b[t + k] << (8 * k);
                if (f.integer) {
                    othOut[o + f.word] = bits >>> 0;
                } else {
                    f32View[0] = decodeBlindValue(bits >>> 0, f.encoding, f.range);
                    othOut[o + f.word] = u32View[0];
                }
            }
        }

        if (colOut) {
            const o = r * colorSw; // [dc0..2, f_rest_0..N]
            colOut[o] = cCode[c0[o4]];
            colOut[o + 1] = cCode[c0[o4 + 1]];
            colOut[o + 2] = cCode[c0[o4 + 2]];
            if (restCount > 0) {
                // palette index: 16-bit in v2, 24-bit in v3
                const label = labelsRGBA![o4] | (labelsRGBA![o4 + 1] << 8) | (v3 ? labelsRGBA![o4 + 2] << 16 : 0);
                if (label < paletteCount) {
                    // centroid texel coordinates: row from the label, column
                    // base from the label's slot, one column per coefficient
                    // (per-coeff work is pure index math — no allocation)
                    const cy = Math.floor(label / 64);
                    const cxBase = (label % 64) * shCoeffs;
                    const cyOk = cy < cH;
                    for (let j = 0; j < shCoeffs; j++) {
                        const cx = cxBase + j;
                        const ok = cyOk && cx < cW;
                        const idx = (cy * cW + cx) * 4;
                        const lr = ok ? centroidsRGBA![idx] : 0;
                        const lg = ok ? centroidsRGBA![idx + 1] : 0;
                        const lb = ok ? centroidsRGBA![idx + 2] : 0;
                        colOut[o + 3 + j] = shCodebook![lr] ?? 0;
                        colOut[o + 3 + j + shCoeffs] = shCodebook![lg] ?? 0;
                        colOut[o + 3 + j + shCoeffs * 2] = shCodebook![lb] ?? 0;
                    }
                } else {
                    for (let k = 0; k < restCount; k++) colOut[o + 3 + k] = 0;
                }
            }
        }
    };

    // Fill the requested layers either from a contiguous chunk or from an
    // explicit index list. The textures are resident, so a gather is index math
    // only (no re-decode) — which is what lets a containerSource of SOG nodes
    // serve the LOD writer's per-unit gather.
    const read = (request: ReadRequest): Promise<void> => {
        if ((request.lod ?? 0) !== 0) {
            throw new Error('readSog: only lod 0 is supported');
        }
        const posOut = request.position ? new Float32Array(request.position.data) : null;
        const geoOut = request.geometric ? new Float32Array(request.geometric.data) : null;
        const colOut = request.color ? new Float32Array(request.color.data) : null;
        const othOut = request.other && otherL ? new Uint32Array(request.other.data) : null;
        if ('indices' in request) {
            const { indices, indexOffset, count: cnt } = request;
            for (let j = 0; j < cnt; j++) decodeInto(posOut, geoOut, colOut, othOut, indices[indexOffset + j], j);
        } else {
            const start = request.chunkIndex * chunkSize;
            const cnt = Math.min(chunkSize, count - start);
            if (cnt <= 0) {
                throw new Error(`readSog: chunkIndex ${request.chunkIndex} out of range`);
            }
            for (let i = 0; i < cnt; i++) decodeInto(posOut, geoOut, colOut, othOut, start + i, i);
        }
        return Promise.resolve();
    };

    return { meta: meta_, read, close: () => Promise.resolve() };
};

// Fetch + parse the SOG meta.json. The basename is used verbatim for the fetch
// so any URL querystring/fragment (e.g. a presigned `?token=...`) is preserved.
const fetchSogMeta = async (fileSystem: ReadFileSystem, filename: string) => {
    const baseDir = dirname(filename);
    const metaName = basename(filename);
    const resolve = (name: string) => (baseDir ? join(baseDir, name) : name);
    const metaBytes = await readFile(fileSystem, resolve(metaName));
    const rawMeta = JSON.parse(new TextDecoder().decode(metaBytes)) as MetaV2 | (MetaV1 & { version?: number });
    return { baseDir, resolve, rawMeta };
};

/**
 * Header-only SOG peek: parse `meta.json` and report the metadata a file-info
 * summary needs — gaussian count, SH bands, splat model and available layers — **without**
 * decoding any WebP texture. SOG always carries position/geometric/color; V3 adds
 * blind-data extras (and `opacityAboveOne`) when its meta declares them. V2/V3 only;
 * legacy V1 (no `version`) returns `null` so the caller can fall back to a full read.
 *
 * @param fileSystem - The file system to read meta.json from.
 * @param filename - Path to meta.json.
 * @returns The meta summary, or `null` for V1 files.
 */
const statSogSource = async (
    fileSystem: ReadFileSystem,
    filename: string
): Promise<Pick<
    ChunkSourceMetadata,
    | 'numGaussians'
    | 'numLods'
    | 'lodCounts'
    | 'shBands'
    | 'model'
    | 'availableLayers'
    | 'extraColumns'
    | 'opacityAboveOne'
> | null> => {
    const { rawMeta } = await fetchSogMeta(fileSystem, filename);
    if (rawMeta.version !== 2 && rawMeta.version !== 3) return null;
    const meta = rawMeta as MetaV2;
    const count = meta.count;
    const extraColumns = extraColumnsFromMeta(meta);
    const availableLayers = new Set<ChunkLayer>(['position', 'geometric', 'color']);
    if (extraColumns.length > 0) availableLayers.add('other');
    return {
        numGaussians: count,
        numLods: 1,
        lodCounts: [count],
        shBands: bandsFromMeta(meta),
        model: modelFromMeta(meta.model),
        availableLayers,
        extraColumns,
        ...(opacityAboveOneFromMeta(meta) ? { opacityAboveOne: true } : {})
    };
};

/**
 * Open a SOG scene as a lazy, layered {@link ChunkSource}.
 *
 * Current (V2) files decode chunk-native: the WebP textures + codebooks stay
 * resident and `read`/`readRows` expand on demand (see {@link readSogSourceV2}).
 * Legacy V1 files (no `version`, per-channel mins/maxs) are decoded eagerly by
 * {@link readSogV1} and bridged to a resident source — they're rare and
 * deprecated, so they keep the simpler whole-scene path.
 *
 * @param fileSystem - The file system to read from.
 * @param filename - Path to meta.json (relative texture paths resolved from its directory).
 * @param pool - Pool whose `chunkSize` sets the chunking granularity.
 * @param options - Options controlling progress reporting (V1 only).
 * @returns A `ChunkSource` over the SOG's gaussians.
 * @ignore
 */
const readSogSource = async (
    fileSystem: ReadFileSystem,
    filename: string,
    pool: ChunkDataPool,
    options: ReadSogOptions = {}
): Promise<ChunkSource> => {
    const { baseDir, resolve, rawMeta } = await fetchSogMeta(fileSystem, filename);
    const version = rawMeta.version;
    if (version === undefined) {
        // Legacy V1: eager decode, bridged resident (deprecated, low volume).
        return dataTableToChunkSource(await readSogV1(fileSystem, baseDir, rawMeta as MetaV1, options), pool.chunkSize);
    }
    if (version !== 2 && version !== 3) {
        throw new Error(`Unsupported SOG meta version: ${version}`);
    }
    return readSogSourceV2(rawMeta as MetaV2, fileSystem, resolve, pool);
};

/**
 * Read a SOG file as a `DataTable` (the eager, whole-scene representation).
 *
 * The DataTable-input adapter / byte-identity oracle: V2 materializes the native
 * {@link readSogSourceV2}; V1 is decoded directly by {@link readSogV1}. New code
 * that wants a streaming source should use {@link readSogSource}.
 *
 * @param fileSystem - The file system to read from.
 * @param filename - Path to meta.json (relative paths resolved from its directory).
 * @param options - Options controlling progress reporting.
 * @returns DataTable with Gaussian splat data.
 * @ignore
 */
const readSog = async (
    fileSystem: ReadFileSystem,
    filename: string,
    options: ReadSogOptions = {}
): Promise<DataTable> => {
    const { baseDir, resolve, rawMeta } = await fetchSogMeta(fileSystem, filename);
    const version = rawMeta.version;
    if (version === undefined) {
        return readSogV1(fileSystem, baseDir, rawMeta as MetaV1, options);
    }
    if (version !== 2 && version !== 3) {
        throw new Error(`Unsupported SOG meta version: ${version}`);
    }
    const pool = createChunkDataPool();
    const src = await readSogSourceV2(rawMeta as MetaV2, fileSystem, resolve, pool);
    return materializeToDataTable(src, pool);
};

export { readSog, readSogSource, statSogSource };
export type { ReadSogOptions };
