import { dataTableToChunkSource, materializeToDataTable } from '../compat/data-table';
import { type DataTable } from '../data-table';
import { basename, dirname, join, type ReadFileSystem, readFile } from '../io/read';
import {
    type ChunkReadRequest,
    type RowReadRequest,
    type ChunkSource,
    type ChunkSourceMetadata,
    type ChunkDataPool,
    type ChunkLayer,
    type LayerLayout,
    type SHBands,
    POSITION_STRIDE,
    GEOMETRIC_STRIDE,
    colorStride,
    positionFields,
    geometricFields,
    colorFields,
    createChunkDataPool
} from '../source';
import { Transform, WebPCodec } from '../utils';
import { readSogV1, type MetaV1 } from './read-sog-v1';

// V2 (current) SOG meta layout - codebook-based quantization for scales /
// sh0 / shN. Legacy V1 uses a different per-channel mins/maxs scheme and is
// handled separately in read-sog-v1.ts.
type MetaV2 = {
    version: 2;
    count: number;
    means: { mins: number[]; maxs: number[]; files: string[] };
    scales: { codebook: number[]; files: string[] };
    quats: { files: string[] };
    sh0: { codebook: number[]; files: string[] };
    shN?: { count: number; bands: number; codebook: number[]; files: string[] };
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

// Inverse of logTransform(x) = sign(x) * ln(|x| + 1)
const invLogTransform = (v: number) => {
    const a = Math.abs(v);
    const e = Math.exp(a) - 1; // |x|
    return v < 0 ? -e : e;
};

const unpackQuat = (px: number, py: number, pz: number, tag: number): [number, number, number, number] => {
    const maxComp = tag - 252;
    const a = px / 255 * 2 - 1;
    const b = py / 255 * 2 - 1;
    const c = pz / 255 * 2 - 1;
    const sqrt2 = Math.sqrt(2);
    const comps = [0, 0, 0, 0];
    const idx = [
        [1, 2, 3],
        [0, 2, 3],
        [0, 1, 3],
        [0, 1, 2]
    ][maxComp];
    comps[idx[0]] = a / sqrt2;
    comps[idx[1]] = b / sqrt2;
    comps[idx[2]] = c / sqrt2;
    // reconstruct max component to make unit length with positive sign
    const t = 1 - (comps[0] * comps[0] + comps[1] * comps[1] + comps[2] * comps[2] + comps[3] * comps[3]);
    comps[maxComp] = Math.sqrt(Math.max(0, t));
    return comps as [number, number, number, number];
};

const sigmoidInv = (y: number) => {
    const e = Math.min(1 - 1e-6, Math.max(1e-6, y));
    return Math.log(e / (1 - e));
};

const SH_COEFFS = [0, 3, 8, 15];

/**
 * Native (V2) SOG `ChunkSource`: the decoded WebP textures (RGBA) + codebooks /
 * SH palette are held resident, and `read`/`readRows` expand only the requested
 * gaussians on demand — the per-row inverse of `writeSog`'s texel encode. Peak
 * resident is the (quantized) textures (~tens of bytes/gaussian), not the f32
 * scene; a position-only pass costs one texel lookup per gaussian. WebP has no
 * sub-image random access, so each texture is decoded once up front.
 *
 * @param meta - The parsed V2 meta.
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
    const xMin = mins[0], xScale = (maxs[0] - mins[0]) || 1;
    const yMin = mins[1], yScale = (maxs[1] - mins[1]) || 1;
    const zMin = mins[2], zScale = (maxs[2] - mins[2]) || 1;

    // quats: 4 bytes/splat, last byte is the largest-component tag (252-255).
    const quats = decoder.decodeRGBA(await load(meta.quats.files[0]));
    const qr = quats.rgba;
    if (quats.width * quats.height < count) throw new Error('SOG quats texture too small for count');

    // scales: 3 bytes index the shared 256-entry codebook.
    const scales = decoder.decodeRGBA(await load(meta.scales.files[0]));
    const sl = scales.rgba;
    if (scales.width * scales.height < count) throw new Error('SOG scales texture too small for count');
    const sCode = new Float32Array(meta.scales.codebook);

    // sh0: 3 color bytes index the codebook; the 4th is a sigmoid'd opacity.
    const sh0 = decoder.decodeRGBA(await load(meta.sh0.files[0]));
    const c0 = sh0.rgba;
    if (sh0.width * sh0.height < count) throw new Error('SOG sh0 texture too small for count');
    const cCode = new Float32Array(meta.sh0.codebook);

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
    if (meta.shN && SH_COEFFS[meta.shN.bands] > 0) {
        shBands = meta.shN.bands as SHBands;
        shCoeffs = SH_COEFFS[shBands];
        restCount = shCoeffs * 3;
        shCodebook = new Float32Array(meta.shN.codebook);
        const cen = decoder.decodeRGBA(await load(meta.shN.files[0]));
        centroidsRGBA = cen.rgba; cW = cen.width; cH = cen.height;
        const lab = decoder.decodeRGBA(await load(meta.shN.files[1]));
        labelsRGBA = lab.rgba;
        if (lab.width * lab.height < count) throw new Error('SOG shN labels texture too small for count');
        if (cW !== 64 * shCoeffs) {
            throw new Error(`SOG shN centroids texture width ${cW} does not match expected ${64 * shCoeffs} for ${shBands}-band palette`);
        }
        paletteCount = meta.shN.count;
    }
    const colorSw = 3 + restCount;

    const getCentroidPixel = (centroidIndex: number, coeff: number): [number, number, number] => {
        const cx = (centroidIndex % 64) * shCoeffs + coeff;
        const cy = Math.floor(centroidIndex / 64);
        if (cx >= cW || cy >= cH) return [0, 0, 0];
        const idx = (cy * cW + cx) * 4;
        return [centroidsRGBA![idx], centroidsRGBA![idx + 1], centroidsRGBA![idx + 2]];
    };

    const layouts: Partial<Record<ChunkLayer, LayerLayout>> = {
        position: { stride: POSITION_STRIDE, fields: positionFields() },
        geometric: { stride: GEOMETRIC_STRIDE, fields: geometricFields() },
        color: { stride: colorStride(shBands), fields: colorFields(shBands) }
    };

    const meta_: ChunkSourceMetadata = {
        numGaussians: count,
        numLods: 1,
        lodCounts: [count],
        chunkSize,
        numChunks: [count === 0 ? 0 : Math.ceil(count / chunkSize)],
        shBands,
        extraColumns: [],
        transform: Transform.PLY.clone(),
        availableLayers: new Set<ChunkLayer>(['position', 'geometric', 'color']),
        layouts
    };

    // Expand source gaussian `g` into output row `r` of the requested layers.
    // Shared by sequential `read` and random-access `readRows`. Every field of a
    // requested layer is written (pool buffers may hold prior data), so the SH
    // rest is explicitly zeroed for an out-of-range label rather than left stale.
    const decodeInto = (
        posOut: Float32Array | null,
        geoOut: Float32Array | null,
        colOut: Float32Array | null,
        g: number,
        r: number
    ): void => {
        const o4 = g * 4;

        if (posOut) {
            const o = r * 3;
            const xv = lo[o4] | (hi[o4] << 8);
            const yv = lo[o4 + 1] | (hi[o4 + 1] << 8);
            const zv = lo[o4 + 2] | (hi[o4 + 2] << 8);
            posOut[o]     = invLogTransform(xMin + xScale * (xv / 65535));
            posOut[o + 1] = invLogTransform(yMin + yScale * (yv / 65535));
            posOut[o + 2] = invLogTransform(zMin + zScale * (zv / 65535));
        }

        if (geoOut) {
            const o = r * 8; // [rot0..3, scale0..2, opacity]
            const tag = qr[o4 + 3];
            if (tag < 252 || tag > 255) {
                geoOut[o] = 1; geoOut[o + 1] = 0; geoOut[o + 2] = 0; geoOut[o + 3] = 0;
            } else {
                const [w, x, y, z] = unpackQuat(qr[o4], qr[o4 + 1], qr[o4 + 2], tag);
                geoOut[o] = w; geoOut[o + 1] = x; geoOut[o + 2] = y; geoOut[o + 3] = z;
            }
            geoOut[o + 4] = sCode[sl[o4]];
            geoOut[o + 5] = sCode[sl[o4 + 1]];
            geoOut[o + 6] = sCode[sl[o4 + 2]];
            geoOut[o + 7] = sigmoidInv(c0[o4 + 3] / 255);
        }

        if (colOut) {
            const o = r * colorSw; // [dc0..2, f_rest_0..N]
            colOut[o]     = cCode[c0[o4]];
            colOut[o + 1] = cCode[c0[o4 + 1]];
            colOut[o + 2] = cCode[c0[o4 + 2]];
            if (restCount > 0) {
                const label = labelsRGBA![o4] | (labelsRGBA![o4 + 1] << 8); // 16-bit palette index
                if (label < paletteCount) {
                    for (let j = 0; j < shCoeffs; j++) {
                        const [lr, lg, lb] = getCentroidPixel(label, j);
                        colOut[o + 3 + j]                = shCodebook![lr] ?? 0;
                        colOut[o + 3 + j + shCoeffs]     = shCodebook![lg] ?? 0;
                        colOut[o + 3 + j + shCoeffs * 2] = shCodebook![lb] ?? 0;
                    }
                } else {
                    for (let k = 0; k < restCount; k++) colOut[o + 3 + k] = 0;
                }
            }
        }
    };

    const read = (request: ChunkReadRequest): Promise<void> => {
        if ((request.lod ?? 0) !== 0) {
            throw new Error('readSog: only lod 0 is supported');
        }
        const start = request.chunkIndex * chunkSize;
        const cnt = Math.min(chunkSize, count - start);
        if (cnt <= 0) {
            throw new Error(`readSog: chunkIndex ${request.chunkIndex} out of range`);
        }
        const posOut = request.position ? new Float32Array(request.position.data) : null;
        const geoOut = request.geometric ? new Float32Array(request.geometric.data) : null;
        const colOut = request.color ? new Float32Array(request.color.data) : null;
        for (let i = 0; i < cnt; i++) decodeInto(posOut, geoOut, colOut, start + i, i);
        return Promise.resolve();
    };

    // Random-access gather (LOD 0): expand arbitrary texels into packed output
    // rows. The textures are resident, so this is index math only (no re-decode) —
    // which is what lets a containerSource of SOG nodes serve the LOD writer.
    const readRows = (request: RowReadRequest): Promise<void> => {
        if ((request.lod ?? 0) !== 0) {
            throw new Error('readSog: readRows only supports lod 0');
        }
        const { indices, indexOffset, count: cnt } = request;
        const posOut = request.position ? new Float32Array(request.position.data) : null;
        const geoOut = request.geometric ? new Float32Array(request.geometric.data) : null;
        const colOut = request.color ? new Float32Array(request.color.data) : null;
        for (let j = 0; j < cnt; j++) decodeInto(posOut, geoOut, colOut, indices[indexOffset + j], j);
        return Promise.resolve();
    };

    return { meta: meta_, read, readRows, close: () => Promise.resolve() };
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
    if (version !== 2) {
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
const readSog = async (fileSystem: ReadFileSystem, filename: string, options: ReadSogOptions = {}): Promise<DataTable> => {
    const { baseDir, resolve, rawMeta } = await fetchSogMeta(fileSystem, filename);
    const version = rawMeta.version;
    if (version === undefined) {
        return readSogV1(fileSystem, baseDir, rawMeta as MetaV1, options);
    }
    if (version !== 2) {
        throw new Error(`Unsupported SOG meta version: ${version}`);
    }
    const pool = createChunkDataPool();
    const src = await readSogSourceV2(rawMeta as MetaV2, fileSystem, resolve, pool);
    return materializeToDataTable(src, pool);
};

export { readSog, readSogSource };
export type { ReadSogOptions };
