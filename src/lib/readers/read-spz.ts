import { decompress as decompressZstd } from 'fzstd';

import { fileChunkSource } from './reader-utils';
import { ReadSource } from '../io/read';
import {
    type ReadRequest,
    type ChunkSource,
    type ChunkSourceMetadata,
    type ChunkDataPool,
    type ChunkLayer,
    type LayerLayout,
    type SHBands,
    SH_REST_COUNTS,
    POSITION_STRIDE,
    GEOMETRIC_STRIDE,
    colorStride,
    positionFields,
    geometricFields,
    colorFields
} from '../source';
import { Transform } from '../utils';

// Niantic Spatial .spz format. Pure-JS, lazy/layered decode (no WASM): v2/v3 are
// a single gzip stream with a 16-byte header; v4 is a 32-byte plaintext header +
// TOC + per-attribute ZSTD streams. The compact packed attribute bytes are held
// resident (decompressed once); each `read` unpacks only the requested layers
// for the requested chunk range — peak memory is the packed blob, not the f32
// scene, and a position-only pass never touches color/SH. Decode is RAW
// (CoordinateSystem UNSPECIFIED) — no coordinate conversion, matching the spec
// default and the `from: UNSPECIFIED` writer. See https://github.com/nianticlabs/spz.

// gzip (v2/v3) via the web-standard streams API — browser + Node, no WASM.
const decompressGzip = async (data: Uint8Array): Promise<Uint8Array> => {
    const ab = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer;
    const stream = new Blob([ab]).stream().pipeThrough(new DecompressionStream('gzip'));
    return new Uint8Array(await new Response(stream).arrayBuffer());
};

// spz quantizes SH-DC colors with this scale (not SH_C0); invert on read.
const SPZ_COLOR_SCALE = 0.15;
const inverseColorFromSpz = (byte: number): number => (byte / 255 - 0.5) / SPZ_COLOR_SCALE;

// Read one signed 24-bit fixed-point position component (little-endian, 3 bytes).
const getFixed24 = (view: DataView, element: number, member: number): number => {
    const o = element * 9 + member * 3;
    let v = view.getUint8(o) | (view.getUint8(o + 1) << 8) | (view.getUint8(o + 2) << 16);
    if (v & 0x800000) v |= 0xff000000; // sign-extend
    return v;
};

// Total SH coefficients (all 3 channels) per gaussian, by SH degree 0..4.
const HARMONICS_COMPONENT_COUNT = [0, 9, 24, 45, 72];

const MIN_SPZ_VERSION = 2;
const LATEST_SPZ_VERSION = 4;
const SPZ_VERSION_ZSTD = 4; // v4+ switched to per-stream ZSTD + 32-byte header

// The resident, decompressed per-attribute byte views for one scene.
type SpzStreams = {
    version: number;
    numSplats: number;
    shBands: SHBands;
    fractionalBits: number;
    positions: DataView;
    alphas: DataView;
    colors: DataView;
    scales: DataView;
    rotations: DataView;
    sh: DataView;
};

// Parse + decompress the whole .spz into resident per-attribute views. Returns
// `null` for a structurally-malformed v4 container (→ empty source, not a crash).
const parseSpz = async (source: ReadSource): Promise<SpzStreams | null> => {
    let fileBuffer = await source.read().readAll();

    // Legacy v2/v3 files are a single gzip stream — inflate first.
    if (fileBuffer.length >= 2 && fileBuffer[0] === 0x1f && fileBuffer[1] === 0x8b) {
        fileBuffer = await decompressGzip(fileBuffer);
    }

    const totalSize = fileBuffer.length;
    const MIN_HEADER_SIZE = 16;
    if (totalSize < MIN_HEADER_SIZE) {
        throw new Error('File too small to be valid .spz format');
    }

    const magicView = new DataView(fileBuffer.buffer, fileBuffer.byteOffset, MIN_HEADER_SIZE);
    if (magicView.getUint32(0, true) !== 0x5053474e) { // 'NGSP'
        throw new Error('invalid .spz file header');
    }

    const version = magicView.getUint32(4, true);
    if (version < MIN_SPZ_VERSION || version > LATEST_SPZ_VERSION) {
        throw new Error(`Unsupported .spz version ${version}`);
    }

    const HEADER_SIZE = version >= SPZ_VERSION_ZSTD ? 32 : MIN_HEADER_SIZE;
    if (totalSize < HEADER_SIZE) {
        throw new Error('File too small to be valid .spz format');
    }

    const header = new DataView(fileBuffer.buffer, fileBuffer.byteOffset, HEADER_SIZE);
    const numSplats = header.getUint32(8, true);
    const shDegree = header.getUint8(12);
    const fractionalBits = header.getUint8(13);
    if (shDegree < 0 || shDegree >= HARMONICS_COMPONENT_COUNT.length) {
        throw new Error(`Unsupported SH degree ${shDegree}`);
    }
    if (shDegree > 3) {
        // SPZ degree 4 (72 coeffs) exceeds splat-transform's SH band-3 model.
        throw new Error('Unsupported .spz SH degree 4 (band 4); splat-transform supports up to band 3');
    }
    const shBands = shDegree as SHBands;

    const harmonicsComponentCount = HARMONICS_COMPONENT_COUNT[shDegree];
    const positionsByteSize = numSplats * 9;                                 // 3 × int24
    const alphasByteSize = numSplats;                                        // u8
    const colorsByteSize = numSplats * 3;                                    // u8 × 3
    const scalesByteSize = numSplats * 3;                                    // u8 × 3
    const rotationsByteSize = numSplats * (version === MIN_SPZ_VERSION ? 3 : 4);
    const shByteSize = numSplats * harmonicsComponentCount;

    if (version >= SPZ_VERSION_ZSTD) {
        // 32-byte header, then optional extensions, then TOC, then ZSTD streams.
        // A malformed container yields null (→ empty source) rather than a crash.
        const numStreams = header.getUint8(15);
        const tocByteOffset = header.getUint32(16, true);
        const expectedNumStreams = shDegree === 0 ? 5 : 6;
        const expectedSizes = [positionsByteSize, alphasByteSize, colorsByteSize, scalesByteSize, rotationsByteSize];
        if (shDegree > 0) expectedSizes.push(shByteSize);

        if (numStreams !== expectedNumStreams || tocByteOffset < HEADER_SIZE) return null;
        const dataStart = tocByteOffset + numStreams * 16;
        if (dataStart > totalSize) return null;

        const streams: DataView[] = [];
        let dataOffset = dataStart;
        for (let i = 0; i < numStreams; i++) {
            const toc = new DataView(fileBuffer.buffer, fileBuffer.byteOffset + tocByteOffset + i * 16, 16);
            const compressedSize = Number(toc.getBigUint64(0, true));
            const uncompressedSize = Number(toc.getBigUint64(8, true));
            if (uncompressedSize !== expectedSizes[i]) return null;
            if (dataOffset + compressedSize > totalSize) return null;
            let decompressed: Uint8Array;
            try {
                decompressed = decompressZstd(fileBuffer.subarray(dataOffset, dataOffset + compressedSize));
            } catch {
                return null;
            }
            if (decompressed.byteLength !== uncompressedSize) return null;
            streams.push(new DataView(decompressed.buffer, decompressed.byteOffset, decompressed.byteLength));
            dataOffset += compressedSize;
        }

        return {
            version,
            numSplats,
            shBands,
            fractionalBits,
            positions: streams[0],
            alphas: streams[1],
            colors: streams[2],
            scales: streams[3],
            rotations: streams[4],
            sh: numStreams > 5 ? streams[5] : new DataView(new ArrayBuffer(0))
        };
    }

    // v2/v3: the (already-gunzipped) buffer holds the attribute blocks sequentially.
    const required = HEADER_SIZE + positionsByteSize + alphasByteSize + colorsByteSize + scalesByteSize + rotationsByteSize + shByteSize;
    if (totalSize < required) {
        throw new Error(`File too small for .spz payload: expected at least ${required} bytes, got ${totalSize}`);
    }
    const buf = fileBuffer.buffer;
    let off = fileBuffer.byteOffset + HEADER_SIZE;
    const positions = new DataView(buf, off, positionsByteSize); off += positionsByteSize;
    const alphas = new DataView(buf, off, alphasByteSize); off += alphasByteSize;
    const colors = new DataView(buf, off, colorsByteSize); off += colorsByteSize;
    const scales = new DataView(buf, off, scalesByteSize); off += scalesByteSize;
    const rotations = new DataView(buf, off, rotationsByteSize); off += rotationsByteSize;
    const sh = new DataView(buf, off, shByteSize);
    return { version, numSplats, shBands, fractionalBits, positions, alphas, colors, scales, rotations, sh };
};

// Reusable scratch for smallest-three quaternion decoding.
const tmpQuat = [0.0, 0.0, 0.0, 0.0];

/**
 * Read a Niantic Spatial `.spz` file as a lazy, layered {@link ChunkSource}.
 *
 * Pure-JS: gzip (v2/v3) via `DecompressionStream`, ZSTD (v4) via `fzstd` — no
 * WASM, no in-memory size ceiling, identical Node/browser. The decompressed
 * (compact) attribute bytes stay resident; each `read` unpacks only the
 * requested layers for the chunk. Decoding is raw (no coordinate conversion);
 * the source carries a pending `Transform.PLY`. SH degree 4 is rejected (the
 * chunk model supports up to band 3).
 *
 * @param source - The read source for the `.spz` file.
 * @param pool - Pool whose `chunkSize` sets the gaussians-per-chunk granularity.
 * @returns A lazy source over the file's gaussians.
 * @ignore
 */
const readSpz = async (source: ReadSource, pool: ChunkDataPool): Promise<ChunkSource> => {
    const streams = await parseSpz(source);

    const chunkSize = pool.chunkSize;
    const numGaussians = streams ? streams.numSplats : 0;
    const shBands: SHBands = streams ? streams.shBands : 0;
    const restCount = SH_REST_COUNTS[shBands];
    const colStride32 = 3 + restCount;
    const numChunks = numGaussians === 0 ? 0 : Math.ceil(numGaussians / chunkSize);

    const layouts: Partial<Record<ChunkLayer, LayerLayout>> = {
        position: { stride: POSITION_STRIDE, fields: positionFields() },
        geometric: { stride: GEOMETRIC_STRIDE, fields: geometricFields() },
        color: { stride: colorStride(shBands), fields: colorFields(shBands) }
    };

    const meta: ChunkSourceMetadata = {
        numGaussians,
        numLods: 1,
        lodCounts: [numGaussians],
        chunkSize,
        numChunks: [numChunks],
        shBands,
        extraColumns: [],
        transform: Transform.PLY.clone(),
        availableLayers: new Set<ChunkLayer>(['position', 'geometric', 'color']),
        layouts
    };

    // Per-source-index decode constants (fixed for the whole scene).
    const positionScale = streams ? 1.0 / (1 << streams.fractionalBits) : 1;
    const perChannel = restCount / 3;

    // Unpack one source gaussian `s` into output row `r` of whichever layer
    // buffers are present. Shared by sequential `read` (s = start + i, r = i) and
    // random-access `readRows` (s = indices[..], r = j), so both decode identically.
    const decodeInto = (
        posOut: Float32Array | null,
        geoOut: Float32Array | null,
        colOut: Float32Array | null,
        s: number,
        r: number
    ): void => {
        const { version, positions, alphas, colors, scales, rotations, sh } = streams!;

        if (posOut) {
            const o = r * 3;
            posOut[o]     = getFixed24(positions, s, 0) * positionScale;
            posOut[o + 1] = getFixed24(positions, s, 1) * positionScale;
            posOut[o + 2] = getFixed24(positions, s, 2) * positionScale;
        }

        if (geoOut) {
            const o = r * 8; // [rot0..3, scale0..2, opacity]

            if (version === MIN_SPZ_VERSION) {
                // v2: 3 × int8 (xyz), w derived
                const x = rotations.getUint8(s * 3 + 0) / 127.5 - 1.0;
                const y = rotations.getUint8(s * 3 + 1) / 127.5 - 1.0;
                const z = rotations.getUint8(s * 3 + 2) / 127.5 - 1.0;
                geoOut[o]     = Math.sqrt(Math.max(0.0, 1.0 - (x * x + y * y + z * z)));
                geoOut[o + 1] = x;
                geoOut[o + 2] = y;
                geoOut[o + 3] = z;
            } else {
                // v3/v4: smallest-three — 2-bit largest index + three 10-bit signed (9-bit mag + sign)
                tmpQuat[0] = tmpQuat[1] = tmpQuat[2] = tmpQuat[3] = 0.0;
                let packed = rotations.getUint32(s * 4, true);
                const cMask = (1 << 9) - 1;
                const largest = packed >>> 30;
                let sumSq = 0;
                for (let j = 3; j >= 0; --j) {
                    if (j !== largest) {
                        const mag = packed & cMask;
                        const neg = (packed >>> 9) & 1;
                        packed >>>= 10;
                        let v = Math.SQRT1_2 * mag / cMask;
                        if (neg === 1) v = -v;
                        tmpQuat[j] = v;
                        sumSq += v * v;
                    }
                }
                tmpQuat[largest] = Math.sqrt(Math.max(0.0, 1.0 - sumSq));
                geoOut[o]     = tmpQuat[3]; // w
                geoOut[o + 1] = tmpQuat[0]; // x
                geoOut[o + 2] = tmpQuat[1]; // y
                geoOut[o + 3] = tmpQuat[2]; // z
            }

            // scales: 8-bit log-encoded
            geoOut[o + 4] = scales.getUint8(s * 3 + 0) / 16.0 - 10.0;
            geoOut[o + 5] = scales.getUint8(s * 3 + 1) / 16.0 - 10.0;
            geoOut[o + 6] = scales.getUint8(s * 3 + 2) / 16.0 - 10.0;

            // opacity: u8 -> inverse sigmoid (logit), clamped off the asymptotes
            const eps = 1e-6;
            const a = Math.max(eps, Math.min(1 - eps, alphas.getUint8(s) / 255.0));
            geoOut[o + 7] = Math.log(a / (1 - a));
        }

        if (colOut) {
            const o = r * colStride32; // [dc0..2, f_rest_0..N]
            colOut[o]     = inverseColorFromSpz(colors.getUint8(s * 3 + 0));
            colOut[o + 1] = inverseColorFromSpz(colors.getUint8(s * 3 + 1));
            colOut[o + 2] = inverseColorFromSpz(colors.getUint8(s * 3 + 2));
            // spherical harmonics: 8-bit signed, channel-inner -> de-interleave to f_rest
            for (let c = 0; c < restCount; c++) {
                const channel = c % 3;
                const coeff = (c / 3) | 0;
                const col = channel * perChannel + coeff;
                colOut[o + 3 + col] = (sh.getUint8(s * restCount + c) - 128) / 128;
            }
        }
    };

    // Fill the requested layers either from a contiguous chunk or from an
    // explicit index list. The compact streams are resident, so a gather is pure
    // index math — no re-decompress — which is what lets a containerSource of SPZ
    // nodes serve the LOD writer's per-unit gather.
    const read = (request: ReadRequest): Promise<void> => {
        if (!streams) return Promise.resolve();
        if ((request.lod ?? 0) !== 0) {
            throw new Error('readSpz: only lod 0 is supported');
        }
        const posOut = request.position ? new Float32Array(request.position.data) : null;
        const geoOut = request.geometric ? new Float32Array(request.geometric.data) : null;
        const colOut = request.color ? new Float32Array(request.color.data) : null;
        if ('indices' in request) {
            const { indices, indexOffset, count } = request;
            for (let j = 0; j < count; j++) {
                decodeInto(posOut, geoOut, colOut, indices[indexOffset + j], j);
            }
        } else {
            const start = request.chunkIndex * chunkSize;
            const count = Math.min(chunkSize, numGaussians - start);
            if (count <= 0) {
                throw new Error(`readSpz: chunkIndex ${request.chunkIndex} out of range`);
            }
            for (let i = 0; i < count; i++) {
                decodeInto(posOut, geoOut, colOut, start + i, i);
            }
        }
        return Promise.resolve();
    };

    return fileChunkSource(source, meta, read);
};

export { readSpz };
