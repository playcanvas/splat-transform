import { isCompressedPly, decompressPly } from './decompress-ply';
import { fileChunkSource, readExact } from './reader-utils';
import {
    type ChunkData,
    type ChunkLayer,
    type ReadRequest,
    type ChunkSource,
    type ChunkSourceMetadata,
    type ChunkDataPool,
    type ExtraColumn,
    type LayerLayout,
    type SHBands,
    SH_REST_COUNTS,
    POSITION_STRIDE,
    GEOMETRIC_STRIDE,
    colorStride,
    positionFields,
    geometricFields,
    colorFields,
    otherLayout
} from '../chunk';
import { Column, DataTable } from '../data-table';
import { type ReadSource, type ReadStream } from '../io/read';
import { logger, Transform } from '../utils';

type PlyProperty = {
    name: string;               // 'x', 'f_dc_0', etc
    type: string;               // 'float', 'char', etc
};

type PlyElement = {
    name: string;               // 'vertex', etc
    count: number;
    properties: PlyProperty[];
};

type PlyHeader = {
    comments: string[];
    elements: PlyElement[];
    headerBytes: number;        // byte offset at which binary data begins
    format: string;             // 'binary_little_endian' | 'binary_big_endian' | 'ascii'
};

// A whole-PLY decode result: one DataTable per element (consumed by decompressPly).
type PlyData = {
    comments: string[];
    elements: {
        name: string,
        dataTable: DataTable
    }[];
};

const GEOMETRIC_COLS = ['rot_0', 'rot_1', 'rot_2', 'rot_3', 'scale_0', 'scale_1', 'scale_2', 'opacity'];
const COLOR_DC_COLS = ['f_dc_0', 'f_dc_1', 'f_dc_2'];

const getDataType = (type: string) => {
    switch (type) {
        case 'char': return Int8Array;
        case 'uchar': return Uint8Array;
        case 'short': return Int16Array;
        case 'ushort': return Uint16Array;
        case 'int': return Int32Array;
        case 'uint': return Uint32Array;
        case 'float': return Float32Array;
        case 'double': return Float64Array;
        default: return null;
    }
};

// Reads one numeric value from a DataView at a byte offset (little-endian).
type ValueReader = (view: DataView, offset: number) => number;

const getReader = (type: string): ValueReader => {
    switch (type) {
        case 'char':   return (v, o) => v.getInt8(o);
        case 'uchar':  return (v, o) => v.getUint8(o);
        case 'short':  return (v, o) => v.getInt16(o, true);
        case 'ushort': return (v, o) => v.getUint16(o, true);
        case 'int':    return (v, o) => v.getInt32(o, true);
        case 'uint':   return (v, o) => v.getUint32(o, true);
        case 'float':  return (v, o) => v.getFloat32(o, true);
        case 'double': return (v, o) => v.getFloat64(o, true);
        default: throw new Error(`readPly: unsupported ply type '${type}'`);
    }
};

const typeSize = (type: string): number => {
    switch (type) {
        case 'char': case 'uchar': return 1;
        case 'short': case 'ushort': return 2;
        case 'int': case 'uint': case 'float': return 4;
        case 'double': return 8;
        default: throw new Error(`readPly: unsupported ply type '${type}'`);
    }
};

const rowSizeOf = (properties: PlyProperty[]): number => properties.reduce((s, p) => s + typeSize(p.type), 0);

const endHeaderMarker = new Uint8Array([10, 101, 110, 100, 95, 104, 101, 97, 100, 101, 114, 10]); // \nend_header\n

const indexOfMarker = (buf: Uint8Array, marker: Uint8Array, limit: number): number => {
    for (let i = 0; i + marker.length <= limit; i++) {
        let match = true;
        for (let j = 0; j < marker.length; j++) {
            if (buf[i + j] !== marker[j]) {
                match = false;
                break;
            }
        }
        if (match) return i;
    }
    return -1;
};

// Parse the PLY header from a 128 KB probe at the start of the source: returns
// every element (name/count/properties), comments, and the byte offset where
// binary data begins. Tolerates trailing/duplicate whitespace (some tools write
// space-padded counts). The source must be seekable.
const readHeader = async (source: ReadSource): Promise<PlyHeader> => {
    const probeLen = Math.min(128 * 1024, source.size ?? 128 * 1024);
    const buf = new Uint8Array(probeLen);
    const read = await readExact(source.read(0, probeLen), buf, 0, probeLen);

    const markerIdx = indexOfMarker(buf, endHeaderMarker, read);
    if (markerIdx < 0) {
        throw new Error('readPly: end_header not found within 128KB probe (non-PLY?)');
    }
    const headerBytes = markerIdx + endHeaderMarker.length;

    const lines = new TextDecoder('ascii').decode(buf.subarray(0, headerBytes)).split('\n').filter(Boolean);
    if (lines[0] !== 'ply') {
        throw new Error('readPly: invalid PLY header');
    }

    const elements: PlyElement[] = [];
    const comments: string[] = [];
    let format = '';
    let current: PlyElement | null = null;
    for (let i = 1; i < lines.length; i++) {
        const words = lines[i].split(' ').filter(Boolean);
        switch (words[0]) {
            case 'ply': case 'end_header':
                break;
            case 'format':
                format = words[1] ?? '';
                break;
            case 'comment':
                comments.push(lines[i].substring(8)); // skip 'comment '
                break;
            case 'element':
                if (words.length !== 3) throw new Error('readPly: invalid element line');
                current = { name: words[1], count: parseInt(words[2], 10), properties: [] };
                elements.push(current);
                break;
            case 'property':
                if (!current || words.length !== 3 || !getDataType(words[1])) throw new Error('readPly: invalid property line');
                current.properties.push({ name: words[2], type: words[1] });
                break;
            default:
                throw new Error(`readPly: unrecognized header line '${words[0]}'`);
        }
    }

    return { comments, elements, headerBytes, format };
};

// Decode `count` rows of an element (the given properties) from a stream into a
// DataTable. Fast path when every property is float (Float32Array view); general
// path for mixed types via DataView readers. `onProgress(rowsDoneInElement)` is
// called once per 1024-row block.
const decodeElement = async (
    stream: ReadStream,
    properties: PlyProperty[],
    count: number,
    onProgress?: (rows: number) => void
): Promise<DataTable> => {
    const columns = properties.map(p => new Column(p.name, new (getDataType(p.type)!)(count)));
    const numProperties = columns.length;
    const allFloat = properties.every(p => p.type === 'float');
    const blockRows = 1024;
    const numBlocks = Math.ceil(count / blockRows);

    if (allFloat) {
        const rowSize = numProperties * 4;
        const blockData = new Uint8Array(blockRows * rowSize);
        const floatData = new Float32Array(blockData.buffer);
        const storage = columns.map(c => c.data as Float32Array);

        for (let b = 0; b < numBlocks; ++b) {
            const rows = Math.min(blockRows, count - b * blockRows);
            const base = b * blockRows;
            await readExact(stream, blockData, 0, rowSize * rows);
            for (let p = 0; p < numProperties; ++p) {
                const s = storage[p];
                for (let r = 0; r < rows; ++r) s[base + r] = floatData[r * numProperties + p];
            }
            onProgress?.(base + rows);
        }
    } else {
        let byteOffset = 0;
        const columnInfo = properties.map((property, idx) => {
            const info = { data: columns[idx].data, byteOffset, reader: getReader(property.type) };
            byteOffset += columns[idx].data.BYTES_PER_ELEMENT;
            return info;
        });
        const rowSize = byteOffset;
        const blockData = new Uint8Array(blockRows * rowSize);

        for (let b = 0; b < numBlocks; ++b) {
            const rows = Math.min(blockRows, count - b * blockRows);
            const base = b * blockRows;
            await readExact(stream, blockData, 0, rowSize * rows);
            const view = new DataView(blockData.buffer, blockData.byteOffset, blockData.byteLength);
            for (let r = 0; r < rows; ++r) {
                const rowByteOffset = r * rowSize;
                for (let p = 0; p < columnInfo.length; ++p) {
                    const info = columnInfo[p];
                    info.data[base + r] = info.reader(view, rowByteOffset + info.byteOffset);
                }
            }
            onProgress?.(base + rows);
        }
    }

    return new DataTable(columns);
};

/**
 * Decode a whole PLY file (standard or compressed) to a `DataTable`.
 *
 * Internal eager decode. {@link readPly} is the public reader; standard PLY is
 * read lazily and compressed PLY is read chunk-by-chunk (see `readCompressedChunked`),
 * so neither uses this path. Retained as the independent test oracle.
 *
 * @param source - A seekable read source over the PLY file.
 * @returns Promise resolving to a DataTable containing the splat data.
 * @ignore
 */
const decodePlyToDataTable = async (source: ReadSource): Promise<DataTable> => {
    const { comments, elements: headerElements, headerBytes } = await readHeader(source);
    const stream = source.read(headerBytes);

    // Single decode bar across all elements (e.g. compressed PLY: chunk + vertex).
    const totalRows = headerElements.reduce((sum, e) => sum + e.count, 0);
    const bar = logger.bar('decoding', totalRows);

    let prior = 0;
    const decoded = [];
    for (const element of headerElements) {
        const priorRows = prior; // snapshot for the progress closure (prior is reassigned below)
        const dataTable = await decodeElement(stream, element.properties, element.count, r => bar.update(priorRows + r));
        prior += element.count;
        decoded.push({ name: element.name, dataTable });
    }

    const plyData = { comments, elements: decoded };

    let result: DataTable;
    if (isCompressedPly(plyData)) {
        result = decompressPly(plyData);
    } else {
        const vertexElement = plyData.elements.find(e => e.name === 'vertex');
        if (!vertexElement) {
            throw new Error('PLY file does not contain vertex element');
        }
        result = vertexElement.dataTable;
    }

    result.transform = Transform.PLY.clone();

    // Close the bar only on success: leaving it open on an earlier error path
    // lets `logger.error() -> unwindAll(true)` mark it as failed.
    bar.end();

    return result;
};

// --- Compressed-PLY dequant primitives (mirror decompress-ply.ts) ---
const SH_C0 = 0.28209479177387814;
const ROT_NORM = 1.0 / (Math.sqrt(2) * 0.5);
const unpackUnorm = (value: number, bits: number): number => {
    const t = (1 << bits) - 1;
    return (value & t) / t;
};
const lerp = (a: number, b: number, t: number): number => a * (1 - t) + b * t;

// Lazy reader for compressed PLY (packed chunk + vertex [+ sh] elements). The
// small `chunk` bounds element is read resident once (indexed by absolute
// 256-block); each `read` range-reads only that output-chunk's packed vertex
// (+ sh, for color) rows and dequantizes **just the requested layer's
// attributes** straight into the layer buffers — no whole-scene decode, no
// intermediate DataTable, no per-gaussian object churn.
const readCompressedChunked = (source: ReadSource, header: PlyHeader, pool: ChunkDataPool): ChunkSource => {
    const chunkEl = header.elements.find(e => e.name === 'chunk');
    const vertexEl = header.elements.find(e => e.name === 'vertex');
    const shEl = header.elements.find(e => e.name === 'sh');
    if (!chunkEl || !vertexEl) {
        throw new Error('readPly: compressed PLY missing chunk/vertex element');
    }

    const chunkRowSize = rowSizeOf(chunkEl.properties);
    const vertexRowSize = rowSizeOf(vertexEl.properties);
    const shRowSize = shEl ? rowSizeOf(shEl.properties) : 0;

    // word index of each packed vertex column within the (all-uint32) record
    const vWord = (name: string): number => {
        let off = 0;
        for (const p of vertexEl.properties) {
            if (p.name === name) return off >> 2;
            off += typeSize(p.type);
        }
        throw new Error(`readPly: compressed vertex missing '${name}'`);
    };
    const ppIdx = vWord('packed_position');
    const prIdx = vWord('packed_rotation');
    const psIdx = vWord('packed_scale');
    const pcIdx = vWord('packed_color');
    const vWords = vertexRowSize >> 2;

    // Binary offset of an element (header order, immediately after the header).
    const elementOffset = (target: PlyElement): number => {
        let off = header.headerBytes;
        for (const e of header.elements) {
            if (e === target) return off;
            off += e.count * rowSizeOf(e.properties);
        }
        throw new Error('readPly: element not found');
    };
    const chunkOffset = elementOffset(chunkEl);
    const vertexOffset = elementOffset(vertexEl);
    const shOffset = shEl ? elementOffset(shEl) : 0;

    const numGaussians = vertexEl.count;
    const restCount = shEl ? shEl.properties.length : 0;
    const shBands: SHBands = (restCount === 0 ? 0 : ({ [SH_REST_COUNTS[1]]: 1, [SH_REST_COUNTS[2]]: 2, [SH_REST_COUNTS[3]]: 3 } as Record<number, SHBands>)[restCount]);
    if (shBands === undefined) {
        throw new Error(`readPly: unrecognized compressed sh column count ${restCount}`);
    }
    // byte offset of each f_rest_* within the sh record (uint8 columns)
    const shByteOffset = (name: string): number => {
        let off = 0;
        for (const p of shEl!.properties) {
            if (p.name === name) return off;
            off += typeSize(p.type);
        }
        throw new Error(`readPly: compressed sh missing '${name}'`);
    };
    const shOff = shEl ? Array.from({ length: restCount }, (_, r) => shByteOffset(`f_rest_${r}`)) : [];

    const chunkSize = pool.chunkSize;
    const numChunks = Math.max(1, Math.ceil(numGaussians / chunkSize));

    const availableLayers = new Set<ChunkLayer>(['position', 'geometric', 'color']);
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
        availableLayers,
        layouts
    };

    // Per-256-block bounds, read resident once (small: count/256 rows).
    let bounds: DataTable | null = null;
    const ensureBounds = async (): Promise<DataTable> => {
        bounds ??= await decodeElement(
            source.read(chunkOffset, chunkOffset + chunkEl.count * chunkRowSize),
            chunkEl.properties,
            chunkEl.count
        );
        return bounds;
    };

    // Reusable scratch (reads are sequential): `vScratch`/`sScratch` for the
    // contiguous chunk read, `gatherV`/`gatherS` for one coalesced gather run.
    let vScratch: Uint8Array | null = null;
    let sScratch: Uint8Array | null = null;
    let gatherV: Uint8Array | null = null;
    let gatherS: Uint8Array | null = null;

    const read = async (request: ReadRequest): Promise<void> => {
        if ((request.lod ?? 0) !== 0) {
            throw new Error('readPly: only lod 0 is supported');
        }

        const pos = request.position ? new Float32Array(request.position.data) : null;
        const geo = request.geometric ? new Float32Array(request.geometric.data) : null;
        const col = request.color ? new Float32Array(request.color.data) : null;
        if (!pos && !geo && !col) return;

        const bnd = await ensureBounds();
        const bcol = (name: string): Float32Array => bnd.getColumnByName(name)!.data as Float32Array;

        // Bound columns, fetched once per read; indexed by 256-block `gi >> 8`.
        const minX = pos ? bcol('min_x') : null, minY = pos ? bcol('min_y') : null, minZ = pos ? bcol('min_z') : null;
        const maxX = pos ? bcol('max_x') : null, maxY = pos ? bcol('max_y') : null, maxZ = pos ? bcol('max_z') : null;
        const minSX = geo ? bcol('min_scale_x') : null, minSY = geo ? bcol('min_scale_y') : null, minSZ = geo ? bcol('min_scale_z') : null;
        const maxSX = geo ? bcol('max_scale_x') : null, maxSY = geo ? bcol('max_scale_y') : null, maxSZ = geo ? bcol('max_scale_z') : null;
        const sw = 3 + restCount;
        const hasColors = col ? bnd.hasColumn('min_r') : false;
        const minR = hasColors ? bcol('min_r') : null, minG = hasColors ? bcol('min_g') : null, minB = hasColors ? bcol('min_b') : null;
        const maxR = hasColors ? bcol('max_r') : null, maxG = hasColors ? bcol('max_g') : null, maxB = hasColors ? bcol('max_b') : null;

        // Decode one packed record: vertex words at local index `vi` in `v`, sh
        // bytes at local index `si` in `sBytes`, bounds from global index `gi`,
        // written to output row `dstRow`. Shared by the chunk and gather paths.
        const decodeRecord = (v: Uint32Array, vi: number, sBytes: Uint8Array | null, si: number, gi: number, dstRow: number): void => {
            const ci = gi >> 8;
            const base = vi * vWords;
            if (pos) {
                const pp = v[base + ppIdx];
                const o = dstRow * 3;
                pos[o]     = lerp(minX![ci], maxX![ci], unpackUnorm(pp >>> 21, 11));
                pos[o + 1] = lerp(minY![ci], maxY![ci], unpackUnorm(pp >>> 11, 10));
                pos[o + 2] = lerp(minZ![ci], maxZ![ci], unpackUnorm(pp, 11));
            }
            if (geo) {
                const o = dstRow * 8;
                // rotation (largest-3 quaternion)
                const pr = v[base + prIdx];
                const ra = (unpackUnorm(pr >>> 20, 10) - 0.5) * ROT_NORM;
                const rb = (unpackUnorm(pr >>> 10, 10) - 0.5) * ROT_NORM;
                const rc = (unpackUnorm(pr, 10) - 0.5) * ROT_NORM;
                const rm = Math.sqrt(Math.max(0, 1 - (ra * ra + rb * rb + rc * rc)));
                switch (pr >>> 30) {
                    case 0: geo[o] = rm; geo[o + 1] = ra; geo[o + 2] = rb; geo[o + 3] = rc; break;
                    case 1: geo[o] = ra; geo[o + 1] = rm; geo[o + 2] = rb; geo[o + 3] = rc; break;
                    case 2: geo[o] = ra; geo[o + 1] = rb; geo[o + 2] = rm; geo[o + 3] = rc; break;
                    default: geo[o] = ra; geo[o + 1] = rb; geo[o + 2] = rc; geo[o + 3] = rm;
                }
                // scale
                const ps = v[base + psIdx];
                geo[o + 4] = lerp(minSX![ci], maxSX![ci], unpackUnorm(ps >>> 21, 11));
                geo[o + 5] = lerp(minSY![ci], maxSY![ci], unpackUnorm(ps >>> 11, 10));
                geo[o + 6] = lerp(minSZ![ci], maxSZ![ci], unpackUnorm(ps, 11));
                // opacity (alpha of packed_color)
                const cw = unpackUnorm(v[base + pcIdx], 8);
                geo[o + 7] = -Math.log(1 / cw - 1);
            }
            if (col) {
                const o = dstRow * sw;
                const pc = v[base + pcIdx];
                const cx = unpackUnorm(pc >>> 24, 8);
                const cy = unpackUnorm(pc >>> 16, 8);
                const cz = unpackUnorm(pc >>> 8, 8);
                const cr = hasColors ? lerp(minR![ci], maxR![ci], cx) : cx;
                const cg = hasColors ? lerp(minG![ci], maxG![ci], cy) : cy;
                const cb = hasColors ? lerp(minB![ci], maxB![ci], cz) : cz;
                col[o] = (cr - 0.5) / SH_C0;
                col[o + 1] = (cg - 0.5) / SH_C0;
                col[o + 2] = (cb - 0.5) / SH_C0;
                if (sBytes) {
                    const sb = si * shRowSize;
                    for (let r = 0; r < restCount; r++) {
                        const byte = sBytes[sb + shOff[r]];
                        const nrm = (byte === 0) ? 0 : (byte === 255) ? 1 : (byte + 0.5) / 256;
                        col[o + 3 + r] = (nrm - 0.5) * 8;
                    }
                }
            }
        };

        if ('indices' in request) {
            // Gather: sort output slots into source-file order, coalesce
            // consecutive records into one range read (vertex + parallel sh).
            const { indices, indexOffset, count } = request;
            if (count <= 0) return;
            const slot = new Uint32Array(count);
            for (let j = 0; j < count; j++) slot[j] = j;
            slot.sort((a, b) => indices[indexOffset + a] - indices[indexOffset + b]);

            let j = 0;
            while (j < count) {
                const first = indices[indexOffset + slot[j]];
                let k = j + 1;
                while (k < count && indices[indexOffset + slot[k]] === first + (k - j)) k++;
                const runLen = k - j;

                const vNeed = runLen * vertexRowSize;
                if (!gatherV || gatherV.length < vNeed) gatherV = new Uint8Array(vNeed);
                const vBytes = gatherV.subarray(0, vNeed);
                if (await readExact(source.read(vertexOffset + first * vertexRowSize, vertexOffset + (first + runLen) * vertexRowSize), vBytes, 0, vNeed) !== vNeed) {
                    throw new Error(`readPly: short gather vertex read at row ${first}`);
                }
                const vU32 = new Uint32Array(gatherV.buffer, gatherV.byteOffset, runLen * vWords);

                let sBytes: Uint8Array | null = null;
                if (col && shEl) {
                    const sNeed = runLen * shRowSize;
                    if (!gatherS || gatherS.length < sNeed) gatherS = new Uint8Array(sNeed);
                    sBytes = gatherS.subarray(0, sNeed);
                    if (await readExact(source.read(shOffset + first * shRowSize, shOffset + (first + runLen) * shRowSize), sBytes, 0, sNeed) !== sNeed) {
                        throw new Error(`readPly: short gather sh read at row ${first}`);
                    }
                }

                for (let t = 0; t < runLen; t++) {
                    decodeRecord(vU32, t, sBytes, t, first + t, slot[j + t]);
                }
                j = k;
            }
            return;
        }

        // Contiguous chunk: rows [start, start + count). The bounds for gaussian
        // g live at absolute 256-block `g >> 8`.
        const start = request.chunkIndex * chunkSize;
        const count = Math.min(chunkSize, numGaussians - start);
        if (count <= 0) {
            throw new Error(`readPly: chunkIndex ${request.chunkIndex} out of range`);
        }

        vScratch ??= new Uint8Array(chunkSize * vertexRowSize);
        const vBytes = vScratch.subarray(0, count * vertexRowSize);
        if (await readExact(source.read(vertexOffset + start * vertexRowSize, vertexOffset + (start + count) * vertexRowSize), vBytes, 0, vBytes.length) !== vBytes.length) {
            throw new Error(`readPly: short vertex read for chunk ${request.chunkIndex}`);
        }
        const vU32 = new Uint32Array(vScratch.buffer, 0, count * vWords);

        let sBytes: Uint8Array | null = null;
        if (col && shEl) {
            sScratch ??= new Uint8Array(chunkSize * shRowSize);
            sBytes = sScratch.subarray(0, count * shRowSize);
            if (await readExact(source.read(shOffset + start * shRowSize, shOffset + (start + count) * shRowSize), sBytes, 0, sBytes.length) !== sBytes.length) {
                throw new Error(`readPly: short sh read for chunk ${request.chunkIndex}`);
            }
        }

        for (let i = 0; i < count; i++) {
            decodeRecord(vU32, i, sBytes, i, start + i, i);
        }
    };

    return fileChunkSource(source, meta, read);
};

// One destination field within a layer: where to read it from a source record,
// and where/how to write it into the layer buffer.
type FillField = { recordOffset: number; read: ValueReader; dstByteOffset: number; uint: boolean };
// `srcIdx`/`dstIdx` are float indices for the all-float fast path; `allFloat`
// is true only when the whole record is float (so a Float32Array view is valid).
type LayerPlan = { stride: number; fields: FillField[]; allFloat: boolean; srcIdx: Uint32Array; dstIdx: Uint32Array };

/**
 * Open a gaussian-splat PLY as a {@link ChunkSource}. The single public PLY reader.
 *
 * Standard uncompressed binary PLY is read **lazily**: only the header is parsed
 * at open time, and each `read` seeks the requested chunk's byte range, pulls
 * just those records, and de-interleaves the requested layers into the caller's
 * buffers. Standard gaussian properties map to the `position`/`geometric`/`color`
 * layers; non-standard properties (e.g. normals) become `other`-layer extras.
 *
 * Compressed PLY (the packed chunk+vertex format) is read lazily too: each chunk
 * is range-read and dequantized on demand (see `readCompressedChunked`). Either
 * way the data is labelled `Transform.PLY`.
 *
 * The source must be `seekable` (range reads).
 *
 * @param source - A seekable read source over the PLY file.
 * @param pool - The chunk-data pool whose `chunkSize` defines the chunking granularity.
 * @returns A lazy `ChunkSource` over the file.
 */
const readPly = async (source: ReadSource, pool: ChunkDataPool): Promise<ChunkSource> => {
    if (!source.seekable) {
        throw new Error('readPly: source must be seekable');
    }

    const header = await readHeader(source);
    const { headerBytes } = header;
    const vertex = header.elements.find(e => e.name === 'vertex');
    if (!vertex) {
        throw new Error('readPly: PLY has no vertex element');
    }
    const properties = vertex.properties;
    const vertexCount = vertex.count;

    // Compressed PLY (packed chunk + vertex elements): lazy dequantizing reader.
    // `packed_position` is the marker.
    if (properties.some(p => p.name === 'packed_position')) {
        return readCompressedChunked(source, header, pool);
    }

    // Integrity guard: an uncompressed binary-little-endian PLY is exactly
    // `headerBytes + Σ element.count * rowStride` bytes. A mismatch means the file
    // is truncated or corrupt — fail fast rather than decode garbage. (ascii /
    // big-endian / compressed layouts have no such simple relation, so skip them.)
    if (header.format === 'binary_little_endian' && source.size !== undefined) {
        const expected = headerBytes + header.elements.reduce((sum, e) => sum + e.count * rowSizeOf(e.properties), 0);
        if (source.size !== expected) {
            throw new Error(`readPly: file size ${source.size} does not match header-implied size ${expected} (truncated or corrupt PLY)`);
        }
    }

    // Record layout: byte offset + reader per property.
    let recordStride = 0;
    const recordOffset = new Map<string, number>();
    const reader = new Map<string, ValueReader>();
    for (const p of properties) {
        recordOffset.set(p.name, recordStride);
        reader.set(p.name, getReader(p.type));
        recordStride += typeSize(p.type);
    }

    // Whole-record-float enables the typed-array fast path (every field 4-byte
    // aligned, so a Float32Array view over the record bytes is valid).
    const recordAllFloat = properties.every(p => p.type === 'float');

    const has = (name: string) => recordOffset.has(name);
    const hasPosition = ['x', 'y', 'z'].every(has);
    const hasGeometric = GEOMETRIC_COLS.every(has);
    const hasColor = COLOR_DC_COLS.every(has);

    // SH band count from the highest f_rest_* index present.
    let highestRest = -1;
    for (const p of properties) {
        const m = p.name.match(/^f_rest_(\d+)$/);
        if (m) highestRest = Math.max(highestRest, parseInt(m[1], 10));
    }
    const restCount = highestRest + 1;
    const shBands: SHBands = (restCount === 0 ? 0 : ({ [SH_REST_COUNTS[1]]: 1, [SH_REST_COUNTS[2]]: 2, [SH_REST_COUNTS[3]]: 3 } as Record<number, SHBands>)[restCount]);
    if (shBands === undefined) {
        throw new Error(`readPly: unrecognized f_rest_* count ${restCount}`);
    }

    // Non-standard columns become `other` extras (in file order).
    const standard = new Set<string>(['x', 'y', 'z', ...GEOMETRIC_COLS, ...COLOR_DC_COLS]);
    const extras: ExtraColumn[] = properties
    .filter(p => !standard.has(p.name) && !/^f_rest_\d+$/.test(p.name))
    .map((p) => {
        const type: 'float32' | 'uint32' = p.type === 'float' || p.type === 'double' ? 'float32' : 'uint32';
        return { name: p.name, type };
    });
    const hasOther = extras.length > 0;

    // Build per-layer fill plans + layouts.
    const fieldNames = (layer: ChunkLayer): string[] => {
        switch (layer) {
            case 'position': return ['x', 'y', 'z'];
            case 'geometric': return GEOMETRIC_COLS;
            case 'color': return [...COLOR_DC_COLS, ...Array.from({ length: SH_REST_COUNTS[shBands] }, (_, k) => `f_rest_${k}`)];
            case 'other': return extras.map(e => e.name);
        }
    };
    const uintByName = new Map(extras.map(e => [e.name, e.type === 'uint32']));

    const layerPlan = (layer: ChunkLayer, stride: number): LayerPlan => {
        const fields: FillField[] = fieldNames(layer).map((name, idx) => ({
            recordOffset: recordOffset.get(name)!,
            read: reader.get(name)!,
            dstByteOffset: idx * 4,
            uint: uintByName.get(name) ?? false
        }));
        const allFloat = recordAllFloat && fields.every(f => !f.uint);
        const srcIdx = new Uint32Array(fields.map(f => f.recordOffset >> 2));
        const dstIdx = new Uint32Array(fields.map(f => f.dstByteOffset >> 2));
        return { stride, fields, allFloat, srcIdx, dstIdx };
    };

    const availableLayers = new Set<ChunkLayer>();
    const layouts: Partial<Record<ChunkLayer, LayerLayout>> = {};
    const plans: Partial<Record<ChunkLayer, LayerPlan>> = {};
    if (hasPosition) {
        availableLayers.add('position');
        layouts.position = { stride: POSITION_STRIDE, fields: positionFields() };
        plans.position = layerPlan('position', POSITION_STRIDE);
    }
    if (hasGeometric) {
        availableLayers.add('geometric');
        layouts.geometric = { stride: GEOMETRIC_STRIDE, fields: geometricFields() };
        plans.geometric = layerPlan('geometric', GEOMETRIC_STRIDE);
    }
    if (hasColor) {
        availableLayers.add('color');
        layouts.color = { stride: colorStride(shBands), fields: colorFields(shBands) };
        plans.color = layerPlan('color', colorStride(shBands));
    }
    if (hasOther) {
        availableLayers.add('other');
        const ol = otherLayout(extras);
        layouts.other = { stride: ol.stride, fields: ol.fields };
        plans.other = layerPlan('other', ol.stride);
    }

    const chunkSize = pool.chunkSize;
    const numChunks = Math.max(1, Math.ceil(vertexCount / chunkSize));

    const meta: ChunkSourceMetadata = {
        numGaussians: vertexCount,
        numLods: 1,
        lodCounts: [vertexCount],
        chunkSize,
        numChunks: [numChunks],
        shBands,
        extraColumns: extras,
        transform: Transform.PLY.clone(),
        availableLayers,
        layouts
    };

    const fill = (recordBytes: Uint8Array, count: number, chunkData: ChunkData, plan: LayerPlan, dstRow: number): void => {
        // Fast path: whole-float record -> de-interleave via Float32Array views
        // (no DataView). Little-endian only, matching the binary PLY format.
        if (plan.allFloat) {
            const recF32 = new Float32Array(recordBytes.buffer, recordBytes.byteOffset, recordBytes.byteLength >> 2);
            const dstF32 = new Float32Array(chunkData.data);
            const sStrideF = recordStride >> 2;
            const dStrideF = plan.stride >> 2;
            const { srcIdx, dstIdx } = plan;
            const nf = srcIdx.length;
            for (let i = 0; i < count; i++) {
                const rb = i * sStrideF;
                const db = (dstRow + i) * dStrideF;
                for (let j = 0; j < nf; j++) {
                    dstF32[db + dstIdx[j]] = recF32[rb + srcIdx[j]];
                }
            }
            return;
        }

        // General path: mixed types via DataView readers.
        const src = new DataView(recordBytes.buffer, recordBytes.byteOffset, recordBytes.byteLength);
        const dst = new DataView(chunkData.data);
        for (let i = 0; i < count; i++) {
            const recBase = i * recordStride;
            const dstBase = (dstRow + i) * plan.stride;
            for (const f of plan.fields) {
                const v = f.read(src, recBase + f.recordOffset);
                if (f.uint) {
                    dst.setUint32(dstBase + f.dstByteOffset, v, true);
                } else {
                    dst.setFloat32(dstBase + f.dstByteOffset, v, true);
                }
            }
        }
    };

    // Sub-block size (gaussians) for the read+de-interleave loop. Bounds the raw
    // record scratch to SUB_BLOCK×recordStride (~16 MB at typical strides),
    // independent of chunkSize, so reading a chunk never materializes the whole
    // interleaved chunk in memory at once.
    const SUB_BLOCK = 1 << 16;

    // Reused scratch buffers. Reads are sequential per the ChunkSource contract,
    // so single shared buffers avoid re-allocating on every read: `recordBuffer`
    // for the contiguous sub-block sweep, `gatherScratch` for one coalesced run.
    let recordBuffer: Uint8Array | null = null;
    let gatherScratch: Uint8Array | null = null;

    const read = async (request: ReadRequest): Promise<void> => {
        if ((request.lod ?? 0) !== 0) {
            throw new Error('readPly: only lod 0 is supported');
        }

        const requested: ChunkLayer[] = (['position', 'geometric', 'color', 'other'] as ChunkLayer[])
        .filter(layer => request[layer]);
        for (const layer of requested) {
            if (!plans[layer]) {
                throw new Error(`readPly: layer '${layer}' not available`);
            }
        }

        if ('indices' in request) {
            // Gather: fetch arbitrary rows by byte-range reads. Output slots are
            // sorted into source-file order so reads run forward and consecutive
            // source rows coalesce into one range read; each record is then
            // de-interleaved into its (scattered) output row. Only the requested
            // rows' bytes (including SH) are read — the basis of the LOD writer's
            // "positions resident, heavy data fetched per output chunk" gather.
            const { indices, indexOffset, count } = request;
            if (count <= 0) return;

            const slot = new Uint32Array(count);
            for (let j = 0; j < count; j++) slot[j] = j;
            slot.sort((a, b) => indices[indexOffset + a] - indices[indexOffset + b]);

            let j = 0;
            while (j < count) {
                const first = indices[indexOffset + slot[j]];
                let k = j + 1;
                while (k < count && indices[indexOffset + slot[k]] === first + (k - j)) k++;
                const runLen = k - j;

                const need = runLen * recordStride;
                if (!gatherScratch || gatherScratch.length < need) {
                    gatherScratch = new Uint8Array(need);
                }
                const recordBytes = gatherScratch.subarray(0, need);
                const got = await readExact(
                    source.read(headerBytes + first * recordStride, headerBytes + (first + runLen) * recordStride),
                    recordBytes, 0, need
                );
                if (got !== need) {
                    throw new Error(`readPly: short gather read (${got}/${need}) at row ${first}`);
                }

                for (let t = 0; t < runLen; t++) {
                    const rec = recordBytes.subarray(t * recordStride, (t + 1) * recordStride);
                    for (const layer of requested) {
                        fill(rec, 1, request[layer]!, plans[layer]!, slot[j + t]);
                    }
                }
                j = k;
            }
            return;
        }

        // Contiguous chunk: rows [start, start + count).
        const start = request.chunkIndex * chunkSize;
        const count = Math.min(chunkSize, vertexCount - start);
        if (count <= 0) {
            throw new Error(`readPly: chunkIndex ${request.chunkIndex} out of range`);
        }

        const byteStart = headerBytes + start * recordStride;
        const byteEnd = byteStart + count * recordStride;
        const stream = source.read(byteStart, byteEnd);

        const block = Math.min(chunkSize, SUB_BLOCK);
        recordBuffer ??= new Uint8Array(block * recordStride);

        // Pull and de-interleave the chunk one sub-block at a time.
        let done = 0;
        while (done < count) {
            const b = Math.min(SUB_BLOCK, count - done);
            const recordBytes = recordBuffer.subarray(0, b * recordStride);
            const got = await readExact(stream, recordBytes, 0, recordBytes.length);
            if (got !== recordBytes.length) {
                throw new Error(`readPly: short read (${got}/${recordBytes.length}) for chunk ${request.chunkIndex}`);
            }
            for (const layer of requested) {
                fill(recordBytes, b, request[layer]!, plans[layer]!, done);
            }
            done += b;
        }
    };

    return fileChunkSource(source, meta, read);
};

export { PlyData, decodePlyToDataTable, readPly };
