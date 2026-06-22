import { type ReadSource, type ReadStream } from '../io/read';
import {
    type ChunkData,
    type ChunkLayer,
    type ChunkReadRequest,
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
} from '../source';
import { Transform } from '../utils';

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
        default: throw new Error(`readPlyChunked: unsupported ply type '${type}'`);
    }
};

const typeSize = (type: string): number => {
    switch (type) {
        case 'char': case 'uchar': return 1;
        case 'short': case 'ushort': return 2;
        case 'int': case 'uint': case 'float': return 4;
        case 'double': return 8;
        default: throw new Error(`readPlyChunked: unsupported ply type '${type}'`);
    }
};

type PlyProperty = { name: string; type: string };
type ParsedHeader = {
    vertexCount: number;
    properties: PlyProperty[];
    headerBytes: number;
};

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

const readExact = async (stream: ReadStream, buffer: Uint8Array, offset: number, length: number): Promise<number> => {
    let total = 0;
    while (total < length) {
        const n = await stream.pull(buffer.subarray(offset + total, offset + length));
        if (n === 0) break;
        total += n;
    }
    return total;
};

// Read and parse the PLY header, returning the vertex element's properties and
// the byte offset at which binary data begins.
const readHeader = async (source: ReadSource): Promise<ParsedHeader> => {
    const probeLen = Math.min(128 * 1024, source.size ?? 128 * 1024);
    const buf = new Uint8Array(probeLen);
    const read = await readExact(source.read(0, probeLen), buf, 0, probeLen);

    const markerIdx = indexOfMarker(buf, endHeaderMarker, read);
    if (markerIdx < 0) {
        throw new Error('readPlyChunked: end_header not found within probe window (compressed or non-PLY?)');
    }
    const headerBytes = markerIdx + endHeaderMarker.length;

    const lines = new TextDecoder('ascii').decode(buf.subarray(0, headerBytes)).split('\n').filter(Boolean);
    if (lines[0] !== 'ply') {
        throw new Error('readPlyChunked: invalid PLY header');
    }

    let vertexCount = -1;
    let properties: PlyProperty[] | null = null;
    let current: { name: string; props: PlyProperty[] } | null = null;

    for (let i = 1; i < lines.length; i++) {
        const words = lines[i].split(' ').filter(Boolean);
        switch (words[0]) {
            case 'format': case 'comment': case 'ply': case 'end_header':
                break;
            case 'element':
                if (words.length !== 3) throw new Error('readPlyChunked: invalid element line');
                current = { name: words[1], props: [] };
                if (words[1] === 'vertex') {
                    vertexCount = parseInt(words[2], 10);
                    properties = current.props;
                }
                break;
            case 'property':
                if (!current || words.length !== 3) throw new Error('readPlyChunked: invalid property line');
                current.props.push({ name: words[2], type: words[1] });
                break;
            default:
                throw new Error(`readPlyChunked: unrecognized header line '${words[0]}'`);
        }
    }

    if (vertexCount < 0 || !properties) {
        throw new Error('readPlyChunked: PLY has no vertex element');
    }
    return { vertexCount, properties, headerBytes };
};

const GEOMETRIC_COLS = ['rot_0', 'rot_1', 'rot_2', 'rot_3', 'scale_0', 'scale_1', 'scale_2', 'opacity'];
const COLOR_DC_COLS = ['f_dc_0', 'f_dc_1', 'f_dc_2'];

// One destination field within a layer: where to read it from a source record,
// and where/how to write it into the layer buffer.
type FillField = { recordOffset: number; read: ValueReader; dstByteOffset: number; uint: boolean };
// `srcIdx`/`dstIdx` are float indices for the all-float fast path; `allFloat`
// is true only when the whole record is float (so a Float32Array view is valid).
type LayerPlan = { stride: number; fields: FillField[]; allFloat: boolean; srcIdx: Uint32Array; dstIdx: Uint32Array };

/**
 * Open a (binary, little-endian) gaussian-splat PLY as a lazy {@link ChunkSource}.
 *
 * Parses only the header at open time; each `read` seeks to the requested
 * chunk's byte range, pulls just those records from the source, and decodes the
 * requested layers into the caller's buffers. Standard gaussian properties map
 * to the `position`/`geometric`/`color` layers; any non-standard properties
 * (e.g. normals) become `other`-layer extras. Matches `readPly` in labelling
 * the data as `Transform.PLY`.
 *
 * The source must be `seekable` (range reads). Targets standard uncompressed
 * gaussian PLY; compressed PLY is out of scope (use `readPly`).
 *
 * @param source - A seekable read source over the PLY file.
 * @param pool - The chunk-data pool whose `chunkSize` defines the chunking granularity.
 * @returns A lazy `ChunkSource` over the file.
 */
const readPlyChunked = async (source: ReadSource, pool: ChunkDataPool): Promise<ChunkSource> => {
    if (!source.seekable) {
        throw new Error('readPlyChunked: source must be seekable');
    }

    const { vertexCount, properties, headerBytes } = await readHeader(source);

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
        throw new Error(`readPlyChunked: unrecognized f_rest_* count ${restCount}`);
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

    const fill = (recordBytes: Uint8Array, count: number, chunkData: ChunkData, plan: LayerPlan): void => {
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
                const db = i * dStrideF;
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
            const dstBase = i * plan.stride;
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

    const read = async (request: ChunkReadRequest): Promise<void> => {
        if ((request.lod ?? 0) !== 0) {
            throw new Error('readPlyChunked: only lod 0 is supported');
        }
        const start = request.chunkIndex * chunkSize;
        const count = Math.min(chunkSize, vertexCount - start);
        if (count <= 0) {
            throw new Error(`readPlyChunked: chunkIndex ${request.chunkIndex} out of range`);
        }

        const byteStart = headerBytes + start * recordStride;
        const byteEnd = byteStart + count * recordStride;
        const recordBytes = new Uint8Array(count * recordStride);
        const got = await readExact(source.read(byteStart, byteEnd), recordBytes, 0, recordBytes.length);
        if (got !== recordBytes.length) {
            throw new Error(`readPlyChunked: short read (${got}/${recordBytes.length}) for chunk ${request.chunkIndex}`);
        }

        const layers: ChunkLayer[] = ['position', 'geometric', 'color', 'other'];
        for (const layer of layers) {
            const cd = request[layer];
            if (!cd) continue;
            const plan = plans[layer];
            if (!plan) {
                throw new Error(`readPlyChunked: layer '${layer}' not available`);
            }
            fill(recordBytes, count, cd, plan);
        }
    };

    const close = (): Promise<void> => {
        source.close();
        return Promise.resolve();
    };

    return { meta, read, close };
};

export { readPlyChunked };
