import { Vec3 } from 'playcanvas';

import { readExact, sortGatherSlots, gatherRuns } from './reader-utils';
import {
    type ChunkData,
    type ChunkDataPool,
    type ChunkLayer,
    type ReadRequest,
    type ChunkSource,
    type ChunkSourceMetadata,
    type ExtraColumn,
    type LayerLayout,
    type SHBands,
    POSITION_STRIDE,
    GEOMETRIC_STRIDE,
    colorStride,
    positionFields,
    geometricFields,
    colorFields,
    otherLayout
} from '../chunk';
import { dataTableToChunkSource } from '../compat/data-table';
import { Column, DataTable } from '../data-table';
import { dirname, join, ReadFileSystem, ReadSource, readFile } from '../io/read';
import { Options } from '../types';
import { logger, Transform } from '../utils';

const kSH_C0 = 0.28209479177387814;
const SQRT_2 = 1.414213562373095;
const SQRT_2_INV = 0.7071067811865475;

const IO_CONCURRENCY = 16;

// lod data in data.bin
type LccLod = {
    points: number;     // number of splats
    offset: bigint;     // offset
    size: number;       // data size
}

// The scene uses a quadtree for spatial partitioning,
// with each unit having its own xy index (starting from 0) and multiple layers of lod data
type LccUnitInfo = {
    x: number;          // x index
    y: number;          // y index
    lods: Array<LccLod>;    //  lods
}

// Used to decompress scale in data.bin and sh in shcoef.bin
type CompressInfo = {
    scaleMin: Vec3;         // min scale
    scaleMax: Vec3;         // max scale
    shMin: Vec3;            // min sh
    shMax: Vec3;            // max sh
    envScaleMin: Vec3;      // min environment scale
    envScaleMax: Vec3;      // max environment scale
    envShMin: Vec3;         // min environment sh
    envShMax: Vec3;         // max environment sh
}

// parse .lcc files, such as meta.lcc
const parseMeta = (obj: any): CompressInfo => {
    const attributes: { [key: string]: any } = {};
    obj.attributes.forEach((attr: any) => {
        attributes[attr.name] = attr;
    });

    const scaleMin = new Vec3(attributes.scale.min);
    const scaleMax = new Vec3(attributes.scale.max);
    const shMin = new Vec3(attributes.shcoef.min);
    const shMax = new Vec3(attributes.shcoef.max);
    const envScaleMin = new Vec3(attributes.envscale?.min ?? attributes.scale.min);
    const envScaleMax = new Vec3(attributes.envscale?.max ?? attributes.scale.max);
    const envShMin = new Vec3(attributes.envshcoef?.min ?? attributes.shcoef.min);
    const envShMax = new Vec3(attributes.envshcoef?.max ?? attributes.shcoef.max);

    return { scaleMin, scaleMax, shMin, shMax, envScaleMin, envScaleMax, envShMin, envShMax };
};

const parseIndexBin = (raw: ArrayBuffer, meta: any): Array<LccUnitInfo> => {
    let offset = 0;

    const buff = new DataView(raw);
    const infos: Array<LccUnitInfo> = [];
    while (true) {
        if (offset > buff.byteLength - 1) {
            break;
        }

        const x = buff.getInt16(offset, true);
        offset += 2;
        const y = buff.getInt16(offset, true);
        offset += 2;

        const lods: Array<LccLod> = [];
        for (let i = 0; i < meta.totalLevel; i++) {
            const ldPoints = buff.getInt32(offset, true);
            offset += 4;

            const ldOffset = buff.getBigInt64(offset, true);
            offset += 8;

            const ldSize = buff.getInt32(offset, true);
            offset += 4;

            lods.push({
                points: ldPoints,
                offset: ldOffset,
                size: ldSize
            });

        }
        const info: LccUnitInfo = {
            x,
            y,
            lods
        };

        infos.push(info);
    }

    return infos;
};

const invSigmoid = (v: number): number => {
    return -Math.log((1.0 - v) / v);
};

const invSH0ToColor = (v: number): number => {
    return (v - 0.5) / kSH_C0;
};

const invLinearScale = (v: number): number => {
    return Math.log(v);
};

const mix = (min: number, max: number, s: number): number => {
    return (1.0 - s) * min + s * max;
};

const floatProps = [
    'x', 'y', 'z',
    'nx', 'ny', 'nz',
    'opacity',
    'rot_0', 'rot_1', 'rot_2', 'rot_3',
    'f_dc_0', 'f_dc_1', 'f_dc_2',
    'scale_0', 'scale_1', 'scale_2'
];

const initProperties = (length: number): Record<string, Float32Array> => {
    const props: Record<string, Float32Array> = {};
    for (const key of floatProps) {
        props[`property_${key}`] = new Float32Array(length);
    }
    return props;
};

// Decode rotation quaternion and write directly to output arrays.
// The encoded value packs 3 quaternion components at 10 bits each, plus a 2-bit index
// indicating which component was omitted (the largest, which is reconstructed).
const decodeRotationInto = (
    v: number,
    rot0: Float32Array, rot1: Float32Array, rot2: Float32Array, rot3: Float32Array,
    idx: number
) => {
    const d0 = (v & 1023) / 1023.0;
    const d1 = ((v >> 10) & 1023) / 1023.0;
    const d2 = ((v >> 20) & 1023) / 1023.0;
    const d3 = (v >> 30) & 3;

    const qx = d0 * SQRT_2 - SQRT_2_INV;
    const qy = d1 * SQRT_2 - SQRT_2_INV;
    const qz = d2 * SQRT_2 - SQRT_2_INV;
    const qw = Math.sqrt(1 - Math.min(1.0, qx * qx + qy * qy + qz * qz));

    // Reconstruct full quaternion with qw inserted at position d3.
    // Output mapping matches original: rot_0 = q[3], rot_1 = q[0], rot_2 = q[1], rot_3 = q[2]
    if (d3 === 0) {
        rot0[idx] = qz; rot1[idx] = qw; rot2[idx] = qx; rot3[idx] = qy;
    } else if (d3 === 1) {
        rot0[idx] = qz; rot1[idx] = qx; rot2[idx] = qw; rot3[idx] = qy;
    } else if (d3 === 2) {
        rot0[idx] = qz; rot1[idx] = qx; rot2[idx] = qy; rot3[idx] = qw;
    } else {
        rot0[idx] = qw; rot1[idx] = qx; rot2[idx] = qy; rot3[idx] = qz;
    }
};

// Decode a unit's splat data and write directly into the global output arrays at propertyOffset.
// Uses typed array views instead of DataView for faster element access (assumes LE host).
const processUnit = async (
    info: LccUnitInfo,
    targetLod: number,
    dataSource: ReadSource,
    shSource: ReadSource | undefined,
    compressInfo: CompressInfo,
    propertyOffset: number,
    properties: Record<string, Float32Array>,
    properties_f_rest: Float32Array[] | null,
    onUnitDone: () => void
) => {
    const lod = info.lods[targetLod];
    const unitSplats = lod.points;
    if (unitSplats === 0) {
        onUnitDone();
        return;
    }

    const offset = Number(lod.offset);
    const size = lod.size;

    // load data using range read
    const dataBytes = await dataSource.read(offset, offset + size).readAll();
    const expectedDataSize = unitSplats * 32;
    if (dataBytes.byteLength < expectedDataSize) {
        throw new Error(`LCC unit data too short: expected ${expectedDataSize} bytes for ${unitSplats} splats, got ${dataBytes.byteLength}`);
    }

    // Typed array views over the same buffer -- avoids DataView overhead.
    // 32-byte record: [f32 x, f32 y, f32 z, u8 r, u8 g, u8 b, u8 opacity,
    //                  u16 s0, u16 s1, u16 s2, u16 rot_lo, u16 rot_hi, u16 nx, u16 ny, u16 nz]
    const f32 = new Float32Array(dataBytes.buffer, dataBytes.byteOffset, dataBytes.byteLength >> 2);
    const u16 = new Uint16Array(dataBytes.buffer, dataBytes.byteOffset, dataBytes.byteLength >> 1);
    const u8 = dataBytes;

    // load sh data using range read
    let shU32: Uint32Array | null = null;
    if (shSource) {
        const shBytes = await shSource.read(offset * 2, offset * 2 + size * 2).readAll();
        const expectedShSize = unitSplats * 64;
        if (shBytes.byteLength < expectedShSize) {
            throw new Error(`LCC unit SH data too short: expected ${expectedShSize} bytes for ${unitSplats} splats, got ${shBytes.byteLength}`);
        }
        shU32 = new Uint32Array(shBytes.buffer, shBytes.byteOffset, shBytes.byteLength >> 2);
    }

    // Extract array references once to avoid repeated property lookups in the hot loop
    const px = properties.property_x;
    const py = properties.property_y;
    const pz = properties.property_z;
    const pnx = properties.property_nx;
    const pny = properties.property_ny;
    const pnz = properties.property_nz;
    const pop = properties.property_opacity;
    const pr0 = properties.property_rot_0;
    const pr1 = properties.property_rot_1;
    const pr2 = properties.property_rot_2;
    const pr3 = properties.property_rot_3;
    const pdc0 = properties.property_f_dc_0;
    const pdc1 = properties.property_f_dc_1;
    const pdc2 = properties.property_f_dc_2;
    const ps0 = properties.property_scale_0;
    const ps1 = properties.property_scale_1;
    const ps2 = properties.property_scale_2;

    const sMinX = compressInfo.scaleMin.x, sMinY = compressInfo.scaleMin.y, sMinZ = compressInfo.scaleMin.z;
    const sMaxX = compressInfo.scaleMax.x, sMaxY = compressInfo.scaleMax.y, sMaxZ = compressInfo.scaleMax.z;
    const shMinX = compressInfo.shMin.x, shMinY = compressInfo.shMin.y, shMinZ = compressInfo.shMin.z;
    const shMaxX = compressInfo.shMax.x, shMaxY = compressInfo.shMax.y, shMaxZ = compressInfo.shMax.z;

    for (let i = 0; i < unitSplats; i++) {
        const g = propertyOffset + i;

        // position: 3 x float32 at byte offsets 0, 4, 8 → f32 indices i*8+{0,1,2}
        const fi = i << 3;
        px[g] = f32[fi];
        py[g] = f32[fi + 1];
        pz[g] = f32[fi + 2];

        // color + opacity: 4 x uint8 at byte offsets 12..15
        const bi = i << 5;
        pdc0[g] = invSH0ToColor(u8[bi + 12] / 255.0);
        pdc1[g] = invSH0ToColor(u8[bi + 13] / 255.0);
        pdc2[g] = invSH0ToColor(u8[bi + 14] / 255.0);
        pop[g] = invSigmoid(u8[bi + 15] / 255.0);

        // scale + rotation + normals: uint16 at byte offsets 16..31 → u16 indices i*16+{8..15}
        const hi = i << 4;
        ps0[g] = invLinearScale(mix(sMinX, sMaxX, u16[hi + 8] / 65535.0));
        ps1[g] = invLinearScale(mix(sMinY, sMaxY, u16[hi + 9] / 65535.0));
        ps2[g] = invLinearScale(mix(sMinZ, sMaxZ, u16[hi + 10] / 65535.0));

        // rotation: uint32 at byte offset 22 (not 4-byte aligned), reconstruct from two uint16s
        decodeRotationInto(u16[hi + 11] | (u16[hi + 12] << 16), pr0, pr1, pr2, pr3, g);

        pnx[g] = u16[hi + 13];
        pny[g] = u16[hi + 14];
        pnz[g] = u16[hi + 15];

        // SH coefficients: 15 x uint32 per splat, 64-byte stride (16 uint32s)
        if (shU32 && properties_f_rest) {
            const si = i << 4;
            for (let j = 0; j < 15; j++) {
                const enc = shU32[si + j];
                properties_f_rest[j][g] = mix(shMinX, shMaxX, (enc & 0x7FF) / 2047.0);
                properties_f_rest[j + 15][g] = mix(shMinY, shMaxY, ((enc >> 11) & 0x3FF) / 1023.0);
                properties_f_rest[j + 30][g] = mix(shMinZ, shMaxZ, ((enc >> 21) & 0x7FF) / 2047.0);
            }
        }
    }

    onUnitDone();
};

// Decode all units for a given LOD into shared global arrays with bounded concurrency.
const decodeUnitsForLod = async (
    unitInfos: LccUnitInfo[],
    targetLod: number,
    dataSource: ReadSource,
    shSource: ReadSource | undefined,
    compressInfo: CompressInfo,
    lodOffset: number,
    properties: Record<string, Float32Array>,
    properties_f_rest: Float32Array[] | null,
    onUnitDone: () => void
) => {
    // Pre-compute write offsets so units can be processed concurrently without data races
    const offsets = new Array<number>(unitInfos.length);
    let unitOffset = lodOffset;
    for (let i = 0; i < unitInfos.length; i++) {
        offsets[i] = unitOffset;
        unitOffset += unitInfos[i].lods[targetLod].points;
    }

    let nextUnit = 0;
    const worker = async () => {
        while (true) {
            const idx = nextUnit++;
            if (idx >= unitInfos.length) break;
            await processUnit(
                unitInfos[idx], targetLod, dataSource, shSource,
                compressInfo, offsets[idx], properties, properties_f_rest,
                onUnitDone
            );
        }
    };
    await Promise.all(
        Array.from({ length: Math.min(IO_CONCURRENCY, unitInfos.length) }, () => worker())
    );
};

const deserializeEnvironment = (raw: Uint8Array, compressInfo: CompressInfo, hasSH: boolean) => {
    const stride = hasSH ? 96 : 32;

    const numGaussians = raw.length / stride;

    if (!Number.isInteger(numGaussians)) {
        throw new Error('Invalid environment data size');
    }

    const columns = [
        'x', 'y', 'z',
        'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
        'scale_0', 'scale_1', 'scale_2',
        'rot_0', 'rot_1', 'rot_2', 'rot_3'
    ].concat(hasSH ? new Array(45).fill('').map((_, i) => `f_rest_${i}`) : []).map(name => new Column(name, new Float32Array(numGaussians)));

    const scaleMin = compressInfo.envScaleMin;
    const scaleMax = compressInfo.envScaleMax;
    const shMin = compressInfo.envShMin;
    const shMax = compressInfo.envShMax;

    const rot0 = columns[10].data as Float32Array;
    const rot1 = columns[11].data as Float32Array;
    const rot2 = columns[12].data as Float32Array;
    const rot3 = columns[13].data as Float32Array;

    // fill data
    const dataView = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    for (let i = 0; i < numGaussians; i++) {
        const off = i * stride;

        columns[0].data[i] = dataView.getFloat32(off + 0, true);   // x
        columns[1].data[i] = dataView.getFloat32(off + 4, true);   // y
        columns[2].data[i] = dataView.getFloat32(off + 8, true);   // z

        columns[3].data[i] = invSH0ToColor(dataView.getUint8(off + 12) / 255.0);   // f_dc_0
        columns[4].data[i] = invSH0ToColor(dataView.getUint8(off + 13) / 255.0);   // f_dc_1
        columns[5].data[i] = invSH0ToColor(dataView.getUint8(off + 14) / 255.0);   // f_dc_2
        columns[6].data[i] = invSigmoid(dataView.getUint8(off + 15) / 255.0);      // opacity

        columns[7].data[i] = invLinearScale(mix(scaleMin.x, scaleMax.x, dataView.getUint16(off + 16, true) / 65535.0)); // scale_0
        columns[8].data[i] = invLinearScale(mix(scaleMin.y, scaleMax.y, dataView.getUint16(off + 18, true) / 65535.0)); // scale_1
        columns[9].data[i] = invLinearScale(mix(scaleMin.z, scaleMax.z, dataView.getUint16(off + 20, true) / 65535.0)); // scale_2

        decodeRotationInto(dataView.getUint32(off + 22, true), rot0, rot1, rot2, rot3, i);

        // skip normal 26-32

        if (hasSH) {
            for (let j = 0; j < 15; ++j) {
                const enc = dataView.getUint32(off + 32 + j * 4, true);
                const nx = (enc & 0x7FF) / 2047.0;
                const ny = ((enc >> 11) & 0x3FF) / 1023.0;
                const nz = ((enc >> 21) & 0x7FF) / 2047.0;
                columns[14 + j].data[i] = mix(shMin.x, shMax.x, nx);
                columns[14 + j + 15].data[i] = mix(shMin.y, shMax.y, ny);
                columns[14 + j + 30].data[i] = mix(shMin.z, shMax.z, nz);
            }
        }
    }

    return new DataTable(columns);
};

/**
 * Reads an XGrids LCC format containing multi-LOD Gaussian splat data.
 *
 * The LCC format uses a quadtree spatial structure with multiple LOD levels.
 * Each LOD level is stored separately in data.bin with optional spherical
 * harmonics in shcoef.bin. Environment splats are stored in environment.bin.
 *
 * All selected LODs are decoded directly into a single pre-allocated DataTable
 * to avoid a costly post-read combine step.
 *
 * @param fileSystem - File system for reading the LCC files.
 * @param filename - Path to the meta.lcc file.
 * @param options - Options including LOD selection via `lodSelect`.
 * @returns Promise resolving to an array of DataTables (combined LODs + environment).
 * @ignore
 */
const readLcc = async (fileSystem: ReadFileSystem, filename: string, options: Options): Promise<DataTable[]> => {
    const lccData = await readFile(fileSystem, filename);
    const lccText = new TextDecoder().decode(lccData);
    const lccJson = JSON.parse(lccText);

    const determineSH = () => {
        if (lccJson.fileType === 'Portable') {
            return false;
        }

        if (lccJson.fileType === 'Quality') {
            return true;
        }

        // before version 4 sh seems to have always been present, but we test for shcoef attribute anyway
        return lccJson.attributes.findIndex((attr: any) => attr.name === 'shcoef') !== -1;
    };

    // FIXME: it seems some meta.lcc files at https://developer.xgrids.com/#/download?page=sampledata do not have
    // 'fileType' field, but do appear to contain spherical harmonics data. So for now assume presence of SH when
    // the field is missing.
    // See https://github.com/xgrids/LCCWhitepaper/issues/3
    const hasSH = determineSH();
    const compressInfo = parseMeta(lccJson);
    const splats = lccJson.splats;

    const baseDir = dirname(filename);
    const relatedFilename = (name: string) => (baseDir ? join(baseDir, name) : name);

    const indexData = await readFile(fileSystem, relatedFilename('index.bin'));

    const openSource = (name: string): Promise<ReadSource> => {
        return fileSystem.createSource(relatedFilename(name));
    };

    let dataSource: ReadSource | null = null;
    let shSource: ReadSource | null = null;
    let mainTable: DataTable;

    try {
        // Open sequentially so a failure on the second open doesn't leak the
        // first source. (The actual bulk reads still happen in parallel below
        // via decodeUnitsForLod's IO_CONCURRENCY workers.)
        dataSource = await openSource('data.bin');
        if (hasSH) shSource = await openSource('shcoef.bin');

        const unitInfos: LccUnitInfo[] = parseIndexBin(indexData.buffer.slice(indexData.byteOffset, indexData.byteOffset + indexData.byteLength) as ArrayBuffer, lccJson);

        // build table of input -> output lods
        const lodSelect = options.lodSelect ?? [];
        const lods = lodSelect.length > 0 ?
            lodSelect
            .map(lod => (lod < 0 ? splats.length + lod : lod))    // negative indices map from the end of lod
            .filter(lod => lod >= 0 && lod < splats.length) :
            new Array(splats.length).fill(0).map((_, i) => i);

        if (lods.length === 0) {
            throw new Error(`No valid LODs selected for LCC input file: ${filename} lods: ${JSON.stringify(lods)}`);
        }

        // Pre-allocate a single set of arrays for all LODs combined
        const grandTotal = lods.reduce((sum, lodIdx) => sum + splats[lodIdx], 0);
        const properties: Record<string, Float32Array> = initProperties(grandTotal);
        const properties_f_rest = shSource ? Array.from({ length: 45 }, () => new Float32Array(grandTotal)) : null;
        const lodColumn = new Float32Array(grandTotal);

        // One bar across all LOD passes: total = (units per LOD) * (LOD count).
        // Bounded concurrency means ticks come in bursts; that's fine for the
        // bar's monotonic update().
        const totalUnits = unitInfos.length * lods.length;
        const bar = logger.bar('decoding', totalUnits);
        let unitsDone = 0;
        const tick = () => {
            bar.update(++unitsDone);
        };

        let lodOffset = 0;
        for (let i = 0; i < lods.length; i++) {
            const inputLod = lods[i];
            const outputLod = i;
            const totalSplats = splats[inputLod];

            await decodeUnitsForLod(
                unitInfos, inputLod, dataSource, shSource ?? undefined,
                compressInfo, lodOffset, properties, properties_f_rest,
                tick
            );

            lodColumn.fill(outputLod, lodOffset, lodOffset + totalSplats);
            lodOffset += totalSplats;
        }

        const columns = [
            ...floatProps.map(name => new Column(name, properties[`property_${name}`])),
            ...(properties_f_rest ? properties_f_rest.map((storage, i) => new Column(`f_rest_${i}`, storage)) : []),
            new Column('lod', lodColumn)
        ];

        mainTable = new DataTable(columns, new Transform().fromEulers(90, 0, 180));
        // Close the bar only on success: leaving it open on the error path
        // lets `logger.error() -> unwindAll(true)` mark it as failed
        // instead of finalizing it as a successful bar first.
        bar.end();
    } finally {
        dataSource?.close();
        shSource?.close();
    }

    const result: DataTable[] = [mainTable];

    // environment.bin is optional - missing file is a normal case (no skybox).
    // Different ReadFileSystem implementations signal "not found" differently:
    //   - NodeReadFileSystem throws an Error with `code === 'ENOENT'`
    //   - Memory/Zip backends throw `Error('Entry not found: ...')`
    //   - UrlReadFileSystem throws `Error('HTTP error 404: ...')`
    // Suppress warnings for any of these.
    try {
        const envData = await readFile(fileSystem, relatedFilename('environment.bin'));
        const envDataTable = deserializeEnvironment(envData, compressInfo, hasSH);
        envDataTable.addColumn(new Column('lod', new Float32Array(envDataTable.numRows).fill(-1)));
        envDataTable.transform = new Transform().fromEulers(90, 0, 180);
        result.push(envDataTable);
    } catch (err) {
        const code = (err as { code?: string })?.code;
        const message = (err as Error)?.message ?? '';
        const isMissing = code === 'ENOENT' ||
            message.startsWith('Entry not found') ||
            message.startsWith('HTTP error 404');
        if (!isMissing) {
            logger.warn(`failed to load environment.bin: ${message || err}`);
        }
    }

    return result;
};

// LCC's fixed coordinate transform (Y-up → engine), applied lazily on consume.
const LCC_TRANSFORM = (): Transform => new Transform().fromEulers(90, 0, 180);

// Whether this LCC scene carries spherical harmonics (mirrors readLcc).
const lccHasSH = (lccJson: any): boolean => {
    if (lccJson.fileType === 'Portable') return false;
    if (lccJson.fileType === 'Quality') return true;
    // Pre-v4 / missing fileType: assume SH if a shcoef attribute is present.
    return lccJson.attributes.findIndex((attr: any) => attr.name === 'shcoef') !== -1;
};

/**
 * Native chunked LCC reader: presents the selected LODs as a flat fixed-stride
 * {@link ChunkSource} over the quadtree's units. Each `read`/`readRows`
 * range-reads only the requested gaussians' 32-byte records from `data.bin`
 * (+ 64-byte SH records from `shcoef.bin`) and dequantizes them straight into the
 * caller's layer buffers — no whole-unit decode, no intermediate `DataTable`.
 * Peak memory is sub-block bounded (like the PLY reader), independent of unit size.
 *
 * The global gaussian order per LOD is the concatenation of the non-empty units
 * in `index.bin` order, matching eager {@link readLcc} (so a materialized
 * comparison is byte-identical, minus the structural `lod` column). LCC's
 * quantization is scene-global (`meta.lcc`), so any record decodes independently
 * of which unit it came from.
 *
 * Standard fields map to position/geometric/color; the LCC normals (nx/ny/nz)
 * become `other`-layer extras (matching the eager table's columns). Data is
 * labelled `LCC_TRANSFORM()` (applied lazily on consume). The optional
 * environment chunk is not returned (only the eager path writes it, for `lod`
 * output).
 *
 * @param fileSystem - File system for reading the LCC files.
 * @param filename - Path to the meta.lcc file.
 * @param options - Options including LOD selection via `lodSelect`.
 * @param pool - Pool whose `chunkSize` sets the chunking granularity.
 * @returns A lazy `ChunkSource` over the selected LODs (`readRows` is LOD-0).
 * @ignore
 */
const readLccSource = async (
    fileSystem: ReadFileSystem,
    filename: string,
    options: Options,
    pool: ChunkDataPool
): Promise<ChunkSource> => {
    const lccJson = JSON.parse(new TextDecoder().decode(await readFile(fileSystem, filename)));
    const hasSH = lccHasSH(lccJson);
    const compressInfo = parseMeta(lccJson);
    const splats = lccJson.splats;

    const baseDir = dirname(filename);
    const relatedFilename = (name: string) => (baseDir ? join(baseDir, name) : name);
    const indexData = await readFile(fileSystem, relatedFilename('index.bin'));
    const unitInfos = parseIndexBin(
        indexData.buffer.slice(indexData.byteOffset, indexData.byteOffset + indexData.byteLength) as ArrayBuffer,
        lccJson
    );

    // input -> output LOD mapping (mirrors readLcc).
    const lodSelect = options.lodSelect ?? [];
    const lods = lodSelect.length > 0 ?
        lodSelect
        .map(lod => (lod < 0 ? splats.length + lod : lod))
        .filter(lod => lod >= 0 && lod < splats.length) :
        new Array(splats.length).fill(0).map((_, i) => i);
    if (lods.length === 0) {
        throw new Error(`No valid LODs selected for LCC input file: ${filename} lods: ${JSON.stringify(lods)}`);
    }

    // Open the bulk sources; kept open for the source's lifetime (read/readRows
    // range-read them), closed on close(). Guard the second open so a missing
    // shcoef.bin (hasSH is heuristic) doesn't leak the data.bin handle.
    const dataSource = await fileSystem.createSource(relatedFilename('data.bin'));
    let shSource: ReadSource | undefined;
    if (hasSH) {
        try {
            shSource = await fileSystem.createSource(relatedFilename('shcoef.bin'));
        } catch (err) {
            dataSource.close();
            throw err;
        }
    }

    // Per selected LOD: the ordered list of non-empty unit runs (index.bin order).
    // `dataByteOffset` is the unit's LOD block start in data.bin (its SH block
    // starts at 2x that); `globalStart` is the prefix-sum gaussian index of the
    // run's first row within the LOD. `starts` mirrors `globalStart` for search.
    type UnitRun = { dataByteOffset: number; points: number; globalStart: number };
    const runsByLod: UnitRun[][] = [];
    const startsByLod: number[][] = [];
    const lodCounts: number[] = [];
    for (const inputLod of lods) {
        const runs: UnitRun[] = [];
        const starts: number[] = [];
        let acc = 0;
        for (const info of unitInfos) {
            const lod = info.lods[inputLod];
            if (lod.points === 0) continue;
            runs.push({ dataByteOffset: Number(lod.offset), points: lod.points, globalStart: acc });
            starts.push(acc);
            acc += lod.points;
        }
        runsByLod.push(runs);
        startsByLod.push(starts);
        lodCounts.push(acc);
    }

    const chunkSize = pool.chunkSize;
    const shBands: SHBands = hasSH ? 3 : 0;
    const restCount = hasSH ? 45 : 0;
    const extraColumns: ExtraColumn[] = [
        { name: 'nx', type: 'float32' },
        { name: 'ny', type: 'float32' },
        { name: 'nz', type: 'float32' }
    ];
    const ol = otherLayout(extraColumns);
    const layouts: Partial<Record<ChunkLayer, LayerLayout>> = {
        position: { stride: POSITION_STRIDE, fields: positionFields() },
        geometric: { stride: GEOMETRIC_STRIDE, fields: geometricFields() },
        color: { stride: colorStride(shBands), fields: colorFields(shBands) },
        other: { stride: ol.stride, fields: ol.fields }
    };

    const meta: ChunkSourceMetadata = {
        numGaussians: lodCounts[0],
        numLods: lods.length,
        lodCounts,
        chunkSize,
        numChunks: lodCounts.map(c => Math.ceil(c / chunkSize)),
        shBands,
        extraColumns,
        transform: LCC_TRANSFORM(),
        availableLayers: new Set<ChunkLayer>(['position', 'geometric', 'color', 'other']),
        layouts
    };

    // Global (scene-wide) dequant constants, hoisted once (see processUnit).
    const sMinX = compressInfo.scaleMin.x, sMinY = compressInfo.scaleMin.y, sMinZ = compressInfo.scaleMin.z;
    const sMaxX = compressInfo.scaleMax.x, sMaxY = compressInfo.scaleMax.y, sMaxZ = compressInfo.scaleMax.z;
    const shMinX = compressInfo.shMin.x, shMinY = compressInfo.shMin.y, shMinZ = compressInfo.shMin.z;
    const shMaxX = compressInfo.shMax.x, shMaxY = compressInfo.shMax.y, shMaxZ = compressInfo.shMax.z;
    const colorSw = 3 + restCount;

    // Largest run index r with starts[r] <= g.
    const findRun = (starts: number[], g: number): number => {
        let lo = 0, hi = starts.length - 1, ans = 0;
        while (lo <= hi) {
            const mid = (lo + hi) >> 1;
            if (starts[mid] <= g) {
                ans = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return ans;
    };

    // Per-record dequantizer bound to the caller's output layer buffers. `srcIdx`
    // indexes the current source views (rebound per sub-block / gather run via
    // setViews); `dstRow` is the output gaussian row. Mirrors processUnit exactly.
    const makeDecoder = (
        position?: ChunkData, geometric?: ChunkData, color?: ChunkData, other?: ChunkData
    ) => {
        const pos = position ? new Float32Array(position.data) : null;
        const geo = geometric ? new Float32Array(geometric.data) : null;
        const col = color ? new Float32Array(color.data) : null;
        const oth = other ? new Float32Array(other.data) : null;
        let f32!: Float32Array;
        let u16!: Uint16Array;
        let u8!: Uint8Array;
        let shU32: Uint32Array | null = null;
        const setViews = (df: Float32Array, du16: Uint16Array, du8: Uint8Array, dsh: Uint32Array | null) => {
            f32 = df; u16 = du16; u8 = du8; shU32 = dsh;
        };
        const decode = (srcIdx: number, dstRow: number) => {
            if (pos) {
                const fi = srcIdx << 3;
                const o = dstRow * 3;
                pos[o] = f32[fi]; pos[o + 1] = f32[fi + 1]; pos[o + 2] = f32[fi + 2];
            }
            if (geo) {
                const bi = srcIdx << 5;
                const hi = srcIdx << 4;
                const o = dstRow << 3;
                // rotation (largest-3 quaternion); the u32 spans bytes 22-25, read
                // as two u16 because byte 22 isn't 4-byte aligned.
                const v = u16[hi + 11] | (u16[hi + 12] << 16);
                const d0 = (v & 1023) / 1023.0;
                const d1 = ((v >> 10) & 1023) / 1023.0;
                const d2 = ((v >> 20) & 1023) / 1023.0;
                const d3 = (v >> 30) & 3;
                const qx = d0 * SQRT_2 - SQRT_2_INV;
                const qy = d1 * SQRT_2 - SQRT_2_INV;
                const qz = d2 * SQRT_2 - SQRT_2_INV;
                const qw = Math.sqrt(1 - Math.min(1.0, qx * qx + qy * qy + qz * qz));
                if (d3 === 0) {
                    geo[o] = qz; geo[o + 1] = qw; geo[o + 2] = qx; geo[o + 3] = qy;
                } else if (d3 === 1) {
                    geo[o] = qz; geo[o + 1] = qx; geo[o + 2] = qw; geo[o + 3] = qy;
                } else if (d3 === 2) {
                    geo[o] = qz; geo[o + 1] = qx; geo[o + 2] = qy; geo[o + 3] = qw;
                } else {
                    geo[o] = qw; geo[o + 1] = qx; geo[o + 2] = qy; geo[o + 3] = qz;
                }
                geo[o + 4] = invLinearScale(mix(sMinX, sMaxX, u16[hi + 8] / 65535.0));
                geo[o + 5] = invLinearScale(mix(sMinY, sMaxY, u16[hi + 9] / 65535.0));
                geo[o + 6] = invLinearScale(mix(sMinZ, sMaxZ, u16[hi + 10] / 65535.0));
                geo[o + 7] = invSigmoid(u8[bi + 15] / 255.0);
            }
            if (col) {
                const bi = srcIdx << 5;
                const o = dstRow * colorSw;
                col[o] = invSH0ToColor(u8[bi + 12] / 255.0);
                col[o + 1] = invSH0ToColor(u8[bi + 13] / 255.0);
                col[o + 2] = invSH0ToColor(u8[bi + 14] / 255.0);
                if (shU32) {
                    const si = srcIdx << 4;
                    for (let j = 0; j < 15; j++) {
                        const enc = shU32[si + j];
                        col[o + 3 + j] = mix(shMinX, shMaxX, (enc & 0x7FF) / 2047.0);
                        col[o + 3 + j + 15] = mix(shMinY, shMaxY, ((enc >> 11) & 0x3FF) / 1023.0);
                        col[o + 3 + j + 30] = mix(shMinZ, shMaxZ, ((enc >> 21) & 0x7FF) / 2047.0);
                    }
                }
            }
            if (oth) {
                const hi = srcIdx << 4;
                const o = dstRow * 3;
                oth[o] = u16[hi + 13]; oth[o + 1] = u16[hi + 14]; oth[o + 2] = u16[hi + 15];
            }
        };
        return { setViews, decode, wantColor: !!col };
    };

    // Sub-block size: bounds the raw record scratch independent of chunkSize.
    const SUB_BLOCK = 1 << 16;
    let dScratch: Uint8Array | null = null;     // SUB_BLOCK * 32 (data records)
    let sScratch: Uint8Array | null = null;     // SUB_BLOCK * 64 (sh records)
    let gScratch: Uint8Array | null = null;     // readRows data gather
    let gShScratch: Uint8Array | null = null;   // readRows sh gather

    const read = async (request: ReadRequest): Promise<void> => {
        const lod = request.lod ?? 0;
        const runs = runsByLod[lod];
        if (!runs) {
            throw new Error(`readLcc: lod ${lod} out of range`);
        }
        if (!request.position && !request.geometric && !request.color && !request.other) {
            return;
        }
        const starts = startsByLod[lod];
        const dec = makeDecoder(request.position, request.geometric, request.color, request.other);

        if ('indices' in request) {
            // Random-access gather from a structural LOD. Resolve each index to
            // its data.bin byte offset, sort output slots by that (forward reads),
            // coalesce nearby records into one bounded range read (data + parallel
            // sh at 2x the offsets), dequant each into its scattered output row.
            const { indices, indexOffset, count } = request;
            if (count <= 0) return;

            const boff = new Float64Array(count);
            for (let j = 0; j < count; j++) {
                const g = indices[indexOffset + j];
                const run = runs[findRun(starts, g)];
                boff[j] = run.dataByteOffset + (g - run.globalStart) * 32;
            }
            const slot = sortGatherSlots(count, s => boff[s]);
            const wantSh = !!(shSource && dec.wantColor);
            const costBytes = 32 + (wantSh ? 64 : 0);

            for (const gr of gatherRuns(count, t => boff[slot[t]], 32, costBytes)) {
                const first = gr.firstByte;
                const rc = gr.recordCount;

                const need = rc * 32;
                if (!gScratch || gScratch.length < need) gScratch = new Uint8Array(need);
                const dataBytes = gScratch.subarray(0, need);
                if (await readExact(dataSource.read(first, first + need), dataBytes, 0, need) !== need) {
                    throw new Error(`readLcc: short gather read at byte ${first}`);
                }
                let shU32: Uint32Array | null = null;
                if (wantSh) {
                    const shNeed = rc * 64;
                    const shStart = first * 2;
                    if (!gShScratch || gShScratch.length < shNeed) gShScratch = new Uint8Array(shNeed);
                    const shBytes = gShScratch.subarray(0, shNeed);
                    if (await readExact(shSource!.read(shStart, shStart + shNeed), shBytes, 0, shNeed) !== shNeed) {
                        throw new Error(`readLcc: short sh gather read at byte ${shStart}`);
                    }
                    shU32 = new Uint32Array(shBytes.buffer, shBytes.byteOffset, rc * 16);
                }
                dec.setViews(
                    new Float32Array(dataBytes.buffer, dataBytes.byteOffset, rc * 8),
                    new Uint16Array(dataBytes.buffer, dataBytes.byteOffset, rc * 16),
                    dataBytes,
                    shU32
                );
                for (let t = gr.j0; t < gr.j1; t++) {
                    dec.decode((boff[slot[t]] - first) / 32, slot[t]);
                }
            }
            return;
        }

        // Contiguous chunk across this LOD's unit runs.
        const start = request.chunkIndex * chunkSize;
        const count = Math.min(chunkSize, lodCounts[lod] - start);
        if (count <= 0) {
            throw new Error(`readLcc: chunkIndex ${request.chunkIndex} out of range`);
        }
        const end = start + count;

        let r = findRun(starts, start);
        while (r < runs.length && runs[r].globalStart < end) {
            const run = runs[r];
            const localBeg = Math.max(0, start - run.globalStart);
            const localEnd = Math.min(run.points, end - run.globalStart);
            const dstRowBase = run.globalStart + localBeg - start;
            for (let off = localBeg; off < localEnd; off += SUB_BLOCK) {
                const bb = Math.min(SUB_BLOCK, localEnd - off);
                const dataStart = run.dataByteOffset + off * 32;
                dScratch ??= new Uint8Array(SUB_BLOCK * 32);
                const dataBytes = dScratch.subarray(0, bb * 32);
                if (await readExact(dataSource.read(dataStart, dataStart + bb * 32), dataBytes, 0, bb * 32) !== bb * 32) {
                    throw new Error(`readLcc: short data read for chunk ${request.chunkIndex}`);
                }
                let shU32: Uint32Array | null = null;
                if (shSource && dec.wantColor) {
                    const shStart = run.dataByteOffset * 2 + off * 64;
                    sScratch ??= new Uint8Array(SUB_BLOCK * 64);
                    const shBytes = sScratch.subarray(0, bb * 64);
                    if (await readExact(shSource.read(shStart, shStart + bb * 64), shBytes, 0, bb * 64) !== bb * 64) {
                        throw new Error(`readLcc: short sh read for chunk ${request.chunkIndex}`);
                    }
                    shU32 = new Uint32Array(shBytes.buffer, shBytes.byteOffset, bb * 16);
                }
                dec.setViews(
                    new Float32Array(dataBytes.buffer, dataBytes.byteOffset, bb * 8),
                    new Uint16Array(dataBytes.buffer, dataBytes.byteOffset, bb * 16),
                    dataBytes,
                    shU32
                );
                const dstBase = dstRowBase + (off - localBeg);
                for (let i = 0; i < bb; i++) {
                    dec.decode(i, dstBase + i);
                }
            }
            r++;
        }
    };

    return {
        meta,
        read,
        close: () => {
            dataSource.close();
            shSource?.close();
            return Promise.resolve();
        }
    };
};

/**
 * Decode an LCC scene's environment (skybox) splats as a resident `ChunkSource`,
 * or `null` if the file has no `environment.bin`. The environment is small (a
 * single contiguous block), so it is decoded eagerly here and bridged — only the
 * main scene needs the streaming {@link readLccSource}. Lets the streaming LOD
 * writer carry the env without eagerly decoding the whole main scene.
 *
 * @param fileSystem - File system for reading the LCC files.
 * @param filename - Path to the meta.lcc file.
 * @param pool - Pool whose `chunkSize` the env source is chunked at.
 * @returns The environment as a resident `ChunkSource`, or `null` if absent/empty.
 * @ignore
 */
const readLccEnvironmentSource = async (
    fileSystem: ReadFileSystem,
    filename: string,
    pool: ChunkDataPool
): Promise<ChunkSource | null> => {
    const lccJson = JSON.parse(new TextDecoder().decode(await readFile(fileSystem, filename)));
    const hasSH = lccHasSH(lccJson);
    const compressInfo = parseMeta(lccJson);

    const baseDir = dirname(filename);
    const relatedFilename = (name: string) => (baseDir ? join(baseDir, name) : name);

    try {
        const envData = await readFile(fileSystem, relatedFilename('environment.bin'));
        const envTable = deserializeEnvironment(envData, compressInfo, hasSH);
        if (envTable.numRows === 0) {
            return null;
        }
        envTable.transform = LCC_TRANSFORM();
        return dataTableToChunkSource(envTable, pool.chunkSize);
    } catch (err) {
        // environment.bin is optional — a missing file is the normal "no skybox"
        // case (signalled differently per backend; mirror readLcc's suppression).
        const code = (err as { code?: string })?.code;
        const message = (err as Error)?.message ?? '';
        const isMissing = code === 'ENOENT' ||
            message.startsWith('Entry not found') ||
            message.startsWith('HTTP error 404');
        if (!isMissing) {
            logger.warn(`failed to load environment.bin: ${message || err}`);
        }
        return null;
    }
};

export { readLcc, readLccSource, readLccEnvironmentSource };
