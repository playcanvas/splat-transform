import { Vec3 } from 'playcanvas';

import { Column, DataTable } from '../data-table/data-table';
import { dirname, join, ReadFileSystem, ReadSource, readFile } from '../io/read';
import { Options } from '../types';

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
const processUnit = async (
    info: LccUnitInfo,
    targetLod: number,
    dataSource: ReadSource,
    shSource: ReadSource | undefined,
    compressInfo: CompressInfo,
    propertyOffset: number,
    properties: Record<string, Float32Array>,
    properties_f_rest: Float32Array[] | null
) => {
    const lod = info.lods[targetLod];
    const unitSplats = lod.points;
    if (unitSplats === 0) return;

    const offset = Number(lod.offset);
    const size = lod.size;

    // load data using range read
    const dataBytes = await dataSource.read(offset, offset + size).readAll();
    const dataView = new DataView(dataBytes.buffer, dataBytes.byteOffset, dataBytes.byteLength);

    // load sh data using range read
    let shDataView: DataView | null = null;
    if (shSource) {
        const shBytes = await shSource.read(offset * 2, offset * 2 + size * 2).readAll();
        shDataView = new DataView(shBytes.buffer, shBytes.byteOffset, shBytes.byteLength);
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
        const off = i * 32;

        px[g] = dataView.getFloat32(off, true);
        py[g] = dataView.getFloat32(off + 4, true);
        pz[g] = dataView.getFloat32(off + 8, true);

        pdc0[g] = invSH0ToColor(dataView.getUint8(off + 12) / 255.0);
        pdc1[g] = invSH0ToColor(dataView.getUint8(off + 13) / 255.0);
        pdc2[g] = invSH0ToColor(dataView.getUint8(off + 14) / 255.0);
        pop[g] = invSigmoid(dataView.getUint8(off + 15) / 255.0);

        ps0[g] = invLinearScale(mix(sMinX, sMaxX, dataView.getUint16(off + 16, true) / 65535.0));
        ps1[g] = invLinearScale(mix(sMinY, sMaxY, dataView.getUint16(off + 18, true) / 65535.0));
        ps2[g] = invLinearScale(mix(sMinZ, sMaxZ, dataView.getUint16(off + 20, true) / 65535.0));

        decodeRotationInto(dataView.getUint32(off + 22, true), pr0, pr1, pr2, pr3, g);

        pnx[g] = dataView.getUint16(off + 26, true);
        pny[g] = dataView.getUint16(off + 28, true);
        pnz[g] = dataView.getUint16(off + 30, true);

        if (shDataView && properties_f_rest) {
            const shOff = i * 64;
            for (let j = 0; j < 15; j++) {
                const enc = shDataView.getUint32(shOff + j * 4, true);
                properties_f_rest[j][g] = mix(shMinX, shMaxX, (enc & 0x7FF) / 2047.0);
                properties_f_rest[j + 15][g] = mix(shMinY, shMaxY, ((enc >> 11) & 0x3FF) / 1023.0);
                properties_f_rest[j + 30][g] = mix(shMinZ, shMaxZ, ((enc >> 21) & 0x7FF) / 2047.0);
            }
        }
    }
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
    properties_f_rest: Float32Array[] | null
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
                compressInfo, offsets[idx], properties, properties_f_rest
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
    const dataSource = await fileSystem.createSource(relatedFilename('data.bin'));
    const shSource = hasSH ? await fileSystem.createSource(relatedFilename('shcoef.bin')) : null;

    const unitInfos: LccUnitInfo[] = parseIndexBin(indexData.buffer.slice(indexData.byteOffset, indexData.byteOffset + indexData.byteLength) as ArrayBuffer, lccJson);

    // build table of input -> output lods
    const lods = options.lodSelect.length > 0 ?
        options.lodSelect
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

    try {
        let lodOffset = 0;
        for (let i = 0; i < lods.length; i++) {
            const inputLod = lods[i];
            const outputLod = i;
            const totalSplats = splats[inputLod];

            await decodeUnitsForLod(
                unitInfos, inputLod, dataSource, shSource ?? undefined,
                compressInfo, lodOffset, properties, properties_f_rest
            );

            lodColumn.fill(outputLod, lodOffset, lodOffset + totalSplats);
            lodOffset += totalSplats;
        }
    } finally {
        dataSource.close();
        if (shSource) {
            shSource.close();
        }
    }

    const columns = [
        ...floatProps.map(name => new Column(name, properties[`property_${name}`])),
        ...(properties_f_rest ? properties_f_rest.map((storage, i) => new Column(`f_rest_${i}`, storage)) : []),
        new Column('lod', lodColumn)
    ];

    const result: DataTable[] = [new DataTable(columns)];

    // load environment and tag as lod -1
    try {
        const envData = await readFile(fileSystem, relatedFilename('environment.bin'));
        const envDataTable = deserializeEnvironment(envData, compressInfo, hasSH);
        envDataTable.addColumn(new Column('lod', new Float32Array(envDataTable.numRows).fill(-1)));
        result.push(envDataTable);
    } catch (err) {
        console.warn('Failed to load environment.bin', err);
    }

    return result;
};

export { readLcc };
