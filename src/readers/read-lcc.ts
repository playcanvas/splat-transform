import { Buffer } from 'node:buffer';
import { FileHandle, open } from 'node:fs/promises';
import { dirname, join } from 'node:path';

import { Vec3 } from 'playcanvas';

import { Column, DataTable } from '../data-table/data-table';
import { Options } from '../types';

const kSH_C0 = 0.28209479177387814;
const SQRT_2 = 1.414213562373095;
const SQRT_2_INV = 0.7071067811865475;

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

// parameters used to convert LCC data into GSplatData
type LccParam = {
    totalSplats: number;
    targetLod: number;
    compressInfo: CompressInfo;
    unitInfos: Array<LccUnitInfo>;
    dataFile: FileHandle;
    shFile?: FileHandle;
}

type ProcessUnitContext = {
    info: LccUnitInfo;
    targetLod: number;
    dataFile: FileHandle;
    shFile?: FileHandle;
    compressInfo: CompressInfo;
    propertyOffset: number;
    properties: Record<string, Float32Array>;
    properties_f_rest: Float32Array[] | null;
}

const readPart = async (fh: FileHandle, start: number, end: number): Promise<Uint8Array> => {
    const buf = Buffer.alloc(end - start);
    await fh.read(buf, 0, end - start, start);
    return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
};

const read = async (fh: FileHandle): Promise<Uint8Array> => {
    const stat = await fh.stat();
    const buf = Buffer.alloc(stat.size);
    await fh.read(buf, 0, stat.size, 0);
    return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
};

const openAndRead = async (pathname: string): Promise<Uint8Array> => {
    const handle = await open(pathname, 'r');
    try {
        return await read(handle);
    } finally {
        await handle.close();
    }
};

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

    const compressInfo: CompressInfo = {
        scaleMin,
        scaleMax,
        shMin,
        shMax,
        envScaleMin,
        envScaleMax,
        envShMin,
        envShMax
    };

    return compressInfo;
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

const mixVec3 = (min: Vec3, max: Vec3, v: Vec3): Vec3 => {
    return new Vec3(
        mix(min.x, max.x, v.x),
        mix(min.y, max.y, v.y),
        mix(min.z, max.z, v.z)
    );
};

const decodePacked_11_10_11 = (res: Vec3, enc: number) => {
    res.x = (enc & 0x7FF) / 2047.0;
    res.y = ((enc >> 11) & 0x3FF) / 1023.0;
    res.z = ((enc >> 21) & 0x7FF) / 2047.0;
};

const decodeRotation = (v: number) => {
    const d0 = (v & 1023) / 1023.0;
    const d1 = ((v >> 10) & 1023) / 1023.0;
    const d2 = ((v >> 20) & 1023) / 1023.0;
    const d3 = (v >> 30) & 3;

    const qx = d0 * SQRT_2 - SQRT_2_INV;
    const qy = d1 * SQRT_2 - SQRT_2_INV;
    const qz = d2 * SQRT_2 - SQRT_2_INV;
    let sum = qx * qx + qy * qy + qz * qz;
    sum = Math.min(1.0, sum);
    const qw = Math.sqrt(1 - sum);

    if (d3 === 0) {
        return [qw, qx, qy, qz];
    } else if (d3 === 1) {
        return [qx, qw, qy, qz];
    } else if (d3 === 2) {
        return [qx, qy, qw, qz];
    }
    return [qx, qy, qz, qw];
};

const floatProps = [
    'x', 'y', 'z',
    'nx', 'ny', 'nz',
    'opacity',
    'rot_0', 'rot_1', 'rot_2', 'rot_3',
    'f_dc_0', 'f_dc_1', 'f_dc_2',
    'scale_0', 'scale_1', 'scale_2'
];

const createStorage = (length: number) => new Float32Array(length);

const initProperties = (length: number): Record<string, Float32Array> => {
    const props: Record<string, Float32Array> = {};
    for (const key of floatProps) {
        props[`property_${key}`] = createStorage(length);
    }
    return props;
};

const vec3 = new Vec3();

const decodeSplat = (
    dataView: DataView,
    shDataView: DataView | null,
    i: number,
    compressInfo: CompressInfo,
    unitProperties: Record<string, Float32Array>,
    unitProperties_f_rest: Float32Array[] | null
) => {
    const off = i * 32;

    // position
    unitProperties.property_x[i] = dataView.getFloat32(off + 0, true);
    unitProperties.property_y[i] = dataView.getFloat32(off + 4, true);
    unitProperties.property_z[i] = dataView.getFloat32(off + 8, true);

    // decode color
    unitProperties.property_f_dc_0[i] = invSH0ToColor(dataView.getUint8(off + 12) / 255.0);
    unitProperties.property_f_dc_1[i] = invSH0ToColor(dataView.getUint8(off + 13) / 255.0);
    unitProperties.property_f_dc_2[i] = invSH0ToColor(dataView.getUint8(off + 14) / 255.0);
    unitProperties.property_opacity[i] = invSigmoid(dataView.getUint8(off + 15) / 255.0);

    // decode scale
    const scaleMin = compressInfo.scaleMin;
    const scaleMax = compressInfo.scaleMax;
    unitProperties.property_scale_0[i] = invLinearScale(mix(scaleMin.x, scaleMax.x, dataView.getUint16(off + 16, true) / 65535.0));
    unitProperties.property_scale_1[i] = invLinearScale(mix(scaleMin.y, scaleMax.y, dataView.getUint16(off + 18, true) / 65535.0));
    unitProperties.property_scale_2[i] = invLinearScale(mix(scaleMin.z, scaleMax.z, dataView.getUint16(off + 20, true) / 65535.0));

    // decode rotation
    const q = decodeRotation(dataView.getUint32(off + 22, true));
    unitProperties.property_rot_0[i] = q[3];// w
    unitProperties.property_rot_1[i] = q[0];// x
    unitProperties.property_rot_2[i] = q[1];// y
    unitProperties.property_rot_3[i] = q[2];// z

    // normal
    unitProperties.property_nx[i] = dataView.getUint16(off + 26, true);
    unitProperties.property_ny[i] = dataView.getUint16(off + 28, true);
    unitProperties.property_nz[i] = dataView.getUint16(off + 30, true);

    // SH
    if (shDataView && unitProperties_f_rest) {
        const shOff = off * 2;
        const SHValues = Array.from({ length: 15 }, (_, idx) => shDataView.getUint32(shOff + idx * 4, true));
        const { shMin, shMax } = compressInfo;
        const vecSHValues = SHValues.map((sh) => {
            decodePacked_11_10_11(vec3, sh);
            return mixVec3(shMin, shMax, vec3);
        });

        for (let j = 0; j < 15; j++) {
            unitProperties_f_rest[j][i] = vecSHValues[j].x;
            unitProperties_f_rest[j + 15][i] = vecSHValues[j].y;
            unitProperties_f_rest[j + 30][i] = vecSHValues[j].z;
        }
    }
};

const processUnit = async (ctx: ProcessUnitContext) => {
    const {
        info,
        targetLod,
        dataFile,
        shFile,
        compressInfo,
        propertyOffset,
        properties,
        properties_f_rest
    } = ctx;

    const lod = info.lods[targetLod];
    const unitSplats = lod.points;
    const offset = Number(lod.offset);
    const size = lod.size;

    if (unitSplats === 0) {
        return propertyOffset;
    }

    // load data
    const dataSource = await readPart(dataFile, offset, offset + size);
    const dataView = new DataView(dataSource.buffer);

    // load sh data
    let shDataView: DataView | null = null;
    if (shFile) {
        const shSource = await readPart(shFile, offset * 2, offset * 2 + size * 2);
        shDataView = new DataView(shSource.buffer);
    }

    const unitProperties = initProperties(unitSplats);
    const unitProperties_f_rest = shFile ? Array.from({ length: 45 }, () => new Float32Array(unitSplats)) : null;

    for (let i = 0; i < unitSplats; i++) {
        decodeSplat(dataView, shDataView, i, compressInfo, unitProperties, unitProperties_f_rest);
    }

    for (const key of floatProps) {
        properties[`property_${key}`].set(unitProperties[`property_${key}`], propertyOffset);
    }

    if (unitProperties_f_rest) {
        for (let j = 0; j < 45; j++) {
            properties_f_rest[j].set(unitProperties_f_rest[j], propertyOffset);
        }
    }

    return propertyOffset + unitSplats;
};

// this function would stream data directly into GSplatData buffers
const deserializeFromLcc = async (param: LccParam) => {
    const { totalSplats, unitInfos, targetLod, dataFile, shFile, compressInfo } = param;

    // properties to GSplatData
    const properties: Record<string, Float32Array> = initProperties(totalSplats);
    const properties_f_rest = shFile ? Array.from({ length: 45 }, () => createStorage(totalSplats)) : null;

    let propertyOffset = 0;
    for (const info of unitInfos) {
        propertyOffset = await processUnit({
            info,
            targetLod,
            dataFile,
            shFile,
            compressInfo,
            propertyOffset,
            properties,
            properties_f_rest
        });
    }

    const columns = [
        ...floatProps.map(name => new Column(name, properties[`property_${name}`])),
        ...(properties_f_rest ? properties_f_rest.map((storage, i) => new Column(`f_rest_${i}`, storage)) : [])
    ];

    return new DataTable(columns);
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

        const q = decodeRotation(dataView.getUint32(off + 22, true));
        columns[10].data[i] = q[3]; // rot_0 (w)
        columns[11].data[i] = q[0]; // rot_1 (x)
        columns[12].data[i] = q[1]; // rot_2 (y)
        columns[13].data[i] = q[2]; // rot_3 (z)

        // skip normal 26-32

        if (hasSH) {
            for (let j = 0; j < 15; ++j) {
                decodePacked_11_10_11(vec3, dataView.getUint32(off + 32 + j * 4, true));

                columns[14 + j].data[i] = mix(shMin.x, shMax.x, vec3.x);
                columns[14 + j + 15].data[i] = mix(shMin.y, shMax.y, vec3.y);
                columns[14 + j + 30].data[i] = mix(shMin.z, shMax.z, vec3.z);
            }
        }
    }

    return new DataTable(columns);
};

const readLcc = async (fileHandle: FileHandle, sourceName: string, options: Options): Promise<DataTable[]> => {
    const lccData = await read(fileHandle);
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

    const relatedFilename = (name: string) => join(dirname(sourceName ?? ''), name);

    const indexData = await openAndRead(relatedFilename('index.bin'));
    const dataFile = await open(relatedFilename('data.bin'), 'r');
    const shFile = hasSH ? await open(relatedFilename('shcoef.bin'), 'r') : null;

    const unitInfos: LccUnitInfo[] = parseIndexBin(indexData.buffer as ArrayBuffer, lccJson);

    // build table of input -> output lods
    const lods = options.lodSelect.length > 0 ?
        options.lodSelect
        .map(lod => (lod < 0 ? splats.length + lod : lod))    // negative indices map from the end of lod
        .filter(lod => lod >= 0 && lod < splats.length) :
        new Array(splats.length).fill(0).map((_, i) => i);

    if (lods.length === 0) {
        throw new Error(`No valid LODs selected for LCC input file: ${sourceName} lods: ${JSON.stringify(lods)}`);
    }

    const result = [];

    for (let i = 0; i < lods.length; i++) {
        const inputLod = lods[i];
        const outputLod = i;
        const totalSplats = splats[inputLod];

        const dataTable = await deserializeFromLcc({
            totalSplats,
            unitInfos,
            targetLod: inputLod,
            dataFile,
            shFile,
            compressInfo
        });

        dataTable.addColumn(new Column('lod', new Float32Array(totalSplats).fill(outputLod)));

        result.push(dataTable);
    }

    // cleanup
    await dataFile.close();
    if (shFile) {
        await shFile.close();
    }

    // load environment and tag as lod -1
    try {
        const envData = await openAndRead(relatedFilename('environment.bin'));
        const envDataTable  = await deserializeEnvironment(envData, compressInfo, hasSH);
        envDataTable.addColumn(new Column('lod', new Float32Array(envDataTable.numRows).fill(-1)));
        result.push(envDataTable);
    } catch (err) {
        console.warn('Failed to load environment.bin', err);
    }

    return result;
};

export { readLcc };
