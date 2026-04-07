import { version } from '../../../package.json';
import { DataTable } from '../data-table/data-table';
import { convertToSpace } from '../data-table/transform';
import { Transform } from '../utils/math';
import { type FileSystem } from '../io/write';
import { sigmoid } from '../utils/math';

const SH_C0 = 0.2820947917738781;

const shRestNames = new Array(45).fill('').map((_: string, i: number) => `f_rest_${i}`);

type WriteGlbOptions = {
    filename: string;
    dataTable: DataTable;
};

// glTF accessor component types
const FLOAT = 5126;
const UNSIGNED_BYTE = 5121;

// glTF buffer view targets
const ARRAY_BUFFER = 34962;

// GLB chunk types
const GLB_MAGIC = 0x46546C67;
const GLB_VERSION = 2;
const JSON_CHUNK_TYPE = 0x4E4F534A;
const BIN_CHUNK_TYPE = 0x004E4942;

/**
 * Determines how many SH bands (0-3) the DataTable contains beyond the DC term.
 * Band detection uses the same channel-major layout as the rest of the codebase:
 * N coefficients per channel, 3 channels, stored as f_rest_0..f_rest_(3N-1).
 *
 * @param dataTable - The DataTable to inspect.
 * @returns The number of SH bands (0-3).
 */
const getSHBands = (dataTable: DataTable): number => {
    const idx = shRestNames.findIndex(v => !dataTable.hasColumn(v));
    return ({ '9': 1, '24': 2, '-1': 3 } as Record<string, number>)[String(idx)] ?? 0;
};

/**
 * Computes POSITION accessor min/max bounds required by the glTF spec.
 *
 * @param x - X position data.
 * @param y - Y position data.
 * @param z - Z position data.
 * @returns Object with min and max arrays.
 */
const computePositionBounds = (x: Float32Array, y: Float32Array, z: Float32Array) => {
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    for (let i = 0; i < x.length; i++) {
        if (x[i] < minX) minX = x[i];
        if (x[i] > maxX) maxX = x[i];
        if (y[i] < minY) minY = y[i];
        if (y[i] > maxY) maxY = y[i];
        if (z[i] < minZ) minZ = z[i];
        if (z[i] > maxZ) maxZ = z[i];
    }

    return {
        min: [minX, minY, minZ],
        max: [maxX, maxY, maxZ]
    };
};

/**
 * Pads a byte length up to the next multiple of 4.
 *
 * @param n - The byte length to align.
 * @returns The aligned byte length.
 */
const align4 = (n: number) => (n + 3) & ~3;

/**
 * Builds the binary buffer containing all attribute data, and returns the
 * list of buffer views / accessor descriptions needed for the glTF JSON.
 *
 * @param dataTable - The DataTable containing splat data.
 * @param numSplats - Number of splats in the table.
 * @param shBands - Number of SH bands beyond DC (0-3).
 * @returns Object with segments, offsets, and the assembled binary buffer.
 */
const buildBinaryBuffer = (dataTable: DataTable, numSplats: number, shBands: number) => {
    const x = dataTable.getColumnByName('x')!.data as Float32Array;
    const y = dataTable.getColumnByName('y')!.data as Float32Array;
    const z = dataTable.getColumnByName('z')!.data as Float32Array;
    const rot0 = dataTable.getColumnByName('rot_0')!.data as Float32Array;
    const rot1 = dataTable.getColumnByName('rot_1')!.data as Float32Array;
    const rot2 = dataTable.getColumnByName('rot_2')!.data as Float32Array;
    const rot3 = dataTable.getColumnByName('rot_3')!.data as Float32Array;
    const scale0 = dataTable.getColumnByName('scale_0')!.data as Float32Array;
    const scale1 = dataTable.getColumnByName('scale_1')!.data as Float32Array;
    const scale2 = dataTable.getColumnByName('scale_2')!.data as Float32Array;
    const opacity = dataTable.getColumnByName('opacity')!.data as Float32Array;
    const fdc0 = dataTable.getColumnByName('f_dc_0')!.data as Float32Array;
    const fdc1 = dataTable.getColumnByName('f_dc_1')!.data as Float32Array;
    const fdc2 = dataTable.getColumnByName('f_dc_2')!.data as Float32Array;

    // Coefficients per channel for each SH band beyond DC
    const coeffsPerChannel = [0, 3, 8, 15][shBands];
    const shCoefCount = [0, 3, 5, 7]; // per-degree coefficient count
    const shDegrees = shBands; // degrees 1..shDegrees have rest coefficients

    // Collect all binary segments and their metadata
    type Segment = {
        name: string;
        data: ArrayBuffer;
        componentType: number;
        type: string;
        count: number;
        normalized?: boolean;
        min?: number[];
        max?: number[];
    };

    const segments: Segment[] = [];

    // POSITION (VEC3 float)
    const posData = new Float32Array(numSplats * 3);
    for (let i = 0; i < numSplats; i++) {
        posData[i * 3] = x[i];
        posData[i * 3 + 1] = y[i];
        posData[i * 3 + 2] = z[i];
    }
    const bounds = computePositionBounds(x, y, z);
    segments.push({
        name: 'POSITION',
        data: posData.buffer,
        componentType: FLOAT,
        type: 'VEC3',
        count: numSplats,
        min: bounds.min,
        max: bounds.max
    });

    // COLOR_0 fallback (VEC4 unsigned byte normalized)
    const colorData = new Uint8Array(numSplats * 4);
    for (let i = 0; i < numSplats; i++) {
        const r = Math.max(0, Math.min(255, Math.round((fdc0[i] * SH_C0 + 0.5) * 255)));
        const g = Math.max(0, Math.min(255, Math.round((fdc1[i] * SH_C0 + 0.5) * 255)));
        const b = Math.max(0, Math.min(255, Math.round((fdc2[i] * SH_C0 + 0.5) * 255)));
        const a = Math.max(0, Math.min(255, Math.round(sigmoid(opacity[i]) * 255)));
        colorData[i * 4] = r;
        colorData[i * 4 + 1] = g;
        colorData[i * 4 + 2] = b;
        colorData[i * 4 + 3] = a;
    }
    segments.push({
        name: 'COLOR_0',
        data: colorData.buffer,
        componentType: UNSIGNED_BYTE,
        type: 'VEC4',
        count: numSplats,
        normalized: true
    });

    // KHR_gaussian_splatting:ROTATION (VEC4 float, xyzw order)
    const rotData = new Float32Array(numSplats * 4);
    for (let i = 0; i < numSplats; i++) {
        rotData[i * 4] = rot1[i];     // x
        rotData[i * 4 + 1] = rot2[i]; // y
        rotData[i * 4 + 2] = rot3[i]; // z
        rotData[i * 4 + 3] = rot0[i]; // w
    }
    segments.push({
        name: 'KHR_gaussian_splatting:ROTATION',
        data: rotData.buffer,
        componentType: FLOAT,
        type: 'VEC4',
        count: numSplats
    });

    // KHR_gaussian_splatting:SCALE (VEC3 float, log-space)
    const scaleData = new Float32Array(numSplats * 3);
    for (let i = 0; i < numSplats; i++) {
        scaleData[i * 3] = scale0[i];
        scaleData[i * 3 + 1] = scale1[i];
        scaleData[i * 3 + 2] = scale2[i];
    }
    segments.push({
        name: 'KHR_gaussian_splatting:SCALE',
        data: scaleData.buffer,
        componentType: FLOAT,
        type: 'VEC3',
        count: numSplats
    });

    // KHR_gaussian_splatting:OPACITY (SCALAR float, sigmoid-activated)
    const opacityData = new Float32Array(numSplats);
    for (let i = 0; i < numSplats; i++) {
        opacityData[i] = sigmoid(opacity[i]);
    }
    segments.push({
        name: 'KHR_gaussian_splatting:OPACITY',
        data: opacityData.buffer,
        componentType: FLOAT,
        type: 'SCALAR',
        count: numSplats
    });

    // SH_DEGREE_0_COEF_0 (VEC3 float, raw DC coefficients)
    const shDC = new Float32Array(numSplats * 3);
    for (let i = 0; i < numSplats; i++) {
        shDC[i * 3] = fdc0[i];
        shDC[i * 3 + 1] = fdc1[i];
        shDC[i * 3 + 2] = fdc2[i];
    }
    segments.push({
        name: 'KHR_gaussian_splatting:SH_DEGREE_0_COEF_0',
        data: shDC.buffer,
        componentType: FLOAT,
        type: 'VEC3',
        count: numSplats
    });

    // Higher-order SH coefficients (degrees 1-3)
    // Internal layout is channel-major: f_rest[k] for k in [0..N-1] is red,
    // [N..2N-1] is green, [2N..3N-1] is blue, where N = coeffsPerChannel.
    if (shDegrees > 0) {
        const restData: Float32Array[] = [];
        for (let k = 0; k < coeffsPerChannel; k++) {
            restData.push(dataTable.getColumnByName(shRestNames[k])!.data as Float32Array);                     // red
            restData.push(dataTable.getColumnByName(shRestNames[k + coeffsPerChannel])!.data as Float32Array);  // green
            restData.push(dataTable.getColumnByName(shRestNames[k + 2 * coeffsPerChannel])!.data as Float32Array); // blue
        }

        let coefOffset = 0;
        for (let degree = 1; degree <= shDegrees; degree++) {
            const numCoefs = shCoefCount[degree];
            for (let c = 0; c < numCoefs; c++) {
                const k = coefOffset + c;
                const rChannel = restData[k * 3];
                const gChannel = restData[k * 3 + 1];
                const bChannel = restData[k * 3 + 2];

                const buf = new Float32Array(numSplats * 3);
                for (let i = 0; i < numSplats; i++) {
                    buf[i * 3] = rChannel[i];
                    buf[i * 3 + 1] = gChannel[i];
                    buf[i * 3 + 2] = bChannel[i];
                }
                segments.push({
                    name: `KHR_gaussian_splatting:SH_DEGREE_${degree}_COEF_${c}`,
                    data: buf.buffer,
                    componentType: FLOAT,
                    type: 'VEC3',
                    count: numSplats
                });
            }
            coefOffset += numCoefs;
        }
    }

    // Compute total binary buffer size (each segment aligned to 4 bytes)
    let totalSize = 0;
    const offsets: number[] = [];
    for (const seg of segments) {
        offsets.push(totalSize);
        totalSize += align4(seg.data.byteLength);
    }

    // Assemble the binary buffer
    const binBuffer = new Uint8Array(totalSize);
    for (let i = 0; i < segments.length; i++) {
        binBuffer.set(new Uint8Array(segments[i].data), offsets[i]);
    }

    return { segments, offsets, binBuffer };
};

/**
 * Writes Gaussian splat data to a GLB file using the KHR_gaussian_splatting extension.
 *
 * @param options - Options including filename and data table to write.
 * @param fs - File system for writing the output file.
 * @ignore
 */
const writeGlb = async (options: WriteGlbOptions, fs: FileSystem) => {
    const { filename } = options;
    const dataTable = convertToSpace(options.dataTable, Transform.IDENTITY);
    const numSplats = dataTable.numRows;
    const shBands = getSHBands(dataTable);

    const { segments, offsets, binBuffer } = buildBinaryBuffer(dataTable, numSplats, shBands);

    // Build glTF JSON
    const bufferViews: any[] = [];
    const accessors: any[] = [];
    const attributes: Record<string, number> = {};

    for (let i = 0; i < segments.length; i++) {
        const seg = segments[i];

        bufferViews.push({
            buffer: 0,
            byteOffset: offsets[i],
            byteLength: seg.data.byteLength,
            target: ARRAY_BUFFER
        });

        const accessor: any = {
            bufferView: i,
            byteOffset: 0,
            componentType: seg.componentType,
            count: seg.count,
            type: seg.type
        };

        if (seg.normalized) {
            accessor.normalized = true;
        }
        if (seg.min) {
            accessor.min = seg.min;
            accessor.max = seg.max;
        }

        accessors.push(accessor);
        attributes[seg.name] = i;
    }

    const gltf: any = {
        asset: {
            version: '2.0',
            generator: `splat-transform ${version}`
        },
        extensionsUsed: ['KHR_gaussian_splatting'],
        scene: 0,
        scenes: [{ nodes: [0] }],
        nodes: [{ mesh: 0 }],
        buffers: [{ byteLength: binBuffer.byteLength }],
        bufferViews,
        accessors,
        meshes: [{
            primitives: [{
                attributes,
                mode: 0,
                extensions: {
                    KHR_gaussian_splatting: {
                        kernel: 'ellipse',
                        colorSpace: 'srgb_rec709_display',
                        sortingMethod: 'cameraDistance',
                        projection: 'perspective'
                    }
                }
            }]
        }]
    };

    const jsonString = JSON.stringify(gltf);
    const jsonEncoder = new TextEncoder();
    const jsonBytes = jsonEncoder.encode(jsonString);
    const jsonPaddedLength = align4(jsonBytes.byteLength);
    const binPaddedLength = align4(binBuffer.byteLength);

    const totalLength = 12 + 8 + jsonPaddedLength + 8 + binPaddedLength;

    const glb = new ArrayBuffer(totalLength);
    const view = new DataView(glb);
    const bytes = new Uint8Array(glb);

    // GLB header (12 bytes)
    view.setUint32(0, GLB_MAGIC, true);
    view.setUint32(4, GLB_VERSION, true);
    view.setUint32(8, totalLength, true);

    // JSON chunk header (8 bytes)
    let offset = 12;
    view.setUint32(offset, jsonPaddedLength, true);
    view.setUint32(offset + 4, JSON_CHUNK_TYPE, true);
    offset += 8;

    // JSON chunk data (padded with spaces per spec)
    bytes.set(jsonBytes, offset);
    for (let i = jsonBytes.byteLength; i < jsonPaddedLength; i++) {
        bytes[offset + i] = 0x20; // space
    }
    offset += jsonPaddedLength;

    // BIN chunk header (8 bytes)
    view.setUint32(offset, binPaddedLength, true);
    view.setUint32(offset + 4, BIN_CHUNK_TYPE, true);
    offset += 8;

    // BIN chunk data (padded with zeros per spec)
    bytes.set(binBuffer, offset);

    // Write the GLB file
    const writer = await fs.createWriter(filename);
    await writer.write(new Uint8Array(glb));
    await writer.close();
};

export { writeGlb };
