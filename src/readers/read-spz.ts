import { Buffer } from 'node:buffer';
import { FileHandle } from 'node:fs/promises';

import { Column, DataTable } from '../data-table';

// See https://github.com/nianticlabs/spz for reference implementation

type SplatData = {
    comments: string[];
    elements: {
        name: string,
        dataTable: DataTable
    }[];
};

const decompressGZIP = async (fileHandle: FileHandle): Promise<Buffer<ArrayBuffer>> => {
    const stats = await fileHandle.stat();
    const zippedSize = stats.size;
    const fileBuffer = Buffer.alloc(zippedSize);
    await fileHandle.read(fileBuffer, 0, zippedSize, 0);

    const blob = new Blob([fileBuffer.buffer], { type: 'application/gzip' });
    const ds = new DecompressionStream('gzip');
    const decompressionStream = blob.stream().pipeThrough(ds);
    const arrayBuffer = await new Response(decompressionStream).arrayBuffer();

    return Buffer.from(arrayBuffer);
};

// Coefficient used by niantic labs spz to have better results with Spherical harmonics.
const SH_C0_2 = 0.15;
function inverseConvertColorFromSPZ(y: number) {
    return (y / 255.0 - 0.5) / SH_C0_2;
}

function getFixed24(positionsView: DataView, elementIndex: number, memberIndex: number) {
    const sizeofMember = 3; // 24 bits is 3 bytes
    const stride = 3 * sizeofMember; // x y z
    let fixed32 = positionsView.getUint8(elementIndex * stride + memberIndex * sizeofMember + 0);
    fixed32 |= positionsView.getUint8(elementIndex * stride + memberIndex * sizeofMember + 1) << 8;
    fixed32 |= positionsView.getUint8(elementIndex * stride + memberIndex * sizeofMember + 2) << 16;
    fixed32 |= (fixed32 & 0x800000) ? 0xff000000 : 0;  // sign extension

    return fixed32;
}

const readSPZ = async (fileHandle: FileHandle): Promise<SplatData> => {
    // Load magic
    const magicSize = 4;
    let fileBuffer = Buffer.alloc(magicSize);
    await fileHandle.read(fileBuffer, 0, magicSize, 0);
    let magicView = new DataView(fileBuffer.buffer, 0, magicSize);

    // If file is GZip compressed, decompress it first.
    let isGZipped = false;
    if (magicView.getUint16(0) === 0x1F8B) { // '1F 8B' is the magic for gzip: https://en.wikipedia.org/wiki/Gzip
        isGZipped = true;
        fileBuffer = await decompressGZIP(fileHandle);

        magicView = new DataView(fileBuffer.buffer, 0, magicSize);
    }

    const magic = magicView.getUint32(0, true);
    if (magic !== 0x5053474e) { // NGSP
        throw new Error('invalid file header');
    }

    const HEADER_SIZE = 16;
    const totalSize = fileBuffer.buffer.byteLength;

    if (totalSize < HEADER_SIZE) {
        throw new Error('File too small to be valid .spz format');
    }

    // Parse header
    if (isGZipped === false) {
        await fileHandle.read(fileBuffer, 0, totalSize, 0);
    }
    const header = new DataView(fileBuffer.buffer, fileBuffer.byteOffset, HEADER_SIZE);

    const version = header.getUint32(4, true);
    if (!(version === 2 || version === 3)) {
        throw new Error(`Unsupported version ${version}`);
    }
    const numSplats = header.getUint32(8, true);
    const shDegree = header.getUint8(12);
    const fractionalBits = header.getUint8(13);
    const unused_flags = header.getUint8(14); // Trained with antialiasing or not
    const unused_reserved = header.getUint8(15); // padding

    const positionsByteSize = numSplats * 3 * 3; // 3 * 24bit values
    const alphasByteSize = numSplats; // u8
    const colorsByteSize = numSplats * 3; // u8 * 3
    const scalesByteSize = numSplats * 3; // u8 * 3
    const rotationsByteSize = numSplats * 3; // u8 * 3
    const shDim = (shDegree === 0) ? 0 :
        (shDegree === 1) ? 9 :
            (shDegree === 2) ? 24 : 45;
    const shByteSize =  numSplats * shDim;

    const positionsView = new DataView(fileBuffer.buffer, HEADER_SIZE, positionsByteSize);
    const alphasView =    new DataView(fileBuffer.buffer, HEADER_SIZE + positionsByteSize, alphasByteSize);
    const colorsView =    new DataView(fileBuffer.buffer, HEADER_SIZE + positionsByteSize + alphasByteSize, colorsByteSize);
    const scalesView =    new DataView(fileBuffer.buffer, HEADER_SIZE + positionsByteSize + alphasByteSize + colorsByteSize, scalesByteSize);
    const rotationsView = new DataView(fileBuffer.buffer, HEADER_SIZE + positionsByteSize + alphasByteSize + colorsByteSize + scalesByteSize, rotationsByteSize);
    const shView =        new DataView(fileBuffer.buffer, HEADER_SIZE + positionsByteSize + alphasByteSize + colorsByteSize + scalesByteSize + rotationsByteSize, shByteSize);

    // Create columns for the standard Gaussian splat data
    const columns = [
        // Position
        new Column('x', new Float32Array(numSplats)),
        new Column('y', new Float32Array(numSplats)),
        new Column('z', new Float32Array(numSplats)),

        // Scale (stored as linear in .splat, convert to log for internal use)
        new Column('scale_0', new Float32Array(numSplats)),
        new Column('scale_1', new Float32Array(numSplats)),
        new Column('scale_2', new Float32Array(numSplats)),

        // Color/opacity
        new Column('f_dc_0', new Float32Array(numSplats)), // Red
        new Column('f_dc_1', new Float32Array(numSplats)), // Green
        new Column('f_dc_2', new Float32Array(numSplats)), // Blue
        new Column('opacity', new Uint8Array(numSplats)),

        // Rotation quaternion
        new Column('rot_0', new Float32Array(numSplats)),
        new Column('rot_1', new Float32Array(numSplats)),
        new Column('rot_2', new Float32Array(numSplats)),
        new Column('rot_3', new Float32Array(numSplats))

        // TODO: Push spherical Harmonics columns
    ];

    const scale = 1.0 / (1 << fractionalBits);
    for (let splatIndex = 0; splatIndex < numSplats; splatIndex++) {
        // Read position (3 × uint24)
        const x = getFixed24(positionsView, splatIndex, 0) * scale;
        const y = getFixed24(positionsView, splatIndex, 1) * scale;
        const z = getFixed24(positionsView, splatIndex, 2) * scale;

        // Read scale (3 × uint8 log encoded)
        const scaleX = scalesView.getUint8(splatIndex * 3 + 0) / 16.0 - 10.0;
        const scaleY = scalesView.getUint8(splatIndex * 3 + 1) / 16.0 - 10.0;
        const scaleZ = scalesView.getUint8(splatIndex * 3 + 2) / 16.0 - 10.0;

        // Read color and opacity (4 × uint8)
        const red =   colorsView.getUint8(splatIndex * 3 + 0);
        const green = colorsView.getUint8(splatIndex * 3 + 1);
        const blue =  colorsView.getUint8(splatIndex * 3 + 2);
        const opacity = alphasView.getUint8(splatIndex);

        // Read rotation quaternion (4 × uint8)
        const rot1 = rotationsView.getUint8(splatIndex * 3 + 0);
        const rot2 = rotationsView.getUint8(splatIndex * 3 + 1);
        const rot3 = rotationsView.getUint8(splatIndex * 3 + 2);

        // Store position
        (columns[0].data as Float32Array)[splatIndex] = x;
        (columns[1].data as Float32Array)[splatIndex] = y;
        (columns[2].data as Float32Array)[splatIndex] = z;

        // Store scale (No need to apply log since they are already log-encoded)
        (columns[3].data as Float32Array)[splatIndex] = scaleX;
        (columns[4].data as Float32Array)[splatIndex] = scaleY;
        (columns[5].data as Float32Array)[splatIndex] = scaleZ;

        // Store color (convert from uint8 back to spherical harmonics)
        // Colors are already between 0 and 255 but multiplied by SH_C0_2. We need to revert the function to apply the correct SH_C0
        (columns[6].data as Float32Array)[splatIndex] = inverseConvertColorFromSPZ(red);
        (columns[7].data as Float32Array)[splatIndex] = inverseConvertColorFromSPZ(green);
        (columns[8].data as Float32Array)[splatIndex] = inverseConvertColorFromSPZ(blue);

        // Store opacity (convert from uint8 to float and apply inverse sigmoid)
        const epsilon = 1e-6;
        const normalizedOpacity = Math.max(epsilon, Math.min(1.0 - epsilon, opacity / 255.0));
        (columns[9].data as Float32Array)[splatIndex] = opacity / 255.0;

        // Store rotation quaternion (convert from uint8 [0,255] to float [-1,1])
        const rot1Norm = (rot1 / 127.5) - 1.0;
        const rot2Norm = (rot2 / 127.5) - 1.0;
        const rot3Norm = (rot3 / 127.5) - 1.0;
        const rotationDot = rot1Norm * rot1Norm + rot2Norm * rot2Norm + rot3Norm * rot3Norm;
        const rot0Norm = Math.sqrt(Math.max(0.0, 1.0 - rotationDot));

        (columns[10].data as Float32Array)[splatIndex] = rot0Norm;
        (columns[11].data as Float32Array)[splatIndex] = rot1Norm;
        (columns[12].data as Float32Array)[splatIndex] = rot2Norm;
        (columns[13].data as Float32Array)[splatIndex] = rot3Norm;

        // TODO: Spherical Harmonics
    }

    return {
        comments: [],
        elements: [{
            name: 'vertex',
            dataTable: new DataTable(columns)
        }]
    };
};

export { SplatData, readSPZ };
