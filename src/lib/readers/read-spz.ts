import { decompress as decompressZstd } from 'fzstd';

import { Column, DataTable } from '../data-table/data-table';
import { ReadSource } from '../io/read';
import { logger, Transform } from '../utils';

const TICK_BATCH = 1 << 16;

// See https://github.com/nianticlabs/spz for reference implementation

const decompressGZIP = async (data: Uint8Array): Promise<Uint8Array> => {
    // Convert to ArrayBuffer slice to satisfy Blob constructor type requirements
    const arrayBuffer = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer;
    const blob = new Blob([arrayBuffer], { type: 'application/gzip' });
    const ds = new DecompressionStream('gzip');
    const decompressionStream = blob.stream().pipeThrough(ds);
    const decompressed = await new Response(decompressionStream).arrayBuffer();
    return new Uint8Array(decompressed);
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

const HARMONICS_COMPONENT_COUNT = [0, 9, 24, 45, 72];

// Reusable scratch array for smallest-three quaternion decoding (avoids per-splat allocations)
const tmpQuat = [0.0, 0.0, 0.0, 0.0];

/**
 * Reads a .spz file containing Niantic Labs compressed Gaussian splat data.
 *
 * The .spz format uses either GZIP (v2/v3) or per-stream ZSTD (v4) compression and
 * fixed-point encoding to achieve compact file sizes. Supports SH degrees 0-4.
 *
 * @see https://github.com/nianticlabs/spz
 *
 * @param source - The read source providing access to the .spz file data.
 * @returns Promise resolving to a DataTable containing the splat data.
 * @ignore
 */
const readSpz = async (source: ReadSource): Promise<DataTable> => {
    // Load complete file
    let fileBuffer = await source.read().readAll();

    // Check if file is GZip compressed (legacy v2/v3 format)
    const magicSize = 4;
    let magicView = new DataView(fileBuffer.buffer, fileBuffer.byteOffset, magicSize);

    // If file is GZip compressed, decompress it first.
    if (magicView.getUint16(0) === 0x1F8B) { // '1F 8B' is the magic for gzip
        fileBuffer = await decompressGZIP(fileBuffer);
        magicView = new DataView(fileBuffer.buffer, fileBuffer.byteOffset, magicSize);
    }

    const magic = magicView.getUint32(0, true);
    if (magic !== 0x5053474e) { // NGSP
        throw new Error('invalid file header');
    }

    const totalSize = fileBuffer.length;

    // v2/v3: 16-byte header, single gzip stream (already decompressed above)
    // v4 and later:    32-byte plaintext header with ZSTD streams
    const MIN_SPZ_VERSION = 2;
    const LATEST_SPZ_VERSION = 4;
    const SPZ_VERSION_INTRODUCED_ZSTD = 4;
    const MIN_HEADER_SIZE = 16;

    if (totalSize < MIN_HEADER_SIZE) {
        throw new Error('File too small to be valid .spz format');
    }
    const version = new DataView(fileBuffer.buffer, fileBuffer.byteOffset, MIN_HEADER_SIZE).getUint32(4, true);
    if (version < MIN_SPZ_VERSION || version > LATEST_SPZ_VERSION) {
        throw new Error(`Unsupported version ${version}`);
    }

    const HEADER_SIZE = version >= SPZ_VERSION_INTRODUCED_ZSTD ? 32 : MIN_HEADER_SIZE;
    if (totalSize < HEADER_SIZE) {
        throw new Error('File too small to be valid .spz format');
    }

    const header = new DataView(fileBuffer.buffer, fileBuffer.byteOffset, HEADER_SIZE);
    const numSplats = header.getUint32(8, true);
    const shDegree = header.getUint8(12);
    const fractionalBits = header.getUint8(13);
    if (shDegree >= HARMONICS_COMPONENT_COUNT.length) {
        throw new Error(`Unsupported SH degree ${shDegree}`);
    }

    const harmonicsComponentCount = HARMONICS_COMPONENT_COUNT[shDegree];

    // Byte sizes of each attribute block
    const positionsByteSize = numSplats * 9;                      // 3 × uint24
    const alphasByteSize = numSplats;                             // u8
    const colorsByteSize = numSplats * 3;                        // u8 × 3
    const scalesByteSize = numSplats * 3;                        // u8 × 3
    const rotationsByteSize = numSplats * (version === MIN_SPZ_VERSION ? 3 : 4);
    const shByteSize = numSplats * harmonicsComponentCount;

    let positionsView: DataView;
    let alphasView: DataView;
    let colorsView: DataView;
    let scalesView: DataView;
    let rotationsView: DataView;
    let shView: DataView;

    if (version >= SPZ_VERSION_INTRODUCED_ZSTD && version <= LATEST_SPZ_VERSION) {
        // 32-byte plaintext header, followed by optional extensions, then TOC, then ZSTD streams.
        const numStreams = header.getUint8(15);
        const tocByteOffset = header.getUint32(16, true);

        const expectedNumStreams = shDegree === 0 ? 5 : 6;
        if (numStreams !== expectedNumStreams) {
            throw new Error(`Unexpected stream count for shDegree ${shDegree}: expected ${expectedNumStreams}, got ${numStreams}`);
        }

        if (tocByteOffset < HEADER_SIZE) {
            throw new Error(`Invalid tocByteOffset ${tocByteOffset}: must be >= ${HEADER_SIZE}`);
        }

        const tocSize = numStreams * 16;
        const dataStart = tocByteOffset + tocSize;
        if (dataStart > totalSize) {
            throw new Error('File too small: TOC exceeds file bounds');
        }

        const expectedSizes = [positionsByteSize, alphasByteSize, colorsByteSize, scalesByteSize, rotationsByteSize];
        if (shDegree > 0) {
            expectedSizes.push(shByteSize);
        }

        const streams: DataView[] = [];
        let dataOffset = dataStart;
        for (let i = 0; i < numStreams; i++) {
            const tocBase = tocByteOffset + i * 16;
            const tocView = new DataView(fileBuffer.buffer, fileBuffer.byteOffset + tocBase, 16);
            const compressedSizeBig = tocView.getBigUint64(0, true);
            const uncompressedSizeBig = tocView.getBigUint64(8, true);
            if (compressedSizeBig > BigInt(Number.MAX_SAFE_INTEGER) || uncompressedSizeBig > BigInt(Number.MAX_SAFE_INTEGER)) {
                throw new Error(`Stream ${i} size exceeds Number.MAX_SAFE_INTEGER`);
            }
            const compressedSize = Number(compressedSizeBig);
            const uncompressedSize = Number(uncompressedSizeBig);

            if (uncompressedSize !== expectedSizes[i]) {
                throw new Error(`Stream ${i} uncompressed size mismatch: expected ${expectedSizes[i]}, got ${uncompressedSize}`);
            }
            if (dataOffset + compressedSize > totalSize) {
                throw new Error(`Stream ${i} extends past end of file (offset ${dataOffset} + ${compressedSize} > ${totalSize})`);
            }

            const compressed = fileBuffer.subarray(dataOffset, dataOffset + compressedSize);
            const decompressed = decompressZstd(compressed);
            if (decompressed.byteLength !== uncompressedSize) {
                throw new Error(`Stream ${i} decompressed size mismatch: expected ${uncompressedSize}, got ${decompressed.byteLength}`);
            }
            streams.push(new DataView(decompressed.buffer, decompressed.byteOffset, decompressed.byteLength));
            dataOffset += compressedSize;
        }

        positionsView = streams[0];
        alphasView = streams[1];
        colorsView = streams[2];
        scalesView = streams[3];
        rotationsView = streams[4];
        shView = numStreams > 5 ? streams[5] : new DataView(new ArrayBuffer(0));
    } else {
        // v2/v3: single buffer already decompressed from gzip, attributes laid out sequentially.
        const requiredSize = HEADER_SIZE + positionsByteSize + alphasByteSize + colorsByteSize + scalesByteSize + rotationsByteSize + shByteSize;
        if (totalSize < requiredSize) {
            throw new Error(`File too small for SPZ payload: expected at least ${requiredSize} bytes, got ${totalSize}`);
        }

        let offset = fileBuffer.byteOffset + HEADER_SIZE;
        positionsView = new DataView(fileBuffer.buffer, offset, positionsByteSize); offset += positionsByteSize;
        alphasView = new DataView(fileBuffer.buffer, offset, alphasByteSize); offset += alphasByteSize;
        colorsView = new DataView(fileBuffer.buffer, offset, colorsByteSize); offset += colorsByteSize;
        scalesView = new DataView(fileBuffer.buffer, offset, scalesByteSize); offset += scalesByteSize;
        rotationsView = new DataView(fileBuffer.buffer, offset, rotationsByteSize); offset += rotationsByteSize;
        shView = new DataView(fileBuffer.buffer, offset, shByteSize);
    }

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
        new Column('opacity', new Float32Array(numSplats)),

        // Rotation quaternion
        new Column('rot_0', new Float32Array(numSplats)),
        new Column('rot_1', new Float32Array(numSplats)),
        new Column('rot_2', new Float32Array(numSplats)),
        new Column('rot_3', new Float32Array(numSplats))
    ];

    // Add spherical harmonics columns based on maximum degree found
    for (let i = 0; i < harmonicsComponentCount; i++) {
        columns.push(new Column(`f_rest_${i}`, new Float32Array(numSplats)));
    }

    const bar = logger.bar('decoding', numSplats);

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
        const red = colorsView.getUint8(splatIndex * 3 + 0);
        const green = colorsView.getUint8(splatIndex * 3 + 1);
        const blue = colorsView.getUint8(splatIndex * 3 + 2);
        const opacity = alphasView.getUint8(splatIndex);

        let rot0Norm = 0.0;
        let rot1Norm = 0.0;
        let rot2Norm = 0.0;
        let rot3Norm = 0.0;
        if (version === MIN_SPZ_VERSION) {
            rot1Norm = (rotationsView.getUint8(splatIndex * 3 + 0) / 127.5) - 1.0;
            rot2Norm = (rotationsView.getUint8(splatIndex * 3 + 1) / 127.5) - 1.0;
            rot3Norm = (rotationsView.getUint8(splatIndex * 3 + 2) / 127.5) - 1.0;
            const dot = rot1Norm * rot1Norm + rot2Norm * rot2Norm + rot3Norm * rot3Norm;
            rot0Norm = Math.sqrt(Math.max(0.0, 1.0 - dot));
        } else if (version >= 3 && version <= LATEST_SPZ_VERSION) {
            // Smallest-three quaternion decode from packed uint32
            // SPZ stores as [x, y, z, w] (indices 0-3)
            tmpQuat[0] = tmpQuat[1] = tmpQuat[2] = tmpQuat[3] = 0.0;
            let packed = rotationsView.getUint32(splatIndex * 4, true);
            const cMask = (1 << 9) - 1;
            const largest = packed >>> 30;
            let sumSq = 0;
            for (let i = 3; i >= 0; --i) {
                if (i !== largest) {
                    const mag = packed & cMask;
                    const neg = (packed >>> 9) & 1;
                    packed >>>= 10;
                    tmpQuat[i] = Math.SQRT1_2 * mag / cMask;
                    if (neg === 1) tmpQuat[i] = -tmpQuat[i];
                    sumSq += tmpQuat[i] * tmpQuat[i];
                }
            }
            tmpQuat[largest] = Math.sqrt(Math.max(0.0, 1.0 - sumSq));
            rot0Norm = tmpQuat[3]; // w
            rot1Norm = tmpQuat[0]; // x
            rot2Norm = tmpQuat[1]; // y
            rot3Norm = tmpQuat[2]; // z
        }

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
        (columns[9].data as Float32Array)[splatIndex] = Math.log(normalizedOpacity / (1.0 - normalizedOpacity));

        (columns[10].data as Float32Array)[splatIndex] = rot0Norm;
        (columns[11].data as Float32Array)[splatIndex] = rot1Norm;
        (columns[12].data as Float32Array)[splatIndex] = rot2Norm;
        (columns[13].data as Float32Array)[splatIndex] = rot3Norm;

        // Store spherical harmonics
        for (let i = 0; i < harmonicsComponentCount; i++) {
            const channel = i % 3;
            const coeff = Math.floor(i / 3);
            const col = channel * (harmonicsComponentCount / 3) + coeff;
            const shCoef = shView.getUint8(splatIndex * harmonicsComponentCount + i);
            (columns[14 + col].data as Float32Array)[splatIndex] = (shCoef - 128) / 128;
        }

        if ((splatIndex & (TICK_BATCH - 1)) === (TICK_BATCH - 1)) {
            bar.update(splatIndex + 1);
        }
    }

    bar.update(numSplats);
    bar.end();

    return new DataTable(columns, Transform.PLY);
};

export { readSpz };
