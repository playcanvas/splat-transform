import { Buffer } from 'node:buffer';
import { FileHandle } from 'node:fs/promises';

import { Column, DataTable } from '../data-table/data-table';

// Format configuration for different compression modes
interface CompressionConfig {
    centerBytes: number;
    scaleBytes: number;
    rotationBytes: number;
    colorBytes: number;
    harmonicsBytes: number;
    scaleStartByte: number;
    rotationStartByte: number;
    colorStartByte: number;
    harmonicsStartByte: number;
    scaleQuantRange: number;
}

// Half-precision floating point decoder
function decodeFloat16(encoded: number): number {
    const signBit = (encoded >> 15) & 1;
    const exponent = (encoded >> 10) & 0x1f;
    const mantissa = encoded & 0x3ff;

    if (exponent === 0) {
        if (mantissa === 0) {
            return signBit ? -0.0 : 0.0;
        }
        // Denormalized number
        let m = mantissa;
        let exp = -14;
        while (!(m & 0x400)) {
            m <<= 1;
            exp--;
        }
        m &= 0x3ff;
        const finalExp = exp + 127;
        const finalMantissa = m << 13;
        const bits = (signBit << 31) | (finalExp << 23) | finalMantissa;
        return new Float32Array(new Uint32Array([bits]).buffer)[0];
    }

    if (exponent === 0x1f) {
        return mantissa === 0 ? (signBit ? -Infinity : Infinity) : NaN;
    }

    const finalExp = exponent - 15 + 127;
    const finalMantissa = mantissa << 13;
    const bits = (signBit << 31) | (finalExp << 23) | finalMantissa;
    return new Float32Array(new Uint32Array([bits]).buffer)[0];
}

const COMPRESSION_MODES: CompressionConfig[] = [
    {
        centerBytes: 12,
        scaleBytes: 12,
        rotationBytes: 16,
        colorBytes: 4,
        harmonicsBytes: 4,
        scaleStartByte: 12,
        rotationStartByte: 24,
        colorStartByte: 40,
        harmonicsStartByte: 44,
        scaleQuantRange: 1
    },
    {
        centerBytes: 6,
        scaleBytes: 6,
        rotationBytes: 8,
        colorBytes: 4,
        harmonicsBytes: 2,
        scaleStartByte: 6,
        rotationStartByte: 12,
        colorStartByte: 20,
        harmonicsStartByte: 24,
        scaleQuantRange: 32767
    },
    {
        centerBytes: 6,
        scaleBytes: 6,
        rotationBytes: 8,
        colorBytes: 4,
        harmonicsBytes: 1,
        scaleStartByte: 6,
        rotationStartByte: 12,
        colorStartByte: 20,
        harmonicsStartByte: 24,
        scaleQuantRange: 32767
    }
];

const HARMONICS_COMPONENT_COUNT = [0, 9, 24, 45];

const readKsplat = async (fileHandle: FileHandle): Promise<DataTable> => {
    const stats = await fileHandle.stat();
    const totalSize = stats.size;

    // Load complete file
    const fileBuffer = Buffer.alloc(totalSize);
    await fileHandle.read(fileBuffer, 0, totalSize, 0);

    const MAIN_HEADER_SIZE = 4096;
    const SECTION_HEADER_SIZE = 1024;

    if (totalSize < MAIN_HEADER_SIZE) {
        throw new Error('File too small to be valid .ksplat format');
    }

    // Parse main header
    const mainHeader = new DataView(fileBuffer.buffer, fileBuffer.byteOffset, MAIN_HEADER_SIZE);

    const majorVersion = mainHeader.getUint8(0);
    const minorVersion = mainHeader.getUint8(1);
    if (majorVersion !== 0 || minorVersion < 1) {
        throw new Error(`Unsupported version ${majorVersion}.${minorVersion}`);
    }

    const maxSections = mainHeader.getUint32(4, true);
    const numSplats = mainHeader.getUint32(16, true);
    const compressionMode = mainHeader.getUint16(20, true);

    if (compressionMode > 2) {
        throw new Error(`Invalid compression mode: ${compressionMode}`);
    }

    const minHarmonicsValue = mainHeader.getFloat32(36, true) || -1.5;
    const maxHarmonicsValue = mainHeader.getFloat32(40, true) || 1.5;

    if (numSplats === 0) {
        throw new Error('Invalid .ksplat file: file is empty');
    }

    // First pass: scan all sections to find maximum harmonics degree
    let maxHarmonicsDegree = 0;
    for (let sectionIdx = 0; sectionIdx < maxSections; sectionIdx++) {
        const sectionHeaderOffset = MAIN_HEADER_SIZE + sectionIdx * SECTION_HEADER_SIZE;
        const sectionHeader = new DataView(fileBuffer.buffer, fileBuffer.byteOffset + sectionHeaderOffset, SECTION_HEADER_SIZE);

        const sectionSplatCount = sectionHeader.getUint32(0, true);
        if (sectionSplatCount === 0) continue; // Skip empty sections

        const harmonicsDegree = sectionHeader.getUint16(40, true);
        maxHarmonicsDegree = Math.max(maxHarmonicsDegree, harmonicsDegree);
    }

    // Initialize data storage with base columns
    const columns: Column[] = [
        new Column('x', new Float32Array(numSplats)),
        new Column('y', new Float32Array(numSplats)),
        new Column('z', new Float32Array(numSplats)),
        new Column('scale_0', new Float32Array(numSplats)),
        new Column('scale_1', new Float32Array(numSplats)),
        new Column('scale_2', new Float32Array(numSplats)),
        new Column('f_dc_0', new Float32Array(numSplats)),
        new Column('f_dc_1', new Float32Array(numSplats)),
        new Column('f_dc_2', new Float32Array(numSplats)),
        new Column('opacity', new Float32Array(numSplats)),
        new Column('rot_0', new Float32Array(numSplats)),
        new Column('rot_1', new Float32Array(numSplats)),
        new Column('rot_2', new Float32Array(numSplats)),
        new Column('rot_3', new Float32Array(numSplats))
    ];

    // Add spherical harmonics columns based on maximum degree found
    const maxHarmonicsComponentCount = HARMONICS_COMPONENT_COUNT[maxHarmonicsDegree];
    for (let i = 0; i < maxHarmonicsComponentCount; i++) {
        columns.push(new Column(`f_rest_${i}`, new Float32Array(numSplats)));
    }

    const {
        centerBytes,
        scaleBytes,
        rotationBytes,
        colorBytes,
        harmonicsBytes,
        scaleStartByte,
        rotationStartByte,
        colorStartByte,
        harmonicsStartByte,
        scaleQuantRange
    } = COMPRESSION_MODES[compressionMode];

    let currentSectionDataOffset = MAIN_HEADER_SIZE + maxSections * SECTION_HEADER_SIZE;
    let splatIndex = 0;

    // Process each section
    for (let sectionIdx = 0; sectionIdx < maxSections; sectionIdx++) {
        const sectionHeaderOffset = MAIN_HEADER_SIZE + sectionIdx * SECTION_HEADER_SIZE;
        const sectionHeader = new DataView(fileBuffer.buffer, fileBuffer.byteOffset + sectionHeaderOffset, SECTION_HEADER_SIZE);

        const sectionSplatCount = sectionHeader.getUint32(0, true);
        const maxSectionSplats = sectionHeader.getUint32(4, true);
        const bucketCapacity = sectionHeader.getUint32(8, true);
        const bucketCount = sectionHeader.getUint32(12, true);
        const spatialBlockSize = sectionHeader.getFloat32(16, true);
        const bucketStorageSize = sectionHeader.getUint16(20, true);
        const quantizationRange = sectionHeader.getUint32(24, true) || scaleQuantRange;
        const fullBuckets = sectionHeader.getUint32(32, true);
        const partialBuckets = sectionHeader.getUint32(36, true);
        const harmonicsDegree = sectionHeader.getUint16(40, true);

        // Calculate layout
        const fullBucketSplats = fullBuckets * bucketCapacity;
        const partialBucketMetaSize = partialBuckets * 4;
        const totalBucketStorageSize = bucketStorageSize * bucketCount + partialBucketMetaSize;
        const harmonicsComponentCount = HARMONICS_COMPONENT_COUNT[harmonicsDegree];
        const bytesPerSplat = centerBytes + scaleBytes + rotationBytes +
                             colorBytes + harmonicsComponentCount * harmonicsBytes;
        const sectionDataSize = bytesPerSplat * maxSectionSplats;

        // Calculate decompression parameters
        const positionScale = spatialBlockSize / 2.0 / quantizationRange;

        // Get bucket centers
        const bucketCentersOffset = currentSectionDataOffset + partialBucketMetaSize;
        const bucketCenters = new Float32Array(fileBuffer.buffer, fileBuffer.byteOffset + bucketCentersOffset, bucketCount * 3);

        // Get partial bucket sizes
        const partialBucketSizes = new Uint32Array(fileBuffer.buffer, fileBuffer.byteOffset + currentSectionDataOffset, partialBuckets);

        // Get splat data
        const splatDataOffset = currentSectionDataOffset + totalBucketStorageSize;
        const splatData = new DataView(fileBuffer.buffer, fileBuffer.byteOffset + splatDataOffset, sectionDataSize);

        // Harmonic value decoder
        const decodeHarmonics = (offset: number, component: number): number => {
            switch (compressionMode) {
                case 0:
                    return splatData.getFloat32(offset + harmonicsStartByte + component * 4, true);
                case 1:
                    return decodeFloat16(splatData.getUint16(offset + harmonicsStartByte + component * 2, true));
                case 2: {
                    const normalized = splatData.getUint8(offset + harmonicsStartByte + component) / 255;
                    return minHarmonicsValue + normalized * (maxHarmonicsValue - minHarmonicsValue);
                }
                default:
                    return 0;
            }
        };

        // Track partial bucket processing
        let currentPartialBucket = fullBuckets;
        let currentPartialBase = fullBucketSplats;

        // Process splats in this section
        for (let splatIdx = 0; splatIdx < sectionSplatCount; splatIdx++) {
            const splatByteOffset = splatIdx * bytesPerSplat;

            // Determine which bucket this splat belongs to
            let bucketIdx: number;
            if (splatIdx < fullBucketSplats) {
                bucketIdx = Math.floor(splatIdx / bucketCapacity);
            } else {
                const currentBucketSize = partialBucketSizes[currentPartialBucket - fullBuckets];
                if (splatIdx >= currentPartialBase + currentBucketSize) {
                    currentPartialBucket++;
                    currentPartialBase += currentBucketSize;
                }
                bucketIdx = currentPartialBucket;
            }

            // Decode position
            let x: number, y: number, z: number;
            if (compressionMode === 0) {
                x = splatData.getFloat32(splatByteOffset, true);
                y = splatData.getFloat32(splatByteOffset + 4, true);
                z = splatData.getFloat32(splatByteOffset + 8, true);
            } else {
                x = (splatData.getUint16(splatByteOffset, true) - quantizationRange) * positionScale + bucketCenters[bucketIdx * 3];
                y = (splatData.getUint16(splatByteOffset + 2, true) - quantizationRange) * positionScale + bucketCenters[bucketIdx * 3 + 1];
                z = (splatData.getUint16(splatByteOffset + 4, true) - quantizationRange) * positionScale + bucketCenters[bucketIdx * 3 + 2];
            }

            // Decode scales
            let scaleX: number, scaleY: number, scaleZ: number;
            if (compressionMode === 0) {
                scaleX = splatData.getFloat32(splatByteOffset + scaleStartByte, true);
                scaleY = splatData.getFloat32(splatByteOffset + scaleStartByte + 4, true);
                scaleZ = splatData.getFloat32(splatByteOffset + scaleStartByte + 8, true);
            } else {
                scaleX = decodeFloat16(splatData.getUint16(splatByteOffset + scaleStartByte, true));
                scaleY = decodeFloat16(splatData.getUint16(splatByteOffset + scaleStartByte + 2, true));
                scaleZ = decodeFloat16(splatData.getUint16(splatByteOffset + scaleStartByte + 4, true));
            }

            // Decode rotation quaternion
            let rot0: number, rot1: number, rot2: number, rot3: number;
            if (compressionMode === 0) {
                rot0 = splatData.getFloat32(splatByteOffset + rotationStartByte, true);
                rot1 = splatData.getFloat32(splatByteOffset + rotationStartByte + 4, true);
                rot2 = splatData.getFloat32(splatByteOffset + rotationStartByte + 8, true);
                rot3 = splatData.getFloat32(splatByteOffset + rotationStartByte + 12, true);
            } else {
                rot0 = decodeFloat16(splatData.getUint16(splatByteOffset + rotationStartByte, true));
                rot1 = decodeFloat16(splatData.getUint16(splatByteOffset + rotationStartByte + 2, true));
                rot2 = decodeFloat16(splatData.getUint16(splatByteOffset + rotationStartByte + 4, true));
                rot3 = decodeFloat16(splatData.getUint16(splatByteOffset + rotationStartByte + 6, true));
            }

            // Decode color and opacity
            const red = splatData.getUint8(splatByteOffset + colorStartByte);
            const green = splatData.getUint8(splatByteOffset + colorStartByte + 1);
            const blue = splatData.getUint8(splatByteOffset + colorStartByte + 2);
            const opacity = splatData.getUint8(splatByteOffset + colorStartByte + 3);

            // Store position
            (columns[0].data as Float32Array)[splatIndex] = x;
            (columns[1].data as Float32Array)[splatIndex] = y;
            (columns[2].data as Float32Array)[splatIndex] = z;

            // Store scale (convert from linear in .ksplat to log scale for internal use)
            (columns[3].data as Float32Array)[splatIndex] = scaleX > 0 ? Math.log(scaleX) : -10;
            (columns[4].data as Float32Array)[splatIndex] = scaleY > 0 ? Math.log(scaleY) : -10;
            (columns[5].data as Float32Array)[splatIndex] = scaleZ > 0 ? Math.log(scaleZ) : -10;

            // Store color (convert from uint8 back to spherical harmonics)
            const SH_C0 = 0.28209479177387814;
            (columns[6].data as Float32Array)[splatIndex] = (red / 255.0 - 0.5) / SH_C0;
            (columns[7].data as Float32Array)[splatIndex] = (green / 255.0 - 0.5) / SH_C0;
            (columns[8].data as Float32Array)[splatIndex] = (blue / 255.0 - 0.5) / SH_C0;

            // Store opacity (convert from uint8 to float and apply inverse sigmoid)
            const epsilon = 1e-6;
            const normalizedOpacity = Math.max(epsilon, Math.min(1.0 - epsilon, opacity / 255.0));
            (columns[9].data as Float32Array)[splatIndex] = Math.log(normalizedOpacity / (1.0 - normalizedOpacity));

            // Store quaternion
            (columns[10].data as Float32Array)[splatIndex] = rot0;
            (columns[11].data as Float32Array)[splatIndex] = rot1;
            (columns[12].data as Float32Array)[splatIndex] = rot2;
            (columns[13].data as Float32Array)[splatIndex] = rot3;

            // Store spherical harmonics
            for (let i = 0; i < harmonicsComponentCount; i++) {
                let channel;
                let coeff;

                // band 0 is packed together, then band 1, then band 2.
                if (i < 9) {
                    channel = Math.floor(i / 3);
                    coeff = i % 3;
                } else if (i < 24) {
                    channel = Math.floor((i - 9) / 5);
                    coeff = (i - 9) % 5 + 3;
                } else {
                    // don't think 3 bands are supported, but here just in case
                    channel = Math.floor((i - 24) / 7);
                    coeff = (i - 24) % 7 + 8;
                }

                const col = channel * (harmonicsComponentCount / 3) + coeff;

                (columns[14 + col].data as Float32Array)[splatIndex] = decodeHarmonics(splatByteOffset, i);
            }

            splatIndex++;
        }

        currentSectionDataOffset += sectionDataSize + totalBucketStorageSize;
    }

    if (splatIndex !== numSplats) {
        throw new Error(`Splat count mismatch: expected ${numSplats}, processed ${splatIndex}`);
    }

    return new DataTable(columns);
};

export { readKsplat };
