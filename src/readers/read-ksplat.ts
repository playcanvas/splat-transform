import { Buffer } from 'node:buffer';
import { FileHandle } from 'node:fs/promises';

import { Column, DataTable } from '../data-table';

type KsplatFileData = {
    comments: string[];
    elements: {
        name: string,
        dataTable: DataTable
    }[];
};

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

const readKsplat = async (fileHandle: FileHandle): Promise<KsplatFileData> => {
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
    const totalSplats = mainHeader.getUint32(16, true);
    const compressionMode = mainHeader.getUint16(20, true);

    if (compressionMode > 2) {
        throw new Error(`Invalid compression mode: ${compressionMode}`);
    }

    const minHarmonicsValue = mainHeader.getFloat32(36, true) || -1.5;
    const maxHarmonicsValue = mainHeader.getFloat32(40, true) || 1.5;

    if (totalSplats === 0) {
        throw new Error('No splats found in file');
    }

    // Initialize data storage
    const dataColumns: Column[] = [
        new Column('x', new Float32Array(totalSplats)),
        new Column('y', new Float32Array(totalSplats)),
        new Column('z', new Float32Array(totalSplats)),
        new Column('scale_0', new Float32Array(totalSplats)),
        new Column('scale_1', new Float32Array(totalSplats)),
        new Column('scale_2', new Float32Array(totalSplats)),
        new Column('f_dc_0', new Float32Array(totalSplats)),
        new Column('f_dc_1', new Float32Array(totalSplats)),
        new Column('f_dc_2', new Float32Array(totalSplats)),
        new Column('opacity', new Float32Array(totalSplats)),
        new Column('rot_0', new Float32Array(totalSplats)),
        new Column('rot_1', new Float32Array(totalSplats)),
        new Column('rot_2', new Float32Array(totalSplats)),
        new Column('rot_3', new Float32Array(totalSplats))
    ];

    const config = COMPRESSION_MODES[compressionMode];
    let currentSectionDataOffset = MAIN_HEADER_SIZE + maxSections * SECTION_HEADER_SIZE;
    let globalSplatIndex = 0;

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
        const quantizationRange = sectionHeader.getUint32(24, true) || config.scaleQuantRange;
        const fullBuckets = sectionHeader.getUint32(32, true);
        const partialBuckets = sectionHeader.getUint32(36, true);
        const harmonicsDegree = sectionHeader.getUint16(40, true);

        // Calculate layout
        const fullBucketSplats = fullBuckets * bucketCapacity;
        const partialBucketMetaSize = partialBuckets * 4;
        const totalBucketStorageSize = bucketStorageSize * bucketCount + partialBucketMetaSize;
        const harmonicsComponentCount = HARMONICS_COMPONENT_COUNT[harmonicsDegree];
        const bytesPerSplat = config.centerBytes + config.scaleBytes + config.rotationBytes +
                             config.colorBytes + harmonicsComponentCount * config.harmonicsBytes;
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
                     return splatData.getFloat32(offset + config.harmonicsStartByte + component * 4, true);
                 case 1:
                     return decodeFloat16(splatData.getUint16(offset + config.harmonicsStartByte + component * 2, true));
                 case 2: {
                     const normalized = splatData.getUint8(offset + config.harmonicsStartByte + component) / 255;
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
            let posX: number, posY: number, posZ: number;
            if (compressionMode === 0) {
                posX = splatData.getFloat32(splatByteOffset, true);
                posY = splatData.getFloat32(splatByteOffset + 4, true);
                posZ = splatData.getFloat32(splatByteOffset + 8, true);
            } else {
                posX = (splatData.getUint16(splatByteOffset, true) - quantizationRange) * positionScale + bucketCenters[bucketIdx * 3];
                posY = (splatData.getUint16(splatByteOffset + 2, true) - quantizationRange) * positionScale + bucketCenters[bucketIdx * 3 + 1];
                posZ = (splatData.getUint16(splatByteOffset + 4, true) - quantizationRange) * positionScale + bucketCenters[bucketIdx * 3 + 2];
            }

            // Decode scales
            let scaleX: number, scaleY: number, scaleZ: number;
            if (compressionMode === 0) {
                scaleX = splatData.getFloat32(splatByteOffset + config.scaleStartByte, true);
                scaleY = splatData.getFloat32(splatByteOffset + config.scaleStartByte + 4, true);
                scaleZ = splatData.getFloat32(splatByteOffset + config.scaleStartByte + 8, true);
            } else {
                scaleX = decodeFloat16(splatData.getUint16(splatByteOffset + config.scaleStartByte, true));
                scaleY = decodeFloat16(splatData.getUint16(splatByteOffset + config.scaleStartByte + 2, true));
                scaleZ = decodeFloat16(splatData.getUint16(splatByteOffset + config.scaleStartByte + 4, true));
            }

            // Decode rotation quaternion
            let rotW: number, rotX: number, rotY: number, rotZ: number;
            if (compressionMode === 0) {
                rotW = splatData.getFloat32(splatByteOffset + config.rotationStartByte, true);
                rotX = splatData.getFloat32(splatByteOffset + config.rotationStartByte + 4, true);
                rotY = splatData.getFloat32(splatByteOffset + config.rotationStartByte + 8, true);
                rotZ = splatData.getFloat32(splatByteOffset + config.rotationStartByte + 12, true);
            } else {
                rotW = decodeFloat16(splatData.getUint16(splatByteOffset + config.rotationStartByte, true));
                rotX = decodeFloat16(splatData.getUint16(splatByteOffset + config.rotationStartByte + 2, true));
                rotY = decodeFloat16(splatData.getUint16(splatByteOffset + config.rotationStartByte + 4, true));
                rotZ = decodeFloat16(splatData.getUint16(splatByteOffset + config.rotationStartByte + 6, true));
            }

            // Decode color and opacity
            const colorR = splatData.getUint8(splatByteOffset + config.colorStartByte) / 255;
            const colorG = splatData.getUint8(splatByteOffset + config.colorStartByte + 1) / 255;
            const colorB = splatData.getUint8(splatByteOffset + config.colorStartByte + 2) / 255;
            const alpha = splatData.getUint8(splatByteOffset + config.colorStartByte + 3) / 255;

            // Store position
            (dataColumns[0].data as Float32Array)[globalSplatIndex] = posX;
            (dataColumns[1].data as Float32Array)[globalSplatIndex] = posY;
            (dataColumns[2].data as Float32Array)[globalSplatIndex] = posZ;

            // Store scale as logarithmic values
            (dataColumns[3].data as Float32Array)[globalSplatIndex] = Math.log(scaleX);
            (dataColumns[4].data as Float32Array)[globalSplatIndex] = Math.log(scaleY);
            (dataColumns[5].data as Float32Array)[globalSplatIndex] = Math.log(scaleZ);

            // Convert color to spherical harmonics coefficients
            const SH_NORMALIZATION = 0.28209479177387814;
            (dataColumns[6].data as Float32Array)[globalSplatIndex] = (colorR - 0.5) / SH_NORMALIZATION;
            (dataColumns[7].data as Float32Array)[globalSplatIndex] = (colorG - 0.5) / SH_NORMALIZATION;
            (dataColumns[8].data as Float32Array)[globalSplatIndex] = (colorB - 0.5) / SH_NORMALIZATION;

            // Convert opacity to sigmoid domain
            const EPSILON = 1e-6;
            const clampedAlpha = Math.max(EPSILON, Math.min(1.0 - EPSILON, alpha));
            (dataColumns[9].data as Float32Array)[globalSplatIndex] = -Math.log(1.0 / clampedAlpha - 1.0);

            // Store quaternion
            (dataColumns[10].data as Float32Array)[globalSplatIndex] = rotX;
            (dataColumns[11].data as Float32Array)[globalSplatIndex] = rotY;
            (dataColumns[12].data as Float32Array)[globalSplatIndex] = rotZ;
            (dataColumns[13].data as Float32Array)[globalSplatIndex] = rotW;

            globalSplatIndex++;
        }

        currentSectionDataOffset += sectionDataSize + totalBucketStorageSize;
    }

    if (globalSplatIndex !== totalSplats) {
        throw new Error(`Splat count mismatch: expected ${totalSplats}, processed ${globalSplatIndex}`);
    }

    const resultTable = new DataTable(dataColumns);

    return {
        comments: [`Loaded ${totalSplats} splats from .ksplat format`],
        elements: [{
            name: 'vertex',
            dataTable: resultTable
        }]
    };
};

export { readKsplat };
