import { Buffer } from 'node:buffer';
import { FileHandle } from 'node:fs/promises';

import { Column, DataTable } from '../data-table';

type SplatData = {
    comments: string[];
    elements: {
        name: string,
        dataTable: DataTable
    }[];
};

const readSplat = async (fileHandle: FileHandle): Promise<SplatData> => {
    // Get file size to determine number of splats
    const fileStats = await fileHandle.stat();
    const fileSize = fileStats.size;

    // Each splat is 32 bytes
    const BYTES_PER_SPLAT = 32;
    if (fileSize % BYTES_PER_SPLAT !== 0) {
        throw new Error('Invalid .splat file: file size is not a multiple of 32 bytes');
    }

    const numSplats = fileSize / BYTES_PER_SPLAT;

    if (numSplats === 0) {
        throw new Error('Invalid .splat file: file is empty');
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

    // Read data in chunks
    const chunkSize = 1024;
    const numChunks = Math.ceil(numSplats / chunkSize);
    const chunkData = Buffer.alloc(chunkSize * BYTES_PER_SPLAT);

    for (let c = 0; c < numChunks; ++c) {
        const numRows = Math.min(chunkSize, numSplats - c * chunkSize);
        const bytesToRead = numRows * BYTES_PER_SPLAT;

        const { bytesRead } = await fileHandle.read(chunkData, 0, bytesToRead);
        if (bytesRead !== bytesToRead) {
            throw new Error('Failed to read expected amount of data from .splat file');
        }

        // Parse each splat in the chunk
        for (let r = 0; r < numRows; ++r) {
            const splatIndex = c * chunkSize + r;
            const offset = r * BYTES_PER_SPLAT;

            // Read position (3 × float32)
            const x = chunkData.readFloatLE(offset + 0);
            const y = chunkData.readFloatLE(offset + 4);
            const z = chunkData.readFloatLE(offset + 8);

            // Read scale (3 × float32)
            const scaleX = chunkData.readFloatLE(offset + 12);
            const scaleY = chunkData.readFloatLE(offset + 16);
            const scaleZ = chunkData.readFloatLE(offset + 20);

            // Read color and opacity (4 × uint8)
            const red = chunkData.readUInt8(offset + 24);
            const green = chunkData.readUInt8(offset + 25);
            const blue = chunkData.readUInt8(offset + 26);
            const opacity = chunkData.readUInt8(offset + 27);

            // Read rotation quaternion (4 × uint8)
            const rot0 = chunkData.readUInt8(offset + 28);
            const rot1 = chunkData.readUInt8(offset + 29);
            const rot2 = chunkData.readUInt8(offset + 30);
            const rot3 = chunkData.readUInt8(offset + 31);

            // Store position
            (columns[0].data as Float32Array)[splatIndex] = x;
            (columns[1].data as Float32Array)[splatIndex] = y;
            (columns[2].data as Float32Array)[splatIndex] = z;

            // Store scale (convert from linear in .splat to log scale for internal use)
            (columns[3].data as Float32Array)[splatIndex] = Math.log(scaleX);
            (columns[4].data as Float32Array)[splatIndex] = Math.log(scaleY);
            (columns[5].data as Float32Array)[splatIndex] = Math.log(scaleZ);

            // Store color (convert from uint8 back to spherical harmonics)
            const SH_C0 = 0.28209479177387814;
            (columns[6].data as Float32Array)[splatIndex] = (red / 255.0 - 0.5) / SH_C0;
            (columns[7].data as Float32Array)[splatIndex] = (green / 255.0 - 0.5) / SH_C0;
            (columns[8].data as Float32Array)[splatIndex] = (blue / 255.0 - 0.5) / SH_C0;

            // Store opacity (convert from uint8 to float and apply inverse sigmoid)
            const epsilon = 1e-6;
            const normalizedOpacity = Math.max(epsilon, Math.min(1.0 - epsilon, opacity / 255.0));
            (columns[9].data as Float32Array)[splatIndex] = Math.log(normalizedOpacity / (1.0 - normalizedOpacity));

            // Store rotation quaternion (convert from uint8 [0,255] to float [-1,1] and normalize)
            const rot0Norm = (rot0 / 255.0) * 2.0 - 1.0;
            const rot1Norm = (rot1 / 255.0) * 2.0 - 1.0;
            const rot2Norm = (rot2 / 255.0) * 2.0 - 1.0;
            const rot3Norm = (rot3 / 255.0) * 2.0 - 1.0;

            // Normalize quaternion
            const length = Math.sqrt(rot0Norm * rot0Norm + rot1Norm * rot1Norm + rot2Norm * rot2Norm + rot3Norm * rot3Norm);
            if (length > 0) {
                (columns[10].data as Float32Array)[splatIndex] = rot0Norm / length;
                (columns[11].data as Float32Array)[splatIndex] = rot1Norm / length;
                (columns[12].data as Float32Array)[splatIndex] = rot2Norm / length;
                (columns[13].data as Float32Array)[splatIndex] = rot3Norm / length;
            } else {
                // Default to identity quaternion if invalid
                (columns[10].data as Float32Array)[splatIndex] = 0.0;
                (columns[11].data as Float32Array)[splatIndex] = 0.0;
                (columns[12].data as Float32Array)[splatIndex] = 0.0;
                (columns[13].data as Float32Array)[splatIndex] = 1.0;
            }
        }
    }

    return {
        comments: [],
        elements: [{
            name: 'vertex',
            dataTable: new DataTable(columns)
        }]
    };
};

export { SplatData, readSplat };
