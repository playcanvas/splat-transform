#!/usr/bin/env node
/**
 * Creates binary test fixtures for input-only formats.
 * Run with: node test/fixtures/create-fixtures.mjs
 *
 * This generates:
 * - minimal.splat - A .splat file with known values
 * - minimal.ksplat - A .ksplat file (compression mode 0)
 * - minimal.spz - A .spz file (if feasible)
 */

import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const outputDir = join(__dirname, 'splat');

// Ensure output directory exists
mkdirSync(outputDir, { recursive: true });

/**
 * Creates a minimal .splat file (antimatter15 format).
 * Format: 32 bytes per splat
 * - 3x float32: position (x, y, z)
 * - 3x float32: scale (linear, not log)
 * - 4x uint8: color (r, g, b, opacity)
 * - 4x uint8: rotation quaternion
 */
function createSplatFixture(count = 4) {
    const BYTES_PER_SPLAT = 32;
    const buffer = new ArrayBuffer(count * BYTES_PER_SPLAT);
    const view = new DataView(buffer);

    const gridSize = Math.ceil(Math.sqrt(count));
    const spacing = 1.0;
    const scale = 0.1;

    for (let i = 0; i < count; i++) {
        const offset = i * BYTES_PER_SPLAT;
        const gx = i % gridSize;
        const gz = Math.floor(i / gridSize);

        // Position (3x float32)
        view.setFloat32(offset + 0, (gx - gridSize / 2) * spacing, true); // x
        view.setFloat32(offset + 4, 0, true); // y
        view.setFloat32(offset + 8, (gz - gridSize / 2) * spacing, true); // z

        // Scale (3x float32, linear values)
        view.setFloat32(offset + 12, scale, true);
        view.setFloat32(offset + 16, scale, true);
        view.setFloat32(offset + 20, scale, true);

        // Color (4x uint8): gradient based on position
        const r = Math.round(((gx + 1) / (gridSize + 1)) * 255);
        const g = Math.round(((gz + 1) / (gridSize + 1)) * 255);
        const b = 128; // 0.5
        const opacity = 230; // ~0.9
        view.setUint8(offset + 24, r);
        view.setUint8(offset + 25, g);
        view.setUint8(offset + 26, b);
        view.setUint8(offset + 27, opacity);

        // Rotation (4x uint8): identity quaternion [0,0,0,1] encoded as [128,128,128,255]
        // Quaternion encoding: (value / 255) * 2 - 1 maps [0,255] to [-1,1]
        // For identity [0,0,0,1]: we need values that map to 0,0,0,1
        // 0 -> (128/255)*2-1 ≈ 0
        // 1 -> (255/255)*2-1 = 1
        view.setUint8(offset + 28, 128); // rot_0 ≈ 0
        view.setUint8(offset + 29, 128); // rot_1 ≈ 0
        view.setUint8(offset + 30, 128); // rot_2 ≈ 0
        view.setUint8(offset + 31, 255); // rot_3 = 1

    }

    return new Uint8Array(buffer);
}

/**
 * Creates a minimal .ksplat file (mkkellogg format, compression mode 0).
 * This is the uncompressed mode which is simpler to generate.
 *
 * Structure:
 * - Main header: 4096 bytes
 * - Section headers: 1024 bytes each
 * - Section data: variable
 */
function createKsplatFixture(count = 4) {
    const MAIN_HEADER_SIZE = 4096;
    const SECTION_HEADER_SIZE = 1024;

    // Compression mode 0 configuration
    const centerBytes = 12;
    const scaleBytes = 12;
    const rotationBytes = 16;
    const colorBytes = 4;
    const harmonicsBytes = 4;
    const bytesPerSplat = centerBytes + scaleBytes + rotationBytes + colorBytes; // 44 bytes

    // Calculate sizes
    const numSections = 1;
    const splatDataSize = count * bytesPerSplat;
    const bucketCount = 1;
    const bucketStorageSize = bucketCount * 12; // 3 floats for bucket center
    const totalSize = MAIN_HEADER_SIZE + (numSections * SECTION_HEADER_SIZE) + bucketStorageSize + splatDataSize;

    const buffer = new ArrayBuffer(totalSize);
    const view = new DataView(buffer);

    // Main header
    view.setUint8(0, 0); // major version
    view.setUint8(1, 1); // minor version
    view.setUint32(4, 1, true); // maxSections
    view.setUint32(16, count, true); // numSplats
    view.setUint16(20, 0, true); // compressionMode (0 = uncompressed)
    view.setFloat32(36, -1.5, true); // minHarmonicsValue
    view.setFloat32(40, 1.5, true); // maxHarmonicsValue

    // Section header (at offset 4096)
    const sectionOffset = MAIN_HEADER_SIZE;
    view.setUint32(sectionOffset + 0, count, true); // sectionSplatCount
    view.setUint32(sectionOffset + 4, count, true); // maxSectionSplats
    view.setUint32(sectionOffset + 8, count, true); // bucketCapacity
    view.setUint32(sectionOffset + 12, bucketCount, true); // bucketCount
    view.setFloat32(sectionOffset + 16, 10.0, true); // spatialBlockSize
    view.setUint16(sectionOffset + 20, 12, true); // bucketStorageSize
    view.setUint32(sectionOffset + 24, 32767, true); // quantizationRange
    view.setUint32(sectionOffset + 32, 1, true); // fullBuckets
    view.setUint32(sectionOffset + 36, 0, true); // partialBuckets
    view.setUint16(sectionOffset + 40, 0, true); // harmonicsDegree (no SH)

    // Bucket centers (one bucket at origin)
    const bucketCentersOffset = MAIN_HEADER_SIZE + SECTION_HEADER_SIZE;
    view.setFloat32(bucketCentersOffset + 0, 0, true); // center x
    view.setFloat32(bucketCentersOffset + 4, 0, true); // center y
    view.setFloat32(bucketCentersOffset + 8, 0, true); // center z

    // Splat data
    const splatDataOffset = bucketCentersOffset + bucketStorageSize;
    const gridSize = Math.ceil(Math.sqrt(count));
    const spacing = 1.0;
    const scale = 0.1;

    for (let i = 0; i < count; i++) {
        const offset = splatDataOffset + i * bytesPerSplat;
        const gx = i % gridSize;
        const gz = Math.floor(i / gridSize);

        // Position (3x float32)
        view.setFloat32(offset + 0, (gx - gridSize / 2) * spacing, true); // x
        view.setFloat32(offset + 4, 0, true); // y
        view.setFloat32(offset + 8, (gz - gridSize / 2) * spacing, true); // z

        // Scale (3x float32, linear values)
        view.setFloat32(offset + 12, scale, true);
        view.setFloat32(offset + 16, scale, true);
        view.setFloat32(offset + 20, scale, true);

        // Rotation (4x float32): identity quaternion
        view.setFloat32(offset + 24, 0, true); // rot_0
        view.setFloat32(offset + 28, 0, true); // rot_1
        view.setFloat32(offset + 32, 0, true); // rot_2
        view.setFloat32(offset + 36, 1, true); // rot_3

        // Color (4x uint8)
        const r = Math.round(((gx + 1) / (gridSize + 1)) * 255);
        const g = Math.round(((gz + 1) / (gridSize + 1)) * 255);
        view.setUint8(offset + 40, r);
        view.setUint8(offset + 41, g);
        view.setUint8(offset + 42, 128); // b = 0.5
        view.setUint8(offset + 43, 230); // opacity ≈ 0.9
    }

    return new Uint8Array(buffer);
}

/**
 * Creates a minimal .spz file (Niantic format, version 2).
 * Format: 16-byte header + position/alpha/color/scale/rotation data
 */
function createSpzFixture(count = 4) {
    const HEADER_SIZE = 16;
    const positionsByteSize = count * 3 * 3; // 3 * 24-bit values
    const alphasByteSize = count;
    const colorsByteSize = count * 3;
    const scalesByteSize = count * 3;
    const rotationsByteSize = count * 3;
    const shDegree = 0; // No spherical harmonics
    const shByteSize = 0;

    const totalSize = HEADER_SIZE + positionsByteSize + alphasByteSize +
                     colorsByteSize + scalesByteSize + rotationsByteSize + shByteSize;

    const buffer = new ArrayBuffer(totalSize);
    const view = new DataView(buffer);
    const bytes = new Uint8Array(buffer);

    // Header
    // Magic: NGSP (0x5053474E little-endian)
    view.setUint32(0, 0x5053474E, true); // NGSP
    view.setUint32(4, 2, true); // version
    view.setUint32(8, count, true); // numSplats
    view.setUint8(12, shDegree); // shDegree
    view.setUint8(13, 8); // fractionalBits (scale = 1/256)

    const gridSize = Math.ceil(Math.sqrt(count));
    const spacing = 1.0;
    const scale = 0.1;
    const fractionalScale = 1 << 8; // 256

    // Positions (24-bit fixed point per component)
    let offset = HEADER_SIZE;
    for (let i = 0; i < count; i++) {
        const gx = i % gridSize;
        const gz = Math.floor(i / gridSize);

        const x = (gx - gridSize / 2) * spacing;
        const y = 0;
        const z = (gz - gridSize / 2) * spacing;

        // Convert to 24-bit fixed point
        const xFixed = Math.round(x * fractionalScale) & 0xFFFFFF;
        const yFixed = Math.round(y * fractionalScale) & 0xFFFFFF;
        const zFixed = Math.round(z * fractionalScale) & 0xFFFFFF;

        const posOffset = offset + i * 9;
        bytes[posOffset + 0] = xFixed & 0xFF;
        bytes[posOffset + 1] = (xFixed >> 8) & 0xFF;
        bytes[posOffset + 2] = (xFixed >> 16) & 0xFF;
        bytes[posOffset + 3] = yFixed & 0xFF;
        bytes[posOffset + 4] = (yFixed >> 8) & 0xFF;
        bytes[posOffset + 5] = (yFixed >> 16) & 0xFF;
        bytes[posOffset + 6] = zFixed & 0xFF;
        bytes[posOffset + 7] = (zFixed >> 8) & 0xFF;
        bytes[posOffset + 8] = (zFixed >> 16) & 0xFF;
    }
    offset += positionsByteSize;

    // Alphas (uint8)
    for (let i = 0; i < count; i++) {
        bytes[offset + i] = 230; // ~0.9 opacity
    }
    offset += alphasByteSize;

    // Colors (3x uint8 per splat)
    for (let i = 0; i < count; i++) {
        const gx = i % gridSize;
        const gz = Math.floor(i / gridSize);

        // SPZ uses SH_C0_2 = 0.15 for color encoding
        const r = Math.round(((gx + 1) / (gridSize + 1)) * 255);
        const g = Math.round(((gz + 1) / (gridSize + 1)) * 255);

        bytes[offset + i * 3 + 0] = r;
        bytes[offset + i * 3 + 1] = g;
        bytes[offset + i * 3 + 2] = 128; // b = 0.5
    }
    offset += colorsByteSize;

    // Scales (3x uint8, log encoded: value/16 - 10)
    // For scale=0.1: log(0.1) ≈ -2.3, encoded as (-2.3 + 10) * 16 ≈ 123
    const logScale = Math.log(scale);
    const encodedScale = Math.round((logScale + 10) * 16);
    for (let i = 0; i < count; i++) {
        bytes[offset + i * 3 + 0] = encodedScale;
        bytes[offset + i * 3 + 1] = encodedScale;
        bytes[offset + i * 3 + 2] = encodedScale;
    }
    offset += scalesByteSize;

    // Rotations (3x uint8 for version 2: rot_1, rot_2, rot_3)
    // Identity quaternion: rot_0 is reconstructed, rot_1=rot_2=rot_3=0
    // Encoding: (value + 1) * 127.5 maps [-1,1] to [0,255]
    // For 0: (0 + 1) * 127.5 = 127.5 ≈ 128
    for (let i = 0; i < count; i++) {
        bytes[offset + i * 3 + 0] = 128; // rot_1 = 0
        bytes[offset + i * 3 + 1] = 128; // rot_2 = 0
        bytes[offset + i * 3 + 2] = 128; // rot_3 = 0
    }

    return new Uint8Array(buffer);
}

// Generate fixtures
console.log('Creating test fixtures...');

const splatData = createSplatFixture(4);
writeFileSync(join(outputDir, 'minimal.splat'), splatData);
console.log(`Created minimal.splat (${splatData.length} bytes, 4 splats)`);

const ksplatData = createKsplatFixture(4);
writeFileSync(join(outputDir, 'minimal.ksplat'), ksplatData);
console.log(`Created minimal.ksplat (${ksplatData.length} bytes, 4 splats)`);

const spzData = createSpzFixture(4);
writeFileSync(join(outputDir, 'minimal.spz'), spzData);
console.log(`Created minimal.spz (${spzData.length} bytes, 4 splats)`);

console.log('Done!');
