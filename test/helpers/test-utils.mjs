/**
 * Test utility functions for splat-transform tests.
 */

import { Column, DataTable } from '../../dist/index.mjs';

/**
 * Creates a minimal DataTable with standard Gaussian splat columns.
 * Values are deterministic based on index for reproducible tests.
 *
 * @param {number} count - Number of splats to create
 * @param {object} [options] - Options for customizing the data
 * @param {boolean} [options.includeSH=false] - Include spherical harmonics columns
 * @param {number} [options.shBands=1] - Number of SH bands (1=9 coeffs, 2=24, 3=45)
 * @returns {DataTable} A DataTable with Gaussian splat data
 */
function createTestDataTable(count, options = {}) {
    const { includeSH = false, shBands = 1 } = options;

    // Constants for encoding
    const SH_C0 = 0.28209479177387814;
    const packClr = (c) => (c - 0.5) / SH_C0;
    const packOpacity = (opacity) => {
        if (opacity <= 0) return -20;
        if (opacity >= 1) return 20;
        return -Math.log(1 / opacity - 1);
    };

    // Create base columns
    const columns = [
        new Column('x', new Float32Array(count)),
        new Column('y', new Float32Array(count)),
        new Column('z', new Float32Array(count)),
        new Column('scale_0', new Float32Array(count)),
        new Column('scale_1', new Float32Array(count)),
        new Column('scale_2', new Float32Array(count)),
        new Column('f_dc_0', new Float32Array(count)),
        new Column('f_dc_1', new Float32Array(count)),
        new Column('f_dc_2', new Float32Array(count)),
        new Column('opacity', new Float32Array(count)),
        new Column('rot_0', new Float32Array(count)),
        new Column('rot_1', new Float32Array(count)),
        new Column('rot_2', new Float32Array(count)),
        new Column('rot_3', new Float32Array(count))
    ];

    // Grid dimensions for positioning
    const gridSize = Math.ceil(Math.sqrt(count));
    const spacing = 1.0;
    const scale = 0.1;

    for (let i = 0; i < count; i++) {
        const gx = i % gridSize;
        const gz = Math.floor(i / gridSize);

        // Position: grid layout
        columns[0].data[i] = (gx - gridSize / 2) * spacing; // x
        columns[1].data[i] = 0; // y
        columns[2].data[i] = (gz - gridSize / 2) * spacing; // z

        // Scale: log-encoded
        columns[3].data[i] = Math.log(scale);
        columns[4].data[i] = Math.log(scale);
        columns[5].data[i] = Math.log(scale);

        // Color: varying based on position
        const r = (gx + 1) / (gridSize + 1);
        const g = (gz + 1) / (gridSize + 1);
        const b = 0.5;
        columns[6].data[i] = packClr(r);
        columns[7].data[i] = packClr(g);
        columns[8].data[i] = packClr(b);

        // Opacity: sigmoid-encoded
        columns[9].data[i] = packOpacity(0.9);

        // Rotation: identity quaternion
        columns[10].data[i] = 0; // rot_0
        columns[11].data[i] = 0; // rot_1
        columns[12].data[i] = 0; // rot_2
        columns[13].data[i] = 1; // rot_3
    }

    // Add spherical harmonics if requested
    if (includeSH) {
        const shCoeffs = [0, 3, 8, 15][shBands];
        for (let c = 0; c < shCoeffs * 3; c++) {
            const shColumn = new Column(`f_rest_${c}`, new Float32Array(count));
            // Small deterministic values
            for (let i = 0; i < count; i++) {
                shColumn.data[i] = ((c + i) % 10 - 5) * 0.01;
            }
            columns.push(shColumn);
        }
    }

    return new DataTable(columns);
}

/**
 * Creates a 4x4 grid of splats (16 total) for minimal testing.
 * @param {object} [options] - Options passed to createTestDataTable
 * @returns {DataTable} A DataTable with 16 Gaussian splats
 */
function createMinimalTestData(options = {}) {
    return createTestDataTable(16, options);
}

/**
 * Encodes a DataTable to PLY binary format.
 * @param {DataTable} dataTable - The data to encode
 * @returns {Uint8Array} PLY file as binary data
 */
function encodePlyBinary(dataTable) {
    const columns = dataTable.columns;
    const numRows = dataTable.numRows;

    // Build header
    const columnTypeToPlyType = (type) => {
        switch (type) {
            case 'float32': return 'float';
            case 'float64': return 'double';
            case 'int8': return 'char';
            case 'uint8': return 'uchar';
            case 'int16': return 'short';
            case 'uint16': return 'ushort';
            case 'int32': return 'int';
            case 'uint32': return 'uint';
        }
    };

    const headerLines = [
        'ply',
        'format binary_little_endian 1.0',
        `element vertex ${numRows}`,
        ...columns.map(c => `property ${columnTypeToPlyType(c.dataType)} ${c.name}`),
        'end_header'
    ];
    const headerStr = headerLines.join('\n') + '\n';
    const headerBytes = new TextEncoder().encode(headerStr);

    // Calculate row size
    const sizes = columns.map(c => c.data.BYTES_PER_ELEMENT);
    const rowSize = sizes.reduce((a, b) => a + b, 0);

    // Create output buffer
    const dataSize = numRows * rowSize;
    const result = new Uint8Array(headerBytes.length + dataSize);

    // Write header
    result.set(headerBytes, 0);

    // Write data
    const buffers = columns.map(c => new Uint8Array(c.data.buffer, c.data.byteOffset, c.data.byteLength));
    let offset = headerBytes.length;

    for (let row = 0; row < numRows; row++) {
        for (let col = 0; col < columns.length; col++) {
            const size = sizes[col];
            const colOffset = row * size;
            result.set(buffers[col].subarray(colOffset, colOffset + size), offset);
            offset += size;
        }
    }

    return result;
}

export { createTestDataTable, createMinimalTestData, encodePlyBinary };
