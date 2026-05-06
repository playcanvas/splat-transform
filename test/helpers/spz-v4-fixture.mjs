const SPZ_V4_HARMONICS_COMPONENT_COUNT = [0, 9, 24, 45, 72];

const SPZ_V3_PACKED_QUATERNIONS = {
    identity: 0xC0000000,
    rotate45Z: 0xF5A40000
};

const wrapZstdRaw = (data) => {
    const len = data.length;
    if (len > 0x1FFFFF) {
        throw new Error(`Raw block too large: ${len}`);
    }

    const out = new Uint8Array(4 + 1 + 4 + 3 + len);
    const view = new DataView(out.buffer);
    view.setUint32(0, 0xFD2FB528, true);
    out[4] = 0xA0;
    view.setUint32(5, len, true);
    const blockHeader = (len << 3) | 0b001;
    out[9] = blockHeader & 0xFF;
    out[10] = (blockHeader >>> 8) & 0xFF;
    out[11] = (blockHeader >>> 16) & 0xFF;
    out.set(data, 12);
    return out;
};

const createSpzV4Fixture = ({ shDegree = 0, extensionBytes = new Uint8Array(0) } = {}) => {
    const count = 2;
    const HEADER_SIZE = 32;

    const positions = new Uint8Array(count * 9);
    const alphas = new Uint8Array(count);
    const colors = new Uint8Array(count * 3);
    const scales = new Uint8Array(count * 3);
    const rotations = new Uint8Array(count * 4);

    const positionValues = [
        [0, 0, 0],
        [1, 0, 0]
    ];
    for (let i = 0; i < count; i++) {
        const values = positionValues[i].map(value => (Math.round(value * 4096) & 0xFFFFFF) >>> 0);
        for (let j = 0; j < 3; j++) {
            positions[i * 9 + j * 3 + 0] = values[j] & 0xFF;
            positions[i * 9 + j * 3 + 1] = (values[j] >> 8) & 0xFF;
            positions[i * 9 + j * 3 + 2] = (values[j] >> 16) & 0xFF;
        }
    }

    alphas[0] = 230;
    alphas[1] = 230;
    colors.set([180, 90, 128, 90, 180, 128]);

    const encodedScale = Math.round((Math.log(0.1) + 10) * 16);
    for (let i = 0; i < count; i++) {
        scales[i * 3 + 0] = encodedScale;
        scales[i * 3 + 1] = encodedScale;
        scales[i * 3 + 2] = encodedScale;
    }

    const rotView = new DataView(rotations.buffer);
    rotView.setUint32(0, SPZ_V3_PACKED_QUATERNIONS.identity, true);
    rotView.setUint32(4, SPZ_V3_PACKED_QUATERNIONS.rotate45Z, true);

    const streams = [positions, alphas, colors, scales, rotations];
    if (shDegree > 0) {
        const sh = new Uint8Array(count * SPZ_V4_HARMONICS_COMPONENT_COUNT[shDegree]);
        sh.fill(128);
        streams.push(sh);
    }

    const compressed = streams.map(wrapZstdRaw);
    const numStreams = compressed.length;
    const tocSize = numStreams * 16;
    const tocByteOffset = HEADER_SIZE + extensionBytes.length;
    const totalDataSize = compressed.reduce((sum, chunk) => sum + chunk.length, 0);
    const totalSize = HEADER_SIZE + extensionBytes.length + tocSize + totalDataSize;

    const buffer = new ArrayBuffer(totalSize);
    const view = new DataView(buffer);
    const bytes = new Uint8Array(buffer);

    view.setUint32(0, 0x5053474E, true);
    view.setUint32(4, 4, true);
    view.setUint32(8, count, true);
    view.setUint8(12, shDegree);
    view.setUint8(13, 12);
    view.setUint8(14, extensionBytes.length > 0 ? 0x2 : 0);
    view.setUint8(15, numStreams);
    view.setUint32(16, tocByteOffset, true);

    bytes.set(extensionBytes, HEADER_SIZE);

    let tocOffset = HEADER_SIZE + extensionBytes.length;
    for (let i = 0; i < numStreams; i++) {
        view.setBigUint64(tocOffset, BigInt(compressed[i].length), true);
        view.setBigUint64(tocOffset + 8, BigInt(streams[i].length), true);
        tocOffset += 16;
    }

    let dataOffset = HEADER_SIZE + extensionBytes.length + tocSize;
    for (const chunk of compressed) {
        bytes.set(chunk, dataOffset);
        dataOffset += chunk.length;
    }

    return new Uint8Array(buffer);
};

export { createSpzV4Fixture };
