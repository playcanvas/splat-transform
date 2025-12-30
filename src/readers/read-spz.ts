import { Column, DataTable } from '../data-table/data-table';
import { ReadFileSystem } from '../serialize/read-file-system';

// See https://github.com/nianticlabs/spz for reference implementation

const decompressGZIP = async (buffer: ArrayBuffer): Promise<ArrayBuffer> => {
    const blob = new Blob([buffer], { type: 'application/gzip' });
    const ds = new DecompressionStream('gzip');
    const decompressionStream = blob.stream().pipeThrough(ds);
    return await new Response(decompressionStream).arrayBuffer();
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

const HARMONICS_COMPONENT_COUNT = [0, 9, 24, 45];

const readSpz = async (filename: string, fs: ReadFileSystem): Promise<DataTable> => {
    const readSource = await fs.createReader(filename);
    let fileBuffer = await readSource.arrayBuffer();

    // Check if file is GZip compressed
    const magicView = new DataView(fileBuffer, 0, 4);
    if (magicView.getUint16(0) === 0x1F8B) { // '1F 8B' is the magic for gzip
        fileBuffer = await decompressGZIP(fileBuffer);
    }

    const headerView = new DataView(fileBuffer, 0, 4);
    const magic = headerView.getUint32(0, true);
    if (magic !== 0x5053474e) { // NGSP
        throw new Error('invalid file header');
    }

    const HEADER_SIZE = 16;
    const totalSize = fileBuffer.byteLength;

    if (totalSize < HEADER_SIZE) {
        throw new Error('File too small to be valid .spz format');
    }

    // Parse header
    const header = new DataView(fileBuffer, 0, HEADER_SIZE);

    const version = header.getUint32(4, true);
    if (!(version === 2 || version === 3)) {
        throw new Error(`Unsupported version ${version}`);
    }

    const numSplats = header.getUint32(8, true);
    const shDegree = header.getUint8(12);
    const fractionalBits = header.getUint8(13);
    // const unused_flags = header.getUint8(14); // Trained with antialiasing or not
    // const unused_reserved = header.getUint8(15); // padding

    const positionsByteSize = numSplats * 3 * 3; // 3 * 24bit values
    const alphasByteSize = numSplats; // u8
    const colorsByteSize = numSplats * 3; // u8 * 3
    const scalesByteSize = numSplats * 3; // u8 * 3
    const rotationsByteSize = numSplats * 3; // u8 * 3
    const harmonicsComponentCount = HARMONICS_COMPONENT_COUNT[shDegree];
    const shByteSize = numSplats * harmonicsComponentCount;

    const positionsView = new DataView(fileBuffer, HEADER_SIZE, positionsByteSize);
    const alphasView = new DataView(fileBuffer, HEADER_SIZE + positionsByteSize, alphasByteSize);
    const colorsView = new DataView(fileBuffer, HEADER_SIZE + positionsByteSize + alphasByteSize, colorsByteSize);
    const scalesView = new DataView(fileBuffer, HEADER_SIZE + positionsByteSize + alphasByteSize + colorsByteSize, scalesByteSize);
    const rotationsView = new DataView(fileBuffer, HEADER_SIZE + positionsByteSize + alphasByteSize + colorsByteSize + scalesByteSize, rotationsByteSize);
    const shView = new DataView(fileBuffer, HEADER_SIZE + positionsByteSize + alphasByteSize + colorsByteSize + scalesByteSize + rotationsByteSize, shByteSize);

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

        // Read rotation quaternion (4 × uint8)
        const rotation = [0.0, 0.0, 0.0, 0.0];
        if (version === 2) {
            rotation[1] = rotationsView.getUint8(splatIndex * 3 + 0);
            rotation[2] = rotationsView.getUint8(splatIndex * 3 + 1);
            rotation[3] = rotationsView.getUint8(splatIndex * 3 + 2);
        } else if (version === 3) {
            // Copied from https://github.com/nianticlabs/spz/blob/e2101d6924b3936a6fb630cd4a881cb610c57d77/src/cc/load-spz.cc#L347
            let packedRotation = rotationsView.getUint32(splatIndex);
            const c_mask = (1 << 9) - 1;

            const largestRotIndex = packedRotation >> 30;
            let sum_squares = 0;
            for (let i = 3; i >= 0; --i) {
                if (i !== largestRotIndex) {
                    const mag = packedRotation & c_mask;
                    const negbit = (packedRotation >> 9) & 1;
                    packedRotation >>= 10;
                    rotation[i] = Math.SQRT1_2 * (mag) / c_mask;
                    if (negbit === 1) {
                        rotation[i] = -rotation[i];
                    }
                    sum_squares += rotation[i] * rotation[i];
                }
            }
            rotation[largestRotIndex] = Math.sqrt(1.0 - sum_squares);
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

        // Store rotation quaternion (convert from uint8 [0,255] to float [-1,1])

        let rot0Norm = 0.0;
        const rot1Norm = (rotation[1] / 127.5) - 1.0;
        const rot2Norm = (rotation[2] / 127.5) - 1.0;
        const rot3Norm = (rotation[3] / 127.5) - 1.0;
        if (version === 2) {
            const rotationDot = rot1Norm * rot1Norm + rot2Norm * rot2Norm + rot3Norm * rot3Norm;
            rot0Norm = Math.sqrt(Math.max(0.0, 1.0 - rotationDot));
        } else if (version === 3) {
            rot0Norm = (rotation[0] / 127.5) - 1.0;
        }

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
    }

    return new DataTable(columns);
};

export { readSpz };
