import type { Bounds } from '../data-table';
import { coplanarMerge, marchingCubes, voxelFaces, type Mesh } from '../mesh';
import type { CollisionMeshShape } from '../types';
import { fmtCount, logger } from '../utils';
import { SparseVoxelGrid } from '../voxel/sparse-voxel-grid';

/**
 * Build a minimal GLB (glTF 2.0 binary) file containing a single triangle mesh.
 *
 * The output contains only positions and triangle indices — no normals,
 * UVs, or materials — suitable for collision meshes.
 *
 * @param positions - Vertex positions (3 floats per vertex)
 * @param indices - Triangle indices (3 per triangle, unsigned 32-bit)
 * @returns GLB file as a Uint8Array
 */
function encodeGlb(positions: Float32Array, indices: Uint32Array): Uint8Array {
    const vertexCount = positions.length / 3;
    const indexCount = indices.length;

    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < positions.length; i += 3) {
        const x = positions[i], y = positions[i + 1], z = positions[i + 2];
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (z < minZ) minZ = z;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
        if (z > maxZ) maxZ = z;
    }

    const positionsByteLength = positions.byteLength;
    const indicesByteLength = indices.byteLength;
    const totalBinSize = positionsByteLength + indicesByteLength;

    const gltf = {
        asset: { version: '2.0', generator: 'splat-transform' },
        scene: 0,
        scenes: [{ nodes: [0] }],
        nodes: [{ mesh: 0 }],
        meshes: [{
            primitives: [{
                attributes: { POSITION: 0 },
                indices: 1
            }]
        }],
        accessors: [
            {
                bufferView: 0,
                componentType: 5126, // FLOAT
                count: vertexCount,
                type: 'VEC3',
                min: [minX, minY, minZ],
                max: [maxX, maxY, maxZ]
            },
            {
                bufferView: 1,
                componentType: 5125, // UNSIGNED_INT
                count: indexCount,
                type: 'SCALAR'
            }
        ],
        bufferViews: [
            {
                buffer: 0,
                byteOffset: 0,
                byteLength: positionsByteLength,
                target: 34962 // ARRAY_BUFFER
            },
            {
                buffer: 0,
                byteOffset: positionsByteLength,
                byteLength: indicesByteLength,
                target: 34963 // ELEMENT_ARRAY_BUFFER
            }
        ],
        buffers: [{ byteLength: totalBinSize }]
    };

    const jsonString = JSON.stringify(gltf);
    const jsonEncoder = new TextEncoder();
    const jsonBytes = jsonEncoder.encode(jsonString);

    // JSON chunk must be padded to 4-byte alignment with spaces (0x20)
    const jsonPadding = (4 - (jsonBytes.length % 4)) % 4;
    const jsonChunkLength = jsonBytes.length + jsonPadding;

    // BIN chunk must be padded to 4-byte alignment with zeros
    const binPadding = (4 - (totalBinSize % 4)) % 4;
    const binChunkLength = totalBinSize + binPadding;

    // GLB layout: header (12) + JSON chunk header (8) + JSON data + BIN chunk header (8) + BIN data
    const totalLength = 12 + 8 + jsonChunkLength + 8 + binChunkLength;
    const buffer = new ArrayBuffer(totalLength);
    const view = new DataView(buffer);
    const byteArray = new Uint8Array(buffer);
    let offset = 0;

    // GLB header
    view.setUint32(offset, 0x46546C67, true); offset += 4; // magic: "glTF"
    view.setUint32(offset, 2, true); offset += 4;           // version: 2
    view.setUint32(offset, totalLength, true); offset += 4;  // total length

    // JSON chunk header
    view.setUint32(offset, jsonChunkLength, true); offset += 4;
    view.setUint32(offset, 0x4E4F534A, true); offset += 4; // type: "JSON"

    // JSON chunk data
    byteArray.set(jsonBytes, offset); offset += jsonBytes.length;
    for (let i = 0; i < jsonPadding; i++) {
        byteArray[offset++] = 0x20;
    }

    // BIN chunk header
    view.setUint32(offset, binChunkLength, true); offset += 4;
    view.setUint32(offset, 0x004E4942, true); offset += 4; // type: "BIN\0"

    // BIN chunk data: positions then indices
    byteArray.set(new Uint8Array(positions.buffer, positions.byteOffset, positionsByteLength), offset);
    offset += positionsByteLength;
    byteArray.set(new Uint8Array(indices.buffer, indices.byteOffset, indicesByteLength), offset);

    return byteArray;
}

/**
 * Extract a collision mesh from voxel data and encode it as a GLB file.
 *
 * Generates collision geometry from voxel data using either the smooth
 * marching-cubes path or a direct watertight voxel-face mesh.
 *
 * @param grid - Voxel grid after filtering / nav phases
 * @param gridBounds - Grid bounds aligned to block boundaries
 * @param voxelResolution - Size of each voxel in world units
 * @param shape - Collision mesh shape to generate
 * @returns GLB bytes, or null if no triangles were generated
 */
const buildCollisionMesh = (
    grid: SparseVoxelGrid,
    gridBounds: Bounds,
    voxelResolution: number,
    shape: CollisionMeshShape = 'smooth'
): Uint8Array | null => {
    const g = logger.group('Collision mesh');

    let finalMesh: Mesh;
    if (shape === 'faces') {
        const extractSub = logger.group('Extracting voxel faces');
        finalMesh = voxelFaces(grid, gridBounds, voxelResolution);
        logger.info(`vertices: ${fmtCount(finalMesh.positions.length / 3)}`);
        logger.info(`triangles: ${fmtCount(finalMesh.indices.length / 3)}`);
        extractSub.end();
    } else {
        const extractSub = logger.group('Extracting');
        const preMergedMesh = marchingCubes(grid, gridBounds, voxelResolution, { mergeFlatFaces: true });
        logger.info(`pre-merged vertices: ${fmtCount(preMergedMesh.positions.length / 3)}`);
        logger.info(`pre-merged triangles: ${fmtCount(preMergedMesh.indices.length / 3)}`);
        const preMergedIndexCount = preMergedMesh.indices.length;
        extractSub.end();

        if (preMergedIndexCount < 3) {
            finalMesh = preMergedMesh;
        } else {
            const mergeSub = logger.group('Merging coplanar faces');
            finalMesh = coplanarMerge(preMergedMesh, voxelResolution);

            const reduction = (1 - finalMesh.indices.length / preMergedIndexCount) * 100;
            logger.info(`merged vertices: ${fmtCount(finalMesh.positions.length / 3)}`);
            logger.info(`merged triangles: ${fmtCount(finalMesh.indices.length / 3)}`);
            logger.info(`reduction: ${reduction.toFixed(0)}%`);
            mergeSub.end();
        }
    }

    if (finalMesh.indices.length < 3) {
        logger.warn('no triangles generated, skipping GLB output');
        g.end();
        return null;
    }

    g.end();
    return encodeGlb(finalMesh.positions, finalMesh.indices);
};

export { buildCollisionMesh };
