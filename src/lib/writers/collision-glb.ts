import * as fs from 'node:fs';
import * as path from 'node:path';

import type { Bounds } from '../data-table';
import { coplanarMerge, greedyVoxelMesh, marchingCubes } from '../mesh';
import { fmtCount, logger } from '../utils';
import { BlockMaskBuffer } from '../voxel/block-mask-buffer';

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
 * Two surface shapes are supported:
 * - `'edge'` (default): greedy voxel meshing. Emits one quad per maximal
 *   coplanar rectangle of exposed voxel faces with shared (welded) corner
 *   vertices. Already minimal; no further reduction pass is run.
 * - `'smooth'`: marching cubes followed by a lossless coplanar-merge pass
 *   that fuses the redundant axis-aligned triangles inside each voxel-face
 *   plane into greedy-style quads while leaving the corner-cutting bevel
 *   triangles untouched. The output surface is identical to the raw MC
 *   surface but typically has 1-2 orders of magnitude fewer triangles.
 *
 * @param blockBuffer - Voxel block data after filtering
 * @param gridBounds - Grid bounds aligned to block boundaries
 * @param voxelResolution - Size of each voxel in world units
 * @param meshType - Surface shape to extract (`'edge'` or `'smooth'`)
 * @returns GLB bytes, or null if no triangles were generated
 */
const buildCollisionMesh = (
    blockBuffer: BlockMaskBuffer,
    gridBounds: Bounds,
    voxelResolution: number,
    meshType: 'edge' | 'smooth' = 'edge'
): Uint8Array | null => {
    const g = logger.group('Collision mesh');

    const extractSub = logger.group(`Extracting (${meshType})`);
    const rawMesh = meshType === 'edge' ?
        greedyVoxelMesh(blockBuffer, gridBounds, voxelResolution) :
        marchingCubes(blockBuffer, gridBounds, voxelResolution);
    logger.info(`raw vertices: ${fmtCount(rawMesh.positions.length / 3)}`);
    logger.info(`raw triangles: ${fmtCount(rawMesh.indices.length / 3)}`);

    if (rawMesh.indices.length < 3) {
        logger.warn('no triangles generated, skipping GLB output');
        extractSub.end();
        g.end();
        return null;
    }
    extractSub.end();

    let finalMesh = rawMesh;
    if (meshType === 'smooth') {
        const mergeSub = logger.group('Merging coplanar faces');
        finalMesh = coplanarMerge(rawMesh, voxelResolution);

        const reduction = (1 - finalMesh.indices.length / rawMesh.indices.length) * 100;
        logger.info(`merged vertices: ${fmtCount(finalMesh.positions.length / 3)}`);
        logger.info(`merged triangles: ${fmtCount(finalMesh.indices.length / 3)}`);
        logger.info(`reduction: ${reduction.toFixed(0)}%`);
        mergeSub.end();

        if (typeof process !== 'undefined' && process.env && process.env.COPLANAR_DUMP_RAW === '1') {
            // Emit raw MC mesh side-by-side for diff inspection. Half-voxel
            // welder applied so positional comparisons match the merged stream.
            const r = voxelResolution;
            const inv = 2 / r;
            const map = new Map<string, number>();
            const newIdx = new Uint32Array(rawMesh.indices.length);
            const newPosTmp: number[] = [];
            for (let i = 0; i < rawMesh.indices.length; i++) {
                const o = rawMesh.indices[i] * 3;
                const x = rawMesh.positions[o];
                const y = rawMesh.positions[o + 1];
                const z = rawMesh.positions[o + 2];
                const key = `${Math.round(x * inv)}_${Math.round(y * inv)}_${Math.round(z * inv)}`;
                let idx = map.get(key);
                if (idx === undefined) {
                    idx = newPosTmp.length / 3;
                    newPosTmp.push(x, y, z);
                    map.set(key, idx);
                }
                newIdx[i] = idx;
            }
            const newPos = new Float32Array(newPosTmp);
            const base = process.env.COPLANAR_DUMP_DIR || '/tmp';
            fs.writeFileSync(path.join(base, 'coplanar-raw.glb'), encodeGlb(newPos, newIdx));
            fs.writeFileSync(path.join(base, 'coplanar-merged.glb'), encodeGlb(finalMesh.positions, finalMesh.indices));
            console.error(`[coplanar-dump] wrote raw=${newPos.length / 3} verts, merged=${finalMesh.positions.length / 3} verts`);
            const rawSet = new Set<string>();
            for (let i = 0; i < newPos.length; i += 3) {
                rawSet.add(`${Math.round(newPos[i] * inv)}_${Math.round(newPos[i + 1] * inv)}_${Math.round(newPos[i + 2] * inv)}`);
            }
            const phantoms: number[][] = [];
            for (let i = 0; i < finalMesh.positions.length; i += 3) {
                const x = finalMesh.positions[i];
                const y = finalMesh.positions[i + 1];
                const z = finalMesh.positions[i + 2];
                const key = `${Math.round(x * inv)}_${Math.round(y * inv)}_${Math.round(z * inv)}`;
                if (!rawSet.has(key)) phantoms.push([x, y, z]);
            }
            console.error(`[coplanar-dump] phantom verts in merged but NOT in raw: ${phantoms.length}`);
            for (const p of phantoms.slice(0, 20)) {
                console.error(`  ${p[0].toFixed(4)}, ${p[1].toFixed(4)}, ${p[2].toFixed(4)}`);
            }
        }
    }

    g.end();
    return encodeGlb(finalMesh.positions, finalMesh.indices);
};

export { buildCollisionMesh };
