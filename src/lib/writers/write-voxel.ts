import { MeshoptSimplifier } from 'meshoptimizer/simplifier';
import { Vec3 } from 'playcanvas';

import { buildCollisionGlb } from './collision-glb';
import { Column, DataTable, computeGaussianExtents, computeWriteTransform, transformColumns } from '../data-table';
import { GpuVoxelization } from '../gpu';
import { type FileSystem, writeFile } from '../io/write';
import { GaussianBVH } from '../spatial';
import type { DeviceCreator } from '../types';
import { logger, Transform } from '../utils';
import { buildSparseOctree, type SparseOctree } from './sparse-octree';
import {
    filterAndFillBlocks,
    alignGridBounds,
    carveInterior,
    fillExterior,
    type NavSeed,
    marchingCubes,
    voxelizeToBuffer
} from '../voxel';

/**
 * Options for writing a voxel octree file.
 */
type WriteVoxelOptions = {
    /** Output filename ending in .voxel.json */
    filename: string;

    /** Gaussian splat data to voxelize */
    dataTable: DataTable;

    /** Size of each voxel in world units. Default: 0.05 */
    voxelResolution?: number;

    /** Opacity threshold for solid voxels - voxels below this are considered empty. Default: 0.5 */
    opacityCutoff?: number;

    /** Optional function to create a GPU device for voxelization */
    createDevice?: DeviceCreator;

    /** Exterior fill radius in world units. Set to 0 to disable exterior fill. Defaults to 1.6 when nav simplification is active. Requires navSeed to be set; ignored without it. */
    navExteriorRadius?: number;

    /** Capsule dimensions for navigation simplification. Height of 0 disables interior carve. When height > 0, only voxels contactable from the seed are kept. */
    navCapsule?: { height: number; radius: number };

    /** Seed position in world space for navigation flood fill. Required when navCapsule is set with height > 0. */
    navSeed?: NavSeed;

    /** Whether to generate a collision mesh (.collision.glb) alongside the voxel data. Default: false */
    collisionMesh?: boolean;

    /** Maximum geometric error for collision mesh simplification as a fraction of voxelResolution. Default: 0.08 */
    meshSimplifyError?: number;
};

/**
 * Metadata for a voxel octree file.
 */
interface VoxelMetadata {
    /** File format version */
    version: string;

    /** Grid bounds aligned to 4x4x4 block boundaries */
    gridBounds: { min: number[]; max: number[] };

    /** Scene bounds (in PlayCanvas coordinate space for v1.1+) */
    sceneBounds: { min: number[]; max: number[] };

    /** Size of each voxel in world units */
    voxelResolution: number;

    /** Voxels per leaf dimension (always 4) */
    leafSize: number;

    /** Maximum tree depth */
    treeDepth: number;

    /** Number of interior nodes */
    numInteriorNodes: number;

    /** Number of mixed leaf nodes */
    numMixedLeaves: number;

    /** Total number of Uint32 entries in the nodes array */
    nodeCount: number;

    /** Total number of Uint32 entries in the leafData array */
    leafDataCount: number;
}

/**
 * Write octree data to files.
 *
 * @param fs - File system for writing output files.
 * @param jsonFilename - Output filename for JSON metadata.
 * @param octree - Sparse octree structure to write.
 */
const writeOctreeFiles = async (
    fs: FileSystem,
    jsonFilename: string,
    octree: SparseOctree
): Promise<void> => {
    // Build metadata object
    const metadata: VoxelMetadata = {
        version: '1.1',
        gridBounds: {
            min: [octree.gridBounds.min.x, octree.gridBounds.min.y, octree.gridBounds.min.z],
            max: [octree.gridBounds.max.x, octree.gridBounds.max.y, octree.gridBounds.max.z]
        },
        sceneBounds: {
            min: [octree.sceneBounds.min.x, octree.sceneBounds.min.y, octree.sceneBounds.min.z],
            max: [octree.sceneBounds.max.x, octree.sceneBounds.max.y, octree.sceneBounds.max.z]
        },
        voxelResolution: octree.voxelResolution,
        leafSize: octree.leafSize,
        treeDepth: octree.treeDepth,
        numInteriorNodes: octree.numInteriorNodes,
        numMixedLeaves: octree.numMixedLeaves,
        nodeCount: octree.nodes.length,
        leafDataCount: octree.leafData.length
    };

    // Write JSON metadata
    logger.log(`writing '${jsonFilename}'...`);
    await writeFile(fs, jsonFilename, JSON.stringify(metadata, null, 2));

    // Write binary data (nodes + leafData concatenated)
    const binFilename = jsonFilename.replace('.voxel.json', '.voxel.bin');
    logger.log(`writing '${binFilename}'...`);

    const binarySize = (octree.nodes.length + octree.leafData.length) * 4;
    const buffer = new ArrayBuffer(binarySize);
    const view = new Uint32Array(buffer);
    view.set(octree.nodes, 0);
    view.set(octree.leafData, octree.nodes.length);

    await writeFile(fs, binFilename, new Uint8Array(buffer));
};

/**
 * Voxelizes Gaussian splat data and writes the result as a sparse voxel octree.
 *
 * This function performs GPU-accelerated voxelization of Gaussian splat data
 * and outputs two or three files:
 * - `filename` (.voxel.json) - JSON metadata including bounds, resolution, and array sizes
 * - Corresponding .voxel.bin - Binary octree data (nodes + leafData as Uint32 arrays)
 * - Corresponding .collision.glb - Triangle mesh extracted via marching cubes (GLB format, optional)
 *
 * The binary file layout is:
 * - Bytes 0 to (nodeCount * 4 - 1): nodes array (Uint32, little-endian)
 * - Bytes (nodeCount * 4) to end: leafData array (Uint32, little-endian)
 *
 * @param options - Options including filename, data, and voxelization settings.
 * @param fs - File system for writing output files.
 *
 * @example
 * ```ts
 * import { writeVoxel, MemoryFileSystem } from '@playcanvas/splat-transform';
 *
 * const fs = new MemoryFileSystem();
 * await writeVoxel({
 *     filename: 'scene.voxel.json',
 *     dataTable: myDataTable,
 *     voxelResolution: 0.05,
 *     opacityCutoff: 0.5,
 *     collisionMesh: true,
 *     createDevice: async () => myGraphicsDevice
 * }, fs);
 * ```
 */
const writeVoxel = async (options: WriteVoxelOptions, fs: FileSystem): Promise<void> => {
    const {
        filename,
        dataTable,
        voxelResolution = 0.05,
        opacityCutoff = 0.5,
        createDevice,
        navExteriorRadius,
        navCapsule,
        navSeed,
        collisionMesh = false,
        meshSimplifyError
    } = options;

    if (!createDevice) {
        throw new Error('writeVoxel requires a createDevice function for GPU voxelization');
    }

    if (navCapsule && !navSeed) {
        logger.warn('writeVoxel: navCapsule requires navSeed for interior nav carving, skipping nav carving');
    }
    const hasNavBase = !!(navCapsule && navSeed);
    const hasNav = hasNavBase && navCapsule!.height > 0;
    const exteriorRadius = hasNav ? (navExteriorRadius ?? 1.6) : navExteriorRadius;
    const hasFillExterior = !!(exteriorRadius && navSeed);
    let stepCount = 5;
    if (collisionMesh) stepCount += 2;
    if (hasFillExterior) stepCount += 1;
    if (hasNav) stepCount += 1;
    logger.progress.begin(stepCount);

    // Build a DataTable in engine space containing only the columns needed
    // for voxelization (no SH, so SH rotation cost is never paid).
    const voxelColumns = [
        'x', 'y', 'z',
        'rot_0', 'rot_1', 'rot_2', 'rot_3',
        'scale_0', 'scale_1', 'scale_2',
        'opacity'
    ];
    const missingColumns = voxelColumns.filter(name => !dataTable.hasColumn(name));
    if (missingColumns.length > 0) {
        throw new Error(`writeVoxel: missing required column(s): ${missingColumns.join(', ')}`);
    }
    const delta = computeWriteTransform(dataTable.transform, Transform.IDENTITY);
    const cols = transformColumns(dataTable, voxelColumns, delta);
    const pcDataTable = new DataTable(voxelColumns.map(name => new Column(name, cols.get(name)!)));

    const extentsResult = computeGaussianExtents(pcDataTable);
    const bounds = extentsResult.sceneBounds;

    logger.progress.step('Building BVH');
    logger.debug(`scene extents: (${bounds.min.x.toFixed(2)},${bounds.min.y.toFixed(2)},${bounds.min.z.toFixed(2)}) - (${bounds.max.x.toFixed(2)},${bounds.max.y.toFixed(2)},${bounds.max.z.toFixed(2)})`);

    const bvh = new GaussianBVH(pcDataTable, extentsResult.extents);
    const device = await createDevice();

    const gpuVoxelization = new GpuVoxelization(device);
    gpuVoxelization.uploadAllGaussians(pcDataTable, extentsResult.extents);

    // Align grid bounds to block boundaries BEFORE voxelization so the
    // block coordinates used during voxelization match what the reader expects.
    const blockSize = 4 * voxelResolution;  // Each block is 4x4x4 voxels
    let gridBounds = alignGridBounds(
        bounds.min.x, bounds.min.y, bounds.min.z,
        bounds.max.x, bounds.max.y, bounds.max.z,
        voxelResolution
    );

    logger.progress.step('Voxelizing');

    let buffer = await voxelizeToBuffer(
        bvh, gpuVoxelization, gridBounds, voxelResolution, opacityCutoff
    );

    gpuVoxelization.destroy();

    logger.progress.step('Filtering');
    buffer = filterAndFillBlocks(buffer);

    if (hasFillExterior) {
        logger.progress.step('Fill exterior');
        const fillResult = fillExterior(
            buffer, gridBounds, voxelResolution,
            exteriorRadius!, navSeed!
        );
        buffer = fillResult.buffer;
        gridBounds = fillResult.gridBounds;
    }

    if (hasNav) {
        logger.progress.step('Carve interior');
        const navResult = carveInterior(
            buffer, gridBounds, voxelResolution,
            navCapsule!.height, navCapsule!.radius,
            navSeed!
        );
        buffer = navResult.buffer;
        gridBounds = navResult.gridBounds;
    }

    let glbBytes: Uint8Array | null = null;

    if (collisionMesh) {
        logger.progress.step('Extracting collision mesh');
        const rawMesh = marchingCubes(buffer, gridBounds, voxelResolution);
        logger.log(`collision mesh (raw): ${rawMesh.positions.length / 3} vertices, ${rawMesh.indices.length / 3} triangles`);

        if (rawMesh.indices.length < 3) {
            logger.progress.step('Simplifying collision mesh');
            logger.log('collision mesh: no triangles generated, skipping GLB output');
        } else {
            logger.progress.step('Simplifying collision mesh');
            await MeshoptSimplifier.ready;

            const errorFraction = Number.isFinite(meshSimplifyError) && meshSimplifyError >= 0 ? meshSimplifyError : 0.08;
            const simplifyError = errorFraction * voxelResolution;
            const [simplifiedIndices] = MeshoptSimplifier.simplify(
                rawMesh.indices,
                rawMesh.positions,
                3,
                0,
                simplifyError,
                ['ErrorAbsolute']
            );

            const vertexRemap = new Map<number, number>();
            let newVertexCount = 0;
            for (let i = 0; i < simplifiedIndices.length; i++) {
                if (!vertexRemap.has(simplifiedIndices[i])) {
                    vertexRemap.set(simplifiedIndices[i], newVertexCount++);
                }
            }
            const compactPositions = new Float32Array(newVertexCount * 3);
            for (const [oldIdx, newIdx] of vertexRemap) {
                compactPositions[newIdx * 3] = rawMesh.positions[oldIdx * 3];
                compactPositions[newIdx * 3 + 1] = rawMesh.positions[oldIdx * 3 + 1];
                compactPositions[newIdx * 3 + 2] = rawMesh.positions[oldIdx * 3 + 2];
            }
            const compactIndices = new Uint32Array(simplifiedIndices.length);
            for (let i = 0; i < simplifiedIndices.length; i++) {
                compactIndices[i] = vertexRemap.get(simplifiedIndices[i])!;
            }

            const reduction = (1 - simplifiedIndices.length / rawMesh.indices.length) * 100;
            logger.log(`collision mesh (simplified): ${newVertexCount} vertices, ${simplifiedIndices.length / 3} triangles (${reduction.toFixed(0)}% reduction)`);

            glbBytes = buildCollisionGlb(compactPositions, compactIndices);
        }
    }

    logger.progress.step('Building octree');
    const octree = buildSparseOctree(
        buffer,
        gridBounds,
        bounds,
        voxelResolution
    );
    buffer.clear();

    logger.log(`octree: depth=${octree.treeDepth}, interior=${octree.numInteriorNodes}, mixed=${octree.numMixedLeaves}`);

    logger.progress.step('Writing');
    await writeOctreeFiles(fs, filename, octree);

    if (glbBytes) {
        const glbFilename = filename.replace('.voxel.json', '.collision.glb');
        logger.log(`writing '${glbFilename}'...`);
        await writeFile(fs, glbFilename, glbBytes);
    }

    const totalBytes = (octree.nodes.length + octree.leafData.length) * 4;
    if (glbBytes) {
        logger.log(`total size: octree ${(totalBytes / 1024).toFixed(1)} KB, collision mesh ${(glbBytes.length / 1024).toFixed(1)} KB`);
    } else {
        logger.log(`total size: ${(totalBytes / 1024).toFixed(1)} KB`);
    }
};

export { writeVoxel, type WriteVoxelOptions, type VoxelMetadata };
