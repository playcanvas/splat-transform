import { Vec3 } from 'playcanvas';

import { buildCollisionMesh } from './collision-glb';
import { Column, DataTable, computeGaussianExtents, computeWriteTransform, transformColumns, type Bounds } from '../data-table';
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
    fillFloor,
    type NavSeed,
    voxelizeToBuffer
} from '../voxel';
import { BlockMaskBuffer } from '../voxel/block-mask-buffer';
import { mortonToXYZ } from '../voxel/morton';
import { SparseVoxelGrid } from '../voxel/sparse-voxel-grid';

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

    /** Exterior fill radius in world units. Enables exterior fill when set. Requires navSeed; ignored without it. */
    navExteriorRadius?: number;

    /** Capsule dimensions for interior carve. Height of 0 disables interior carve. When height > 0, only voxels contactable from the seed are kept. Requires navSeed. */
    navCapsule?: { height: number; radius: number };

    /** Seed position in world space for exterior fill and interior carve flood fill. */
    navSeed?: NavSeed;

    /** Fill each voxel column upward from the bottom until hitting solid. Runs after interior carve. Default: false */
    floorFill?: boolean;

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
 * Crop a voxel buffer and its grid bounds to the occupied block range.
 * Removes empty padding that arises from Gaussian 3-sigma extents being
 * much larger than the actual solid voxel footprint.
 *
 * @param buffer - Voxelized scene data.
 * @param gridBounds - Axis-aligned bounds of the voxel grid.
 * @param voxelResolution - Size of each voxel in world units.
 * @returns Cropped buffer and grid bounds.
 */
const cropToOccupied = (
    buffer: BlockMaskBuffer,
    gridBounds: Bounds,
    voxelResolution: number
): { buffer: BlockMaskBuffer; gridBounds: Bounds } => {
    if (buffer.count === 0) {
        return { buffer, gridBounds };
    }

    const nx = Math.round((gridBounds.max.x - gridBounds.min.x) / voxelResolution);
    const ny = Math.round((gridBounds.max.y - gridBounds.min.y) / voxelResolution);
    const nz = Math.round((gridBounds.max.z - gridBounds.min.z) / voxelResolution);
    const nbx = nx >> 2;
    const nby = ny >> 2;
    const nbz = nz >> 2;

    let minBx = nbx, minBy = nby, minBz = nbz;
    let maxBx = 0, maxBy = 0, maxBz = 0;

    const scanMorton = (morton: number) => {
        const [bx, by, bz] = mortonToXYZ(morton);
        if (bx < minBx) minBx = bx;
        if (by < minBy) minBy = by;
        if (bz < minBz) minBz = bz;
        if (bx > maxBx) maxBx = bx;
        if (by > maxBy) maxBy = by;
        if (bz > maxBz) maxBz = bz;
    };

    const solidMortons = buffer.getSolidBlocks();
    for (let i = 0; i < solidMortons.length; i++) scanMorton(solidMortons[i]);

    const mixed = buffer.getMixedBlocks();
    for (let i = 0; i < mixed.morton.length; i++) scanMorton(mixed.morton[i]);

    if (minBx > maxBx) {
        return { buffer, gridBounds };
    }

    const cropMaxBx = maxBx + 1;
    const cropMaxBy = maxBy + 1;
    const cropMaxBz = maxBz + 1;

    if (minBx === 0 && minBy === 0 && minBz === 0 &&
        cropMaxBx === nbx && cropMaxBy === nby && cropMaxBz === nbz) {
        return { buffer, gridBounds };
    }

    const grid = SparseVoxelGrid.fromBuffer(buffer, nx, ny, nz);
    const croppedBuffer = grid.toBuffer(minBx, minBy, minBz, cropMaxBx, cropMaxBy, cropMaxBz);

    const blockSize = 4 * voxelResolution;
    const croppedMin = new Vec3(
        gridBounds.min.x + minBx * blockSize,
        gridBounds.min.y + minBy * blockSize,
        gridBounds.min.z + minBz * blockSize
    );
    const croppedBounds: Bounds = {
        min: croppedMin,
        max: new Vec3(
            croppedMin.x + (cropMaxBx - minBx) * blockSize,
            croppedMin.y + (cropMaxBy - minBy) * blockSize,
            croppedMin.z + (cropMaxBz - minBz) * blockSize
        )
    };

    return { buffer: croppedBuffer, gridBounds: croppedBounds };
};

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
        floorFill = false,
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
    const hasNav = !!(navCapsule && navSeed && navCapsule.height > 0);
    const hasFillExterior = !!(navExteriorRadius && navSeed);
    const hasFloorFill = floorFill;
    let stepCount = 5;
    if (collisionMesh) stepCount += 2;
    if (hasFillExterior) stepCount += 1;
    if (hasFloorFill) stepCount += 1;
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

    let gpuVoxelization: GpuVoxelization | null = new GpuVoxelization(device);
    let progressComplete = false;
    try {
        gpuVoxelization.uploadAllGaussians(pcDataTable, extentsResult.extents);

        // Align grid bounds to block boundaries BEFORE voxelization so the
        // block coordinates used during voxelization match what the reader expects.
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
        gpuVoxelization = null;

        logger.progress.step('Filtering');
        // buffer = filterAndFillBlocks(buffer);

        const cropResult = cropToOccupied(buffer, gridBounds, voxelResolution);
        buffer = cropResult.buffer;
        gridBounds = cropResult.gridBounds;

        if (hasFillExterior) {
            logger.progress.step('Fill exterior');
            const fillResult = fillExterior(
                buffer, gridBounds, voxelResolution,
                navExteriorRadius!, navSeed!
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

        if (hasFloorFill) {
            logger.progress.step('Fill floor');
            const floorResult = fillFloor(
                buffer, gridBounds, voxelResolution
            );
            buffer = floorResult.buffer;
            gridBounds = floorResult.gridBounds;
        }

        const glbBytes = collisionMesh ?
            await buildCollisionMesh(buffer, gridBounds, voxelResolution, meshSimplifyError) :
            null;

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

        progressComplete = true;
    } catch (e) {
        gpuVoxelization?.destroy();
        throw e;
    } finally {
        if (!progressComplete) {
            logger.progress.cancel();
        }
    }
};

export { writeVoxel, writeOctreeFiles, type WriteVoxelOptions, type VoxelMetadata };
