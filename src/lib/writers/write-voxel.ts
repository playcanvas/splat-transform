import { basename } from 'pathe';
import { Vec3 } from 'playcanvas';

import { buildCollisionMesh } from './collision-glb';
import { logWrittenFile } from './utils';
import { Column, DataTable, computeGaussianExtents, computeWriteTransform, transformColumns, type Bounds } from '../data-table';
import { GpuVoxelization } from '../gpu';
import { type FileSystem, writeFile } from '../io/write';
import { GaussianBVH } from '../spatial';
import type { DeviceCreator } from '../types';
import { fmtCount, logger, Transform } from '../utils';
import { buildSparseOctree, type SparseOctree } from './sparse-octree';
import {
    filterAndFillBlocks,
    alignGridBounds,
    carve,
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

    /** Opacity threshold for solid voxels - voxels below this are considered empty. Default: 0.1 */
    opacityCutoff?: number;

    /** Optional function to create a GPU device for voxelization */
    createDevice?: DeviceCreator;

    /** Exterior fill radius in world units. Enables exterior fill when set. Requires navSeed; ignored without it. */
    navExteriorRadius?: number;

    /** Capsule dimensions for carve. Height of 0 disables carve. When height > 0, only voxels contactable from the seed are kept. Requires navSeed. */
    navCapsule?: { height: number; radius: number };

    /** Seed position in world space for exterior fill and carve flood fill. */
    navSeed?: NavSeed;

    /** Fill each voxel column upward from the bottom until hitting solid. Runs before carve so the carve's BFS is confined to the actual navigable bubble. Default: false */
    floorFill?: boolean;

    /** When `floorFill` is enabled, dilation radius in world units used to identify "interior" XZ columns to patch. Empty XZ areas larger than `2 * floorFillDilation` from any solid column are treated as exterior and left empty. Default: 0 (patch every empty column). */
    floorFillDilation?: number;

    /** Shape of the collision mesh (.collision.glb). When set, a collision mesh is generated. `edge` = axis-aligned greedy voxel surface, `smooth` = marching cubes followed by lossless coplanar merge. */
    meshType?: 'edge' | 'smooth';
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
 * Crop a voxel buffer to fit the navigable (non-fully-solid) region tightly.
 * Since the runtime treats outside-the-grid as solid, we only need to include
 * blocks that contain at least one empty voxel. Fully-solid blocks beyond the
 * navigable boundary are redundant.
 *
 * @param buffer - Voxelized scene data.
 * @param gridBounds - Axis-aligned bounds of the voxel grid.
 * @param voxelResolution - Size of each voxel in world units.
 * @returns Cropped buffer and grid bounds.
 */
const cropToNavigable = (
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

    const grid = SparseVoxelGrid.fromBuffer(buffer, nx, ny, nz);

    const navBounds = grid.getNavigableBlockBounds();
    if (!navBounds) {
        return { buffer, gridBounds };
    }

    const { minBx, minBy, minBz, maxBx, maxBy, maxBz } = navBounds;
    const cropMaxBx = maxBx + 1;
    const cropMaxBy = maxBy + 1;
    const cropMaxBz = maxBz + 1;

    if (minBx === 0 && minBy === 0 && minBz === 0 &&
        cropMaxBx === nbx && cropMaxBy === nby && cropMaxBz === nbz) {
        return { buffer, gridBounds };
    }

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

    const jsonBytes = (new TextEncoder()).encode(JSON.stringify(metadata, null, 2));
    await writeFile(fs, jsonFilename, jsonBytes);
    logWrittenFile(basename(jsonFilename), jsonBytes.byteLength);

    const binFilename = jsonFilename.replace('.voxel.json', '.voxel.bin');

    const binarySize = (octree.nodes.length + octree.leafData.length) * 4;
    const buffer = new ArrayBuffer(binarySize);
    const view = new Uint32Array(buffer);
    view.set(octree.nodes, 0);
    view.set(octree.leafData, octree.nodes.length);

    await writeFile(fs, binFilename, new Uint8Array(buffer));
    logWrittenFile(basename(binFilename), binarySize);
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
 *     opacityCutoff: 0.1,
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
        opacityCutoff = 0.1,
        createDevice,
        navExteriorRadius,
        floorFill = false,
        floorFillDilation = 0,
        navCapsule,
        navSeed,
        meshType
    } = options;

    if (!createDevice) {
        throw new Error('writeVoxel requires a createDevice function for GPU voxelization');
    }

    if (navCapsule && !navSeed) {
        logger.warn('navCapsule requires navSeed for nav carving, skipping nav carving');
    }
    const hasNav = !!(navCapsule && navSeed && navCapsule.height > 0);
    const hasFillExterior = !!(navExteriorRadius && navSeed);
    const hasFloorFill = floorFill;

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

    const g = logger.group('Build voxels');

    // gpuVoxelization is the only resource not owned by a scope; its
    // destruction is the sole job of the finally below. Open scopes on the
    // error path are reaped by the embedder's logger.error() -> unwindAll.
    let gpuVoxelization: GpuVoxelization | null = null;
    try {
        const bvhSub = logger.group('Building BVH');
        logger.debug(`scene extents: (${bounds.min.x.toFixed(2)},${bounds.min.y.toFixed(2)},${bounds.min.z.toFixed(2)}) - (${bounds.max.x.toFixed(2)},${bounds.max.y.toFixed(2)},${bounds.max.z.toFixed(2)})`);

        const bvh = new GaussianBVH(pcDataTable, extentsResult.extents);
        bvhSub.end();

        const device = await createDevice();
        gpuVoxelization = new GpuVoxelization(device);
        gpuVoxelization.uploadAllGaussians(pcDataTable, extentsResult.extents);

        // Align grid bounds to block boundaries BEFORE voxelization so the
        // block coordinates used during voxelization match what the reader expects.
        // When fillExterior runs, pad by halfExtent + 1 voxels per side so the
        // boundary-face flood seeds survive the dilation (notably below the floor).
        const exteriorPad = hasFillExterior ?
            (Math.ceil(navExteriorRadius! / voxelResolution) + 1) * voxelResolution :
            0;
        let gridBounds = alignGridBounds(
            bounds.min.x - exteriorPad, bounds.min.y - exteriorPad, bounds.min.z - exteriorPad,
            bounds.max.x + exteriorPad, bounds.max.y + exteriorPad, bounds.max.z + exteriorPad,
            voxelResolution
        );

        let buffer = await voxelizeToBuffer(
            bvh, gpuVoxelization, gridBounds, voxelResolution, opacityCutoff
        );

        gpuVoxelization.destroy();
        gpuVoxelization = null;

        const filterSub = logger.group('Filtering');
        buffer = filterAndFillBlocks(buffer);
        filterSub.end();

        if (hasFillExterior) {
            const sub = logger.group('Fill exterior');
            const fillResult = fillExterior(
                buffer, gridBounds, voxelResolution,
                navExteriorRadius!, navSeed!
            );
            buffer = fillResult.buffer;
            gridBounds = fillResult.gridBounds;
            sub.end();
        }

        if (hasFloorFill) {
            const sub = logger.group('Fill floor');
            const floorResult = fillFloor(
                buffer, gridBounds, voxelResolution, floorFillDilation
            );
            buffer = floorResult.buffer;
            gridBounds = floorResult.gridBounds;
            sub.end();
        }

        if (hasNav) {
            const sub = logger.group('Carve');
            const navResult = carve(
                buffer, gridBounds, voxelResolution,
                navCapsule!.height, navCapsule!.radius,
                navSeed!
            );
            buffer = navResult.buffer;
            gridBounds = navResult.gridBounds;
            sub.end();
        }

        const finalCrop = hasFillExterior || hasFloorFill ?
            cropToNavigable(buffer, gridBounds, voxelResolution) :
            cropToOccupied(buffer, gridBounds, voxelResolution);
        buffer = finalCrop.buffer;
        gridBounds = finalCrop.gridBounds;

        const glbBytes = meshType ?
            buildCollisionMesh(buffer, gridBounds, voxelResolution, meshType) :
            null;

        const octree = buildSparseOctree(
            buffer,
            gridBounds,
            bounds,
            voxelResolution
        );
        buffer.clear();

        logger.info(`octree depth: ${octree.treeDepth}`);
        logger.info(`interior nodes: ${fmtCount(octree.numInteriorNodes)}`);
        logger.info(`mixed leaves: ${fmtCount(octree.numMixedLeaves)}`);

        const writingSub = logger.group('Writing');
        await writeOctreeFiles(fs, filename, octree);

        if (glbBytes) {
            const glbFilename = filename.replace('.voxel.json', '.collision.glb');
            await writeFile(fs, glbFilename, glbBytes);
            logWrittenFile(basename(glbFilename), glbBytes.length);
        }
        writingSub.end();

        g.end();
    } finally {
        gpuVoxelization?.destroy();
    }
};

export { writeVoxel, type WriteVoxelOptions, type VoxelMetadata };
