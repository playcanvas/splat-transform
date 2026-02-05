import { dirname, resolve } from 'pathe';
import { GraphicsDevice } from 'playcanvas';

import { DataTable } from '../data-table/data-table';
import { type FileSystem, writeFile } from '../io/write';
import { logger } from '../utils/logger';
import {
    computeGaussianExtents,
    GaussianBVH,
    GpuVoxelization,
    BlockAccumulator,
    buildSparseOctree,
    alignGridBounds,
    xyzToMorton,
    type SparseOctree
} from '../voxel/index';

/**
 * A function that creates a PlayCanvas GraphicsDevice on demand.
 *
 * Used for GPU-accelerated voxelization.
 * The application is responsible for caching if needed.
 *
 * @returns Promise resolving to a GraphicsDevice instance.
 */
type DeviceCreator = () => Promise<GraphicsDevice>;

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
};

/**
 * Metadata for a voxel octree file.
 */
interface VoxelMetadata {
    /** File format version */
    version: string;

    /** Grid bounds aligned to 4x4x4 block boundaries */
    gridBounds: { min: number[]; max: number[] };

    /** Original Gaussian scene bounds */
    gaussianBounds: { min: number[]; max: number[] };

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
 */
const writeOctreeFiles = async (
    fs: FileSystem,
    jsonFilename: string,
    octree: SparseOctree
): Promise<void> => {
    // Build metadata object
    const metadata: VoxelMetadata = {
        version: '1.0',
        gridBounds: {
            min: [octree.gridBounds.min.x, octree.gridBounds.min.y, octree.gridBounds.min.z],
            max: [octree.gridBounds.max.x, octree.gridBounds.max.y, octree.gridBounds.max.z]
        },
        gaussianBounds: {
            min: [octree.gaussianBounds.min.x, octree.gaussianBounds.min.y, octree.gaussianBounds.min.z],
            max: [octree.gaussianBounds.max.x, octree.gaussianBounds.max.y, octree.gaussianBounds.max.z]
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
 * and outputs two files:
 * - `filename` (.voxel.json) - JSON metadata including bounds, resolution, and array sizes
 * - Corresponding .voxel.bin - Binary octree data (nodes + leafData as Uint32 arrays)
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
        createDevice
    } = options;

    if (!createDevice) {
        throw new Error('writeVoxel requires a createDevice function for GPU voxelization');
    }

    logger.log(`voxelizing scene (resolution: ${voxelResolution}, opacity cutoff: ${opacityCutoff})...`);

    // Phase 1: Compute Gaussian extents
    logger.log('computing Gaussian extents...');
    const extentsResult = computeGaussianExtents(dataTable);
    const bounds = extentsResult.sceneBounds;

    // Phase 2: Build BVH
    logger.log('building BVH...');
    const bvh = new GaussianBVH(dataTable, extentsResult.extents);

    // Phase 3: Create GPU device and voxelization
    logger.log('creating GPU device...');
    const device = await createDevice();

    const gpuVoxelization = new GpuVoxelization(device);
    gpuVoxelization.uploadAllGaussians(dataTable, extentsResult.extents);

    // Calculate grid dimensions
    const blockSize = 4 * voxelResolution;  // Each block is 4x4x4 voxels
    const numBlocksX = Math.ceil((bounds.max.x - bounds.min.x) / blockSize);
    const numBlocksY = Math.ceil((bounds.max.y - bounds.min.y) / blockSize);
    const numBlocksZ = Math.ceil((bounds.max.z - bounds.min.z) / blockSize);

    logger.log(`grid: ${numBlocksX} x ${numBlocksY} x ${numBlocksZ} blocks`);

    // Phase 4: Voxelization with BlockAccumulator
    const accumulator = new BlockAccumulator();
    const batchSize = 16;  // 16x16x16 = 4096 blocks max per batch

    logger.log('voxelizing...');

    // Process the entire scene in batches
    for (let bz = 0; bz < numBlocksZ; bz += batchSize) {
        for (let by = 0; by < numBlocksY; by += batchSize) {
            for (let bx = 0; bx < numBlocksX; bx += batchSize) {
                const currBatchX = Math.min(batchSize, numBlocksX - bx);
                const currBatchY = Math.min(batchSize, numBlocksY - by);
                const currBatchZ = Math.min(batchSize, numBlocksZ - bz);

                const blockMinX = bounds.min.x + bx * blockSize;
                const blockMinY = bounds.min.y + by * blockSize;
                const blockMinZ = bounds.min.z + bz * blockSize;
                const blockMaxX = blockMinX + currBatchX * blockSize;
                const blockMaxY = blockMinY + currBatchY * blockSize;
                const blockMaxZ = blockMinZ + currBatchZ * blockSize;

                // Query BVH for overlapping Gaussians
                const overlapping = bvh.queryOverlappingRaw(
                    blockMinX, blockMinY, blockMinZ,
                    blockMaxX, blockMaxY, blockMaxZ
                );

                if (overlapping.length === 0) {
                    // Empty batch - skip GPU work
                    continue;
                }

                // Run GPU voxelization
                const result = await gpuVoxelization.voxelizeBlocks(
                    overlapping,
                    { x: blockMinX, y: blockMinY, z: blockMinZ },
                    currBatchX, currBatchY, currBatchZ,
                    voxelResolution,
                    opacityCutoff
                );

                // Accumulate blocks with Morton codes
                for (let blockIdx = 0; blockIdx < result.blocks.length; blockIdx++) {
                    const block = result.blocks[blockIdx];
                    const maskLo = result.masks[blockIdx * 2];
                    const maskHi = result.masks[blockIdx * 2 + 1];

                    // Calculate absolute block coordinates
                    const absBlockX = bx + block.x;
                    const absBlockY = by + block.y;
                    const absBlockZ = bz + block.z;

                    // Convert to Morton code and accumulate
                    const morton = xyzToMorton(absBlockX, absBlockY, absBlockZ);
                    accumulator.addBlock(morton, maskLo, maskHi);
                }
            }
        }
    }

    // Cleanup GPU resources (device lifecycle managed by caller)
    gpuVoxelization.destroy();

    logger.log(`voxelization complete: ${accumulator.count} non-empty blocks`);

    // Phase 5: Build sparse octree
    logger.log('building sparse octree...');

    // Align grid bounds to block boundaries
    const gridBounds = alignGridBounds(
        bounds.min.x, bounds.min.y, bounds.min.z,
        bounds.max.x, bounds.max.y, bounds.max.z,
        voxelResolution
    );

    // Build the sparse octree
    const octree = buildSparseOctree(
        accumulator,
        gridBounds,
        bounds,  // Original Gaussian bounds
        voxelResolution
    );

    logger.log(`octree: depth=${octree.treeDepth}, interior=${octree.numInteriorNodes}, mixed=${octree.numMixedLeaves}`);

    // Phase 6: Write output files
    await writeOctreeFiles(fs, filename, octree);

    const totalBytes = (octree.nodes.length + octree.leafData.length) * 4;
    logger.log(`total size: ${(totalBytes / 1024).toFixed(1)} KB`);
};

export { writeVoxel, type WriteVoxelOptions, type VoxelMetadata, type DeviceCreator };
