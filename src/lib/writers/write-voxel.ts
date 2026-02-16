import { dirname, resolve } from 'pathe';
import { GraphicsDevice } from 'playcanvas';

import { DataTable } from '../data-table/data-table';
import { type FileSystem, writeFile } from '../io/write';
import { logger } from '../utils/logger';
import {
    computeGaussianExtents,
    GaussianBVH,
    GpuVoxelization,
    buildSparseOctree,
    alignGridBounds,
    type BatchSpec,
    type MultiBatchResult
} from '../voxel/index';
import {
    BlockAccumulator,
    xyzToMorton,
    type SparseOctree
} from '../voxel/sparse-octree';

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
        version: '1.0',
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

    logger.progress.begin(6);

    // Phase 1: Compute Gaussian extents
    logger.progress.step('Computing Gaussian extents');
    const extentsResult = computeGaussianExtents(dataTable);
    const bounds = extentsResult.sceneBounds;

    // Phase 2: Build BVH
    logger.progress.step('Building BVH');
    const bvh = new GaussianBVH(dataTable, extentsResult.extents);

    // Phase 3: Create GPU device and voxelization
    logger.progress.step('Creating GPU device');
    const device = await createDevice();

    const gpuVoxelization = new GpuVoxelization(device);
    gpuVoxelization.uploadAllGaussians(dataTable, extentsResult.extents);

    // Align grid bounds to block boundaries BEFORE voxelization so the
    // block coordinates used during voxelization match what the reader expects.
    const blockSize = 4 * voxelResolution;  // Each block is 4x4x4 voxels
    const gridBounds = alignGridBounds(
        bounds.min.x, bounds.min.y, bounds.min.z,
        bounds.max.x, bounds.max.y, bounds.max.z,
        voxelResolution
    );

    const numBlocksX = Math.round((gridBounds.max.x - gridBounds.min.x) / blockSize);
    const numBlocksY = Math.round((gridBounds.max.y - gridBounds.min.y) / blockSize);
    const numBlocksZ = Math.round((gridBounds.max.z - gridBounds.min.z) / blockSize);

    logger.log(`grid: ${numBlocksX} x ${numBlocksY} x ${numBlocksZ} blocks`);

    // Phase 4: Double-buffered pipelined voxelization
    // Uses two GPU dispatch slots so the CPU can prepare the next mega-dispatch
    // (BVH queries + index copying) while the GPU executes the current one.
    const accumulator = new BlockAccumulator();
    const batchSize = 16;  // 16x16x16 = 4096 blocks max per batch

    logger.progress.step('Voxelizing');

    // Mega-dispatch thresholds: flush when either limit is reached
    const MEGA_MAX_BATCHES = 512;
    const MEGA_MAX_INDICES = 4 * 1024 * 1024;  // 4M indices

    const maxBlocks = GpuVoxelization.MAX_BLOCKS_PER_BATCH;
    const numSlots = GpuVoxelization.NUM_SLOTS;

    // Batch collection state — per-slot for double buffering
    interface PendingBatch extends BatchSpec {
        bx: number;  // absolute block start X (for Morton codes)
        by: number;
        bz: number;
        numBlocksX: number;
        numBlocksY: number;
        numBlocksZ: number;
    }

    // Per-slot CPU-side index arrays
    const slotIndexArrays: Uint32Array[] = [];
    const slotCapacities: number[] = [];
    for (let i = 0; i < numSlots; i++) {
        slotCapacities.push(1024 * 1024);
        slotIndexArrays.push(new Uint32Array(1024 * 1024));
    }

    let currentSlot = 0;
    let indexOffset = 0;
    const pendingBatches: PendingBatch[] = [];
    let megaDispatchCount = 0;
    let skippedEmpty = 0;

    // Inflight dispatch from previous flush (for pipelining)
    let inflight: {
        resultPromise: Promise<MultiBatchResult>;
        batches: PendingBatch[];
    } | null = null;

    /**
     * Process GPU results back into the block accumulator.
     */
    const processResults = (masks: Uint32Array, batches: PendingBatch[]): void => {
        for (let b = 0; b < batches.length; b++) {
            const batch = batches[b];
            const batchResultOffset = b * maxBlocks * 2;
            const totalBatchBlocks = batch.numBlocksX * batch.numBlocksY * batch.numBlocksZ;

            for (let blockIdx = 0; blockIdx < totalBatchBlocks; blockIdx++) {
                const maskLo = masks[batchResultOffset + blockIdx * 2];
                const maskHi = masks[batchResultOffset + blockIdx * 2 + 1];

                // Skip empty blocks — vast majority are empty
                if (maskLo === 0 && maskHi === 0) continue;

                // Compute block coordinates directly from flat index
                const localX = blockIdx % batch.numBlocksX;
                const localY = (blockIdx / batch.numBlocksX | 0) % batch.numBlocksY;
                const localZ = (blockIdx / (batch.numBlocksX * batch.numBlocksY)) | 0;

                const absBlockX = batch.bx + localX;
                const absBlockY = batch.by + localY;
                const absBlockZ = batch.bz + localZ;

                const morton = xyzToMorton(absBlockX, absBlockY, absBlockZ);
                accumulator.addBlock(morton, maskLo, maskHi);
            }
        }
    };

    /**
     * Submit pending batches as a GPU mega-dispatch and set up pipelining.
     * The GPU dispatch is fire-and-forget — the result promise is stored
     * in `inflight` and awaited on the NEXT flush, allowing the CPU to
     * prepare more batches while the GPU is busy.
     */
    const flushPendingBatches = async (): Promise<void> => {
        if (pendingBatches.length === 0) return;

        // Capture current batch state before clearing
        const batchesToSubmit = pendingBatches.slice();
        const submitSlot = currentSlot;
        const submitIndexArray = slotIndexArrays[submitSlot];
        const submitIndexCount = indexOffset;

        // Switch to the other slot for collecting the next round
        currentSlot = (currentSlot + 1) % numSlots;
        pendingBatches.length = 0;
        indexOffset = 0;

        // Submit new dispatch FIRST so GPU starts working immediately
        const resultPromise = gpuVoxelization.submitMultiBatch(
            submitSlot,
            submitIndexArray,
            submitIndexCount,
            batchesToSubmit,
            voxelResolution,
            opacityCutoff
        );

        // THEN await and process the previous inflight dispatch
        // (the GPU was computing it while the CPU prepared this batch)
        if (inflight) {
            const result = await inflight.resultPromise;
            processResults(result.masks, inflight.batches);
            megaDispatchCount++;
        }

        inflight = { resultPromise, batches: batchesToSubmit };
    };

    // Inner progress for voxelization batches
    const numZBatches = Math.max(1, Math.ceil(numBlocksZ / batchSize));
    logger.progress.begin(10);
    let lastVoxelStep = 0;
    let totalBatches = 0;

    // Process the entire scene, collecting batches for multi-dispatch
    for (let bz = 0; bz < numBlocksZ; bz += batchSize) {
        // Report inner progress scaled to 10 steps
        const currentVoxelStep = Math.min(10, Math.floor(((bz / batchSize) + 1) / numZBatches * 10));
        while (lastVoxelStep < currentVoxelStep) {
            logger.progress.step();
            lastVoxelStep++;
        }

        for (let by = 0; by < numBlocksY; by += batchSize) {
            for (let bx = 0; bx < numBlocksX; bx += batchSize) {
                const currBatchX = Math.min(batchSize, numBlocksX - bx);
                const currBatchY = Math.min(batchSize, numBlocksY - by);
                const currBatchZ = Math.min(batchSize, numBlocksZ - bz);

                const blockMinX = gridBounds.min.x + bx * blockSize;
                const blockMinY = gridBounds.min.y + by * blockSize;
                const blockMinZ = gridBounds.min.z + bz * blockSize;
                const blockMaxX = blockMinX + currBatchX * blockSize;
                const blockMaxY = blockMinY + currBatchY * blockSize;
                const blockMaxZ = blockMinZ + currBatchZ * blockSize;

                // Query BVH for overlapping Gaussians
                const overlapping = bvh.queryOverlappingRaw(
                    blockMinX, blockMinY, blockMinZ,
                    blockMaxX, blockMaxY, blockMaxZ
                );

                if (overlapping.length === 0) {
                    skippedEmpty++;
                    continue;
                }

                // Ensure current slot's index array has enough capacity
                const needed = indexOffset + overlapping.length;
                if (needed > slotCapacities[currentSlot]) {
                    slotCapacities[currentSlot] = Math.max(slotCapacities[currentSlot] * 2, needed);
                    const newArray = new Uint32Array(slotCapacities[currentSlot]);
                    newArray.set(slotIndexArrays[currentSlot].subarray(0, indexOffset));
                    slotIndexArrays[currentSlot] = newArray;
                }

                // Copy overlapping Gaussian indices into current slot's array
                slotIndexArrays[currentSlot].set(overlapping, indexOffset);

                // Record batch metadata
                pendingBatches.push({
                    indexOffset,
                    indexCount: overlapping.length,
                    blockMin: { x: blockMinX, y: blockMinY, z: blockMinZ },
                    numBlocksX: currBatchX,
                    numBlocksY: currBatchY,
                    numBlocksZ: currBatchZ,
                    bx, by, bz
                });

                indexOffset += overlapping.length;
                totalBatches++;

                // Flush when mega-dispatch thresholds are reached
                if (pendingBatches.length >= MEGA_MAX_BATCHES || indexOffset >= MEGA_MAX_INDICES) {
                    await flushPendingBatches();
                }
            }
        }
    }

    // Flush remaining pending batches
    await flushPendingBatches();

    // Await and process the final inflight dispatch
    if (inflight) {
        const result = await inflight.resultPromise;
        processResults(result.masks, inflight.batches);
        megaDispatchCount++;
        inflight = null;
    }

    // Flush remaining inner voxelization progress steps
    while (lastVoxelStep < 10) {
        logger.progress.step();
        lastVoxelStep++;
    }

    // Cleanup GPU resources (device lifecycle managed by caller)
    gpuVoxelization.destroy();

    logger.log(`voxelization complete: ${accumulator.count} non-empty blocks`);
    logger.log(`pipelined: ${totalBatches} batches in ${megaDispatchCount} mega-dispatches, ${skippedEmpty} empty skips`);

    // Phase 5: Build sparse octree
    logger.progress.step('Building sparse octree');

    // Build the sparse octree (gridBounds already computed before voxelization)
    const octree = buildSparseOctree(
        accumulator,
        gridBounds,
        bounds,  // Original scene bounds
        voxelResolution
    );

    logger.log(`octree: depth=${octree.treeDepth}, interior=${octree.numInteriorNodes}, mixed=${octree.numMixedLeaves}`);

    // Phase 6: Write output files
    logger.progress.step('Writing output');
    await writeOctreeFiles(fs, filename, octree);

    const totalBytes = (octree.nodes.length + octree.leafData.length) * 4;
    logger.log(`total size: ${(totalBytes / 1024).toFixed(1)} KB`);
};

export { writeVoxel, type WriteVoxelOptions, type VoxelMetadata, type DeviceCreator };
