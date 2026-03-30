import { MeshoptSimplifier } from 'meshoptimizer/simplifier';
import { Quat, Vec3 } from 'playcanvas';

import type { DeviceCreator } from './write-sog';
import { DataTable } from '../data-table/data-table';
import { transform } from '../data-table/transform';
import { type FileSystem, writeFile } from '../io/write';
import { logger } from '../utils/logger';
import { buildCollisionGlb } from '../voxel/collision-glb';
import {
    computeGaussianExtents,
    GaussianBVH,
    GpuVoxelization,
    buildSparseOctree,
    alignGridBounds,
    type BatchSpec,
    type MultiBatchResult
} from '../voxel/index';
import { marchingCubes } from '../voxel/marching-cubes';
import { fillExterior, simplifyForCapsule, type NavSeed } from '../voxel/nav-simplify';
import {
    BlockAccumulator,
    xyzToMorton,
    type SparseOctree
} from '../voxel/sparse-octree';
import { filterAndFillBlocks } from '../voxel/voxel-filter';

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

    /** Whether to generate a collision mesh (.collision.glb) alongside the voxel data. Default: false */
    collisionMesh?: boolean;

    /** Maximum geometric error for collision mesh simplification as a fraction of voxelResolution. Default: 0.08 */
    meshSimplifyError?: number;

    /** Capsule dimensions for navigation simplification. When set, only voxels contactable from the seed are kept. */
    navCapsule?: { height: number; radius: number };

    /** Seed position in world space for navigation flood fill. Required when navCapsule is set. */
    navSeed?: NavSeed;

    /** Dilation distance in world units for exterior void filling. Defaults to 1.6 when nav simplification is active. */
    navFillDilation?: number;

    /** Stop nav processing at this stage and output intermediate state. 1-5: fillExterior, 6-10: simplifyForCapsule. */
    navDebugStage?: number;
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
        collisionMesh = false,
        meshSimplifyError,
        navCapsule,
        navSeed,
        navFillDilation,
        navDebugStage
    } = options;

    if (!createDevice) {
        throw new Error('writeVoxel requires a createDevice function for GPU voxelization');
    }

    if ((navCapsule && !navSeed) || (!navCapsule && navSeed)) {
        logger.warn('writeVoxel: both navCapsule and navSeed must be provided for nav simplification, skipping');
    }
    const fillDebugStage = navDebugStage && navDebugStage <= 5 ? navDebugStage : undefined;
    const simplifyDebugStage = navDebugStage && navDebugStage >= 6 ? navDebugStage : undefined;
    const hasNavBase = !!(navCapsule && navSeed);
    const fillDilation = hasNavBase ? (navFillDilation ?? 1.6) : navFillDilation;
    const hasFillExterior = !!(fillDilation && navSeed);
    const hasNav = hasNavBase && !fillDebugStage;
    let stepCount = 5;
    if (collisionMesh) stepCount += 2;
    if (hasFillExterior) stepCount += 1;
    if (hasNav) stepCount += 1;
    logger.progress.begin(stepCount);

    // Build a copy containing only the columns needed for voxelization and
    // transform it by (0, 0, 180) to orient data into PlayCanvas space.
    // Using a copy avoids mutating the caller's DataTable and keeps SH out
    // of the transform since those columns aren't present.
    const pcDataTable = dataTable.clone({
        columns: [
            'x', 'y', 'z',
            'rot_0', 'rot_1', 'rot_2', 'rot_3',
            'scale_0', 'scale_1', 'scale_2',
            'opacity'
        ]
    });
    transform(pcDataTable, Vec3.ZERO, new Quat().setFromEulerAngles(0, 0, 180), 1);

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

    const numBlocksX = Math.round((gridBounds.max.x - gridBounds.min.x) / blockSize);
    const numBlocksY = Math.round((gridBounds.max.y - gridBounds.min.y) / blockSize);
    const numBlocksZ = Math.round((gridBounds.max.z - gridBounds.min.z) / blockSize);

    // Phase 4: Double-buffered pipelined voxelization
    // Uses two GPU dispatch slots so the CPU can prepare the next mega-dispatch
    // (BVH queries + index copying) while the GPU executes the current one.
    let accumulator = new BlockAccumulator();
    const batchSize = 16;  // 16x16x16 = 4096 blocks max per batch

    logger.progress.step('Voxelizing');
    logger.debug(`blocks: ${numBlocksX} x ${numBlocksY} x ${numBlocksZ} (${(numBlocksX * numBlocksY * numBlocksZ / 1e6).toFixed(1)}M)`);
    logger.debug(`voxels: ${numBlocksX * 4} x ${numBlocksY * 4} x ${numBlocksZ * 4} (${(numBlocksX * numBlocksY * numBlocksZ * 64 / 1e9).toFixed(2)}B)`);

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
     *
     * @param masks - Raw Uint32 voxel masks from the GPU readback.
     * @param batches - Batch metadata used to decode block positions.
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
                    bx,
                    by,
                    bz
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

    logger.progress.step('Filtering');
    accumulator = filterAndFillBlocks(accumulator);

    if (hasFillExterior) {
        logger.progress.step('Fill exterior');
        const fillResult = fillExterior(
            accumulator, gridBounds, voxelResolution,
            fillDilation!, navSeed!,
            fillDebugStage
        );
        accumulator = fillResult.accumulator;
        gridBounds = fillResult.gridBounds;
    }

    if (hasNav) {
        logger.progress.step('Carve interior');
        const navResult = simplifyForCapsule(
            accumulator, gridBounds, voxelResolution,
            navCapsule!.height, navCapsule!.radius,
            navSeed!,
            simplifyDebugStage
        );
        accumulator = navResult.accumulator;
        gridBounds = navResult.gridBounds;
    }

    let glbBytes: Uint8Array | null = null;

    if (collisionMesh) {
        logger.progress.step('Extracting collision mesh');
        const rawMesh = marchingCubes(accumulator, gridBounds, voxelResolution);
        logger.log(`collision mesh (raw): ${rawMesh.positions.length / 3} vertices, ${rawMesh.indices.length / 3} triangles`);

        if (rawMesh.indices.length < 3) {
            logger.progress.step('Simplifying collision mesh');
            logger.log('collision mesh: no triangles generated, skipping GLB output');
        } else {
            logger.progress.step('Simplifying collision mesh');
            await MeshoptSimplifier.ready;

            const simplifyError = (meshSimplifyError ?? 0.08) * voxelResolution;
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
        accumulator,
        gridBounds,
        bounds,
        voxelResolution
    );
    accumulator.clear();

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
