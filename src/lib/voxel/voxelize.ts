import { Vec3 } from 'playcanvas';

import { BlockMaskBuffer } from './block-mask-buffer';
import { xyzToMorton } from './morton';
import type { Bounds } from '../data-table';
import {
    GpuVoxelization,
    type BatchSpec,
    type MultiBatchResult
} from '../gpu';
import { type GaussianBVH } from '../spatial';
import { logger } from '../utils';

interface PendingBatch extends BatchSpec {
    bx: number;
    by: number;
    bz: number;
    numBlocksX: number;
    numBlocksY: number;
    numBlocksZ: number;
}

/**
 * GPU-accelerated voxelization of Gaussian splat data into a block mask buffer.
 *
 * Uses double-buffered pipelining: the CPU prepares the next mega-dispatch
 * (BVH queries + index copying) while the GPU executes the current one.
 *
 * @param bvh - Gaussian bounding volume hierarchy for spatial queries.
 * @param gpuVoxelization - GPU voxelization pipeline with Gaussians already uploaded.
 * @param gridBounds - Block-aligned grid bounds to voxelize within.
 * @param voxelResolution - Size of each voxel in world units.
 * @param opacityCutoff - Opacity threshold for solid voxels.
 * @returns Block mask buffer containing voxelization results.
 */
const voxelizeToBuffer = async (
    bvh: GaussianBVH,
    gpuVoxelization: GpuVoxelization,
    gridBounds: Bounds,
    voxelResolution: number,
    opacityCutoff: number
): Promise<BlockMaskBuffer> => {
    const blockSize = 4 * voxelResolution;
    const numBlocksX = Math.round((gridBounds.max.x - gridBounds.min.x) / blockSize);
    const numBlocksY = Math.round((gridBounds.max.y - gridBounds.min.y) / blockSize);
    const numBlocksZ = Math.round((gridBounds.max.z - gridBounds.min.z) / blockSize);

    const buffer = new BlockMaskBuffer();
    const batchSize = 16;

    const MEGA_MAX_BATCHES = 512;
    const MEGA_MAX_INDICES = 4 * 1024 * 1024;

    const maxBlocks = GpuVoxelization.MAX_BLOCKS_PER_BATCH;
    const numSlots = GpuVoxelization.NUM_SLOTS;

    const slotIndexArrays: Uint32Array[] = [];
    const slotCapacities: number[] = [];
    for (let i = 0; i < numSlots; i++) {
        slotCapacities.push(1024 * 1024);
        slotIndexArrays.push(new Uint32Array(1024 * 1024));
    }

    let currentSlot = 0;
    let indexOffset = 0;
    const pendingBatches: PendingBatch[] = [];

    let inflight: {
        resultPromise: Promise<MultiBatchResult>;
        batches: PendingBatch[];
    } | null = null;

    const processResults = (masks: Uint32Array, batches: PendingBatch[]): void => {
        for (let b = 0; b < batches.length; b++) {
            const batch = batches[b];
            const batchResultOffset = b * maxBlocks * 2;
            const totalBatchBlocks = batch.numBlocksX * batch.numBlocksY * batch.numBlocksZ;

            for (let blockIdx = 0; blockIdx < totalBatchBlocks; blockIdx++) {
                const maskLo = masks[batchResultOffset + blockIdx * 2];
                const maskHi = masks[batchResultOffset + blockIdx * 2 + 1];

                if (maskLo === 0 && maskHi === 0) continue;

                const localX = blockIdx % batch.numBlocksX;
                const localY = (blockIdx / batch.numBlocksX | 0) % batch.numBlocksY;
                const localZ = (blockIdx / (batch.numBlocksX * batch.numBlocksY)) | 0;

                const absBlockX = batch.bx + localX;
                const absBlockY = batch.by + localY;
                const absBlockZ = batch.bz + localZ;

                const morton = xyzToMorton(absBlockX, absBlockY, absBlockZ);
                buffer.addBlock(morton, maskLo, maskHi);
            }
        }
    };

    const flushPendingBatches = async (): Promise<void> => {
        if (pendingBatches.length === 0) return;

        const batchesToSubmit = pendingBatches.slice();
        const submitSlot = currentSlot;
        const submitIndexArray = slotIndexArrays[submitSlot];
        const submitIndexCount = indexOffset;

        currentSlot = (currentSlot + 1) % numSlots;
        pendingBatches.length = 0;
        indexOffset = 0;

        const resultPromise = gpuVoxelization.submitMultiBatch(
            submitSlot,
            submitIndexArray,
            submitIndexCount,
            batchesToSubmit,
            voxelResolution,
            opacityCutoff
        );

        if (inflight) {
            const result = await inflight.resultPromise;
            processResults(result.masks, inflight.batches);
        }

        inflight = { resultPromise, batches: batchesToSubmit };
    };

    const numZBatches = Math.max(1, Math.ceil(numBlocksZ / batchSize));
    const bar = logger.bar('Voxelizing', numZBatches);
    for (let bz = 0; bz < numBlocksZ; bz += batchSize) {
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

                const overlapping = bvh.queryOverlappingRaw(
                    blockMinX, blockMinY, blockMinZ,
                    blockMaxX, blockMaxY, blockMaxZ
                );

                if (overlapping.length === 0) {
                    continue;
                }

                const needed = indexOffset + overlapping.length;
                if (needed > slotCapacities[currentSlot]) {
                    slotCapacities[currentSlot] = Math.max(slotCapacities[currentSlot] * 2, needed);
                    const newArray = new Uint32Array(slotCapacities[currentSlot]);
                    newArray.set(slotIndexArrays[currentSlot].subarray(0, indexOffset));
                    slotIndexArrays[currentSlot] = newArray;
                }

                slotIndexArrays[currentSlot].set(overlapping, indexOffset);

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

                if (pendingBatches.length >= MEGA_MAX_BATCHES || indexOffset >= MEGA_MAX_INDICES) {
                    await flushPendingBatches();
                }
            }
        }

        await flushPendingBatches();
        bar.tick();
    }

    await flushPendingBatches();

    if (inflight) {
        const result = await inflight.resultPromise;
        processResults(result.masks, inflight.batches);
        inflight = null;
    }

    bar.end();
    return buffer;
};

/**
 * Align bounds to 4x4x4 block boundaries.
 *
 * @param minX - Scene minimum X
 * @param minY - Scene minimum Y
 * @param minZ - Scene minimum Z
 * @param maxX - Scene maximum X
 * @param maxY - Scene maximum Y
 * @param maxZ - Scene maximum Z
 * @param voxelResolution - Size of each voxel
 * @returns Aligned bounds
 */
function alignGridBounds(
    minX: number, minY: number, minZ: number,
    maxX: number, maxY: number, maxZ: number,
    voxelResolution: number
): Bounds {
    const blockSize = 4 * voxelResolution;
    return {
        min: new Vec3(
            Math.floor(minX / blockSize) * blockSize,
            Math.floor(minY / blockSize) * blockSize,
            Math.floor(minZ / blockSize) * blockSize
        ),
        max: new Vec3(
            Math.ceil(maxX / blockSize) * blockSize,
            Math.ceil(maxY / blockSize) * blockSize,
            Math.ceil(maxZ / blockSize) * blockSize
        )
    };
}

export { voxelizeToBuffer, alignGridBounds };
