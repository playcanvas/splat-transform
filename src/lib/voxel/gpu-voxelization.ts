import {
    BUFFERUSAGE_COPY_DST,
    BUFFERUSAGE_COPY_SRC,
    SHADERLANGUAGE_WGSL,
    SHADERSTAGE_COMPUTE,
    UNIFORMTYPE_FLOAT,
    UNIFORMTYPE_UINT,
    BindGroupFormat,
    BindStorageBufferFormat,
    BindUniformBufferFormat,
    Compute,
    GraphicsDevice,
    Shader,
    StorageBuffer,
    UniformBufferFormat,
    UniformFormat
} from 'playcanvas';

import { DataTable } from '../data-table/data-table.js';

/**
 * WGSL shader for multi-batch voxelization of 4x4x4 blocks.
 *
 * Each workgroup processes one block in one batch.
 * - workgroup_id.z = batch index
 * - workgroup_id.x = flat block index within the batch
 * Per-batch metadata (index range, block origin, dimensions) comes from a storage buffer,
 * allowing many batches to be dispatched in a single GPU call.
 *
 * @returns WGSL shader code
 */
const voxelizeMultiBatchWgsl = () => {
    return /* wgsl */ `

struct Uniforms {
    opacityCutoff: f32,
    voxelResolution: f32,
    maxBlocksPerBatch: u32
}

struct Gaussian {
    posX: f32,
    posY: f32,
    posZ: f32,
    opacityLogit: f32,
    rotW: f32,
    rotX: f32,
    rotY: f32,
    rotZ: f32,
    scaleX: f32,
    scaleY: f32,
    scaleZ: f32,
    extentX: f32,
    extentY: f32,
    extentZ: f32,
    _padding0: f32,
    _padding1: f32
}

struct BatchInfo {
    indexOffset: u32,
    indexCount: u32,
    numBlocksX: u32,
    numBlocksY: u32,
    numBlocksZ: u32,
    blockMinX: f32,
    blockMinY: f32,
    blockMinZ: f32
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> allGaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> results: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> batchInfos: array<BatchInfo>;

// Shared memory for cooperative Gaussian loading.
// All 64 threads in a workgroup load one Gaussian each, then all threads
// evaluate against the shared chunk — reducing global memory reads by 64x.
// 64 Gaussians * 64 bytes each = 4 KB (well within 16 KB WebGPU minimum).
const tileSize = 64u;
var<workgroup> sharedGaussians: array<Gaussian, tileSize>;

fn mortonToXYZ(m: u32) -> vec3u {
    return vec3u(
        (m & 1u) | ((m >> 2u) & 2u),
        ((m >> 1u) & 1u) | ((m >> 3u) & 2u),
        ((m >> 2u) & 1u) | ((m >> 4u) & 2u)
    );
}

// Evaluate Gaussian contribution to a voxel.
// Uses pre-computed AABB extents for accurate overlap check.
// Returns raw density contribution for extinction-based accumulation.
fn evaluateGaussianForVoxel(voxelCenter: vec3f, voxelHalfSize: f32, g: Gaussian) -> f32 {
    let gaussianCenter = vec3f(g.posX, g.posY, g.posZ);
    let diff = voxelCenter - gaussianCenter;
    
    // Use pre-computed world-space AABB half-extents (3-sigma, accounts for rotation)
    let extent = vec3f(g.extentX, g.extentY, g.extentZ);
    
    // Per-axis AABB overlap check
    if (any(abs(diff) > (extent + voxelHalfSize))) {
        return 0.0;
    }
    
    // Find closest point in voxel to Gaussian center
    let closestPoint = clamp(gaussianCenter, voxelCenter - voxelHalfSize, voxelCenter + voxelHalfSize);
    let closestDiff = closestPoint - gaussianCenter;
    
    // Inverse rotation using cross-product formula (Rodrigues rotation)
    // For inverse: negate xyz components of quaternion
    let qxyz = vec3f(-g.rotX, -g.rotY, -g.rotZ);
    let t = 2.0 * cross(qxyz, closestDiff);
    let localDiff = closestDiff + g.rotW * t + cross(qxyz, t);
    
    // Calculate Mahalanobis distance squared
    let invScale = vec3f(exp(-g.scaleX), exp(-g.scaleY), exp(-g.scaleZ));
    let scaled = localDiff * invScale;
    let d2 = dot(scaled, scaled);
    
    // Get Gaussian opacity and return density contribution
    let opacity = 1.0 / (1.0 + exp(-g.opacityLogit));
    return opacity * exp(-0.5 * d2);
}

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_index) voxelIdx: u32,
    @builtin(workgroup_id) wgId: vec3u
) {
    let batchIdx = wgId.z;
    let flatBlockId = wgId.x;
    let info = batchInfos[batchIdx];
    
    // Skip padded workgroups beyond the batch's actual block count
    let totalBlocks = info.numBlocksX * info.numBlocksY * info.numBlocksZ;
    if (flatBlockId >= totalBlocks) { return; }
    
    // Decompose flat block ID to 3D coordinates within the batch
    let blockX = flatBlockId % info.numBlocksX;
    let blockY = (flatBlockId / info.numBlocksX) % info.numBlocksY;
    let blockZ = flatBlockId / (info.numBlocksX * info.numBlocksY);
    
    let localPos = mortonToXYZ(voxelIdx);
    let blockMin = vec3f(info.blockMinX, info.blockMinY, info.blockMinZ);
    let blockOffset = vec3f(f32(blockX), f32(blockY), f32(blockZ)) * 4.0 * uniforms.voxelResolution;
    let voxelCenter = blockMin + blockOffset + (vec3f(localPos) + 0.5) * uniforms.voxelResolution;
    let voxelHalfSize = uniforms.voxelResolution * 0.5;
    
    // Extinction-based density accumulation with shared memory tiling.
    // Instead of each thread independently reading Gaussians from global memory,
    // all 64 threads cooperatively load a tile of 64 Gaussians into shared memory,
    // then all threads evaluate against the shared tile before loading the next.
    var totalSigma = 0.0;
    let numIndices = info.indexCount;
    let numTiles = (numIndices + tileSize - 1u) / tileSize;
    
    for (var tile = 0u; tile < numTiles; tile++) {
        // Cooperative load: each thread loads one Gaussian into shared memory
        let loadIdx = tile * tileSize + voxelIdx;
        if (loadIdx < numIndices) {
            let gaussianIdx = indices[info.indexOffset + loadIdx];
            sharedGaussians[voxelIdx] = allGaussians[gaussianIdx];
        }
        
        // Wait for all threads to finish loading the tile
        workgroupBarrier();
        
        // Evaluate all Gaussians in this tile (skip math if already saturated)
        if (totalSigma < 7.0) {
            let thisTileSize = min(tileSize, numIndices - tile * tileSize);
            for (var c = 0u; c < thisTileSize; c++) {
                totalSigma += evaluateGaussianForVoxel(voxelCenter, voxelHalfSize, sharedGaussians[c]);
                if (totalSigma >= 7.0) {
                    break;
                }
            }
        }
        
        // Wait before next tile overwrites shared memory
        workgroupBarrier();
    }
    
    // Convert accumulated density to opacity using Beer-Lambert law
    let finalOpacity = 1.0 - exp(-totalSigma);
    
    // Determine if voxel is solid
    let isSolid = finalOpacity >= uniforms.opacityCutoff;
    
    // Write result bit to output using linear indexing (z*16 + y*4 + x)
    // Each batch's results are at batchIdx * maxBlocksPerBatch * 2 in the results array
    let linearIdx = localPos.z * 16u + localPos.y * 4u + localPos.x;
    let resultBase = batchIdx * uniforms.maxBlocksPerBatch * 2u;
    let wordIndex = resultBase + flatBlockId * 2u + (linearIdx >> 5u);
    let bitIndex = linearIdx & 31u;
    
    if (isSolid) {
        atomicOr(&results[wordIndex], 1u << bitIndex);
    }
}
`;
};

/**
 * Result of voxelizing a batch of 4x4x4 blocks.
 */
interface VoxelizationResult {
    /** Block coordinates (x, y, z) for each block */
    blocks: Array<{ x: number; y: number; z: number }>;

    /**
     * Interleaved u32 voxel masks for each block.
     * For block at index i: masks[i*2] = low bits (voxels 0-31), masks[i*2+1] = high bits (voxels 32-63)
     */
    masks: Uint32Array;
}

/**
 * Specification for a single batch in a multi-batch dispatch.
 */
interface BatchSpec {
    /** Offset into the concatenated index array */
    indexOffset: number;

    /** Number of Gaussian indices for this batch */
    indexCount: number;

    /** World-space minimum corner of the first block */
    blockMin: { x: number; y: number; z: number };

    /** Number of blocks in X direction */
    numBlocksX: number;

    /** Number of blocks in Y direction */
    numBlocksY: number;

    /** Number of blocks in Z direction */
    numBlocksZ: number;
}

/**
 * Result of a multi-batch voxelization dispatch.
 */
interface MultiBatchResult {
    /**
     * Raw u32 masks for all batches.
     * For batch i, block j: masks[(i * maxBlocksPerBatch + j) * 2] = low, [+1] = high
     */
    masks: Uint32Array;

    /** Maximum blocks per batch, used for offset calculation */
    maxBlocksPerBatch: number;
}

/**
 * A set of GPU buffers and compute instance for one dispatch slot.
 * Two slots allow double-buffered pipelining: while the GPU executes
 * a dispatch on slot A, the CPU can prepare data for slot B.
 */
interface DispatchSlot {
    indexBuffer: StorageBuffer;
    resultsBuffer: StorageBuffer;
    batchInfoBuffer: StorageBuffer;
    compute: Compute;
    indexBufferSize: number;
    resultsBufferSize: number;
    batchInfoBufferSize: number;
}

/**
 * GPU-accelerated voxelization of Gaussian splat data.
 *
 * Uploads all Gaussian data once, then dispatches many batches in a single
 * GPU call using per-batch metadata to minimize CPU-GPU synchronization.
 *
 * Supports double-buffered pipelining via two dispatch slots: the CPU can
 * prepare the next mega-dispatch while the GPU is still executing the current one.
 */
class GpuVoxelization {
    private device: GraphicsDevice;
    private shader: Shader;
    private bindGroupFormat: BindGroupFormat;

    // Shared Gaussian buffer (read-only, uploaded once)
    private gaussianBuffer: StorageBuffer | null = null;

    // Double-buffered dispatch slots
    private slots: DispatchSlot[];

    private totalGaussians: number = 0;

    // Reusable zero buffer for clearing results (grown as needed)
    private zeroBuffer: Uint32Array;

    /** Floats per Gaussian in the interleaved buffer (16 for alignment) */
    private static readonly FLOATS_PER_GAUSSIAN = 16;

    /** Maximum blocks per batch (16^3 = 4096 for 4x4x4 voxel blocks in a 16-block batch) */
    static readonly MAX_BLOCKS_PER_BATCH = 4096;

    /** Number of u32 fields per BatchInfo struct */
    private static readonly BATCH_INFO_U32S = 8;

    /** Number of dispatch slots (for double-buffered pipelining) */
    static readonly NUM_SLOTS = 2;

    /**
     * Create a GPU voxelization instance.
     *
     * @param device - PlayCanvas graphics device (must support WebGPU compute)
     */
    constructor(device: GraphicsDevice) {
        this.device = device;

        // Create shared bind group format with 5 bindings:
        // 0: uniforms, 1: allGaussians (read), 2: indices (read), 3: results (read_write), 4: batchInfos (read)
        this.bindGroupFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('allGaussians', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('indices', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('results', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('batchInfos', SHADERSTAGE_COMPUTE, true)
        ]);

        // Create shared shader
        this.shader = new Shader(device, {
            name: 'voxelize-multi-batch',
            shaderLanguage: SHADERLANGUAGE_WGSL,
            cshader: voxelizeMultiBatchWgsl(),
            // @ts-ignore - computeUniformBufferFormats not in types
            computeUniformBufferFormats: {
                uniforms: new UniformBufferFormat(device, [
                    new UniformFormat('opacityCutoff', UNIFORMTYPE_FLOAT),
                    new UniformFormat('voxelResolution', UNIFORMTYPE_FLOAT),
                    new UniformFormat('maxBlocksPerBatch', UNIFORMTYPE_UINT)
                ])
            },
            // @ts-ignore - computeBindGroupFormat not in types
            computeBindGroupFormat: this.bindGroupFormat
        });

        // Create double-buffered dispatch slots
        this.slots = [];
        for (let i = 0; i < GpuVoxelization.NUM_SLOTS; i++) {
            const indexBufferSize = 1024 * 1024 * 4;  // 1M indices = 4 MB
            const indexBuffer = new StorageBuffer(device, indexBufferSize, BUFFERUSAGE_COPY_DST);

            // 64 batches × 4096 blocks × 2 u32 × 4 bytes = 2 MB
            const resultsBufferSize = 64 * GpuVoxelization.MAX_BLOCKS_PER_BATCH * 2 * 4;
            const resultsBuffer = new StorageBuffer(device, resultsBufferSize, BUFFERUSAGE_COPY_SRC | BUFFERUSAGE_COPY_DST);

            // 64 batches × 8 fields × 4 bytes = 2 KB
            const batchInfoBufferSize = 64 * GpuVoxelization.BATCH_INFO_U32S * 4;
            const batchInfoBuffer = new StorageBuffer(device, batchInfoBufferSize, BUFFERUSAGE_COPY_DST);

            const compute = new Compute(device, this.shader, `voxelize-slot-${i}`);
            compute.setParameter('indices', indexBuffer);
            compute.setParameter('results', resultsBuffer);
            compute.setParameter('batchInfos', batchInfoBuffer);

            this.slots.push({
                indexBuffer, resultsBuffer, batchInfoBuffer, compute,
                indexBufferSize, resultsBufferSize, batchInfoBufferSize
            });
        }

        // Pre-allocate zero buffer for clearing results
        const initialResultsU32 = 64 * GpuVoxelization.MAX_BLOCKS_PER_BATCH * 2;
        this.zeroBuffer = new Uint32Array(initialResultsU32);
    }

    /**
     * Upload all Gaussian data to GPU once.
     * Must be called before any dispatch methods.
     *
     * @param dataTable - DataTable containing all Gaussian properties
     * @param extents - DataTable containing pre-computed AABB extents (extent_x, extent_y, extent_z)
     */
    uploadAllGaussians(dataTable: DataTable, extents: DataTable): void {
        const numGaussians = dataTable.numRows;
        this.totalGaussians = numGaussians;

        // Create buffer sized for ALL Gaussians
        const bufferSize = numGaussians * GpuVoxelization.FLOATS_PER_GAUSSIAN * 4;

        // Destroy old buffer if it exists
        if (this.gaussianBuffer) {
            this.gaussianBuffer.destroy();
        }

        this.gaussianBuffer = new StorageBuffer(this.device, bufferSize, BUFFERUSAGE_COPY_DST);

        // Interleave and upload all Gaussian data
        const interleavedData = new Float32Array(numGaussians * GpuVoxelization.FLOATS_PER_GAUSSIAN);

        const x = dataTable.getColumnByName('x').data;
        const y = dataTable.getColumnByName('y').data;
        const z = dataTable.getColumnByName('z').data;
        const opacity = dataTable.getColumnByName('opacity').data;
        const rotW = dataTable.getColumnByName('rot_0').data;
        const rotX = dataTable.getColumnByName('rot_1').data;
        const rotY = dataTable.getColumnByName('rot_2').data;
        const rotZ = dataTable.getColumnByName('rot_3').data;
        const scaleX = dataTable.getColumnByName('scale_0').data;
        const scaleY = dataTable.getColumnByName('scale_1').data;
        const scaleZ = dataTable.getColumnByName('scale_2').data;
        const extentX = extents.getColumnByName('extent_x').data;
        const extentY = extents.getColumnByName('extent_y').data;
        const extentZ = extents.getColumnByName('extent_z').data;

        for (let i = 0; i < numGaussians; i++) {
            const offset = i * GpuVoxelization.FLOATS_PER_GAUSSIAN;
            interleavedData[offset + 0] = x[i];
            interleavedData[offset + 1] = y[i];
            interleavedData[offset + 2] = z[i];
            interleavedData[offset + 3] = opacity[i];

            // Normalize quaternion — the Rodrigues rotation in the shader
            // assumes unit quaternions; non-normalized ones (common in MCMC
            // training) would produce incorrect Mahalanobis distances.
            const qw = rotW[i], qx = rotX[i], qy = rotY[i], qz = rotZ[i];
            const qlen = Math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
            const invLen = qlen > 0 ? 1 / qlen : 0;
            interleavedData[offset + 4] = qw * invLen;
            interleavedData[offset + 5] = qx * invLen;
            interleavedData[offset + 6] = qy * invLen;
            interleavedData[offset + 7] = qz * invLen;
            interleavedData[offset + 8] = scaleX[i];
            interleavedData[offset + 9] = scaleY[i];
            interleavedData[offset + 10] = scaleZ[i];
            interleavedData[offset + 11] = extentX[i];
            interleavedData[offset + 12] = extentY[i];
            interleavedData[offset + 13] = extentZ[i];
            interleavedData[offset + 14] = 0; // padding
            interleavedData[offset + 15] = 0; // padding
        }

        this.gaussianBuffer.write(0, interleavedData, 0, interleavedData.length);

        // Bind the shared Gaussian buffer to ALL slot compute instances
        for (const slot of this.slots) {
            slot.compute.setParameter('allGaussians', this.gaussianBuffer);
        }
    }

    /**
     * Ensure a slot's buffer is at least the given size, growing if needed.
     */
    private ensureSlotBuffer(
        slot: DispatchSlot,
        bufferKey: 'indexBuffer' | 'resultsBuffer' | 'batchInfoBuffer',
        sizeKey: 'indexBufferSize' | 'resultsBufferSize' | 'batchInfoBufferSize',
        neededSize: number,
        usage: number,
        paramName: string
    ): void {
        if (neededSize <= slot[sizeKey]) return;

        slot[bufferKey].destroy();
        const newBuffer = new StorageBuffer(this.device, neededSize, usage);
        slot.compute.setParameter(paramName, newBuffer);
        slot[bufferKey] = newBuffer;
        slot[sizeKey] = neededSize;
    }

    /**
     * Submit a multi-batch dispatch on the specified slot.
     *
     * Returns a Promise that resolves when GPU results are ready. The caller
     * should NOT await immediately — do CPU work first (BVH queries, index
     * copying for the next dispatch) to overlap CPU and GPU execution.
     *
     * Use different slot indices for consecutive calls to avoid buffer conflicts.
     *
     * @param slotIndex - Which dispatch slot to use (0 or 1)
     * @param concatenatedIndices - Pre-built Uint32Array of all Gaussian indices
     * @param totalIndices - Number of valid indices in the array
     * @param batches - Per-batch metadata (index offset/count, block origin, dimensions)
     * @param voxelResolution - Size of each voxel in world units
     * @param opacityCutoff - Opacity threshold for solid voxels (0.0-1.0)
     * @returns Promise resolving to multi-batch voxelization results
     */
    submitMultiBatch(
        slotIndex: number,
        concatenatedIndices: Uint32Array,
        totalIndices: number,
        batches: BatchSpec[],
        voxelResolution: number,
        opacityCutoff: number
    ): Promise<MultiBatchResult> {
        if (!this.gaussianBuffer) {
            throw new Error('uploadAllGaussians must be called before submitMultiBatch');
        }

        const slot = this.slots[slotIndex];
        const maxBlocks = GpuVoxelization.MAX_BLOCKS_PER_BATCH;
        const numBatches = batches.length;

        if (numBatches === 0) {
            return Promise.resolve({ masks: new Uint32Array(0), maxBlocksPerBatch: maxBlocks });
        }

        // Ensure slot buffers are large enough
        this.ensureSlotBuffer(
            slot, 'indexBuffer', 'indexBufferSize',
            totalIndices * 4, BUFFERUSAGE_COPY_DST, 'indices'
        );

        const resultsU32Count = numBatches * maxBlocks * 2;
        this.ensureSlotBuffer(
            slot, 'resultsBuffer', 'resultsBufferSize',
            resultsU32Count * 4, BUFFERUSAGE_COPY_SRC | BUFFERUSAGE_COPY_DST, 'results'
        );

        const batchInfoU32Count = numBatches * GpuVoxelization.BATCH_INFO_U32S;
        this.ensureSlotBuffer(
            slot, 'batchInfoBuffer', 'batchInfoBufferSize',
            batchInfoU32Count * 4, BUFFERUSAGE_COPY_DST, 'batchInfos'
        );

        // Ensure zero buffer is large enough for clearing results
        if (this.zeroBuffer.length < resultsU32Count) {
            this.zeroBuffer = new Uint32Array(resultsU32Count);
        }

        // Clear results buffer
        slot.resultsBuffer.write(0, this.zeroBuffer, 0, resultsU32Count);

        // Upload concatenated indices
        slot.indexBuffer.write(0, concatenatedIndices, 0, totalIndices);

        // Build and upload batch info
        const batchInfoF32 = new Float32Array(batchInfoU32Count);
        const batchInfoU32 = new Uint32Array(batchInfoF32.buffer);

        for (let i = 0; i < numBatches; i++) {
            const batch = batches[i];
            const base = i * GpuVoxelization.BATCH_INFO_U32S;

            batchInfoU32[base + 0] = batch.indexOffset;
            batchInfoU32[base + 1] = batch.indexCount;
            batchInfoU32[base + 2] = batch.numBlocksX;
            batchInfoU32[base + 3] = batch.numBlocksY;
            batchInfoU32[base + 4] = batch.numBlocksZ;
            batchInfoF32[base + 5] = batch.blockMin.x;
            batchInfoF32[base + 6] = batch.blockMin.y;
            batchInfoF32[base + 7] = batch.blockMin.z;
        }

        slot.batchInfoBuffer.write(0, batchInfoU32, 0, batchInfoU32Count);

        // Set uniforms (global values only — per-batch values are in batchInfos)
        slot.compute.setParameter('opacityCutoff', opacityCutoff);
        slot.compute.setParameter('voxelResolution', voxelResolution);
        slot.compute.setParameter('maxBlocksPerBatch', maxBlocks);

        // Dispatch: x = max blocks per batch (padded), y = 1, z = num batches
        slot.compute.setupDispatch(maxBlocks, 1, numBatches);
        this.device.computeDispatch([slot.compute], `voxelize-slot-${slotIndex}`);

        // Return promise for deferred readback — the caller controls when to await
        const readSize = resultsU32Count * 4;
        return slot.resultsBuffer.read(0, readSize, null, true).then((readData: Uint8Array) => {
            const masks = new Uint32Array(readData.buffer, readData.byteOffset, resultsU32Count);
            return { masks, maxBlocksPerBatch: maxBlocks };
        });
    }

    /**
     * Voxelize many batches of 4x4x4 blocks in a single GPU dispatch.
     * Synchronous wrapper that awaits the result immediately.
     *
     * @param concatenatedIndices - Pre-built Uint32Array of all Gaussian indices
     * @param totalIndices - Number of valid indices in the array
     * @param batches - Per-batch metadata (index offset/count, block origin, dimensions)
     * @param voxelResolution - Size of each voxel in world units
     * @param opacityCutoff - Opacity threshold for solid voxels (0.0-1.0)
     * @returns Promise resolving to multi-batch voxelization results
     */
    async voxelizeMultiBatch(
        concatenatedIndices: Uint32Array,
        totalIndices: number,
        batches: BatchSpec[],
        voxelResolution: number,
        opacityCutoff: number
    ): Promise<MultiBatchResult> {
        return this.submitMultiBatch(0, concatenatedIndices, totalIndices, batches, voxelResolution, opacityCutoff);
    }

    /**
     * Voxelize a single batch of 4x4x4 blocks.
     * Convenience wrapper around voxelizeMultiBatch for backward compatibility.
     *
     * @param gaussianIndices - Indices of Gaussians to evaluate for this chunk
     * @param blockMin - World-space minimum corner of the first block
     * @param blockMin.x - X coordinate of block minimum
     * @param blockMin.y - Y coordinate of block minimum
     * @param blockMin.z - Z coordinate of block minimum
     * @param numBlocksX - Number of blocks in X direction
     * @param numBlocksY - Number of blocks in Y direction
     * @param numBlocksZ - Number of blocks in Z direction
     * @param voxelResolution - Size of each voxel in world units
     * @param opacityCutoff - Opacity threshold for solid voxels (0.0-1.0)
     * @returns Promise resolving to voxelization results
     */
    async voxelizeBlocks(
        gaussianIndices: number[],
        blockMin: { x: number; y: number; z: number },
        numBlocksX: number,
        numBlocksY: number,
        numBlocksZ: number,
        voxelResolution: number,
        opacityCutoff: number
    ): Promise<VoxelizationResult> {
        const indicesU32 = new Uint32Array(gaussianIndices);
        const result = await this.voxelizeMultiBatch(
            indicesU32,
            gaussianIndices.length,
            [{
                indexOffset: 0,
                indexCount: gaussianIndices.length,
                blockMin,
                numBlocksX,
                numBlocksY,
                numBlocksZ
            }],
            voxelResolution,
            opacityCutoff
        );

        const totalBlocks = numBlocksX * numBlocksY * numBlocksZ;
        const blocks: Array<{ x: number; y: number; z: number }> = [];
        const masks = new Uint32Array(totalBlocks * 2);

        let blockIdx = 0;
        for (let z = 0; z < numBlocksZ; z++) {
            for (let y = 0; y < numBlocksY; y++) {
                for (let x = 0; x < numBlocksX; x++) {
                    blocks.push({ x, y, z });
                    masks[blockIdx * 2] = result.masks[blockIdx * 2];
                    masks[blockIdx * 2 + 1] = result.masks[blockIdx * 2 + 1];
                    blockIdx++;
                }
            }
        }

        return { blocks, masks };
    }

    /**
     * Get the total number of Gaussians uploaded.
     *
     * @returns Total Gaussian count
     */
    get numGaussians(): number {
        return this.totalGaussians;
    }

    /**
     * Destroy GPU resources.
     */
    destroy(): void {
        if (this.gaussianBuffer) {
            this.gaussianBuffer.destroy();
        }
        for (const slot of this.slots) {
            slot.indexBuffer.destroy();
            slot.resultsBuffer.destroy();
            slot.batchInfoBuffer.destroy();
        }
        this.shader.destroy();
        this.bindGroupFormat.destroy();
    }
}

export { GpuVoxelization, type VoxelizationResult, type BatchSpec, type MultiBatchResult };
