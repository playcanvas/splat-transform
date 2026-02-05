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

import { DataTable, TypedArray } from '../data-table/data-table.js';

/**
 * WGSL shader for voxelizing 4x4x4 blocks with extinction-based density accumulation.
 * Uses indirection through an index buffer to access a global Gaussian buffer.
 *
 * @returns WGSL shader code
 */
const voxelizeWgsl = () => {
    return /* wgsl */ `

struct Uniforms {
    numIndices: u32,
    numBlocksX: u32,
    numBlocksY: u32,
    opacityCutoff: f32,
    voxelResolution: f32,
    blockMinX: f32,
    blockMinY: f32,
    blockMinZ: f32
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

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> allGaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> results: array<atomic<u32>>;

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
    @builtin(workgroup_id) blockId: vec3u
) {
    let localPos = mortonToXYZ(voxelIdx);
    let blockMin = vec3f(uniforms.blockMinX, uniforms.blockMinY, uniforms.blockMinZ);
    let blockOffset = vec3f(f32(blockId.x), f32(blockId.y), f32(blockId.z)) * 4.0 * uniforms.voxelResolution;
    let voxelCenter = blockMin + blockOffset + (vec3f(localPos) + 0.5) * uniforms.voxelResolution;
    let voxelHalfSize = uniforms.voxelResolution * 0.5;
    
    // Extinction-based density accumulation
    // Accumulate raw density (sigma), then convert to opacity at the end
    var totalSigma = 0.0;
    for (var i = 0u; i < uniforms.numIndices; i++) {
        let gaussianIdx = indices[i];
        let g = allGaussians[gaussianIdx];
        totalSigma += evaluateGaussianForVoxel(voxelCenter, voxelHalfSize, g);
        
        // Early exit if density is already very high
        // sigma of 7 gives opacity > 0.999
        if (totalSigma >= 7.0) {
            break;
        }
    }
    
    // Convert accumulated density to opacity using Beer-Lambert law
    // The contributions are already opacity-like values, so no depth scaling needed
    let finalOpacity = 1.0 - exp(-totalSigma);
    
    // Determine if voxel is solid
    let isSolid = finalOpacity >= uniforms.opacityCutoff;
    
    // Write result bit to output
    let blockIndex = blockId.x + blockId.y * uniforms.numBlocksX + blockId.z * uniforms.numBlocksX * uniforms.numBlocksY;
    let wordIndex = blockIndex * 2u + (voxelIdx >> 5u);
    let bitIndex = voxelIdx & 31u;
    
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
 * GPU-accelerated voxelization of Gaussian splat data.
 *
 * Uploads all Gaussian data once, then uses per-chunk index buffers
 * for extinction-based density accumulation.
 */
class GpuVoxelization {
    private device: GraphicsDevice;
    private shader: Shader;
    private compute: Compute;
    private bindGroupFormat: BindGroupFormat;

    // Buffers
    private gaussianBuffer: StorageBuffer | null = null;
    private indexBuffer: StorageBuffer;
    private resultsBuffer: StorageBuffer;

    private maxIndicesPerDispatch: number;
    private maxBlocksPerDispatch: number;
    private totalGaussians: number = 0;

    /** Floats per Gaussian in the interleaved buffer (16 for alignment) */
    private static readonly FLOATS_PER_GAUSSIAN = 16;

    /** Maximum indices per dispatch (controls index buffer size) */
    private static readonly MAX_INDICES_DISPATCH = 65536;

    /** Maximum blocks per dispatch */
    private static readonly MAX_BLOCKS_DISPATCH = 4096;

    /**
     * Create a GPU voxelization instance.
     *
     * @param device - PlayCanvas graphics device (must support WebGPU compute)
     */
    constructor(device: GraphicsDevice) {
        this.device = device;
        this.maxIndicesPerDispatch = GpuVoxelization.MAX_INDICES_DISPATCH;
        this.maxBlocksPerDispatch = GpuVoxelization.MAX_BLOCKS_DISPATCH;

        // Create bind group format with 4 bindings:
        // 0: uniforms, 1: allGaussians (read), 2: indices (read), 3: results (read_write)
        this.bindGroupFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('allGaussians', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('indices', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('results', SHADERSTAGE_COMPUTE)
        ]);

        // Create shader
        this.shader = new Shader(device, {
            name: 'voxelize-block',
            shaderLanguage: SHADERLANGUAGE_WGSL,
            cshader: voxelizeWgsl(),
            // @ts-ignore - computeUniformBufferFormats not in types
            computeUniformBufferFormats: {
                uniforms: new UniformBufferFormat(device, [
                    new UniformFormat('numIndices', UNIFORMTYPE_UINT),
                    new UniformFormat('numBlocksX', UNIFORMTYPE_UINT),
                    new UniformFormat('numBlocksY', UNIFORMTYPE_UINT),
                    new UniformFormat('opacityCutoff', UNIFORMTYPE_FLOAT),
                    new UniformFormat('voxelResolution', UNIFORMTYPE_FLOAT),
                    new UniformFormat('blockMinX', UNIFORMTYPE_FLOAT),
                    new UniformFormat('blockMinY', UNIFORMTYPE_FLOAT),
                    new UniformFormat('blockMinZ', UNIFORMTYPE_FLOAT)
                ])
            },
            // @ts-ignore - computeBindGroupFormat not in types
            computeBindGroupFormat: this.bindGroupFormat
        });

        // Create index buffer (u32 per index)
        const indexBufferSize = this.maxIndicesPerDispatch * 4;
        this.indexBuffer = new StorageBuffer(device, indexBufferSize, BUFFERUSAGE_COPY_DST);

        // Results buffer: 2 u32s per block (64 bits)
        const resultsBufferSize = this.maxBlocksPerDispatch * 2 * 4;
        this.resultsBuffer = new StorageBuffer(device, resultsBufferSize, BUFFERUSAGE_COPY_SRC | BUFFERUSAGE_COPY_DST);

        // Create compute instance (Gaussian buffer will be set after upload)
        this.compute = new Compute(device, this.shader, 'voxelize-block');
        this.compute.setParameter('indices', this.indexBuffer);
        this.compute.setParameter('results', this.resultsBuffer);
    }

    /**
     * Upload all Gaussian data to GPU once.
     * Must be called before voxelizeBlocks.
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
            interleavedData[offset + 4] = rotW[i];
            interleavedData[offset + 5] = rotX[i];
            interleavedData[offset + 6] = rotY[i];
            interleavedData[offset + 7] = rotZ[i];
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

        // Bind the Gaussian buffer to the compute shader
        this.compute.setParameter('allGaussians', this.gaussianBuffer);
    }

    /**
     * Voxelize a batch of 4x4x4 blocks using proper alpha blending.
     * uploadAllGaussians must be called first.
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
        if (!this.gaussianBuffer) {
            throw new Error('uploadAllGaussians must be called before voxelizeBlocks');
        }

        const totalBlocks = numBlocksX * numBlocksY * numBlocksZ;

        if (totalBlocks > this.maxBlocksPerDispatch) {
            throw new Error(`Too many blocks: ${totalBlocks} > ${this.maxBlocksPerDispatch}`);
        }

        if (gaussianIndices.length > this.maxIndicesPerDispatch) {
            throw new Error(`Too many indices: ${gaussianIndices.length} > ${this.maxIndicesPerDispatch}. Consider batching.`);
        }

        // Clear results buffer
        const zeroResults = new Uint32Array(totalBlocks * 2);
        this.resultsBuffer.write(0, zeroResults, 0, totalBlocks * 2);

        // Upload indices for this chunk
        const indicesU32 = new Uint32Array(gaussianIndices);
        this.indexBuffer.write(0, indicesU32, 0, gaussianIndices.length);

        // Set uniforms
        this.compute.setParameter('numIndices', gaussianIndices.length);
        this.compute.setParameter('numBlocksX', numBlocksX);
        this.compute.setParameter('numBlocksY', numBlocksY);
        this.compute.setParameter('opacityCutoff', opacityCutoff);
        this.compute.setParameter('voxelResolution', voxelResolution);
        this.compute.setParameter('blockMinX', blockMin.x);
        this.compute.setParameter('blockMinY', blockMin.y);
        this.compute.setParameter('blockMinZ', blockMin.z);

        // Dispatch compute
        this.compute.setupDispatch(numBlocksX, numBlocksY, numBlocksZ);
        this.device.computeDispatch([this.compute], 'voxelize-dispatch');

        // Read results
        const readData = await this.resultsBuffer.read(0, totalBlocks * 2 * 4, null, true);
        const resultsU32 = new Uint32Array(readData.buffer, readData.byteOffset, totalBlocks * 2);

        // Convert to result format
        const blocks: Array<{ x: number; y: number; z: number }> = [];
        const masks = new Uint32Array(totalBlocks * 2);

        let blockIdx = 0;
        for (let z = 0; z < numBlocksZ; z++) {
            for (let y = 0; y < numBlocksY; y++) {
                for (let x = 0; x < numBlocksX; x++) {
                    blocks.push({ x, y, z });
                    masks[blockIdx * 2] = resultsU32[blockIdx * 2];
                    masks[blockIdx * 2 + 1] = resultsU32[blockIdx * 2 + 1];
                    blockIdx++;
                }
            }
        }

        return { blocks, masks };
    }

    /**
     * Get the maximum number of indices that can be processed in one dispatch.
     *
     * @returns Maximum index count per dispatch
     */
    get maxIndices(): number {
        return this.maxIndicesPerDispatch;
    }

    /**
     * Get the maximum number of blocks that can be processed in one dispatch.
     *
     * @returns Maximum block count
     */
    get maxBlocks(): number {
        return this.maxBlocksPerDispatch;
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
        this.indexBuffer.destroy();
        this.resultsBuffer.destroy();
        this.shader.destroy();
        this.bindGroupFormat.destroy();
    }
}

export { GpuVoxelization, VoxelizationResult };
