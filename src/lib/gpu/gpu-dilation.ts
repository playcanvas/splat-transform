import {
    BUFFERUSAGE_COPY_DST,
    BUFFERUSAGE_COPY_SRC,
    SHADERLANGUAGE_WGSL,
    SHADERSTAGE_COMPUTE,
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

/**
 * WGSL compute shader for separable 1D dilation along an axis.
 *
 * Each thread emits one output bit using atomicOr; the destination buffer
 * must be cleared to zero before dispatch. Reads up to `2 * halfExtent + 1`
 * input bits along the dilation axis (early-exits on the first set bit).
 *
 * Axis is encoded by `stride` (= 1 for X, nx for Y, nx*ny for Z) and
 * `axisLen` (= nx, ny, or nz). `gid.x/y/z` is the voxel index in the chunk.
 *
 * @returns WGSL shader code
 */
/**
 * Clear shader — writes 0 to every word in the destination buffer up to
 * `numWords`. Dispatched in the same command encoder as the dilation passes
 * so it's ordered with them on the GPU; using `queue.writeBuffer` for inter-
 * pass clears would race because writes are queued separately from encoder
 * commands and execute *all writes first*, then the command buffer.
 */
const clearWgsl = () => /* wgsl */`
struct ClearUniforms {
    clearNumWords: u32,
    clearRowStride: u32,
    _pad0: u32,
    _pad1: u32
}

@group(0) @binding(0) var<uniform> u: ClearUniforms;
@group(0) @binding(1) var<storage, read_write> clearDst: array<u32>;

// 2D dispatch (rowStride = wgX * 256) so we can clear buffers larger than
// 65535 * 256 words without exceeding the WebGPU per-dimension workgroup
// limit. Linear word index = gid.x + gid.y * rowStride.
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x + gid.y * u.clearRowStride;
    if (i >= u.clearNumWords) {
        return;
    }
    clearDst[i] = 0u;
}
`;

const dilateWgsl = () => /* wgsl */`
struct Uniforms {
    nx: u32,
    ny: u32,
    nz: u32,
    halfExtent: u32,
    stride: u32,
    axisLen: u32,
    _pad0: u32,
    _pad1: u32
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<atomic<u32>>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= u.nx || gid.y >= u.ny || gid.z >= u.nz) {
        return;
    }

    let outBit = i32(gid.x) + i32(gid.y) * i32(u.nx) + i32(gid.z) * i32(u.nx) * i32(u.ny);

    var pos: u32;
    if (u.stride == 1u) {
        pos = gid.x;
    } else if (u.stride == u.nx) {
        pos = gid.y;
    } else {
        pos = gid.z;
    }

    let r = i32(u.halfExtent);
    let lo = max(0, i32(pos) - r);
    let hi = min(i32(u.axisLen) - 1, i32(pos) + r);

    let baseBit = outBit - i32(pos) * i32(u.stride);

    var any: u32 = 0u;
    for (var p: i32 = lo; p <= hi; p = p + 1) {
        let bit = baseBit + p * i32(u.stride);
        let word = src[bit >> 5];
        any = any | ((word >> (u32(bit) & 31u)) & 1u);
        if (any != 0u) {
            break;
        }
    }

    if (any != 0u) {
        atomicOr(&dst[outBit >> 5], 1u << (u32(outBit) & 31u));
    }
}
`;

/**
 * Separable 3D dilation on the GPU using a dense bit grid (1 bit per voxel
 * packed into u32). Caller is responsible for converting to/from sparse
 * representations and chunking.
 *
 * Each call to `dilateChunk` runs three separable passes (X, Z, Y) on the
 * GPU back-to-back with no CPU round-trip between them. The destination
 * buffer (`bufferB`) must be zeroed before the first pass — `clearBufferB`
 * does this, and the X/Z/Y passes ping-pong between A and B. The final
 * result lives in `bufferB` after `dilateChunk`.
 */
class GpuDilation {
    private device: GraphicsDevice;
    private dilateShader: Shader;
    private clearShader: Shader;
    private dilateBindGroupFormat: BindGroupFormat;
    private clearBindGroupFormat: BindGroupFormat;

    private bufferA: StorageBuffer;
    private bufferB: StorageBuffer;
    private bufferCapacity: number;

    // Three Compute instances (one per axis pass) so each has its own uniform
    // buffer. With a single shared Compute, PlayCanvas's setParameter routes
    // through `queue.writeBuffer(uniformBuffer)`, and all three pre-submit
    // writes end up overwriting each other — every dispatch then reads the
    // last-written uniforms (Y-pass with halfExtent=0 → identity).
    private dilateComputes: Compute[];
    private clearCompute: Compute;

    constructor(device: GraphicsDevice) {
        this.device = device;

        this.dilateBindGroupFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('src', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('dst', SHADERSTAGE_COMPUTE)
        ]);

        this.clearBindGroupFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('clearDst', SHADERSTAGE_COMPUTE)
        ]);

        this.dilateShader = new Shader(device, {
            name: 'gpu-dilation',
            shaderLanguage: SHADERLANGUAGE_WGSL,
            cshader: dilateWgsl(),
            // @ts-ignore
            computeUniformBufferFormats: {
                uniforms: new UniformBufferFormat(device, [
                    new UniformFormat('nx', UNIFORMTYPE_UINT),
                    new UniformFormat('ny', UNIFORMTYPE_UINT),
                    new UniformFormat('nz', UNIFORMTYPE_UINT),
                    new UniformFormat('halfExtent', UNIFORMTYPE_UINT),
                    new UniformFormat('stride', UNIFORMTYPE_UINT),
                    new UniformFormat('axisLen', UNIFORMTYPE_UINT)
                ])
            },
            // @ts-ignore
            computeBindGroupFormat: this.dilateBindGroupFormat
        });

        this.clearShader = new Shader(device, {
            name: 'gpu-dilation-clear',
            shaderLanguage: SHADERLANGUAGE_WGSL,
            cshader: clearWgsl(),
            // @ts-ignore
            computeUniformBufferFormats: {
                uniforms: new UniformBufferFormat(device, [
                    new UniformFormat('clearNumWords', UNIFORMTYPE_UINT),
                    new UniformFormat('clearRowStride', UNIFORMTYPE_UINT)
                ])
            },
            // @ts-ignore
            computeBindGroupFormat: this.clearBindGroupFormat
        });

        this.dilateComputes = [
            new Compute(device, this.dilateShader, 'gpu-dilate-x'),
            new Compute(device, this.dilateShader, 'gpu-dilate-z'),
            new Compute(device, this.dilateShader, 'gpu-dilate-y')
        ];
        this.clearCompute = new Compute(device, this.clearShader, 'gpu-dilate-clear');

        // Initial buffer size — grown lazily in ensureBuffers.
        this.bufferCapacity = 1024 * 1024 * 4; // 4 MB = 1M u32 = 32M bits
        this.bufferA = new StorageBuffer(device, this.bufferCapacity, BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC);
        this.bufferB = new StorageBuffer(device, this.bufferCapacity, BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC);
    }

    /**
     * Ensure the GPU buffers are large enough for `numWords` u32 entries.
     * Recreates buffers (and the zero buffer) only when growing.
     *
     * @param numWords - Required size in u32 words.
     */
    private ensureBuffers(numWords: number): void {
        const neededBytes = numWords * 4;
        if (neededBytes <= this.bufferCapacity) return;

        // Round up to power of two for fewer reallocs across chunks.
        let cap = this.bufferCapacity;
        while (cap < neededBytes) cap *= 2;

        this.bufferA.destroy();
        this.bufferB.destroy();
        this.bufferA = new StorageBuffer(this.device, cap, BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC);
        this.bufferB = new StorageBuffer(this.device, cap, BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC);
        this.bufferCapacity = cap;
    }

    /**
     * Dispatch a compute clear of `dst` to zero for the first `numWords` words.
     * Uses the command encoder so it's correctly ordered with subsequent
     * dilation passes (unlike `queue.writeBuffer`, which is queued separately
     * and would race against the dispatches).
     */
    private dispatchClear(dst: StorageBuffer, numWords: number): void {
        // 2D dispatch with workgroup_size(256, 1, 1). Each row of workgroups
        // covers `wgX * 256` words; we cap wgX at the WebGPU 65535 limit and
        // grow wgY as needed.
        const totalWg = Math.ceil(numWords / 256);
        const MAX_DIM = 65535;
        const wgX = Math.min(totalWg, MAX_DIM);
        const wgY = Math.ceil(totalWg / wgX);
        const rowStride = wgX * 256;

        this.clearCompute.setParameter('clearDst', dst);
        this.clearCompute.setParameter('clearNumWords', numWords);
        this.clearCompute.setParameter('clearRowStride', rowStride);
        this.clearCompute.setupDispatch(wgX, wgY, 1);
        this.device.computeDispatch([this.clearCompute], 'gpu-dilate-clear');
    }

    /**
     * Run the three separable dilation passes on a dense bit chunk.
     *
     * @param srcBits - Dense bit grid (1 bit per voxel, packed in u32 words,
     *  size >= ceil(nx*ny*nz/32)). Uploaded to GPU; returned readback bytes
     *  contain the dilated result for the same voxel layout.
     * @param nx - Chunk width in voxels.
     * @param ny - Chunk height in voxels.
     * @param nz - Chunk depth in voxels.
     * @param halfExtentXZ - Dilation half-extent for X and Z passes.
     * @param halfExtentY - Dilation half-extent for Y pass.
     * @returns Promise resolving to the dilated dense bit grid.
     */
    async dilateChunk(
        srcBits: Uint32Array,
        nx: number, ny: number, nz: number,
        halfExtentXZ: number,
        halfExtentY: number
    ): Promise<Uint32Array> {
        const totalVoxels = nx * ny * nz;
        const numWords = (totalVoxels + 31) >>> 5;
        this.ensureBuffers(numWords);

        // Upload source into bufferA. Inter-pass clears use compute dispatches
        // (see `dispatchClear`) so they're encoder-ordered with the dilation
        // dispatches.
        this.bufferA.write(0, srcBits, 0, numWords);

        // X-pass: A -> B. Clear B before dispatch so atomicOr starts from zero.
        this.dispatchClear(this.bufferB, numWords);
        this.dispatch(0, this.bufferA, this.bufferB, nx, ny, nz, halfExtentXZ, 1, nx);

        // Z-pass: B -> A. Clear A.
        this.dispatchClear(this.bufferA, numWords);
        this.dispatch(1, this.bufferB, this.bufferA, nx, ny, nz, halfExtentXZ, nx * ny, nz);

        // Y-pass: A -> B. Clear B.
        this.dispatchClear(this.bufferB, numWords);
        this.dispatch(2, this.bufferA, this.bufferB, nx, ny, nz, halfExtentY, nx, ny);

        const readData = await this.bufferB.read(0, numWords * 4, null, true) as Uint8Array;
        return new Uint32Array(readData.buffer, readData.byteOffset, numWords);
    }

    private dispatch(
        passIdx: number,
        src: StorageBuffer, dst: StorageBuffer,
        nx: number, ny: number, nz: number,
        halfExtent: number,
        stride: number, axisLen: number
    ): void {
        const compute = this.dilateComputes[passIdx];
        compute.setParameter('src', src);
        compute.setParameter('dst', dst);
        compute.setParameter('nx', nx);
        compute.setParameter('ny', ny);
        compute.setParameter('nz', nz);
        compute.setParameter('halfExtent', halfExtent);
        compute.setParameter('stride', stride);
        compute.setParameter('axisLen', axisLen);

        // setupDispatch takes workgroup counts; workgroup_size = (8, 8, 4).
        const wgX = Math.ceil(nx / 8);
        const wgY = Math.ceil(ny / 8);
        const wgZ = Math.ceil(nz / 4);
        compute.setupDispatch(wgX, wgY, wgZ);
        this.device.computeDispatch([compute], `gpu-dilate-pass-${passIdx}`);
    }

    destroy(): void {
        this.bufferA.destroy();
        this.bufferB.destroy();
        for (const c of this.dilateComputes) c.destroy();
        this.clearCompute.destroy();
        this.dilateShader.destroy();
        this.clearShader.destroy();
        this.dilateBindGroupFormat.destroy();
        this.clearBindGroupFormat.destroy();
    }
}

export { GpuDilation };
