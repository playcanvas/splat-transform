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

/**
 * X-axis dilation shader — per-word.
 *
 * Each thread produces one 32-bit output word at `(xWord, y, z)` and writes
 * it directly (no atomics). The output bit at relative X position `b` (in
 * `[0, 31]`) is the OR of input bits in `[xWord*32 + b - r, xWord*32 + b + r]`,
 * which lives across up to three input words: `W_prev` (xWord-1), `W` (xWord),
 * and `W_next` (xWord+1). For each shift `d` in `[1, r]` we OR in two shifted
 * views: rightward (`(W >> d) | (W_next << (32 - d))`) for `d` positive, and
 * leftward (`(W << d) | (W_prev >> (32 - d))`) for `d` negative. `d == 32`
 * is special-cased because WGSL `u32 << 32` is UB.
 *
 * Bound by the chunk's `numXWords` (= ceil(nx / 32)). Out-of-bounds neighbors
 * are read as 0.
 */
const dilateXWgsl = () => /* wgsl */`
struct DilateXUniforms {
    numXWords: u32,
    ny: u32,
    nz: u32,
    halfExtent: u32
}

@group(0) @binding(0) var<uniform> u: DilateXUniforms;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(8, 4, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= u.numXWords || gid.y >= u.ny || gid.z >= u.nz) {
        return;
    }

    let xWord = gid.x;
    let y = gid.y;
    let z = gid.z;
    let rowStride = u.numXWords;
    let planeStride = rowStride * u.ny;
    let rowOffset = y * rowStride + z * planeStride;

    let W = src[rowOffset + xWord];
    var W_prev: u32 = 0u;
    if (xWord > 0u) {
        W_prev = src[rowOffset + xWord - 1u];
    }
    var W_next: u32 = 0u;
    if (xWord + 1u < u.numXWords) {
        W_next = src[rowOffset + xWord + 1u];
    }

    var output: u32 = W;
    let r = u.halfExtent;
    for (var d: u32 = 1u; d <= r; d = d + 1u) {
        // Rightward shift: bit b ← input bit b+d. Need (W >> d) | (W_next << (32-d)).
        var shifted_pos: u32;
        if (d >= 32u) {
            shifted_pos = W_next;
        } else {
            shifted_pos = (W >> d) | (W_next << (32u - d));
        }
        // Leftward shift: bit b ← input bit b-d. Need (W << d) | (W_prev >> (32-d)).
        var shifted_neg: u32;
        if (d >= 32u) {
            shifted_neg = W_prev;
        } else {
            shifted_neg = (W << d) | (W_prev >> (32u - d));
        }
        output = output | shifted_pos | shifted_neg;
        if (output == 0xFFFFFFFFu) {
            break;
        }
    }

    dst[rowOffset + xWord] = output;
}
`;

/**
 * Y/Z-axis dilation shader — per-word.
 *
 * Each thread reads up to `2 * halfExtent + 1` input words at the same
 * `xWord` along the chosen axis (Y or Z) and OR's them into one output word.
 * No bit shifts needed because words at the same `xWord` are bit-aligned
 * across rows (row stride is `numXWords` words). Caller picks the axis by
 * setting `stride` and `axisLen`:
 *  - Y-pass: `stride = numXWords`, `axisLen = ny`.
 *  - Z-pass: `stride = numXWords * ny`, `axisLen = nz`.
 */
const dilateYZWgsl = () => /* wgsl */`
struct DilateYZUniforms {
    numXWords: u32,
    ny: u32,
    nz: u32,
    halfExtent: u32,
    stride: u32,
    axisLen: u32,
    _pad0: u32,
    _pad1: u32
}

@group(0) @binding(0) var<uniform> u: DilateYZUniforms;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(8, 4, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= u.numXWords || gid.y >= u.ny || gid.z >= u.nz) {
        return;
    }

    let xWord = gid.x;
    let y = gid.y;
    let z = gid.z;
    let rowStride = u.numXWords;
    let planeStride = rowStride * u.ny;
    let outIdx = i32(xWord) + i32(y) * i32(rowStride) + i32(z) * i32(planeStride);

    var pos: u32;
    if (u.stride == rowStride) {
        pos = y;
    } else {
        pos = z;
    }

    let r = i32(u.halfExtent);
    let lo = max(0, i32(pos) - r);
    let hi = min(i32(u.axisLen) - 1, i32(pos) + r);

    let baseIdx = outIdx - i32(pos) * i32(u.stride);
    var output: u32 = 0u;
    for (var p: i32 = lo; p <= hi; p = p + 1) {
        output = output | src[baseIdx + p * i32(u.stride)];
        if (output == 0xFFFFFFFFu) {
            break;
        }
    }

    dst[outIdx] = output;
}
`;

/**
 * Separable 3D dilation on the GPU using a row-aligned dense bit grid
 * (1 bit per voxel, packed into u32 words; each row of bits along X starts
 * on a word boundary so per-word access is trivial). Each pass owns its
 * own `Compute` instance — see `dilateChunk` for why this matters.
 */
class GpuDilation {
    private device: GraphicsDevice;
    private dilateXShader: Shader;
    private dilateYZShader: Shader;
    private clearShader: Shader;
    private dilateXBindGroupFormat: BindGroupFormat;
    private dilateYZBindGroupFormat: BindGroupFormat;
    private clearBindGroupFormat: BindGroupFormat;

    private bufferA: StorageBuffer;
    private bufferB: StorageBuffer;
    private bufferCapacity: number;

    // Three Compute instances (one per axis pass) so each has its own uniform
    // buffer. With a single shared Compute, PlayCanvas's setParameter routes
    // through `queue.writeBuffer(uniformBuffer)`, and all three pre-submit
    // writes end up overwriting each other — every dispatch then reads the
    // last-written uniforms.
    private dilateXCompute: Compute;
    private dilateYCompute: Compute;
    private dilateZCompute: Compute;
    private clearCompute: Compute;

    constructor(device: GraphicsDevice) {
        this.device = device;

        this.dilateXBindGroupFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('src', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('dst', SHADERSTAGE_COMPUTE)
        ]);

        this.dilateYZBindGroupFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('src', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('dst', SHADERSTAGE_COMPUTE)
        ]);

        this.clearBindGroupFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('clearDst', SHADERSTAGE_COMPUTE)
        ]);

        this.dilateXShader = new Shader(device, {
            name: 'gpu-dilation-x',
            shaderLanguage: SHADERLANGUAGE_WGSL,
            cshader: dilateXWgsl(),
            // @ts-ignore
            computeUniformBufferFormats: {
                uniforms: new UniformBufferFormat(device, [
                    new UniformFormat('numXWords', UNIFORMTYPE_UINT),
                    new UniformFormat('ny', UNIFORMTYPE_UINT),
                    new UniformFormat('nz', UNIFORMTYPE_UINT),
                    new UniformFormat('halfExtent', UNIFORMTYPE_UINT)
                ])
            },
            // @ts-ignore
            computeBindGroupFormat: this.dilateXBindGroupFormat
        });

        this.dilateYZShader = new Shader(device, {
            name: 'gpu-dilation-yz',
            shaderLanguage: SHADERLANGUAGE_WGSL,
            cshader: dilateYZWgsl(),
            // @ts-ignore
            computeUniformBufferFormats: {
                uniforms: new UniformBufferFormat(device, [
                    new UniformFormat('numXWords', UNIFORMTYPE_UINT),
                    new UniformFormat('ny', UNIFORMTYPE_UINT),
                    new UniformFormat('nz', UNIFORMTYPE_UINT),
                    new UniformFormat('halfExtent', UNIFORMTYPE_UINT),
                    new UniformFormat('stride', UNIFORMTYPE_UINT),
                    new UniformFormat('axisLen', UNIFORMTYPE_UINT)
                ])
            },
            // @ts-ignore
            computeBindGroupFormat: this.dilateYZBindGroupFormat
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

        this.dilateXCompute = new Compute(device, this.dilateXShader, 'gpu-dilate-x');
        this.dilateYCompute = new Compute(device, this.dilateYZShader, 'gpu-dilate-y');
        this.dilateZCompute = new Compute(device, this.dilateYZShader, 'gpu-dilate-z');
        this.clearCompute = new Compute(device, this.clearShader, 'gpu-dilate-clear');

        // Initial buffer size — grown lazily in ensureBuffers.
        this.bufferCapacity = 1024 * 1024 * 4;
        this.bufferA = new StorageBuffer(device, this.bufferCapacity, BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC);
        this.bufferB = new StorageBuffer(device, this.bufferCapacity, BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC);
    }

    private ensureBuffers(numWords: number): void {
        const neededBytes = numWords * 4;
        if (neededBytes <= this.bufferCapacity) return;

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
     * @param srcBits - Row-aligned dense bit grid (`numXWords * ny * nz`
     *  u32 words). Bit at voxel (x, y, z) lives at word index
     *  `(x >> 5) + y * numXWords + z * numXWords * ny`, bit `x & 31`.
     * @param numXWords - Number of u32 words per row (= ceil(nx / 32)).
     * @param ny - Chunk height in voxels.
     * @param nz - Chunk depth in voxels.
     * @param halfExtentXZ - Dilation half-extent for X and Z passes.
     * @param halfExtentY - Dilation half-extent for Y pass.
     * @returns Promise resolving to the dilated dense bit grid.
     */
    async dilateChunk(
        srcBits: Uint32Array,
        numXWords: number, ny: number, nz: number,
        halfExtentXZ: number,
        halfExtentY: number
    ): Promise<Uint32Array> {
        const numWords = numXWords * ny * nz;
        this.ensureBuffers(numWords);

        // Upload source into bufferA. Inter-pass clears use compute dispatches
        // so they're encoder-ordered with the dilation dispatches.
        this.bufferA.write(0, srcBits, 0, numWords);

        // X-pass: A -> B. Per-word, no atomics, but B must be a clean
        // destination buffer; clear keeps it consistent across calls.
        this.dispatchClear(this.bufferB, numWords);
        this.dispatchX(this.bufferA, this.bufferB, numXWords, ny, nz, halfExtentXZ);

        // Z-pass: B -> A.
        this.dispatchClear(this.bufferA, numWords);
        this.dispatchYZ(this.dilateZCompute, this.bufferB, this.bufferA,
            numXWords, ny, nz, halfExtentXZ, numXWords * ny, nz);

        // Y-pass: A -> B.
        this.dispatchClear(this.bufferB, numWords);
        this.dispatchYZ(this.dilateYCompute, this.bufferA, this.bufferB,
            numXWords, ny, nz, halfExtentY, numXWords, ny);

        const readData = await this.bufferB.read(0, numWords * 4, null, true) as Uint8Array;
        return new Uint32Array(readData.buffer, readData.byteOffset, numWords);
    }

    private dispatchX(
        src: StorageBuffer, dst: StorageBuffer,
        numXWords: number, ny: number, nz: number,
        halfExtent: number
    ): void {
        const c = this.dilateXCompute;
        c.setParameter('src', src);
        c.setParameter('dst', dst);
        c.setParameter('numXWords', numXWords);
        c.setParameter('ny', ny);
        c.setParameter('nz', nz);
        c.setParameter('halfExtent', halfExtent);

        const wgX = Math.ceil(numXWords / 8);
        const wgY = Math.ceil(ny / 4);
        const wgZ = Math.ceil(nz / 8);
        c.setupDispatch(wgX, wgY, wgZ);
        this.device.computeDispatch([c], 'gpu-dilate-x');
    }

    private dispatchYZ(
        compute: Compute,
        src: StorageBuffer, dst: StorageBuffer,
        numXWords: number, ny: number, nz: number,
        halfExtent: number,
        stride: number, axisLen: number
    ): void {
        compute.setParameter('src', src);
        compute.setParameter('dst', dst);
        compute.setParameter('numXWords', numXWords);
        compute.setParameter('ny', ny);
        compute.setParameter('nz', nz);
        compute.setParameter('halfExtent', halfExtent);
        compute.setParameter('stride', stride);
        compute.setParameter('axisLen', axisLen);

        const wgX = Math.ceil(numXWords / 8);
        const wgY = Math.ceil(ny / 4);
        const wgZ = Math.ceil(nz / 8);
        compute.setupDispatch(wgX, wgY, wgZ);
        this.device.computeDispatch([compute], compute.name);
    }

    destroy(): void {
        this.bufferA.destroy();
        this.bufferB.destroy();
        this.dilateXCompute.destroy();
        this.dilateYCompute.destroy();
        this.dilateZCompute.destroy();
        this.clearCompute.destroy();
        this.dilateXShader.destroy();
        this.dilateYZShader.destroy();
        this.clearShader.destroy();
        this.dilateXBindGroupFormat.destroy();
        this.dilateYZBindGroupFormat.destroy();
        this.clearBindGroupFormat.destroy();
    }
}

export { GpuDilation };
