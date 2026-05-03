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

import type { SparseVoxelGrid } from '../voxel/sparse-voxel-grid';

/**
 * Extract shader — converts a `SparseVoxelGrid` (uploaded as types + open-
 * addressed mask hash) directly into a row-aligned dense bit buffer for one
 * outer chunk. One thread per source block in the chunk's outer block range.
 *
 * For MIXED blocks the shader does Fibonacci-hash linear-probe lookup against
 * the uploaded `srcKeys`/`srcLo`/`srcHi` arrays (matches the CPU
 * `BlockMaskMap.slot` formula bit-for-bit). The block's 4×4 X-row pattern
 * lands in a single dense word at bit offset `(blockX*4) & 31`; multiple
 * blocks share the same dense word at non-overlapping bit positions, so the
 * write is `atomicOr`. Caller must clear the dense buffer first.
 * @returns WGSL source for the extract compute shader.
 */
const extractWgsl = () => /* wgsl */`
struct ExtractUniforms {
    minBx: i32, minBy: i32, minBz: i32,
    _pad0: u32,

    outerBx: u32, outerBy: u32, outerBz: u32,
    numXWords: u32,

    srcNbx: u32, srcNby: u32, srcNbz: u32,
    srcBStride: u32,

    srcCapMinusOne: u32,
    _pad1: u32, _pad2: u32, _pad3: u32
}

@group(0) @binding(0) var<uniform> u: ExtractUniforms;
@group(0) @binding(1) var<storage, read> srcTypes: array<u32>;
@group(0) @binding(2) var<storage, read> srcKeys: array<u32>;
@group(0) @binding(3) var<storage, read> srcLo: array<u32>;
@group(0) @binding(4) var<storage, read> srcHi: array<u32>;
@group(0) @binding(5) var<storage, read_write> dstDense: array<atomic<u32>>;

@compute @workgroup_size(8, 4, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= u.outerBx || gid.y >= u.outerBy || gid.z >= u.outerBz) {
        return;
    }

    let chunkBx = i32(gid.x);
    let chunkBy = i32(gid.y);
    let chunkBz = i32(gid.z);

    let globalBx = u.minBx + chunkBx;
    let globalBy = u.minBy + chunkBy;
    let globalBz = u.minBz + chunkBz;

    if (globalBx < 0 || globalBy < 0 || globalBz < 0) { return; }
    if (globalBx >= i32(u.srcNbx) || globalBy >= i32(u.srcNby) || globalBz >= i32(u.srcNbz)) { return; }

    let blockIdx = u32(globalBx) + u32(globalBy) * u.srcNbx + u32(globalBz) * u.srcBStride;

    let typeWord = srcTypes[blockIdx >> 4u];
    let bt = (typeWord >> ((blockIdx & 15u) * 2u)) & 3u;

    if (bt == 0u) { return; }  // BLOCK_EMPTY

    var lo: u32; var hi: u32;
    if (bt == 1u) {  // BLOCK_SOLID
        lo = 0xFFFFFFFFu;
        hi = 0xFFFFFFFFu;
    } else {
        // BLOCK_MIXED → hash lookup. Same Fibonacci constant as CPU.
        var i = (blockIdx * 0x9E3779B9u) & u.srcCapMinusOne;
        loop {
            let k = srcKeys[i];
            if (k == blockIdx) {
                lo = srcLo[i];
                hi = srcHi[i];
                break;
            }
            if (k == 0xFFFFFFFFu) {
                return;  // not found (shouldn't happen for MIXED)
            }
            i = (i + 1u) & u.srcCapMinusOne;
        }
    }

    // Write 64 voxel bits to dense at chunk-local position. dx0 is 4-aligned
    // so each (ly, lz) row's 4 X-bits live in a single dense word.
    let dx0 = u32(chunkBx) * 4u;
    let wordOffsetX = dx0 / 32u;
    let bitShiftX = dx0 & 31u;

    let outerNy = u.outerBy * 4u;
    let planeWords = u.numXWords * outerNy;

    for (var lz: u32 = 0u; lz < 4u; lz = lz + 1u) {
        let dz = u32(chunkBz) * 4u + lz;
        let zBitBase = (lz & 1u) * 16u;
        let word = select(lo, hi, lz >= 2u);
        for (var ly: u32 = 0u; ly < 4u; ly = ly + 1u) {
            let dy = u32(chunkBy) * 4u + ly;
            let bitBase = zBitBase + ly * 4u;
            let pattern = (word >> bitBase) & 0xFu;
            if (pattern == 0u) { continue; }
            let wordIdx = wordOffsetX + dy * u.numXWords + dz * planeWords;
            atomicOr(&dstDense[wordIdx], pattern << bitShiftX);
        }
    }
}
`;

/**
 * Compact shader — converts a dilated dense bit buffer back into per-block
 * `(type, lo, hi)` form for the chunk's INNER block region. One thread per
 * inner block; reads its 16 dense-word patterns to assemble the block's
 * 64-bit mask, classifies as EMPTY/SOLID/MIXED, and writes to two parallel
 * outputs:
 *   - `typesOut`: 2-bit-per-block packed (matches `dst.types` layout).
 *     Multiple threads write the same word, so atomicOr (caller clears).
 *   - `masksOut`: `[lo, hi]` pairs per inner block, indexed by inner-local
 *     block index. Always written (non-atomic; one thread per slot).
 * @returns WGSL source for the compact compute shader.
 */
const compactWgsl = () => /* wgsl */`
struct CompactUniforms {
    haloBx: u32, haloBy: u32, haloBz: u32,
    numXWords: u32,             // outer chunk's

    innerBx: u32, innerBy: u32, innerBz: u32,
    outerBy: u32                 // for plane stride into dilatedDense
}

@group(0) @binding(0) var<uniform> u: CompactUniforms;
@group(0) @binding(1) var<storage, read> dilatedDense: array<u32>;
@group(0) @binding(2) var<storage, read_write> typesOut: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> masksOut: array<u32>;

@compute @workgroup_size(8, 4, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= u.innerBx || gid.y >= u.innerBy || gid.z >= u.innerBz) {
        return;
    }

    let innerBlockIdx = gid.x + gid.y * u.innerBx + gid.z * u.innerBx * u.innerBy;

    // Outer block coords (inner shifted by halo).
    let outerBx = gid.x + u.haloBx;
    let outerBy = gid.y + u.haloBy;
    let outerBz = gid.z + u.haloBz;

    let dx0 = outerBx * 4u;
    let wordOffsetX = dx0 / 32u;
    let bitShiftX = dx0 & 31u;

    let outerNy = u.outerBy * 4u;
    let numXWords = u.numXWords;
    let planeWords = numXWords * outerNy;

    var lo: u32 = 0u;
    var hi: u32 = 0u;
    for (var lz: u32 = 0u; lz < 4u; lz = lz + 1u) {
        let dz = outerBz * 4u + lz;
        let zBitBase = (lz & 1u) * 16u;
        let inHi = lz >= 2u;
        let planeBase = dz * planeWords;
        for (var ly: u32 = 0u; ly < 4u; ly = ly + 1u) {
            let dy = outerBy * 4u + ly;
            let bitBase = zBitBase + ly * 4u;
            let wordIdx = wordOffsetX + dy * numXWords + planeBase;
            let pattern = (dilatedDense[wordIdx] >> bitShiftX) & 0xFu;
            let bits = pattern << bitBase;
            if (inHi) { hi = hi | bits; }
            else { lo = lo | bits; }
        }
    }

    masksOut[innerBlockIdx * 2u] = lo;
    masksOut[innerBlockIdx * 2u + 1u] = hi;

    var bt: u32 = 0u;  // EMPTY
    if (lo != 0u || hi != 0u) {
        if (lo == 0xFFFFFFFFu && hi == 0xFFFFFFFFu) {
            bt = 1u;  // SOLID
        } else {
            bt = 2u;  // MIXED
        }
    }

    let typeWordIdx = innerBlockIdx >> 4u;
    let typeBitShift = (innerBlockIdx & 15u) * 2u;
    atomicOr(&typesOut[typeWordIdx], bt << typeBitShift);
}
`;

/**
 * Clear shader — writes 0 to every word in the destination buffer up to
 * `numWords`. Dispatched in the same command encoder as the dilation passes
 * so it's ordered with them on the GPU; using `queue.writeBuffer` for inter-
 * pass clears would race because writes are queued separately from encoder
 * commands and execute *all writes first*, then the command buffer.
 * @returns WGSL source for the clear compute shader.
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
 * `[0, 31]`) is the OR of input bits in `[xWord*32 + b - r, xWord*32 + b + r]`.
 * For each distance `d` in `[1, r]`, the shader reads the source word(s)
 * containing bits shifted by `d`, so radii can span any number of 32-bit words.
 *
 * Bound by the chunk's `numXWords` (= ceil(nx / 32)). Out-of-bounds neighbors
 * are read as 0.
 * @returns WGSL source for the X-axis dilation compute shader.
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

fn readWord(rowOffset: u32, word: i32) -> u32 {
    if (word < 0 || word >= i32(u.numXWords)) {
        return 0u;
    }
    return src[rowOffset + u32(word)];
}

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

    var output: u32 = src[rowOffset + xWord];
    let rowBits = u.numXWords * 32u;
    let r = min(u.halfExtent, rowBits);
    for (var d: u32 = 1u; d <= r; d = d + 1u) {
        let wordOffset = i32(d >> 5u);
        let bitShift = d & 31u;
        let baseWord = i32(xWord);

        // Rightward shift: bit b ← input bit b+d.
        var shifted_pos = readWord(rowOffset, baseWord + wordOffset);
        if (bitShift != 0u) {
            shifted_pos = (shifted_pos >> bitShift) |
                (readWord(rowOffset, baseWord + wordOffset + 1) << (32u - bitShift));
        }

        // Leftward shift: bit b ← input bit b-d.
        var shifted_neg = readWord(rowOffset, baseWord - wordOffset);
        if (bitShift != 0u) {
            shifted_neg = (shifted_neg << bitShift) |
                (readWord(rowOffset, baseWord - wordOffset - 1) >> (32u - bitShift));
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
 * @returns WGSL source for the Y/Z-axis dilation compute shader.
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
 * One double-buffered slot — the four `Compute` instances (X, Y, Z, clear)
 * each own a uniform buffer that mustn't be overwritten by a sibling
 * dispatch on the same submit, plus the ping-pong storage buffers. Two
 * slots let the CPU prepare chunk N+1 while the GPU is busy with chunk N.
 */
interface DilationSlot {
    bufferA: StorageBuffer;
    bufferB: StorageBuffer;
    capacity: number;
    dilateXCompute: Compute;
    dilateYCompute: Compute;
    dilateZCompute: Compute;
    clearCompute: Compute;
    extractCompute: Compute;
    compactCompute: Compute;

    typesOutBuffer: StorageBuffer;
    masksOutBuffer: StorageBuffer;
    typesOutCapacity: number;
    masksOutCapacity: number;
}

/**
 * Separable 3D dilation on the GPU using a row-aligned dense bit grid
 * (1 bit per voxel, packed into u32 words; each row of bits along X starts
 * on a word boundary so per-word access is trivial). Each pass owns its
 * own `Compute` instance because their uniform buffers must not collide
 * within a single submit.
 */
class GpuDilation {
    private device: GraphicsDevice;
    private dilateXShader: Shader;
    private dilateYZShader: Shader;
    private clearShader: Shader;
    private extractShader: Shader;
    private compactShader: Shader;
    private dilateXBindGroupFormat: BindGroupFormat;
    private dilateYZBindGroupFormat: BindGroupFormat;
    private clearBindGroupFormat: BindGroupFormat;
    private extractBindGroupFormat: BindGroupFormat;
    private compactBindGroupFormat: BindGroupFormat;

    private slots: DilationSlot[];

    // Source SparseVoxelGrid uploaded once per `gpuDilate3` call (shared
    // across all chunks and slots). `extract` shaders read these.
    private srcTypesBuffer: StorageBuffer | null = null;
    private srcKeysBuffer: StorageBuffer | null = null;
    private srcLoBuffer: StorageBuffer | null = null;
    private srcHiBuffer: StorageBuffer | null = null;
    private srcTypesCapacity = 0;
    private srcMasksCapacity = 0;
    private srcMeta = { nbx: 0, nby: 0, nbz: 0, bStride: 0, capMinusOne: 0 };

    /** Number of double-buffered dispatch slots. */
    static readonly NUM_SLOTS = 2;

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

        this.extractBindGroupFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('srcTypes', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('srcKeys', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('srcLo', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('srcHi', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('dstDense', SHADERSTAGE_COMPUTE)
        ]);

        this.compactBindGroupFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('dilatedDense', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('typesOut', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('masksOut', SHADERSTAGE_COMPUTE)
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

        this.extractShader = new Shader(device, {
            name: 'gpu-dilation-extract',
            shaderLanguage: SHADERLANGUAGE_WGSL,
            cshader: extractWgsl(),
            // @ts-ignore
            computeUniformBufferFormats: {
                uniforms: new UniformBufferFormat(device, [
                    new UniformFormat('minBx', UNIFORMTYPE_UINT),  // signed reinterpret
                    new UniformFormat('minBy', UNIFORMTYPE_UINT),
                    new UniformFormat('minBz', UNIFORMTYPE_UINT),
                    new UniformFormat('_pad0', UNIFORMTYPE_UINT),
                    new UniformFormat('outerBx', UNIFORMTYPE_UINT),
                    new UniformFormat('outerBy', UNIFORMTYPE_UINT),
                    new UniformFormat('outerBz', UNIFORMTYPE_UINT),
                    new UniformFormat('numXWords', UNIFORMTYPE_UINT),
                    new UniformFormat('srcNbx', UNIFORMTYPE_UINT),
                    new UniformFormat('srcNby', UNIFORMTYPE_UINT),
                    new UniformFormat('srcNbz', UNIFORMTYPE_UINT),
                    new UniformFormat('srcBStride', UNIFORMTYPE_UINT),
                    new UniformFormat('srcCapMinusOne', UNIFORMTYPE_UINT)
                ])
            },
            // @ts-ignore
            computeBindGroupFormat: this.extractBindGroupFormat
        });

        this.compactShader = new Shader(device, {
            name: 'gpu-dilation-compact',
            shaderLanguage: SHADERLANGUAGE_WGSL,
            cshader: compactWgsl(),
            // @ts-ignore
            computeUniformBufferFormats: {
                uniforms: new UniformBufferFormat(device, [
                    new UniformFormat('haloBx', UNIFORMTYPE_UINT),
                    new UniformFormat('haloBy', UNIFORMTYPE_UINT),
                    new UniformFormat('haloBz', UNIFORMTYPE_UINT),
                    new UniformFormat('numXWords', UNIFORMTYPE_UINT),
                    new UniformFormat('innerBx', UNIFORMTYPE_UINT),
                    new UniformFormat('innerBy', UNIFORMTYPE_UINT),
                    new UniformFormat('innerBz', UNIFORMTYPE_UINT),
                    new UniformFormat('outerBy', UNIFORMTYPE_UINT)
                ])
            },
            // @ts-ignore
            computeBindGroupFormat: this.compactBindGroupFormat
        });

        this.slots = [];
        for (let i = 0; i < GpuDilation.NUM_SLOTS; i++) {
            const initialCapacity = 1024 * 1024 * 4;
            const initialTypesOut = 64 * 1024;       // 16K blocks worth packed
            const initialMasksOut = 1024 * 1024;     // 128K blocks worth (lo, hi)
            this.slots.push({
                bufferA: new StorageBuffer(device, initialCapacity, BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC),
                bufferB: new StorageBuffer(device, initialCapacity, BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC),
                capacity: initialCapacity,
                dilateXCompute: new Compute(device, this.dilateXShader, `gpu-dilate-x-${i}`),
                dilateYCompute: new Compute(device, this.dilateYZShader, `gpu-dilate-y-${i}`),
                dilateZCompute: new Compute(device, this.dilateYZShader, `gpu-dilate-z-${i}`),
                clearCompute: new Compute(device, this.clearShader, `gpu-dilate-clear-${i}`),
                extractCompute: new Compute(device, this.extractShader, `gpu-dilate-extract-${i}`),
                compactCompute: new Compute(device, this.compactShader, `gpu-dilate-compact-${i}`),
                typesOutBuffer: new StorageBuffer(device, initialTypesOut, BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC),
                masksOutBuffer: new StorageBuffer(device, initialMasksOut, BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC),
                typesOutCapacity: initialTypesOut,
                masksOutCapacity: initialMasksOut
            });
        }
    }

    private ensureSlotBuffers(slot: DilationSlot, numWords: number): void {
        const neededBytes = numWords * 4;
        if (neededBytes <= slot.capacity) return;

        let cap = slot.capacity;
        while (cap < neededBytes) cap *= 2;

        slot.bufferA.destroy();
        slot.bufferB.destroy();
        slot.bufferA = new StorageBuffer(this.device, cap, BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC);
        slot.bufferB = new StorageBuffer(this.device, cap, BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC);
        slot.capacity = cap;
    }

    /**
     * Dispatch a compute clear of `dst` to zero for the first `numWords` words.
     * Uses the command encoder so it's correctly ordered with subsequent
     * dilation passes (unlike `queue.writeBuffer`, which is queued separately
     * and would race against the dispatches).
     * @param slot - Per-chunk slot whose `clearCompute` pipeline is dispatched.
     * @param dst - Destination buffer to zero.
     * @param numWords - Number of leading u32 words to clear.
     */
    private dispatchClear(slot: DilationSlot, dst: StorageBuffer, numWords: number): void {
        const totalWg = Math.ceil(numWords / 256);
        const MAX_DIM = 65535;
        const wgX = Math.min(totalWg, MAX_DIM);
        const wgY = Math.ceil(totalWg / wgX);
        const rowStride = wgX * 256;

        slot.clearCompute.setParameter('clearDst', dst);
        slot.clearCompute.setParameter('clearNumWords', numWords);
        slot.clearCompute.setParameter('clearRowStride', rowStride);
        slot.clearCompute.setupDispatch(wgX, wgY, 1);
        this.device.computeDispatch([slot.clearCompute], 'gpu-dilate-clear');
    }

    /**
     * Upload a `SparseVoxelGrid` to GPU storage buffers used by the extract
     * shader. Reuses the existing buffers if they're large enough; otherwise
     * destroys and reallocates. Designed to be called once per
     * `gpuDilate3` call (the same `src` is read across all chunks).
     * @param src - Source sparse grid to upload.
     */
    uploadSrc(src: SparseVoxelGrid): void {
        const types = src.types;
        const keys = src.masks.keys;     // Int32Array; -1 sentinel reads as 0xFFFFFFFF when interpreted as u32
        const lo = src.masks.lo;
        const hi = src.masks.hi;

        const typesBytes = types.byteLength;
        if (this.srcTypesBuffer === null || this.srcTypesCapacity < typesBytes) {
            this.srcTypesBuffer?.destroy();
            this.srcTypesBuffer = new StorageBuffer(this.device, typesBytes, BUFFERUSAGE_COPY_DST);
            this.srcTypesCapacity = typesBytes;
        }
        this.srcTypesBuffer.write(0, types, 0, types.length);

        const masksBytes = keys.byteLength;
        if (this.srcKeysBuffer === null || this.srcMasksCapacity < masksBytes) {
            this.srcKeysBuffer?.destroy();
            this.srcLoBuffer?.destroy();
            this.srcHiBuffer?.destroy();
            this.srcKeysBuffer = new StorageBuffer(this.device, masksBytes, BUFFERUSAGE_COPY_DST);
            this.srcLoBuffer = new StorageBuffer(this.device, masksBytes, BUFFERUSAGE_COPY_DST);
            this.srcHiBuffer = new StorageBuffer(this.device, masksBytes, BUFFERUSAGE_COPY_DST);
            this.srcMasksCapacity = masksBytes;
        }
        // Treat keys (Int32) as Uint32 — same byte pattern; -1 reads as 0xFFFFFFFF.
        const keysU32 = new Uint32Array(keys.buffer, keys.byteOffset, keys.length);
        this.srcKeysBuffer.write(0, keysU32, 0, keys.length);
        this.srcLoBuffer.write(0, lo, 0, lo.length);
        this.srcHiBuffer.write(0, hi, 0, hi.length);

        this.srcMeta = {
            nbx: src.nbx,
            nby: src.nby,
            nbz: src.nbz,
            bStride: src.bStride,
            capMinusOne: keys.length - 1
        };
    }

    /** Free uploaded `src` buffers. Caller can call after `gpuDilate3` finishes. */
    releaseSrc(): void {
        this.srcTypesBuffer?.destroy();
        this.srcKeysBuffer?.destroy();
        this.srcLoBuffer?.destroy();
        this.srcHiBuffer?.destroy();
        this.srcTypesBuffer = null;
        this.srcKeysBuffer = null;
        this.srcLoBuffer = null;
        this.srcHiBuffer = null;
        this.srcTypesCapacity = 0;
        this.srcMasksCapacity = 0;
    }

    private ensureSlotOutputBuffers(slot: DilationSlot, innerBlocks: number): void {
        // typesOut: 2 bits per inner block, packed into u32 words.
        const typesBytes = (((innerBlocks + 15) >>> 4) * 4);
        if (slot.typesOutCapacity < typesBytes) {
            slot.typesOutBuffer.destroy();
            slot.typesOutBuffer = new StorageBuffer(this.device, typesBytes, BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC);
            slot.typesOutCapacity = typesBytes;
        }
        // masksOut: (lo, hi) per inner block.
        const masksBytes = innerBlocks * 8;
        if (slot.masksOutCapacity < masksBytes) {
            slot.masksOutBuffer.destroy();
            slot.masksOutBuffer = new StorageBuffer(this.device, masksBytes, BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC);
            slot.masksOutCapacity = masksBytes;
        }
    }

    /**
     * Sparse-path submit. Reads from the previously-uploaded `src` (via
     * `uploadSrc`), runs extract → dilate → compact as GPU passes, and returns
     * Promises for the per-block `typesOut` (packed 2-bit) and `masksOut`
     * (lo/hi pairs). Caller integrates these into `dst` directly.
     * @param slotIdx - Round-robin slot index (`0..NUM_SLOTS-1`).
     * @param minBx - Outer chunk origin block X (in `src`'s block coords).
     * @param minBy - Outer chunk origin block Y.
     * @param minBz - Outer chunk origin block Z.
     * @param outerBx - Outer chunk size in blocks along X.
     * @param outerBy - Outer chunk size in blocks along Y.
     * @param outerBz - Outer chunk size in blocks along Z.
     * @param haloBx - Halo size in blocks along X (one side).
     * @param haloBy - Halo size in blocks along Y (one side).
     * @param haloBz - Halo size in blocks along Z (one side).
     * @param innerBx - Inner (output) region size in blocks along X.
     * @param innerBy - Inner region size in blocks along Y.
     * @param innerBz - Inner region size in blocks along Z.
     * @param halfExtentXZ - Dilation half-extent in voxels along X and Z.
     * @param halfExtentY - Dilation half-extent in voxels along Y.
     * @returns Promises for the inner region's packed types and `[lo, hi]` masks.
     */
    submitChunkSparse(
        slotIdx: number,
        // outer chunk in block coords (each is voxel/4)
        minBx: number, minBy: number, minBz: number,
        outerBx: number, outerBy: number, outerBz: number,
        // halo in blocks
        haloBx: number, haloBy: number, haloBz: number,
        // inner chunk in block coords
        innerBx: number, innerBy: number, innerBz: number,
        halfExtentXZ: number,
        halfExtentY: number
    ): { types: Promise<Uint32Array>, masks: Promise<Uint32Array> } {
        if (this.srcTypesBuffer === null) {
            throw new Error('GpuDilation: must call uploadSrc() before submitChunkSparse()');
        }
        const slot = this.slots[slotIdx];

        const outerNx = outerBx * 4;
        const outerNy = outerBy * 4;
        const outerNz = outerBz * 4;
        const numXWords = (outerNx + 31) >>> 5;
        const numWords = numXWords * outerNy * outerNz;
        this.ensureSlotBuffers(slot, numWords);

        const innerBlocks = innerBx * innerBy * innerBz;
        this.ensureSlotOutputBuffers(slot, innerBlocks);

        const typesOutWords = (innerBlocks + 15) >>> 4;

        // Extract: clear bufferA, dispatch extract from sparse src into bufferA.
        this.dispatchClear(slot, slot.bufferA, numWords);
        this.dispatchExtract(slot, minBx, minBy, minBz, outerBx, outerBy, outerBz, numXWords);

        // Tiny throwaway readback forces a queue submit between extract
        // and dilate. WITHOUT this, atomicOr writes from the extract pass
        // aren't reliably visible to the next dilate pass (apparent
        // missing memory barrier in PlayCanvas/Dawn for cross-pass atomic
        // writes). Promise is intentionally not awaited — only the
        // implicit submit+barrier matters. 16 bytes ≈ negligible PCIe.
        slot.bufferA.read(0, 16, null, true).catch(() => { /* ignore */ });

        // X-pass: A → B
        this.dispatchX(slot, slot.bufferA, slot.bufferB, numXWords, outerNy, outerNz, halfExtentXZ);

        // Z-pass: B → A
        this.dispatchYZ(slot.dilateZCompute, slot.bufferB, slot.bufferA,
            numXWords, outerNy, outerNz, halfExtentXZ, numXWords * outerNy, outerNz);

        // Y-pass: A → B
        this.dispatchYZ(slot.dilateYCompute, slot.bufferA, slot.bufferB,
            numXWords, outerNy, outerNz, halfExtentY, numXWords, outerNy);

        // Compact: clear typesOut (atomicOr accumulates), dispatch compact.
        // masksOut is written non-atomically — no clear needed.
        this.dispatchClear(slot, slot.typesOutBuffer, typesOutWords);
        this.dispatchCompact(slot, haloBx, haloBy, haloBz, innerBx, innerBy, innerBz, numXWords, outerBy);

        const typesPromise = slot.typesOutBuffer.read(0, typesOutWords * 4, null, true)
        .then((readData: Uint8Array) => new Uint32Array(readData.buffer, readData.byteOffset, typesOutWords));
        const masksPromise = slot.masksOutBuffer.read(0, innerBlocks * 8, null, true)
        .then((readData: Uint8Array) => new Uint32Array(readData.buffer, readData.byteOffset, innerBlocks * 2));

        return { types: typesPromise, masks: masksPromise };
    }

    private dispatchExtract(
        slot: DilationSlot,
        minBx: number, minBy: number, minBz: number,
        outerBx: number, outerBy: number, outerBz: number,
        numXWords: number
    ): void {
        const c = slot.extractCompute;
        c.setParameter('srcTypes', this.srcTypesBuffer!);
        c.setParameter('srcKeys', this.srcKeysBuffer!);
        c.setParameter('srcLo', this.srcLoBuffer!);
        c.setParameter('srcHi', this.srcHiBuffer!);
        c.setParameter('dstDense', slot.bufferA);
        // Reinterpret signed minB* as u32 bits via the i32 parameter slot —
        // PlayCanvas treats UNIFORMTYPE_UINT as raw u32, and the WGSL struct
        // declares these as i32 so they read back signed.
        c.setParameter('minBx', (minBx >>> 0));
        c.setParameter('minBy', (minBy >>> 0));
        c.setParameter('minBz', (minBz >>> 0));
        c.setParameter('_pad0', 0);
        c.setParameter('outerBx', outerBx);
        c.setParameter('outerBy', outerBy);
        c.setParameter('outerBz', outerBz);
        c.setParameter('numXWords', numXWords);
        c.setParameter('srcNbx', this.srcMeta.nbx);
        c.setParameter('srcNby', this.srcMeta.nby);
        c.setParameter('srcNbz', this.srcMeta.nbz);
        c.setParameter('srcBStride', this.srcMeta.bStride);
        c.setParameter('srcCapMinusOne', this.srcMeta.capMinusOne);

        const wgX = Math.ceil(outerBx / 8);
        const wgY = Math.ceil(outerBy / 4);
        const wgZ = Math.ceil(outerBz / 8);
        c.setupDispatch(wgX, wgY, wgZ);
        this.device.computeDispatch([c], c.name);
    }

    private dispatchCompact(
        slot: DilationSlot,
        haloBx: number, haloBy: number, haloBz: number,
        innerBx: number, innerBy: number, innerBz: number,
        numXWords: number,
        outerBy: number
    ): void {
        const c = slot.compactCompute;
        c.setParameter('dilatedDense', slot.bufferB);
        c.setParameter('typesOut', slot.typesOutBuffer);
        c.setParameter('masksOut', slot.masksOutBuffer);
        c.setParameter('haloBx', haloBx);
        c.setParameter('haloBy', haloBy);
        c.setParameter('haloBz', haloBz);
        c.setParameter('numXWords', numXWords);
        c.setParameter('innerBx', innerBx);
        c.setParameter('innerBy', innerBy);
        c.setParameter('innerBz', innerBz);
        c.setParameter('outerBy', outerBy);

        const wgX = Math.ceil(innerBx / 8);
        const wgY = Math.ceil(innerBy / 4);
        const wgZ = Math.ceil(innerBz / 8);
        c.setupDispatch(wgX, wgY, wgZ);
        this.device.computeDispatch([c], c.name);
    }

    private dispatchX(
        slot: DilationSlot,
        src: StorageBuffer, dst: StorageBuffer,
        numXWords: number, ny: number, nz: number,
        halfExtent: number
    ): void {
        const c = slot.dilateXCompute;
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
        this.device.computeDispatch([c], c.name);
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
        this.releaseSrc();
        for (const slot of this.slots) {
            slot.bufferA.destroy();
            slot.bufferB.destroy();
            slot.typesOutBuffer.destroy();
            slot.masksOutBuffer.destroy();
            slot.dilateXCompute.destroy();
            slot.dilateYCompute.destroy();
            slot.dilateZCompute.destroy();
            slot.clearCompute.destroy();
            slot.extractCompute.destroy();
            slot.compactCompute.destroy();
        }
        this.dilateXShader.destroy();
        this.dilateYZShader.destroy();
        this.clearShader.destroy();
        this.extractShader.destroy();
        this.compactShader.destroy();
        this.dilateXBindGroupFormat.destroy();
        this.dilateYZBindGroupFormat.destroy();
        this.clearBindGroupFormat.destroy();
        this.extractBindGroupFormat.destroy();
        this.compactBindGroupFormat.destroy();
    }
}

export { GpuDilation };
