/**
 * WGSL sources for the 5 compute shaders that make up the GPU dilation
 * pipeline (extract → clear → dilateX → dilateZ → dilateY → compact),
 * plus the small block-type + Fibonacci-hash constants block that the
 * extract and compact shaders share. Plain TS template-string
 * composition — the constants block is interpolated into each consuming
 * shader via `${dilationConstants}` rather than going through the
 * engine's `#include` preprocessor.
 *
 * The orchestrator class lives in `gpu-dilation.ts` and imports each
 * `xxxWgsl()` generator individually.
 */

/**
 * Block-type enum and Fibonacci-hash constant, shared between the
 * extract and compact shaders. Mirrors the values used by the CPU
 * `BlockMaskMap.slot` formula so the GPU's hash probe lands in the same
 * slot.
 */
const dilationConstants = /* wgsl */`
const BLOCK_EMPTY: u32 = 0u;
const BLOCK_SOLID: u32 = 1u;
const BLOCK_MIXED: u32 = 2u;
const FIBONACCI_HASH: u32 = 0x9E3779B9u;
`;

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
 *
 * @returns WGSL source for the extract compute shader.
 */
const extractWgsl = () => /* wgsl */`
${dilationConstants}

struct ExtractUniforms {
    minBx: i32, minBy: i32, minBz: i32,

    outerBx: u32, outerBy: u32, outerBz: u32,
    numXWords: u32,

    srcNbx: u32, srcNby: u32, srcNbz: u32,
    srcBStride: u32,

    srcCapMinusOne: u32
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

    if (bt == BLOCK_EMPTY) { return; }

    var lo: u32; var hi: u32;
    if (bt == BLOCK_SOLID) {
        lo = 0xFFFFFFFFu;
        hi = 0xFFFFFFFFu;
    } else {
        // BLOCK_MIXED → hash lookup. Same constant as CPU BlockMaskMap.
        var i = (blockIdx * FIBONACCI_HASH) & u.srcCapMinusOne;
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
 *
 * @returns WGSL source for the compact compute shader.
 */
const compactWgsl = () => /* wgsl */`
${dilationConstants}

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

    var bt: u32 = BLOCK_EMPTY;
    if (lo != 0u || hi != 0u) {
        if (lo == 0xFFFFFFFFu && hi == 0xFFFFFFFFu) {
            bt = BLOCK_SOLID;
        } else {
            bt = BLOCK_MIXED;
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
 *
 * @returns WGSL source for the clear compute shader.
 */
const clearWgsl = () => /* wgsl */`
struct ClearUniforms {
    clearNumWords: u32,
    clearRowStride: u32
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
 *
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
 *
 * @returns WGSL source for the Y/Z-axis dilation compute shader.
 */
const dilateYZWgsl = () => /* wgsl */`
struct DilateYZUniforms {
    numXWords: u32,
    ny: u32,
    nz: u32,
    halfExtent: u32,
    stride: u32,
    axisLen: u32
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

export { extractWgsl, compactWgsl, clearWgsl, dilateXWgsl, dilateYZWgsl };
