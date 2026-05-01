import {
    BLOCK_EMPTY,
    BLOCK_SOLID,
    BLOCKS_PER_WORD,
    EVEN_BITS,
    SOLID_HI,
    SOLID_LO,
    SOLID_WORD,
    SparseVoxelGrid,
    readBlockType
} from './sparse-voxel-grid';
import { GpuDilation } from '../gpu';
import { logger } from '../utils';

// ============================================================================
// Active Pair Computation
//
// Each function iterates `grid.types` (packed 2-bit blockTypes, 16 blocks
// per word) and derives a per-word "non-empty lane" mask using the trick
//   nonEmpty = (word & 0x55555555) | ((word >>> 1) & 0x55555555)
// which sets bit 2k iff lane k is non-zero. The set bits are then walked
// the same way the old occupancy bitfield was.
// ============================================================================

function getActiveYZPairs(grid: SparseVoxelGrid): Set<number> {
    const pairs = new Set<number>();
    const { nbx } = grid;
    const totalBlocks = grid.nbx * grid.nby * grid.nbz;
    const types = grid.types;
    for (let w = 0; w < types.length; w++) {
        const word = types[w];
        if (word === 0) continue;
        let nonEmpty = ((word & EVEN_BITS) | ((word >>> 1) & EVEN_BITS)) >>> 0;
        const baseIdx = w * BLOCKS_PER_WORD;
        while (nonEmpty) {
            const bp = 31 - Math.clz32(nonEmpty & -nonEmpty);
            const blockIdx = baseIdx + (bp >>> 1);
            if (blockIdx < totalBlocks) {
                pairs.add((blockIdx / nbx) | 0);
            }
            nonEmpty &= nonEmpty - 1;
        }
    }
    return pairs;
}

function getActiveXZPairs(grid: SparseVoxelGrid): Set<number> {
    const pairs = new Set<number>();
    const { nbx, bStride } = grid;
    const totalBlocks = grid.nbx * grid.nby * grid.nbz;
    const types = grid.types;
    for (let w = 0; w < types.length; w++) {
        const word = types[w];
        if (word === 0) continue;
        let nonEmpty = ((word & EVEN_BITS) | ((word >>> 1) & EVEN_BITS)) >>> 0;
        const baseIdx = w * BLOCKS_PER_WORD;
        while (nonEmpty) {
            const bp = 31 - Math.clz32(nonEmpty & -nonEmpty);
            const blockIdx = baseIdx + (bp >>> 1);
            if (blockIdx < totalBlocks) {
                const bx = blockIdx % nbx;
                const bz = (blockIdx / bStride) | 0;
                pairs.add(bx + bz * nbx);
            }
            nonEmpty &= nonEmpty - 1;
        }
    }
    return pairs;
}

function getActiveXYPairs(grid: SparseVoxelGrid): Set<number> {
    const pairs = new Set<number>();
    const { nbx, nby } = grid;
    const totalBlocks = grid.nbx * grid.nby * grid.nbz;
    const types = grid.types;
    for (let w = 0; w < types.length; w++) {
        const word = types[w];
        if (word === 0) continue;
        let nonEmpty = ((word & EVEN_BITS) | ((word >>> 1) & EVEN_BITS)) >>> 0;
        const baseIdx = w * BLOCKS_PER_WORD;
        while (nonEmpty) {
            const bp = 31 - Math.clz32(nonEmpty & -nonEmpty);
            const blockIdx = baseIdx + (bp >>> 1);
            if (blockIdx < totalBlocks) {
                const bx = blockIdx % nbx;
                const by = ((blockIdx / nbx) | 0) % nby;
                pairs.add(bx + by * nbx);
            }
            nonEmpty &= nonEmpty - 1;
        }
    }
    return pairs;
}

// ============================================================================
// Line Extraction / Write-back
//
// `readBlockType` reads a packed 2-bit block type. As a module-level arrow
// function it inlines under V8's JIT, matching the perf of the previous
// direct Uint8Array read modulo the bit math.
// ============================================================================

function extractLineX(grid: SparseVoxelGrid, iy: number, iz: number, buf: Uint32Array): void {
    const by = iy >> 2, bz = iz >> 2;
    const bitBase = ((iz & 3) << 4) + ((iy & 3) << 2);
    const inHi = bitBase >= 32;
    const shift = inHi ? bitBase - 32 : bitBase;
    const lineBase = by * grid.nbx + bz * grid.bStride;
    const types = grid.types;
    for (let bx = 0; bx < grid.nbx; bx++) {
        const blockIdx = lineBase + bx;
        const bt = readBlockType(types, blockIdx);
        if (bt === BLOCK_EMPTY) continue;
        let row4: number;
        if (bt === BLOCK_SOLID) {
            row4 = 0xF;
        } else {
            const s = grid.masks.slot(blockIdx);
            row4 = ((inHi ? grid.masks.hi[s] : grid.masks.lo[s]) >>> shift) & 0xF;
        }
        if (row4) {
            const ix = bx << 2;
            buf[ix >>> 5] |= (row4 << (ix & 31));
        }
    }
}

function writeLineX(grid: SparseVoxelGrid, iy: number, iz: number, buf: Uint32Array): void {
    const by = iy >> 2, bz = iz >> 2;
    const bitBase = ((iz & 3) << 4) + ((iy & 3) << 2);
    const inHi = bitBase >= 32;
    const shift = inHi ? bitBase - 32 : bitBase;
    const lineBase = by * grid.nbx + bz * grid.bStride;
    for (let bx = 0; bx < grid.nbx; bx++) {
        const ix = bx << 2;
        const row4 = (buf[ix >>> 5] >>> (ix & 31)) & 0xF;
        if (!row4) continue;
        const blockIdx = lineBase + bx;
        grid.orBlock(blockIdx,
            inHi ? 0 : (row4 << shift) >>> 0,
            inHi ? (row4 << shift) >>> 0 : 0
        );
    }
}

function extractLineY(grid: SparseVoxelGrid, ix: number, iz: number, buf: Uint32Array): void {
    const bx = ix >> 2, bz = iz >> 2;
    const lx = ix & 3, lz = iz & 3;
    const inHi = lz >= 2;
    const base = lx + (lz & 1) * 16;
    const types = grid.types;
    for (let by = 0; by < grid.nby; by++) {
        const blockIdx = bx + by * grid.nbx + bz * grid.bStride;
        const bt = readBlockType(types, blockIdx);
        if (bt === BLOCK_EMPTY) continue;
        let row4: number;
        if (bt === BLOCK_SOLID) {
            row4 = 0xF;
        } else {
            const s = grid.masks.slot(blockIdx);
            const word = inHi ? grid.masks.hi[s] : grid.masks.lo[s];
            row4 = ((word >>> base) & 1) |
                   (((word >>> (base + 4)) & 1) << 1) |
                   (((word >>> (base + 8)) & 1) << 2) |
                   (((word >>> (base + 12)) & 1) << 3);
        }
        if (row4) {
            const iy = by << 2;
            buf[iy >>> 5] |= (row4 << (iy & 31));
        }
    }
}

function writeLineY(grid: SparseVoxelGrid, ix: number, iz: number, buf: Uint32Array): void {
    const bx = ix >> 2, bz = iz >> 2;
    const lx = ix & 3, lz = iz & 3;
    const inHi = lz >= 2;
    const base = lx + (lz & 1) * 16;
    for (let by = 0; by < grid.nby; by++) {
        const iy = by << 2;
        const row4 = (buf[iy >>> 5] >>> (iy & 31)) & 0xF;
        if (!row4) continue;
        const blockIdx = bx + by * grid.nbx + bz * grid.bStride;
        const bits = ((row4 & 1) << base) |
                     (((row4 >>> 1) & 1) << (base + 4)) |
                     (((row4 >>> 2) & 1) << (base + 8)) |
                     (((row4 >>> 3) & 1) << (base + 12));
        grid.orBlock(blockIdx,
            inHi ? 0 : bits >>> 0,
            inHi ? bits >>> 0 : 0
        );
    }
}

function extractLineZ(grid: SparseVoxelGrid, ix: number, iy: number, buf: Uint32Array): void {
    const bx = ix >> 2, by = iy >> 2;
    const base = (ix & 3) + ((iy & 3) << 2);
    const types = grid.types;
    for (let bz = 0; bz < grid.nbz; bz++) {
        const blockIdx = bx + by * grid.nbx + bz * grid.bStride;
        const bt = readBlockType(types, blockIdx);
        if (bt === BLOCK_EMPTY) continue;
        let row4: number;
        if (bt === BLOCK_SOLID) {
            row4 = 0xF;
        } else {
            const s = grid.masks.slot(blockIdx);
            row4 = ((grid.masks.lo[s] >>> base) & 1) |
                   (((grid.masks.lo[s] >>> (base + 16)) & 1) << 1) |
                   (((grid.masks.hi[s] >>> base) & 1) << 2) |
                   (((grid.masks.hi[s] >>> (base + 16)) & 1) << 3);
        }
        if (row4) {
            const iz = bz << 2;
            buf[iz >>> 5] |= (row4 << (iz & 31));
        }
    }
}

function writeLineZ(grid: SparseVoxelGrid, ix: number, iy: number, buf: Uint32Array): void {
    const bx = ix >> 2, by = iy >> 2;
    const base = (ix & 3) + ((iy & 3) << 2);
    for (let bz = 0; bz < grid.nbz; bz++) {
        const iz = bz << 2;
        const row4 = (buf[iz >>> 5] >>> (iz & 31)) & 0xF;
        if (!row4) continue;
        const blockIdx = bx + by * grid.nbx + bz * grid.bStride;
        let lo = 0, hi = 0;
        if (row4 & 1) lo |= (1 << base);
        if (row4 & 2) lo |= (1 << (base + 16));
        if (row4 & 4) hi |= (1 << base);
        if (row4 & 8) hi |= (1 << (base + 16));
        grid.orBlock(blockIdx, lo >>> 0, hi >>> 0);
    }
}

// ============================================================================
// 1D Sliding Window Operations
// ============================================================================

function flatDilate1D(src: Uint32Array, dst: Uint32Array, n: number, halfExtent: number): void {
    let count = 0;
    const winEnd = Math.min(halfExtent, n - 1);
    for (let i = 0; i <= winEnd; i++) {
        if ((src[i >>> 5] >>> (i & 31)) & 1) count++;
    }
    for (let i = 0; i < n; i++) {
        if (count > 0) dst[i >>> 5] |= (1 << (i & 31));
        const exitI = i - halfExtent;
        if (exitI >= 0 && (src[exitI >>> 5] >>> (exitI & 31)) & 1) count--;
        const enterI = i + halfExtent + 1;
        if (enterI < n && (src[enterI >>> 5] >>> (enterI & 31)) & 1) count++;
    }
}

// ============================================================================
// Sparse Dilation
// ============================================================================

function sparseDilateX(src: SparseVoxelGrid, dst: SparseVoxelGrid, halfExtent: number): void {
    const { nx, ny, nz, nbx, nby, bStride } = src;
    const lineWords = (nx + 31) >>> 5;
    const srcBuf = new Uint32Array(lineWords);
    const dstBuf = new Uint32Array(lineWords);
    const srcTypes = src.types;
    const activePairs = getActiveYZPairs(src);
    for (const key of activePairs) {
        const by = key % nby;
        const bz = (key / nby) | 0;

        const lineBase = by * nbx + bz * bStride;
        let allSolid = true;
        for (let bx = 0; bx < nbx; bx++) {
            const idx = lineBase + bx;
            if (readBlockType(srcTypes, idx) !== BLOCK_SOLID) {
                allSolid = false;
                break;
            }
        }
        if (allSolid) {
            for (let bx = 0; bx < nbx; bx++) {
                dst.orBlock(lineBase + bx, SOLID_LO, SOLID_HI);
            }
            continue;
        }

        for (let ly = 0; ly < 4; ly++) {
            const iy = (by << 2) + ly;
            if (iy >= ny) continue;
            for (let lz = 0; lz < 4; lz++) {
                const iz = (bz << 2) + lz;
                if (iz >= nz) continue;
                srcBuf.fill(0);
                dstBuf.fill(0);
                extractLineX(src, iy, iz, srcBuf);
                flatDilate1D(srcBuf, dstBuf, nx, halfExtent);
                writeLineX(dst, iy, iz, dstBuf);
            }
        }
    }
}

function sparseDilateZ(src: SparseVoxelGrid, dst: SparseVoxelGrid, halfExtent: number): void {
    const { nx, ny, nz, nbx, nbz, bStride } = src;
    const lineWords = (nz + 31) >>> 5;
    const srcBuf = new Uint32Array(lineWords);
    const dstBuf = new Uint32Array(lineWords);
    const srcTypes = src.types;
    const activePairs = getActiveXYPairs(src);
    for (const key of activePairs) {
        const bx = key % nbx;
        const by = (key / nbx) | 0;

        const lineStart = bx + by * nbx;
        let allSolid = true;
        for (let bz = 0; bz < nbz; bz++) {
            const idx = lineStart + bz * bStride;
            if (readBlockType(srcTypes, idx) !== BLOCK_SOLID) {
                allSolid = false;
                break;
            }
        }
        if (allSolid) {
            for (let bz = 0; bz < nbz; bz++) {
                dst.orBlock(lineStart + bz * bStride, SOLID_LO, SOLID_HI);
            }
            continue;
        }

        for (let lx = 0; lx < 4; lx++) {
            const ix = (bx << 2) + lx;
            if (ix >= nx) continue;
            for (let ly = 0; ly < 4; ly++) {
                const iy = (by << 2) + ly;
                if (iy >= ny) continue;
                srcBuf.fill(0);
                dstBuf.fill(0);
                extractLineZ(src, ix, iy, srcBuf);
                flatDilate1D(srcBuf, dstBuf, nz, halfExtent);
                writeLineZ(dst, ix, iy, dstBuf);
            }
        }
    }
}

function sparseDilateY(src: SparseVoxelGrid, dst: SparseVoxelGrid, halfExtent: number): void {
    const { nx, ny, nz, nbx, nby, bStride } = src;
    const lineWords = (ny + 31) >>> 5;
    const srcBuf = new Uint32Array(lineWords);
    const dstBuf = new Uint32Array(lineWords);
    const srcTypes = src.types;
    const activePairs = getActiveXZPairs(src);
    for (const key of activePairs) {
        const bx = key % nbx;
        const bz = (key / nbx) | 0;

        const lineStart = bx + bz * bStride;
        let allSolid = true;
        for (let by = 0; by < nby; by++) {
            const idx = lineStart + by * nbx;
            if (readBlockType(srcTypes, idx) !== BLOCK_SOLID) {
                allSolid = false;
                break;
            }
        }
        if (allSolid) {
            for (let by = 0; by < nby; by++) {
                dst.orBlock(lineStart + by * nbx, SOLID_LO, SOLID_HI);
            }
            continue;
        }

        for (let lx = 0; lx < 4; lx++) {
            const ix = (bx << 2) + lx;
            if (ix >= nx) continue;
            for (let lz = 0; lz < 4; lz++) {
                const iz = (bz << 2) + lz;
                if (iz >= nz) continue;
                srcBuf.fill(0);
                dstBuf.fill(0);
                extractLineY(src, ix, iz, srcBuf);
                flatDilate1D(srcBuf, dstBuf, ny, halfExtent);
                writeLineY(dst, ix, iz, dstBuf);
            }
        }
    }
}

/**
 * 3D dilation by separable 1D passes (X then Z then Y).
 *
 * Allocates one fresh working grid (`a`) and uses one more (`b`) for the
 * intermediate. When `consumeSrc` is true the caller relinquishes `src`
 * — it gets cleared and reused as the second working grid, saving one
 * full SparseVoxelGrid allocation per call. After the function returns
 * with `consumeSrc=true`, `src` references an empty grid and must not
 * be read by the caller.
 *
 * @param src - Input grid. Read-only when `consumeSrc=false`; consumed
 * (cleared and used as scratch) when `consumeSrc=true`.
 * @param halfExtentXZ - Dilation half-extent in voxels along X and Z.
 * @param halfExtentY - Dilation half-extent in voxels along Y.
 * @param consumeSrc - If true, the function may reuse `src`'s memory as
 * a working buffer. Saves ~one full grid's worth of allocation. The
 * caller must not read `src` after the call.
 * @returns Newly allocated dilated grid.
 */
function sparseDilate3(
    src: SparseVoxelGrid,
    halfExtentXZ: number,
    halfExtentY: number,
    consumeSrc: boolean = false
): SparseVoxelGrid {
    const { nx, ny, nz } = src;
    const a = new SparseVoxelGrid(nx, ny, nz);
    const bar = logger.bar('Dilating', 3);
    sparseDilateX(src, a, halfExtentXZ);
    bar.tick();
    let b: SparseVoxelGrid;
    if (consumeSrc) {
        src.clear();
        b = src;
    } else {
        b = new SparseVoxelGrid(nx, ny, nz);
    }
    sparseDilateZ(a, b, halfExtentXZ);
    a.clear();
    bar.tick();
    sparseDilateY(b, a, halfExtentY);
    b.clear();
    bar.tick();
    bar.end();
    return a;
}

export { sparseDilate3, gpuDilate3 };

// ============================================================================
// GPU Dilation
//
// Chunked GPU separable 3D dilation that mirrors `sparseDilate3`'s API
// (returns a fresh `SparseVoxelGrid`). Extracts each chunk (with halo) from
// the source as a dense bit grid, runs X/Z/Y compute passes on the GPU
// without a CPU round-trip between them, then OR's the inner-only output
// back into the destination. Source is read-only across the whole call;
// destination is built up chunk by chunk.
// ============================================================================

/** Inner chunk size in voxels per axis (must be a multiple of 4). */
const CHUNK_INNER = 512;

/**
 * When set (e.g. `GPU_DILATE_DEBUG=1`), each `gpuDilate3` chunk additionally
 * runs the CPU `sparseDilate3` on the same input chunk and diffs the dense
 * outputs. Catches GPU/CPU correctness bugs by pinpointing the first chunk
 * (and bit) where they disagree. Slow — disable for non-debug runs.
 */
const GPU_DILATE_DEBUG = (typeof process !== 'undefined' && !!process.env?.GPU_DILATE_DEBUG);

/**
 * Convert a row-aligned dense bit chunk back into a `SparseVoxelGrid` of the
 * chunk's outer dims. Used in `GPU_DILATE_DEBUG` to feed the same input into
 * the CPU dilation for diffing.
 */
function denseToSparseGrid(
    dense: Uint32Array,
    nx: number, ny: number, nz: number, numXWords: number
): SparseVoxelGrid {
    const grid = new SparseVoxelGrid(nx, ny, nz);
    const planeWords = numXWords * ny;
    for (let bz = 0; bz < grid.nbz; bz++) {
        for (let by = 0; by < grid.nby; by++) {
            for (let bx = 0; bx < grid.nbx; bx++) {
                const baseGx = bx * 4;
                const baseGy = by * 4;
                const baseGz = bz * 4;
                let lo = 0;
                let hi = 0;
                for (let lz = 0; lz < 4; lz++) {
                    const gz = baseGz + lz;
                    if (gz >= nz) continue;
                    const inHi = lz >= 2;
                    const zBitBase = (lz & 1) * 16;
                    for (let ly = 0; ly < 4; ly++) {
                        const gy = baseGy + ly;
                        if (gy >= ny) continue;
                        const bitBase = zBitBase + ly * 4;
                        for (let lx = 0; lx < 4; lx++) {
                            const gx = baseGx + lx;
                            if (gx >= nx) continue;
                            const wordIdx = (gx >>> 5) + gy * numXWords + gz * planeWords;
                            if (!((dense[wordIdx] >>> (gx & 31)) & 1)) continue;
                            const bit = 1 << (bitBase + lx);
                            if (inHi) hi |= bit;
                            else lo |= bit;
                        }
                    }
                }
                if (lo || hi) {
                    const blockIdx = bx + by * grid.nbx + bz * grid.bStride;
                    grid.orBlock(blockIdx, lo >>> 0, hi >>> 0);
                }
            }
        }
    }
    return grid;
}

/**
 * Fast empty-chunk check. Scans only `types` (no mask reads) for the source
 * blocks that overlap the chunk's outer region; returns true if every block
 * is `BLOCK_EMPTY`. Lets `gpuDilate3` skip extract / dispatch / insert for
 * chunks far from the scene's occupied region.
 */
function chunkIsEmpty(
    src: SparseVoxelGrid,
    ox: number, oy: number, oz: number,
    cx: number, cy: number, cz: number
): boolean {
    const minBx = Math.max(0, Math.floor(ox / 4));
    const minBy = Math.max(0, Math.floor(oy / 4));
    const minBz = Math.max(0, Math.floor(oz / 4));
    const maxBx = Math.min(src.nbx, Math.ceil((ox + cx) / 4));
    const maxBy = Math.min(src.nby, Math.ceil((oy + cy) / 4));
    const maxBz = Math.min(src.nbz, Math.ceil((oz + cz) / 4));
    if (maxBx <= minBx || maxBy <= minBy || maxBz <= minBz) return true;

    const types = src.types;
    for (let bz = minBz; bz < maxBz; bz++) {
        for (let by = minBy; by < maxBy; by++) {
            for (let bx = minBx; bx < maxBx; bx++) {
                const blockIdx = bx + by * src.nbx + bz * src.bStride;
                if (readBlockType(types, blockIdx) !== BLOCK_EMPTY) {
                    return false;
                }
            }
        }
    }
    return true;
}

/**
 * Fast saturated-chunk check. Returns true if the chunk's outer region is
 * entirely within `src` bounds AND every overlapping block is `BLOCK_SOLID`
 * (so extract would produce all 1s). Dilation of an all-solid input is
 * trivially all-solid, letting the caller skip GPU dispatch and write
 * `BLOCK_SOLID` directly into the destination's inner region.
 */
function chunkIsSaturated(
    src: SparseVoxelGrid,
    ox: number, oy: number, oz: number,
    cx: number, cy: number, cz: number
): boolean {
    // Halo extends past grid → those bits are 0, not saturated.
    if (ox < 0 || oy < 0 || oz < 0) return false;
    if (ox + cx > src.nx || oy + cy > src.ny || oz + cz > src.nz) return false;

    const minBx = ox >> 2;
    const minBy = oy >> 2;
    const minBz = oz >> 2;
    const maxBx = (ox + cx + 3) >> 2;
    const maxBy = (oy + cy + 3) >> 2;
    const maxBz = (oz + cz + 3) >> 2;

    const types = src.types;
    for (let bz = minBz; bz < maxBz; bz++) {
        for (let by = minBy; by < maxBy; by++) {
            for (let bx = minBx; bx < maxBx; bx++) {
                const blockIdx = bx + by * src.nbx + bz * src.bStride;
                if (readBlockType(types, blockIdx) !== BLOCK_SOLID) {
                    return false;
                }
            }
        }
    }
    return true;
}

/**
 * Insert a fully-solid inner region into `dst` without going through the
 * dense path. Used when the source chunk is saturated and dilation is
 * trivially saturated.
 *
 * For each X-row of inner blocks (varying `bx`, fixed `by, bz`), the
 * `dst.types` slots are contiguous; we set 2 bits per block to `BLOCK_SOLID`
 * via word-level writes (`SOLID_WORD = 0x55555555`) instead of millions of
 * `orBlock` calls. Chunks here are always disjoint and dst blocks empty
 * before this call, so no merge logic is needed.
 */
function insertSaturatedInner(
    dst: SparseVoxelGrid,
    innerOx: number, innerOy: number, innerOz: number,
    innerCx: number, innerCy: number, innerCz: number
): void {
    const minBx = Math.max(0, innerOx >> 2);
    const minBy = Math.max(0, innerOy >> 2);
    const minBz = Math.max(0, innerOz >> 2);
    const maxBx = Math.min(dst.nbx, (innerOx + innerCx + 3) >> 2);
    const maxBy = Math.min(dst.nby, (innerOy + innerCy + 3) >> 2);
    const maxBz = Math.min(dst.nbz, (innerOz + innerCz + 3) >> 2);

    const types = dst.types;
    const nbx = dst.nbx;
    const bStride = dst.bStride;

    for (let bz = minBz; bz < maxBz; bz++) {
        for (let by = minBy; by < maxBy; by++) {
            const rowBase = by * nbx + bz * bStride;
            const startIdx = rowBase + minBx;
            const endIdx = rowBase + maxBx; // exclusive
            let blockIdx = startIdx;
            while (blockIdx < endIdx) {
                const w = blockIdx >>> 4;
                const shift = (blockIdx & 15) << 1;
                const remainingInWord = 16 - (blockIdx & 15);
                const remainingInRow = endIdx - blockIdx;
                const blocksToWrite = remainingInWord < remainingInRow ? remainingInWord : remainingInRow;
                if (blocksToWrite === 16) {
                    types[w] = SOLID_WORD;
                } else {
                    const bits = blocksToWrite << 1;
                    const mask = (((1 << bits) - 1) >>> 0) << shift;
                    types[w] = ((types[w] & ~mask) | (SOLID_WORD & mask)) >>> 0;
                }
                blockIdx += blocksToWrite;
            }
        }
    }
}

/** Convert a `SparseVoxelGrid` back to a row-aligned dense bit buffer for diffing. */
function sparseGridToDense(grid: SparseVoxelGrid, numXWords: number): Uint32Array {
    const dense = new Uint32Array(numXWords * grid.ny * grid.nz);
    const planeWords = numXWords * grid.ny;
    const types = grid.types;
    for (let bz = 0; bz < grid.nbz; bz++) {
        for (let by = 0; by < grid.nby; by++) {
            for (let bx = 0; bx < grid.nbx; bx++) {
                const blockIdx = bx + by * grid.nbx + bz * grid.bStride;
                const bt = readBlockType(types, blockIdx);
                if (bt === BLOCK_EMPTY) continue;
                let lo: number, hi: number;
                if (bt === BLOCK_SOLID) {
                    lo = SOLID_LO;
                    hi = SOLID_HI;
                } else {
                    const s = grid.masks.slot(blockIdx);
                    lo = grid.masks.lo[s];
                    hi = grid.masks.hi[s];
                }
                const baseGx = bx * 4;
                const baseGy = by * 4;
                const baseGz = bz * 4;
                for (let lz = 0; lz < 4; lz++) {
                    const gz = baseGz + lz;
                    if (gz >= grid.nz) continue;
                    const word = lz < 2 ? lo : hi;
                    const zBitBase = (lz & 1) * 16;
                    for (let ly = 0; ly < 4; ly++) {
                        const gy = baseGy + ly;
                        if (gy >= grid.ny) continue;
                        const bitBase = zBitBase + ly * 4;
                        for (let lx = 0; lx < 4; lx++) {
                            if (!((word >>> (bitBase + lx)) & 1)) continue;
                            const gx = baseGx + lx;
                            if (gx >= grid.nx) continue;
                            const wordIdx = (gx >>> 5) + gy * numXWords + gz * planeWords;
                            dense[wordIdx] |= (1 << (gx & 31));
                        }
                    }
                }
            }
        }
    }
    return dense;
}

/**
 * Run CPU sparseDilate3 on the same dense chunk and diff against the GPU
 * output. Logs the first few divergent voxels. Only called when
 * `GPU_DILATE_DEBUG` is set.
 */
function debugCompareDilate(
    chunkIdx: number, srcDense: Uint32Array, gpuDense: Uint32Array,
    nx: number, ny: number, nz: number, numXWords: number,
    halfExtentXZ: number, halfExtentY: number
): void {
    const inputGrid = denseToSparseGrid(srcDense, nx, ny, nz, numXWords);
    const cpuGrid = sparseDilate3(inputGrid, halfExtentXZ, halfExtentY);
    const cpuDense = sparseGridToDense(cpuGrid, numXWords);

    const planeWords = numXWords * ny;
    let mismatches = 0;
    let firstMismatchAt = -1;
    let cpuOnly = 0;
    let gpuOnly = 0;
    for (let i = 0; i < gpuDense.length; i++) {
        const cpu = cpuDense[i];
        const gpu = gpuDense[i];
        if (cpu !== gpu) {
            mismatches++;
            if (firstMismatchAt < 0) firstMismatchAt = i;
            cpuOnly += popcount32(cpu & ~gpu);
            gpuOnly += popcount32(gpu & ~cpu);
        }
    }

    if (mismatches > 0) {
        const fmZ = Math.floor(firstMismatchAt / planeWords);
        const fmRowOff = firstMismatchAt - fmZ * planeWords;
        const fmY = Math.floor(fmRowOff / numXWords);
        const fmXWord = fmRowOff - fmY * numXWords;
        logger.warn(`gpuDilate3 chunk ${chunkIdx} mismatch: ${mismatches} words diverge; cpu-only bits=${cpuOnly}, gpu-only bits=${gpuOnly}; first divergent word at xWord=${fmXWord},y=${fmY},z=${fmZ}`);
    } else {
        logger.debug(`gpuDilate3 chunk ${chunkIdx}: GPU matches CPU ✓`);
    }
}

const popcount32 = (w: number): number => {
    w = w - ((w >>> 1) & 0x55555555);
    w = (w & 0x33333333) + ((w >>> 2) & 0x33333333);
    return (((w + (w >>> 4)) & 0x0F0F0F0F) * 0x01010101) >>> 24;
};

/**
 * Extract a dense bit chunk (1 bit per voxel) from a `SparseVoxelGrid` using
 * a row-aligned layout: each row of bits along X starts on a 32-bit word
 * boundary. Bit at chunk-local (dx, dy, dz) lives at word index
 * `(dx >> 5) + dy * numXWords + dz * numXWords * cy`, bit `dx & 31`. Voxels
 * outside the source grid are written as 0.
 *
 * @param src - Source sparse grid.
 * @param ox - Chunk origin X in voxels (may be negative if halo extends beyond grid).
 * @param oy - Chunk origin Y in voxels.
 * @param oz - Chunk origin Z in voxels.
 * @param cx - Chunk size X in voxels (outer = inner + 2 * halo).
 * @param cy - Chunk size Y in voxels.
 * @param cz - Chunk size Z in voxels.
 * @param numXWords - Words per row (= ceil(cx / 32)).
 * @returns Dense bit grid of length `numXWords * cy * cz`.
 */
function extractDenseChunk(
    src: SparseVoxelGrid,
    ox: number, oy: number, oz: number,
    cx: number, cy: number, cz: number,
    numXWords: number
): Uint32Array {
    const planeWords = numXWords * cy;
    const dense = new Uint32Array(numXWords * cy * cz);

    // Iterate over source blocks that overlap the outer chunk.
    const minBx = Math.max(0, Math.floor(ox / 4));
    const minBy = Math.max(0, Math.floor(oy / 4));
    const minBz = Math.max(0, Math.floor(oz / 4));
    const maxBx = Math.min(src.nbx, Math.ceil((ox + cx) / 4));
    const maxBy = Math.min(src.nby, Math.ceil((oy + cy) / 4));
    const maxBz = Math.min(src.nbz, Math.ceil((oz + cz) / 4));

    const types = src.types;
    for (let bz = minBz; bz < maxBz; bz++) {
        const baseGz = bz * 4;
        for (let by = minBy; by < maxBy; by++) {
            const baseGy = by * 4;
            for (let bx = minBx; bx < maxBx; bx++) {
                const blockIdx = bx + by * src.nbx + bz * src.bStride;
                const bt = readBlockType(types, blockIdx);
                if (bt === BLOCK_EMPTY) continue;

                let lo: number, hi: number;
                if (bt === BLOCK_SOLID) {
                    lo = SOLID_LO;
                    hi = SOLID_HI;
                } else {
                    const s = src.masks.slot(blockIdx);
                    lo = src.masks.lo[s];
                    hi = src.masks.hi[s];
                }

                const baseGx = bx * 4;
                const dx0 = baseGx - ox;
                // baseGx is 4-aligned and ox is integer, so dx0 is 4-aligned.
                // The 4 bits along X for each (ly,lz) row therefore live in a
                // single dense word at bit positions [dx0&31 .. dx0&31+3].
                const dxFastPath = dx0 >= 0 && dx0 + 4 <= cx;
                const wordOffsetX = dx0 >>> 5;
                const bitShiftX = dx0 & 31;

                for (let lz = 0; lz < 4; lz++) {
                    const dz = baseGz + lz - oz;
                    if (dz < 0 || dz >= cz) continue;
                    const word = lz < 2 ? lo : hi;
                    const zBitBase = (lz & 1) * 16;
                    const planeBase = dz * planeWords;
                    for (let ly = 0; ly < 4; ly++) {
                        const dy = baseGy + ly - oy;
                        if (dy < 0 || dy >= cy) continue;
                        const bitBase = zBitBase + ly * 4;
                        const pattern = (word >>> bitBase) & 0xF;
                        if (pattern === 0) continue;

                        if (dxFastPath) {
                            const wordIdx = wordOffsetX + dy * numXWords + planeBase;
                            dense[wordIdx] |= (pattern << bitShiftX);
                        } else {
                            // Edge: clip per-bit (chunks straddling grid edge).
                            for (let lx = 0; lx < 4; lx++) {
                                if (!((pattern >>> lx) & 1)) continue;
                                const dx = dx0 + lx;
                                if (dx < 0 || dx >= cx) continue;
                                const wordIdx = (dx >>> 5) + dy * numXWords + planeBase;
                                dense[wordIdx] |= (1 << (dx & 31));
                            }
                        }
                    }
                }
            }
        }
    }
    return dense;
}

/**
 * OR the inner region of a dense bit chunk into a `SparseVoxelGrid`. Iterates
 * over destination blocks that intersect the inner region (chunk minus halo)
 * and OR's the corresponding bits in.
 *
 * @param dst - Destination sparse grid (mutated).
 * @param dense - Dilated dense bit grid for the outer chunk.
 * @param ox - Outer chunk origin X in voxels.
 * @param oy - Outer chunk origin Y.
 * @param oz - Outer chunk origin Z.
 * @param cx - Outer chunk size X.
 * @param cy - Outer chunk size Y.
 * @param cz - Outer chunk size Z.
 * @param innerOx - Inner region origin X (= ox + haloX, in voxels).
 * @param innerOy - Inner region origin Y.
 * @param innerOz - Inner region origin Z.
 * @param innerCx - Inner region size X.
 * @param innerCy - Inner region size Y.
 * @param innerCz - Inner region size Z.
 */
function insertDenseChunk(
    dst: SparseVoxelGrid,
    dense: Uint32Array,
    ox: number, oy: number, oz: number,
    cx: number, cy: number, cz: number,
    numXWords: number,
    innerOx: number, innerOy: number, innerOz: number,
    innerCx: number, innerCy: number, innerCz: number
): void {
    const planeWords = numXWords * cy;

    // Inner region must be 4-aligned (caller guarantees), so iterate full blocks.
    const minBx = Math.max(0, innerOx >> 2);
    const minBy = Math.max(0, innerOy >> 2);
    const minBz = Math.max(0, innerOz >> 2);
    const maxBx = Math.min(dst.nbx, (innerOx + innerCx + 3) >> 2);
    const maxBy = Math.min(dst.nby, (innerOy + innerCy + 3) >> 2);
    const maxBz = Math.min(dst.nbz, (innerOz + innerCz + 3) >> 2);

    for (let bz = minBz; bz < maxBz; bz++) {
        const baseGz = bz * 4;
        for (let by = minBy; by < maxBy; by++) {
            const baseGy = by * 4;
            for (let bx = minBx; bx < maxBx; bx++) {
                const baseGx = bx * 4;
                const dx0 = baseGx - ox;
                const dxFastPath = dx0 >= 0 && dx0 + 4 <= cx;
                const wordOffsetX = dx0 >>> 5;
                const bitShiftX = dx0 & 31;

                let lo = 0;
                let hi = 0;
                for (let lz = 0; lz < 4; lz++) {
                    const dz = baseGz + lz - oz;
                    if (dz < 0 || dz >= cz) continue;
                    const zBitBase = (lz & 1) * 16;
                    const inHi = lz >= 2;
                    const planeBase = dz * planeWords;
                    for (let ly = 0; ly < 4; ly++) {
                        const dy = baseGy + ly - oy;
                        if (dy < 0 || dy >= cy) continue;
                        const bitBase = zBitBase + ly * 4;

                        let pattern: number;
                        if (dxFastPath) {
                            const wordIdx = wordOffsetX + dy * numXWords + planeBase;
                            pattern = (dense[wordIdx] >>> bitShiftX) & 0xF;
                        } else {
                            // Edge: read each bit individually with clipping.
                            pattern = 0;
                            for (let lx = 0; lx < 4; lx++) {
                                const dx = dx0 + lx;
                                if (dx < 0 || dx >= cx) continue;
                                const wordIdx = (dx >>> 5) + dy * numXWords + planeBase;
                                if ((dense[wordIdx] >>> (dx & 31)) & 1) {
                                    pattern |= (1 << lx);
                                }
                            }
                        }
                        if (pattern === 0) continue;

                        const bits = pattern << bitBase;
                        if (inHi) hi |= bits;
                        else lo |= bits;
                    }
                }
                if (lo || hi) {
                    const blockIdx = bx + by * dst.nbx + bz * dst.bStride;
                    dst.orBlock(blockIdx, lo >>> 0, hi >>> 0);
                }
            }
        }
    }
}

/**
 * GPU separable 3D dilation. Chunks the grid into ~1024³ inner regions plus
 * a halo on each side, runs three GPU passes per chunk, and OR's the
 * dilated inner region into a fresh destination `SparseVoxelGrid`. Mirrors
 * `sparseDilate3`'s API (returns a new grid).
 *
 * @param gpu - Reusable GPU dilation context (compiled shader + buffers).
 * @param src - Input sparse grid (read-only across the call).
 * @param halfExtentXZ - Dilation half-extent in voxels along X and Z.
 * @param halfExtentY - Dilation half-extent in voxels along Y.
 * @returns Newly allocated dilated sparse grid.
 */
/**
 * Apply a chunk's GPU-produced output (`typesOut` + `masksOut`) to `dst`.
 * Iterates inner blocks; for each, reads the precomputed type and (if MIXED)
 * the precomputed `lo`/`hi`, then writes directly into `dst.types` and
 * `dst.masks`. Replaces the dense-bit-reading hot loop with O(blocks) work
 * dominated by hash inserts for mixed blocks.
 */
function applyChunkToDst(
    dst: SparseVoxelGrid,
    typesOut: Uint32Array,
    masksOut: Uint32Array,
    cx: number, cy: number, cz: number,
    innerNx: number, innerNy: number, innerNz: number
): void {
    const innerBx = innerNx >> 2;
    const innerBy = innerNy >> 2;
    const innerBz = innerNz >> 2;
    const dstNbx = dst.nbx;
    const dstBStride = dst.bStride;
    const dstTypes = dst.types;
    const dstMasks = dst.masks;

    const baseBx = cx >> 2;
    const baseBy = cy >> 2;
    const baseBz = cz >> 2;

    let innerIdx = 0;
    for (let bz = 0; bz < innerBz; bz++) {
        const globalBz = baseBz + bz;
        for (let by = 0; by < innerBy; by++) {
            const globalBy = baseBy + by;
            const baseGlobalIdx = baseBx + globalBy * dstNbx + globalBz * dstBStride;
            for (let bx = 0; bx < innerBx; bx++, innerIdx++) {
                const wordIdx = innerIdx >>> 4;
                const bitShift = (innerIdx & 15) << 1;
                const bt = (typesOut[wordIdx] >>> bitShift) & 3;
                if (bt === 0) continue;  // EMPTY

                const globalBlockIdx = baseGlobalIdx + bx;
                const w = globalBlockIdx >>> 4;
                const shift = (globalBlockIdx & 15) << 1;
                dstTypes[w] = ((dstTypes[w] & ~(3 << shift)) | (bt << shift)) >>> 0;

                if (bt === 2) {  // MIXED
                    const m2 = innerIdx * 2;
                    dstMasks.set(globalBlockIdx, masksOut[m2], masksOut[m2 + 1]);
                }
            }
        }
    }
}

async function gpuDilate3(
    gpu: GpuDilation,
    src: SparseVoxelGrid,
    halfExtentXZ: number,
    halfExtentY: number
): Promise<SparseVoxelGrid> {
    const dst = new SparseVoxelGrid(src.nx, src.ny, src.nz);

    if (halfExtentXZ % 4 !== 0 && halfExtentXZ !== 0) {
        // Halo must be block-aligned for the sparse path's chunk math.
        // Current callers (fill-floor 32, carve XZ 4) satisfy this; assert.
        throw new Error(`gpuDilate3: halfExtentXZ=${halfExtentXZ} must be a multiple of 4`);
    }
    if (halfExtentY % 4 !== 0 && halfExtentY !== 0) {
        throw new Error(`gpuDilate3: halfExtentY=${halfExtentY} must be a multiple of 4`);
    }

    const haloX = halfExtentXZ;
    const haloY = halfExtentY;
    const haloZ = halfExtentXZ;
    const haloBx = haloX >> 2;
    const haloBy = haloY >> 2;
    const haloBz = haloZ >> 2;

    // Round inner chunk down to multiple of 4 (block alignment).
    const innerStep = CHUNK_INNER & ~3;

    const numChunksX = Math.ceil(src.nx / innerStep);
    const numChunksY = Math.ceil(src.ny / innerStep);
    const numChunksZ = Math.ceil(src.nz / innerStep);
    const totalChunks = numChunksX * numChunksY * numChunksZ;

    // Phase timings accumulated across all chunks in this dilate3 call.
    let tSubmit = 0;
    let tAwait = 0;
    let tApply = 0;

    interface InFlight {
        typesPromise: Promise<Uint32Array>;
        masksPromise: Promise<Uint32Array>;
        chunkIdx: number;
        cx: number; cy: number; cz: number;
        innerNx: number; innerNy: number; innerNz: number;
    }

    let currentSlot = 0;
    let inflight: InFlight | null = null;

    const drainInflight = async (): Promise<void> => {
        if (!inflight) return;
        const f = inflight;
        inflight = null;
        const tAwaitStart = performance.now();
        const [typesOut, masksOut] = await Promise.all([f.typesPromise, f.masksPromise]);
        tAwait += performance.now() - tAwaitStart;
        const tApplyStart = performance.now();
        applyChunkToDst(dst, typesOut, masksOut, f.cx, f.cy, f.cz, f.innerNx, f.innerNy, f.innerNz);
        tApply += performance.now() - tApplyStart;
    };

    // BUG-HUNTING: Setting GPU_DILATE_CPU_EXTRACT=1 routes extract through CPU
    // (current production path) and uses GPU only for dilate + compact. Lets us
    // narrow whether the new bug is in `extract` shader or downstream.
    const useCpuExtract = process.env.GPU_DILATE_CPU_EXTRACT === '1';

    const tUploadStart = performance.now();
    if (!useCpuExtract) gpu.uploadSrc(src);
    const tUpload = performance.now() - tUploadStart;

    const bar = logger.bar('Dilating', totalChunks);
    try {
        let chunkIdx = 0;
        for (let cz = 0; cz < src.nz; cz += innerStep) {
            for (let cy = 0; cy < src.ny; cy += innerStep) {
                for (let cx = 0; cx < src.nx; cx += innerStep) {
                    const innerNx = Math.min(innerStep, src.nx - cx);
                    const innerNy = Math.min(innerStep, src.ny - cy);
                    const innerNz = Math.min(innerStep, src.nz - cz);

                    const ox = cx - haloX;
                    const oy = cy - haloY;
                    const oz = cz - haloZ;
                    const outerNx = innerNx + 2 * haloX;
                    const outerNy = innerNy + 2 * haloY;
                    const outerNz = innerNz + 2 * haloZ;

                    if (chunkIsEmpty(src, ox, oy, oz, outerNx, outerNy, outerNz)) {
                        chunkIdx++;
                        bar.tick();
                        continue;
                    }
                    if (chunkIsSaturated(src, ox, oy, oz, outerNx, outerNy, outerNz)) {
                        insertSaturatedInner(dst, cx, cy, cz, innerNx, innerNy, innerNz);
                        chunkIdx++;
                        bar.tick();
                        continue;
                    }

                    const innerBx = innerNx >> 2;
                    const innerBy = innerNy >> 2;
                    const innerBz = innerNz >> 2;
                    const outerBx = outerNx >> 2;
                    const outerBy = outerNy >> 2;
                    const outerBz = outerNz >> 2;
                    const minBx = ox >> 2;
                    const minBy = oy >> 2;
                    const minBz = oz >> 2;

                    const tSubmitStart = performance.now();
                    let debugDense: Uint32Array | undefined;
                    if (useCpuExtract) {
                        debugDense = extractDenseChunk(src, ox, oy, oz, outerNx, outerNy, outerNz, (outerNx + 31) >>> 5);
                    }
                    const { types: typesPromise, masks: masksPromise } = gpu.submitChunkSparse(
                        currentSlot,
                        minBx, minBy, minBz,
                        outerBx, outerBy, outerBz,
                        haloBx, haloBy, haloBz,
                        innerBx, innerBy, innerBz,
                        halfExtentXZ, halfExtentY,
                        debugDense
                    );
                    tSubmit += performance.now() - tSubmitStart;

                    if (inflight) {
                        await drainInflight();
                    }

                    inflight = {
                        typesPromise,
                        masksPromise,
                        chunkIdx,
                        cx, cy, cz,
                        innerNx, innerNy, innerNz
                    };
                    currentSlot = (currentSlot + 1) % GpuDilation.NUM_SLOTS;

                    chunkIdx++;
                    bar.tick();
                }
            }
        }

        if (inflight) {
            await drainInflight();
        }
    } finally {
        bar.end();
        gpu.releaseSrc();
    }
    const fmt = (ms: number) => `${ms.toFixed(0)}ms`;
    logger.info(`gpuDilate3 timing: upload=${fmt(tUpload)} submit=${fmt(tSubmit)} await=${fmt(tAwait)} apply=${fmt(tApply)}`);
    return dst;
}
