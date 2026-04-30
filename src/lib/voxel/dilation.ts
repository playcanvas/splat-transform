import {
    BLOCK_EMPTY,
    BLOCK_SOLID,
    BLOCKS_PER_WORD,
    EVEN_BITS,
    SOLID_HI,
    SOLID_LO,
    SparseVoxelGrid
} from './sparse-voxel-grid';
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
// `getType` reads a packed 2-bit block type. V8 should JIT this into a few
// inline ops per call; small overhead vs the old direct Uint8Array read.
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
        const bt = (types[blockIdx >>> 4] >>> ((blockIdx & 15) << 1)) & 0x3;
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
        const bt = (types[blockIdx >>> 4] >>> ((blockIdx & 15) << 1)) & 0x3;
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
        const bt = (types[blockIdx >>> 4] >>> ((blockIdx & 15) << 1)) & 0x3;
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
            if (((srcTypes[idx >>> 4] >>> ((idx & 15) << 1)) & 0x3) !== BLOCK_SOLID) {
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
            if (((srcTypes[idx >>> 4] >>> ((idx & 15) << 1)) & 0x3) !== BLOCK_SOLID) {
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
            if (((srcTypes[idx >>> 4] >>> ((idx & 15) << 1)) & 0x3) !== BLOCK_SOLID) {
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

export { sparseDilate3 };
