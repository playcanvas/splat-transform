import {
    BLOCK_EMPTY,
    BLOCK_MIXED,
    BLOCK_SOLID,
    SOLID_HI,
    SOLID_LO,
    SparseVoxelGrid
} from './sparse-voxel-grid';
import { logger } from '../utils';

// ============================================================================
// Active Pair Computation
// ============================================================================

function getActiveYZPairs(grid: SparseVoxelGrid): Set<number> {
    const pairs = new Set<number>();
    const { nbx } = grid;
    const totalBlocks = grid.nbx * grid.nby * grid.nbz;
    for (let w = 0; w < grid.occupancy.length; w++) {
        let bits = grid.occupancy[w];
        while (bits) {
            const bitPos = 31 - Math.clz32(bits & -bits);
            const blockIdx = w * 32 + bitPos;
            if (blockIdx < totalBlocks) {
                pairs.add((blockIdx / nbx) | 0);
            }
            bits &= bits - 1;
        }
    }
    return pairs;
}

function getActiveXZPairs(grid: SparseVoxelGrid): Set<number> {
    const pairs = new Set<number>();
    const { nbx, bStride } = grid;
    const totalBlocks = grid.nbx * grid.nby * grid.nbz;
    for (let w = 0; w < grid.occupancy.length; w++) {
        let bits = grid.occupancy[w];
        while (bits) {
            const bitPos = 31 - Math.clz32(bits & -bits);
            const blockIdx = w * 32 + bitPos;
            if (blockIdx < totalBlocks) {
                const bx = blockIdx % nbx;
                const bz = (blockIdx / bStride) | 0;
                pairs.add(bx + bz * nbx);
            }
            bits &= bits - 1;
        }
    }
    return pairs;
}

function getActiveXYPairs(grid: SparseVoxelGrid): Set<number> {
    const pairs = new Set<number>();
    const { nbx, nby } = grid;
    const totalBlocks = grid.nbx * grid.nby * grid.nbz;
    for (let w = 0; w < grid.occupancy.length; w++) {
        let bits = grid.occupancy[w];
        while (bits) {
            const bitPos = 31 - Math.clz32(bits & -bits);
            const blockIdx = w * 32 + bitPos;
            if (blockIdx < totalBlocks) {
                const bx = blockIdx % nbx;
                const by = ((blockIdx / nbx) | 0) % nby;
                pairs.add(bx + by * nbx);
            }
            bits &= bits - 1;
        }
    }
    return pairs;
}

// ============================================================================
// Line Extraction / Write-back
// ============================================================================

function extractLineX(grid: SparseVoxelGrid, iy: number, iz: number, buf: Uint32Array): void {
    const by = iy >> 2, bz = iz >> 2;
    const bitBase = ((iz & 3) << 4) + ((iy & 3) << 2);
    const inHi = bitBase >= 32;
    const shift = inHi ? bitBase - 32 : bitBase;
    const lineBase = by * grid.nbx + bz * grid.bStride;
    for (let bx = 0; bx < grid.nbx; bx++) {
        const blockIdx = lineBase + bx;
        const bt = grid.blockType[blockIdx];
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
    for (let by = 0; by < grid.nby; by++) {
        const blockIdx = bx + by * grid.nbx + bz * grid.bStride;
        const bt = grid.blockType[blockIdx];
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
    for (let bz = 0; bz < grid.nbz; bz++) {
        const blockIdx = bx + by * grid.nbx + bz * grid.bStride;
        const bt = grid.blockType[blockIdx];
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
    const srcBT = src.blockType;
    const activePairs = getActiveYZPairs(src);
    for (const key of activePairs) {
        const by = key % nby;
        const bz = (key / nby) | 0;

        const lineBase = by * nbx + bz * bStride;
        let allSolid = true;
        for (let bx = 0; bx < nbx; bx++) {
            if (srcBT[lineBase + bx] !== BLOCK_SOLID) {
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
    const srcBT = src.blockType;
    const activePairs = getActiveXYPairs(src);
    for (const key of activePairs) {
        const bx = key % nbx;
        const by = (key / nbx) | 0;

        const lineStart = bx + by * nbx;
        let allSolid = true;
        for (let bz = 0; bz < nbz; bz++) {
            if (srcBT[lineStart + bz * bStride] !== BLOCK_SOLID) {
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
    const srcBT = src.blockType;
    const activePairs = getActiveXZPairs(src);
    for (const key of activePairs) {
        const bx = key % nbx;
        const bz = (key / nbx) | 0;

        const lineStart = bx + bz * bStride;
        let allSolid = true;
        for (let by = 0; by < nby; by++) {
            if (srcBT[lineStart + by * nbx] !== BLOCK_SOLID) {
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

function sparseDilate3(
    src: SparseVoxelGrid,
    halfExtentXZ: number,
    halfExtentY: number
): SparseVoxelGrid {
    const { nx, ny, nz } = src;
    const a = new SparseVoxelGrid(nx, ny, nz);
    sparseDilateX(src, a, halfExtentXZ);
    logger.progress.step();
    const b = new SparseVoxelGrid(nx, ny, nz);
    sparseDilateZ(a, b, halfExtentXZ);
    a.clear();
    logger.progress.step();
    sparseDilateY(b, a, halfExtentY);
    b.clear();
    logger.progress.step();
    return a;
}

export { sparseDilate3 };
