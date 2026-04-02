import { Vec3 } from 'playcanvas';

import {
    BlockAccumulator,
    xyzToMorton,
    type Bounds
} from './sparse-octree';
import {
    BLOCK_EMPTY,
    BLOCK_MIXED,
    BLOCK_SOLID,
    FACE_MASKS_HI,
    FACE_MASKS_LO,
    SOLID_HI,
    SOLID_LO,
    SparseVoxelGrid
} from './sparse-voxel-grid';
import { logger } from '../utils/logger';

type NavSeed = {
    x: number;
    y: number;
    z: number;
};

type NavSimplifyResult = {
    accumulator: BlockAccumulator;
    gridBounds: Bounds;
};

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

// ============================================================================
// Sparse Grid Combination
// ============================================================================

function computeEmptyGrid(visited: SparseVoxelGrid, blocked: SparseVoxelGrid): SparseVoxelGrid {
    const empty = new SparseVoxelGrid(visited.nx, visited.ny, visited.nz);
    const totalBlocks = visited.nbx * visited.nby * visited.nbz;
    for (let w = 0; w < visited.occupancy.length; w++) {
        let bits = visited.occupancy[w];
        while (bits) {
            const bitPos = 31 - Math.clz32(bits & -bits);
            const blockIdx = w * 32 + bitPos;
            if (blockIdx >= totalBlocks) break;
            const vbt = visited.blockType[blockIdx];
            let vLo: number, vHi: number;
            if (vbt === BLOCK_SOLID) {
                vLo = SOLID_LO;
                vHi = SOLID_HI;
            } else {
                const vs = visited.masks.slot(blockIdx);
                vLo = visited.masks.lo[vs];
                vHi = visited.masks.hi[vs];
            }
            const bbt = blocked.blockType[blockIdx];
            let lo: number, hi: number;
            if (bbt === BLOCK_EMPTY) {
                lo = vLo;
                hi = vHi;
            } else if (bbt === BLOCK_SOLID) {
                lo = 0;
                hi = 0;
            } else {
                const bs = blocked.masks.slot(blockIdx);
                lo = (vLo & ~blocked.masks.lo[bs]) >>> 0;
                hi = (vHi & ~blocked.masks.hi[bs]) >>> 0;
            }
            if (lo || hi) {
                empty.orBlock(blockIdx, lo, hi);
            }
            bits &= bits - 1;
        }
    }
    return empty;
}

function sparseOrGrids(a: SparseVoxelGrid, b: SparseVoxelGrid): SparseVoxelGrid {
    const result = a.clone();
    const totalBlocks = b.nbx * b.nby * b.nbz;
    for (let w = 0; w < b.occupancy.length; w++) {
        let bits = b.occupancy[w];
        while (bits) {
            const bitPos = 31 - Math.clz32(bits & -bits);
            const blockIdx = w * 32 + bitPos;
            if (blockIdx >= totalBlocks) break;
            const bt = b.blockType[blockIdx];
            if (bt === BLOCK_SOLID) {
                result.orBlock(blockIdx, SOLID_LO, SOLID_HI);
            } else {
                const s = b.masks.slot(blockIdx);
                result.orBlock(blockIdx, b.masks.lo[s], b.masks.hi[s]);
            }
            bits &= bits - 1;
        }
    }
    return result;
}

// ============================================================================
// Utilities
// ============================================================================

function findNearestFreeCellSparse(
    blocked: SparseVoxelGrid,
    seedIx: number, seedIy: number, seedIz: number,
    maxRadius: number
): { ix: number; iy: number; iz: number } | null {
    const { nx, ny, nz } = blocked;
    for (let r = 1; r <= maxRadius; r++) {
        for (let dz = -r; dz <= r; dz++) {
            for (let dy = -r; dy <= r; dy++) {
                for (let dx = -r; dx <= r; dx++) {
                    if (Math.abs(dx) !== r && Math.abs(dy) !== r && Math.abs(dz) !== r) continue;
                    const ix = seedIx + dx;
                    const iy = seedIy + dy;
                    const iz = seedIz + dz;
                    if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) continue;
                    if (!blocked.getVoxel(ix, iy, iz)) return { ix, iy, iz };
                }
            }
        }
    }
    return null;
}

function getOccupiedBlockBounds(grid: SparseVoxelGrid): {
    minBx: number; minBy: number; minBz: number;
    maxBx: number; maxBy: number; maxBz: number;
} | null {
    const { nbx, nby } = grid;
    const totalBlocks = grid.nbx * grid.nby * grid.nbz;
    let minBx = nbx, minBy = nby, minBz = grid.nbz;
    let maxBx = 0, maxBy = 0, maxBz = 0;
    for (let w = 0; w < grid.occupancy.length; w++) {
        let bits = grid.occupancy[w];
        while (bits) {
            const bitPos = 31 - Math.clz32(bits & -bits);
            const blockIdx = w * 32 + bitPos;
            if (blockIdx >= totalBlocks) break;
            const bx = blockIdx % nbx;
            const byBz = (blockIdx / nbx) | 0;
            const by = byBz % nby;
            const bz = (byBz / nby) | 0;
            if (bx < minBx) minBx = bx;
            if (bx > maxBx) maxBx = bx;
            if (by < minBy) minBy = by;
            if (by > maxBy) maxBy = by;
            if (bz < minBz) minBz = bz;
            if (bz > maxBz) maxBz = bz;
            bits &= bits - 1;
        }
    }
    return minBx <= maxBx ? { minBx, minBy, minBz, maxBx, maxBy, maxBz } : null;
}

// ============================================================================
// Two-Level BFS
//
// Exploits the fact that most blocks in the blocked grid are BLOCK_EMPTY (all
// 64 voxels free). When BFS reaches such a block, ALL voxels are reachable
// (mutually face-connected within the 4x4x4 cube), so we mark the entire
// block as visited and propagate at block granularity — 64x fewer operations.
//
// Two queues:
//   Block queue  — processes BLOCK_EMPTY (in blocked) blocks at block level
//   Voxel queue  — processes individual voxels in BLOCK_MIXED blocks
//
// When the voxel BFS enters a BLOCK_EMPTY block it switches to block fill.
// When block fill hits a BLOCK_MIXED neighbor it enqueues free face voxels.
// ============================================================================

function twoLevelBFS(
    blocked: SparseVoxelGrid,
    blockSeeds: number[],
    voxelSeeds: { ix: number; iy: number; iz: number }[],
    nx: number, ny: number, nz: number
): SparseVoxelGrid {
    const visited = new SparseVoxelGrid(nx, ny, nz);
    const nbx = nx >> 2;
    const nby = ny >> 2;
    const nbz = nz >> 2;
    const bStride = nbx * nby;

    const blockedBT = blocked.blockType;
    const bMasks = blocked.masks;
    const visitedBT = visited.blockType;
    const vMasks = visited.masks;
    const visitedOcc = visited.occupancy;

    // Block queue (ring buffer of block indices)
    let bqCap = 1 << 14;
    let bqBuf = new Uint32Array(bqCap);
    let bqMask = bqCap - 1;
    let bqHead = 0;
    let bqTail = 0;
    let bqSize = 0;

    // Voxel queue (ring buffer of coordinates)
    let vqCap = 1 << 14;
    let vqIx = new Uint32Array(vqCap);
    let vqIy = new Uint32Array(vqCap);
    let vqIz = new Uint32Array(vqCap);
    let vqMask = vqCap - 1;
    let vqHead = 0;
    let vqTail = 0;
    let vqSize = 0;

    const growBlockQueue = (): void => {
        const newCap = bqCap << 1;
        const nb = new Uint32Array(newCap);
        for (let i = 0; i < bqSize; i++) nb[i] = bqBuf[(bqHead + i) & bqMask];
        bqBuf = nb;
        bqCap = newCap;
        bqMask = newCap - 1;
        bqHead = 0;
        bqTail = bqSize;
    };

    const growVoxelQueue = (): void => {
        const newCap = vqCap << 1;
        const nix = new Uint32Array(newCap);
        const niy = new Uint32Array(newCap);
        const niz = new Uint32Array(newCap);
        for (let i = 0; i < vqSize; i++) {
            const j = (vqHead + i) & vqMask;
            nix[i] = vqIx[j];
            niy[i] = vqIy[j];
            niz[i] = vqIz[j];
        }
        vqIx = nix;
        vqIy = niy;
        vqIz = niz;
        vqCap = newCap;
        vqMask = newCap - 1;
        vqHead = 0;
        vqTail = vqSize;
    };

    const enqueueVoxel = (ix: number, iy: number, iz: number): void => {
        if (vqSize >= vqCap) growVoxelQueue();
        vqIx[vqTail] = ix;
        vqIy[vqTail] = iy;
        vqIz[vqTail] = iz;
        vqTail = (vqTail + 1) & vqMask;
        vqSize++;
    };

    // Mark a BLOCK_EMPTY (in blocked) block as fully visited and enqueue it
    // for block-level neighbor propagation. Returns true if the block was filled.
    const tryFillBlock = (blockIdx: number): boolean => {
        if (blockedBT[blockIdx] !== BLOCK_EMPTY) return false;
        if (visitedBT[blockIdx] !== BLOCK_EMPTY) return false;
        visitedBT[blockIdx] = BLOCK_SOLID;
        visitedOcc[blockIdx >>> 5] |= (1 << (blockIdx & 31));
        if (bqSize >= bqCap) growBlockQueue();
        bqBuf[bqTail] = blockIdx;
        bqTail = (bqTail + 1) & bqMask;
        bqSize++;
        return true;
    };

    // Enqueue free, unvisited voxels on one face of a BLOCK_MIXED neighbor.
    // `face` indexes into FACE_MASKS_LO/HI (0=-X, 1=+X, 2=-Y, 3=+Y, 4=-Z, 5=+Z).
    const enqueueFaceVoxels = (nBlockIdx: number, face: number, nBx: number, nBy: number, nBz: number): void => {
        const vbt = visitedBT[nBlockIdx];
        if (vbt === BLOCK_SOLID) return;

        const bs = bMasks.slot(nBlockIdx);
        let vLo = 0, vHi = 0;
        let vs = -1;
        if (vbt === BLOCK_MIXED) {
            vs = vMasks.slot(nBlockIdx);
            vLo = vMasks.lo[vs];
            vHi = vMasks.hi[vs];
        }

        const freeLo = (FACE_MASKS_LO[face] & ~bMasks.lo[bs] & ~vLo) >>> 0;
        const freeHi = (FACE_MASKS_HI[face] & ~bMasks.hi[bs] & ~vHi) >>> 0;
        if (freeLo === 0 && freeHi === 0) return;

        if (vbt === BLOCK_EMPTY) {
            visitedBT[nBlockIdx] = BLOCK_MIXED;
            visitedOcc[nBlockIdx >>> 5] |= (1 << (nBlockIdx & 31));
            vMasks.set(nBlockIdx, freeLo, freeHi);
        } else {
            vMasks.lo[vs] = (vMasks.lo[vs] | freeLo) >>> 0;
            vMasks.hi[vs] = (vMasks.hi[vs] | freeHi) >>> 0;
            if (vMasks.lo[vs] === SOLID_LO && vMasks.hi[vs] === SOLID_HI) {
                vMasks.removeAt(vs);
                visitedBT[nBlockIdx] = BLOCK_SOLID;
            }
        }

        const baseIx = nBx << 2;
        const baseIy = nBy << 2;
        const baseIz = nBz << 2;

        let bits = freeLo;
        while (bits) {
            const bp = 31 - Math.clz32(bits & -bits);
            enqueueVoxel(baseIx + (bp & 3), baseIy + ((bp >> 2) & 3), baseIz + (bp >> 4));
            bits &= bits - 1;
        }
        bits = freeHi;
        while (bits) {
            const bp = 31 - Math.clz32(bits & -bits);
            const bi = bp + 32;
            enqueueVoxel(baseIx + (bi & 3), baseIy + ((bi >> 2) & 3), baseIz + (bi >> 4));
            bits &= bits - 1;
        }
    };

    // Process a block's 6 face-neighbors at block level.
    const processBlock = (blockIdx: number): void => {
        const bx = blockIdx % nbx;
        const byBz = (blockIdx / nbx) | 0;
        const by = byBz % nby;
        const bz = (byBz / nby) | 0;

        // For each of the 6 directions: check the neighbor block.
        // The `face` parameter is the face of the NEIGHBOR that touches us.
        // -X neighbor: neighbor's +X face (face=1)
        if (bx > 0) {
            const ni = blockIdx - 1;
            const nbt = blockedBT[ni];
            if (nbt === BLOCK_EMPTY) tryFillBlock(ni);
            else if (nbt === BLOCK_MIXED) enqueueFaceVoxels(ni, 1, bx - 1, by, bz);
        }
        // +X neighbor: neighbor's -X face (face=0)
        if (bx < nbx - 1) {
            const ni = blockIdx + 1;
            const nbt = blockedBT[ni];
            if (nbt === BLOCK_EMPTY) tryFillBlock(ni);
            else if (nbt === BLOCK_MIXED) enqueueFaceVoxels(ni, 0, bx + 1, by, bz);
        }
        // -Y neighbor: neighbor's +Y face (face=3)
        if (by > 0) {
            const ni = blockIdx - nbx;
            const nbt = blockedBT[ni];
            if (nbt === BLOCK_EMPTY) tryFillBlock(ni);
            else if (nbt === BLOCK_MIXED) enqueueFaceVoxels(ni, 3, bx, by - 1, bz);
        }
        // +Y neighbor: neighbor's -Y face (face=2)
        if (by < nby - 1) {
            const ni = blockIdx + nbx;
            const nbt = blockedBT[ni];
            if (nbt === BLOCK_EMPTY) tryFillBlock(ni);
            else if (nbt === BLOCK_MIXED) enqueueFaceVoxels(ni, 2, bx, by + 1, bz);
        }
        // -Z neighbor: neighbor's +Z face (face=5)
        if (bz > 0) {
            const ni = blockIdx - bStride;
            const nbt = blockedBT[ni];
            if (nbt === BLOCK_EMPTY) tryFillBlock(ni);
            else if (nbt === BLOCK_MIXED) enqueueFaceVoxels(ni, 5, bx, by, bz - 1);
        }
        // +Z neighbor: neighbor's -Z face (face=4)
        if (bz < nbz - 1) {
            const ni = blockIdx + bStride;
            const nbt = blockedBT[ni];
            if (nbt === BLOCK_EMPTY) tryFillBlock(ni);
            else if (nbt === BLOCK_MIXED) enqueueFaceVoxels(ni, 4, bx, by, bz + 1);
        }
    };

    // Try to enqueue a single voxel (used by voxel-level BFS).
    // If the voxel's block is BLOCK_EMPTY in blocked, triggers block fill.
    const tryEnqueueVoxel = (ix: number, iy: number, iz: number): void => {
        const blockIdx = (ix >> 2) + (iy >> 2) * nbx + (iz >> 2) * bStride;

        const bbt = blockedBT[blockIdx];
        if (bbt === BLOCK_SOLID) return;
        if (bbt === BLOCK_EMPTY) {
            tryFillBlock(blockIdx);
            return;
        }

        // BLOCK_MIXED: check specific voxel against blocked mask
        const bs = bMasks.slot(blockIdx);
        const bitIdx = (ix & 3) + ((iy & 3) << 2) + ((iz & 3) << 4);
        if (bitIdx < 32 ? (bMasks.lo[bs] >>> bitIdx) & 1 : (bMasks.hi[bs] >>> (bitIdx - 32)) & 1) return;

        // Check and set visited
        const vbt = visitedBT[blockIdx];
        if (vbt === BLOCK_SOLID) return;
        if (vbt === BLOCK_MIXED) {
            const vs = vMasks.slot(blockIdx);
            if (bitIdx < 32 ? (vMasks.lo[vs] >>> bitIdx) & 1 : (vMasks.hi[vs] >>> (bitIdx - 32)) & 1) return;
            if (bitIdx < 32) vMasks.lo[vs] = (vMasks.lo[vs] | (1 << bitIdx)) >>> 0;
            else vMasks.hi[vs] = (vMasks.hi[vs] | (1 << (bitIdx - 32))) >>> 0;
            if (vMasks.lo[vs] === SOLID_LO && vMasks.hi[vs] === SOLID_HI) {
                vMasks.removeAt(vs);
                visitedBT[blockIdx] = BLOCK_SOLID;
            }
        } else {
            visitedBT[blockIdx] = BLOCK_MIXED;
            visitedOcc[blockIdx >>> 5] |= (1 << (blockIdx & 31));
            vMasks.set(blockIdx,
                bitIdx < 32 ? (1 << bitIdx) >>> 0 : 0,
                bitIdx >= 32 ? (1 << (bitIdx - 32)) >>> 0 : 0
            );
        }

        enqueueVoxel(ix, iy, iz);
    };

    // --- Seeding ---

    for (let i = 0; i < blockSeeds.length; i++) {
        tryFillBlock(blockSeeds[i]);
    }

    for (let i = 0; i < voxelSeeds.length; i++) {
        const s = voxelSeeds[i];
        tryEnqueueVoxel(s.ix, s.iy, s.iz);
    }

    // --- Main BFS loop ---
    // Process block queue first (fast propagation through empty space),
    // then voxel queue (mixed blocks). Interleave until both are empty.

    while (bqSize > 0 || vqSize > 0) {
        while (bqSize > 0) {
            const blockIdx = bqBuf[bqHead];
            bqHead = (bqHead + 1) & bqMask;
            bqSize--;
            processBlock(blockIdx);
        }

        if (vqSize > 0) {
            const ix = vqIx[vqHead];
            const iy = vqIy[vqHead];
            const iz = vqIz[vqHead];
            vqHead = (vqHead + 1) & vqMask;
            vqSize--;

            if (ix > 0) tryEnqueueVoxel(ix - 1, iy, iz);
            if (ix < nx - 1) tryEnqueueVoxel(ix + 1, iy, iz);
            if (iy > 0) tryEnqueueVoxel(ix, iy - 1, iz);
            if (iy < ny - 1) tryEnqueueVoxel(ix, iy + 1, iz);
            if (iz > 0) tryEnqueueVoxel(ix, iy, iz - 1);
            if (iz < nz - 1) tryEnqueueVoxel(ix, iy, iz + 1);
        }
    }

    return visited;
}

// ============================================================================
// simplifyForCapsule
// ============================================================================

const simplifyForCapsule = (
    accumulator: BlockAccumulator,
    gridBounds: Bounds,
    voxelResolution: number,
    capsuleHeight: number,
    capsuleRadius: number,
    seed: NavSeed
): NavSimplifyResult => {
    if (!Number.isFinite(voxelResolution) || voxelResolution <= 0) {
        throw new Error(`nav simplify: voxelResolution must be finite and > 0, got ${voxelResolution}`);
    }
    if (!Number.isFinite(capsuleHeight) || capsuleHeight <= 0) {
        throw new Error(`nav simplify: capsuleHeight must be finite and > 0, got ${capsuleHeight}`);
    }
    if (!Number.isFinite(capsuleRadius) || capsuleRadius < 0) {
        throw new Error(`nav simplify: capsuleRadius must be finite and >= 0, got ${capsuleRadius}`);
    }

    const nx = Math.round((gridBounds.max.x - gridBounds.min.x) / voxelResolution);
    const ny = Math.round((gridBounds.max.y - gridBounds.min.y) / voxelResolution);
    const nz = Math.round((gridBounds.max.z - gridBounds.min.z) / voxelResolution);

    if (nx % 4 !== 0 || ny % 4 !== 0 || nz % 4 !== 0) {
        throw new Error(`Grid dimensions must be multiples of 4, got ${nx}x${ny}x${nz}`);
    }

    if (accumulator.count === 0) {
        return { accumulator, gridBounds };
    }

    const kernelR = Math.ceil(capsuleRadius / voxelResolution);
    const yHalfExtent = Math.ceil(capsuleHeight / (2 * voxelResolution));
    const nbx = nx >> 2;
    const nby = ny >> 2;
    const nbz = nz >> 2;

    logger.progress.begin(10);
    let progressComplete = false;

    try {
        // Phase 6: build sparse grid from accumulator
        const gridA = SparseVoxelGrid.fromAccumulator(accumulator, nx, ny, nz);
        logger.progress.step();

        // Phase 7: capsule clearance grid (Minkowski dilation)
        const blocked = sparseDilate3(gridA, kernelR, yHalfExtent);

        // Phase 8: BFS flood fill from seed
        let seedIx = Math.floor((seed.x - gridBounds.min.x) / voxelResolution);
        let seedIy = Math.floor((seed.y - gridBounds.min.y) / voxelResolution);
        let seedIz = Math.floor((seed.z - gridBounds.min.z) / voxelResolution);

        if (seedIx < 0 || seedIx >= nx || seedIy < 0 || seedIy >= ny || seedIz < 0 || seedIz >= nz) {
            logger.warn(`nav simplify: seed (${seed.x}, ${seed.y}, ${seed.z}) outside grid, skipping`);
            return { accumulator, gridBounds };
        }

        if (blocked.getVoxel(seedIx, seedIy, seedIz)) {
            const maxRadius = Math.max(kernelR, yHalfExtent) * 2;
            const found = findNearestFreeCellSparse(blocked, seedIx, seedIy, seedIz, maxRadius);
            if (!found) {
                logger.warn(`nav simplify: seed (${seed.x}, ${seed.y}, ${seed.z}) blocked after dilation, no free cell within ${maxRadius} voxels, skipping`);
                return { accumulator, gridBounds };
            }
            seedIx = found.ix;
            seedIy = found.iy;
            seedIz = found.iz;
        }

        const seedBlockIdx = (seedIx >> 2) + (seedIy >> 2) * nbx + (seedIz >> 2) * (nbx * nby);
        const seedBt = blocked.blockType[seedBlockIdx];
        const bSeeds = seedBt === BLOCK_EMPTY ? [seedBlockIdx] : [];
        const vSeeds = seedBt === BLOCK_EMPTY ? [] : [{ ix: seedIx, iy: seedIy, iz: seedIz }];
        const visited = twoLevelBFS(blocked, bSeeds, vSeeds, nx, ny, nz);
        logger.progress.step();

        // Phase 9: erode(blocked | ~visited) = NOT dilate(visited & ~blocked)
        const emptyGrid = computeEmptyGrid(visited, blocked);
        logger.progress.step();

        const navRegion = sparseDilate3(emptyGrid, kernelR, yHalfExtent);

        // Phase 10: crop to bounding box of navigable cells
        const navBounds = getOccupiedBlockBounds(navRegion);

        if (!navBounds) {
            logger.warn('nav simplify: no navigable cells remain, returning empty result');
            logger.progress.step();
            progressComplete = true;
            return {
                accumulator: new BlockAccumulator(),
                gridBounds: { min: gridBounds.min.clone(), max: gridBounds.min.clone() }
            };
        }

        const MARGIN = 1;
        const cropMinBx = Math.max(0, navBounds.minBx - MARGIN);
        const cropMinBy = Math.max(0, navBounds.minBy - MARGIN);
        const cropMinBz = Math.max(0, navBounds.minBz - MARGIN);
        const cropMaxBx = Math.min(nbx, navBounds.maxBx + 1 + MARGIN);
        const cropMaxBy = Math.min(nby, navBounds.maxBy + 1 + MARGIN);
        const cropMaxBz = Math.min(nbz, navBounds.maxBz + 1 + MARGIN);

        const blockSize = 4 * voxelResolution;
        const croppedMin = new Vec3(
            gridBounds.min.x + cropMinBx * blockSize,
            gridBounds.min.y + cropMinBy * blockSize,
            gridBounds.min.z + cropMinBz * blockSize
        );
        const croppedBounds: Bounds = {
            min: croppedMin,
            max: new Vec3(
                croppedMin.x + (cropMaxBx - cropMinBx) * blockSize,
                croppedMin.y + (cropMaxBy - cropMinBy) * blockSize,
                croppedMin.z + (cropMaxBz - cropMinBz) * blockSize
            )
        };

        logger.progress.step();
        progressComplete = true;

        return {
            accumulator: navRegion.toAccumulatorInverted(
                cropMinBx, cropMinBy, cropMinBz,
                cropMaxBx, cropMaxBy, cropMaxBz
            ),
            gridBounds: croppedBounds
        };

    } finally {
        if (!progressComplete) {
            logger.progress.cancel();
        }
    }
};

// ============================================================================
// fillExterior
// ============================================================================

const fillExterior = (
    accumulator: BlockAccumulator,
    gridBounds: Bounds,
    voxelResolution: number,
    dilation: number,
    seed: NavSeed
): NavSimplifyResult => {
    if (!Number.isFinite(voxelResolution) || voxelResolution <= 0) {
        throw new Error(`fillExterior: voxelResolution must be finite and > 0, got ${voxelResolution}`);
    }
    if (!Number.isFinite(dilation) || dilation <= 0) {
        throw new Error(`fillExterior: dilation must be finite and > 0, got ${dilation}`);
    }

    const nx = Math.round((gridBounds.max.x - gridBounds.min.x) / voxelResolution);
    const ny = Math.round((gridBounds.max.y - gridBounds.min.y) / voxelResolution);
    const nz = Math.round((gridBounds.max.z - gridBounds.min.z) / voxelResolution);

    if (nx % 4 !== 0 || ny % 4 !== 0 || nz % 4 !== 0) {
        throw new Error(`Grid dimensions must be multiples of 4, got ${nx}x${ny}x${nz}`);
    }

    if (accumulator.count === 0) {
        return { accumulator, gridBounds };
    }

    const halfExtent = Math.ceil(dilation / voxelResolution);
    const nbx = nx >> 2;
    const nby = ny >> 2;
    const nbz = nz >> 2;

    logger.progress.begin(10);
    let progressComplete = false;

    try {
        // Stage 1: build sparse grid from accumulator
        const gridOriginal = SparseVoxelGrid.fromAccumulator(accumulator, nx, ny, nz);
        logger.progress.step();

        // Stage 2: uniform dilation
        const dilated = sparseDilate3(gridOriginal, halfExtent, halfExtent);

        // Stage 3: BFS flood fill from all 6 boundary faces (block-level seeding)
        const bStride = nbx * nby;
        const blockSeeds: number[] = [];
        const faceVoxelSeeds: { ix: number; iy: number; iz: number }[] = [];

        const seedBoundaryBlock = (blockIdx: number, bx: number, by: number, bz: number, face: number): void => {
            const bt = dilated.blockType[blockIdx];
            if (bt === BLOCK_SOLID) return;
            if (bt === BLOCK_EMPTY) {
                blockSeeds.push(blockIdx);
                return;
            }
            // BLOCK_MIXED: seed only the free voxels on the grid boundary face
            const ms = dilated.masks.slot(blockIdx);
            const faceLo = FACE_MASKS_LO[face];
            const faceHi = FACE_MASKS_HI[face];
            let freeLo = (faceLo & ~dilated.masks.lo[ms]) >>> 0;
            let freeHi = (faceHi & ~dilated.masks.hi[ms]) >>> 0;
            if (freeLo === 0 && freeHi === 0) return;
            const baseIx = bx << 2;
            const baseIy = by << 2;
            const baseIz = bz << 2;
            while (freeLo) {
                const bp = 31 - Math.clz32(freeLo & -freeLo);
                faceVoxelSeeds.push({ ix: baseIx + (bp & 3), iy: baseIy + ((bp >> 2) & 3), iz: baseIz + (bp >> 4) });
                freeLo &= freeLo - 1;
            }
            while (freeHi) {
                const bp = 31 - Math.clz32(freeHi & -freeHi);
                const bi = bp + 32;
                faceVoxelSeeds.push({ ix: baseIx + (bi & 3), iy: baseIy + ((bi >> 2) & 3), iz: baseIz + (bi >> 4) });
                freeHi &= freeHi - 1;
            }
        };

        // -X face (bx=0): grid boundary is lx=0 face of these blocks (face=0)
        for (let bz = 0; bz < nbz; bz++) {
            for (let by = 0; by < nby; by++) {
                seedBoundaryBlock(by * nbx + bz * bStride, 0, by, bz, 0);
            }
        }

        // +X face (bx=nbx-1): grid boundary is lx=3 face (face=1)
        for (let bz = 0; bz < nbz; bz++) {
            for (let by = 0; by < nby; by++) {
                seedBoundaryBlock((nbx - 1) + by * nbx + bz * bStride, nbx - 1, by, bz, 1);
            }
        }

        // -Y face (by=0): grid boundary is ly=0 face (face=2)
        for (let bz = 0; bz < nbz; bz++) {
            for (let bx = 0; bx < nbx; bx++) {
                seedBoundaryBlock(bx + bz * bStride, bx, 0, bz, 2);
            }
        }

        // +Y face (by=nby-1): grid boundary is ly=3 face (face=3)
        for (let bz = 0; bz < nbz; bz++) {
            for (let bx = 0; bx < nbx; bx++) {
                seedBoundaryBlock(bx + (nby - 1) * nbx + bz * bStride, bx, nby - 1, bz, 3);
            }
        }

        // -Z face (bz=0): grid boundary is lz=0 face (face=4)
        for (let by = 0; by < nby; by++) {
            for (let bx = 0; bx < nbx; bx++) {
                seedBoundaryBlock(bx + by * nbx, bx, by, 0, 4);
            }
        }

        // +Z face (bz=nbz-1): grid boundary is lz=3 face (face=5)
        for (let by = 0; by < nby; by++) {
            for (let bx = 0; bx < nbx; bx++) {
                seedBoundaryBlock(bx + by * nbx + (nbz - 1) * bStride, bx, by, nbz - 1, 5);
            }
        }

        const visited = twoLevelBFS(dilated, blockSeeds, faceVoxelSeeds, nx, ny, nz);

        // Check if seed is reachable from outside
        const seedIx = Math.floor((seed.x - gridBounds.min.x) / voxelResolution);
        const seedIy = Math.floor((seed.y - gridBounds.min.y) / voxelResolution);
        const seedIz = Math.floor((seed.z - gridBounds.min.z) / voxelResolution);

        if (seedIx >= 0 && seedIx < nx && seedIy >= 0 && seedIy < ny && seedIz >= 0 && seedIz < nz) {
            if (visited.getVoxel(seedIx, seedIy, seedIz)) {
                logger.log('fillExterior: seed reachable from outside, skipping');
                logger.progress.cancel();
                progressComplete = true;
                return { accumulator, gridBounds };
            }
        } else {
            logger.log('fillExterior: seed outside grid bounds, skipping exterior fill');
            logger.progress.cancel();
            progressComplete = true;
            return { accumulator, gridBounds };
        }

        logger.progress.step();

        // Stage 4: dilate BFS-visited
        const dilatedVisited = sparseDilate3(visited, halfExtent, halfExtent);

        // Stage 5: combine with original
        const combined = sparseOrGrids(gridOriginal, dilatedVisited);
        logger.progress.step();

        // Stage 6: crop to bounding box of empty (navigable) cells
        let minIx = nx, minIy = ny, minIz = nz;
        let maxIx = 0, maxIy = 0, maxIz = 0;

        for (let bz = 0; bz < nbz; bz++) {
            for (let by = 0; by < nby; by++) {
                for (let bx = 0; bx < nbx; bx++) {
                    const blockIdx = bx + by * nbx + bz * combined.bStride;
                    const bt = combined.blockType[blockIdx];
                    if (bt === BLOCK_SOLID) continue;
                    if (bt === BLOCK_MIXED) {
                        const cs = combined.masks.slot(blockIdx);
                        if (combined.masks.lo[cs] === SOLID_LO && combined.masks.hi[cs] === SOLID_HI) continue;
                    }
                    const baseX = bx << 2;
                    const baseY = by << 2;
                    const baseZ = bz << 2;
                    if (baseX < minIx) minIx = baseX;
                    if (baseX + 3 > maxIx) maxIx = baseX + 3;
                    if (baseY < minIy) minIy = baseY;
                    if (baseY + 3 > maxIy) maxIy = baseY + 3;
                    if (baseZ < minIz) minIz = baseZ;
                    if (baseZ + 3 > maxIz) maxIz = baseZ + 3;
                }
            }
        }

        if (minIx > maxIx) {
            logger.warn('fillExterior: no navigable cells remain, returning empty result');
            logger.progress.step();
            progressComplete = true;
            return {
                accumulator: new BlockAccumulator(),
                gridBounds: { min: gridBounds.min.clone(), max: gridBounds.min.clone() }
            };
        }

        const MARGIN = 1;
        const cropMinBx = Math.max(0, (minIx >> 2) - MARGIN);
        const cropMinBy = Math.max(0, (minIy >> 2) - MARGIN);
        const cropMinBz = Math.max(0, (minIz >> 2) - MARGIN);
        const cropMaxBx = Math.min(nbx, (maxIx >> 2) + 1 + MARGIN);
        const cropMaxBy = Math.min(nby, (maxIy >> 2) + 1 + MARGIN);
        const cropMaxBz = Math.min(nbz, (maxIz >> 2) + 1 + MARGIN);

        const blockSize = 4 * voxelResolution;
        const croppedMin = new Vec3(
            gridBounds.min.x + cropMinBx * blockSize,
            gridBounds.min.y + cropMinBy * blockSize,
            gridBounds.min.z + cropMinBz * blockSize
        );
        const croppedBounds: Bounds = {
            min: croppedMin,
            max: new Vec3(
                croppedMin.x + (cropMaxBx - cropMinBx) * blockSize,
                croppedMin.y + (cropMaxBy - cropMinBy) * blockSize,
                croppedMin.z + (cropMaxBz - cropMinBz) * blockSize
            )
        };

        logger.progress.step();
        progressComplete = true;

        return {
            accumulator: combined.toAccumulator(
                cropMinBx, cropMinBy, cropMinBz,
                cropMaxBx, cropMaxBy, cropMaxBz
            ),
            gridBounds: croppedBounds
        };

    } finally {
        if (!progressComplete) {
            logger.progress.cancel();
        }
    }
};

export { fillExterior, simplifyForCapsule };
export type { NavSeed, NavSimplifyResult };
