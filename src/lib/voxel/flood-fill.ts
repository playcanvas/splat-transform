import {
    BLOCK_EMPTY,
    BLOCK_MIXED,
    BLOCK_SOLID,
    FACE_MASKS_HI,
    FACE_MASKS_LO,
    SOLID_HI,
    SOLID_LO,
    SparseVoxelGrid,
    readBlockType,
    writeBlockType
} from './sparse-voxel-grid';

/**
 * Two-level BFS on a sparse voxel grid.
 *
 * Exploits the fact that most blocks in the blocked grid are BLOCK_EMPTY (all
 * 64 voxels free). When BFS reaches such a block, ALL voxels are reachable
 * (mutually face-connected within the 4x4x4 cube), so we mark the entire
 * block as visited and propagate at block granularity -- 64x fewer operations.
 *
 * Two queues:
 *   Block queue  -- processes BLOCK_EMPTY (in blocked) blocks at block level
 *   Voxel queue  -- processes individual voxels in BLOCK_MIXED blocks
 *
 * When the voxel BFS enters a BLOCK_EMPTY block it switches to block fill.
 * When block fill hits a BLOCK_MIXED neighbor it enqueues free face voxels.
 *
 * @param blocked - Sparse voxel grid marking blocked voxels.
 * @param blockSeeds - Block indices to seed block-level BFS from.
 * @param voxelSeeds - Voxel coordinates to seed voxel-level BFS from.
 * @param nx - Grid dimension X in voxels.
 * @param ny - Grid dimension Y in voxels.
 * @param nz - Grid dimension Z in voxels.
 * @returns Sparse voxel grid marking all reachable voxels.
 */
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

    const blockedTypes = blocked.types;
    const bMasks = blocked.masks;
    const visitedTypes = visited.types;
    const vMasks = visited.masks;

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

    const tryFillBlock = (blockIdx: number): boolean => {
        if (readBlockType(blockedTypes, blockIdx) !== BLOCK_EMPTY) return false;
        if (readBlockType(visitedTypes, blockIdx) !== BLOCK_EMPTY) return false;
        writeBlockType(visitedTypes, blockIdx, BLOCK_SOLID);
        if (bqSize >= bqCap) growBlockQueue();
        bqBuf[bqTail] = blockIdx;
        bqTail = (bqTail + 1) & bqMask;
        bqSize++;
        return true;
    };

    const enqueueFaceVoxels = (nBlockIdx: number, face: number, nBx: number, nBy: number, nBz: number): void => {
        const vbt = readBlockType(visitedTypes, nBlockIdx);
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
            writeBlockType(visitedTypes, nBlockIdx, BLOCK_MIXED);
            vMasks.set(nBlockIdx, freeLo, freeHi);
        } else {
            vMasks.lo[vs] = (vMasks.lo[vs] | freeLo) >>> 0;
            vMasks.hi[vs] = (vMasks.hi[vs] | freeHi) >>> 0;
            if (vMasks.lo[vs] === SOLID_LO && vMasks.hi[vs] === SOLID_HI) {
                vMasks.removeAt(vs);
                writeBlockType(visitedTypes, nBlockIdx, BLOCK_SOLID);
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

    const processBlock = (blockIdx: number): void => {
        const bx = blockIdx % nbx;
        const byBz = (blockIdx / nbx) | 0;
        const by = byBz % nby;
        const bz = (byBz / nby) | 0;

        if (bx > 0) {
            const ni = blockIdx - 1;
            const nbt = readBlockType(blockedTypes, ni);
            if (nbt === BLOCK_EMPTY) tryFillBlock(ni);
            else if (nbt === BLOCK_MIXED) enqueueFaceVoxels(ni, 1, bx - 1, by, bz);
        }
        if (bx < nbx - 1) {
            const ni = blockIdx + 1;
            const nbt = readBlockType(blockedTypes, ni);
            if (nbt === BLOCK_EMPTY) tryFillBlock(ni);
            else if (nbt === BLOCK_MIXED) enqueueFaceVoxels(ni, 0, bx + 1, by, bz);
        }
        if (by > 0) {
            const ni = blockIdx - nbx;
            const nbt = readBlockType(blockedTypes, ni);
            if (nbt === BLOCK_EMPTY) tryFillBlock(ni);
            else if (nbt === BLOCK_MIXED) enqueueFaceVoxels(ni, 3, bx, by - 1, bz);
        }
        if (by < nby - 1) {
            const ni = blockIdx + nbx;
            const nbt = readBlockType(blockedTypes, ni);
            if (nbt === BLOCK_EMPTY) tryFillBlock(ni);
            else if (nbt === BLOCK_MIXED) enqueueFaceVoxels(ni, 2, bx, by + 1, bz);
        }
        if (bz > 0) {
            const ni = blockIdx - bStride;
            const nbt = readBlockType(blockedTypes, ni);
            if (nbt === BLOCK_EMPTY) tryFillBlock(ni);
            else if (nbt === BLOCK_MIXED) enqueueFaceVoxels(ni, 5, bx, by, bz - 1);
        }
        if (bz < nbz - 1) {
            const ni = blockIdx + bStride;
            const nbt = readBlockType(blockedTypes, ni);
            if (nbt === BLOCK_EMPTY) tryFillBlock(ni);
            else if (nbt === BLOCK_MIXED) enqueueFaceVoxels(ni, 4, bx, by, bz + 1);
        }
    };

    const tryEnqueueVoxel = (ix: number, iy: number, iz: number): void => {
        const blockIdx = (ix >> 2) + (iy >> 2) * nbx + (iz >> 2) * bStride;

        const bbt = readBlockType(blockedTypes, blockIdx);
        if (bbt === BLOCK_SOLID) return;
        if (bbt === BLOCK_EMPTY) {
            tryFillBlock(blockIdx);
            return;
        }

        const bs = bMasks.slot(blockIdx);
        const bitIdx = (ix & 3) + ((iy & 3) << 2) + ((iz & 3) << 4);
        if (bitIdx < 32 ? (bMasks.lo[bs] >>> bitIdx) & 1 : (bMasks.hi[bs] >>> (bitIdx - 32)) & 1) return;

        const vbt = readBlockType(visitedTypes, blockIdx);
        if (vbt === BLOCK_SOLID) return;
        if (vbt === BLOCK_MIXED) {
            const vs = vMasks.slot(blockIdx);
            if (bitIdx < 32 ? (vMasks.lo[vs] >>> bitIdx) & 1 : (vMasks.hi[vs] >>> (bitIdx - 32)) & 1) return;
            if (bitIdx < 32) vMasks.lo[vs] = (vMasks.lo[vs] | (1 << bitIdx)) >>> 0;
            else vMasks.hi[vs] = (vMasks.hi[vs] | (1 << (bitIdx - 32))) >>> 0;
            if (vMasks.lo[vs] === SOLID_LO && vMasks.hi[vs] === SOLID_HI) {
                vMasks.removeAt(vs);
                writeBlockType(visitedTypes, blockIdx, BLOCK_SOLID);
            }
        } else {
            writeBlockType(visitedTypes, blockIdx, BLOCK_MIXED);
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

export { twoLevelBFS };
