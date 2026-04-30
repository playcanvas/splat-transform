import {
    BLOCK_EMPTY,
    BLOCK_SOLID,
    BLOCKS_PER_WORD,
    EVEN_BITS,
    SOLID_HI,
    SOLID_LO,
    SparseVoxelGrid,
    TYPE_MASK,
    readBlockType
} from './sparse-voxel-grid';

/**
 * Compute the grid of voxels that are visited but not blocked
 * (i.e. reachable empty voxels).
 *
 * @param visited - Grid of visited voxels (from BFS).
 * @param blocked - Grid of blocked voxels (obstacles).
 * @returns Grid containing only voxels that are visited AND not blocked.
 */
function computeEmptyGrid(visited: SparseVoxelGrid, blocked: SparseVoxelGrid): SparseVoxelGrid {
    const empty = new SparseVoxelGrid(visited.nx, visited.ny, visited.nz);
    const totalBlocks = visited.nbx * visited.nby * visited.nbz;
    const visitedTypes = visited.types;
    const blockedTypes = blocked.types;
    for (let w = 0; w < visitedTypes.length; w++) {
        const word = visitedTypes[w];
        if (word === 0) continue;
        let nonEmpty = ((word & EVEN_BITS) | ((word >>> 1) & EVEN_BITS)) >>> 0;
        const baseIdx = w * BLOCKS_PER_WORD;
        while (nonEmpty) {
            const bp = 31 - Math.clz32(nonEmpty & -nonEmpty);
            const lane = bp >>> 1;
            const blockIdx = baseIdx + lane;
            if (blockIdx >= totalBlocks) {
                nonEmpty = 0;
                break;
            }
            const vbt = (word >>> (lane << 1)) & TYPE_MASK;
            let vLo: number, vHi: number;
            if (vbt === BLOCK_SOLID) {
                vLo = SOLID_LO;
                vHi = SOLID_HI;
            } else {
                const vs = visited.masks.slot(blockIdx);
                vLo = visited.masks.lo[vs];
                vHi = visited.masks.hi[vs];
            }
            const bbt = readBlockType(blockedTypes, blockIdx);
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
            nonEmpty &= nonEmpty - 1;
        }
    }
    return empty;
}

/**
 * Compute the union of two sparse voxel grids (bitwise OR).
 *
 * @param a - First grid. By default a fresh clone is taken as the base;
 * with `consumeA=true` it is mutated in place and returned, saving one
 * full grid's worth of clone allocation.
 * @param b - Second grid (OR'd into the result).
 * @param consumeA - If true, `a` is mutated in place and returned. The
 * caller must not subsequently read `a` as an independent value
 * (the returned grid IS `a`).
 * @returns Grid containing the union of both inputs. Equal to `a` when
 * `consumeA=true`, otherwise a freshly cloned grid.
 */
function sparseOrGrids(
    a: SparseVoxelGrid,
    b: SparseVoxelGrid,
    consumeA: boolean = false
): SparseVoxelGrid {
    const result = consumeA ? a : a.clone();
    const totalBlocks = b.nbx * b.nby * b.nbz;
    const bTypes = b.types;
    for (let w = 0; w < bTypes.length; w++) {
        const word = bTypes[w];
        if (word === 0) continue;
        let nonEmpty = ((word & EVEN_BITS) | ((word >>> 1) & EVEN_BITS)) >>> 0;
        const baseIdx = w * BLOCKS_PER_WORD;
        while (nonEmpty) {
            const bp = 31 - Math.clz32(nonEmpty & -nonEmpty);
            const lane = bp >>> 1;
            const blockIdx = baseIdx + lane;
            if (blockIdx >= totalBlocks) {
                nonEmpty = 0;
                break;
            }
            const bt = (word >>> (lane << 1)) & TYPE_MASK;
            if (bt === BLOCK_SOLID) {
                result.orBlock(blockIdx, SOLID_LO, SOLID_HI);
            } else {
                const s = b.masks.slot(blockIdx);
                result.orBlock(blockIdx, b.masks.lo[s], b.masks.hi[s]);
            }
            nonEmpty &= nonEmpty - 1;
        }
    }
    return result;
}

export { computeEmptyGrid, sparseOrGrids };
