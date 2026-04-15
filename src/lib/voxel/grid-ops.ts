import {
    BLOCK_EMPTY,
    BLOCK_MIXED,
    BLOCK_SOLID,
    SOLID_HI,
    SOLID_LO,
    SparseVoxelGrid
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

/**
 * Compute the union of two sparse voxel grids (bitwise OR).
 *
 * @param a - First grid (cloned as the base).
 * @param b - Second grid (OR'd into the clone of a).
 * @returns New grid containing the union of both inputs.
 */
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

export { computeEmptyGrid, sparseOrGrids };
