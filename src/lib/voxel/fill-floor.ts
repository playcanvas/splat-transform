import { BlockMaskBuffer } from './block-mask-buffer';
import type { NavSimplifyResult } from './fill-exterior';
import type { Bounds } from '../data-table';
import {
    BLOCK_EMPTY,
    BLOCK_MIXED,
    BLOCK_SOLID,
    SOLID_HI,
    SOLID_LO,
    SparseVoxelGrid
} from './sparse-voxel-grid';
import { logger } from '../utils';

// 1D sliding-window minimum using a monotone deque.
// Window is symmetric: [i - R, i + R] clamped to [0, n).
const minFilter1D = (
    src: Int32Array, sOff: number,
    dst: Int32Array, dOff: number,
    n: number, R: number,
    deque: Int32Array
): void => {
    let dqH = 0, dqT = 0;

    for (let j = 0; j < Math.min(R, n); j++) {
        while (dqH < dqT && src[sOff + deque[dqT - 1]] >= src[sOff + j]) dqT--;
        deque[dqT++] = j;
    }

    for (let i = 0; i < n; i++) {
        const enter = i + R;
        if (enter < n) {
            while (dqH < dqT && src[sOff + deque[dqT - 1]] >= src[sOff + enter]) dqT--;
            deque[dqT++] = enter;
        }
        while (deque[dqH] < i - R) dqH++;
        dst[dOff + i] = src[sOff + deque[dqH]];
    }
};

// Separable 2D minimum filter on row-major Int32Array(nx * nz).
const minFilter2D = (
    src: Int32Array, dst: Int32Array,
    nx: number, nz: number, R: number
): void => {
    const deque = new Int32Array(Math.max(nx, nz));

    for (let iz = 0; iz < nz; iz++) {
        const off = iz * nx;
        minFilter1D(src, off, dst, off, nx, R, deque);
    }

    const colSrc = new Int32Array(nz);
    const colDst = new Int32Array(nz);
    for (let ix = 0; ix < nx; ix++) {
        for (let iz = 0; iz < nz; iz++) colSrc[iz] = dst[iz * nx + ix];
        minFilter1D(colSrc, 0, colDst, 0, nz, R, deque);
        for (let iz = 0; iz < nz; iz++) dst[iz * nx + ix] = colDst[iz];
    }
};

// Block-level floor scan: for each voxel column, find Y of the first solid
// voxel scanning upward from the bottom. Sentinel `ny` = no floor found.
const scanFloorHeights = (grid: SparseVoxelGrid, floorY: Int32Array, nx: number, ny: number, nz: number): void => {
    const { nbx, nby, nbz, bStride, blockType, masks } = grid;

    for (let bz = 0; bz < nbz; bz++) {
        for (let bx = 0; bx < nbx; bx++) {
            let remaining = 0xFFFF;

            for (let by = 0; by < nby && remaining; by++) {
                const blockIdx = bx + by * nbx + bz * bStride;
                const bt = blockType[blockIdx];
                if (bt === BLOCK_EMPTY) continue;

                const baseY = by << 2;

                if (bt === BLOCK_SOLID) {
                    for (let lz = 0; lz < 4; lz++) {
                        for (let lx = 0; lx < 4; lx++) {
                            if (remaining & (1 << (lz * 4 + lx))) {
                                floorY[((bz << 2) + lz) * nx + (bx << 2) + lx] = baseY;
                            }
                        }
                    }
                    break;
                }

                // BLOCK_MIXED
                const s = masks.slot(blockIdx);
                const lo = masks.lo[s];
                const hi = masks.hi[s];

                for (let lz = 0; lz < 4; lz++) {
                    const inHi = lz >= 2;
                    const word = inHi ? hi : lo;
                    const base = (lz & 1) << 4;

                    for (let lx = 0; lx < 4; lx++) {
                        const bit = 1 << (lz * 4 + lx);
                        if (!(remaining & bit)) continue;

                        const shift = base + lx;
                        for (let ly = 0; ly < 4; ly++) {
                            if ((word >>> (shift + (ly << 2))) & 1) {
                                floorY[((bz << 2) + lz) * nx + (bx << 2) + lx] = baseY + ly;
                                remaining &= ~bit;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
};

// Block-level fill for gap/no-floor columns. Only fills columns where
// the original floor is absent but the dilated floor exists.
const fillGapColumns = (
    grid: SparseVoxelGrid,
    originalFloorY: Int32Array,
    dilatedFloorY: Int32Array,
    nx: number, ny: number, nz: number
): void => {
    const { nbx, nby, nbz, bStride, blockType } = grid;

    for (let bz = 0; bz < nbz; bz++) {
        for (let bx = 0; bx < nbx; bx++) {
            const fills = new Int32Array(16);
            let minFill = ny + 1, maxFill = 0;

            for (let lz = 0; lz < 4; lz++) {
                for (let lx = 0; lx < 4; lx++) {
                    const col = ((bz << 2) + lz) * nx + (bx << 2) + lx;
                    let fty = 0;
                    if (originalFloorY[col] >= ny && dilatedFloorY[col] < ny) {
                        fty = dilatedFloorY[col];
                    }
                    const idx = lz * 4 + lx;
                    fills[idx] = fty;
                    if (fty > maxFill) maxFill = fty;
                    if (fty < minFill) minFill = fty;
                }
            }

            if (maxFill === 0) continue;

            const maxBlockY = Math.min((maxFill - 1) >> 2, nby - 1);
            for (let by = 0; by <= maxBlockY; by++) {
                const blockIdx = bx + by * nbx + bz * bStride;
                if (blockType[blockIdx] === BLOCK_SOLID) continue;

                const blockTopY = (by + 1) << 2;
                if (blockTopY <= minFill) {
                    grid.orBlock(blockIdx, SOLID_LO, SOLID_HI);
                } else {
                    let lo = 0, hi = 0;
                    const blockBaseY = by << 2;
                    for (let lz = 0; lz < 4; lz++) {
                        for (let lx = 0; lx < 4; lx++) {
                            const fty = fills[lz * 4 + lx];
                            for (let ly = 0; ly < 4; ly++) {
                                if (blockBaseY + ly < fty) {
                                    const bitIdx = lx + (ly << 2) + (lz << 4);
                                    if (bitIdx < 32) lo |= (1 << bitIdx);
                                    else hi |= (1 << (bitIdx - 32));
                                }
                            }
                        }
                    }
                    if (lo || hi) {
                        grid.orBlock(blockIdx, lo >>> 0, hi >>> 0);
                    }
                }
            }
        }
    }
};

/**
 * Fill below the floor surface to block outdoor scene edges.
 *
 * Uses a 2D floor-height map instead of 3D dilation for scalability:
 * 1. Block-level scan to find per-column floor heights
 * 2. 2D separable min-filter to bridge floor gaps (same effect as XZ dilation)
 * 3. Block-level fill for gap/no-floor columns only
 *
 * Columns that already have a floor are not modified -- the floor itself is
 * the barrier and `carveInterior` handles sub-floor space by inversion.
 *
 * @param buffer - Voxelized scene data.
 * @param gridBounds - Axis-aligned bounds of the voxel grid.
 * @param voxelResolution - Size of each voxel in world units.
 * @param dilation - XZ dilation radius in world units for bridging floor gaps.
 * @returns Modified buffer with gap/edge columns filled solid.
 */
const fillFloor = (
    buffer: BlockMaskBuffer,
    gridBounds: Bounds,
    voxelResolution: number,
    dilation: number
): NavSimplifyResult => {
    if (!Number.isFinite(voxelResolution) || voxelResolution <= 0) {
        throw new Error(`fillFloor: voxelResolution must be finite and > 0, got ${voxelResolution}`);
    }
    if (!Number.isFinite(dilation) || dilation <= 0) {
        throw new Error(`fillFloor: dilation must be finite and > 0, got ${dilation}`);
    }

    const nx = Math.round((gridBounds.max.x - gridBounds.min.x) / voxelResolution);
    const ny = Math.round((gridBounds.max.y - gridBounds.min.y) / voxelResolution);
    const nz = Math.round((gridBounds.max.z - gridBounds.min.z) / voxelResolution);

    if (nx % 4 !== 0 || ny % 4 !== 0 || nz % 4 !== 0) {
        throw new Error(`Grid dimensions must be multiples of 4, got ${nx}x${ny}x${nz}`);
    }

    if (buffer.count === 0) {
        return { buffer, gridBounds };
    }

    const halfExtent = Math.ceil(dilation / voxelResolution);
    const nbx = nx >> 2;
    const nby = ny >> 2;
    const nbz = nz >> 2;

    logger.progress.begin(4);
    let progressComplete = false;

    try {
        const grid = SparseVoxelGrid.fromBuffer(buffer, nx, ny, nz);
        logger.progress.step();

        const floorY = new Int32Array(nx * nz).fill(ny);
        scanFloorHeights(grid, floorY, nx, ny, nz);
        logger.progress.step();

        const dilatedFloorY = new Int32Array(nx * nz);
        minFilter2D(floorY, dilatedFloorY, nx, nz, halfExtent);
        fillGapColumns(grid, floorY, dilatedFloorY, nx, ny, nz);
        logger.progress.step();

        const result = grid.toBuffer(0, 0, 0, nbx, nby, nbz);
        logger.progress.step();
        progressComplete = true;

        return { buffer: result, gridBounds };
    } finally {
        if (!progressComplete) {
            logger.progress.cancel();
        }
    }
};

export { fillFloor };
