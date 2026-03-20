import {
    BlockAccumulator,
    mortonToXYZ,
    xyzToMorton,
    type Bounds
} from './sparse-octree';
import { logger } from '../utils/logger';

/**
 * Seed position for capsule navigation simplification.
 */
type NavSeed = {
    x: number;
    y: number;
    z: number;
};

/**
 * Build a list of (dx, dz) offsets forming a filled circle in the XZ plane.
 *
 * @param radius - Circle radius in voxel units.
 * @returns Array of [dx, dz] offset pairs within the circle.
 */
const buildCircularKernel = (radius: number): number[][] => {
    const offsets: number[][] = [];
    const r2 = radius * radius;
    for (let dx = -radius; dx <= radius; dx++) {
        for (let dz = -radius; dz <= radius; dz++) {
            if (dx * dx + dz * dz <= r2) {
                offsets.push([dx, dz]);
            }
        }
    }
    return offsets;
};

/**
 * Populate a dense solid grid from a BlockAccumulator.
 *
 * @param accumulator - Source block data.
 * @param grid - Pre-allocated Uint8Array (nx * ny * nz), zeroed.
 * @param nx - Grid X dimension in voxels.
 * @param ny - Grid Y dimension.
 * @param nz - Grid Z dimension.
 */
const fillDenseSolidGrid = (
    accumulator: BlockAccumulator,
    grid: Uint8Array,
    nx: number, ny: number, nz: number
): void => {
    const stride = nx * ny;

    const solidMortons = accumulator.getSolidBlocks();
    for (let i = 0; i < solidMortons.length; i++) {
        const [bx, by, bz] = mortonToXYZ(solidMortons[i]);
        const baseX = bx << 2;
        const baseY = by << 2;
        const baseZ = bz << 2;
        for (let lz = 0; lz < 4; lz++) {
            const iz = baseZ + lz;
            if (iz >= nz) continue;
            for (let ly = 0; ly < 4; ly++) {
                const iy = baseY + ly;
                if (iy >= ny) continue;
                const rowOff = iz * stride + iy * nx;
                for (let lx = 0; lx < 4; lx++) {
                    const ix = baseX + lx;
                    if (ix < nx) grid[rowOff + ix] = 1;
                }
            }
        }
    }

    const mixed = accumulator.getMixedBlocks();
    for (let i = 0; i < mixed.morton.length; i++) {
        const [bx, by, bz] = mortonToXYZ(mixed.morton[i]);
        const lo = mixed.masks[i * 2];
        const hi = mixed.masks[i * 2 + 1];
        const baseX = bx << 2;
        const baseY = by << 2;
        const baseZ = bz << 2;
        for (let lz = 0; lz < 4; lz++) {
            const iz = baseZ + lz;
            if (iz >= nz) continue;
            for (let ly = 0; ly < 4; ly++) {
                const iy = baseY + ly;
                if (iy >= ny) continue;
                const rowOff = iz * stride + iy * nx;
                for (let lx = 0; lx < 4; lx++) {
                    const bitIdx = lx + (ly << 2) + (lz << 4);
                    const word = bitIdx < 32 ? lo : hi;
                    const bit = bitIdx < 32 ? bitIdx : bitIdx - 32;
                    if ((word >>> bit) & 1) {
                        const ix = baseX + lx;
                        if (ix < nx) grid[rowOff + ix] = 1;
                    }
                }
            }
        }
    }
};

/**
 * XZ morphological dilation: for each cell matching `matchValue` in `src`,
 * mark all cells within the circular kernel in `dst`.
 *
 * @param src - Source grid.
 * @param dst - Destination grid (must be pre-zeroed).
 * @param nx - Grid X dimension.
 * @param ny - Grid Y dimension.
 * @param nz - Grid Z dimension.
 * @param kernel - Circular kernel offsets [dx, dz].
 * @param matchValue - Value to match in src.
 */
const dilateXZ = (
    src: Uint8Array,
    dst: Uint8Array,
    nx: number, ny: number, nz: number,
    kernel: number[][],
    matchValue: number
): void => {
    const stride = nx * ny;
    const kLen = kernel.length;

    for (let iz = 0; iz < nz; iz++) {
        const zOff = iz * stride;
        for (let iy = 0; iy < ny; iy++) {
            const yzOff = zOff + iy * nx;
            for (let ix = 0; ix < nx; ix++) {
                if (src[yzOff + ix] !== matchValue) continue;
                for (let k = 0; k < kLen; k++) {
                    const kx = ix + kernel[k][0];
                    const kz = iz + kernel[k][1];
                    if (kx >= 0 && kx < nx && kz >= 0 && kz < nz) {
                        dst[kz * stride + iy * nx + kx] = 1;
                    }
                }
            }
        }
    }
};

/**
 * Y-axis morphological dilation via sliding window.
 * For each column, a cell is marked if any cell within `halfExtent` in Y
 * is set in the source.
 *
 * @param src - Source grid.
 * @param dst - Destination grid (must be pre-zeroed).
 * @param nx - Grid X dimension.
 * @param ny - Grid Y dimension.
 * @param nz - Grid Z dimension.
 * @param halfExtent - Half-window size in voxels.
 */
const dilateY = (
    src: Uint8Array,
    dst: Uint8Array,
    nx: number, ny: number, nz: number,
    halfExtent: number
): void => {
    const stride = nx * ny;

    for (let iz = 0; iz < nz; iz++) {
        const zOff = iz * stride;
        for (let ix = 0; ix < nx; ix++) {
            let count = 0;
            const winEnd = Math.min(halfExtent, ny - 1);
            for (let iy = 0; iy <= winEnd; iy++) {
                if (src[zOff + iy * nx + ix]) count++;
            }

            for (let iy = 0; iy < ny; iy++) {
                if (count > 0) dst[zOff + iy * nx + ix] = 1;

                const exitY = iy - halfExtent;
                if (exitY >= 0 && src[zOff + exitY * nx + ix]) count--;

                const enterY = iy + halfExtent + 1;
                if (enterY < ny && src[zOff + enterY * nx + ix]) count++;
            }
        }
    }
};

/**
 * XZ morphological erosion: a cell remains solid only if ALL cells within the
 * circular kernel are solid in `src`. Out-of-bounds cells are treated as solid
 * (grid boundary convention).
 *
 * @param src - Source grid.
 * @param dst - Destination grid (must be pre-zeroed).
 * @param nx - Grid X dimension.
 * @param ny - Grid Y dimension.
 * @param nz - Grid Z dimension.
 * @param kernel - Circular kernel offsets [dx, dz].
 */
const erodeXZ = (
    src: Uint8Array,
    dst: Uint8Array,
    nx: number, ny: number, nz: number,
    kernel: number[][]
): void => {
    const stride = nx * ny;
    const kLen = kernel.length;

    for (let iz = 0; iz < nz; iz++) {
        const zOff = iz * stride;
        for (let iy = 0; iy < ny; iy++) {
            const yzOff = zOff + iy * nx;
            for (let ix = 0; ix < nx; ix++) {
                let allSolid = true;
                for (let k = 0; k < kLen; k++) {
                    const kx = ix + kernel[k][0];
                    const kz = iz + kernel[k][1];
                    if (kx >= 0 && kx < nx && kz >= 0 && kz < nz) {
                        if (!src[kz * stride + iy * nx + kx]) {
                            allSolid = false;
                            break;
                        }
                    }
                }
                if (allSolid) dst[yzOff + ix] = 1;
            }
        }
    }
};

/**
 * Y-axis morphological erosion via sliding window.
 * A cell remains solid only if ALL cells within `halfExtent` in Y are solid.
 * Out-of-bounds cells are treated as solid (grid boundary convention).
 *
 * @param src - Source grid.
 * @param dst - Destination grid (must be pre-zeroed).
 * @param nx - Grid X dimension.
 * @param ny - Grid Y dimension.
 * @param nz - Grid Z dimension.
 * @param halfExtent - Half-window size in voxels.
 */
const erodeY = (
    src: Uint8Array,
    dst: Uint8Array,
    nx: number, ny: number, nz: number,
    halfExtent: number
): void => {
    const stride = nx * ny;

    for (let iz = 0; iz < nz; iz++) {
        const zOff = iz * stride;
        for (let ix = 0; ix < nx; ix++) {
            let zeroCount = 0;
            const winEnd = Math.min(halfExtent, ny - 1);
            for (let iy = 0; iy <= winEnd; iy++) {
                if (!src[zOff + iy * nx + ix]) zeroCount++;
            }

            for (let iy = 0; iy < ny; iy++) {
                if (zeroCount === 0) dst[zOff + iy * nx + ix] = 1;

                const exitY = iy - halfExtent;
                if (exitY >= 0 && !src[zOff + exitY * nx + ix]) zeroCount--;

                const enterY = iy + halfExtent + 1;
                if (enterY < ny && !src[zOff + enterY * nx + ix]) zeroCount++;
            }
        }
    }
};

/**
 * Convert a dense boolean grid back into a BlockAccumulator.
 *
 * @param grid - Dense grid with 1 = solid.
 * @param nx - Grid X dimension (must be divisible by 4).
 * @param ny - Grid Y dimension (must be divisible by 4).
 * @param nz - Grid Z dimension (must be divisible by 4).
 * @returns New BlockAccumulator with blocks matching the grid.
 */
const denseGridToAccumulator = (
    grid: Uint8Array,
    nx: number, ny: number, nz: number
): BlockAccumulator => {
    const acc = new BlockAccumulator();
    const nbx = nx >> 2;
    const nby = ny >> 2;
    const nbz = nz >> 2;
    const stride = nx * ny;

    for (let bz = 0; bz < nbz; bz++) {
        for (let by = 0; by < nby; by++) {
            for (let bx = 0; bx < nbx; bx++) {
                let lo = 0;
                let hi = 0;
                const baseX = bx << 2;
                const baseY = by << 2;
                const baseZ = bz << 2;

                for (let lz = 0; lz < 4; lz++) {
                    for (let ly = 0; ly < 4; ly++) {
                        for (let lx = 0; lx < 4; lx++) {
                            if (grid[(baseX + lx) + (baseY + ly) * nx + (baseZ + lz) * stride]) {
                                const bitIdx = lx + (ly << 2) + (lz << 4);
                                if (bitIdx < 32) {
                                    lo |= (1 << bitIdx);
                                } else {
                                    hi |= (1 << (bitIdx - 32));
                                }
                            }
                        }
                    }
                }

                if (lo !== 0 || hi !== 0) {
                    acc.addBlock(xyzToMorton(bx, by, bz), lo, hi);
                }
            }
        }
    }

    return acc;
};

const FREE = 0;
const BLOCKED = 1;
const REACHABLE = 2;

/**
 * Simplify voxel collision data for upright capsule navigation.
 *
 * Algorithm:
 * 1. Build dense solid grid from the accumulator.
 * 2. Dilate solid by the capsule shape (Minkowski sum) to get the clearance
 *    grid -- cells where the capsule center cannot be placed.
 * 3. BFS flood fill from the seed through free (non-blocked) cells to find
 *    all reachable capsule-center positions.
 * 4. Invert: every non-reachable cell becomes solid (negative space carving).
 * 5. Erode the solid by the capsule shape (Minkowski subtraction) to shrink
 *    surfaces back to their original positions, undoing the inflation from
 *    step 2 so the runtime capsule query produces correct collisions.
 *
 * Grid boundaries are treated as solid, so the fill is always bounded even
 * in unsealed scenes.
 *
 * @param accumulator - BlockAccumulator with filtered voxelization results.
 * @param gridBounds - Grid bounds aligned to block boundaries.
 * @param voxelResolution - Size of each voxel in world units.
 * @param capsuleHeight - Total capsule height in world units.
 * @param capsuleRadius - Capsule radius in world units.
 * @param seed - Seed position in world space (must be in a free region).
 * @returns New BlockAccumulator with simplified collision voxels.
 */
const simplifyForCapsule = (
    accumulator: BlockAccumulator,
    gridBounds: Bounds,
    voxelResolution: number,
    capsuleHeight: number,
    capsuleRadius: number,
    seed: NavSeed
): BlockAccumulator => {
    const nx = Math.round((gridBounds.max.x - gridBounds.min.x) / voxelResolution);
    const ny = Math.round((gridBounds.max.y - gridBounds.min.y) / voxelResolution);
    const nz = Math.round((gridBounds.max.z - gridBounds.min.z) / voxelResolution);
    const totalVoxels = nx * ny * nz;
    const stride = nx * ny;

    const kernelR = Math.ceil(capsuleRadius / voxelResolution) + 1;
    const yHalfExtent = Math.ceil(capsuleHeight / (2 * voxelResolution)) + 1;
    const kernel = buildCircularKernel(kernelR);

    const memoryMB = Math.round(totalVoxels * 3 / (1024 * 1024));
    logger.debug(`nav simplify: grid ${nx}x${ny}x${nz} (${totalVoxels} voxels, ~${memoryMB} MB), clearance r=${kernelR} (${kernel.length} cells), y half=${yHalfExtent}`);

    if (memoryMB > 512) {
        logger.warn(`nav simplify: large grid requires ~${memoryMB} MB. Consider using a coarser -R value to reduce memory.`);
    }

    // Phase 1: build dense solid grid from accumulator
    const solidGrid = new Uint8Array(totalVoxels);
    fillDenseSolidGrid(accumulator, solidGrid, nx, ny, nz);

    let solidCount = 0;
    for (let i = 0; i < totalVoxels; i++) {
        if (solidGrid[i]) solidCount++;
    }
    logger.debug(`nav simplify: ${solidCount} solid voxels`);

    // Phase 2: capsule clearance grid (Minkowski dilation of solid by capsule)
    const tempA = new Uint8Array(totalVoxels);
    dilateXZ(solidGrid, tempA, nx, ny, nz, kernel, 1);

    const tempB = new Uint8Array(totalVoxels);
    dilateY(tempA, tempB, nx, ny, nz, yHalfExtent);

    // Phase 3: flood fill from seed through free (non-blocked) cells
    const seedIx = Math.floor((seed.x - gridBounds.min.x) / voxelResolution);
    const seedIy = Math.floor((seed.y - gridBounds.min.y) / voxelResolution);
    const seedIz = Math.floor((seed.z - gridBounds.min.z) / voxelResolution);

    if (seedIx < 0 || seedIx >= nx || seedIy < 0 || seedIy >= ny || seedIz < 0 || seedIz >= nz) {
        logger.warn(`nav simplify: seed (${seed.x}, ${seed.y}, ${seed.z}) outside grid, skipping`);
        return accumulator;
    }

    const seedIdx = seedIx + seedIy * nx + seedIz * stride;
    if (tempB[seedIdx] !== FREE) {
        logger.warn(`nav simplify: seed (${seed.x}, ${seed.y}, ${seed.z}) in blocked region, skipping`);
        return accumulator;
    }

    // BFS flood fill using a Uint32Array circular buffer.
    // A JS Array would OOM on large grids because it stores every visited
    // cell. The circular buffer only holds the active frontier.
    const QUEUE_BITS = 25;
    const QUEUE_CAP = 1 << QUEUE_BITS; // 32M entries = 128 MB
    const QUEUE_MASK = QUEUE_CAP - 1;
    const bfsQueue = new Uint32Array(QUEUE_CAP);
    let qHead = 0;
    let qTail = 0;
    let reachableCount = 0;

    tempB[seedIdx] = REACHABLE;
    bfsQueue[qTail] = seedIdx;
    qTail = (qTail + 1) & QUEUE_MASK;

    while (qHead !== qTail) {
        const idx = bfsQueue[qHead];
        qHead = (qHead + 1) & QUEUE_MASK;
        reachableCount++;

        const ix = idx % nx;
        const iy = Math.floor((idx % stride) / nx);
        const iz = Math.floor(idx / stride);

        if (ix > 0 && tempB[idx - 1] === FREE) {
            tempB[idx - 1] = REACHABLE;
            bfsQueue[qTail] = idx - 1;
            qTail = (qTail + 1) & QUEUE_MASK;
        }
        if (ix < nx - 1 && tempB[idx + 1] === FREE) {
            tempB[idx + 1] = REACHABLE;
            bfsQueue[qTail] = idx + 1;
            qTail = (qTail + 1) & QUEUE_MASK;
        }
        if (iy > 0 && tempB[idx - nx] === FREE) {
            tempB[idx - nx] = REACHABLE;
            bfsQueue[qTail] = idx - nx;
            qTail = (qTail + 1) & QUEUE_MASK;
        }
        if (iy < ny - 1 && tempB[idx + nx] === FREE) {
            tempB[idx + nx] = REACHABLE;
            bfsQueue[qTail] = idx + nx;
            qTail = (qTail + 1) & QUEUE_MASK;
        }
        if (iz > 0 && tempB[idx - stride] === FREE) {
            tempB[idx - stride] = REACHABLE;
            bfsQueue[qTail] = idx - stride;
            qTail = (qTail + 1) & QUEUE_MASK;
        }
        if (iz < nz - 1 && tempB[idx + stride] === FREE) {
            tempB[idx + stride] = REACHABLE;
            bfsQueue[qTail] = idx + stride;
            qTail = (qTail + 1) & QUEUE_MASK;
        }
    }

    logger.debug(`nav simplify: ${reachableCount} reachable cells (${(reachableCount / totalVoxels * 100).toFixed(1)}%)`);

    // Phase 4: invert reachable to solid
    // Everything the capsule cannot reach becomes solid. This produces a
    // "negative space" carving: the reachable volume is empty, everything
    // else is filled. Large contiguous solid regions compress to single
    // SOLID_LEAF_MARKER nodes in the octree, and there are no surface
    // holes since every non-reachable cell is solid.
    let outputCount = 0;
    for (let i = 0; i < totalVoxels; i++) {
        if (tempB[i] !== REACHABLE) {
            solidGrid[i] = 1;
            outputCount++;
        } else {
            solidGrid[i] = 0;
        }
    }

    logger.debug(`nav simplify: ${outputCount} solid voxels after inversion`);

    // Phase 5: erode solid by capsule shape (Minkowski subtraction)
    // The inversion inflated the solid boundary by the capsule clearance.
    // Eroding by the same kernel shrinks it back to the original surface
    // positions so the runtime capsule query produces correct collisions.
    tempA.fill(0);
    erodeXZ(solidGrid, tempA, nx, ny, nz, kernel);

    solidGrid.fill(0);
    erodeY(tempA, solidGrid, nx, ny, nz, yHalfExtent);

    let finalCount = 0;
    for (let i = 0; i < totalVoxels; i++) {
        if (solidGrid[i]) finalCount++;
    }

    logger.log(`nav simplify: ${finalCount} solid voxels (from ${solidCount} original, ${reachableCount} reachable carved out)`);

    return denseGridToAccumulator(solidGrid, nx, ny, nz);
};

export { simplifyForCapsule };
export type { NavSeed };
