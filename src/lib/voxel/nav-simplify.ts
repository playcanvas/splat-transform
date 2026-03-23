import { Vec3 } from 'playcanvas';

import {
    BlockAccumulator,
    mortonToXYZ,
    popcount,
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
 * Result of capsule navigation simplification.
 */
type NavSimplifyResult = {
    accumulator: BlockAccumulator;
    gridBounds: Bounds;
};

/**
 * Populate a bitfield grid from a BlockAccumulator.
 * Each bit in the Uint32Array represents one voxel (1 = solid).
 *
 * @param accumulator - Source block data.
 * @param grid - Pre-allocated Uint32Array (ceil(nx*ny*nz / 32)), zeroed.
 * @param nx - Grid X dimension in voxels.
 * @param ny - Grid Y dimension.
 * @param nz - Grid Z dimension.
 */
const fillDenseSolidGrid = (
    accumulator: BlockAccumulator,
    grid: Uint32Array,
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
                    if (ix < nx) {
                        const idx = rowOff + ix;
                        grid[idx >>> 5] |= (1 << (idx & 31));
                    }
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
                        if (ix < nx) {
                            const idx = rowOff + ix;
                            grid[idx >>> 5] |= (1 << (idx & 31));
                        }
                    }
                }
            }
        }
    }
};

/**
 * X-axis morphological dilation via sliding window (bitfield version).
 * A cell is marked if any cell within `halfExtent` in X is set.
 *
 * @param src - Source bitfield.
 * @param dst - Destination bitfield (must be pre-zeroed).
 * @param nx - Grid X dimension.
 * @param ny - Grid Y dimension.
 * @param nz - Grid Z dimension.
 * @param halfExtent - Half-window size in voxels.
 */
const dilateX = (
    src: Uint32Array, dst: Uint32Array,
    nx: number, ny: number, nz: number,
    halfExtent: number
): void => {
    const stride = nx * ny;
    for (let iz = 0; iz < nz; iz++) {
        for (let iy = 0; iy < ny; iy++) {
            const rowOff = iz * stride + iy * nx;
            let count = 0;
            const winEnd = Math.min(halfExtent, nx - 1);
            for (let ix = 0; ix <= winEnd; ix++) {
                const idx = rowOff + ix;
                if ((src[idx >>> 5] >>> (idx & 31)) & 1) count++;
            }
            for (let ix = 0; ix < nx; ix++) {
                const idx = rowOff + ix;
                if (count > 0) dst[idx >>> 5] |= (1 << (idx & 31));
                const exitX = ix - halfExtent;
                if (exitX >= 0) {
                    const ei = rowOff + exitX;
                    if ((src[ei >>> 5] >>> (ei & 31)) & 1) count--;
                }
                const enterX = ix + halfExtent + 1;
                if (enterX < nx) {
                    const ni = rowOff + enterX;
                    if ((src[ni >>> 5] >>> (ni & 31)) & 1) count++;
                }
            }
        }
    }
};

/**
 * Y-axis morphological dilation via sliding window (bitfield version).
 * A cell is marked if any cell within `halfExtent` in Y is set.
 *
 * @param src - Source bitfield.
 * @param dst - Destination bitfield (must be pre-zeroed).
 * @param nx - Grid X dimension.
 * @param ny - Grid Y dimension.
 * @param nz - Grid Z dimension.
 * @param halfExtent - Half-window size in voxels.
 */
const dilateY = (
    src: Uint32Array, dst: Uint32Array,
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
                const idx = zOff + iy * nx + ix;
                if ((src[idx >>> 5] >>> (idx & 31)) & 1) count++;
            }
            for (let iy = 0; iy < ny; iy++) {
                const idx = zOff + iy * nx + ix;
                if (count > 0) dst[idx >>> 5] |= (1 << (idx & 31));
                const exitY = iy - halfExtent;
                if (exitY >= 0) {
                    const ei = zOff + exitY * nx + ix;
                    if ((src[ei >>> 5] >>> (ei & 31)) & 1) count--;
                }
                const enterY = iy + halfExtent + 1;
                if (enterY < ny) {
                    const ni = zOff + enterY * nx + ix;
                    if ((src[ni >>> 5] >>> (ni & 31)) & 1) count++;
                }
            }
        }
    }
};

/**
 * Z-axis morphological dilation via sliding window (bitfield version).
 * A cell is marked if any cell within `halfExtent` in Z is set.
 *
 * @param src - Source bitfield.
 * @param dst - Destination bitfield (must be pre-zeroed).
 * @param nx - Grid X dimension.
 * @param ny - Grid Y dimension.
 * @param nz - Grid Z dimension.
 * @param halfExtent - Half-window size in voxels.
 */
const dilateZ = (
    src: Uint32Array, dst: Uint32Array,
    nx: number, ny: number, nz: number,
    halfExtent: number
): void => {
    const stride = nx * ny;
    for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
            let count = 0;
            const winEnd = Math.min(halfExtent, nz - 1);
            for (let iz = 0; iz <= winEnd; iz++) {
                const idx = iz * stride + iy * nx + ix;
                if ((src[idx >>> 5] >>> (idx & 31)) & 1) count++;
            }
            for (let iz = 0; iz < nz; iz++) {
                const idx = iz * stride + iy * nx + ix;
                if (count > 0) dst[idx >>> 5] |= (1 << (idx & 31));
                const exitZ = iz - halfExtent;
                if (exitZ >= 0) {
                    const ei = exitZ * stride + iy * nx + ix;
                    if ((src[ei >>> 5] >>> (ei & 31)) & 1) count--;
                }
                const enterZ = iz + halfExtent + 1;
                if (enterZ < nz) {
                    const ni = enterZ * stride + iy * nx + ix;
                    if ((src[ni >>> 5] >>> (ni & 31)) & 1) count++;
                }
            }
        }
    }
};

/**
 * X-axis morphological erosion via sliding window (bitfield version).
 * A cell remains solid only if ALL cells within `halfExtent` in X are solid.
 * Out-of-bounds cells are treated as solid (grid boundary convention).
 *
 * @param src - Source bitfield.
 * @param dst - Destination bitfield (must be pre-zeroed).
 * @param nx - Grid X dimension.
 * @param ny - Grid Y dimension.
 * @param nz - Grid Z dimension.
 * @param halfExtent - Half-window size in voxels.
 */
const erodeX = (
    src: Uint32Array, dst: Uint32Array,
    nx: number, ny: number, nz: number,
    halfExtent: number
): void => {
    const stride = nx * ny;
    for (let iz = 0; iz < nz; iz++) {
        for (let iy = 0; iy < ny; iy++) {
            const rowOff = iz * stride + iy * nx;
            let zeroCount = 0;
            const winEnd = Math.min(halfExtent, nx - 1);
            for (let ix = 0; ix <= winEnd; ix++) {
                const idx = rowOff + ix;
                if (!((src[idx >>> 5] >>> (idx & 31)) & 1)) zeroCount++;
            }
            for (let ix = 0; ix < nx; ix++) {
                const idx = rowOff + ix;
                if (zeroCount === 0) dst[idx >>> 5] |= (1 << (idx & 31));
                const exitX = ix - halfExtent;
                if (exitX >= 0) {
                    const ei = rowOff + exitX;
                    if (!((src[ei >>> 5] >>> (ei & 31)) & 1)) zeroCount--;
                }
                const enterX = ix + halfExtent + 1;
                if (enterX < nx) {
                    const ni = rowOff + enterX;
                    if (!((src[ni >>> 5] >>> (ni & 31)) & 1)) zeroCount++;
                }
            }
        }
    }
};

/**
 * Y-axis morphological erosion via sliding window (bitfield version).
 * A cell remains solid only if ALL cells within `halfExtent` in Y are solid.
 * Out-of-bounds cells are treated as solid (grid boundary convention).
 *
 * @param src - Source bitfield.
 * @param dst - Destination bitfield (must be pre-zeroed).
 * @param nx - Grid X dimension.
 * @param ny - Grid Y dimension.
 * @param nz - Grid Z dimension.
 * @param halfExtent - Half-window size in voxels.
 */
const erodeY = (
    src: Uint32Array, dst: Uint32Array,
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
                const idx = zOff + iy * nx + ix;
                if (!((src[idx >>> 5] >>> (idx & 31)) & 1)) zeroCount++;
            }
            for (let iy = 0; iy < ny; iy++) {
                const idx = zOff + iy * nx + ix;
                if (zeroCount === 0) dst[idx >>> 5] |= (1 << (idx & 31));
                const exitY = iy - halfExtent;
                if (exitY >= 0) {
                    const ei = zOff + exitY * nx + ix;
                    if (!((src[ei >>> 5] >>> (ei & 31)) & 1)) zeroCount--;
                }
                const enterY = iy + halfExtent + 1;
                if (enterY < ny) {
                    const ni = zOff + enterY * nx + ix;
                    if (!((src[ni >>> 5] >>> (ni & 31)) & 1)) zeroCount++;
                }
            }
        }
    }
};

/**
 * Z-axis morphological erosion via sliding window (bitfield version).
 * A cell remains solid only if ALL cells within `halfExtent` in Z are solid.
 * Out-of-bounds cells are treated as solid (grid boundary convention).
 *
 * @param src - Source bitfield.
 * @param dst - Destination bitfield (must be pre-zeroed).
 * @param nx - Grid X dimension.
 * @param ny - Grid Y dimension.
 * @param nz - Grid Z dimension.
 * @param halfExtent - Half-window size in voxels.
 */
const erodeZ = (
    src: Uint32Array, dst: Uint32Array,
    nx: number, ny: number, nz: number,
    halfExtent: number
): void => {
    const stride = nx * ny;
    for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
            let zeroCount = 0;
            const winEnd = Math.min(halfExtent, nz - 1);
            for (let iz = 0; iz <= winEnd; iz++) {
                const idx = iz * stride + iy * nx + ix;
                if (!((src[idx >>> 5] >>> (idx & 31)) & 1)) zeroCount++;
            }
            for (let iz = 0; iz < nz; iz++) {
                const idx = iz * stride + iy * nx + ix;
                if (zeroCount === 0) dst[idx >>> 5] |= (1 << (idx & 31));
                const exitZ = iz - halfExtent;
                if (exitZ >= 0) {
                    const ei = exitZ * stride + iy * nx + ix;
                    if (!((src[ei >>> 5] >>> (ei & 31)) & 1)) zeroCount--;
                }
                const enterZ = iz + halfExtent + 1;
                if (enterZ < nz) {
                    const ni = enterZ * stride + iy * nx + ix;
                    if (!((src[ni >>> 5] >>> (ni & 31)) & 1)) zeroCount++;
                }
            }
        }
    }
};

/**
 * Convert a cropped region of a bitfield grid into a BlockAccumulator.
 * Block coordinates in the output start at (0,0,0).
 *
 * @param grid - Bitfield with 1 = solid.
 * @param nx - Full grid X dimension.
 * @param ny - Full grid Y dimension.
 * @param nz - Full grid Z dimension.
 * @param cropMinBx - Crop region start block X.
 * @param cropMinBy - Crop region start block Y.
 * @param cropMinBz - Crop region start block Z.
 * @param cropMaxBx - Crop region end block X (exclusive).
 * @param cropMaxBy - Crop region end block Y (exclusive).
 * @param cropMaxBz - Crop region end block Z (exclusive).
 * @returns New BlockAccumulator with blocks from the cropped region.
 */
const denseGridToAccumulator = (
    grid: Uint32Array,
    nx: number, ny: number, nz: number,
    cropMinBx: number, cropMinBy: number, cropMinBz: number,
    cropMaxBx: number, cropMaxBy: number, cropMaxBz: number
): BlockAccumulator => {
    const acc = new BlockAccumulator();
    const stride = nx * ny;

    for (let bz = cropMinBz; bz < cropMaxBz; bz++) {
        for (let by = cropMinBy; by < cropMaxBy; by++) {
            for (let bx = cropMinBx; bx < cropMaxBx; bx++) {
                let lo = 0;
                let hi = 0;
                const baseX = bx << 2;
                const baseY = by << 2;
                const baseZ = bz << 2;

                for (let lz = 0; lz < 4; lz++) {
                    for (let ly = 0; ly < 4; ly++) {
                        for (let lx = 0; lx < 4; lx++) {
                            const idx = (baseX + lx) + (baseY + ly) * nx + (baseZ + lz) * stride;
                            if ((grid[idx >>> 5] >>> (idx & 31)) & 1) {
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
                    acc.addBlock(
                        xyzToMorton(bx - cropMinBx, by - cropMinBy, bz - cropMinBz),
                        lo, hi
                    );
                }
            }
        }
    }

    return acc;
};

/**
 * Search outward from a blocked seed in expanding Chebyshev shells to find
 * the nearest free (non-blocked) voxel in the dilated clearance grid.
 *
 * @param blocked - Dilated bitfield (1 = blocked).
 * @param seedIx - Seed voxel X index.
 * @param seedIy - Seed voxel Y index.
 * @param seedIz - Seed voxel Z index.
 * @param nx - Grid X dimension.
 * @param ny - Grid Y dimension.
 * @param nz - Grid Z dimension.
 * @param stride - Row stride (nx * ny).
 * @param maxRadius - Maximum Chebyshev distance to search.
 * @returns Grid coordinates of the nearest free cell, or null if none found.
 */
const findNearestFreeCell = (
    blocked: Uint32Array,
    seedIx: number, seedIy: number, seedIz: number,
    nx: number, ny: number, nz: number, stride: number,
    maxRadius: number
): { ix: number; iy: number; iz: number } | null => {
    for (let r = 1; r <= maxRadius; r++) {
        for (let dz = -r; dz <= r; dz++) {
            for (let dy = -r; dy <= r; dy++) {
                for (let dx = -r; dx <= r; dx++) {
                    if (Math.abs(dx) !== r && Math.abs(dy) !== r && Math.abs(dz) !== r) continue;
                    const ix = seedIx + dx;
                    const iy = seedIy + dy;
                    const iz = seedIz + dz;
                    if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) continue;
                    const idx = ix + iy * nx + iz * stride;
                    if (!((blocked[idx >>> 5] >>> (idx & 31)) & 1)) {
                        return { ix, iy, iz };
                    }
                }
            }
        }
    }
    return null;
};

/**
 * Simplify voxel collision data for upright capsule navigation.
 *
 * Uses bitfield storage (1 bit per voxel) to reduce memory by 8x compared
 * to byte-per-voxel. Two Uint32Array buffers are ping-ponged through the
 * dilation, BFS, inversion, and erosion phases.
 *
 * Algorithm:
 * 1. Build dense bitfield grid from the accumulator.
 * 2. Dilate solid by the capsule shape (Minkowski sum) to get the clearance
 *    grid -- cells where the capsule center cannot be placed.
 * 3. BFS flood fill from the seed through free (non-blocked) cells to find
 *    all reachable capsule-center positions (uses a separate visited bitfield).
 * 4. Invert: every non-reachable cell becomes solid (negative space carving),
 *    computed as a single bitwise operation per word.
 * 5. Erode the solid by the capsule shape (Minkowski subtraction) to shrink
 *    surfaces back to their original positions, undoing the inflation from
 *    step 2 so the runtime capsule query produces correct collisions.
 * 6. Crop to bounding box of navigable cells.
 *
 * Grid boundaries are treated as solid, so the fill is always bounded even
 * in unsealed scenes.
 *
 * @param accumulator - BlockAccumulator with filtered voxelization results.
 * @param gridBounds - Grid bounds aligned to block boundaries (not mutated).
 * @param voxelResolution - Size of each voxel in world units.
 * @param capsuleHeight - Total capsule height in world units.
 * @param capsuleRadius - Capsule radius in world units.
 * @param seed - Seed position in world space (must be in a free region).
 * @returns Simplified accumulator and cropped grid bounds.
 */
const simplifyForCapsule = (
    accumulator: BlockAccumulator,
    gridBounds: Bounds,
    voxelResolution: number,
    capsuleHeight: number,
    capsuleRadius: number,
    seed: NavSeed
): NavSimplifyResult => {
    const nx = Math.round((gridBounds.max.x - gridBounds.min.x) / voxelResolution);
    const ny = Math.round((gridBounds.max.y - gridBounds.min.y) / voxelResolution);
    const nz = Math.round((gridBounds.max.z - gridBounds.min.z) / voxelResolution);
    const totalVoxels = nx * ny * nz;
    const stride = nx * ny;
    const wordCount = (totalVoxels + 31) >>> 5;

    // Capsule approximated as an axis-aligned box (square XZ cross-section).
    // Conservative: may reject narrow diagonal passages a true capsule could fit.
    const kernelR = Math.ceil(capsuleRadius / voxelResolution);
    const yHalfExtent = Math.ceil(capsuleHeight / (2 * voxelResolution));

    const memoryMB = Math.round(wordCount * 4 * 2 / (1024 * 1024));
    logger.debug(`nav simplify: grid ${nx}x${ny}x${nz} (${totalVoxels} voxels, ~${memoryMB} MB bitfield), clearance r=${kernelR}, y half=${yHalfExtent}`);

    // Phase 1: build dense bitfield grid from accumulator
    let t0 = performance.now();
    const bitA = new Uint32Array(wordCount);
    fillDenseSolidGrid(accumulator, bitA, nx, ny, nz);

    let solidCount = 0;
    for (let w = 0; w < wordCount; w++) {
        solidCount += popcount(bitA[w]);
    }
    logger.debug(`nav simplify: phase 1 (dense grid) ${(performance.now() - t0).toFixed(0)}ms, ${solidCount} solid voxels`);

    // Phase 2: capsule clearance grid (Minkowski dilation of solid by capsule)
    // Three separable 1D sliding window passes (X, Z, Y).
    t0 = performance.now();
    const bitB = new Uint32Array(wordCount);

    dilateX(bitA, bitB, nx, ny, nz, kernelR);
    bitA.fill(0);
    dilateZ(bitB, bitA, nx, ny, nz, kernelR);
    bitB.fill(0);
    dilateY(bitA, bitB, nx, ny, nz, yHalfExtent);
    logger.debug(`nav simplify: phase 2 (dilation) ${(performance.now() - t0).toFixed(0)}ms`);

    // Phase 3: BFS flood fill from seed through free (non-blocked) cells.
    // Uses bitB as blocked mask and bitA as visited mask.
    let seedIx = Math.floor((seed.x - gridBounds.min.x) / voxelResolution);
    let seedIy = Math.floor((seed.y - gridBounds.min.y) / voxelResolution);
    let seedIz = Math.floor((seed.z - gridBounds.min.z) / voxelResolution);

    if (seedIx < 0 || seedIx >= nx || seedIy < 0 || seedIy >= ny || seedIz < 0 || seedIz >= nz) {
        logger.warn(`nav simplify: seed (${seed.x}, ${seed.y}, ${seed.z}) outside grid, skipping`);
        return { accumulator, gridBounds };
    }

    let seedIdx = seedIx + seedIy * nx + seedIz * stride;
    if ((bitB[seedIdx >>> 5] >>> (seedIdx & 31)) & 1) {
        const maxRadius = Math.max(kernelR, yHalfExtent) * 2;
        const found = findNearestFreeCell(bitB, seedIx, seedIy, seedIz, nx, ny, nz, stride, maxRadius);
        if (!found) {
            logger.warn(`nav simplify: seed (${seed.x}, ${seed.y}, ${seed.z}) blocked after dilation, no free cell within ${maxRadius} voxels, skipping`);
            return { accumulator, gridBounds };
        }
        const dist = Math.max(Math.abs(found.ix - seedIx), Math.abs(found.iy - seedIy), Math.abs(found.iz - seedIz));
        const worldX = (gridBounds.min.x + (found.ix + 0.5) * voxelResolution).toFixed(2);
        const worldY = (gridBounds.min.y + (found.iy + 0.5) * voxelResolution).toFixed(2);
        const worldZ = (gridBounds.min.z + (found.iz + 0.5) * voxelResolution).toFixed(2);
        logger.log(`nav simplify: seed (${seed.x}, ${seed.y}, ${seed.z}) blocked after dilation, adjusted to (${worldX}, ${worldY}, ${worldZ}) (distance: ${dist} voxels)`);
        seedIx = found.ix;
        seedIy = found.iy;
        seedIz = found.iz;
        seedIdx = seedIx + seedIy * nx + seedIz * stride;
    }

    t0 = performance.now();
    bitA.fill(0); // reuse as visited bitfield

    const MAX_QUEUE = 1 << 25;
    const queueCap = Math.min(1 << (32 - Math.clz32(totalVoxels - 1)), MAX_QUEUE);
    const queueMask = queueCap - 1;
    const bfsQueue = new Uint32Array(queueCap);
    let qHead = 0;
    let qTail = 0;
    let queueSize = 0;
    let reachableCount = 0;
    let overflowed = false;

    const enqueue = (nIdx: number) => {
        const w = nIdx >>> 5;
        const m = 1 << (nIdx & 31);
        if (!((bitB[w] | bitA[w]) & m)) {
            if (queueSize >= queueCap) {
                if (!overflowed) {
                    logger.warn(`nav simplify: BFS queue overflow (cap ${queueCap}), results may be incomplete`);
                    overflowed = true;
                }
                return;
            }
            bitA[w] |= m;
            bfsQueue[qTail] = nIdx;
            qTail = (qTail + 1) & queueMask;
            queueSize++;
        }
    };

    bitA[seedIdx >>> 5] |= (1 << (seedIdx & 31));
    bfsQueue[qTail] = seedIdx;
    qTail = (qTail + 1) & queueMask;
    queueSize++;

    while (qHead !== qTail) {
        const idx = bfsQueue[qHead];
        qHead = (qHead + 1) & queueMask;
        queueSize--;
        reachableCount++;

        const ix = idx % nx;
        const iy = Math.floor((idx % stride) / nx);
        const iz = Math.floor(idx / stride);

        if (ix > 0) enqueue(idx - 1);
        if (ix < nx - 1) enqueue(idx + 1);
        if (iy > 0) enqueue(idx - nx);
        if (iy < ny - 1) enqueue(idx + nx);
        if (iz > 0) enqueue(idx - stride);
        if (iz < nz - 1) enqueue(idx + stride);
    }

    logger.debug(`nav simplify: phase 3 (flood fill) ${(performance.now() - t0).toFixed(0)}ms, ${reachableCount} reachable cells (${(reachableCount / totalVoxels * 100).toFixed(1)}%)`);

    // Phase 4: invert reachable to solid (bitwise operation).
    // Reachable = visited AND NOT blocked = bitA AND NOT bitB.
    // Solid = NOT reachable = NOT bitA OR bitB = ~bitA | bitB.
    t0 = performance.now();
    for (let w = 0; w < wordCount; w++) {
        bitB[w] |= ~bitA[w];
    }

    // Clear padding bits in the last word to avoid phantom solids
    const tailBits = totalVoxels & 31;
    if (tailBits) {
        bitB[wordCount - 1] &= (1 << tailBits) - 1;
    }

    let outputCount = 0;
    for (let w = 0; w < wordCount; w++) {
        outputCount += popcount(bitB[w]);
    }
    logger.debug(`nav simplify: phase 4 (invert) ${(performance.now() - t0).toFixed(0)}ms, ${outputCount} solid voxels`);

    // Phase 5: erode solid by capsule shape (Minkowski subtraction)
    t0 = performance.now();
    bitA.fill(0);
    erodeX(bitB, bitA, nx, ny, nz, kernelR);

    bitB.fill(0);
    erodeZ(bitA, bitB, nx, ny, nz, kernelR);

    bitA.fill(0);
    erodeY(bitB, bitA, nx, ny, nz, yHalfExtent);
    logger.debug(`nav simplify: phase 5 (erosion) ${(performance.now() - t0).toFixed(0)}ms`);

    // Phase 6: crop to bounding box of empty (navigable) cells
    t0 = performance.now();
    let minIx = nx, minIy = ny, minIz = nz;
    let maxIx = 0, maxIy = 0, maxIz = 0;

    for (let iz = 0; iz < nz; iz++) {
        const zOff = iz * stride;
        for (let iy = 0; iy < ny; iy++) {
            const rowOff = zOff + iy * nx;
            for (let ix = 0; ix < nx; ix++) {
                const idx = rowOff + ix;
                if (!((bitA[idx >>> 5] >>> (idx & 31)) & 1)) {
                    if (ix < minIx) minIx = ix;
                    if (ix > maxIx) maxIx = ix;
                    if (iy < minIy) minIy = iy;
                    if (iy > maxIy) maxIy = iy;
                    if (iz < minIz) minIz = iz;
                    if (iz > maxIz) maxIz = iz;
                }
            }
        }
    }

    const nbx = nx >> 2;
    const nby = ny >> 2;
    const nbz = nz >> 2;

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

    const croppedBlocks = (cropMaxBx - cropMinBx) * (cropMaxBy - cropMinBy) * (cropMaxBz - cropMinBz);
    const totalBlocks = nbx * nby * nbz;
    logger.log(`nav simplify: phase 6 (crop) ${(performance.now() - t0).toFixed(0)}ms, ${cropMaxBx - cropMinBx}x${cropMaxBy - cropMinBy}x${cropMaxBz - cropMinBz} blocks (${croppedBlocks} of ${totalBlocks})`);

    return {
        accumulator: denseGridToAccumulator(
            bitA, nx, ny, nz,
            cropMinBx, cropMinBy, cropMinBz,
            cropMaxBx, cropMaxBy, cropMaxBz
        ),
        gridBounds: croppedBounds
    };
};

export { simplifyForCapsule };
export type { NavSeed, NavSimplifyResult };
