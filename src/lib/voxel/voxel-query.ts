import { ABSENT, BlockIndexMap, SOLID } from './block-index-map';
import { BlockMaskBuffer } from './block-mask-buffer';
import {
    computeGaussianInverse,
    evaluateGaussianAt,
    type GaussianColumns
} from '../data-table';

/**
 * Pre-computed lookup structures for efficient voxel block queries.
 */
interface BlockLookup {
    /** Block index -> state: `SOLID`, a mixed-array slot (`>= 0`), or `ABSENT`. */
    blocks: BlockIndexMap;
    masks: Uint32Array;
    solidCount: number;
    mixedCount: number;
}

/**
 * Grid parameters for block-based voxel queries.
 */
interface BlockGridParams {
    gridMinX: number;
    gridMinY: number;
    gridMinZ: number;
    blockSize: number;
    voxelResolution: number;
    numBlocksX: number;
    numBlocksY: number;
    numBlocksZ: number;
    strideY: number;
    strideZ: number;
}

/**
 * Build block lookup structures from the buffer's linear block indices.
 * The buffer's keys are already linear block indices, so this is a direct
 * copy into a typed-array hash for O(1) random access (no V8 Set/Map cap).
 *
 * @param buffer - Block mask buffer containing voxelized blocks.
 * @returns Block-state lookup, masks array, and solid/mixed counts.
 */
const buildBlockLookup = (
    buffer: BlockMaskBuffer
): BlockLookup => {
    const solidIdx = buffer.getSolidBlocks();
    const mixed = buffer.getMixedBlocks();
    const blocks = new BlockIndexMap(solidIdx.length + mixed.blockIdx.length);
    for (let i = 0; i < solidIdx.length; i++) {
        blocks.set(solidIdx[i], SOLID);
    }
    for (let i = 0; i < mixed.blockIdx.length; i++) {
        blocks.set(mixed.blockIdx[i], i);
    }
    return { blocks, masks: mixed.masks, solidCount: solidIdx.length, mixedCount: mixed.blockIdx.length };
};

/**
 * Test whether a Gaussian's center lies inside an occupied voxel.
 *
 * @param px - Gaussian center x.
 * @param py - Gaussian center y.
 * @param pz - Gaussian center z.
 * @param grid - Block grid parameters.
 * @param lookup - Block lookup structures.
 * @param blockFilter - Optional set of block indices to restrict the test to.
 * @returns True if the center is in an occupied (and optionally filtered) voxel.
 */
const isCenterInOccupiedVoxel = (
    px: number, py: number, pz: number,
    grid: BlockGridParams,
    lookup: BlockLookup,
    blockFilter?: Set<number>
): boolean => {
    const centerBx = Math.floor((px - grid.gridMinX) / grid.blockSize);
    const centerBy = Math.floor((py - grid.gridMinY) / grid.blockSize);
    const centerBz = Math.floor((pz - grid.gridMinZ) / grid.blockSize);

    if (centerBx < 0 || centerBx >= grid.numBlocksX ||
        centerBy < 0 || centerBy >= grid.numBlocksY ||
        centerBz < 0 || centerBz >= grid.numBlocksZ) {
        return false;
    }

    const centerBlockIdx = centerBx + centerBy * grid.strideY + centerBz * grid.strideZ;
    if (blockFilter && !blockFilter.has(centerBlockIdx)) {
        return false;
    }

    const centerState = lookup.blocks.get(centerBlockIdx);
    if (centerState === SOLID) {
        return true;
    }

    if (centerState !== ABSENT) {
        const centerMixedIdx = centerState;
        const lx = Math.floor((px - grid.gridMinX - centerBx * grid.blockSize) / grid.voxelResolution);
        const ly = Math.floor((py - grid.gridMinY - centerBy * grid.blockSize) / grid.voxelResolution);
        const lz = Math.floor((pz - grid.gridMinZ - centerBz * grid.blockSize) / grid.voxelResolution);
        const bitIdx = (lx & 3) + (ly & 3) * 4 + (lz & 3) * 16;
        const word = bitIdx < 32 ? lookup.masks[centerMixedIdx * 2] : lookup.masks[centerMixedIdx * 2 + 1];
        if ((word >>> (bitIdx & 31)) & 1) {
            return true;
        }
    }

    return false;
};

/**
 * Test whether a Gaussian has meaningful contribution at occupied voxel
 * centers in blocks that overlap its AABB.
 *
 * Iterates over blocks that overlap the Gaussian's AABB, then evaluates the
 * Gaussian's opacity contribution at each occupied voxel center in those
 * blocks. Returns true once `minHits` qualifying voxels are found. With the
 * default `minHits = 1` this short-circuits on the first hit; larger values
 * let callers reject elongated outliers (e.g. spikes) whose tails clip only
 * a single cluster voxel.
 *
 * @param gaussianIdx - Index of the Gaussian.
 * @param columns - Gaussian column data arrays.
 * @param grid - Block grid parameters.
 * @param lookup - Block lookup structures.
 * @param minContribution - Minimum contribution threshold.
 * @param blockFilter - Optional set of block indices to restrict the test to.
 * @param minHits - Minimum number of qualifying voxels required. Default 1.
 * @returns True if at least `minHits` qualifying voxels were found.
 */
const gaussianContributesToVoxels = (
    gaussianIdx: number,
    columns: GaussianColumns,
    grid: BlockGridParams,
    lookup: BlockLookup,
    minContribution: number,
    blockFilter?: Set<number>,
    minHits: number = 1
): boolean => {
    const px = columns.posX[gaussianIdx];
    const py = columns.posY[gaussianIdx];
    const pz = columns.posZ[gaussianIdx];
    const ex = columns.extentX[gaussianIdx];
    const ey = columns.extentY[gaussianIdx];
    const ez = columns.extentZ[gaussianIdx];

    const aabbMinBx = Math.max(0, Math.floor((px - ex - grid.gridMinX) / grid.blockSize));
    const aabbMaxBx = Math.min(grid.numBlocksX - 1, Math.floor((px + ex - grid.gridMinX) / grid.blockSize));
    const aabbMinBy = Math.max(0, Math.floor((py - ey - grid.gridMinY) / grid.blockSize));
    const aabbMaxBy = Math.min(grid.numBlocksY - 1, Math.floor((py + ey - grid.gridMinY) / grid.blockSize));
    const aabbMinBz = Math.max(0, Math.floor((pz - ez - grid.gridMinZ) / grid.blockSize));
    const aabbMaxBz = Math.min(grid.numBlocksZ - 1, Math.floor((pz + ez - grid.gridMinZ) / grid.blockSize));

    const g = computeGaussianInverse(
        columns.rotW[gaussianIdx], columns.rotX[gaussianIdx],
        columns.rotY[gaussianIdx], columns.rotZ[gaussianIdx],
        columns.scaleX[gaussianIdx], columns.scaleY[gaussianIdx],
        columns.scaleZ[gaussianIdx], columns.opacity[gaussianIdx]
    );

    let hits = 0;
    for (let bbz = aabbMinBz; bbz <= aabbMaxBz; bbz++) {
        const zOff = bbz * grid.strideZ;
        for (let bby = aabbMinBy; bby <= aabbMaxBy; bby++) {
            const yzOff = bby * grid.strideY + zOff;
            for (let bbx = aabbMinBx; bbx <= aabbMaxBx; bbx++) {
                const blockIdx = bbx + yzOff;
                if (blockFilter && !blockFilter.has(blockIdx)) continue;
                const state = lookup.blocks.get(blockIdx);
                if (state === ABSENT) continue;
                const isSolid = state === SOLID;

                const blockOriginX = grid.gridMinX + bbx * grid.blockSize;
                const blockOriginY = grid.gridMinY + bby * grid.blockSize;
                const blockOriginZ = grid.gridMinZ + bbz * grid.blockSize;

                const lo = isSolid ? 0xFFFFFFFF : lookup.masks[state * 2];
                const hi = isSolid ? 0xFFFFFFFF : lookup.masks[state * 2 + 1];

                for (let lz = 0; lz < 4; lz++) {
                    const vz = blockOriginZ + (lz + 0.5) * grid.voxelResolution;
                    const word = lz < 2 ? lo : hi;
                    const zBitBase = (lz & 1) * 16;

                    for (let ly = 0; ly < 4; ly++) {
                        const bitBase = zBitBase + ly * 4;
                        const vy = blockOriginY + (ly + 0.5) * grid.voxelResolution;

                        for (let lx = 0; lx < 4; lx++) {
                            if (!((word >>> (bitBase + lx)) & 1)) continue;

                            const vx = blockOriginX + (lx + 0.5) * grid.voxelResolution;

                            if (evaluateGaussianAt(g, px, py, pz, vx, vy, vz) >= minContribution) {
                                if (++hits >= minHits) return true;
                            }
                        }
                    }
                }
            }
        }
    }

    return false;
};

export {
    buildBlockLookup,
    isCenterInOccupiedVoxel,
    gaussianContributesToVoxels,
    type BlockLookup,
    type BlockGridParams
};
