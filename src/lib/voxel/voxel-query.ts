import { BlockMaskBuffer } from './block-mask-buffer';
import { mortonToXYZ } from './morton';
import {
    computeGaussianInverse,
    evaluateGaussianAt,
    type GaussianColumns
} from '../data-table';

/**
 * Pre-computed lookup structures for efficient voxel block queries.
 */
interface BlockLookup {
    solidSet: Set<number>;
    mixedMap: Map<number, number>;
    masks: Uint32Array;
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
 * Build block lookup structures from the buffer's Morton codes.
 *
 * @param buffer - Block mask buffer containing voxelized blocks.
 * @param strideY - numBlocksX (stride for Y dimension).
 * @param strideZ - numBlocksX * numBlocksY (stride for Z dimension).
 * @returns Solid block set, mixed block map (linear index to masks array index), and masks.
 */
const buildBlockLookup = (
    buffer: BlockMaskBuffer,
    strideY: number,
    strideZ: number
): BlockLookup => {
    const solidSet = new Set<number>();
    const solidMortons = buffer.getSolidBlocks();
    for (let i = 0; i < solidMortons.length; i++) {
        const [bx, by, bz] = mortonToXYZ(solidMortons[i]);
        solidSet.add(bx + by * strideY + bz * strideZ);
    }
    const mixed = buffer.getMixedBlocks();
    const mixedMap = new Map<number, number>();
    for (let i = 0; i < mixed.morton.length; i++) {
        const [bx, by, bz] = mortonToXYZ(mixed.morton[i]);
        mixedMap.set(bx + by * strideY + bz * strideZ, i);
    }
    return { solidSet, mixedMap, masks: mixed.masks };
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

    if (lookup.solidSet.has(centerBlockIdx)) {
        return true;
    }

    const centerMixedIdx = lookup.mixedMap.get(centerBlockIdx);
    if (centerMixedIdx !== undefined) {
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
 * Test whether a Gaussian has meaningful contribution at any occupied voxel
 * center within its AABB range.
 *
 * Iterates over blocks that overlap the Gaussian's AABB, then evaluates the
 * Gaussian's opacity contribution at each occupied voxel center in those blocks.
 *
 * @param gaussianIdx - Index of the Gaussian.
 * @param columns - Gaussian column data arrays.
 * @param grid - Block grid parameters.
 * @param lookup - Block lookup structures.
 * @param minContribution - Minimum contribution threshold.
 * @param blockFilter - Optional set of block indices to restrict the test to.
 * @returns True if the Gaussian contributes above threshold at any qualifying voxel.
 */
const gaussianContributesToVoxels = (
    gaussianIdx: number,
    columns: GaussianColumns,
    grid: BlockGridParams,
    lookup: BlockLookup,
    minContribution: number,
    blockFilter?: Set<number>
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

    for (let bbz = aabbMinBz; bbz <= aabbMaxBz; bbz++) {
        const zOff = bbz * grid.strideZ;
        for (let bby = aabbMinBy; bby <= aabbMaxBy; bby++) {
            const yzOff = bby * grid.strideY + zOff;
            for (let bbx = aabbMinBx; bbx <= aabbMaxBx; bbx++) {
                const blockIdx = bbx + yzOff;
                if (blockFilter && !blockFilter.has(blockIdx)) continue;
                const isSolid = lookup.solidSet.has(blockIdx);
                const mixedIdx = isSolid ? -1 : lookup.mixedMap.get(blockIdx);
                if (!isSolid && mixedIdx === undefined) continue;

                const blockOriginX = grid.gridMinX + bbx * grid.blockSize;
                const blockOriginY = grid.gridMinY + bby * grid.blockSize;
                const blockOriginZ = grid.gridMinZ + bbz * grid.blockSize;

                const lo = isSolid ? 0xFFFFFFFF : lookup.masks[mixedIdx * 2];
                const hi = isSolid ? 0xFFFFFFFF : lookup.masks[mixedIdx * 2 + 1];

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
                                return true;
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
