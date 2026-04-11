import { computeGaussianExtents } from './gaussian-aabb';
import { GaussianBVH } from './gaussian-bvh';
import { GpuVoxelization } from './gpu-voxelization';
import {
    BlockAccumulator,
    mortonToXYZ,
    alignGridBounds
} from './sparse-octree';
import { voxelizeToAccumulator } from './voxelize';
import { Column, DataTable } from '../data-table/data-table';
import { computeWriteTransform, transformColumns } from '../data-table/transform';
import type { DeviceCreator } from '../types';
import { logger } from '../utils/logger';
import { sigmoid, Transform } from '../utils/math';

/**
 * Build block lookup structures from the accumulator's Morton codes.
 *
 * @param accumulator - Block accumulator containing voxelized blocks.
 * @param strideY - numBlocksX (stride for Y dimension).
 * @param strideZ - numBlocksX * numBlocksY (stride for Z dimension).
 * @returns Solid block set and mixed block map (linear index to masks array index).
 */
const buildBlockLookup = (
    accumulator: BlockAccumulator,
    strideY: number,
    strideZ: number
): { solidSet: Set<number>; mixedMap: Map<number, number>; masks: number[] } => {
    const solidSet = new Set<number>();
    const solidMortons = accumulator.getSolidBlocks();
    for (let i = 0; i < solidMortons.length; i++) {
        const [bx, by, bz] = mortonToXYZ(solidMortons[i]);
        solidSet.add(bx + by * strideY + bz * strideZ);
    }
    const mixed = accumulator.getMixedBlocks();
    const mixedMap = new Map<number, number>();
    for (let i = 0; i < mixed.morton.length; i++) {
        const [bx, by, bz] = mortonToXYZ(mixed.morton[i]);
        mixedMap.set(bx + by * strideY + bz * strideZ, i);
    }
    return { solidSet, mixedMap, masks: mixed.masks };
};

/**
 * Remove Gaussians that don't meaningfully contribute to any solid voxel.
 *
 * GPU-voxelizes the scene at a given resolution, then for each Gaussian evaluates
 * its opacity contribution at each occupied voxel center in its AABB range.
 * Discards Gaussians whose contribution is below `minContribution` at every
 * solid voxel.
 *
 * @param dataTable - Input Gaussian splat data.
 * @param createDevice - Function to create a GPU device for voxelization.
 * @param voxelSize - Voxel size in world units. Default: 0.05.
 * @param opacityCutoff - Opacity threshold for solid voxels. Default: 0.1.
 * @param minContribution - Minimum Gaussian contribution at a voxel center to be kept. Default: 1/255.
 * @returns Filtered DataTable with floaters removed.
 */
const filterFloaters = async (
    dataTable: DataTable,
    createDevice: DeviceCreator,
    voxelSize: number = 0.05,
    opacityCutoff: number = 0.1,
    minContribution: number = 1 / 255
): Promise<DataTable> => {
    const numRows = dataTable.numRows;
    if (numRows === 0) return dataTable;

    logger.progress.begin(4);

    logger.progress.step('Computing extents');

    const voxelColumns = [
        'x', 'y', 'z',
        'rot_0', 'rot_1', 'rot_2', 'rot_3',
        'scale_0', 'scale_1', 'scale_2',
        'opacity'
    ];
    const delta = computeWriteTransform(dataTable.transform, Transform.IDENTITY);
    const cols = transformColumns(dataTable, voxelColumns, delta);
    const pcDataTable = new DataTable(voxelColumns.map(name => new Column(name, cols.get(name)!)));

    const extentsResult = computeGaussianExtents(pcDataTable);
    const sceneBounds = extentsResult.sceneBounds;

    const extentX = extentsResult.extents.getColumnByName('extent_x').data;
    const extentY = extentsResult.extents.getColumnByName('extent_y').data;
    const extentZ = extentsResult.extents.getColumnByName('extent_z').data;
    const posX = pcDataTable.getColumnByName('x').data;
    const posY = pcDataTable.getColumnByName('y').data;
    const posZ = pcDataTable.getColumnByName('z').data;
    const rotW = pcDataTable.getColumnByName('rot_0').data;
    const rotX = pcDataTable.getColumnByName('rot_1').data;
    const rotY = pcDataTable.getColumnByName('rot_2').data;
    const rotZ = pcDataTable.getColumnByName('rot_3').data;
    const scaleX = pcDataTable.getColumnByName('scale_0').data;
    const scaleY = pcDataTable.getColumnByName('scale_1').data;
    const scaleZ = pcDataTable.getColumnByName('scale_2').data;
    const opacityData = pcDataTable.getColumnByName('opacity').data;

    const blockSize = 4 * voxelSize;

    logger.log(`filterFloaters: voxel size ${voxelSize}m, block size ${blockSize}m, minContribution ${minContribution.toFixed(6)}`);

    logger.progress.step('Building BVH');

    const bvh = new GaussianBVH(pcDataTable, extentsResult.extents);
    const device = await createDevice();

    const gpuVoxelization = new GpuVoxelization(device);
    gpuVoxelization.uploadAllGaussians(pcDataTable, extentsResult.extents);

    const gridBounds = alignGridBounds(
        sceneBounds.min.x, sceneBounds.min.y, sceneBounds.min.z,
        sceneBounds.max.x, sceneBounds.max.y, sceneBounds.max.z,
        voxelSize
    );

    logger.progress.step('Voxelizing');

    const accumulator = await voxelizeToAccumulator(
        bvh, gpuVoxelization, gridBounds, voxelSize, opacityCutoff
    );

    gpuVoxelization.destroy();

    const numBlocksX = Math.round((gridBounds.max.x - gridBounds.min.x) / blockSize);
    const numBlocksY = Math.round((gridBounds.max.y - gridBounds.min.y) / blockSize);
    const numBlocksZ = Math.round((gridBounds.max.z - gridBounds.min.z) / blockSize);
    const strideY = numBlocksX;
    const strideZ = numBlocksX * numBlocksY;

    const { solidSet, mixedMap, masks } = buildBlockLookup(accumulator, strideY, strideZ);

    logger.log(`filterFloaters: ${solidSet.size + mixedMap.size} occupied blocks (${solidSet.size} solid, ${mixedMap.size} mixed)`);

    logger.progress.step('Filtering Gaussians');

    const gridMinX = gridBounds.min.x;
    const gridMinY = gridBounds.min.y;
    const gridMinZ = gridBounds.min.z;

    const keepIndices: number[] = [];

    for (let i = 0; i < numRows; i++) {
        const px = posX[i];
        const py = posY[i];
        const pz = posZ[i];
        const ex = extentX[i];
        const ey = extentY[i];
        const ez = extentZ[i];

        const aabbMinBx = Math.max(0, Math.floor((px - ex - gridMinX) / blockSize));
        const aabbMaxBx = Math.min(numBlocksX - 1, Math.floor((px + ex - gridMinX) / blockSize));
        const aabbMinBy = Math.max(0, Math.floor((py - ey - gridMinY) / blockSize));
        const aabbMaxBy = Math.min(numBlocksY - 1, Math.floor((py + ey - gridMinY) / blockSize));
        const aabbMinBz = Math.max(0, Math.floor((pz - ez - gridMinZ) / blockSize));
        const aabbMaxBz = Math.min(numBlocksZ - 1, Math.floor((pz + ez - gridMinZ) / blockSize));

        // Fast path: if the Gaussian's center is inside an occupied voxel, it
        // clearly contributes to solid geometry — keep it without evaluation.
        const centerBx = Math.floor((px - gridMinX) / blockSize);
        const centerBy = Math.floor((py - gridMinY) / blockSize);
        const centerBz = Math.floor((pz - gridMinZ) / blockSize);

        if (centerBx >= 0 && centerBx < numBlocksX &&
            centerBy >= 0 && centerBy < numBlocksY &&
            centerBz >= 0 && centerBz < numBlocksZ) {
            const centerBlockIdx = centerBx + centerBy * strideY + centerBz * strideZ;
            if (solidSet.has(centerBlockIdx)) {
                keepIndices.push(i);
                continue;
            }
            const centerMixedIdx = mixedMap.get(centerBlockIdx);
            if (centerMixedIdx !== undefined) {
                const lx = Math.floor((px - gridMinX - centerBx * blockSize) / voxelSize);
                const ly = Math.floor((py - gridMinY - centerBy * blockSize) / voxelSize);
                const lz = Math.floor((pz - gridMinZ - centerBz * blockSize) / voxelSize);
                const bitIdx = (lx & 3) + (ly & 3) * 4 + (lz & 3) * 16;
                const word = bitIdx < 32 ? masks[centerMixedIdx * 2] : masks[centerMixedIdx * 2 + 1];
                if ((word >>> (bitIdx & 31)) & 1) {
                    keepIndices.push(i);
                    continue;
                }
            }
        }

        // Slow path: Gaussian center is not in an occupied voxel.
        // Check if it has meaningful contribution at any occupied voxel center.

        // Normalize quaternion — Rodrigues formula requires unit quaternions;
        // non-normalized ones (common in MCMC training) produce wrong distances.
        const rw = rotW[i], rx = rotX[i], ry = rotY[i], rz = rotZ[i];
        const qlen = Math.sqrt(rw * rw + rx * rx + ry * ry + rz * rz);
        const invLen = qlen > 0 ? 1 / qlen : 0;

        // For inverse rotation: negate xyz components of the normalized quaternion
        const qw = rw * invLen;
        const qx = -rx * invLen;
        const qy = -ry * invLen;
        const qz = -rz * invLen;

        // Inverse scale: exp(-log_scale)
        const isx = Math.exp(-scaleX[i]);
        const isy = Math.exp(-scaleY[i]);
        const isz = Math.exp(-scaleZ[i]);

        const alpha = sigmoid(opacityData[i]);

        let found = false;
        for (let bbz = aabbMinBz; bbz <= aabbMaxBz && !found; bbz++) {
            const zOff = bbz * strideZ;
            for (let bby = aabbMinBy; bby <= aabbMaxBy && !found; bby++) {
                const yzOff = bby * strideY + zOff;
                for (let bbx = aabbMinBx; bbx <= aabbMaxBx && !found; bbx++) {
                    const blockIdx = bbx + yzOff;
                    const isSolid = solidSet.has(blockIdx);
                    const mixedIdx = isSolid ? -1 : mixedMap.get(blockIdx);
                    if (!isSolid && mixedIdx === undefined) continue;

                    const blockOriginX = gridMinX + bbx * blockSize;
                    const blockOriginY = gridMinY + bby * blockSize;
                    const blockOriginZ = gridMinZ + bbz * blockSize;

                    const lo = isSolid ? 0xFFFFFFFF : masks[mixedIdx * 2];
                    const hi = isSolid ? 0xFFFFFFFF : masks[mixedIdx * 2 + 1];

                    for (let lz = 0; lz < 4 && !found; lz++) {
                        const vz = blockOriginZ + (lz + 0.5) * voxelSize;
                        const word = lz < 2 ? lo : hi;
                        const zBitBase = (lz & 1) * 16;

                        for (let ly = 0; ly < 4 && !found; ly++) {
                            const bitBase = zBitBase + ly * 4;
                            const vy = blockOriginY + (ly + 0.5) * voxelSize;

                            for (let lx = 0; lx < 4 && !found; lx++) {
                                if (!((word >>> (bitBase + lx)) & 1)) continue;

                                const vx = blockOriginX + (lx + 0.5) * voxelSize;

                                // Displacement from Gaussian center to voxel center
                                const dx = vx - px;
                                const dy = vy - py;
                                const dz = vz - pz;

                                // Inverse rotation via Rodrigues cross-product formula
                                const tx = 2 * (qy * dz - qz * dy);
                                const ty = 2 * (qz * dx - qx * dz);
                                const tz = 2 * (qx * dy - qy * dx);

                                const ldx = dx + qw * tx + (qy * tz - qz * ty);
                                const ldy = dy + qw * ty + (qz * tx - qx * tz);
                                const ldz = dz + qw * tz + (qx * ty - qy * tx);

                                // Mahalanobis distance squared
                                const sdx = ldx * isx;
                                const sdy = ldy * isy;
                                const sdz = ldz * isz;
                                const d2 = sdx * sdx + sdy * sdy + sdz * sdz;

                                if (alpha * Math.exp(-0.5 * d2) >= minContribution) {
                                    found = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        if (found) {
            keepIndices.push(i);
        }
    }

    const removed = numRows - keepIndices.length;
    logger.log(`filterFloaters: keeping ${keepIndices.length} of ${numRows} Gaussians (removed ${removed})`);

    if (removed === 0) return dataTable;

    return dataTable.clone({ rows: keepIndices });
};

export { filterFloaters };
