import { Vec3 } from 'playcanvas';

import { computeGaussianExtents } from './gaussian-aabb';
import { GaussianBVH } from './gaussian-bvh';
import { GpuVoxelization } from './gpu-voxelization';
import {
    BlockAccumulator,
    mortonToXYZ,
    alignGridBounds
} from './sparse-octree';
import { filterAndFillBlocks } from './voxel-filter';
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
 * @returns Solid block set, mixed block map (linear index to masks array index), and masks.
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
 * Find the connected component of occupied blocks reachable from a seed block
 * via 6-connected DFS.
 *
 * @param occupied - Set of linear indices for occupied blocks.
 * @param seedBx - Seed block X coordinate.
 * @param seedBy - Seed block Y coordinate.
 * @param seedBz - Seed block Z coordinate.
 * @param numBlocksX - Grid dimension X.
 * @param numBlocksY - Grid dimension Y.
 * @param numBlocksZ - Grid dimension Z.
 * @returns Set of linear indices for blocks in the connected component.
 */
const findClusterFromSeed = (
    occupied: Set<number>,
    seedBx: number,
    seedBy: number,
    seedBz: number,
    numBlocksX: number,
    numBlocksY: number,
    numBlocksZ: number
): Set<number> => {
    const strideY = numBlocksX;
    const strideZ = numBlocksX * numBlocksY;

    const seedIdx = seedBx + seedBy * strideY + seedBz * strideZ;
    if (!occupied.has(seedIdx)) {
        return new Set<number>();
    }

    const ccSet = new Set<number>();
    const visited = new Set<number>();
    const stack: [number, number, number][] = [[seedBx, seedBy, seedBz]];
    visited.add(seedIdx);

    while (stack.length > 0) {
        const [bx, by, bz] = stack.pop()!;
        const idx = bx + by * strideY + bz * strideZ;
        if (occupied.has(idx)) {
            ccSet.add(idx);
        }

        if (bx > 0) {
            const ni = idx - 1;
            if (!visited.has(ni) && occupied.has(ni)) {
                visited.add(ni);
                stack.push([bx - 1, by, bz]);
            }
        }
        if (bx < numBlocksX - 1) {
            const ni = idx + 1;
            if (!visited.has(ni) && occupied.has(ni)) {
                visited.add(ni);
                stack.push([bx + 1, by, bz]);
            }
        }
        if (by > 0) {
            const ni = idx - strideY;
            if (!visited.has(ni) && occupied.has(ni)) {
                visited.add(ni);
                stack.push([bx, by - 1, bz]);
            }
        }
        if (by < numBlocksY - 1) {
            const ni = idx + strideY;
            if (!visited.has(ni) && occupied.has(ni)) {
                visited.add(ni);
                stack.push([bx, by + 1, bz]);
            }
        }
        if (bz > 0) {
            const ni = idx - strideZ;
            if (!visited.has(ni) && occupied.has(ni)) {
                visited.add(ni);
                stack.push([bx, by, bz - 1]);
            }
        }
        if (bz < numBlocksZ - 1) {
            const ni = idx + strideZ;
            if (!visited.has(ni) && occupied.has(ni)) {
                visited.add(ni);
                stack.push([bx, by, bz + 1]);
            }
        }
    }

    return ccSet;
};

/**
 * Find the nearest occupied block to a given position using expanding cube shells.
 *
 * @param occupied - Set of linear indices for occupied blocks.
 * @param seedBx - Starting block X.
 * @param seedBy - Starting block Y.
 * @param seedBz - Starting block Z.
 * @param maxRadius - Maximum search radius in blocks.
 * @param numBlocksX - Grid dimension X.
 * @param numBlocksY - Grid dimension Y.
 * @param numBlocksZ - Grid dimension Z.
 * @returns Coordinates of nearest occupied block, or null.
 */
const findNearestOccupiedBlock = (
    occupied: Set<number>,
    seedBx: number,
    seedBy: number,
    seedBz: number,
    maxRadius: number,
    numBlocksX: number,
    numBlocksY: number,
    numBlocksZ: number
): { bx: number; by: number; bz: number } | null => {
    const strideY = numBlocksX;
    const strideZ = numBlocksX * numBlocksY;

    if (occupied.has(seedBx + seedBy * strideY + seedBz * strideZ)) {
        return { bx: seedBx, by: seedBy, bz: seedBz };
    }

    for (let r = 1; r <= maxRadius; r++) {
        for (let dz = -r; dz <= r; dz++) {
            const nz = seedBz + dz;
            if (nz < 0 || nz >= numBlocksZ) continue;
            for (let dy = -r; dy <= r; dy++) {
                const ny = seedBy + dy;
                if (ny < 0 || ny >= numBlocksY) continue;
                for (let dx = -r; dx <= r; dx++) {
                    if (Math.abs(dx) !== r && Math.abs(dy) !== r && Math.abs(dz) !== r) continue;
                    const nx = seedBx + dx;
                    if (nx < 0 || nx >= numBlocksX) continue;
                    if (occupied.has(nx + ny * strideY + nz * strideZ)) {
                        return { bx: nx, by: ny, bz: nz };
                    }
                }
            }
        }
    }

    return null;
};

/**
 * Filter a Gaussian splat DataTable to keep only Gaussians that contribute to
 * the connected component found by GPU voxelization from a seed position.
 *
 * @param dataTable - Input Gaussian splat data.
 * @param createDevice - Function to create a GPU device for voxelization.
 * @param maxDimension - Max voxels per axis for coarse voxelization. Default: 1024.
 * @param seed - Seed position in world space. Default: (0,0,0).
 * @param opacityCutoff - Opacity threshold for solid voxels. Default: 0.1.
 * @returns Filtered DataTable containing only Gaussians in the seed's cluster.
 */
const filterCluster = async (
    dataTable: DataTable,
    createDevice: DeviceCreator,
    maxDimension: number = 1024,
    seed: Vec3 = Vec3.ZERO,
    opacityCutoff: number = 0.1
): Promise<DataTable> => {
    const numRows = dataTable.numRows;
    if (numRows === 0) return dataTable;

    logger.progress.begin(5);

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

    const sceneExtentX = sceneBounds.max.x - sceneBounds.min.x;
    const sceneExtentY = sceneBounds.max.y - sceneBounds.min.y;
    const sceneExtentZ = sceneBounds.max.z - sceneBounds.min.z;
    const maxSceneExtent = Math.max(sceneExtentX, sceneExtentY, sceneExtentZ);

    const coarseVoxelSize = Math.max(0.01, maxSceneExtent / maxDimension);
    const blockSize = 4 * coarseVoxelSize;

    logger.log(`filterCluster: scene extent ${maxSceneExtent.toFixed(2)}m, coarse voxel size ${coarseVoxelSize.toFixed(4)}m`);

    logger.progress.step('Building BVH');

    const bvh = new GaussianBVH(pcDataTable, extentsResult.extents);
    const device = await createDevice();

    const gpuVoxelization = new GpuVoxelization(device);
    gpuVoxelization.uploadAllGaussians(pcDataTable, extentsResult.extents);

    const gridBounds = alignGridBounds(
        sceneBounds.min.x, sceneBounds.min.y, sceneBounds.min.z,
        sceneBounds.max.x, sceneBounds.max.y, sceneBounds.max.z,
        coarseVoxelSize
    );

    logger.progress.step('Voxelizing');

    let accumulator = await voxelizeToAccumulator(
        bvh, gpuVoxelization, gridBounds, coarseVoxelSize, opacityCutoff
    );

    gpuVoxelization.destroy();

    accumulator = filterAndFillBlocks(accumulator);

    logger.progress.step('Finding cluster');

    let seedBx = Math.floor((seed.x - gridBounds.min.x) / blockSize);
    let seedBy = Math.floor((seed.y - gridBounds.min.y) / blockSize);
    let seedBz = Math.floor((seed.z - gridBounds.min.z) / blockSize);

    const numBlocksX = Math.round((gridBounds.max.x - gridBounds.min.x) / blockSize);
    const numBlocksY = Math.round((gridBounds.max.y - gridBounds.min.y) / blockSize);
    const numBlocksZ = Math.round((gridBounds.max.z - gridBounds.min.z) / blockSize);
    const strideY = numBlocksX;
    const strideZ = numBlocksX * numBlocksY;

    seedBx = Math.max(0, Math.min(seedBx, numBlocksX - 1));
    seedBy = Math.max(0, Math.min(seedBy, numBlocksY - 1));
    seedBz = Math.max(0, Math.min(seedBz, numBlocksZ - 1));

    const { solidSet, mixedMap, masks } = buildBlockLookup(accumulator, strideY, strideZ);
    const occupied = new Set<number>([...solidSet, ...mixedMap.keys()]);

    const maxSearchRadius = Math.max(numBlocksX, numBlocksY, numBlocksZ);
    const nearest = findNearestOccupiedBlock(occupied, seedBx, seedBy, seedBz, maxSearchRadius, numBlocksX, numBlocksY, numBlocksZ);

    if (!nearest) {
        logger.warn('filterCluster: no occupied blocks found, returning empty result');
        return dataTable.clone({ rows: [] });
    }

    if (nearest.bx !== seedBx || nearest.by !== seedBy || nearest.bz !== seedBz) {
        const worldX = gridBounds.min.x + (nearest.bx + 0.5) * blockSize;
        const worldY = gridBounds.min.y + (nearest.by + 0.5) * blockSize;
        const worldZ = gridBounds.min.z + (nearest.bz + 0.5) * blockSize;
        logger.log(`filterCluster: seed block unoccupied, using nearest at (${worldX.toFixed(2)}, ${worldY.toFixed(2)}, ${worldZ.toFixed(2)})`);
    }

    const ccSet = findClusterFromSeed(occupied, nearest.bx, nearest.by, nearest.bz, numBlocksX, numBlocksY, numBlocksZ);
    logger.log(`filterCluster: cluster has ${ccSet.size} blocks out of ${accumulator.count} total`);

    if (ccSet.size === accumulator.count) {
        logger.log('filterCluster: all blocks in one cluster, no filtering needed');
        logger.progress.step();
        return dataTable;
    }

    logger.progress.step('Filtering Gaussians');

    const gridMinX = gridBounds.min.x;
    const gridMinY = gridBounds.min.y;
    const gridMinZ = gridBounds.min.z;
    const minContribution = 1 / 255;

    const keepIndices: number[] = [];

    for (let i = 0; i < numRows; i++) {
        const px = posX[i];
        const py = posY[i];
        const pz = posZ[i];
        const ex = extentX[i];
        const ey = extentY[i];
        const ez = extentZ[i];

        // Fast path: if center is in a cluster voxel, keep immediately
        const centerBx = Math.floor((px - gridMinX) / blockSize);
        const centerBy = Math.floor((py - gridMinY) / blockSize);
        const centerBz = Math.floor((pz - gridMinZ) / blockSize);

        if (centerBx >= 0 && centerBx < numBlocksX &&
            centerBy >= 0 && centerBy < numBlocksY &&
            centerBz >= 0 && centerBz < numBlocksZ) {
            const centerBlockIdx = centerBx + centerBy * strideY + centerBz * strideZ;
            if (ccSet.has(centerBlockIdx)) {
                if (solidSet.has(centerBlockIdx)) {
                    keepIndices.push(i);
                    continue;
                }
                const centerMixedIdx = mixedMap.get(centerBlockIdx);
                if (centerMixedIdx !== undefined) {
                    const lx = Math.floor((px - gridMinX - centerBx * blockSize) / coarseVoxelSize);
                    const ly = Math.floor((py - gridMinY - centerBy * blockSize) / coarseVoxelSize);
                    const lz = Math.floor((pz - gridMinZ - centerBz * blockSize) / coarseVoxelSize);
                    const bitIdx = (lx & 3) + (ly & 3) * 4 + (lz & 3) * 16;
                    const word = bitIdx < 32 ? masks[centerMixedIdx * 2] : masks[centerMixedIdx * 2 + 1];
                    if ((word >>> (bitIdx & 31)) & 1) {
                        keepIndices.push(i);
                        continue;
                    }
                }
            }
        }

        // Slow path: evaluate Gaussian contribution at cluster voxel centers
        const aabbMinBx = Math.max(0, Math.floor((px - ex - gridMinX) / blockSize));
        const aabbMaxBx = Math.min(numBlocksX - 1, Math.floor((px + ex - gridMinX) / blockSize));
        const aabbMinBy = Math.max(0, Math.floor((py - ey - gridMinY) / blockSize));
        const aabbMaxBy = Math.min(numBlocksY - 1, Math.floor((py + ey - gridMinY) / blockSize));
        const aabbMinBz = Math.max(0, Math.floor((pz - ez - gridMinZ) / blockSize));
        const aabbMaxBz = Math.min(numBlocksZ - 1, Math.floor((pz + ez - gridMinZ) / blockSize));

        const rw = rotW[i], rx = rotX[i], ry = rotY[i], rz = rotZ[i];
        const qlen = Math.sqrt(rw * rw + rx * rx + ry * ry + rz * rz);
        const invLen = qlen > 0 ? 1 / qlen : 0;

        const qw = rw * invLen;
        const qx = -rx * invLen;
        const qy = -ry * invLen;
        const qz = -rz * invLen;

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
                    if (!ccSet.has(blockIdx)) continue;
                    const isSolid = solidSet.has(blockIdx);
                    const mixedIdx = isSolid ? -1 : mixedMap.get(blockIdx);
                    if (!isSolid && mixedIdx === undefined) continue;

                    const blockOriginX = gridMinX + bbx * blockSize;
                    const blockOriginY = gridMinY + bby * blockSize;
                    const blockOriginZ = gridMinZ + bbz * blockSize;

                    const lo = isSolid ? 0xFFFFFFFF : masks[mixedIdx * 2];
                    const hi = isSolid ? 0xFFFFFFFF : masks[mixedIdx * 2 + 1];

                    for (let lz = 0; lz < 4 && !found; lz++) {
                        const vz = blockOriginZ + (lz + 0.5) * coarseVoxelSize;
                        const word = lz < 2 ? lo : hi;
                        const zBitBase = (lz & 1) * 16;

                        for (let ly = 0; ly < 4 && !found; ly++) {
                            const bitBase = zBitBase + ly * 4;
                            const vy = blockOriginY + (ly + 0.5) * coarseVoxelSize;

                            for (let lx = 0; lx < 4 && !found; lx++) {
                                if (!((word >>> (bitBase + lx)) & 1)) continue;

                                const vx = blockOriginX + (lx + 0.5) * coarseVoxelSize;

                                const dx = vx - px;
                                const dy = vy - py;
                                const dz = vz - pz;

                                const tx = 2 * (qy * dz - qz * dy);
                                const ty = 2 * (qz * dx - qx * dz);
                                const tz = 2 * (qx * dy - qy * dx);

                                const ldx = dx + qw * tx + (qy * tz - qz * ty);
                                const ldy = dy + qw * ty + (qz * tx - qx * tz);
                                const ldz = dz + qw * tz + (qx * ty - qy * tx);

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
    logger.log(`filterCluster: keeping ${keepIndices.length} of ${numRows} Gaussians (removed ${removed})`);

    if (removed === 0) return dataTable;

    return dataTable.clone({ rows: keepIndices });
};

export { filterCluster };
