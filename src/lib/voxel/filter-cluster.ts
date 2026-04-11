import { Vec3 } from 'playcanvas';

import { computeGaussianExtents } from './gaussian-aabb';
import { GaussianBVH } from './gaussian-bvh';
import { GpuVoxelization } from './gpu-voxelization';
import {
    BlockAccumulator,
    xyzToMorton,
    alignGridBounds
} from './sparse-octree';
import { filterAndFillBlocks } from './voxel-filter';
import { voxelizeToAccumulator } from './voxelize';
import { Column, DataTable } from '../data-table/data-table';
import { computeWriteTransform, transformColumns } from '../data-table/transform';
import type { DeviceCreator } from '../types';
import { logger } from '../utils/logger';
import { Transform } from '../utils/math';

/**
 * Find the connected component of occupied blocks reachable from a seed block
 * via 6-connected DFS.
 *
 * @param accumulator - Block accumulator containing voxelized blocks.
 * @param seedBx - Seed block X coordinate.
 * @param seedBy - Seed block Y coordinate.
 * @param seedBz - Seed block Z coordinate.
 * @returns Set of Morton codes for blocks in the connected component.
 */
const findClusterFromSeed = (
    accumulator: BlockAccumulator,
    seedBx: number,
    seedBy: number,
    seedBz: number
): Set<number> => {
    const occupied = new Set<number>();
    const solidMortons = accumulator.getSolidBlocks();
    for (let i = 0; i < solidMortons.length; i++) {
        occupied.add(solidMortons[i]);
    }
    const mixed = accumulator.getMixedBlocks();
    for (let i = 0; i < mixed.morton.length; i++) {
        occupied.add(mixed.morton[i]);
    }

    const seedMorton = xyzToMorton(seedBx, seedBy, seedBz);
    if (!occupied.has(seedMorton)) {
        return new Set<number>();
    }

    const ccSet = new Set<number>();
    const visited = new Set<number>();
    const stack: [number, number, number][] = [[seedBx, seedBy, seedBz]];
    visited.add(seedMorton);

    while (stack.length > 0) {
        const [bx, by, bz] = stack.pop()!;
        const morton = xyzToMorton(bx, by, bz);
        if (occupied.has(morton)) {
            ccSet.add(morton);
        }

        const neighbors: [number, number, number][] = [
            [bx - 1, by, bz], [bx + 1, by, bz],
            [bx, by - 1, bz], [bx, by + 1, bz],
            [bx, by, bz - 1], [bx, by, bz + 1]
        ];

        for (const [nx, ny, nz] of neighbors) {
            const nm = xyzToMorton(nx, ny, nz);
            if (!visited.has(nm) && occupied.has(nm)) {
                visited.add(nm);
                stack.push([nx, ny, nz]);
            }
        }
    }

    return ccSet;
};

/**
 * Find the nearest occupied block to a given position using expanding cube shells.
 *
 * @param accumulator - Block accumulator.
 * @param seedBx - Starting block X.
 * @param seedBy - Starting block Y.
 * @param seedBz - Starting block Z.
 * @param maxRadius - Maximum search radius in blocks.
 * @returns Coordinates of nearest occupied block, or null.
 */
const findNearestOccupiedBlock = (
    accumulator: BlockAccumulator,
    seedBx: number,
    seedBy: number,
    seedBz: number,
    maxRadius: number
): { bx: number; by: number; bz: number } | null => {
    const occupied = new Set<number>();
    const solidMortons = accumulator.getSolidBlocks();
    for (let i = 0; i < solidMortons.length; i++) {
        occupied.add(solidMortons[i]);
    }
    const mixed = accumulator.getMixedBlocks();
    for (let i = 0; i < mixed.morton.length; i++) {
        occupied.add(mixed.morton[i]);
    }

    if (occupied.has(xyzToMorton(seedBx, seedBy, seedBz))) {
        return { bx: seedBx, by: seedBy, bz: seedBz };
    }

    for (let r = 1; r <= maxRadius; r++) {
        for (let dz = -r; dz <= r; dz++) {
            for (let dy = -r; dy <= r; dy++) {
                for (let dx = -r; dx <= r; dx++) {
                    if (Math.abs(dx) !== r && Math.abs(dy) !== r && Math.abs(dz) !== r) continue;
                    const nx = seedBx + dx;
                    const ny = seedBy + dy;
                    const nz = seedBz + dz;
                    if (occupied.has(xyzToMorton(nx, ny, nz))) {
                        return { bx: nx, by: ny, bz: nz };
                    }
                }
            }
        }
    }

    return null;
};

/**
 * Filter a Gaussian splat DataTable to keep only Gaussians whose AABB overlaps
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

    seedBx = Math.max(0, Math.min(seedBx, numBlocksX - 1));
    seedBy = Math.max(0, Math.min(seedBy, numBlocksY - 1));
    seedBz = Math.max(0, Math.min(seedBz, numBlocksZ - 1));

    const maxSearchRadius = Math.max(numBlocksX, numBlocksY, numBlocksZ);
    const nearest = findNearestOccupiedBlock(accumulator, seedBx, seedBy, seedBz, maxSearchRadius);

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

    const ccSet = findClusterFromSeed(accumulator, nearest.bx, nearest.by, nearest.bz);
    logger.log(`filterCluster: cluster has ${ccSet.size} blocks out of ${accumulator.count} total`);

    if (ccSet.size === accumulator.count) {
        logger.log('filterCluster: all blocks in one cluster, no filtering needed');
        logger.progress.step();
        return dataTable;
    }

    logger.progress.step('Filtering Gaussians');

    const keepIndices: number[] = [];

    for (let i = 0; i < numRows; i++) {
        const px = posX[i];
        const py = posY[i];
        const pz = posZ[i];
        const ex = extentX[i];
        const ey = extentY[i];
        const ez = extentZ[i];

        const aabbMinBx = Math.max(0, Math.floor((px - ex - gridBounds.min.x) / blockSize));
        const aabbMaxBx = Math.min(numBlocksX - 1, Math.floor((px + ex - gridBounds.min.x) / blockSize));
        const aabbMinBy = Math.max(0, Math.floor((py - ey - gridBounds.min.y) / blockSize));
        const aabbMaxBy = Math.min(numBlocksY - 1, Math.floor((py + ey - gridBounds.min.y) / blockSize));
        const aabbMinBz = Math.max(0, Math.floor((pz - ez - gridBounds.min.z) / blockSize));
        const aabbMaxBz = Math.min(numBlocksZ - 1, Math.floor((pz + ez - gridBounds.min.z) / blockSize));

        let found = false;
        for (let bz = aabbMinBz; bz <= aabbMaxBz && !found; bz++) {
            for (let by = aabbMinBy; by <= aabbMaxBy && !found; by++) {
                for (let bx = aabbMinBx; bx <= aabbMaxBx && !found; bx++) {
                    if (ccSet.has(xyzToMorton(bx, by, bz))) {
                        found = true;
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
