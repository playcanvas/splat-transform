import { Vec3 } from 'playcanvas';

import {
    setupVoxelFilter,
    buildGaussianColumns,
    buildBlockGridParams,
    type VoxelFilterContext
} from './filter-pipeline';
import {
    buildBlockLookup,
    isCenterInOccupiedVoxel,
    gaussianContributesToVoxels
} from './voxel-query';
import { alignGridBounds, voxelizeToBuffer } from './voxelize';
import { DataTable } from '../data-table';
import type { DeviceCreator } from '../types';
import { logger } from '../utils';

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
 * @param voxelResolution - Voxel size in world units for coarse voxelization. Default: 1.0.
 * @param seed - Seed position in world space. Default: (0,0,0).
 * @param opacityCutoff - Opacity threshold for solid voxels. Default: 0.99.
 * @param minContribution - Minimum Gaussian contribution at a cluster voxel center to be kept. Default: 1/255.
 * @returns Filtered DataTable containing only Gaussians in the seed's cluster.
 */
const filterCluster = async (
    dataTable: DataTable,
    createDevice: DeviceCreator,
    voxelResolution: number = 1.0,
    seed: Vec3 = Vec3.ZERO,
    opacityCutoff: number = 0.99,
    minContribution: number = 1 / 255
): Promise<DataTable> => {
    const numRows = dataTable.numRows;
    if (numRows === 0) return dataTable;

    logger.progress.begin(5);

    let ctx: VoxelFilterContext | undefined;
    try {
        logger.progress.step('Computing extents');

        ctx = await setupVoxelFilter(dataTable, createDevice);

        const clampedResolution = Math.max(0.01, voxelResolution);
        const blockSize = 4 * clampedResolution;
        const maxGridExtent = 8192 * clampedResolution;

        const sceneExtentX = ctx.sceneBounds.max.x - ctx.sceneBounds.min.x;
        const sceneExtentY = ctx.sceneBounds.max.y - ctx.sceneBounds.min.y;
        const sceneExtentZ = ctx.sceneBounds.max.z - ctx.sceneBounds.min.z;
        const maxSceneExtent = Math.max(sceneExtentX, sceneExtentY, sceneExtentZ);

        logger.log(`filterCluster: scene extent ${maxSceneExtent.toFixed(2)}m, voxel resolution ${clampedResolution.toFixed(4)}m`);

        logger.progress.step('Building BVH');

        const gridBounds = alignGridBounds(
            ctx.sceneBounds.min.x, ctx.sceneBounds.min.y, ctx.sceneBounds.min.z,
            ctx.sceneBounds.max.x, ctx.sceneBounds.max.y, ctx.sceneBounds.max.z,
            clampedResolution
        );

        const clampAxis = (min: number, max: number) => {
            const extent = max - min;
            if (extent > maxGridExtent) {
                const center = (min + max) * 0.5;
                const half = maxGridExtent * 0.5;
                return { min: center - half, max: center + half };
            }
            return { min, max };
        };

        const cx = clampAxis(gridBounds.min.x, gridBounds.max.x);
        const cy = clampAxis(gridBounds.min.y, gridBounds.max.y);
        const cz = clampAxis(gridBounds.min.z, gridBounds.max.z);
        gridBounds.min.set(cx.min, cy.min, cz.min);
        gridBounds.max.set(cx.max, cy.max, cz.max);

        logger.progress.step('Voxelizing');

        const buffer = await voxelizeToBuffer(
            ctx.bvh, ctx.gpuVoxelization, gridBounds, clampedResolution, opacityCutoff
        );

        ctx.gpuVoxelization.destroy();
        ctx.gpuVoxelization = null;

        logger.progress.step('Finding cluster');

        const grid = buildBlockGridParams(gridBounds, clampedResolution);

        let seedBx = Math.floor((seed.x - gridBounds.min.x) / blockSize);
        let seedBy = Math.floor((seed.y - gridBounds.min.y) / blockSize);
        let seedBz = Math.floor((seed.z - gridBounds.min.z) / blockSize);

        seedBx = Math.max(0, Math.min(seedBx, grid.numBlocksX - 1));
        seedBy = Math.max(0, Math.min(seedBy, grid.numBlocksY - 1));
        seedBz = Math.max(0, Math.min(seedBz, grid.numBlocksZ - 1));

        const lookup = buildBlockLookup(buffer, grid.strideY, grid.strideZ);
        const occupied = new Set<number>([...lookup.solidSet, ...lookup.mixedMap.keys()]);

        const maxSearchRadius = Math.max(grid.numBlocksX, grid.numBlocksY, grid.numBlocksZ);
        const nearest = findNearestOccupiedBlock(occupied, seedBx, seedBy, seedBz, maxSearchRadius, grid.numBlocksX, grid.numBlocksY, grid.numBlocksZ);

        if (!nearest) {
            logger.warn('filterCluster: no occupied blocks found, returning empty result');
            logger.progress.cancel();
            return dataTable.clone({ rows: [] });
        }

        if (nearest.bx !== seedBx || nearest.by !== seedBy || nearest.bz !== seedBz) {
            const worldX = gridBounds.min.x + (nearest.bx + 0.5) * blockSize;
            const worldY = gridBounds.min.y + (nearest.by + 0.5) * blockSize;
            const worldZ = gridBounds.min.z + (nearest.bz + 0.5) * blockSize;
            logger.log(`filterCluster: seed block unoccupied, using nearest at (${worldX.toFixed(2)}, ${worldY.toFixed(2)}, ${worldZ.toFixed(2)})`);
        }

        const ccSet = findClusterFromSeed(occupied, nearest.bx, nearest.by, nearest.bz, grid.numBlocksX, grid.numBlocksY, grid.numBlocksZ);
        logger.log(`filterCluster: cluster has ${ccSet.size} blocks out of ${buffer.count} total`);

        if (ccSet.size === buffer.count) {
            logger.log('filterCluster: all blocks in one cluster, no filtering needed');
            logger.progress.step();
            return dataTable;
        }

        logger.progress.step('Filtering Gaussians');

        const gaussianCols = buildGaussianColumns(ctx);
        const keepIndices: number[] = [];

        for (let i = 0; i < numRows; i++) {
            const px = gaussianCols.posX[i];
            const py = gaussianCols.posY[i];
            const pz = gaussianCols.posZ[i];

            if (isCenterInOccupiedVoxel(px, py, pz, grid, lookup, ccSet)) {
                keepIndices.push(i);
                continue;
            }

            if (gaussianContributesToVoxels(i, gaussianCols, grid, lookup, minContribution, ccSet)) {
                keepIndices.push(i);
            }
        }

        const removed = numRows - keepIndices.length;
        logger.log(`filterCluster: keeping ${keepIndices.length} of ${numRows} Gaussians (removed ${removed})`);

        if (removed === 0) return dataTable;

        return dataTable.clone({ rows: keepIndices });
    } catch (e) {
        ctx?.gpuVoxelization?.destroy();
        logger.progress.cancel();
        throw e;
    }
};

export { filterCluster };
