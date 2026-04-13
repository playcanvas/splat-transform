import {
    setupVoxelFilter,
    buildGaussianColumns,
    buildBlockGridParams
} from './filter-pipeline';
import {
    alignGridBounds
} from './sparse-octree';
import {
    buildBlockLookup,
    isCenterInOccupiedVoxel,
    gaussianContributesToVoxels
} from './voxel-query';
import { voxelizeToBuffer } from './voxelize';
import { DataTable } from '../data-table/data-table';
import type { DeviceCreator } from '../types';
import { logger } from '../utils/logger';

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
 * @param voxelResolution - Voxel size in world units. Default: 0.05.
 * @param opacityCutoff - Opacity threshold for solid voxels. Default: 0.1.
 * @param minContribution - Minimum Gaussian contribution at a voxel center to be kept. Default: 1/255.
 * @returns Filtered DataTable with floaters removed.
 */
const filterFloaters = async (
    dataTable: DataTable,
    createDevice: DeviceCreator,
    voxelResolution: number = 0.05,
    opacityCutoff: number = 0.1,
    minContribution: number = 1 / 255
): Promise<DataTable> => {
    const numRows = dataTable.numRows;
    if (numRows === 0) return dataTable;

    logger.progress.begin(4);

    logger.progress.step('Computing extents');

    const ctx = await setupVoxelFilter(dataTable, createDevice);

    const blockSize = 4 * voxelResolution;

    logger.log(`filterFloaters: voxel size ${voxelResolution}m, block size ${blockSize}m, minContribution ${minContribution.toFixed(6)}`);

    logger.progress.step('Building BVH');

    const gridBounds = alignGridBounds(
        ctx.sceneBounds.min.x, ctx.sceneBounds.min.y, ctx.sceneBounds.min.z,
        ctx.sceneBounds.max.x, ctx.sceneBounds.max.y, ctx.sceneBounds.max.z,
        voxelResolution
    );

    logger.progress.step('Voxelizing');

    const buffer = await voxelizeToBuffer(
        ctx.bvh, ctx.gpuVoxelization, gridBounds, voxelResolution, opacityCutoff
    );

    ctx.gpuVoxelization.destroy();

    const grid = buildBlockGridParams(gridBounds, voxelResolution);
    const lookup = buildBlockLookup(buffer, grid.strideY, grid.strideZ);

    logger.log(`filterFloaters: ${lookup.solidSet.size + lookup.mixedMap.size} occupied blocks (${lookup.solidSet.size} solid, ${lookup.mixedMap.size} mixed)`);

    logger.progress.step('Filtering Gaussians');

    const gaussianCols = buildGaussianColumns(ctx);
    const keepIndices: number[] = [];

    for (let i = 0; i < numRows; i++) {
        const px = gaussianCols.posX[i];
        const py = gaussianCols.posY[i];
        const pz = gaussianCols.posZ[i];

        if (isCenterInOccupiedVoxel(px, py, pz, grid, lookup)) {
            keepIndices.push(i);
            continue;
        }

        if (gaussianContributesToVoxels(i, gaussianCols, grid, lookup, minContribution)) {
            keepIndices.push(i);
        }
    }

    const removed = numRows - keepIndices.length;
    logger.log(`filterFloaters: keeping ${keepIndices.length} of ${numRows} Gaussians (removed ${removed})`);

    if (removed === 0) return dataTable;

    return dataTable.clone({ rows: keepIndices });
};

export { filterFloaters };
