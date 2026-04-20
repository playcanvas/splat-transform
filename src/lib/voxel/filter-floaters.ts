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
    if (!Number.isFinite(voxelResolution) || voxelResolution <= 0) {
        throw new Error(`filterFloaters: voxelResolution must be a positive finite number, got ${voxelResolution}`);
    }
    if (!Number.isFinite(opacityCutoff) || opacityCutoff < 0 || opacityCutoff > 1) {
        throw new Error(`filterFloaters: opacityCutoff must be a finite number in [0, 1], got ${opacityCutoff}`);
    }
    if (!Number.isFinite(minContribution) || minContribution < 0) {
        throw new Error(`filterFloaters: minContribution must be a non-negative finite number, got ${minContribution}`);
    }

    const numRows = dataTable.numRows;
    if (numRows === 0) return dataTable;

    const g = logger.group('Filter floaters');
    let ctx: VoxelFilterContext | undefined;
    try {
        g.step('Initializing voxel pipeline');
        ctx = await setupVoxelFilter(dataTable, createDevice);

        const blockSize = 4 * voxelResolution;
        logger.info(`voxel size: ${voxelResolution}m`);
        logger.info(`block size: ${blockSize}m`);
        logger.info(`min contribution: ${minContribution.toFixed(6)}`);

        g.step('Aligning grid bounds');

        const gridBounds = alignGridBounds(
            ctx.sceneBounds.min.x, ctx.sceneBounds.min.y, ctx.sceneBounds.min.z,
            ctx.sceneBounds.max.x, ctx.sceneBounds.max.y, ctx.sceneBounds.max.z,
            voxelResolution
        );

        g.step('Voxelizing');

        const buffer = await voxelizeToBuffer(
            ctx.bvh, ctx.gpuVoxelization!, gridBounds, voxelResolution, opacityCutoff
        );

        ctx.gpuVoxelization.destroy();
        ctx.gpuVoxelization = null;

        const grid = buildBlockGridParams(gridBounds, voxelResolution);
        const lookup = buildBlockLookup(buffer, grid.strideY, grid.strideZ);

        logger.info(`occupied blocks: ${lookup.solidSet.size + lookup.mixedMap.size} (${lookup.solidSet.size} solid, ${lookup.mixedMap.size} mixed)`);

        g.step('Filtering Gaussians');

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
        logger.info(`kept gaussians: ${keepIndices.length} of ${numRows} (removed ${removed})`);

        if (removed === 0) return dataTable;

        return dataTable.clone({ rows: keepIndices });
    } catch (e) {
        ctx?.gpuVoxelization?.destroy();
        throw e;
    } finally {
        g.end();
    }
};

export { filterFloaters };
