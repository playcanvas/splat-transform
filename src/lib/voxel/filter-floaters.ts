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
import { fmtCount, fmtDistance, logger } from '../utils';

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

    // Emit the action's gaussian delta inside its own group, then close it.
    // The "filter-floaters:" prefix would just restate the group header.
    const finish = (out: DataTable): DataTable => {
        const removed = numRows - out.numRows;
        if (removed > 0) {
            logger.info(`removed ${fmtCount(removed)} gaussians`);
        }
        g.end();
        return out;
    };

    let ctx: VoxelFilterContext | undefined;
    try {
        ctx = await setupVoxelFilter(dataTable, createDevice);

        const sceneExtentX = ctx.sceneBounds.max.x - ctx.sceneBounds.min.x;
        const sceneExtentY = ctx.sceneBounds.max.y - ctx.sceneBounds.min.y;
        const sceneExtentZ = ctx.sceneBounds.max.z - ctx.sceneBounds.min.z;

        const gridBounds = alignGridBounds(
            ctx.sceneBounds.min.x, ctx.sceneBounds.min.y, ctx.sceneBounds.min.z,
            ctx.sceneBounds.max.x, ctx.sceneBounds.max.y, ctx.sceneBounds.max.z,
            voxelResolution
        );

        const grid = buildBlockGridParams(gridBounds, voxelResolution);
        const nx = grid.numBlocksX * 4;
        const ny = grid.numBlocksY * 4;
        const nz = grid.numBlocksZ * 4;
        const totalVoxels = nx * ny * nz;

        logger.info(`scene: ${fmtDistance(sceneExtentX)} x ${fmtDistance(sceneExtentY)} x ${fmtDistance(sceneExtentZ)}, grid: ${nx} x ${ny} x ${nz} voxels (${fmtCount(totalVoxels)}) @ ${fmtDistance(voxelResolution)}`);

        const buffer = await voxelizeToBuffer(
            ctx.bvh, ctx.gpuVoxelization!, gridBounds, voxelResolution, opacityCutoff
        );

        ctx.gpuVoxelization.destroy();
        ctx.gpuVoxelization = null;

        const lookup = buildBlockLookup(buffer, grid.strideY, grid.strideZ);

        logger.info(`occupied blocks: ${fmtCount(lookup.solidSet.size + lookup.mixedMap.size)} (${fmtCount(lookup.solidSet.size)} solid, ${fmtCount(lookup.mixedMap.size)} mixed)`);

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

        if (keepIndices.length === numRows) {
            return finish(dataTable);
        }
        return finish(dataTable.clone({ rows: keepIndices }));
    } catch (e) {
        ctx?.gpuVoxelization?.destroy();
        throw e;
    }
};

export { filterFloaters };
