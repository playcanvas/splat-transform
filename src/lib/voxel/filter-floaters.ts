import {
    buildBlockLookup,
    isCenterInOccupiedVoxel,
    gaussianContributesToVoxels,
    type BlockGridParams,
    type GaussianColumns
} from './block-lookup';
import { GpuVoxelization } from './gpu-voxelization';
import {
    alignGridBounds
} from './sparse-octree';
import { voxelizeToAccumulator } from './voxelize';
import { Column, DataTable } from '../data-table/data-table';
import { computeGaussianExtents } from '../data-table/gaussian-aabb';
import { computeWriteTransform, transformColumns } from '../data-table/transform';
import { GaussianBVH } from '../spatial/gaussian-bvh';
import type { DeviceCreator } from '../types';
import { logger } from '../utils/logger';
import { Transform } from '../utils/math';

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

    const lookup = buildBlockLookup(accumulator, strideY, strideZ);

    logger.log(`filterFloaters: ${lookup.solidSet.size + lookup.mixedMap.size} occupied blocks (${lookup.solidSet.size} solid, ${lookup.mixedMap.size} mixed)`);

    logger.progress.step('Filtering Gaussians');

    const gaussianCols: GaussianColumns = {
        posX: pcDataTable.getColumnByName('x').data,
        posY: pcDataTable.getColumnByName('y').data,
        posZ: pcDataTable.getColumnByName('z').data,
        rotW: pcDataTable.getColumnByName('rot_0').data,
        rotX: pcDataTable.getColumnByName('rot_1').data,
        rotY: pcDataTable.getColumnByName('rot_2').data,
        rotZ: pcDataTable.getColumnByName('rot_3').data,
        scaleX: pcDataTable.getColumnByName('scale_0').data,
        scaleY: pcDataTable.getColumnByName('scale_1').data,
        scaleZ: pcDataTable.getColumnByName('scale_2').data,
        opacity: pcDataTable.getColumnByName('opacity').data,
        extentX: extentsResult.extents.getColumnByName('extent_x').data,
        extentY: extentsResult.extents.getColumnByName('extent_y').data,
        extentZ: extentsResult.extents.getColumnByName('extent_z').data
    };

    const grid: BlockGridParams = {
        gridMinX: gridBounds.min.x,
        gridMinY: gridBounds.min.y,
        gridMinZ: gridBounds.min.z,
        blockSize,
        voxelSize,
        numBlocksX,
        numBlocksY,
        numBlocksZ,
        strideY,
        strideZ
    };

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
