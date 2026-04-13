import { type GaussianColumns } from './gaussian-eval';
import { type BlockGridParams } from './voxel-query';
import { Column, DataTable } from '../data-table/data-table';
import { computeGaussianExtents, type Bounds, type GaussianExtentsResult } from '../data-table/gaussian-aabb';
import { computeWriteTransform, transformColumns } from '../data-table/transform';
import { GpuVoxelization } from '../gpu/gpu-voxelization';
import { GaussianBVH } from '../spatial/gaussian-bvh';
import type { DeviceCreator } from '../types';
import { Transform } from '../utils/math';

/**
 * Context produced by the shared voxel filter setup pipeline.
 */
interface VoxelFilterContext {
    pcDataTable: DataTable;
    extentsResult: GaussianExtentsResult;
    sceneBounds: Bounds;
    bvh: GaussianBVH;
    gpuVoxelization: GpuVoxelization;
}

/**
 * Set up the common voxelization pipeline used by both filterCluster and filterFloaters.
 *
 * Transforms columns to world space, computes Gaussian extents, builds a BVH,
 * creates a GPU device, and uploads all Gaussians.
 *
 * @param dataTable - Input Gaussian splat data.
 * @param createDevice - Function to create a GPU device for voxelization.
 * @returns Context containing all shared resources.
 */
const setupVoxelFilter = async (
    dataTable: DataTable,
    createDevice: DeviceCreator
): Promise<VoxelFilterContext> => {
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

    const bvh = new GaussianBVH(pcDataTable, extentsResult.extents);
    const device = await createDevice();

    const gpuVoxelization = new GpuVoxelization(device);
    gpuVoxelization.uploadAllGaussians(pcDataTable, extentsResult.extents);

    return { pcDataTable, extentsResult, sceneBounds, bvh, gpuVoxelization };
};

/**
 * Build GaussianColumns from a VoxelFilterContext.
 *
 * @param ctx - Voxel filter context.
 * @returns Column arrays for Gaussian evaluation.
 */
const buildGaussianColumns = (ctx: VoxelFilterContext): GaussianColumns => ({
    posX: ctx.pcDataTable.getColumnByName('x').data,
    posY: ctx.pcDataTable.getColumnByName('y').data,
    posZ: ctx.pcDataTable.getColumnByName('z').data,
    rotW: ctx.pcDataTable.getColumnByName('rot_0').data,
    rotX: ctx.pcDataTable.getColumnByName('rot_1').data,
    rotY: ctx.pcDataTable.getColumnByName('rot_2').data,
    rotZ: ctx.pcDataTable.getColumnByName('rot_3').data,
    scaleX: ctx.pcDataTable.getColumnByName('scale_0').data,
    scaleY: ctx.pcDataTable.getColumnByName('scale_1').data,
    scaleZ: ctx.pcDataTable.getColumnByName('scale_2').data,
    opacity: ctx.pcDataTable.getColumnByName('opacity').data,
    extentX: ctx.extentsResult.extents.getColumnByName('extent_x').data,
    extentY: ctx.extentsResult.extents.getColumnByName('extent_y').data,
    extentZ: ctx.extentsResult.extents.getColumnByName('extent_z').data
});

/**
 * Build BlockGridParams from grid bounds and voxel size.
 *
 * @param gridBounds - Block-aligned grid bounds.
 * @param voxelSize - Size of each voxel in world units.
 * @returns Block grid parameters.
 */
const buildBlockGridParams = (gridBounds: Bounds, voxelSize: number): BlockGridParams => {
    const blockSize = 4 * voxelSize;
    const numBlocksX = Math.round((gridBounds.max.x - gridBounds.min.x) / blockSize);
    const numBlocksY = Math.round((gridBounds.max.y - gridBounds.min.y) / blockSize);
    const numBlocksZ = Math.round((gridBounds.max.z - gridBounds.min.z) / blockSize);
    return {
        gridMinX: gridBounds.min.x,
        gridMinY: gridBounds.min.y,
        gridMinZ: gridBounds.min.z,
        blockSize,
        voxelSize,
        numBlocksX,
        numBlocksY,
        numBlocksZ,
        strideY: numBlocksX,
        strideZ: numBlocksX * numBlocksY
    };
};

export {
    setupVoxelFilter,
    buildGaussianColumns,
    buildBlockGridParams,
    type VoxelFilterContext
};
