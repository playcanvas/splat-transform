import { Vec3 } from 'playcanvas';

import { BlockMaskBuffer } from './block-mask-buffer';
import {
    setupVoxelFilter,
    buildGaussianColumns,
    buildBlockGridParams,
    type VoxelFilterContext
} from './filter-pipeline';
import { twoLevelBFS } from './flood-fill';
import { mortonToXYZ } from './morton';
import {
    BLOCK_EMPTY,
    BLOCK_MIXED,
    BLOCK_SOLID,
    SparseVoxelGrid
} from './sparse-voxel-grid';
import {
    buildBlockLookup,
    isCenterInOccupiedVoxel,
    gaussianContributesToVoxels
} from './voxel-query';
import { alignGridBounds, voxelizeToBuffer } from './voxelize';
import { DataTable } from '../data-table';
import type { DeviceCreator } from '../types';
import { logger } from '../utils';
import { fmtCount, fmtDistance } from '../utils/logger';

/**
 * Build an inverted SparseVoxelGrid from a BlockMaskBuffer for flood-filling
 * through occupied voxels. In the returned grid, originally-occupied voxels
 * are free (unblocked) and empty space is blocked.
 *
 * @param buffer - Block mask buffer with voxelization results.
 * @param nx - Grid dimension X in voxels.
 * @param ny - Grid dimension Y in voxels.
 * @param nz - Grid dimension Z in voxels.
 * @returns Inverted grid suitable for twoLevelBFS.
 */
const buildInvertedGrid = (
    buffer: BlockMaskBuffer,
    nx: number, ny: number, nz: number
): SparseVoxelGrid => {
    const grid = new SparseVoxelGrid(nx, ny, nz);

    // Default blockType is BLOCK_EMPTY (0). For the inverted grid,
    // non-occupied blocks must be BLOCK_SOLID (fully blocked).
    grid.blockType.fill(BLOCK_SOLID);
    grid.occupancy.fill(0xFFFFFFFF >>> 0);
    const totalBlocks = grid.nbx * grid.nby * grid.nbz;
    const lastWord = totalBlocks >>> 5;
    const remainder = totalBlocks & 31;
    if (remainder > 0) {
        grid.occupancy[lastWord] = ((1 << remainder) - 1) >>> 0;
    }

    const solidMortons = buffer.getSolidBlocks();
    for (let i = 0; i < solidMortons.length; i++) {
        const [bx, by, bz] = mortonToXYZ(solidMortons[i]);
        const blockIdx = bx + by * grid.nbx + bz * grid.bStride;
        grid.blockType[blockIdx] = BLOCK_EMPTY;
        grid.occupancy[blockIdx >>> 5] &= ~(1 << (blockIdx & 31));
    }

    const mixed = buffer.getMixedBlocks();
    for (let i = 0; i < mixed.morton.length; i++) {
        const [bx, by, bz] = mortonToXYZ(mixed.morton[i]);
        const blockIdx = bx + by * grid.nbx + bz * grid.bStride;
        grid.blockType[blockIdx] = BLOCK_MIXED;
        grid.masks.set(blockIdx, (~mixed.masks[i * 2]) >>> 0, (~mixed.masks[i * 2 + 1]) >>> 0);
    }

    return grid;
};

/**
 * Find the connected component of occupied voxels reachable from a seed
 * position via 6-connected voxel-level flood fill. Returns the set of block
 * linear indices that contain at least one reachable voxel, the visited grid,
 * and the resolved seed position.
 *
 * If the seed voxel is not occupied, finds the nearest occupied voxel first.
 *
 * @param buffer - Block mask buffer with voxelization results.
 * @param nx - Grid dimension X in voxels.
 * @param ny - Grid dimension Y in voxels.
 * @param nz - Grid dimension Z in voxels.
 * @param seedIx - Seed voxel X coordinate.
 * @param seedIy - Seed voxel Y coordinate.
 * @param seedIz - Seed voxel Z coordinate.
 * @returns Object with ccSet, visited grid, and the resolved seed, or null if no occupied voxel found.
 */
const findClusterVoxelFlood = (
    buffer: BlockMaskBuffer,
    nx: number, ny: number, nz: number,
    seedIx: number, seedIy: number, seedIz: number
): { ccSet: Set<number>; visited: SparseVoxelGrid; resolvedSeed: { ix: number; iy: number; iz: number } } | null => {
    const blocked = buildInvertedGrid(buffer, nx, ny, nz);
    const nbx = nx >> 2;
    const bStride = nbx * (ny >> 2);

    // In the inverted grid, occupied voxels are "free" (unblocked).
    // If seed is blocked (unoccupied), find nearest free (occupied) voxel.
    if (blocked.getVoxel(seedIx, seedIy, seedIz)) {
        const maxRadius = Math.max(nx, ny, nz);
        const nearest = SparseVoxelGrid.findNearestFreeCell(blocked, seedIx, seedIy, seedIz, maxRadius);
        if (!nearest) return null;
        seedIx = nearest.ix;
        seedIy = nearest.iy;
        seedIz = nearest.iz;
    }

    const seedBlockIdx = (seedIx >> 2) + (seedIy >> 2) * nbx + (seedIz >> 2) * bStride;
    const seedBt = blocked.blockType[seedBlockIdx];
    const blockSeeds = seedBt === BLOCK_EMPTY ? [seedBlockIdx] : [];
    const voxelSeeds = seedBt === BLOCK_EMPTY ? [] : [{ ix: seedIx, iy: seedIy, iz: seedIz }];

    const visited = twoLevelBFS(blocked, blockSeeds, voxelSeeds, nx, ny, nz);

    const ccSet = new Set<number>();
    const totalBlocks = nbx * (ny >> 2) * (nz >> 2);
    for (let w = 0; w < visited.occupancy.length; w++) {
        let bits = visited.occupancy[w];
        while (bits) {
            const bitPos = 31 - Math.clz32(bits & -bits);
            const blockIdx = w * 32 + bitPos;
            if (blockIdx < totalBlocks) {
                ccSet.add(blockIdx);
            }
            bits &= bits - 1;
        }
    }

    return { ccSet, visited, resolvedSeed: { ix: seedIx, iy: seedIy, iz: seedIz } };
};

/**
 * Filter a Gaussian splat DataTable to keep only Gaussians that contribute to
 * the connected component found by GPU voxelization from a seed position.
 *
 * @param dataTable - Input Gaussian splat data.
 * @param createDevice - Function to create a GPU device for voxelization.
 * @param voxelResolution - Voxel size in world units for coarse voxelization. Default: 1.0.
 * @param seed - Seed position in world space. Default: (0,0,0).
 * @param opacityCutoff - Opacity threshold for solid voxels. Default: 0.999.
 * @param minContribution - Minimum Gaussian contribution at a cluster voxel center to be kept. Default: 0.1.
 * @returns Filtered DataTable containing only Gaussians in the seed's cluster.
 */
const filterCluster = async (
    dataTable: DataTable,
    createDevice: DeviceCreator,
    voxelResolution: number = 1.0,
    seed: Vec3 = Vec3.ZERO,
    opacityCutoff: number = 0.999,
    minContribution: number = 0.1
): Promise<DataTable> => {
    if (!Number.isFinite(voxelResolution) || voxelResolution <= 0) {
        throw new Error(`filterCluster: voxelResolution must be a positive finite number, got ${voxelResolution}`);
    }
    if (!Number.isFinite(opacityCutoff) || opacityCutoff < 0 || opacityCutoff > 1) {
        throw new Error(`filterCluster: opacityCutoff must be a finite number in [0, 1], got ${opacityCutoff}`);
    }
    if (!Number.isFinite(minContribution) || minContribution < 0) {
        throw new Error(`filterCluster: minContribution must be a non-negative finite number, got ${minContribution}`);
    }

    const numRows = dataTable.numRows;
    if (numRows === 0) return dataTable;

    const g = logger.group('Filter cluster');

    // Emit the action's gaussian delta inside its own group, then close it.
    // The "filter-cluster:" prefix would just restate the group header.
    const finish = (out: DataTable): DataTable => {
        const removed = numRows - out.numRows;
        if (removed > 0) {
            logger.info(`removed ${removed} gaussians`);
        }
        g.end();
        return out;
    };

    let ctx: VoxelFilterContext | undefined;
    try {
        ctx = await setupVoxelFilter(dataTable, createDevice);

        const clampedResolution = Math.max(0.01, voxelResolution);
        const maxGridExtent = 8192 * clampedResolution;

        const sceneExtentX = ctx.sceneBounds.max.x - ctx.sceneBounds.min.x;
        const sceneExtentY = ctx.sceneBounds.max.y - ctx.sceneBounds.min.y;
        const sceneExtentZ = ctx.sceneBounds.max.z - ctx.sceneBounds.min.z;

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

        const grid = buildBlockGridParams(gridBounds, clampedResolution);
        const nbx = grid.numBlocksX;
        const nby = grid.numBlocksY;
        const nbz = grid.numBlocksZ;
        const nx = nbx * 4;
        const ny = nby * 4;
        const nz = nbz * 4;

        const totalVoxels = nx * ny * nz;
        logger.info(`scene: ${fmtDistance(sceneExtentX)} x ${fmtDistance(sceneExtentY)} x ${fmtDistance(sceneExtentZ)}, grid: ${nx} x ${ny} x ${nz} voxels (${fmtCount(totalVoxels)}) @ ${fmtDistance(clampedResolution)}`);

        const buffer = await voxelizeToBuffer(
            ctx.bvh, ctx.gpuVoxelization!, gridBounds, clampedResolution, opacityCutoff
        );

        ctx.gpuVoxelization.destroy();
        ctx.gpuVoxelization = null;

        if (buffer.count === 0) {
            logger.warn('no occupied blocks found, returning empty result');
            return finish(dataTable.clone({ rows: [] }));
        }

        const seedIx = Math.max(0, Math.min(Math.floor((seed.x - gridBounds.min.x) / clampedResolution), nx - 1));
        const seedIy = Math.max(0, Math.min(Math.floor((seed.y - gridBounds.min.y) / clampedResolution), ny - 1));
        const seedIz = Math.max(0, Math.min(Math.floor((seed.z - gridBounds.min.z) / clampedResolution), nz - 1));

        const floodResult = findClusterVoxelFlood(buffer, nx, ny, nz, seedIx, seedIy, seedIz);
        if (!floodResult) {
            logger.warn('no occupied voxel found near seed, returning empty result');
            return finish(dataTable.clone({ rows: [] }));
        }

        const { ccSet, visited: visitedGrid, resolvedSeed } = floodResult;
        if (resolvedSeed.ix !== seedIx || resolvedSeed.iy !== seedIy || resolvedSeed.iz !== seedIz) {
            const worldX = gridBounds.min.x + (resolvedSeed.ix + 0.5) * clampedResolution;
            const worldY = gridBounds.min.y + (resolvedSeed.iy + 0.5) * clampedResolution;
            const worldZ = gridBounds.min.z + (resolvedSeed.iz + 0.5) * clampedResolution;
            logger.warn(`seed (${seed.x.toFixed(2)}, ${seed.y.toFixed(2)}, ${seed.z.toFixed(2)}) unoccupied; resolved to nearest at (${worldX.toFixed(2)}, ${worldY.toFixed(2)}, ${worldZ.toFixed(2)})`);
        }

        logger.info(`cluster is ${ccSet.size} of ${buffer.count} blocks`);

        // Build lookup from visited voxels only (not all original voxels in ccSet blocks).
        // Every visited voxel is originally-occupied (the BFS only traverses through them),
        // so the visited grid is a correct subset of the original buffer.
        const visitedBuffer = visitedGrid.toBuffer(0, 0, 0, nbx, nby, nbz);
        const lookup = buildBlockLookup(visitedBuffer, grid.strideY, grid.strideZ);

        if (ccSet.size === buffer.count) {
            logger.info('all blocks in one cluster, no filtering needed');
            return finish(dataTable);
        }

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
        g.end();
        throw e;
    }
};

export { buildInvertedGrid, findClusterVoxelFlood, filterCluster };
