import { Vec3 } from 'playcanvas';

import { BlockMaskBuffer } from './block-mask-buffer';
import {
    setupVoxelFilter,
    buildGaussianColumns,
    buildBlockGridParams,
    type VoxelFilterContext
} from './filter-pipeline';
import { twoLevelBFS } from './flood-fill';
import {
    BLOCK_EMPTY,
    BLOCK_MIXED,
    BLOCK_SOLID,
    BLOCKS_PER_WORD,
    EVEN_BITS,
    SOLID_WORD,
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
import { fmtCount, fmtDistance, logger } from '../utils';

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

    // Inverted grid: every block defaults to SOLID (fully blocked). SOLID is
    // 0b01 in each 2-bit lane, so the SOLID-everywhere word is `SOLID_WORD`
    // (0x55555555). Subsequent code clears the lanes corresponding to
    // originally-occupied (i.e. unblocked-in-the-inverted-world) blocks.
    grid.types.fill(SOLID_WORD);
    // Trim the final word so the trailing lanes (past totalBlocks) read
    // back as EMPTY rather than SOLID.
    const totalBlocks = grid.nbx * grid.nby * grid.nbz;
    if (totalBlocks === 0) {
        return grid;
    }
    const lastWord = grid.types.length - 1;
    const lastLanes = totalBlocks - lastWord * BLOCKS_PER_WORD;
    if (lastLanes < BLOCKS_PER_WORD) {
        const validBits = (1 << (lastLanes * 2)) - 1;
        grid.types[lastWord] = (grid.types[lastWord] & validBits) >>> 0;
    }

    const solidIdx = buffer.getSolidBlocks();
    for (let i = 0; i < solidIdx.length; i++) {
        grid.setBlockType(solidIdx[i], BLOCK_EMPTY);
    }

    const mixed = buffer.getMixedBlocks();
    for (let i = 0; i < mixed.blockIdx.length; i++) {
        const blockIdx = mixed.blockIdx[i];
        grid.setBlockType(blockIdx, BLOCK_MIXED);
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
    const seedBt = blocked.getBlockType(seedBlockIdx);
    const blockSeeds = seedBt === BLOCK_EMPTY ? [seedBlockIdx] : [];
    const voxelSeeds = seedBt === BLOCK_EMPTY ? [] : [{ ix: seedIx, iy: seedIy, iz: seedIz }];

    const visited = twoLevelBFS(blocked, blockSeeds, voxelSeeds, nx, ny, nz);

    const ccSet = new Set<number>();
    const totalBlocks = nbx * (ny >> 2) * (nz >> 2);
    const visitedTypes = visited.types;
    for (let w = 0; w < visitedTypes.length; w++) {
        const word = visitedTypes[w];
        if (word === 0) continue;
        let nonEmpty = ((word & EVEN_BITS) | ((word >>> 1) & EVEN_BITS)) >>> 0;
        const baseIdx = w * BLOCKS_PER_WORD;
        while (nonEmpty) {
            const bp = 31 - Math.clz32(nonEmpty & -nonEmpty);
            const blockIdx = baseIdx + (bp >>> 1);
            if (blockIdx < totalBlocks) {
                ccSet.add(blockIdx);
            }
            nonEmpty &= nonEmpty - 1;
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
            logger.info(`removed ${fmtCount(removed)} gaussians`);
        }
        g.end();
        return out;
    };

    let ctx: VoxelFilterContext | undefined;
    try {
        ctx = await setupVoxelFilter(dataTable, createDevice);

        const clampedResolution = Math.max(0.01, voxelResolution);
        // Per-axis cap: 4096 voxels = 1024 blocks. Total blocks <= 1024^3 = 2^30,
        // safely under the 2^32 limit imposed by uint32 block indexing in
        // SparseVoxelGrid (readBlockType/writeBlockType use `blockIdx >>> 4`,
        // which truncates indices >= 2^32 and corrupts visited-state tracking).
        const maxGridExtent = 4096 * clampedResolution;

        const sceneExtentX = ctx.sceneBounds.max.x - ctx.sceneBounds.min.x;
        const sceneExtentY = ctx.sceneBounds.max.y - ctx.sceneBounds.min.y;
        const sceneExtentZ = ctx.sceneBounds.max.z - ctx.sceneBounds.min.z;

        const gridBounds = alignGridBounds(
            ctx.sceneBounds.min.x, ctx.sceneBounds.min.y, ctx.sceneBounds.min.z,
            ctx.sceneBounds.max.x, ctx.sceneBounds.max.y, ctx.sceneBounds.max.z,
            clampedResolution
        );

        const clampAxis = (min: number, max: number, seedV: number) => {
            const extent = max - min;
            if (extent > maxGridExtent) {
                const half = maxGridExtent * 0.5;
                const c = Math.max(min + half, Math.min(seedV, max - half));
                return { min: c - half, max: c + half };
            }
            return { min, max };
        };

        const cx = clampAxis(gridBounds.min.x, gridBounds.max.x, seed.x);
        const cy = clampAxis(gridBounds.min.y, gridBounds.max.y, seed.y);
        const cz = clampAxis(gridBounds.min.z, gridBounds.max.z, seed.z);
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

        logger.info(`cluster is ${fmtCount(ccSet.size)} of ${fmtCount(buffer.count)} blocks`);

        // Build lookup from visited voxels only (not all original voxels in ccSet blocks).
        // Every visited voxel is originally-occupied (the BFS only traverses through them),
        // so the visited grid is a correct subset of the original buffer.
        const visitedBuffer = visitedGrid.toBuffer(0, 0, 0, nbx, nby, nbz);
        const lookup = buildBlockLookup(visitedBuffer);

        if (ccSet.size === buffer.count) {
            logger.info('all blocks in one cluster, no filtering needed');
            return finish(dataTable);
        }

        const gaussianCols = buildGaussianColumns(ctx);
        const keepIndices: number[] = [];

        // Gaussians whose AABB exceeds `largeThreshold` on any axis must hit
        // at least `minOccupancyRatio` of the voxels in their AABB to be kept,
        // rejecting elongated outliers (spikes whose tails clip a single
        // cluster voxel) while preserving structural large gaussians.
        const largeThreshold = 2.0 * clampedResolution;
        const minOccupancyRatio = 0.1;
        const invVoxel = 1 / clampedResolution;

        for (let i = 0; i < numRows; i++) {
            const px = gaussianCols.posX[i];
            const py = gaussianCols.posY[i];
            const pz = gaussianCols.posZ[i];

            if (isCenterInOccupiedVoxel(px, py, pz, grid, lookup)) {
                keepIndices.push(i);
                continue;
            }

            const ex = gaussianCols.extentX[i];
            const ey = gaussianCols.extentY[i];
            const ez = gaussianCols.extentZ[i];

            let minHits = 1;
            if (Math.max(ex, ey, ez) * 2 > largeThreshold) {
                const aabbVoxels = (2 * ex * invVoxel) * (2 * ey * invVoxel) * (2 * ez * invVoxel);
                minHits = Math.max(1, Math.ceil(aabbVoxels * minOccupancyRatio));
            }

            if (gaussianContributesToVoxels(i, gaussianCols, grid, lookup, minContribution, undefined, minHits)) {
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

export { buildInvertedGrid, findClusterVoxelFlood, filterCluster };
