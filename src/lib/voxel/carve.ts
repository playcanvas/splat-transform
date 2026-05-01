import { Vec3 } from 'playcanvas';

import { BlockMaskBuffer } from './block-mask-buffer';
import { gpuDilate3 } from './dilation';
import type { NavSeed, NavSimplifyResult } from './fill-exterior';
import { twoLevelBFS } from './flood-fill';
import { computeEmptyGrid } from './grid-ops';
import type { Bounds } from '../data-table';
import type { GpuDilation } from '../gpu';
import {
    BLOCK_EMPTY,
    SparseVoxelGrid
} from './sparse-voxel-grid';
import { logger } from '../utils';

const carve = async (
    buffer: BlockMaskBuffer,
    gridBounds: Bounds,
    voxelResolution: number,
    capsuleHeight: number,
    capsuleRadius: number,
    seed: NavSeed,
    gpu: GpuDilation
): Promise<NavSimplifyResult> => {
    if (!Number.isFinite(voxelResolution) || voxelResolution <= 0) {
        throw new Error(`carve: voxelResolution must be finite and > 0, got ${voxelResolution}`);
    }
    if (!Number.isFinite(capsuleHeight) || capsuleHeight <= 0) {
        throw new Error(`carve: capsuleHeight must be finite and > 0, got ${capsuleHeight}`);
    }
    if (!Number.isFinite(capsuleRadius) || capsuleRadius < 0) {
        throw new Error(`carve: capsuleRadius must be finite and >= 0, got ${capsuleRadius}`);
    }

    const nx = Math.round((gridBounds.max.x - gridBounds.min.x) / voxelResolution);
    const ny = Math.round((gridBounds.max.y - gridBounds.min.y) / voxelResolution);
    const nz = Math.round((gridBounds.max.z - gridBounds.min.z) / voxelResolution);

    if (nx % 4 !== 0 || ny % 4 !== 0 || nz % 4 !== 0) {
        throw new Error(`Grid dimensions must be multiples of 4, got ${nx}x${ny}x${nz}`);
    }

    if (buffer.count === 0) {
        return { buffer, gridBounds };
    }

    const kernelR = Math.ceil(capsuleRadius / voxelResolution);
    const yHalfExtent = Math.ceil(capsuleHeight / (2 * voxelResolution));
    const nbx = nx >> 2;
    const nby = ny >> 2;
    const nbz = nz >> 2;

    const fromBufBar = logger.bar('Loading grid', Math.max(1, buffer.count));
    const gridA = SparseVoxelGrid.fromBuffer(
        buffer, nx, ny, nz,
        (done, total) => fromBufBar.update(Math.min(done, total))
    );
    fromBufBar.end();

    const blocked = await gpuDilate3(gpu, gridA, kernelR, yHalfExtent);

    let seedIx = Math.floor((seed.x - gridBounds.min.x) / voxelResolution);
    let seedIy = Math.floor((seed.y - gridBounds.min.y) / voxelResolution);
    let seedIz = Math.floor((seed.z - gridBounds.min.z) / voxelResolution);

    if (seedIx < 0 || seedIx >= nx || seedIy < 0 || seedIy >= ny || seedIz < 0 || seedIz >= nz) {
        logger.warn(`seed (${seed.x}, ${seed.y}, ${seed.z}) outside grid, skipping carve`);
        return { buffer, gridBounds };
    }

    if (blocked.getVoxel(seedIx, seedIy, seedIz)) {
        const maxRadius = Math.max(kernelR, yHalfExtent) * 2;
        const found = SparseVoxelGrid.findNearestFreeCell(blocked, seedIx, seedIy, seedIz, maxRadius);
        if (!found) {
            logger.warn(`seed (${seed.x}, ${seed.y}, ${seed.z}) blocked after dilation, no free cell within ${maxRadius} voxels, skipping carve`);
            return { buffer, gridBounds };
        }
        seedIx = found.ix;
        seedIy = found.iy;
        seedIz = found.iz;
    }

    const seedBlockIdx = (seedIx >> 2) + (seedIy >> 2) * nbx + (seedIz >> 2) * (nbx * nby);
    const seedBt = blocked.getBlockType(seedBlockIdx);
    const bSeeds = seedBt === BLOCK_EMPTY ? [seedBlockIdx] : [];
    const vSeeds = seedBt === BLOCK_EMPTY ? [] : [{ ix: seedIx, iy: seedIy, iz: seedIz }];

    // Approximate total: nbx*nby*nbz is an upper bound on whole-block fills,
    // so the bar usually finishes shy of 100% (mixed/solid blocks aren't
    // counted). Good enough for visual feedback on a long BFS.
    const bfsBar = logger.bar('BFS', nbx * nby * nbz);
    const visited = twoLevelBFS(
        blocked, bSeeds, vSeeds, nx, ny, nz,
        count => bfsBar.update(count)
    );
    bfsBar.end();

    const emptyGrid = computeEmptyGrid(visited, blocked);

    const navRegion = await gpuDilate3(gpu, emptyGrid, kernelR, yHalfExtent);

    const boundsBar = logger.bar('Scanning bounds', navRegion.types.length);
    const navBounds = navRegion.getOccupiedBlockBounds(done => boundsBar.update(done));
    boundsBar.end();

    if (!navBounds) {
        logger.warn('no navigable cells remain after carve, returning empty result');
        return {
            buffer: new BlockMaskBuffer(),
            gridBounds: { min: gridBounds.min.clone(), max: gridBounds.min.clone() }
        };
    }

    const MARGIN = 1;
    const cropMinBx = Math.max(0, navBounds.minBx - MARGIN);
    const cropMinBy = Math.max(0, navBounds.minBy - MARGIN);
    const cropMinBz = Math.max(0, navBounds.minBz - MARGIN);
    const cropMaxBx = Math.min(nbx, navBounds.maxBx + 1 + MARGIN);
    const cropMaxBy = Math.min(nby, navBounds.maxBy + 1 + MARGIN);
    const cropMaxBz = Math.min(nbz, navBounds.maxBz + 1 + MARGIN);

    const blockSize = 4 * voxelResolution;
    const croppedMin = new Vec3(
        gridBounds.min.x + cropMinBx * blockSize,
        gridBounds.min.y + cropMinBy * blockSize,
        gridBounds.min.z + cropMinBz * blockSize
    );
    const croppedBounds: Bounds = {
        min: croppedMin,
        max: new Vec3(
            croppedMin.x + (cropMaxBx - cropMinBx) * blockSize,
            croppedMin.y + (cropMaxBy - cropMinBy) * blockSize,
            croppedMin.z + (cropMaxBz - cropMinBz) * blockSize
        )
    };

    const buildBar = logger.bar('Building buffer', cropMaxBz - cropMinBz);
    const outBuffer = navRegion.toBufferInverted(
        cropMinBx, cropMinBy, cropMinBz,
        cropMaxBx, cropMaxBy, cropMaxBz,
        done => buildBar.update(done)
    );
    buildBar.end();

    return {
        buffer: outBuffer,
        gridBounds: croppedBounds
    };
};

export { carve };
