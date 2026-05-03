/**
 * Tests for capsule-based navigation carving.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';

import { Vec3 } from 'playcanvas';

import { carve } from '../src/lib/voxel/carve.js';
import {
    BLOCK_MIXED,
    SOLID_HI,
    SOLID_LO,
    SparseVoxelGrid
} from '../src/lib/voxel/sparse-voxel-grid.js';

const boundsForGrid = (grid, voxelResolution = 1) => ({
    min: new Vec3(0, 0, 0),
    max: new Vec3(
        grid.nx * voxelResolution,
        grid.ny * voxelResolution,
        grid.nz * voxelResolution
    )
});

const voxelCenter = (ix, iy, iz, voxelResolution = 1) => ({
    x: (ix + 0.5) * voxelResolution,
    y: (iy + 0.5) * voxelResolution,
    z: (iz + 0.5) * voxelResolution
});

const solidBlock = (grid, bx, by, bz) => {
    const blockIdx = bx + by * grid.nbx + bz * grid.bStride;
    grid.orBlock(blockIdx, SOLID_LO, SOLID_HI);
};

const buildHollowBox = sizeBlocks => {
    const grid = new SparseVoxelGrid(sizeBlocks * 4, sizeBlocks * 4, sizeBlocks * 4);
    for (let bz = 0; bz < sizeBlocks; bz++) {
        for (let by = 0; by < sizeBlocks; by++) {
            for (let bx = 0; bx < sizeBlocks; bx++) {
                const wall = bx === 0 || by === 0 || bz === 0 ||
                    bx === sizeBlocks - 1 || by === sizeBlocks - 1 || bz === sizeBlocks - 1;
                if (wall) solidBlock(grid, bx, by, bz);
            }
        }
    }
    return { grid, bounds: boundsForGrid(grid) };
};

const cpuDilate = (src, halfExtentXZ, halfExtentY) => {
    const dst = new SparseVoxelGrid(src.nx, src.ny, src.nz);
    for (let z = 0; z < src.nz; z++) {
        for (let y = 0; y < src.ny; y++) {
            for (let x = 0; x < src.nx; x++) {
                if (!src.getVoxel(x, y, z)) continue;
                const minX = Math.max(0, x - halfExtentXZ);
                const maxX = Math.min(src.nx - 1, x + halfExtentXZ);
                const minY = Math.max(0, y - halfExtentY);
                const maxY = Math.min(src.ny - 1, y + halfExtentY);
                const minZ = Math.max(0, z - halfExtentXZ);
                const maxZ = Math.min(src.nz - 1, z + halfExtentXZ);
                for (let dz = minZ; dz <= maxZ; dz++) {
                    for (let dy = minY; dy <= maxY; dy++) {
                        for (let dx = minX; dx <= maxX; dx++) {
                            dst.setVoxel(dx, dy, dz);
                        }
                    }
                }
            }
        }
    }
    return dst;
};

class CpuDilation {
    src = null;

    uploadSrc(src) {
        this.src = src;
    }

    releaseSrc() {
        this.src = null;
    }

    submitChunkSparse(
        slotIdx,
        minBx, minBy, minBz,
        outerBx, outerBy, outerBz,
        haloBx, haloBy, haloBz,
        innerBx, innerBy, innerBz,
        halfExtentXZ, halfExtentY
    ) {
        assert.ok(this.src, 'source grid must be uploaded before submit');
        const dilated = cpuDilate(this.src, halfExtentXZ, halfExtentY);
        const innerBlocks = innerBx * innerBy * innerBz;
        const types = new Uint32Array((innerBlocks + 15) >>> 4);
        const masks = new Uint32Array(innerBlocks * 2);

        let innerIdx = 0;
        for (let bz = 0; bz < innerBz; bz++) {
            const globalBz = minBz + haloBz + bz;
            for (let by = 0; by < innerBy; by++) {
                const globalBy = minBy + haloBy + by;
                for (let bx = 0; bx < innerBx; bx++, innerIdx++) {
                    const globalBx = minBx + haloBx + bx;
                    if (globalBx < 0 || globalBy < 0 || globalBz < 0 ||
                        globalBx >= dilated.nbx || globalBy >= dilated.nby || globalBz >= dilated.nbz) {
                        continue;
                    }

                    const blockIdx = globalBx + globalBy * dilated.nbx + globalBz * dilated.bStride;
                    const bt = dilated.getBlockType(blockIdx);
                    types[innerIdx >>> 4] |= bt << ((innerIdx & 15) << 1);
                    if (bt === BLOCK_MIXED) {
                        const s = dilated.masks.slot(blockIdx);
                        masks[innerIdx * 2] = dilated.masks.lo[s];
                        masks[innerIdx * 2 + 1] = dilated.masks.hi[s];
                    }
                }
            }
        }

        return {
            types: Promise.resolve(types),
            masks: Promise.resolve(masks)
        };
    }
}

const resultVoxelAtSourceVoxel = (result, ix, iy, iz, voxelResolution = 1) => {
    const world = voxelCenter(ix, iy, iz, voxelResolution);
    const rx = Math.floor((world.x - result.gridBounds.min.x) / voxelResolution);
    const ry = Math.floor((world.y - result.gridBounds.min.y) / voxelResolution);
    const rz = Math.floor((world.z - result.gridBounds.min.z) / voxelResolution);
    if (rx < 0 || ry < 0 || rz < 0 ||
        rx >= result.grid.nx || ry >= result.grid.ny || rz >= result.grid.nz) {
        return 0;
    }
    return result.grid.getVoxel(rx, ry, rz);
};

describe('carve', function () {
    it('leaves the seed voxel unoccupied and preserves exterior walls', async function () {
        const { grid, bounds } = buildHollowBox(6);
        const seed = voxelCenter(12, 12, 12);

        const result = await carve(grid, bounds, 1, 1, 0, seed, new CpuDilation());

        assert.strictEqual(resultVoxelAtSourceVoxel(result, 12, 12, 12), 0,
            'seed voxel should remain navigable');
        assert.strictEqual(resultVoxelAtSourceVoxel(result, 0, 12, 12), 1,
            'exterior wall should remain solid');
    });

    it('returns the original grid when the seed is outside the grid', async function () {
        const grid = new SparseVoxelGrid(8, 8, 8);
        const bounds = boundsForGrid(grid);
        const gpu = {
            uploadSrc() {
                throw new Error('dilation should not run for an out-of-grid seed');
            }
        };

        const result = await carve(grid, bounds, 1, 1, 0, { x: -1, y: 2, z: 2 }, gpu);

        assert.strictEqual(result.grid, grid);
        assert.strictEqual(result.gridBounds, bounds);
    });

    it('validates parameters before dilation', async function () {
        const grid = new SparseVoxelGrid(4, 4, 4);
        const bounds = boundsForGrid(grid);
        const seed = voxelCenter(1, 1, 1);

        await assert.rejects(
            () => carve(grid, bounds, 0, 1, 0, seed, null),
            /voxelResolution must be finite and > 0/
        );
        await assert.rejects(
            () => carve(grid, bounds, 1, 0, 0, seed, null),
            /capsuleHeight must be finite and > 0/
        );
        await assert.rejects(
            () => carve(grid, bounds, 1, 1, -1, seed, null),
            /capsuleRadius must be finite and >= 0/
        );
        await assert.rejects(
            () => carve(new SparseVoxelGrid(4, 6, 4), bounds, 1, 1, 0, seed, null),
            /Grid dimensions must be multiples of 4/
        );
    });
});
