/**
 * Tests for floor filling without GPU dilation.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';

import { Vec3 } from 'playcanvas';

import { fillFloor } from '../src/lib/voxel/fill-floor.js';
import { SparseVoxelGrid } from '../src/lib/voxel/sparse-voxel-grid.js';

const boundsForGrid = (grid, voxelResolution = 1) => ({
    min: new Vec3(0, 0, 0),
    max: new Vec3(
        grid.nx * voxelResolution,
        grid.ny * voxelResolution,
        grid.nz * voxelResolution
    )
});

describe('fillFloor', function () {
    it('fills each XZ sub-column only up to the first solid voxel', async function () {
        const grid = new SparseVoxelGrid(4, 8, 4);
        grid.setVoxel(1, 5, 2);
        grid.setVoxel(1, 7, 2);

        const { grid: result } = await fillFloor(grid, boundsForGrid(grid), 1, 0, null);

        for (let y = 0; y <= 5; y++) {
            assert.strictEqual(result.getVoxel(1, y, 2), 1, `y=${y} should be filled`);
        }
        assert.strictEqual(result.getVoxel(1, 6, 2), 0, 'fill must stop above the first solid voxel');
        assert.strictEqual(result.getVoxel(1, 7, 2), 1, 'original solids above the stop point must be preserved');
    });

    it('mutates and returns the input grid for zero dilation', async function () {
        const grid = new SparseVoxelGrid(4, 4, 4);
        const bounds = boundsForGrid(grid);
        grid.setVoxel(2, 3, 1);

        const result = await fillFloor(grid, bounds, 1, 0, null);

        assert.strictEqual(result.grid, grid);
        assert.strictEqual(result.grid.getVoxel(2, 0, 1), 1);
        assert.strictEqual(result.gridBounds, bounds);
    });

    it('validates parameters before walking columns', async function () {
        const grid = new SparseVoxelGrid(4, 4, 4);
        const bounds = boundsForGrid(grid);

        await assert.rejects(
            () => fillFloor(grid, bounds, 0, 0, null),
            /voxelResolution must be finite and > 0/
        );
        await assert.rejects(
            () => fillFloor(grid, bounds, 1, -1, null),
            /dilation must be finite and >= 0/
        );
        await assert.rejects(
            () => fillFloor(new SparseVoxelGrid(4, 6, 4), bounds, 1, 0, null),
            /Grid dimensions must be multiples of 4/
        );
    });
});
