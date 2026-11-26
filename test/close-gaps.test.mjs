/**
 * Tests for closing the holes and gaps in the solid voxels.
 */

import { after, before, describe, it } from 'node:test';
import assert from 'node:assert';

import { GpuDilation } from '../src/lib/gpu/gpu-dilation.js';
import { closeGaps, closeGapsHalfExtent } from '../src/lib/voxel/close-gaps.js';
import { SparseVoxelGrid } from '../src/lib/voxel/sparse-voxel-grid.js';

let device = null;

before(async () => {
    try {
        const { createDevice } = await import('../src/cli/node-device.js');
        device = await createDevice();
    } catch {
        device = null;
    }
});

after(() => {
    device?.destroy?.();
});

// a floor one voxel thick at y = 4, with holes of 1, 2 and 4 voxels along x
const FLOOR_Y = 4;
const holes = [
    { x0: 3, x1: 4 },
    { x0: 8, x1: 10 },
    { x0: 16, x1: 20 }
];
const inHole = (x, z) => z >= 6 && z < 10 && holes.some(h => x >= h.x0 && x < h.x1);

const buildFloor = () => {
    const grid = new SparseVoxelGrid(24, 12, 16);
    for (let z = 0; z < grid.nz; z++) {
        for (let x = 0; x < grid.nx; x++) {
            if (!inHole(x, z)) grid.setVoxel(x, FLOOR_Y, z);
        }
    }
    return grid;
};

describe('closeGapsHalfExtent', function () {
    it('grows by half the width, in whole voxels, by at least one', function () {
        assert.strictEqual(closeGapsHalfExtent(0.1, 0.05), 1);
        assert.strictEqual(closeGapsHalfExtent(0.2, 0.05), 2);
        assert.strictEqual(closeGapsHalfExtent(0.25, 0.05), 3);
        assert.strictEqual(closeGapsHalfExtent(0.01, 0.05), 1);
    });
});

describe('closeGaps', function () {
    it('validates its parameters', async function () {
        const grid = new SparseVoxelGrid(4, 4, 4);
        await assert.rejects(() => closeGaps(grid, 0, 0.1, null), /voxelResolution must be finite and > 0/);
        await assert.rejects(() => closeGaps(grid, 0.05, 0, null), /width must be finite and > 0/);
    });

    it('closes the holes up to the width, and keeps the surfaces in place', { timeout: 60000 }, async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');

        const gpu = new GpuDilation(device);
        try {
            const grid = buildFloor();
            const closed = await closeGaps(grid, 1, 2, gpu);

            for (let z = 0; z < grid.nz; z++) {
                for (let y = 0; y < grid.ny; y++) {
                    for (let x = 0; x < grid.nx; x++) {
                        // the 1 and 2 voxel holes close, the 4 voxel one stays open, and
                        // nothing is added above or below the floor
                        const expected = y === FLOOR_Y && !(inHole(x, z) && x >= 16) ? 1 : 0;
                        assert.strictEqual(closed.getVoxel(x, y, z), expected, `voxel ${x} ${y} ${z}`);
                    }
                }
            }

            // the input is not modified
            assert.strictEqual(grid.getVoxel(3, FLOOR_Y, 7), 0);
        } finally {
            gpu.destroy();
        }
    });
});
