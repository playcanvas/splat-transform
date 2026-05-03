/**
 * Tests for sparse voxel grid operations and CPU-observable dilation shortcuts.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';

import { gpuDilate3 } from '../src/lib/voxel/dilation.js';
import { sparseOrGrids } from '../src/lib/voxel/grid-ops.js';
import {
    SOLID_HI,
    SOLID_LO,
    SparseVoxelGrid
} from '../src/lib/voxel/sparse-voxel-grid.js';

class NoSubmitDilation {
    uploadSrc() {}

    releaseSrc() {}

    submitChunkSparse() {
        throw new Error('GPU submit should be skipped for this chunk');
    }
}

describe('sparseOrGrids consumeA', function () {
    it('should match non-consumed result and mutate a in place', function () {
        const buildA = () => {
            const g = new SparseVoxelGrid(8, 8, 8);
            g.setVoxel(0, 0, 0);
            g.setVoxel(1, 1, 1);
            return g;
        };
        const buildB = () => {
            const g = new SparseVoxelGrid(8, 8, 8);
            g.setVoxel(7, 7, 7);
            g.setVoxel(1, 1, 1); // overlap with a
            return g;
        };

        const aRef = buildA();
        const bRef = buildB();
        const aMut = buildA();
        const bMut = buildB();

        const reference = sparseOrGrids(aRef, bRef, false);
        const consumed = sparseOrGrids(aMut, bMut, true);

        // Reference output must match consumed output voxel-for-voxel.
        for (let z = 0; z < 8; z++) {
            for (let y = 0; y < 8; y++) {
                for (let x = 0; x < 8; x++) {
                    assert.strictEqual(
                        consumed.getVoxel(x, y, z),
                        reference.getVoxel(x, y, z),
                        `voxel (${x},${y},${z}) diverges between consumed/non-consumed`
                    );
                }
            }
        }

        // consumeA=true returns the same object as `a`.
        assert.strictEqual(consumed, aMut,
            'consumed result must alias the input a');

        // Non-consumed `aRef` must NOT have b's contributions.
        assert.strictEqual(aRef.getVoxel(7, 7, 7), 0,
            'non-consumed a must not have b\'s voxels');
    });
});

describe('gpuDilate3 sparse chunk shortcuts', function () {
    it('skips empty chunks without submitting GPU work', async function () {
        const grid = new SparseVoxelGrid(8, 4, 4);

        const result = await gpuDilate3(new NoSubmitDilation(), grid, 0, 0);

        for (let z = 0; z < grid.nz; z++) {
            for (let y = 0; y < grid.ny; y++) {
                for (let x = 0; x < grid.nx; x++) {
                    assert.strictEqual(result.getVoxel(x, y, z), 0);
                }
            }
        }
    });

    it('inserts saturated chunks directly, including partial type words', async function () {
        const grid = new SparseVoxelGrid(8, 4, 4);
        grid.orBlock(0, SOLID_LO, SOLID_HI);
        grid.orBlock(1, SOLID_LO, SOLID_HI);

        const result = await gpuDilate3(new NoSubmitDilation(), grid, 0, 0);

        for (let z = 0; z < grid.nz; z++) {
            for (let y = 0; y < grid.ny; y++) {
                for (let x = 0; x < grid.nx; x++) {
                    assert.strictEqual(result.getVoxel(x, y, z), 1);
                }
            }
        }
    });
});
