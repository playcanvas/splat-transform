/**
 * Tests for the `consumeA` opt-in input-reuse flag on `sparseOrGrids`.
 * Pins the contract so a future regression that reads from a consumed
 * input is caught here.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';

import { sparseOrGrids } from '../src/lib/voxel/grid-ops.js';
import { SparseVoxelGrid } from '../src/lib/voxel/sparse-voxel-grid.js';

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
