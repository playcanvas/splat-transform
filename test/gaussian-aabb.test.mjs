/**
 * Tests for Gaussian AABB calculation (Phase 1 of voxelizer).
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';

import { Column, DataTable } from '../src/lib/index.js';
import {
    computeGaussianExtents
} from '../src/lib/data-table/gaussian-aabb.js';

describe('GaussianAABB', function () {
    describe('computeGaussianExtents', function () {
        it('should compute extents for a single axis-aligned Gaussian', function () {
            // Create a Gaussian at origin with no rotation and scale (1, 2, 3)
            const dataTable = new DataTable([
                new Column('x', new Float32Array([0])),
                new Column('y', new Float32Array([0])),
                new Column('z', new Float32Array([0])),
                new Column('rot_0', new Float32Array([1])),  // w = 1 (identity quaternion)
                new Column('rot_1', new Float32Array([0])),  // x = 0
                new Column('rot_2', new Float32Array([0])),  // y = 0
                new Column('rot_3', new Float32Array([0])),  // z = 0
                new Column('scale_0', new Float32Array([Math.log(1)])),  // exp(log(1)) = 1
                new Column('scale_1', new Float32Array([Math.log(2)])),  // exp(log(2)) = 2
                new Column('scale_2', new Float32Array([Math.log(3)])),  // exp(log(3)) = 3
                new Column('f_dc_0', new Float32Array([0])),
                new Column('f_dc_1', new Float32Array([0])),
                new Column('f_dc_2', new Float32Array([0])),
                new Column('opacity', new Float32Array([0]))
            ]);

            const result = computeGaussianExtents(dataTable);

            assert.strictEqual(result.extents.numRows, 1);
            assert.strictEqual(result.invalidCount, 0);

            // Get extent columns
            const extentX = result.extents.getColumnByName('extent_x').data;
            const extentY = result.extents.getColumnByName('extent_y').data;
            const extentZ = result.extents.getColumnByName('extent_z').data;

            // Half-extents should be 3-sigma (3 * scale) for Gaussian rendering
            assert.ok(Math.abs(extentX[0] - 3) < 0.001, `Expected extent X ~3, got ${extentX[0]}`);
            assert.ok(Math.abs(extentY[0] - 6) < 0.001, `Expected extent Y ~6, got ${extentY[0]}`);
            assert.ok(Math.abs(extentZ[0] - 9) < 0.001, `Expected extent Z ~9, got ${extentZ[0]}`);

            // Scene bounds should be position +/- extents (3-sigma)
            assert.ok(Math.abs(result.sceneBounds.min.x - (-3)) < 0.001);
            assert.ok(Math.abs(result.sceneBounds.min.y - (-6)) < 0.001);
            assert.ok(Math.abs(result.sceneBounds.min.z - (-9)) < 0.001);
            assert.ok(Math.abs(result.sceneBounds.max.x - 3) < 0.001);
            assert.ok(Math.abs(result.sceneBounds.max.y - 6) < 0.001);
            assert.ok(Math.abs(result.sceneBounds.max.z - 9) < 0.001);
        });

        it('should compute extents for a translated Gaussian', function () {
            // Gaussian at (10, 20, 30) with scale (1, 1, 1)
            const dataTable = new DataTable([
                new Column('x', new Float32Array([10])),
                new Column('y', new Float32Array([20])),
                new Column('z', new Float32Array([30])),
                new Column('rot_0', new Float32Array([1])),
                new Column('rot_1', new Float32Array([0])),
                new Column('rot_2', new Float32Array([0])),
                new Column('rot_3', new Float32Array([0])),
                new Column('scale_0', new Float32Array([0])),  // exp(0) = 1
                new Column('scale_1', new Float32Array([0])),
                new Column('scale_2', new Float32Array([0])),
                new Column('f_dc_0', new Float32Array([0])),
                new Column('f_dc_1', new Float32Array([0])),
                new Column('f_dc_2', new Float32Array([0])),
                new Column('opacity', new Float32Array([0]))
            ]);

            const result = computeGaussianExtents(dataTable);

            const extentX = result.extents.getColumnByName('extent_x').data;
            const extentY = result.extents.getColumnByName('extent_y').data;
            const extentZ = result.extents.getColumnByName('extent_z').data;

            // Half-extents should be 3-sigma (3 * scale = 3)
            assert.ok(Math.abs(extentX[0] - 3) < 0.001);
            assert.ok(Math.abs(extentY[0] - 3) < 0.001);
            assert.ok(Math.abs(extentZ[0] - 3) < 0.001);

            // Scene bounds should be (10, 20, 30) +/- (3, 3, 3)
            assert.ok(Math.abs(result.sceneBounds.min.x - 7) < 0.001);
            assert.ok(Math.abs(result.sceneBounds.min.y - 17) < 0.001);
            assert.ok(Math.abs(result.sceneBounds.min.z - 27) < 0.001);
            assert.ok(Math.abs(result.sceneBounds.max.x - 13) < 0.001);
            assert.ok(Math.abs(result.sceneBounds.max.y - 23) < 0.001);
            assert.ok(Math.abs(result.sceneBounds.max.z - 33) < 0.001);
        });

        it('should compute larger extents for rotated Gaussians', function () {
            // 45-degree rotation around Z axis
            // Quaternion for 45 deg around Z: (cos(22.5°), 0, 0, sin(22.5°))
            const angle = Math.PI / 4;  // 45 degrees
            const halfAngle = angle / 2;
            const w = Math.cos(halfAngle);
            const z = Math.sin(halfAngle);

            // Gaussian with scale (2, 1, 1) rotated 45 degrees around Z
            // The AABB should expand due to rotation
            const dataTable = new DataTable([
                new Column('x', new Float32Array([0])),
                new Column('y', new Float32Array([0])),
                new Column('z', new Float32Array([0])),
                new Column('rot_0', new Float32Array([w])),  // w
                new Column('rot_1', new Float32Array([0])),  // x
                new Column('rot_2', new Float32Array([0])),  // y
                new Column('rot_3', new Float32Array([z])),  // z
                new Column('scale_0', new Float32Array([Math.log(2)])),  // X scale = 2
                new Column('scale_1', new Float32Array([Math.log(1)])),  // Y scale = 1
                new Column('scale_2', new Float32Array([Math.log(1)])),  // Z scale = 1
                new Column('f_dc_0', new Float32Array([0])),
                new Column('f_dc_1', new Float32Array([0])),
                new Column('f_dc_2', new Float32Array([0])),
                new Column('opacity', new Float32Array([0]))
            ]);

            const result = computeGaussianExtents(dataTable);

            const extentX = result.extents.getColumnByName('extent_x').data;
            const extentY = result.extents.getColumnByName('extent_y').data;
            const extentZ = result.extents.getColumnByName('extent_z').data;

            // After 45 deg rotation around Z, a box with halfExtents (6, 3, 3) (3-sigma) should have
            // larger X and Y extents due to rotation mixing X and Y
            // The rotated AABB should have X and Y extents > 4.2 (sqrt(2) * min of 3)
            assert.ok(extentX[0] > 4.2, `X extent should expand, got ${extentX[0]}`);
            assert.ok(extentY[0] > 4.2, `Y extent should expand, got ${extentY[0]}`);
            assert.ok(Math.abs(extentZ[0] - 3) < 0.001, `Z extent should stay ~3, got ${extentZ[0]}`);
        });

        it('should compute scene bounds for multiple Gaussians', function () {
            // Two Gaussians at different positions
            const dataTable = new DataTable([
                new Column('x', new Float32Array([-5, 10])),
                new Column('y', new Float32Array([0, 0])),
                new Column('z', new Float32Array([0, 0])),
                new Column('rot_0', new Float32Array([1, 1])),
                new Column('rot_1', new Float32Array([0, 0])),
                new Column('rot_2', new Float32Array([0, 0])),
                new Column('rot_3', new Float32Array([0, 0])),
                new Column('scale_0', new Float32Array([0, 0])),  // extent = 1
                new Column('scale_1', new Float32Array([0, 0])),
                new Column('scale_2', new Float32Array([0, 0])),
                new Column('f_dc_0', new Float32Array([0, 0])),
                new Column('f_dc_1', new Float32Array([0, 0])),
                new Column('f_dc_2', new Float32Array([0, 0])),
                new Column('opacity', new Float32Array([0, 0]))
            ]);

            const result = computeGaussianExtents(dataTable);

            assert.strictEqual(result.extents.numRows, 2);

            // Scene bounds should span both Gaussians with 3-sigma extent (3)
            // First at -5: min = -5 - 3 = -8, Second at 10: max = 10 + 3 = 13
            assert.ok(Math.abs(result.sceneBounds.min.x - (-8)) < 0.001, `Min X should be -8, got ${result.sceneBounds.min.x}`);
            assert.ok(Math.abs(result.sceneBounds.max.x - 13) < 0.001, `Max X should be 13, got ${result.sceneBounds.max.x}`);
        });

        it('should return DataTable with correct column names', function () {
            const dataTable = new DataTable([
                new Column('x', new Float32Array([0])),
                new Column('y', new Float32Array([0])),
                new Column('z', new Float32Array([0])),
                new Column('rot_0', new Float32Array([1])),
                new Column('rot_1', new Float32Array([0])),
                new Column('rot_2', new Float32Array([0])),
                new Column('rot_3', new Float32Array([0])),
                new Column('scale_0', new Float32Array([0])),
                new Column('scale_1', new Float32Array([0])),
                new Column('scale_2', new Float32Array([0])),
                new Column('f_dc_0', new Float32Array([0])),
                new Column('f_dc_1', new Float32Array([0])),
                new Column('f_dc_2', new Float32Array([0])),
                new Column('opacity', new Float32Array([0]))
            ]);

            const result = computeGaussianExtents(dataTable);

            // Verify column names exist
            assert.ok(result.extents.getColumnByName('extent_x'), 'Should have extent_x column');
            assert.ok(result.extents.getColumnByName('extent_y'), 'Should have extent_y column');
            assert.ok(result.extents.getColumnByName('extent_z'), 'Should have extent_z column');
        });
    });

});
