/**
 * Tests for Gaussian AABB calculation (Phase 1 of voxelizer).
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';

import { Vec3 } from 'playcanvas';

import {
    Column,
    DataTable,
    computeGaussianExtents,
    getGaussianAABB,
    gaussianOverlapsBox
} from '../dist/index.mjs';

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

            // Half-extents should match the scale (no rotation)
            assert.ok(Math.abs(extentX[0] - 1) < 0.001, `Expected extent X ~1, got ${extentX[0]}`);
            assert.ok(Math.abs(extentY[0] - 2) < 0.001, `Expected extent Y ~2, got ${extentY[0]}`);
            assert.ok(Math.abs(extentZ[0] - 3) < 0.001, `Expected extent Z ~3, got ${extentZ[0]}`);

            // Scene bounds should be position +/- extents
            assert.ok(Math.abs(result.sceneBounds.min.x - (-1)) < 0.001);
            assert.ok(Math.abs(result.sceneBounds.min.y - (-2)) < 0.001);
            assert.ok(Math.abs(result.sceneBounds.min.z - (-3)) < 0.001);
            assert.ok(Math.abs(result.sceneBounds.max.x - 1) < 0.001);
            assert.ok(Math.abs(result.sceneBounds.max.y - 2) < 0.001);
            assert.ok(Math.abs(result.sceneBounds.max.z - 3) < 0.001);
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

            // Half-extents should be 1, 1, 1
            assert.ok(Math.abs(extentX[0] - 1) < 0.001);
            assert.ok(Math.abs(extentY[0] - 1) < 0.001);
            assert.ok(Math.abs(extentZ[0] - 1) < 0.001);

            // Scene bounds should be (10, 20, 30) +/- (1, 1, 1)
            assert.ok(Math.abs(result.sceneBounds.min.x - 9) < 0.001);
            assert.ok(Math.abs(result.sceneBounds.min.y - 19) < 0.001);
            assert.ok(Math.abs(result.sceneBounds.min.z - 29) < 0.001);
            assert.ok(Math.abs(result.sceneBounds.max.x - 11) < 0.001);
            assert.ok(Math.abs(result.sceneBounds.max.y - 21) < 0.001);
            assert.ok(Math.abs(result.sceneBounds.max.z - 31) < 0.001);
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

            // After 45 deg rotation around Z, a box with halfExtents (2, 1, 1) should have
            // larger X and Y extents
            assert.ok(extentX[0] > 1.4, `X extent should expand, got ${extentX[0]}`);
            assert.ok(extentY[0] > 1.4, `Y extent should expand, got ${extentY[0]}`);
            assert.ok(Math.abs(extentZ[0] - 1) < 0.001, `Z extent should stay ~1, got ${extentZ[0]}`);
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

            // Scene bounds should span both Gaussians
            assert.ok(Math.abs(result.sceneBounds.min.x - (-6)) < 0.001, `Min X should be -6, got ${result.sceneBounds.min.x}`);
            assert.ok(Math.abs(result.sceneBounds.max.x - 11) < 0.001, `Max X should be 11, got ${result.sceneBounds.max.x}`);
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

    describe('getGaussianAABB', function () {
        it('should reconstruct AABB from extents and position', function () {
            const dataTable = new DataTable([
                new Column('x', new Float32Array([5])),
                new Column('y', new Float32Array([10])),
                new Column('z', new Float32Array([15])),
                new Column('rot_0', new Float32Array([1])),
                new Column('rot_1', new Float32Array([0])),
                new Column('rot_2', new Float32Array([0])),
                new Column('rot_3', new Float32Array([0])),
                new Column('scale_0', new Float32Array([Math.log(1)])),
                new Column('scale_1', new Float32Array([Math.log(2)])),
                new Column('scale_2', new Float32Array([Math.log(3)])),
                new Column('f_dc_0', new Float32Array([0])),
                new Column('f_dc_1', new Float32Array([0])),
                new Column('f_dc_2', new Float32Array([0])),
                new Column('opacity', new Float32Array([0]))
            ]);

            const result = computeGaussianExtents(dataTable);
            const min = new Vec3();
            const max = new Vec3();

            getGaussianAABB(result.extents, dataTable, 0, min, max);

            assert.ok(Math.abs(min.x - 4) < 0.001, `Min X should be 4, got ${min.x}`);
            assert.ok(Math.abs(min.y - 8) < 0.001, `Min Y should be 8, got ${min.y}`);
            assert.ok(Math.abs(min.z - 12) < 0.001, `Min Z should be 12, got ${min.z}`);
            assert.ok(Math.abs(max.x - 6) < 0.001, `Max X should be 6, got ${max.x}`);
            assert.ok(Math.abs(max.y - 12) < 0.001, `Max Y should be 12, got ${max.y}`);
            assert.ok(Math.abs(max.z - 18) < 0.001, `Max Z should be 18, got ${max.z}`);
        });
    });

    describe('gaussianOverlapsBox', function () {
        it('should detect overlapping boxes', function () {
            const dataTable = new DataTable([
                new Column('x', new Float32Array([0])),
                new Column('y', new Float32Array([0])),
                new Column('z', new Float32Array([0])),
                new Column('rot_0', new Float32Array([1])),
                new Column('rot_1', new Float32Array([0])),
                new Column('rot_2', new Float32Array([0])),
                new Column('rot_3', new Float32Array([0])),
                new Column('scale_0', new Float32Array([0])),  // extent = 1
                new Column('scale_1', new Float32Array([0])),
                new Column('scale_2', new Float32Array([0])),
                new Column('f_dc_0', new Float32Array([0])),
                new Column('f_dc_1', new Float32Array([0])),
                new Column('f_dc_2', new Float32Array([0])),
                new Column('opacity', new Float32Array([0]))
            ]);

            const result = computeGaussianExtents(dataTable);

            // Gaussian AABB is (-1, -1, -1) to (1, 1, 1)

            // Overlapping box (contained)
            assert.strictEqual(
                gaussianOverlapsBox(result.extents, dataTable, 0,
                    new Vec3(-0.5, -0.5, -0.5),
                    new Vec3(0.5, 0.5, 0.5)),
                true,
                'Contained box should overlap'
            );

            // Touching box (edge contact)
            assert.strictEqual(
                gaussianOverlapsBox(result.extents, dataTable, 0,
                    new Vec3(1, 0, 0),
                    new Vec3(2, 1, 1)),
                true,
                'Edge-touching box should overlap'
            );

            // Non-overlapping box (completely to the right)
            assert.strictEqual(
                gaussianOverlapsBox(result.extents, dataTable, 0,
                    new Vec3(2, 0, 0),
                    new Vec3(3, 1, 1)),
                false,
                'Box to the right should not overlap'
            );

            // Non-overlapping box (completely above)
            assert.strictEqual(
                gaussianOverlapsBox(result.extents, dataTable, 0,
                    new Vec3(0, 2, 0),
                    new Vec3(1, 3, 1)),
                false,
                'Box above should not overlap'
            );
        });
    });
});
