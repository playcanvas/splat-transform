/**
 * Tests for Gaussian BVH (Phase 2 of voxelizer).
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';

import { Vec3 } from 'playcanvas';

import { Column, DataTable } from '../src/lib/index.js';
import { computeGaussianExtents } from '../src/lib/data-table/gaussian-aabb.js';
import { GaussianBVH } from '../src/lib/spatial/gaussian-bvh.js';

/**
 * Create a test DataTable with Gaussians at specified positions.
 * All Gaussians have identity rotation and unit scale (extent = 1).
 */
function createTestData(positions) {
    const count = positions.length;
    const x = new Float32Array(count);
    const y = new Float32Array(count);
    const z = new Float32Array(count);

    for (let i = 0; i < count; i++) {
        x[i] = positions[i][0];
        y[i] = positions[i][1];
        z[i] = positions[i][2];
    }

    return new DataTable([
        new Column('x', x),
        new Column('y', y),
        new Column('z', z),
        new Column('rot_0', new Float32Array(count).fill(1)),  // w = 1 (identity)
        new Column('rot_1', new Float32Array(count).fill(0)),
        new Column('rot_2', new Float32Array(count).fill(0)),
        new Column('rot_3', new Float32Array(count).fill(0)),
        new Column('scale_0', new Float32Array(count).fill(0)),  // exp(0) = 1
        new Column('scale_1', new Float32Array(count).fill(0)),
        new Column('scale_2', new Float32Array(count).fill(0)),
        new Column('f_dc_0', new Float32Array(count).fill(0)),
        new Column('f_dc_1', new Float32Array(count).fill(0)),
        new Column('f_dc_2', new Float32Array(count).fill(0)),
        new Column('opacity', new Float32Array(count).fill(0))
    ]);
}

describe('GaussianBVH', function () {
    describe('constructor', function () {
        it('should build BVH from Gaussian data', function () {
            const dataTable = createTestData([
                [0, 0, 0],
                [10, 0, 0],
                [0, 10, 0]
            ]);
            const { extents } = computeGaussianExtents(dataTable);

            const bvh = new GaussianBVH(dataTable, extents);

            assert.strictEqual(bvh.count, 3);
        });

        it('should compute correct scene bounds', function () {
            const dataTable = createTestData([
                [0, 0, 0],
                [10, 0, 0]
            ]);
            const { extents } = computeGaussianExtents(dataTable);

            const bvh = new GaussianBVH(dataTable, extents);
            const bounds = bvh.sceneBounds;

            // With 3-sigma extent (3 for scale=1):
            // First Gaussian at (0,0,0): AABB (-3, -3, -3) to (3, 3, 3)
            // Second Gaussian at (10,0,0): AABB (7, -3, -3) to (13, 3, 3)
            // Combined: (-3, -3, -3) to (13, 3, 3)
            assert.ok(Math.abs(bounds.minX - (-3)) < 0.001);
            assert.ok(Math.abs(bounds.minY - (-3)) < 0.001);
            assert.ok(Math.abs(bounds.minZ - (-3)) < 0.001);
            assert.ok(Math.abs(bounds.maxX - 13) < 0.001);
            assert.ok(Math.abs(bounds.maxY - 3) < 0.001);
            assert.ok(Math.abs(bounds.maxZ - 3) < 0.001);
        });

        it('should handle large datasets', function () {
            // Create 1000 random Gaussians
            const positions = [];
            for (let i = 0; i < 1000; i++) {
                positions.push([
                    Math.random() * 100 - 50,
                    Math.random() * 100 - 50,
                    Math.random() * 100 - 50
                ]);
            }

            const dataTable = createTestData(positions);
            const { extents } = computeGaussianExtents(dataTable);

            const bvh = new GaussianBVH(dataTable, extents);

            assert.strictEqual(bvh.count, 1000);
        });
    });

    describe('queryOverlapping', function () {
        it('should find overlapping Gaussians', function () {
            const dataTable = createTestData([
                [0, 0, 0],    // AABB: (-1, -1, -1) to (1, 1, 1)
                [5, 0, 0],    // AABB: (4, -1, -1) to (6, 1, 1)
                [10, 0, 0]    // AABB: (9, -1, -1) to (11, 1, 1)
            ]);
            const { extents } = computeGaussianExtents(dataTable);
            const bvh = new GaussianBVH(dataTable, extents);

            // Query box that overlaps only the first Gaussian
            const result1 = bvh.queryOverlapping(
                new Vec3(-2, -2, -2),
                new Vec3(0, 0, 0)
            );
            assert.strictEqual(result1.length, 1);
            assert.ok(result1.includes(0));

            // Query box that overlaps first two Gaussians
            const result2 = bvh.queryOverlapping(
                new Vec3(-1, -1, -1),
                new Vec3(5, 1, 1)
            );
            assert.strictEqual(result2.length, 2);
            assert.ok(result2.includes(0));
            assert.ok(result2.includes(1));

            // Query box that overlaps all Gaussians
            const result3 = bvh.queryOverlapping(
                new Vec3(-2, -2, -2),
                new Vec3(12, 2, 2)
            );
            assert.strictEqual(result3.length, 3);
        });

        it('should return empty array for non-overlapping query', function () {
            const dataTable = createTestData([
                [0, 0, 0],
                [5, 0, 0]
            ]);
            const { extents } = computeGaussianExtents(dataTable);
            const bvh = new GaussianBVH(dataTable, extents);

            // Query box far away from all Gaussians
            const result = bvh.queryOverlapping(
                new Vec3(100, 100, 100),
                new Vec3(110, 110, 110)
            );
            assert.strictEqual(result.length, 0);
        });

        it('should handle edge-touching queries', function () {
            const dataTable = createTestData([
                [0, 0, 0]  // AABB: (-1, -1, -1) to (1, 1, 1)
            ]);
            const { extents } = computeGaussianExtents(dataTable);
            const bvh = new GaussianBVH(dataTable, extents);

            // Query box that just touches the edge
            const result = bvh.queryOverlapping(
                new Vec3(1, 0, 0),  // Touches at x=1
                new Vec3(2, 1, 1)
            );
            assert.strictEqual(result.length, 1);
        });
    });

    describe('queryOverlappingRaw', function () {
        it('should work with raw coordinates', function () {
            const dataTable = createTestData([
                [0, 0, 0],
                [5, 0, 0],
                [10, 0, 0]
            ]);
            const { extents } = computeGaussianExtents(dataTable);
            const bvh = new GaussianBVH(dataTable, extents);

            // Same query as Vec3 version but with raw coordinates
            const result = bvh.queryOverlappingRaw(-1, -1, -1, 5, 1, 1);
            assert.strictEqual(result.length, 2);
            assert.ok(result.includes(0));
            assert.ok(result.includes(1));
        });
    });

    describe('BVH efficiency', function () {
        it('should efficiently query large datasets', function () {
            // Create a grid of Gaussians
            const positions = [];
            for (let x = 0; x < 10; x++) {
                for (let y = 0; y < 10; y++) {
                    for (let z = 0; z < 10; z++) {
                        positions.push([x * 5, y * 5, z * 5]);
                    }
                }
            }

            const dataTable = createTestData(positions);
            const { extents } = computeGaussianExtents(dataTable);
            const bvh = new GaussianBVH(dataTable, extents);

            assert.strictEqual(bvh.count, 1000);

            // Query a small box that should only contain a few Gaussians
            const result = bvh.queryOverlapping(
                new Vec3(0, 0, 0),
                new Vec3(6, 6, 6)
            );

            // Should find Gaussians at (0,0,0), (5,0,0), (0,5,0), (5,5,0),
            // (0,0,5), (5,0,5), (0,5,5), (5,5,5) = 8 Gaussians
            assert.strictEqual(result.length, 8);
        });

        it('should handle empty regions efficiently', function () {
            // Create Gaussians only in positive quadrant
            const positions = [];
            for (let i = 0; i < 100; i++) {
                positions.push([
                    Math.random() * 50 + 50,
                    Math.random() * 50 + 50,
                    Math.random() * 50 + 50
                ]);
            }

            const dataTable = createTestData(positions);
            const { extents } = computeGaussianExtents(dataTable);
            const bvh = new GaussianBVH(dataTable, extents);

            // Query the negative quadrant (should be empty)
            const result = bvh.queryOverlapping(
                new Vec3(-100, -100, -100),
                new Vec3(-50, -50, -50)
            );

            assert.strictEqual(result.length, 0);
        });
    });
});
