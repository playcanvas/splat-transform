import { describe, it } from 'node:test';
import assert from 'node:assert';

import { Vec3 } from 'playcanvas';

import { BlockMaskBuffer } from '../src/lib/voxel/block-mask-buffer.js';
import { xyzToMorton } from '../src/lib/voxel/morton.js';
import { greedyVoxelMesh } from '../src/lib/mesh/greedy-mesh.js';

const SOLID_LO = 0xFFFFFFFF >>> 0;
const SOLID_HI = 0xFFFFFFFF >>> 0;

const makeGridBounds = (minX, minY, minZ, maxX, maxY, maxZ) => ({
    min: new Vec3(minX, minY, minZ),
    max: new Vec3(maxX, maxY, maxZ)
});

describe('greedyVoxelMesh', () => {
    it('returns empty mesh for empty buffer', () => {
        const buffer = new BlockMaskBuffer();
        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const mesh = greedyVoxelMesh(buffer, bounds, 1.0);

        assert.strictEqual(mesh.positions.length, 0);
        assert.strictEqual(mesh.indices.length, 0);
    });

    it('produces 12 triangles for a single isolated voxel', () => {
        const buffer = new BlockMaskBuffer();
        // Voxel (0,0,0): bitIdx=0 → lo bit 0
        buffer.addBlock(xyzToMorton(0, 0, 0), 1, 0);

        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const mesh = greedyVoxelMesh(buffer, bounds, 1.0);

        // 6 faces × 1 quad each × 2 triangles = 12 triangles. With vertex
        // welding, the cube's 8 corners are shared across all 6 faces.
        assert.strictEqual(mesh.indices.length / 3, 12, 'should have 12 triangles');
        assert.strictEqual(mesh.positions.length / 3, 8, 'should have 8 welded cube corners');
    });

    it('produces 12 triangles for a single solid 4x4x4 block', () => {
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);

        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const mesh = greedyVoxelMesh(buffer, bounds, 1.0);

        // 6 faces, each 4x4 → 1 quad each → 12 triangles. Welded vertices
        // share the cube's 8 corners across all 6 faces.
        assert.strictEqual(mesh.indices.length / 3, 12, 'should have 12 triangles for solid block');
        assert.strictEqual(mesh.positions.length / 3, 8, 'should have 8 welded cube corners');
    });

    it('produces 12 triangles for two adjacent solid blocks (8x4x4)', () => {
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);
        buffer.addBlock(xyzToMorton(1, 0, 0), SOLID_LO, SOLID_HI);

        const bounds = makeGridBounds(0, 0, 0, 8, 4, 4);
        const mesh = greedyVoxelMesh(buffer, bounds, 1.0);

        // Coplanar +Y/-Y/+Z/-Z quads should merge across the block boundary,
        // and the shared interior face produces no quads. Result is one quad
        // per face of the merged 8x4x4 box, with the 8 box corners welded.
        assert.strictEqual(mesh.indices.length / 3, 12, 'merged box should have 12 triangles');
        assert.strictEqual(mesh.positions.length / 3, 8, 'merged box should have 8 welded corners');
    });

    it('merges coplanar quads across blocks for an L-shape', () => {
        // Three solid blocks forming an L in the XY plane: (0,0), (1,0), (0,1).
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);
        buffer.addBlock(xyzToMorton(1, 0, 0), SOLID_LO, SOLID_HI);
        buffer.addBlock(xyzToMorton(0, 1, 0), SOLID_LO, SOLID_HI);

        const bounds = makeGridBounds(0, 0, 0, 8, 8, 4);
        const mesh = greedyVoxelMesh(buffer, bounds, 1.0);

        // Expected exposed quads:
        //   +X: 2 (at x=8 for y=0..4, and at x=4 for y=4..8)
        //   -X: 1 (at x=0 for y=0..8 — merged across two blocks)
        //   +Y: 2 (at y=8 for x=0..4, and at y=4 for x=4..8)
        //   -Y: 1 (at y=0 for x=0..8 — merged)
        //   +Z: 2 (L-shape at z=4 splits into two greedy rectangles)
        //   -Z: 2 (same L-shape at z=0)
        // Total: 10 quads = 20 triangles. The L-shape splits into two greedy
        // rectangles whose combined corner set has 7 unique (x,y) points
        // (the 6 outline corners plus the interior split point), duplicated
        // across the z=0 and z=4 planes for a welded total of 14 vertices.
        assert.strictEqual(mesh.indices.length / 3, 20, 'L-shape should have 20 triangles');
        assert.strictEqual(mesh.positions.length / 3, 14, 'L-shape should have 14 welded corners');
    });

    it('produces 12 triangles for a single voxel inside a mixed block', () => {
        const buffer = new BlockMaskBuffer();
        // Set voxel (lx=2, ly=2, lz=2): bitIdx = 2 + 8 + 32 = 42 → hi bit (42-32)=10
        buffer.addBlock(xyzToMorton(0, 0, 0), 0, 1 << 10);

        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const mesh = greedyVoxelMesh(buffer, bounds, 1.0);

        assert.strictEqual(mesh.indices.length / 3, 12, 'inner voxel should expose 12 triangles');
        assert.strictEqual(mesh.positions.length / 3, 8, 'inner voxel should have 8 welded corners');

        // All vertex positions must lie strictly inside the gridBounds box.
        for (let i = 0; i < mesh.positions.length; i += 3) {
            const x = mesh.positions[i];
            const y = mesh.positions[i + 1];
            const z = mesh.positions[i + 2];
            assert.ok(x >= 0 && x <= 4, `x=${x} out of grid bounds`);
            assert.ok(y >= 0 && y <= 4, `y=${y} out of grid bounds`);
            assert.ok(z >= 0 && z <= 4, `z=${z} out of grid bounds`);
        }
    });

    it('emits vertex positions on integer multiples of voxel resolution', () => {
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);
        buffer.addBlock(xyzToMorton(1, 0, 0), 1, 0); // single voxel at (4,0,0)

        const res = 0.25;
        const bounds = makeGridBounds(0, 0, 0, 2, 1, 1);
        const mesh = greedyVoxelMesh(buffer, bounds, res);

        // Greedy meshing is exact: every position component is originX + k*res
        // for some integer k. With origin at 0, that means position / res is an integer.
        const epsilon = 1e-6;
        for (let i = 0; i < mesh.positions.length; i++) {
            const v = mesh.positions[i] / res;
            const r = Math.round(v);
            assert.ok(Math.abs(v - r) < epsilon,
                `position ${mesh.positions[i]} not on voxel grid (k=${v})`);
        }
    });

    it('respects gridBounds origin offset', () => {
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);

        const bounds = makeGridBounds(10, 20, 30, 14, 24, 34);
        const mesh = greedyVoxelMesh(buffer, bounds, 1.0);

        assert.strictEqual(mesh.indices.length / 3, 12);

        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        let minZ = Infinity, maxZ = -Infinity;
        for (let i = 0; i < mesh.positions.length; i += 3) {
            minX = Math.min(minX, mesh.positions[i]);
            maxX = Math.max(maxX, mesh.positions[i]);
            minY = Math.min(minY, mesh.positions[i + 1]);
            maxY = Math.max(maxY, mesh.positions[i + 1]);
            minZ = Math.min(minZ, mesh.positions[i + 2]);
            maxZ = Math.max(maxZ, mesh.positions[i + 2]);
        }

        assert.strictEqual(minX, 10);
        assert.strictEqual(maxX, 14);
        assert.strictEqual(minY, 20);
        assert.strictEqual(maxY, 24);
        assert.strictEqual(minZ, 30);
        assert.strictEqual(maxZ, 34);
    });

    it('produces outward-pointing triangle normals', () => {
        // Single solid block centered around (2, 2, 2) in voxel coords.
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);

        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const mesh = greedyVoxelMesh(buffer, bounds, 1.0);

        const cx = 2, cy = 2, cz = 2;
        const numTris = mesh.indices.length / 3;
        for (let t = 0; t < numTris; t++) {
            const i0 = mesh.indices[t * 3] * 3;
            const i1 = mesh.indices[t * 3 + 1] * 3;
            const i2 = mesh.indices[t * 3 + 2] * 3;

            const ax = mesh.positions[i1] - mesh.positions[i0];
            const ay = mesh.positions[i1 + 1] - mesh.positions[i0 + 1];
            const az = mesh.positions[i1 + 2] - mesh.positions[i0 + 2];
            const bx = mesh.positions[i2] - mesh.positions[i0];
            const by = mesh.positions[i2 + 1] - mesh.positions[i0 + 1];
            const bz = mesh.positions[i2 + 2] - mesh.positions[i0 + 2];

            const nx = ay * bz - az * by;
            const ny = az * bx - ax * bz;
            const nz = ax * by - ay * bx;

            // Triangle centroid
            const tx = (mesh.positions[i0] + mesh.positions[i1] + mesh.positions[i2]) / 3;
            const ty = (mesh.positions[i0 + 1] + mesh.positions[i1 + 1] + mesh.positions[i2 + 1]) / 3;
            const tz = (mesh.positions[i0 + 2] + mesh.positions[i1 + 2] + mesh.positions[i2 + 2]) / 3;

            // Vector from box center to triangle centroid.
            const dx = tx - cx;
            const dy = ty - cy;
            const dz = tz - cz;

            const dot = nx * dx + ny * dy + nz * dz;
            assert.ok(dot > 0, `triangle ${t} has inward-pointing normal (dot=${dot})`);
        }
    });
});
