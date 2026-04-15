import { describe, it } from 'node:test';
import assert from 'node:assert';

import { Vec3 } from 'playcanvas';

import { BlockMaskBuffer } from '../src/lib/voxel/block-mask-buffer.js';
import { xyzToMorton } from '../src/lib/voxel/morton.js';
import { marchingCubes } from '../src/lib/mesh/marching-cubes.js';
import { simplifyMesh } from '../src/lib/mesh/simplify.js';

const SOLID_LO = 0xFFFFFFFF >>> 0;
const SOLID_HI = 0xFFFFFFFF >>> 0;

const makeGridBounds = (minX, minY, minZ, maxX, maxY, maxZ) => ({
    min: new Vec3(minX, minY, minZ),
    max: new Vec3(maxX, maxY, maxZ)
});

describe('marchingCubes', () => {
    it('should return empty mesh for empty buffer', () => {
        const buffer = new BlockMaskBuffer();
        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const mesh = marchingCubes(buffer, bounds, 1.0);

        assert.strictEqual(mesh.positions.length, 0);
        assert.strictEqual(mesh.indices.length, 0);
    });

    it('should generate triangles for a single solid block', () => {
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);

        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const mesh = marchingCubes(buffer, bounds, 1.0);

        assert.ok(mesh.positions.length > 0, 'should have vertices');
        assert.ok(mesh.indices.length > 0, 'should have indices');
        assert.strictEqual(mesh.indices.length % 3, 0, 'indices should be multiple of 3');
        assert.strictEqual(mesh.positions.length % 3, 0, 'positions should be multiple of 3');
    });

    it('should generate a closed surface for a solid cube', () => {
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);

        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const mesh = marchingCubes(buffer, bounds, 1.0);

        const numTriangles = mesh.indices.length / 3;
        // A 4x4x4 solid cube produces boundary triangles on all 6 faces.
        // Corners share edges, so the exact count depends on marching cubes
        // table indexing. Verify it's in a reasonable range for a cube surface.
        assert.ok(numTriangles >= 180 && numTriangles <= 200,
            `expected ~192 triangles for solid 4x4x4 cube, got ${numTriangles}`);
    });

    it('should place vertices within grid bounds', () => {
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);

        const res = 0.5;
        const bounds = makeGridBounds(0, 0, 0, 2, 2, 2);
        const mesh = marchingCubes(buffer, bounds, res);

        for (let i = 0; i < mesh.positions.length; i += 3) {
            const x = mesh.positions[i];
            const y = mesh.positions[i + 1];
            const z = mesh.positions[i + 2];
            assert.ok(x >= -res && x <= 2 + res, `x=${x} out of range`);
            assert.ok(y >= -res && y <= 2 + res, `y=${y} out of range`);
            assert.ok(z >= -res && z <= 2 + res, `z=${z} out of range`);
        }
    });

    it('should produce more triangles for multiple blocks than one', () => {
        const buffer1 = new BlockMaskBuffer();
        buffer1.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);
        const mesh1 = marchingCubes(buffer1, makeGridBounds(0, 0, 0, 8, 4, 4), 1.0);

        const buffer2 = new BlockMaskBuffer();
        buffer2.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);
        buffer2.addBlock(xyzToMorton(1, 0, 0), SOLID_LO, SOLID_HI);
        const mesh2 = marchingCubes(buffer2, makeGridBounds(0, 0, 0, 8, 4, 4), 1.0);

        // Two adjacent blocks form an 8x4x4 solid. The shared face has no
        // boundary, so the total triangle count is less than 2x a single block.
        assert.ok(mesh2.indices.length > mesh1.indices.length,
            'two adjacent blocks should produce more triangles than one');
        assert.ok(mesh2.indices.length < mesh1.indices.length * 2,
            'adjacent blocks should share the internal face');
    });

    it('should handle a single-voxel mixed block', () => {
        const buffer = new BlockMaskBuffer();
        // Set only voxel (0,0,0): bitIdx = 0 + 0*4 + 0*16 = 0 → lo bit 0
        buffer.addBlock(xyzToMorton(0, 0, 0), 1, 0);

        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const mesh = marchingCubes(buffer, bounds, 1.0);

        assert.ok(mesh.positions.length > 0, 'should have vertices for single voxel');
        assert.ok(mesh.indices.length > 0, 'should have triangles for single voxel');
        const numTriangles = mesh.indices.length / 3;
        // A single isolated voxel produces triangles for each exposed face.
        // Marching cubes with binary fields may triangulate corners differently.
        assert.ok(numTriangles >= 6 && numTriangles <= 12,
            `single voxel should produce 6-12 triangles, got ${numTriangles}`);
    });

    it('should handle non-unit voxel resolution', () => {
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);

        const res = 0.25;
        const bounds = makeGridBounds(0, 0, 0, 1, 1, 1);
        const mesh = marchingCubes(buffer, bounds, res);

        assert.ok(mesh.positions.length > 0);

        for (let i = 0; i < mesh.positions.length; i += 3) {
            assert.ok(mesh.positions[i] >= -res && mesh.positions[i] <= 1 + res);
            assert.ok(mesh.positions[i + 1] >= -res && mesh.positions[i + 1] <= 1 + res);
            assert.ok(mesh.positions[i + 2] >= -res && mesh.positions[i + 2] <= 1 + res);
        }
    });
});

describe('simplifyMesh', () => {
    it('should return a valid mesh', async () => {
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);

        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const mesh = marchingCubes(buffer, bounds, 1.0);

        const simplified = await simplifyMesh(mesh, 1.0);

        assert.ok(simplified.positions.length > 0, 'simplified mesh should have vertices');
        assert.ok(simplified.indices.length > 0, 'simplified mesh should have indices');
        assert.strictEqual(simplified.indices.length % 3, 0, 'indices should be multiple of 3');
        assert.strictEqual(simplified.positions.length % 3, 0, 'positions should be multiple of 3');
    });

    it('should reduce triangle count', async () => {
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);

        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const mesh = marchingCubes(buffer, bounds, 1.0);

        const simplified = await simplifyMesh(mesh, 2.0);

        assert.ok(simplified.indices.length <= mesh.indices.length,
            'simplified mesh should have fewer or equal triangles');
        assert.ok(simplified.indices.length < mesh.indices.length,
            `expected fewer triangles: original=${mesh.indices.length / 3}, simplified=${simplified.indices.length / 3}`);
    });

    it('should keep all vertex indices within bounds', async () => {
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);
        buffer.addBlock(xyzToMorton(1, 0, 0), SOLID_LO, SOLID_HI);

        const bounds = makeGridBounds(0, 0, 0, 8, 4, 4);
        const mesh = marchingCubes(buffer, bounds, 1.0);
        const simplified = await simplifyMesh(mesh, 1.0);

        const numVertices = simplified.positions.length / 3;
        for (let i = 0; i < simplified.indices.length; i++) {
            assert.ok(simplified.indices[i] < numVertices,
                `index ${simplified.indices[i]} out of bounds (${numVertices} vertices)`);
        }
    });

    it('should handle small error threshold without crashing', async () => {
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);

        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const mesh = marchingCubes(buffer, bounds, 1.0);

        const simplified = await simplifyMesh(mesh, 0.001);

        assert.ok(simplified.positions.length > 0);
        assert.ok(simplified.indices.length > 0);
    });
});
