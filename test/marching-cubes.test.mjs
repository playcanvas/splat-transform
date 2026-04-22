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

    it('should preserve a feature bump while collapsing flat slab regions', async () => {
        // Build a 2x2 grid of blocks (8x4x8 voxels) whose ly=0 layer is a
        // fully solid 8x8 slab one voxel thick - this gives the simplifier a
        // large flat area to flush. Then add a 3-voxel-tall "bump" column
        // protruding from the slab to act as a feature with unique side and
        // top normals. Normal-weighted simplification should aggressively
        // collapse the slab while leaving the bump column intact.
        //
        // ly=0 slab voxel bits, derived from `bitIdx = lx + ly*4 + lz*16`:
        //   lz=0: bits  0..3   → lo 0x0000_000F
        //   lz=1: bits 16..19  → lo 0x000F_0000
        //   lz=2: bits 32..35  → hi 0x0000_000F
        //   lz=3: bits 48..51  → hi 0x000F_0000
        const slabLo = 0x000F_000F >>> 0;
        const slabHi = 0x000F_000F >>> 0;

        // Bump column inside block (1,0,1) at (lx=0, lz=0), ly = 1..3:
        //   ly=1: bitIdx =  4 → lo bit 4
        //   ly=2: bitIdx =  8 → lo bit 8
        //   ly=3: bitIdx = 12 → lo bit 12
        const bumpLo = ((1 << 4) | (1 << 8) | (1 << 12)) >>> 0;

        const buffer = new BlockMaskBuffer();
        buffer.addBlock(xyzToMorton(0, 0, 0), slabLo, slabHi);
        buffer.addBlock(xyzToMorton(1, 0, 0), slabLo, slabHi);
        buffer.addBlock(xyzToMorton(0, 0, 1), slabLo, slabHi);
        buffer.addBlock(xyzToMorton(1, 0, 1), (slabLo | bumpLo) >>> 0, slabHi);

        const bounds = makeGridBounds(0, 0, 0, 8, 4, 8);
        const mesh = marchingCubes(buffer, bounds, 1.0);
        const originalTriangles = mesh.indices.length / 3;

        // Generous budget so the simplifier can fully flush the coplanar
        // slab triangles. A naive (position-only) simplifier with this
        // budget happily collapses the bump column down into the slab
        // plane, since the per-edge QEM along the bump's vertical edges is
        // small. Normal-weighted simplification should refuse those
        // collapses because the bump's side and top normals diverge sharply
        // from the slab's.
        const simplified = await simplifyMesh(mesh, 0.5);
        const simplifiedTriangles = simplified.indices.length / 3;

        assert.ok(simplifiedTriangles <= originalTriangles * 0.5,
            `expected >=50% triangle reduction; got ${simplifiedTriangles} of ${originalTriangles}`);

        let maxY = -Infinity;
        for (let i = 1; i < simplified.positions.length; i += 3) {
            if (simplified.positions[i] > maxY) maxY = simplified.positions[i];
        }
        // Bump apex was at y = 4. Allow ~1 voxel of accumulated error
        // (meshopt's per-collapse budget compounds across the column), but
        // demand the bulk of the bump survives.
        assert.ok(maxY >= 3.0,
            `bump apex should survive normal-weighted simplification: maxY=${maxY}, expected >= 3.0`);
    });

    it('should produce a valid mesh in sloppy mode', async () => {
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);
        buffer.addBlock(xyzToMorton(1, 0, 0), SOLID_LO, SOLID_HI);
        buffer.addBlock(xyzToMorton(0, 1, 0), SOLID_LO, SOLID_HI);

        const bounds = makeGridBounds(0, 0, 0, 8, 8, 4);
        const mesh = marchingCubes(buffer, bounds, 1.0);

        const simplified = await simplifyMesh(mesh, 0.5, { sloppy: true });

        assert.ok(simplified.positions.length > 0, 'sloppy mesh should have vertices');
        assert.ok(simplified.indices.length > 0, 'sloppy mesh should have indices');
        assert.strictEqual(simplified.indices.length % 3, 0, 'indices should be multiple of 3');
        assert.strictEqual(simplified.positions.length % 3, 0, 'positions should be multiple of 3');

        const numVertices = simplified.positions.length / 3;
        for (let i = 0; i < simplified.indices.length; i++) {
            assert.ok(simplified.indices[i] < numVertices,
                `index ${simplified.indices[i]} out of bounds (${numVertices} vertices)`);
        }
    });
});
