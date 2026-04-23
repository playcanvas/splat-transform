import { describe, it } from 'node:test';
import assert from 'node:assert';

import { Vec3 } from 'playcanvas';

import { BlockMaskBuffer } from '../src/lib/voxel/block-mask-buffer.js';
import { xyzToMorton } from '../src/lib/voxel/morton.js';
import { marchingCubes } from '../src/lib/mesh/marching-cubes.js';
import { coplanarMerge } from '../src/lib/mesh/coplanar-merge.js';

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

/**
 * Compute triangle count, vertex count, and AABB for a mesh.
 *
 * @param {{ positions: Float32Array, indices: Uint32Array }} mesh - The mesh
 *   to scan.
 * @returns {{ tris: number, verts: number, min: number[], max: number[] }}
 *   Triangle count, vertex count, and AABB extremes.
 */
const meshStats = (mesh) => {
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < mesh.positions.length; i += 3) {
        for (let a = 0; a < 3; a++) {
            const v = mesh.positions[i + a];
            if (v < min[a]) min[a] = v;
            if (v > max[a]) max[a] = v;
        }
    }
    return {
        tris: mesh.indices.length / 3,
        verts: mesh.positions.length / 3,
        min,
        max
    };
};

/**
 * Count triangles whose face normal is not aligned with any cardinal axis
 * (i.e. bevel / corner-cutting triangles produced by marching cubes).
 *
 * @param {{ positions: Float32Array, indices: Uint32Array }} mesh - The mesh
 *   to scan.
 * @returns {number} Count of bevel triangles.
 */
const countBevelTris = (mesh) => {
    const { positions, indices } = mesh;
    let count = 0;
    for (let i = 0; i < indices.length; i += 3) {
        const ia = indices[i] * 3;
        const ib = indices[i + 1] * 3;
        const ic = indices[i + 2] * 3;
        const ex = positions[ib] - positions[ia];
        const ey = positions[ib + 1] - positions[ia + 1];
        const ez = positions[ib + 2] - positions[ia + 2];
        const fx = positions[ic] - positions[ia];
        const fy = positions[ic + 1] - positions[ia + 1];
        const fz = positions[ic + 2] - positions[ia + 2];
        const nx = ey * fz - ez * fy;
        const ny = ez * fx - ex * fz;
        const nz = ex * fy - ey * fx;
        const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
        if (len < 1e-6) continue;
        const ax = Math.abs(nx) / len;
        const ay = Math.abs(ny) / len;
        const az = Math.abs(nz) / len;
        const m = Math.max(ax, ay, az);
        if (m < 1 - 1e-3) count++;
    }
    return count;
};

describe('coplanarMerge', () => {
    it('should return an empty mesh when given an empty mesh', () => {
        const empty = { positions: new Float32Array(0), indices: new Uint32Array(0) };
        const merged = coplanarMerge(empty, 1.0);

        assert.strictEqual(merged.positions.length, 0);
        assert.strictEqual(merged.indices.length, 0);
    });

    it('should fuse the flat faces of a fully-occupied 4x4x4 slab', () => {
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);

        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const raw = marchingCubes(buffer, bounds, 1.0);
        const merged = coplanarMerge(raw, 1.0);

        const rawStats = meshStats(raw);
        const mergedStats = meshStats(merged);
        const rawBevels = countBevelTris(raw);
        const mergedBevels = countBevelTris(merged);

        // Each of the 6 flat faces collapses to a single quad (2 tris), so
        // the merged output has exactly 12 axis-aligned face triangles.
        // Bevel triangles around the cube edges and corners pass through
        // verbatim and dominate the absolute count.
        const mergedFaceTris = mergedStats.tris - mergedBevels;
        assert.strictEqual(mergedFaceTris, 12,
            `expected exactly 12 fused face tris (2 per face); got ${mergedFaceTris}`);
        assert.strictEqual(mergedBevels, rawBevels,
            `bevel tris must pass through unchanged: raw=${rawBevels}, merged=${mergedBevels}`);

        // Surface AABB must be preserved exactly (lossless).
        for (let a = 0; a < 3; a++) {
            assert.strictEqual(mergedStats.min[a], rawStats.min[a],
                `min[${a}] changed: ${rawStats.min[a]} -> ${mergedStats.min[a]}`);
            assert.strictEqual(mergedStats.max[a], rawStats.max[a],
                `max[${a}] changed: ${rawStats.max[a]} -> ${mergedStats.max[a]}`);
        }
    });

    it('should preserve bevel triangles around isolated voxels', () => {
        // A single voxel produces a marching-cubes surface with both
        // axis-aligned face triangles and corner / edge bevel triangles.
        // The merge pass must leave the bevels untouched.
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(xyzToMorton(0, 0, 0), 1, 0);

        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const raw = marchingCubes(buffer, bounds, 1.0);
        const merged = coplanarMerge(raw, 1.0);

        const rawBevels = countBevelTris(raw);
        const mergedBevels = countBevelTris(merged);

        assert.ok(rawBevels > 0, 'single voxel should have bevel tris in raw MC output');
        assert.strictEqual(mergedBevels, rawBevels,
            `bevel tris must pass through unchanged: raw=${rawBevels}, merged=${mergedBevels}`);
    });

    it('should preserve a feature bump while collapsing flat slab regions', () => {
        // 2x2 grid of blocks (8x4x8 voxels) whose ly=0 layer is a fully
        // solid 8x8 slab one voxel thick. Add a 3-voxel-tall "bump" column
        // poking up from the slab. The merge pass must fuse the slab's
        // many coplanar triangles into a handful of quads while leaving
        // the bump's bevels and side faces intact.
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
        const raw = marchingCubes(buffer, bounds, 1.0);
        const merged = coplanarMerge(raw, 1.0);

        const rawStats = meshStats(raw);
        const mergedStats = meshStats(merged);
        const rawBevels = countBevelTris(raw);
        const mergedBevels = countBevelTris(merged);
        const rawFaces = rawStats.tris - rawBevels;
        const mergedFaces = mergedStats.tris - mergedBevels;

        // The flat axis-aligned face count is what the merge attacks; bevels
        // are passed through untouched. Demand a deep reduction (>=80%) on
        // the face triangles alone.
        assert.ok(mergedFaces <= rawFaces * 0.2,
            `expected >=80% face-triangle reduction; got ${mergedFaces} of ${rawFaces}`);
        assert.strictEqual(mergedBevels, rawBevels,
            `bevel tris must pass through unchanged: raw=${rawBevels}, merged=${mergedBevels}`);

        // Bump apex must survive losslessly. MC places the surface at
        // voxel-centre boundaries, so the bump apex sits at y=3.5 (midpoint
        // of the topmost in-corner and the empty corner above).
        assert.strictEqual(mergedStats.max[1], rawStats.max[1],
            `bump apex must be preserved exactly: raw=${rawStats.max[1]}, merged=${mergedStats.max[1]}`);
    });
});
