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
        // The 12 cube edges each fuse their 4 cells of edge-bevel triangles
        // into a single quad, and the 8 corner-cap bevels pass through, so
        // the bevel count drops materially without disappearing entirely.
        const mergedFaceTris = mergedStats.tris - mergedBevels;
        assert.strictEqual(mergedFaceTris, 12,
            `expected exactly 12 fused face tris (2 per face); got ${mergedFaceTris}`);
        assert.ok(mergedBevels < rawBevels,
            `expected bevel reduction; got raw=${rawBevels}, merged=${mergedBevels}`);
        assert.ok(mergedBevels > 0,
            `corner-cap bevels must still be present; got merged=${mergedBevels}`);

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

    it('should fuse axis-diagonal edge bevels along a long column', () => {
        // A 1x8x1 column of voxels along Y has 4 vertical edges that
        // produce long axis-diagonal bevel runs (one per edge, 8 cells
        // long). Each raw run of 16 bevel triangles must collapse to a
        // single fused quad (2 tris) without altering the surface AABB.
        //
        // bitIdx = lx + ly*4 + lz*16; column at (lx=0, lz=0), ly=0..3
        // gives bits 0, 4, 8, 12 -> lo = 0x0000_1111.
        const colLo = ((1 << 0) | (1 << 4) | (1 << 8) | (1 << 12)) >>> 0;
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(xyzToMorton(0, 0, 0), colLo, 0);
        buffer.addBlock(xyzToMorton(0, 1, 0), colLo, 0);

        const bounds = makeGridBounds(0, 0, 0, 4, 8, 4);
        const raw = marchingCubes(buffer, bounds, 1.0);
        const merged = coplanarMerge(raw, 1.0);

        const rawStats = meshStats(raw);
        const mergedStats = meshStats(merged);
        const rawBevels = countBevelTris(raw);
        const mergedBevels = countBevelTris(merged);

        assert.ok(rawBevels > 0, 'column should have bevel tris in raw MC output');

        // The 4 vertical edges produce long bevel runs along Y. Each run
        // collapses to a single fused quad (2 tris), so the merged bevel
        // count drops to a small constant dominated by the end-cap and
        // corner triangles. Demand at least a 4x reduction.
        assert.ok(mergedBevels * 4 <= rawBevels,
            `expected >=4x bevel reduction; raw=${rawBevels}, merged=${mergedBevels}`);

        // Surface AABB must be preserved exactly (lossless).
        for (let a = 0; a < 3; a++) {
            assert.strictEqual(mergedStats.min[a], rawStats.min[a],
                `min[${a}] changed: ${rawStats.min[a]} -> ${mergedStats.min[a]}`);
            assert.strictEqual(mergedStats.max[a], rawStats.max[a],
                `max[${a}] changed: ${rawStats.max[a]} -> ${mergedStats.max[a]}`);
        }
    });

    it('should keep convex and concave bevel runs in separate buckets', () => {
        // Two parallel 1x4x1 columns separated by a 1-voxel-wide empty
        // gap along X produce both convex bevels (on the outer corners
        // of each column) and concave bevels (in the gap between them).
        // Convex runs have one set of (axis-diagonal) plane offsets,
        // concave runs have a different set, and the merge must not
        // accidentally fuse a convex run into a concave one. We verify
        // this by demanding the AABB stays exact and the bevel count
        // shrinks materially without going to zero.
        //
        // Voxels at (0, 0..3, 0) and (2, 0..3, 0):
        //   col A bitIdx = 0 + ly*4 + 0  -> bits 0, 4, 8, 12
        //   col B bitIdx = 2 + ly*4 + 0  -> bits 2, 6, 10, 14
        const colMask = (
            (1 << 0) | (1 << 4) | (1 << 8) | (1 << 12) |
            (1 << 2) | (1 << 6) | (1 << 10) | (1 << 14)
        ) >>> 0;
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(xyzToMorton(0, 0, 0), colMask, 0);

        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const raw = marchingCubes(buffer, bounds, 1.0);
        const merged = coplanarMerge(raw, 1.0);

        const rawStats = meshStats(raw);
        const mergedStats = meshStats(merged);
        const rawBevels = countBevelTris(raw);
        const mergedBevels = countBevelTris(merged);

        assert.ok(mergedBevels > 0, 'corner-cap and short-run bevels must remain');
        assert.ok(mergedBevels < rawBevels,
            `bevel count must drop: raw=${rawBevels}, merged=${mergedBevels}`);

        for (let a = 0; a < 3; a++) {
            assert.strictEqual(mergedStats.min[a], rawStats.min[a],
                `min[${a}] changed: ${rawStats.min[a]} -> ${mergedStats.min[a]}`);
            assert.strictEqual(mergedStats.max[a], rawStats.max[a],
                `max[${a}] changed: ${rawStats.max[a]} -> ${mergedStats.max[a]}`);
        }
    });

    it('should not fuse rogue axis-diagonal triangles into shifted quads', () => {
        // Some MC cube configurations (e.g. cubeIndex 29 = c0+c2+c3+c4)
        // emit a triangle whose face normal is axis-diagonal but whose
        // vertices do NOT lie on the canonical 2-tri edge-bevel wedge
        // vertex set. Without a guard, coplanarMerge would bucket such
        // a triangle as if it were a wedge half and emit a fused quad
        // shifted by ~half a voxel from the original geometry, breaking
        // watertightness.
        //
        // Place voxels so the cube cell at (cellX=0, cellY=0, cellZ=0)
        // resolves to cubeIndex 29:
        //   c0 (0,0,0), c2 (1,1,0), c3 (0,1,0), c4 (0,0,1) all solid;
        //   c1, c5, c6, c7 all empty.
        // bitIdx = lx + ly*4 + lz*16 within the (0,0,0) block:
        //   (0,0,0) →  0 → lo bit 0
        //   (0,1,0) →  4 → lo bit 4
        //   (1,1,0) →  5 → lo bit 5
        //   (0,0,1) → 16 → lo bit 16
        const lo = ((1 << 0) | (1 << 4) | (1 << 5) | (1 << 16)) >>> 0;
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(xyzToMorton(0, 0, 0), lo, 0);

        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const raw = marchingCubes(buffer, bounds, 1.0);
        const merged = coplanarMerge(raw, 1.0);

        const rawStats = meshStats(raw);
        const mergedStats = meshStats(merged);

        // AABB must be preserved exactly. A misaligned fused quad would
        // typically pull a +X extreme in from x=1 to x=0.5 (or similar).
        for (let a = 0; a < 3; a++) {
            assert.strictEqual(mergedStats.min[a], rawStats.min[a],
                `min[${a}] changed: ${rawStats.min[a]} -> ${mergedStats.min[a]}`);
            assert.strictEqual(mergedStats.max[a], rawStats.max[a],
                `max[${a}] changed: ${rawStats.max[a]} -> ${mergedStats.max[a]}`);
        }

        // The rogue triangle in cubeIndex 29 has normal (+X, -Y, 0)/sqrt(2)
        // and would be (incorrectly) bucketed as a (pair=XY, su=+1, sv=-1)
        // wedge in cell (cellU=0, cellV=0, cellE=0). The fused "phantom"
        // quad has corners at (0.5, 1, 0), (0, 0.5, 0), (0, 0.5, 1) and
        // (0.5, 1, 1). Three of those four positions cannot be vertices
        // of the genuine MC surface for this voxel set:
        //   (0.5, 1, 0): edge (0,1,0)-(1,1,0), both solid -> no MC vertex
        //   (0, 0.5, 0): edge (0,0,0)-(0,1,0), both solid -> no MC vertex
        //   (0.5, 1, 1): edge (0,1,1)-(1,1,1), both empty -> no MC vertex
        // If any of these positions appear in the merged mesh, the
        // rogue diagonal was fused into a shifted quad.
        const phantoms = [
            [0.5, 1, 0],
            [0, 0.5, 0],
            [0.5, 1, 1]
        ];
        for (const [px, py, pz] of phantoms) {
            for (let i = 0; i < merged.positions.length; i += 3) {
                const dx = Math.abs(merged.positions[i] - px);
                const dy = Math.abs(merged.positions[i + 1] - py);
                const dz = Math.abs(merged.positions[i + 2] - pz);
                assert.ok(dx > 1e-6 || dy > 1e-6 || dz > 1e-6,
                    `merged vertex at phantom position (${px},${py},${pz}) ` +
                    'indicates a rogue diagonal triangle was fused');
            }
        }
    });

    it('should not fuse butterfly diagonal pairs that share a wedge side', () => {
        // A canonical edge-bevel wedge in cell (cellU=0, cellV=0, cellE=0)
        // for pair=XY, su=+1, sv=+1 has 4 corners:
        //   A_lo = (0.5, 0,   0)  mask bit 0x1
        //   A_hi = (0.5, 0,   1)  mask bit 0x2
        //   B_lo = (0,   0.5, 0)  mask bit 0x4
        //   B_hi = (0,   0.5, 1)  mask bit 0x8
        // Two triangles with masks 0x7 = {A_lo, A_hi, B_lo} and 0xB =
        // {A_lo, A_hi, B_hi} both have the (+x, +y, 0)/sqrt(2) normal and
        // pass the per-tri vertex check. Their masks union to 0xF (full
        // coverage) but their intersection is 0x3 = {A_lo, A_hi} - they
        // share the A SIDE of the wedge, not a diagonal. Fusing them into
        // the canonical 4-corner quad would replace a butterfly-shaped
        // region with a parallelogram offset half a voxel from the
        // original surface. The merge must reject this and emit both
        // tris verbatim.
        const positions = new Float32Array([
            0.5, 0,   0,   // 0: A_lo
            0.5, 0,   1,   // 1: A_hi
            0,   0.5, 0,   // 2: B_lo
            0,   0.5, 1    // 3: B_hi
        ]);
        // Winding chosen so each tri's face normal is (+x, +y, 0)/sqrt(2):
        //   tri1: A_lo, B_lo, A_hi -> cross = (+0.5, +0.5, 0)
        //   tri2: A_lo, B_hi, A_hi -> cross = (+0.5, +0.5, 0)
        const indices = new Uint32Array([
            0, 2, 1,
            0, 3, 1
        ]);
        const input = { positions, indices };

        const merged = coplanarMerge(input, 1.0);

        // Both tris must survive verbatim: 2 tris, 4 distinct welded
        // vertices, exactly the input positions, no spurious geometry.
        const mergedStats = meshStats(merged);
        assert.strictEqual(mergedStats.tris, 2,
            `butterfly pair must not fuse; got ${mergedStats.tris} tris`);
        assert.strictEqual(mergedStats.verts, 4,
            `expected 4 welded verts; got ${mergedStats.verts}`);

        // The fused phantom quad would introduce no new vertex positions
        // (its corners are exactly A_lo/A_hi/B_lo/B_hi), so a positional
        // check alone is not enough. Instead, verify each input triangle
        // is preserved by checking that the merged mesh has triangles
        // covering both {A_lo, B_lo, A_hi} and {A_lo, B_hi, A_hi} corner
        // sets - the fused quad would only cover the {A_lo,A_hi,B_lo,B_hi}
        // parallelogram with two triangles sharing the A_lo-B_hi (or
        // A_hi-B_lo) diagonal, neither of which equals the input pair.
        const cornerSets = new Set();
        const keyOf = (i) => {
            const x = merged.positions[i];
            const y = merged.positions[i + 1];
            const z = merged.positions[i + 2];
            return `${x},${y},${z}`;
        };
        for (let i = 0; i < merged.indices.length; i += 3) {
            const k0 = keyOf(merged.indices[i] * 3);
            const k1 = keyOf(merged.indices[i + 1] * 3);
            const k2 = keyOf(merged.indices[i + 2] * 3);
            cornerSets.add([k0, k1, k2].sort().join('|'));
        }
        const tri1Key = ['0.5,0,0', '0,0.5,0', '0.5,0,1'].sort().join('|');
        const tri2Key = ['0.5,0,0', '0,0.5,1', '0.5,0,1'].sort().join('|');
        assert.ok(cornerSets.has(tri1Key),
            'tri1 {A_lo,B_lo,A_hi} must survive verbatim');
        assert.ok(cornerSets.has(tri2Key),
            'tri2 {A_lo,B_hi,A_hi} must survive verbatim');
    });

    it('should not fuse half-cell axis-aligned face triangles into shifted full quads', () => {
        // A canonical face quad in cell (cellU=0, cellV=0) on the z=0.5
        // plane has 4 corners:
        //   c0 = (0, 0, 0.5)  mask bit 0x1 (u_lo, v_lo)
        //   c1 = (1, 0, 0.5)  mask bit 0x2 (u_hi, v_lo)
        //   c2 = (0, 1, 0.5)  mask bit 0x4 (u_lo, v_hi)
        //   c3 = (1, 1, 0.5)  mask bit 0x8 (u_hi, v_hi)
        // MC configurations like cubeIndex 31 (c0+c1+c2+c3+c4 solid) emit a
        // single half-cell triangle covering only 3 of the 4 corners on
        // this plane - the 4th corner (vertex on edge 8 of the cube) is
        // suppressed because its endpoints are both solid. Naively
        // bucketing this single triangle would mark the entire 1x1 cell
        // occupied and emit a full face quad, fabricating the missing
        // corner vertex and doubling the surface area at that cell. The
        // merge must reject this and emit the half-cell tri verbatim.
        const positions = new Float32Array([
            1, 0, 0.5,   // 0: c1 (u_hi, v_lo)
            1, 1, 0.5,   // 1: c3 (u_hi, v_hi)
            0, 1, 0.5    // 2: c2 (u_lo, v_hi)
        ]);
        // Winding for upward-facing normal (+z): c1 -> c3 -> c2 has
        // cross = (+1, 0, 0) x (-1, 0, 0) ... actually compute explicitly:
        // e = c3 - c1 = (0, 1, 0); f = c2 - c1 = (-1, 1, 0);
        // n = e x f = (1*0 - 0*1, 0*(-1) - 0*0, 0*1 - 1*(-1)) = (0, 0, 1).
        const indices = new Uint32Array([0, 1, 2]);
        const input = { positions, indices };

        const merged = coplanarMerge(input, 1.0);

        // The single half-cell triangle must survive verbatim, NOT be
        // expanded into a full 1x1 quad with a fabricated 4th corner.
        const mergedStats = meshStats(merged);
        assert.strictEqual(mergedStats.tris, 1,
            `half-cell tri must not fuse to full quad; got ${mergedStats.tris} tris`);
        assert.strictEqual(mergedStats.verts, 3,
            `expected 3 welded verts (no fabricated 4th corner); got ${mergedStats.verts}`);

        // Verify the missing corner (0, 0, 0.5) was NOT introduced.
        const mergedKeys = new Set();
        for (let i = 0; i < merged.positions.length; i += 3) {
            mergedKeys.add(`${merged.positions[i]},${merged.positions[i + 1]},${merged.positions[i + 2]}`);
        }
        assert.ok(!mergedKeys.has('0,0,0.5'),
            'missing corner (0,0,0.5) must NOT be fabricated by the merger');
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

        // The flat axis-aligned face count is what the merge attacks first;
        // demand a deep reduction (>=80%) on the face triangles alone. The
        // bump's vertical edges form long axis-diagonal bevel runs that
        // also collapse, so the bevel count must drop too.
        assert.ok(mergedFaces <= rawFaces * 0.2,
            `expected >=80% face-triangle reduction; got ${mergedFaces} of ${rawFaces}`);
        assert.ok(mergedBevels < rawBevels,
            `expected bevel reduction; got raw=${rawBevels}, merged=${mergedBevels}`);

        // Bump apex must survive losslessly. MC places the surface at
        // voxel-centre boundaries, so the bump apex sits at y=3.5 (midpoint
        // of the topmost in-corner and the empty corner above).
        assert.strictEqual(mergedStats.max[1], rawStats.max[1],
            `bump apex must be preserved exactly: raw=${rawStats.max[1]}, merged=${mergedStats.max[1]}`);
    });
});
