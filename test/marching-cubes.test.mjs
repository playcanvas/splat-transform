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
        const rawFaceTris = rawStats.tris - rawBevels;
        const mergedFaceTris = mergedStats.tris - mergedBevels;

        // Lossless edge-collapse removes the strictly interior face vertices
        // of each face (the 2x2 inner grid whose fan is purely axis-aligned).
        // Demand a substantial face-tri reduction.
        assert.ok(mergedFaceTris < rawFaceTris,
            `expected face-tri reduction; got raw=${rawFaceTris}, merged=${mergedFaceTris}`);
        assert.ok(mergedFaceTris <= rawFaceTris * 0.7,
            `expected >=30% face-tri reduction; got ${mergedFaceTris} of ${rawFaceTris}`);

        // K=2 edge-collinear collapse merges each long bevel ridge of the
        // 4x4x4 cube into a single quad: the 12 cube edges contribute 24
        // tris and the 8 K>=3 corners are non-removable, so the bevel
        // count drops from rawBevels to roughly 32. Demand a clear
        // reduction with the corner bevels still present.
        assert.ok(mergedBevels < rawBevels,
            `expected bevel-tri reduction from K=2 collapse; raw=${rawBevels}, merged=${mergedBevels}`);
        assert.ok(mergedBevels >= 8,
            `at least 8 corner bevels must survive (K>=3 corners); got ${mergedBevels}`);

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

    it('should collapse the long bevel ridges of a thin column', () => {
        // A 1x8x1 column of voxels along Y has 4 vertical bevel ridges,
        // each running 8 voxel-steps tall with intermediate vertices
        // collinear along the ridge. The K=2 edge-collinear pass collapses
        // each ridge to a single quad; only the K>=3 endpoint corners
        // survive. Net effect: the bevel count drops dramatically while
        // the AABB and the corner geometry are preserved exactly.
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

        // Tri count must drop substantially via K=2 ridge collapse.
        assert.ok(mergedStats.tris < rawStats.tris * 0.5,
            `expected >=50% tri reduction from K=2 ridge collapse; raw=${rawStats.tris}, merged=${mergedStats.tris}`);

        // AABB preserved exactly (lossless).
        for (let a = 0; a < 3; a++) {
            assert.strictEqual(mergedStats.min[a], rawStats.min[a],
                `min[${a}] changed: ${rawStats.min[a]} -> ${mergedStats.min[a]}`);
            assert.strictEqual(mergedStats.max[a], rawStats.max[a],
                `max[${a}] changed: ${rawStats.max[a]} -> ${mergedStats.max[a]}`);
        }

        // No fabricated vertex positions (every output position must exist
        // verbatim in the raw input).
        const rawKeys = new Set();
        for (let i = 0; i < raw.positions.length; i += 3) {
            rawKeys.add(`${raw.positions[i]},${raw.positions[i + 1]},${raw.positions[i + 2]}`);
        }
        for (let i = 0; i < merged.positions.length; i += 3) {
            const key = `${merged.positions[i]},${merged.positions[i + 1]},${merged.positions[i + 2]}`;
            assert.ok(rawKeys.has(key),
                `merged vertex (${key}) was not in raw input (fabricated)`);
        }
    });

    it('should preserve the convex/concave bevel topology of twin columns', () => {
        // Two parallel 1x4x1 columns separated by a 1-voxel-wide empty
        // gap along X produce both convex bevels (on the outer corners
        // of each column) and concave bevels (in the gap between them).
        // The lossless edge-collapse pass must NOT confuse convex and
        // concave bevels even though they share a plane offset. K=2
        // collapse legitimately merges each long vertical bevel ridge
        // into a single quad; assert the AABB is preserved exactly and
        // no vertex is fabricated.
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

        // K=2 collapses each vertical bevel ridge to a quad; tris reduce.
        assert.ok(mergedStats.tris < rawStats.tris,
            `expected tri reduction from K=2 collapse; raw=${rawStats.tris}, merged=${mergedStats.tris}`);

        // AABB preserved exactly (lossless).
        for (let a = 0; a < 3; a++) {
            assert.strictEqual(mergedStats.min[a], rawStats.min[a],
                `min[${a}] changed: ${rawStats.min[a]} -> ${mergedStats.min[a]}`);
            assert.strictEqual(mergedStats.max[a], rawStats.max[a],
                `max[${a}] changed: ${rawStats.max[a]} -> ${mergedStats.max[a]}`);
        }

        // No fabricated vertices: convex/concave bevels must not be
        // confused into emitting positions absent from the raw input.
        const rawKeys = new Set();
        for (let i = 0; i < raw.positions.length; i += 3) {
            rawKeys.add(`${raw.positions[i]},${raw.positions[i + 1]},${raw.positions[i + 2]}`);
        }
        for (let i = 0; i < merged.positions.length; i += 3) {
            const key = `${merged.positions[i]},${merged.positions[i + 1]},${merged.positions[i + 2]}`;
            assert.ok(rawKeys.has(key),
                `merged vertex (${key}) was not in raw input (fabricated)`);
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

        // The flat axis-aligned face count is what the merge attacks first.
        // The 8x8 top face has a large interior (5x5 of strictly-inner
        // vertices, plus more outside the bump's hole) whose fans are
        // purely +Y and so collapse losslessly. Demand a deep reduction.
        assert.ok(mergedFaces <= rawFaces * 0.4,
            `expected >=60% face-triangle reduction; got ${mergedFaces} of ${rawFaces}`);

        // K=2 edge-collinear collapse merges long bevel ridges (slab
        // perimeter, bump vertical edges) into quads. Bevels reduce
        // significantly.
        assert.ok(mergedBevels < rawBevels,
            `expected bevel reduction from K=2 collapse; raw=${rawBevels}, merged=${mergedBevels}`);

        // Bump apex must survive losslessly. MC places the surface at
        // voxel-centre boundaries, so the bump apex sits at y=3.5 (midpoint
        // of the topmost in-corner and the empty corner above).
        assert.strictEqual(mergedStats.max[1], rawStats.max[1],
            `bump apex must be preserved exactly: raw=${rawStats.max[1]}, merged=${mergedStats.max[1]}`);
    });

    it('should produce a T-junction-free output', () => {
        // For a manifold mesh with no T-junctions, every undirected edge
        // appears in exactly 2 incident triangles. The lossless edge-collapse
        // pass is the inverse of vertex split, so it preserves manifoldness
        // by construction; assert this property end-to-end on a slab+bump
        // scene that exercises both flat-face collapses and bevel passthrough.
        const slabLo = 0x000F_000F >>> 0;
        const slabHi = 0x000F_000F >>> 0;
        const bumpLo = ((1 << 4) | (1 << 8) | (1 << 12)) >>> 0;
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(xyzToMorton(0, 0, 0), slabLo, slabHi);
        buffer.addBlock(xyzToMorton(1, 0, 0), slabLo, slabHi);
        buffer.addBlock(xyzToMorton(0, 0, 1), slabLo, slabHi);
        buffer.addBlock(xyzToMorton(1, 0, 1), (slabLo | bumpLo) >>> 0, slabHi);

        const bounds = makeGridBounds(0, 0, 0, 8, 4, 8);
        const raw = marchingCubes(buffer, bounds, 1.0);
        const merged = coplanarMerge(raw, 1.0);

        const edgeCount = new Map();
        const indices = merged.indices;
        for (let i = 0; i < indices.length; i += 3) {
            const a = indices[i];
            const b = indices[i + 1];
            const c = indices[i + 2];
            const addEdge = (u, v) => {
                const key = u < v ? `${u},${v}` : `${v},${u}`;
                edgeCount.set(key, (edgeCount.get(key) ?? 0) + 1);
            };
            addEdge(a, b);
            addEdge(b, c);
            addEdge(c, a);
        }
        for (const [key, count] of edgeCount) {
            assert.strictEqual(count, 2,
                `edge ${key} has ${count} incident tris (T-junction or boundary)`);
        }
    });

    it('should collapse collinear vertices on a 2-plane seam', () => {
        // Build a 90-degree wedge by hand: plane A on z=0 (+z normal) and
        // plane B on y=0 (+y normal), sharing the long edge x in [0..6],
        // y=z=0. Sub-divide the seam at x = 0,1,2,...,6 (5 strictly
        // interior vertices, all collinear with their direct seam
        // neighbours). Each interior seam vertex has K=2 with collinear
        // crease neighbours and so must be removed by the K=2 pass.
        //
        // After the worklist converges, both planes should fully simplify
        // to a single quad each (4 tris total, 8 verts total). The seam
        // becomes a single edge (0,0,0)-(6,0,0) with no interior breaks.
        const positions = new Float32Array([
            // 0..6: seam vertices
            0, 0, 0,
            1, 0, 0,
            2, 0, 0,
            3, 0, 0,
            4, 0, 0,
            5, 0, 0,
            6, 0, 0,
            // 7..8: plane A far edge (y=1)
            0, 1, 0,
            6, 1, 0,
            // 9..10: plane B far edge (z=1)
            0, 0, 1,
            6, 0, 1
        ]);
        // Plane A (+z normal): fan-triangulate from the far-edge endpoints.
        //   (0,7,1), (1,7,8), (1,8,2)? -- need consistent CCW from +z view.
        //
        // Looking down +z axis: y goes "up", x goes "right". CCW order
        // around the +z plane normal is the usual x-right, y-up convention.
        // Plane A polygon in CCW order: 0,1,2,3,4,5,6,8,7 (seam left-to-
        // right along y=0, then far edge right-to-left along y=1).
        // Triangulate via fan from vertex 7 (top-left).
        //   tri (7, 0, 1): e=(0,-1,0), f=(1,-1,0); n=(-1*0-0*-1, 0*1-0*0, 0*-1-(-1)*1)=(0,0,1) +z OK
        //   tri (7, 1, 2): same pattern +z OK
        //   ...continue for (7, i, i+1) for i in 0..5, then (7, 6, 8).
        //
        // Plane B (+y normal): polygon CCW from +y is (x-right, z-into-screen).
        // Looking down +y axis at the (x,z) plane: CCW means x-right, z-up
        // is actually CW in standard convention. Let me derive winding by
        // requiring positive cross product = +y.
        //   tri (0, 1, 9) at (0,0,0)-(1,0,0)-(0,0,1): e=(1,0,0), f=(0,0,1)
        //     n = (0*1-0*0, 0*0-1*1, 1*0-0*0) = (0,-1,0). Wrong; flip.
        //   tri (0, 9, 1): e=(0,0,1), f=(1,0,0); n=(0*0-1*0, 1*1-0*0, 0*0-0*1)=(0,1,0) +y OK
        // Plane B polygon CCW from +y: 0,9,10,6,5,4,3,2,1.
        // Fan from vertex 9: (9, 10, 6), (9, 6, 5), ..., (9, 1, 0).
        const indices = new Uint32Array([
            // Plane A fan from vertex 7
            7, 0, 1,
            7, 1, 2,
            7, 2, 3,
            7, 3, 4,
            7, 4, 5,
            7, 5, 6,
            7, 6, 8,
            // Plane B fan from vertex 9
            9, 10, 6,
            9, 6, 5,
            9, 5, 4,
            9, 4, 3,
            9, 3, 2,
            9, 2, 1,
            9, 1, 0
        ]);
        const input = { positions, indices };

        const merged = coplanarMerge(input, 1.0);
        const stats = meshStats(merged);

        // Each plane should collapse to a single quad (2 tris). Total: 4.
        assert.strictEqual(stats.tris, 4,
            `wedge with collinear seam should collapse to 4 tris; got ${stats.tris}`);

        // All 5 interior seam vertices (1, 2, 3, 4, 5 in the input) must
        // be absent from the merged mesh.
        const mergedKeys = new Set();
        for (let i = 0; i < merged.positions.length; i += 3) {
            mergedKeys.add(`${merged.positions[i]},${merged.positions[i + 1]},${merged.positions[i + 2]}`);
        }
        for (let x = 1; x <= 5; x++) {
            assert.ok(!mergedKeys.has(`${x},0,0`),
                `interior seam vertex (${x},0,0) should have been collapsed`);
        }

        // The two seam endpoints and the four far corners must survive.
        for (const key of ['0,0,0', '6,0,0', '0,1,0', '6,1,0', '0,0,1', '6,0,1']) {
            assert.ok(mergedKeys.has(key),
                `corner vertex (${key}) must survive`);
        }
    });

    it('should not collapse a kinked seam', () => {
        // Same wedge as the previous test, but with a kink in the middle
        // of the seam: vertex 4 is moved off the y=z=0 line. Verify the
        // K=2 collinearity check rejects the kink (and the two seam
        // vertices flanking the kink, since their direct neighbours are
        // no longer collinear), while the strictly-collinear vertices
        // away from the kink (1, 2, 6) still collapse.
        //
        // Seam x = 0,1,2,3, then kink at x=4 (y=0.5), then 5,6,7. So:
        //   0=(0,0,0)  1=(1,0,0)  2=(2,0,0)  3=(3,0,0)  KINK 4=(4,0.5,0)
        //   5=(5,0,0)  6=(6,0,0)  7=(7,0,0)
        //   8=(0,1,0)  9=(7,1,0)         <- plane A far edge
        //   10=(0,0,1) 11=(7,0,1)        <- plane B far edge
        const positions = new Float32Array([
            0, 0, 0,
            1, 0, 0,
            2, 0, 0,
            3, 0, 0,
            4, 0.5, 0,
            5, 0, 0,
            6, 0, 0,
            7, 0, 0,
            0, 1, 0,
            7, 1, 0,
            0, 0, 1,
            7, 0, 1
        ]);
        // Plane A (+z): fan from vertex 8.
        //   (8, 0, 1), (8, 1, 2), ..., (8, 6, 7), (8, 7, 9).
        // Plane B (+y): fan from vertex 10.
        //   (10, 11, 7), (10, 7, 6), ..., (10, 1, 0).
        const indices = new Uint32Array([
            // Plane A fan from vertex 8
            8, 0, 1,
            8, 1, 2,
            8, 2, 3,
            8, 3, 4,
            8, 4, 5,
            8, 5, 6,
            8, 6, 7,
            8, 7, 9,
            // Plane B fan from vertex 10
            10, 11, 7,
            10, 7, 6,
            10, 6, 5,
            10, 5, 4,
            10, 4, 3,
            10, 3, 2,
            10, 2, 1,
            10, 1, 0
        ]);
        const input = { positions, indices };

        const merged = coplanarMerge(input, 1.0);

        const mergedKeys = new Set();
        for (let i = 0; i < merged.positions.length; i += 3) {
            mergedKeys.add(`${merged.positions[i]},${merged.positions[i + 1]},${merged.positions[i + 2]}`);
        }

        // The kink vertex (4, 0.5, 0) must survive: its seam neighbours
        // (3,0,0) and (5,0,0) flank it, but (3, kink, 5) is not collinear.
        assert.ok(mergedKeys.has('4,0.5,0'),
            'kink vertex (4, 0.5, 0) must survive (not collinear with its seam neighbours)');

        // The seam vertices DIRECTLY ADJACENT to the kink (vertices 3 and
        // 5) also fail the K=2 collinearity test (their other seam
        // neighbour through the kink is off-axis), so they must survive.
        assert.ok(mergedKeys.has('3,0,0'),
            'seam vertex (3,0,0) adjacent to kink must survive');
        assert.ok(mergedKeys.has('5,0,0'),
            'seam vertex (5,0,0) adjacent to kink must survive');

        // The strictly-collinear seam vertices away from the kink (1, 2,
        // 6) still satisfy K=2 once the worklist propagates and so should
        // be removed.
        for (const x of [1, 2, 6]) {
            assert.ok(!mergedKeys.has(`${x},0,0`),
                `collinear seam vertex (${x},0,0) should still collapse`);
        }
    });

    it('should not produce sliver triangles', () => {
        // The lossless collapse must not output near-degenerate triangles
        // for a structured 2-plane wedge. After convergence, every output
        // triangle should have area >= voxelResolution^2 * 1e-6.
        const positions = new Float32Array([
            0, 0, 0,
            1, 0, 0,
            2, 0, 0,
            3, 0, 0,
            4, 0, 0,
            0, 1, 0,
            4, 1, 0,
            0, 0, 1,
            4, 0, 1
        ]);
        const indices = new Uint32Array([
            // Plane A (+z) fan from vertex 5
            5, 0, 1,
            5, 1, 2,
            5, 2, 3,
            5, 3, 4,
            5, 4, 6,
            // Plane B (+y) fan from vertex 7
            7, 8, 4,
            7, 4, 3,
            7, 3, 2,
            7, 2, 1,
            7, 1, 0
        ]);
        const input = { positions, indices };
        const voxelResolution = 1.0;
        const merged = coplanarMerge(input, voxelResolution);

        const minArea = voxelResolution * voxelResolution * 1e-6;
        for (let i = 0; i < merged.indices.length; i += 3) {
            const ia = merged.indices[i] * 3;
            const ib = merged.indices[i + 1] * 3;
            const ic = merged.indices[i + 2] * 3;
            const ex = merged.positions[ib] - merged.positions[ia];
            const ey = merged.positions[ib + 1] - merged.positions[ia + 1];
            const ez = merged.positions[ib + 2] - merged.positions[ia + 2];
            const fx = merged.positions[ic] - merged.positions[ia];
            const fy = merged.positions[ic + 1] - merged.positions[ia + 1];
            const fz = merged.positions[ic + 2] - merged.positions[ia + 2];
            const cx = ey * fz - ez * fy;
            const cy = ez * fx - ex * fz;
            const cz = ex * fy - ey * fx;
            const area = 0.5 * Math.sqrt(cx * cx + cy * cy + cz * cz);
            assert.ok(area >= minArea,
                `tri ${i / 3} area ${area} < threshold ${minArea} (sliver)`);
        }
    });

    it('should never fabricate vertex positions', () => {
        // Lossless edge-collapse only re-triangulates among existing vertices;
        // it never moves a vertex or creates a new position. Assert that
        // every output vertex of the merged mesh corresponds bit-exactly to
        // a vertex that exists in the raw MC output.
        const slabLo = 0x000F_000F >>> 0;
        const slabHi = 0x000F_000F >>> 0;
        const bumpLo = ((1 << 4) | (1 << 8) | (1 << 12)) >>> 0;
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(xyzToMorton(0, 0, 0), slabLo, slabHi);
        buffer.addBlock(xyzToMorton(1, 0, 0), slabLo, slabHi);
        buffer.addBlock(xyzToMorton(0, 0, 1), slabLo, slabHi);
        buffer.addBlock(xyzToMorton(1, 0, 1), (slabLo | bumpLo) >>> 0, slabHi);

        const bounds = makeGridBounds(0, 0, 0, 8, 4, 8);
        const raw = marchingCubes(buffer, bounds, 1.0);
        const merged = coplanarMerge(raw, 1.0);

        const rawKeys = new Set();
        for (let i = 0; i < raw.positions.length; i += 3) {
            rawKeys.add(`${raw.positions[i]},${raw.positions[i + 1]},${raw.positions[i + 2]}`);
        }
        let fabricated = 0;
        for (let i = 0; i < merged.positions.length; i += 3) {
            const key = `${merged.positions[i]},${merged.positions[i + 1]},${merged.positions[i + 2]}`;
            if (!rawKeys.has(key)) fabricated++;
        }
        assert.strictEqual(fabricated, 0,
            `merged mesh fabricated ${fabricated} vertex positions not present in raw input`);
    });
});
