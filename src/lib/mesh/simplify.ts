import { MeshoptSimplifier } from 'meshoptimizer/simplifier';

import type { Mesh } from './marching-cubes';

/**
 * Options controlling mesh simplification behaviour.
 */
type SimplifyOptions = {
    /**
     * If true, use {@link MeshoptSimplifier.simplifySloppy} which is typically
     * several times faster than the standard simplifier but produces lower
     * fidelity geometry. Suitable when the output is used as a coarse
     * collision proxy.
     */
    sloppy?: boolean;
};

// Weight applied to each component of the per-vertex normal attribute when
// driving meshopt's QEM cost. With this weight, collapsing across a 90 degree
// crease costs roughly the same as a full `maxError` of position deviation,
// which keeps real corners intact while still letting the simplifier eat all
// of the coplanar redundancy that marching cubes leaves behind on flat
// surfaces.
const NORMAL_WEIGHT = 1.0;

/**
 * Compute area-weighted per-vertex normals for a triangle mesh.
 *
 * Iterates the triangle list once, accumulating each face's (un-normalised)
 * cross product into its three vertices. Because the cross product magnitude
 * equals 2x the triangle area, the contribution is naturally area-weighted
 * without explicit normalisation per face. Each vertex normal is normalised
 * once at the end.
 *
 * @param positions - Vertex positions (3 floats per vertex)
 * @param indices - Triangle indices (3 indices per triangle)
 * @returns Tightly-packed vertex normals (3 floats per vertex)
 */
const computeVertexNormals = (positions: Float32Array, indices: Uint32Array): Float32Array => {
    const vertexCount = positions.length / 3;
    const normals = new Float32Array(vertexCount * 3);

    for (let i = 0; i < indices.length; i += 3) {
        const ia = indices[i] * 3;
        const ib = indices[i + 1] * 3;
        const ic = indices[i + 2] * 3;

        const ax = positions[ia], ay = positions[ia + 1], az = positions[ia + 2];
        const bx = positions[ib], by = positions[ib + 1], bz = positions[ib + 2];
        const cx = positions[ic], cy = positions[ic + 1], cz = positions[ic + 2];

        const ux = bx - ax, uy = by - ay, uz = bz - az;
        const vx = cx - ax, vy = cy - ay, vz = cz - az;

        const nx = uy * vz - uz * vy;
        const ny = uz * vx - ux * vz;
        const nz = ux * vy - uy * vx;

        normals[ia] += nx; normals[ia + 1] += ny; normals[ia + 2] += nz;
        normals[ib] += nx; normals[ib + 1] += ny; normals[ib + 2] += nz;
        normals[ic] += nx; normals[ic + 1] += ny; normals[ic + 2] += nz;
    }

    for (let i = 0; i < normals.length; i += 3) {
        const nx = normals[i], ny = normals[i + 1], nz = normals[i + 2];
        const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
        if (len > 0) {
            const inv = 1 / len;
            normals[i] = nx * inv;
            normals[i + 1] = ny * inv;
            normals[i + 2] = nz * inv;
        }
    }

    return normals;
};

/**
 * Simplify a triangle mesh using meshoptimizer.
 *
 * Reduces the triangle count while keeping geometric error within the
 * specified threshold. The non-sloppy path feeds per-vertex normals as an
 * attribute so coplanar (flush) collapses are exhausted before the simplifier
 * touches edges with real curvature - the practical effect is that flat
 * regions of marching-cubes output are flushed first, while corners and
 * features survive. Vertex positions are compacted so only referenced
 * vertices remain in the output.
 *
 * @param mesh - Input mesh with positions and indices.
 * @param maxError - Maximum geometric error (absolute world-space distance).
 * @param options - Optional simplification tuning.
 * @returns A new mesh with simplified geometry.
 */
const simplifyMesh = async (mesh: Mesh, maxError: number, options?: SimplifyOptions): Promise<Mesh> => {
    await MeshoptSimplifier.ready;

    let simplifiedIndices: Uint32Array;
    if (options?.sloppy) {
        simplifiedIndices = MeshoptSimplifier.simplifySloppy(
            mesh.indices, mesh.positions, 3, null, 0, maxError
        )[0];
    } else {
        const normals = computeVertexNormals(mesh.positions, mesh.indices);
        simplifiedIndices = MeshoptSimplifier.simplifyWithAttributes(
            mesh.indices,
            mesh.positions,
            3,
            normals,
            3,
            [NORMAL_WEIGHT, NORMAL_WEIGHT, NORMAL_WEIGHT],
            null,
            0,
            maxError,
            ['ErrorAbsolute', 'Prune']
        )[0];
    }

    // Compact vertices using meshopt's native remap. The returned remap maps
    // old vertex index -> new vertex index (or 0xFFFFFFFF for unreferenced
    // vertices) and rewrites simplifiedIndices in place.
    const [remap, uniqueCount] = MeshoptSimplifier.compactMesh(simplifiedIndices);

    const compactPositions = new Float32Array(uniqueCount * 3);
    const sourcePositions = mesh.positions;
    const oldVertexCount = remap.length;
    for (let oldIdx = 0; oldIdx < oldVertexCount; oldIdx++) {
        const newIdx = remap[oldIdx];
        if (newIdx === 0xFFFFFFFF) continue;
        const src = oldIdx * 3;
        const dst = newIdx * 3;
        compactPositions[dst]     = sourcePositions[src];
        compactPositions[dst + 1] = sourcePositions[src + 1];
        compactPositions[dst + 2] = sourcePositions[src + 2];
    }

    return {
        positions: compactPositions,
        indices: simplifiedIndices
    };
};

export { simplifyMesh, type SimplifyOptions };
