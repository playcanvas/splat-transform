import { MeshoptSimplifier } from 'meshoptimizer/simplifier';

import type { Mesh } from './marching-cubes';

/**
 * Simplify a triangle mesh using meshoptimizer.
 *
 * Reduces the triangle count while keeping geometric error within the
 * specified threshold. Vertex positions are compacted so only referenced
 * vertices remain in the output.
 *
 * @param mesh - Input mesh with positions and indices.
 * @param maxError - Maximum geometric error (absolute world-space distance).
 * @returns A new mesh with simplified geometry.
 */
const simplifyMesh = async (mesh: Mesh, maxError: number): Promise<Mesh> => {
    await MeshoptSimplifier.ready;

    const [simplifiedIndices] = MeshoptSimplifier.simplify(
        mesh.indices,
        mesh.positions,
        3,
        0,
        maxError,
        ['ErrorAbsolute']
    );

    const vertexRemap = new Map<number, number>();
    let newVertexCount = 0;
    for (let i = 0; i < simplifiedIndices.length; i++) {
        if (!vertexRemap.has(simplifiedIndices[i])) {
            vertexRemap.set(simplifiedIndices[i], newVertexCount++);
        }
    }

    const compactPositions = new Float32Array(newVertexCount * 3);
    for (const [oldIdx, newIdx] of vertexRemap) {
        compactPositions[newIdx * 3] = mesh.positions[oldIdx * 3];
        compactPositions[newIdx * 3 + 1] = mesh.positions[oldIdx * 3 + 1];
        compactPositions[newIdx * 3 + 2] = mesh.positions[oldIdx * 3 + 2];
    }

    const compactIndices = new Uint32Array(simplifiedIndices.length);
    for (let i = 0; i < simplifiedIndices.length; i++) {
        compactIndices[i] = vertexRemap.get(simplifiedIndices[i])!;
    }

    return {
        positions: compactPositions,
        indices: compactIndices
    };
};

export { simplifyMesh };
