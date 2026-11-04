import { RadixSortScratch, radixSortIndicesByFloat } from '../spatial/radix-sort';

import type { CameraBasis, Projection } from './camera';

/**
 * Reusable scratch for `sortCandidatesByDepth`. Bundles the parallel-keys
 * Float32 depth buffer with the shared `RadixSortScratch`. One instance is
 * created per render in the orchestrator and reused across every group's
 * sort to avoid allocating multi-MB typed arrays on every call. Buffers
 * grow on demand and never shrink.
 */
class SortScratch {
    /** Float32 depths in candidate order; parallel to the indices being sorted. */
    depth: Float32Array;
    /** Shared radix-sort working buffers. */
    radix: RadixSortScratch;

    constructor() {
        this.depth = new Float32Array(0);
        this.radix = new RadixSortScratch();
    }

    ensure(count: number): void {
        if (count > this.depth.length) {
            this.depth = new Float32Array(count);
        }
        this.radix.ensure(count);
    }
}

/**
 * Sort `candidateIndices[0..count)` by view depth ascending
 * (front-to-back) so splats are streamed in correct blend
 * order. Mutates `candidateIndices` in place.
 *
 * Depth metric depends on the projection. Pinhole uses
 * `forward · (pos − eye)` (camera-space z). Equirect uses radial
 * distance `‖pos − eye‖` — the natural front-to-back ordering for a
 * spherical projection, since "in front of" is defined per direction
 * rather than per camera-z plane. Delegates the actual sort to the
 * shared `radixSortIndicesByFloat`, providing depths as the parallel
 * Float32 keys.
 *
 * @param cols - Position columns (`x`, `y`, `z`).
 * @param candidateIndices - Indices into the scene rows (mutated).
 * @param count - Number of valid entries.
 * @param camera - Camera basis (forward used for pinhole, eye for both).
 * @param projection - Projection mode; selects the depth metric.
 * @param scratch - Reusable scratch buffers, grown on demand.
 */
const sortCandidatesByDepth = (
    cols: { x: Float32Array; y: Float32Array; z: Float32Array },
    candidateIndices: Uint32Array,
    count: number,
    camera: CameraBasis,
    projection: Projection,
    scratch: SortScratch
): void => {
    if (count < 2) return;
    scratch.ensure(count);
    const { x, y, z } = cols;
    const { depth, radix } = scratch;
    const ex = camera.eye.x,
        ey = camera.eye.y,
        ez = camera.eye.z;

    if (projection === 'pinhole') {
        const fx = camera.forward.x,
            fy = camera.forward.y,
            fz = camera.forward.z;
        for (let i = 0; i < count; i++) {
            const s = candidateIndices[i];
            depth[i] = fx * (x[s] - ex) + fy * (y[s] - ey) + fz * (z[s] - ez);
        }
    } else {
        // Equirect: radial squared distance from the camera. r² is
        // monotonic in r (all non-negative), so sorting on r² gives the
        // same front-to-back order as sorting on r — saves the sqrt.
        for (let i = 0; i < count; i++) {
            const s = candidateIndices[i];
            const dx = x[s] - ex;
            const dy = y[s] - ey;
            const dz = z[s] - ez;
            depth[i] = dx * dx + dy * dy + dz * dz;
        }
    }

    radixSortIndicesByFloat(candidateIndices, depth, count, radix);
};

export { sortCandidatesByDepth, SortScratch };
