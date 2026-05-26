import { DataTable, getSHBands } from '../data-table';
import { type CameraBasis, type Projection } from './camera';
import { RadixSortScratch, radixSortIndicesByFloat } from '../spatial/radix-sort';

/**
 * Number of SH coefficients per color channel for the scene's SH band
 * count. Matches the `f_rest_*` layout in DataTable (channel-major).
 *
 * @param bands - 0–3.
 * @returns Coefficient count per channel.
 */
const numSHCoeffsPerChannel = (bands: number): number => {
    return bands === 0 ? 0 : bands === 1 ? 3 : bands === 2 ? 8 : 15;
};

/**
 * Floats per gaussian in the chunk input buffer that gets uploaded to the
 * GPU project shader. Layout:
 *   [0..2]   pos.xyz
 *   [3..6]   rot.w, rot.x, rot.y, rot.z  (rot_0..rot_3)
 *   [7..9]   log_scale.xyz
 *   [10]     opacity (logit)
 *   [11..13] f_dc.rgb
 *   [14..14+3·N-1]  SH coefficients, channel-major: R[0..N-1], G[0..N-1], B[0..N-1]
 *
 * @param numSHBands - 0–3.
 * @returns Per-gaussian stride in 32-bit floats.
 */
const splatInputStride = (numSHBands: number): number => {
    return 14 + 3 * numSHCoeffsPerChannel(numSHBands);
};

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
 * (front-to-back) so chunked dispatches process splats in correct blend
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
 * @param cols - Pre-resolved column references (only `x`, `y`, `z` are read).
 * @param candidateIndices - Indices into dataTable rows (mutated).
 * @param count - Number of valid entries.
 * @param camera - Camera basis (forward used for pinhole, eye for both).
 * @param projection - Projection mode; selects the depth metric.
 * @param scratch - Reusable scratch buffers, grown on demand.
 */
const sortCandidatesByDepth = (
    cols: SplatColumnRefs,
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
    const ex = camera.eye.x, ey = camera.eye.y, ez = camera.eye.z;

    if (projection === 'pinhole') {
        const fx = camera.forward.x, fy = camera.forward.y, fz = camera.forward.z;
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

/**
 * Cached typed-array references for the columns that `packChunkInput`
 * reads. Built once per render in the orchestrator and reused across
 * every group/chunk to avoid `getColumnByName` lookups and the SH-rest
 * array allocation on the hot path.
 */
type SplatColumnRefs = {
    x: Float32Array;
    y: Float32Array;
    z: Float32Array;
    rotW: Float32Array;
    rotX: Float32Array;
    rotY: Float32Array;
    rotZ: Float32Array;
    scaleX: Float32Array;
    scaleY: Float32Array;
    scaleZ: Float32Array;
    opacity: Float32Array;
    fdcR: Float32Array;
    fdcG: Float32Array;
    fdcB: Float32Array;
    /** Channel-major SH coefficients: indices `[0..N-1]` red, then green, then blue. Empty when `numSHBands === 0`. */
    shRest: Float32Array[];
};

/**
 * Resolve the typed-array references for every column `packChunkInput`
 * touches. Call once per render; reuse the result across all chunks.
 *
 * @param dataTable - Source splat data.
 * @param numSHBands - Scene's SH band count (0–3).
 * @returns Cached column references.
 */
const getSplatColumnRefs = (dataTable: DataTable, numSHBands: number): SplatColumnRefs => {
    const coeffsPerChannel = numSHCoeffsPerChannel(numSHBands);
    const shRest: Float32Array[] = new Array(coeffsPerChannel * 3);
    for (let i = 0; i < coeffsPerChannel * 3; i++) {
        shRest[i] = dataTable.getColumnByName(`f_rest_${i}`)!.data as Float32Array;
    }
    return {
        x: dataTable.getColumnByName('x')!.data as Float32Array,
        y: dataTable.getColumnByName('y')!.data as Float32Array,
        z: dataTable.getColumnByName('z')!.data as Float32Array,
        rotW: dataTable.getColumnByName('rot_0')!.data as Float32Array,
        rotX: dataTable.getColumnByName('rot_1')!.data as Float32Array,
        rotY: dataTable.getColumnByName('rot_2')!.data as Float32Array,
        rotZ: dataTable.getColumnByName('rot_3')!.data as Float32Array,
        scaleX: dataTable.getColumnByName('scale_0')!.data as Float32Array,
        scaleY: dataTable.getColumnByName('scale_1')!.data as Float32Array,
        scaleZ: dataTable.getColumnByName('scale_2')!.data as Float32Array,
        opacity: dataTable.getColumnByName('opacity')!.data as Float32Array,
        fdcR: dataTable.getColumnByName('f_dc_0')!.data as Float32Array,
        fdcG: dataTable.getColumnByName('f_dc_1')!.data as Float32Array,
        fdcB: dataTable.getColumnByName('f_dc_2')!.data as Float32Array,
        shRest
    };
};

/**
 * Pack a chunk's raw splat fields into a flat Float32Array suitable for
 * upload to the GPU project shader. The layout matches `splatInputStride`.
 *
 * @param cols - Pre-resolved column references (build once via `getSplatColumnRefs`).
 * @param chunkIndices - Indices into dataTable for this chunk's splats.
 * @param chunkStart - Offset into `chunkIndices` where the chunk begins.
 * @param chunkSize - Number of splats in this chunk.
 * @param numSHBands - Scene's SH band count (0–3).
 * @param out - Output buffer, length ≥ `chunkSize · stride`.
 */
const packChunkInput = (
    cols: SplatColumnRefs,
    chunkIndices: Uint32Array,
    chunkStart: number,
    chunkSize: number,
    numSHBands: number,
    out: Float32Array
): void => {
    const stride = splatInputStride(numSHBands);
    const numShFloats = numSHCoeffsPerChannel(numSHBands) * 3;
    const {
        x, y, z, rotW, rotX, rotY, rotZ,
        scaleX, scaleY, scaleZ, opacity,
        fdcR, fdcG, fdcB, shRest
    } = cols;

    for (let i = 0; i < chunkSize; i++) {
        const s = chunkIndices[chunkStart + i];
        const base = i * stride;
        out[base + 0] = x[s];
        out[base + 1] = y[s];
        out[base + 2] = z[s];
        out[base + 3] = rotW[s];
        out[base + 4] = rotX[s];
        out[base + 5] = rotY[s];
        out[base + 6] = rotZ[s];
        out[base + 7] = scaleX[s];
        out[base + 8] = scaleY[s];
        out[base + 9] = scaleZ[s];
        out[base + 10] = opacity[s];
        out[base + 11] = fdcR[s];
        out[base + 12] = fdcG[s];
        out[base + 13] = fdcB[s];
        // SH coefficients in channel-major order: f_rest_0..coeffsPerChannel-1 = red, etc.
        for (let k = 0; k < numShFloats; k++) {
            out[base + 14 + k] = shRest[k][s];
        }
    }
};

/**
 * Convenience: derive the scene's SH band count from the DataTable's
 * `f_rest_*` columns. Used by the renderer to size the input stride.
 *
 * @param dataTable - Splat data.
 * @returns SH band count clamped to 0–3.
 */
const sceneSHBands = (dataTable: DataTable): 0 | 1 | 2 | 3 => {
    return getSHBands(dataTable) as 0 | 1 | 2 | 3;
};

export {
    splatInputStride,
    numSHCoeffsPerChannel,
    sortCandidatesByDepth,
    SortScratch,
    getSplatColumnRefs,
    packChunkInput,
    sceneSHBands,
    type SplatColumnRefs
};
