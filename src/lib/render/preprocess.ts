import { DataTable, getSHBands } from '../data-table';
import { type CameraBasis } from './camera';

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
 * Sort `candidateIndices[0..count)` by camera-space depth ascending
 * (front-to-back) so chunked dispatches process splats in correct blend
 * order. Mutates `candidateIndices` in place.
 *
 * Depth is `forward · (pos − eye)` — no projection needed.
 *
 * @param dataTable - Splat data in PlayCanvas-identity space.
 * @param candidateIndices - Indices into dataTable rows (mutated).
 * @param count - Number of valid entries.
 * @param camera - Camera basis (only forward + eye used).
 */
const sortCandidatesByDepth = (
    dataTable: DataTable,
    candidateIndices: Uint32Array,
    count: number,
    camera: CameraBasis
): void => {
    if (count < 2) return;
    const x = dataTable.getColumnByName('x')!.data;
    const y = dataTable.getColumnByName('y')!.data;
    const z = dataTable.getColumnByName('z')!.data;
    const fx = camera.forward.x, fy = camera.forward.y, fz = camera.forward.z;
    const ex = camera.eye.x, ey = camera.eye.y, ez = camera.eye.z;

    const depth = new Float32Array(count);
    for (let i = 0; i < count; i++) {
        const s = candidateIndices[i];
        depth[i] = fx * (x[s] - ex) + fy * (y[s] - ey) + fz * (z[s] - ez);
    }

    const orderArr = new Array(count);
    for (let i = 0; i < count; i++) orderArr[i] = i;
    orderArr.sort((a, b) => depth[a] - depth[b]);

    const tmp = new Uint32Array(count);
    for (let i = 0; i < count; i++) tmp[i] = candidateIndices[orderArr[i]];
    candidateIndices.set(tmp.subarray(0, count));
};

/**
 * Pack a chunk's raw splat fields into a flat Float32Array suitable for
 * upload to the GPU project shader. The layout matches `splatInputStride`.
 *
 * @param dataTable - Source splat data.
 * @param chunkIndices - Indices into dataTable for this chunk's splats.
 * @param chunkStart - Offset into `chunkIndices` where the chunk begins.
 * @param chunkSize - Number of splats in this chunk.
 * @param numSHBands - Scene's SH band count (0–3).
 * @param out - Output buffer, length ≥ `chunkSize · stride`.
 */
const packChunkInput = (
    dataTable: DataTable,
    chunkIndices: Uint32Array,
    chunkStart: number,
    chunkSize: number,
    numSHBands: number,
    out: Float32Array
): void => {
    const stride = splatInputStride(numSHBands);
    const coeffsPerChannel = numSHCoeffsPerChannel(numSHBands);

    const x = dataTable.getColumnByName('x')!.data;
    const y = dataTable.getColumnByName('y')!.data;
    const z = dataTable.getColumnByName('z')!.data;
    const rotW = dataTable.getColumnByName('rot_0')!.data;
    const rotX = dataTable.getColumnByName('rot_1')!.data;
    const rotY = dataTable.getColumnByName('rot_2')!.data;
    const rotZ = dataTable.getColumnByName('rot_3')!.data;
    const scaleX = dataTable.getColumnByName('scale_0')!.data;
    const scaleY = dataTable.getColumnByName('scale_1')!.data;
    const scaleZ = dataTable.getColumnByName('scale_2')!.data;
    const opacity = dataTable.getColumnByName('opacity')!.data;
    const fdcR = dataTable.getColumnByName('f_dc_0')!.data;
    const fdcG = dataTable.getColumnByName('f_dc_1')!.data;
    const fdcB = dataTable.getColumnByName('f_dc_2')!.data;

    const shRest: Float32Array[] = [];
    for (let i = 0; i < coeffsPerChannel * 3; i++) {
        shRest.push(dataTable.getColumnByName(`f_rest_${i}`)!.data as Float32Array);
    }

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
        for (let k = 0; k < coeffsPerChannel * 3; k++) {
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
    packChunkInput,
    sceneSHBands
};
