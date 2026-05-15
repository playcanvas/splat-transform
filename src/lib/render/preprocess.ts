import { DataTable, getSHBands } from '../data-table';
import { type CameraBasis } from './camera';

const SH_C0 = 0.28209479177387814;

const SH_C1 = 0.4886025119029199;

const SH_C2_0 = 1.0925484305920792;
const SH_C2_1 = -1.0925484305920792;
const SH_C2_2 = 0.31539156525252005;
const SH_C2_3 = -1.0925484305920792;
const SH_C2_4 = 0.5462742152960396;

const SH_C3_0 = -0.5900435899266435;
const SH_C3_1 = 2.890611442640554;
const SH_C3_2 = -0.4570457994644658;
const SH_C3_3 = 0.3731763325901154;
const SH_C3_4 = -0.4570457994644658;
const SH_C3_5 = 1.445305721320277;
const SH_C3_6 = -0.5900435899266435;

/**
 * Stride of the GPU record buffer in 32-bit elements. Matches the WGSL
 * unpacking in rasterize.wgsl which reads three vec4s per record:
 *   [cx, cy, covInvA, covInvB] [covInvC, r, g, b] [alpha, _, _, _].
 */
const RECORD_STRIDE_F32 = 12;

/** Tile size in pixels along each axis. Workgroup size in WGSL must match. */
const TILE_SIZE = 16;

/**
 * Output of the CPU preprocess: the data to upload to the GPU plus the
 * per-tile range table the rasterizer indexes into.
 */
type PreprocessResult = {
    /** Packed record buffer, length = numRecords * RECORD_STRIDE_F32. */
    records: Float32Array;
    /** Per-tile [start, end) pairs as a flat Uint32Array of length 2*numTiles. */
    tileRanges: Uint32Array;
    /** Number of records (gaussian × overlapped tile). */
    numRecords: number;
    /** Number of tiles along X. */
    tilesX: number;
    /** Number of tiles along Y. */
    tilesY: number;
    /** Number of visible (post-cull) gaussians that contributed at least one record. */
    numVisible: number;
};

/**
 * Project every gaussian in `dataTable` through the camera, evaluate
 * view-dependent color, derive a screen-space 2D covariance and footprint,
 * and bin into tiles. Records within each tile are sorted by depth
 * ascending so the rasterizer can iterate front-to-back.
 *
 * All math runs on the CPU — no GraphicsDevice is touched here.
 *
 * @param dataTable - Gaussian splat data with the standard columns.
 * @param camera - Pre-built camera basis (right, down, forward + focals).
 * @param width - Output image width in pixels.
 * @param height - Output image height in pixels.
 * @param near - Near clip distance; splats with camera-space depth <= near are culled.
 * @returns Packed records, per-tile ranges and tile dimensions ready for GPU upload.
 */
const preprocess = (
    dataTable: DataTable,
    camera: CameraBasis,
    width: number,
    height: number,
    near: number
): PreprocessResult => {
    const numSplats = dataTable.numRows;

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
    const fdcR = dataTable.getColumnByName('f_dc_0')!.data;
    const fdcG = dataTable.getColumnByName('f_dc_1')!.data;
    const fdcB = dataTable.getColumnByName('f_dc_2')!.data;
    const opacity = dataTable.getColumnByName('opacity')!.data;

    const shBands = getSHBands(dataTable);
    const coeffsPerChannel = shBands === 0 ? 0 : shBands === 1 ? 3 : shBands === 2 ? 8 : 15;
    const shRest: Float32Array[] = [];
    if (shBands > 0) {
        for (let i = 0; i < coeffsPerChannel * 3; i++) {
            shRest.push(dataTable.getColumnByName(`f_rest_${i}`)!.data as Float32Array);
        }
    }

    const rx = camera.right.x, ry = camera.right.y, rz = camera.right.z;
    const dx = camera.down.x, dy = camera.down.y, dz = camera.down.z;
    const fx = camera.forward.x, fy = camera.forward.y, fz = camera.forward.z;
    const ex = camera.eye.x, ey = camera.eye.y, ez = camera.eye.z;
    const focalX = camera.focalX;
    const focalY = camera.focalY;
    const cx0 = width * 0.5;
    const cy0 = height * 0.5;

    const tilesX = Math.ceil(width / TILE_SIZE);
    const tilesY = Math.ceil(height / TILE_SIZE);
    const numTiles = tilesX * tilesY;

    const splatCenter = new Float32Array(numSplats * 2);
    const splatCovInv = new Float32Array(numSplats * 3);
    const splatColor = new Float32Array(numSplats * 3);
    const splatAlpha = new Float32Array(numSplats);
    const splatDepth = new Float32Array(numSplats);
    const splatTileBounds = new Int32Array(numSplats * 4);

    const lowpass = 0.3;

    let numVisible = 0;

    for (let i = 0; i < numSplats; i++) {
        splatTileBounds[i * 4 + 0] = 0;
        splatTileBounds[i * 4 + 1] = 0;
        splatTileBounds[i * 4 + 2] = -1;
        splatTileBounds[i * 4 + 3] = -1;

        const px = x[i], py = y[i], pz = z[i];

        // World → camera (V = [right; down; forward] basis, translation -V*eye)
        const wx = px - ex, wy = py - ey, wz = pz - ez;
        const cx = rx * wx + ry * wy + rz * wz;
        const cy = dx * wx + dy * wy + dz * wz;
        const cz = fx * wx + fy * wy + fz * wz;

        if (cz <= near) continue;

        // Project
        const invZ = 1.0 / cz;
        const screenX = focalX * cx * invZ + cx0;
        const screenY = focalY * cy * invZ + cy0;

        // Quaternion (qw, qx, qy, qz), normalized in case input wasn't.
        let qw = rotW[i], qx = rotX[i], qy = rotY[i], qz = rotZ[i];
        const qlen = Math.hypot(qw, qx, qy, qz);
        if (qlen === 0) continue;
        const invQ = 1.0 / qlen;
        qw *= invQ; qx *= invQ; qy *= invQ; qz *= invQ;

        // Rotation matrix R (world-space) from quaternion.
        const xx = qx * qx, yy = qy * qy, zz = qz * qz;
        const xy = qx * qy, xzq = qx * qz, yz = qy * qz;
        const wxq = qw * qx, wy_ = qw * qy, wzq = qw * qz;
        const r00 = 1 - 2 * (yy + zz);
        const r01 = 2 * (xy - wzq);
        const r02 = 2 * (xzq + wy_);
        const r10 = 2 * (xy + wzq);
        const r11 = 1 - 2 * (xx + zz);
        const r12 = 2 * (yz - wxq);
        const r20 = 2 * (xzq - wy_);
        const r21 = 2 * (yz + wxq);
        const r22 = 1 - 2 * (xx + yy);

        // Scale (exp of log-scale).
        const sx = Math.exp(scaleX[i]);
        const sy = Math.exp(scaleY[i]);
        const sz = Math.exp(scaleZ[i]);

        // M = R * diag(s). Σ_3d = M * M^T.
        const m00 = r00 * sx, m01 = r01 * sy, m02 = r02 * sz;
        const m10 = r10 * sx, m11 = r11 * sy, m12 = r12 * sz;
        const m20 = r20 * sx, m21 = r21 * sy, m22 = r22 * sz;

        const sig00 = m00 * m00 + m01 * m01 + m02 * m02;
        const sig01 = m00 * m10 + m01 * m11 + m02 * m12;
        const sig02 = m00 * m20 + m01 * m21 + m02 * m22;
        const sig11 = m10 * m10 + m11 * m11 + m12 * m12;
        const sig12 = m10 * m20 + m11 * m21 + m12 * m22;
        const sig22 = m20 * m20 + m21 * m21 + m22 * m22;

        // Transform Σ_3d to camera space: Σ_cam = V * Σ_3d * V^T, where V's
        // rows are (right, down, forward).
        const v00 = rx, v01 = ry, v02 = rz;
        const v10 = dx, v11 = dy, v12 = dz;
        const v20 = fx, v21 = fy, v22 = fz;

        // T = V * Σ_3d (3x3)
        const t00 = v00 * sig00 + v01 * sig01 + v02 * sig02;
        const t01 = v00 * sig01 + v01 * sig11 + v02 * sig12;
        const t02 = v00 * sig02 + v01 * sig12 + v02 * sig22;
        const t10 = v10 * sig00 + v11 * sig01 + v12 * sig02;
        const t11 = v10 * sig01 + v11 * sig11 + v12 * sig12;
        const t12 = v10 * sig02 + v11 * sig12 + v12 * sig22;
        const t20 = v20 * sig00 + v21 * sig01 + v22 * sig02;
        const t21 = v20 * sig01 + v21 * sig11 + v22 * sig12;
        const t22 = v20 * sig02 + v21 * sig12 + v22 * sig22;

        // Σ_cam = T * V^T (we only need entries (0,0), (0,1), (0,2), (1,1), (1,2), (2,2))
        const c00 = t00 * v00 + t01 * v01 + t02 * v02;
        const c01 = t00 * v10 + t01 * v11 + t02 * v12;
        const c02 = t00 * v20 + t01 * v21 + t02 * v22;
        const c11 = t10 * v10 + t11 * v11 + t12 * v12;
        const c12 = t10 * v20 + t11 * v21 + t12 * v22;
        const c22 = t20 * v20 + t21 * v21 + t22 * v22;

        // 2D covariance via Jacobian J = [[fx/z, 0, -fx*x/z^2], [0, fy/z, -fy*y/z^2]]
        const jx0 = focalX * invZ;
        const jx2 = -focalX * cx * invZ * invZ;
        const jy1 = focalY * invZ;
        const jy2 = -focalY * cy * invZ * invZ;

        // Σ_2d = J * Σ_cam * J^T (entries before low-pass).
        // Row 0 of J times Σ_cam:
        const u00 = jx0 * c00 + jx2 * c02;
        const u01 = jx0 * c01 + jx2 * c12;
        const u02 = jx0 * c02 + jx2 * c22;
        const u10 = jy1 * c01 + jy2 * c02;
        const u11 = jy1 * c11 + jy2 * c12;
        const u12 = jy1 * c12 + jy2 * c22;

        let cov00 = u00 * jx0 + u02 * jx2;
        const cov01 = u00 * 0 + u01 * jy1 + u02 * jy2;
        let cov11 = u11 * jy1 + u12 * jy2;

        // Low-pass anti-aliasing filter (3DGS reference uses 0.3).
        cov00 += lowpass;
        cov11 += lowpass;

        const det = cov00 * cov11 - cov01 * cov01;
        if (det <= 0) continue;

        const invDet = 1.0 / det;
        const covInvA = cov11 * invDet;
        const covInvB = -cov01 * invDet;
        const covInvC = cov00 * invDet;

        // 3σ screen-space radius from max eigenvalue.
        const mid = 0.5 * (cov00 + cov11);
        const disc = Math.sqrt(Math.max(0.1, mid * mid - det));
        const lambdaMax = mid + disc;
        const radius = Math.ceil(3 * Math.sqrt(lambdaMax));

        const minX = Math.floor((screenX - radius) / TILE_SIZE);
        const maxX = Math.floor((screenX + radius) / TILE_SIZE);
        const minY = Math.floor((screenY - radius) / TILE_SIZE);
        const maxY = Math.floor((screenY + radius) / TILE_SIZE);

        const minTX = Math.max(0, minX);
        const minTY = Math.max(0, minY);
        const maxTX = Math.min(tilesX - 1, maxX);
        const maxTY = Math.min(tilesY - 1, maxY);

        if (maxTX < minTX || maxTY < minTY) continue;

        // View direction (camera → splat) in world space for SH eval.
        const dpx = px - ex, dpy = py - ey, dpz = pz - ez;
        const dLen = Math.hypot(dpx, dpy, dpz) || 1;
        const dirX = dpx / dLen;
        const dirY = dpy / dLen;
        const dirZ = dpz / dLen;

        let cr = SH_C0 * fdcR[i];
        let cg = SH_C0 * fdcG[i];
        let cb = SH_C0 * fdcB[i];

        if (shBands >= 1) {
            const n = coeffsPerChannel;
            // Channel-major: red = shRest[0..n-1], green = shRest[n..2n-1], blue = shRest[2n..3n-1].
            const b0 = -SH_C1 * dirY;
            const b1 = SH_C1 * dirZ;
            const b2 = -SH_C1 * dirX;
            cr += b0 * shRest[0][i] + b1 * shRest[1][i] + b2 * shRest[2][i];
            cg += b0 * shRest[n][i] + b1 * shRest[n + 1][i] + b2 * shRest[n + 2][i];
            cb += b0 * shRest[2 * n][i] + b1 * shRest[2 * n + 1][i] + b2 * shRest[2 * n + 2][i];

            if (shBands >= 2) {
                const xx2 = dirX * dirX, yy2 = dirY * dirY, zz2 = dirZ * dirZ;
                const xy2 = dirX * dirY, yz2 = dirY * dirZ, xz2 = dirX * dirZ;
                const b3 = SH_C2_0 * xy2;
                const b4 = SH_C2_1 * yz2;
                const b5 = SH_C2_2 * (2 * zz2 - xx2 - yy2);
                const b6 = SH_C2_3 * xz2;
                const b7 = SH_C2_4 * (xx2 - yy2);
                cr += b3 * shRest[3][i] + b4 * shRest[4][i] + b5 * shRest[5][i] + b6 * shRest[6][i] + b7 * shRest[7][i];
                cg += b3 * shRest[n + 3][i] + b4 * shRest[n + 4][i] + b5 * shRest[n + 5][i] + b6 * shRest[n + 6][i] + b7 * shRest[n + 7][i];
                cb += b3 * shRest[2 * n + 3][i] + b4 * shRest[2 * n + 4][i] + b5 * shRest[2 * n + 5][i] + b6 * shRest[2 * n + 6][i] + b7 * shRest[2 * n + 7][i];

                if (shBands >= 3) {
                    const b8 = SH_C3_0 * dirY * (3 * xx2 - yy2);
                    const b9 = SH_C3_1 * xy2 * dirZ;
                    const b10 = SH_C3_2 * dirY * (4 * zz2 - xx2 - yy2);
                    const b11 = SH_C3_3 * dirZ * (2 * zz2 - 3 * xx2 - 3 * yy2);
                    const b12 = SH_C3_4 * dirX * (4 * zz2 - xx2 - yy2);
                    const b13 = SH_C3_5 * dirZ * (xx2 - yy2);
                    const b14 = SH_C3_6 * dirX * (xx2 - 3 * yy2);
                    cr += b8 * shRest[8][i] + b9 * shRest[9][i] + b10 * shRest[10][i] + b11 * shRest[11][i] +
                          b12 * shRest[12][i] + b13 * shRest[13][i] + b14 * shRest[14][i];
                    cg += b8 * shRest[n + 8][i] + b9 * shRest[n + 9][i] + b10 * shRest[n + 10][i] + b11 * shRest[n + 11][i] +
                          b12 * shRest[n + 12][i] + b13 * shRest[n + 13][i] + b14 * shRest[n + 14][i];
                    cb += b8 * shRest[2 * n + 8][i] + b9 * shRest[2 * n + 9][i] + b10 * shRest[2 * n + 10][i] + b11 * shRest[2 * n + 11][i] +
                          b12 * shRest[2 * n + 12][i] + b13 * shRest[2 * n + 13][i] + b14 * shRest[2 * n + 14][i];
                }
            }
        }

        // Add the constant offset and clamp. 3DGS convention: stored f_dc
        // is centered around 0 (encoded value); reconstructed color is
        // SH_C0*f_dc + 0.5 + view-dependent terms.
        const colR = Math.max(0, cr + 0.5);
        const colG = Math.max(0, cg + 0.5);
        const colB = Math.max(0, cb + 0.5);

        // Sigmoid of opacity.
        const alpha = 1.0 / (1.0 + Math.exp(-opacity[i]));

        splatCenter[i * 2 + 0] = screenX;
        splatCenter[i * 2 + 1] = screenY;
        splatCovInv[i * 3 + 0] = covInvA;
        splatCovInv[i * 3 + 1] = covInvB;
        splatCovInv[i * 3 + 2] = covInvC;
        splatColor[i * 3 + 0] = colR;
        splatColor[i * 3 + 1] = colG;
        splatColor[i * 3 + 2] = colB;
        splatAlpha[i] = alpha;
        splatDepth[i] = cz;
        splatTileBounds[i * 4 + 0] = minTX;
        splatTileBounds[i * 4 + 1] = minTY;
        splatTileBounds[i * 4 + 2] = maxTX;
        splatTileBounds[i * 4 + 3] = maxTY;

        numVisible++;
    }

    // Count records per tile.
    const tileCount = new Uint32Array(numTiles);
    let numRecords = 0;
    for (let i = 0; i < numSplats; i++) {
        const minTX = splatTileBounds[i * 4 + 0];
        const minTY = splatTileBounds[i * 4 + 1];
        const maxTX = splatTileBounds[i * 4 + 2];
        const maxTY = splatTileBounds[i * 4 + 3];
        if (maxTX < minTX || maxTY < minTY) continue;
        for (let ty = minTY; ty <= maxTY; ty++) {
            for (let tx = minTX; tx <= maxTX; tx++) {
                tileCount[ty * tilesX + tx]++;
                numRecords++;
            }
        }
    }

    // Prefix sum → tile offsets.
    const tileOffsets = new Uint32Array(numTiles + 1);
    let running = 0;
    for (let t = 0; t < numTiles; t++) {
        tileOffsets[t] = running;
        running += tileCount[t];
    }
    tileOffsets[numTiles] = running;

    // Emit records into per-tile bins (unsorted within tile yet).
    const recordSplatIdx = new Uint32Array(numRecords);
    const recordDepth = new Float32Array(numRecords);
    const tileCursor = new Uint32Array(numTiles);

    for (let i = 0; i < numSplats; i++) {
        const minTX = splatTileBounds[i * 4 + 0];
        const minTY = splatTileBounds[i * 4 + 1];
        const maxTX = splatTileBounds[i * 4 + 2];
        const maxTY = splatTileBounds[i * 4 + 3];
        if (maxTX < minTX || maxTY < minTY) continue;
        const d = splatDepth[i];
        for (let ty = minTY; ty <= maxTY; ty++) {
            for (let tx = minTX; tx <= maxTX; tx++) {
                const t = ty * tilesX + tx;
                const pos = tileOffsets[t] + tileCursor[t];
                recordSplatIdx[pos] = i;
                recordDepth[pos] = d;
                tileCursor[t]++;
            }
        }
    }

    // Per-tile sort by depth ascending. Sort the splat indices using a
    // scratch index array, then permute.
    const scratch: number[] = [];
    for (let t = 0; t < numTiles; t++) {
        const start = tileOffsets[t];
        const end = tileOffsets[t + 1];
        const count = end - start;
        if (count < 2) continue;
        scratch.length = count;
        for (let k = 0; k < count; k++) scratch[k] = k;
        scratch.sort((a, b) => recordDepth[start + a] - recordDepth[start + b]);

        const tmpSplat = new Uint32Array(count);
        const tmpDepth = new Float32Array(count);
        for (let k = 0; k < count; k++) {
            tmpSplat[k] = recordSplatIdx[start + scratch[k]];
            tmpDepth[k] = recordDepth[start + scratch[k]];
        }
        recordSplatIdx.set(tmpSplat, start);
        recordDepth.set(tmpDepth, start);
    }

    // Emit final GPU records by walking the sorted splat indices.
    const records = new Float32Array(numRecords * RECORD_STRIDE_F32);
    for (let r = 0; r < numRecords; r++) {
        const sIdx = recordSplatIdx[r];
        const base = r * RECORD_STRIDE_F32;
        records[base + 0] = splatCenter[sIdx * 2 + 0];
        records[base + 1] = splatCenter[sIdx * 2 + 1];
        records[base + 2] = splatCovInv[sIdx * 3 + 0];
        records[base + 3] = splatCovInv[sIdx * 3 + 1];
        records[base + 4] = splatCovInv[sIdx * 3 + 2];
        records[base + 5] = splatColor[sIdx * 3 + 0];
        records[base + 6] = splatColor[sIdx * 3 + 1];
        records[base + 7] = splatColor[sIdx * 3 + 2];
        records[base + 8] = splatAlpha[sIdx];
        // records[base + 9..11] left as 0 padding.
    }

    // Build flat tileRanges as [start0, end0, start1, end1, ...].
    const tileRanges = new Uint32Array(numTiles * 2);
    for (let t = 0; t < numTiles; t++) {
        tileRanges[t * 2 + 0] = tileOffsets[t];
        tileRanges[t * 2 + 1] = tileOffsets[t + 1];
    }

    return { records, tileRanges, numRecords, tilesX, tilesY, numVisible };
};

export { preprocess, RECORD_STRIDE_F32, TILE_SIZE, type PreprocessResult };
