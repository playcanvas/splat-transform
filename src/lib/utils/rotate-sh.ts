import type { Mat3 } from 'playcanvas';

const kSqrt03_02 = Math.sqrt(3.0 / 2.0);
const kSqrt01_03 = Math.sqrt(1.0 / 3.0);
const kSqrt02_03 = Math.sqrt(2.0 / 3.0);
const kSqrt04_03 = Math.sqrt(4.0 / 3.0);
const kSqrt01_04 = Math.sqrt(1.0 / 4.0);
const kSqrt03_04 = Math.sqrt(3.0 / 4.0);
const kSqrt01_05 = Math.sqrt(1.0 / 5.0);
const kSqrt03_05 = Math.sqrt(3.0 / 5.0);
const kSqrt06_05 = Math.sqrt(6.0 / 5.0);
const kSqrt08_05 = Math.sqrt(8.0 / 5.0);
const kSqrt09_05 = Math.sqrt(9.0 / 5.0);
const kSqrt01_06 = Math.sqrt(1.0 / 6.0);
const kSqrt05_06 = Math.sqrt(5.0 / 6.0);
const kSqrt03_08 = Math.sqrt(3.0 / 8.0);
const kSqrt05_08 = Math.sqrt(5.0 / 8.0);
const kSqrt09_08 = Math.sqrt(9.0 / 8.0);
const kSqrt05_09 = Math.sqrt(5.0 / 9.0);
const kSqrt08_09 = Math.sqrt(8.0 / 9.0);
const kSqrt01_10 = Math.sqrt(1.0 / 10.0);
const kSqrt03_10 = Math.sqrt(3.0 / 10.0);
const kSqrt01_12 = Math.sqrt(1.0 / 12.0);
const kSqrt04_15 = Math.sqrt(4.0 / 15.0);
const kSqrt01_16 = Math.sqrt(1.0 / 16.0);
const kSqrt15_16 = Math.sqrt(15.0 / 16.0);
const kSqrt01_18 = Math.sqrt(1.0 / 18.0);
const kSqrt01_60 = Math.sqrt(1.0 / 60.0);

const dp = (n: number, start: number, a: number[] | Float32Array, b: number[] | Float32Array) => {
    let sum = 0;
    for (let i = 0; i < n; i++) {
        sum += a[start + i] * b[i];
    }
    return sum;
};

const coeffsIn = new Float32Array(24);

// ---- band 4 -------------------------------------------------------------
//
// sh-lib stops at band 3, and its hand-unrolled recurrence does not extend
// cleanly, so band 4 is computed by projection instead: the rotated
// coefficients c' satisfy sum_m c'_m Y_m(d) = sum_m c_m Y_m(R^T d) for every
// direction d, which a least-squares fit over a fixed set of directions solves
// exactly (band 4 is a 9-dimensional space, closed under rotation). Y is the
// degree-4 basis renderers evaluate (3DGS / gsplat sh_utils; constants
// omitted — they scale rows and columns equally and cancel). The same
// projection reproduces the band 1-3 matrices above, which the tests check.

const BAND4_DIRS = 64;

// Degree-4 real SH basis at unit direction (x, y, z), without constants.
const band4Basis = (x: number, y: number, z: number, out: Float64Array, o: number) => {
    const xx = x * x,
        yy = y * y,
        zz = z * z;
    out[o] = 2.5033429417967046 * x * y * (xx - yy);
    out[o + 1] = -1.7701307697799304 * y * z * (3 * xx - yy);
    out[o + 2] = 0.9461746957575601 * x * y * (7 * zz - 1);
    out[o + 3] = -0.6690465435572892 * y * z * (7 * zz - 3);
    out[o + 4] = 0.10578554691520431 * (zz * (35 * zz - 30) + 3);
    out[o + 5] = -0.6690465435572892 * x * z * (7 * zz - 3);
    out[o + 6] = 0.47308734787878004 * (xx - yy) * (7 * zz - 1);
    out[o + 7] = -1.7701307697799304 * x * z * (xx - 3 * yy);
    out[o + 8] = 0.6258357354491761 * (xx * (xx - 3 * yy) - yy * (3 * xx - yy));
};

// Fibonacci-sphere directions (flat xyz) and the 9 x N pseudo-inverse of
// their basis matrix, built once on first use.
let band4: { dirs: Float64Array; pinv: Float64Array } | null = null;

const getBand4 = () => {
    if (band4) return band4;
    const n = BAND4_DIRS;
    const dirs = new Float64Array(n * 3);
    const A = new Float64Array(n * 9);
    const golden = Math.PI * (3 - Math.sqrt(5));
    for (let k = 0; k < n; k++) {
        const z = 1 - (2 * (k + 0.5)) / n;
        const r = Math.sqrt(1 - z * z);
        const x = r * Math.cos(golden * k);
        const y = r * Math.sin(golden * k);
        dirs.set([x, y, z], k * 3);
        band4Basis(x, y, z, A, k * 9);
    }
    // pinv = (A^T A)^-1 A^T, inverting the 9x9 normal matrix by Gauss-Jordan
    const M = new Float64Array(9 * 18);
    for (let i = 0; i < 9; i++) {
        for (let j = 0; j < 9; j++) {
            let sum = 0;
            for (let k = 0; k < n; k++) sum += A[k * 9 + i] * A[k * 9 + j];
            M[i * 18 + j] = sum;
        }
        M[i * 18 + 9 + i] = 1;
    }
    for (let c = 0; c < 9; c++) {
        let piv = c;
        for (let r = c + 1; r < 9; r++) if (Math.abs(M[r * 18 + c]) > Math.abs(M[piv * 18 + c])) piv = r;
        for (let j = 0; j < 18; j++) {
            const t = M[c * 18 + j];
            M[c * 18 + j] = M[piv * 18 + j];
            M[piv * 18 + j] = t;
        }
        const inv = 1 / M[c * 18 + c];
        for (let j = 0; j < 18; j++) M[c * 18 + j] *= inv;
        for (let r = 0; r < 9; r++) {
            if (r === c) continue;
            const f = M[r * 18 + c];
            if (f !== 0) for (let j = 0; j < 18; j++) M[r * 18 + j] -= f * M[c * 18 + j];
        }
    }
    const pinv = new Float64Array(9 * n);
    for (let i = 0; i < 9; i++) {
        for (let k = 0; k < n; k++) {
            let sum = 0;
            for (let j = 0; j < 9; j++) sum += M[i * 18 + 9 + j] * A[k * 9 + j];
            pinv[i * n + k] = sum;
        }
    }
    band4 = { dirs, pinv };
    return band4;
};

/**
 * The 9x9 band-4 SH rotation matrix for a column-major rotation, as rows.
 *
 * @param rot - Column-major 3x3 rotation (Mat3 data).
 * @returns Row-major matrix rows, `sh4[i][j]`.
 */
const band4Matrix = (rot: Float32Array): number[][] => {
    const { dirs, pinv } = getBand4();
    const n = BAND4_DIRS;
    const B = new Float64Array(n * 9);
    for (let k = 0; k < n; k++) {
        const x = dirs[k * 3],
            y = dirs[k * 3 + 1],
            z = dirs[k * 3 + 2];
        // R^T d, with R column-major: (R^T d)_i = sum_j R[j][i] d_j = sum_j rot[i*3+j] d_j
        band4Basis(
            rot[0] * x + rot[1] * y + rot[2] * z,
            rot[3] * x + rot[4] * y + rot[5] * z,
            rot[6] * x + rot[7] * y + rot[8] * z,
            B,
            k * 9
        );
    }
    const rows: number[][] = [];
    for (let i = 0; i < 9; i++) {
        const row: number[] = [];
        for (let j = 0; j < 9; j++) {
            let sum = 0;
            for (let k = 0; k < n; k++) sum += pinv[i * n + k] * B[k * 9 + j];
            row.push(sum);
        }
        rows.push(row);
    }
    return rows;
};

// Build a sparse representation of the SH rotation matrices. For axis-aligned
// rotations the matrices are highly sparse, so iterating only over non-zero
// entries is significantly faster than the full dot-product approach.
const buildSparse = (sh1: number[][], sh2: number[][], sh3: number[][], sh4: number[][]) => {
    const counts: number[] = [];
    const indices: number[] = [];
    const values: number[] = [];

    const addBand = (matrix: number[][], size: number, base: number) => {
        for (let i = 0; i < size; i++) {
            let count = 0;
            for (let j = 0; j < size; j++) {
                if (Math.abs(matrix[i][j]) > 1e-10) {
                    indices.push(base + j);
                    values.push(matrix[i][j]);
                    count++;
                }
            }
            counts.push(count);
        }
    };

    addBand(sh1, 3, 0);
    addBand(sh2, 5, 3);
    addBand(sh3, 7, 8);
    addBand(sh4, 9, 15);

    return { counts, indices, values };
};

// Returns true if the rotation matrix is a signed permutation (every entry is 0 or ±1),
// i.e. the rotation maps each axis to ±another axis (multiples of 90°).
const isAxisAligned = (rot: Float32Array) => {
    for (let i = 0; i < 9; i++) {
        const a = Math.abs(rot[i]);
        if (a > 0.01 && Math.abs(a - 1) > 0.01) return false;
    }
    return true;
};

// Rotate spherical harmonics up to band 4: bands 1-3 based on
// https://github.com/andrewwillmott/sh-lib, band 4 by projection (see band4Matrix).
//
// This implementation calculates the rotation factors during construction which can then
// be used to rotate multiple spherical harmonics cheaply.
class RotateSH {
    apply: (result: Float32Array | number[], src?: Float32Array | number[]) => void;

    constructor(mat: Mat3) {
        const rot = mat.data;

        // band 1
        const sh1 = [
            [rot[4], -rot[7], rot[1]],
            [-rot[5], rot[8], -rot[2]],
            [rot[3], -rot[6], rot[0]]
        ];

        // band 2
        const sh2 = [
            [
                kSqrt01_04 *
                    (sh1[2][2] * sh1[0][0] + sh1[2][0] * sh1[0][2] + (sh1[0][2] * sh1[2][0] + sh1[0][0] * sh1[2][2])),
                sh1[2][1] * sh1[0][0] + sh1[0][1] * sh1[2][0],
                kSqrt03_04 * (sh1[2][1] * sh1[0][1] + sh1[0][1] * sh1[2][1]),
                sh1[2][1] * sh1[0][2] + sh1[0][1] * sh1[2][2],
                kSqrt01_04 *
                    (sh1[2][2] * sh1[0][2] - sh1[2][0] * sh1[0][0] + (sh1[0][2] * sh1[2][2] - sh1[0][0] * sh1[2][0]))
            ],
            [
                kSqrt01_04 *
                    (sh1[1][2] * sh1[0][0] + sh1[1][0] * sh1[0][2] + (sh1[0][2] * sh1[1][0] + sh1[0][0] * sh1[1][2])),
                sh1[1][1] * sh1[0][0] + sh1[0][1] * sh1[1][0],
                kSqrt03_04 * (sh1[1][1] * sh1[0][1] + sh1[0][1] * sh1[1][1]),
                sh1[1][1] * sh1[0][2] + sh1[0][1] * sh1[1][2],
                kSqrt01_04 *
                    (sh1[1][2] * sh1[0][2] - sh1[1][0] * sh1[0][0] + (sh1[0][2] * sh1[1][2] - sh1[0][0] * sh1[1][0]))
            ],
            [
                kSqrt01_03 * (sh1[1][2] * sh1[1][0] + sh1[1][0] * sh1[1][2]) -
                    kSqrt01_12 *
                        (sh1[2][2] * sh1[2][0] +
                            sh1[2][0] * sh1[2][2] +
                            (sh1[0][2] * sh1[0][0] + sh1[0][0] * sh1[0][2])),
                kSqrt04_03 * sh1[1][1] * sh1[1][0] - kSqrt01_03 * (sh1[2][1] * sh1[2][0] + sh1[0][1] * sh1[0][0]),
                sh1[1][1] * sh1[1][1] - kSqrt01_04 * (sh1[2][1] * sh1[2][1] + sh1[0][1] * sh1[0][1]),
                kSqrt04_03 * sh1[1][1] * sh1[1][2] - kSqrt01_03 * (sh1[2][1] * sh1[2][2] + sh1[0][1] * sh1[0][2]),
                kSqrt01_03 * (sh1[1][2] * sh1[1][2] - sh1[1][0] * sh1[1][0]) -
                    kSqrt01_12 *
                        (sh1[2][2] * sh1[2][2] -
                            sh1[2][0] * sh1[2][0] +
                            (sh1[0][2] * sh1[0][2] - sh1[0][0] * sh1[0][0]))
            ],
            [
                kSqrt01_04 *
                    (sh1[1][2] * sh1[2][0] + sh1[1][0] * sh1[2][2] + (sh1[2][2] * sh1[1][0] + sh1[2][0] * sh1[1][2])),
                sh1[1][1] * sh1[2][0] + sh1[2][1] * sh1[1][0],
                kSqrt03_04 * (sh1[1][1] * sh1[2][1] + sh1[2][1] * sh1[1][1]),
                sh1[1][1] * sh1[2][2] + sh1[2][1] * sh1[1][2],
                kSqrt01_04 *
                    (sh1[1][2] * sh1[2][2] - sh1[1][0] * sh1[2][0] + (sh1[2][2] * sh1[1][2] - sh1[2][0] * sh1[1][0]))
            ],
            [
                kSqrt01_04 *
                    (sh1[2][2] * sh1[2][0] + sh1[2][0] * sh1[2][2] - (sh1[0][2] * sh1[0][0] + sh1[0][0] * sh1[0][2])),
                sh1[2][1] * sh1[2][0] - sh1[0][1] * sh1[0][0],
                kSqrt03_04 * (sh1[2][1] * sh1[2][1] - sh1[0][1] * sh1[0][1]),
                sh1[2][1] * sh1[2][2] - sh1[0][1] * sh1[0][2],
                kSqrt01_04 *
                    (sh1[2][2] * sh1[2][2] - sh1[2][0] * sh1[2][0] - (sh1[0][2] * sh1[0][2] - sh1[0][0] * sh1[0][0]))
            ]
        ];

        // band 3
        const sh3 = [
            [
                kSqrt01_04 *
                    (sh1[2][2] * sh2[0][0] + sh1[2][0] * sh2[0][4] + (sh1[0][2] * sh2[4][0] + sh1[0][0] * sh2[4][4])),
                kSqrt03_02 * (sh1[2][1] * sh2[0][0] + sh1[0][1] * sh2[4][0]),
                kSqrt15_16 * (sh1[2][1] * sh2[0][1] + sh1[0][1] * sh2[4][1]),
                kSqrt05_06 * (sh1[2][1] * sh2[0][2] + sh1[0][1] * sh2[4][2]),
                kSqrt15_16 * (sh1[2][1] * sh2[0][3] + sh1[0][1] * sh2[4][3]),
                kSqrt03_02 * (sh1[2][1] * sh2[0][4] + sh1[0][1] * sh2[4][4]),
                kSqrt01_04 *
                    (sh1[2][2] * sh2[0][4] - sh1[2][0] * sh2[0][0] + (sh1[0][2] * sh2[4][4] - sh1[0][0] * sh2[4][0]))
            ],
            [
                kSqrt01_06 * (sh1[1][2] * sh2[0][0] + sh1[1][0] * sh2[0][4]) +
                    kSqrt01_06 *
                        (sh1[2][2] * sh2[1][0] +
                            sh1[2][0] * sh2[1][4] +
                            (sh1[0][2] * sh2[3][0] + sh1[0][0] * sh2[3][4])),
                sh1[1][1] * sh2[0][0] + (sh1[2][1] * sh2[1][0] + sh1[0][1] * sh2[3][0]),
                kSqrt05_08 * sh1[1][1] * sh2[0][1] + kSqrt05_08 * (sh1[2][1] * sh2[1][1] + sh1[0][1] * sh2[3][1]),
                kSqrt05_09 * sh1[1][1] * sh2[0][2] + kSqrt05_09 * (sh1[2][1] * sh2[1][2] + sh1[0][1] * sh2[3][2]),
                kSqrt05_08 * sh1[1][1] * sh2[0][3] + kSqrt05_08 * (sh1[2][1] * sh2[1][3] + sh1[0][1] * sh2[3][3]),
                sh1[1][1] * sh2[0][4] + (sh1[2][1] * sh2[1][4] + sh1[0][1] * sh2[3][4]),
                kSqrt01_06 * (sh1[1][2] * sh2[0][4] - sh1[1][0] * sh2[0][0]) +
                    kSqrt01_06 *
                        (sh1[2][2] * sh2[1][4] -
                            sh1[2][0] * sh2[1][0] +
                            (sh1[0][2] * sh2[3][4] - sh1[0][0] * sh2[3][0]))
            ],
            [
                kSqrt04_15 * (sh1[1][2] * sh2[1][0] + sh1[1][0] * sh2[1][4]) +
                    kSqrt01_05 * (sh1[0][2] * sh2[2][0] + sh1[0][0] * sh2[2][4]) -
                    kSqrt01_60 *
                        (sh1[2][2] * sh2[0][0] +
                            sh1[2][0] * sh2[0][4] -
                            (sh1[0][2] * sh2[4][0] + sh1[0][0] * sh2[4][4])),
                kSqrt08_05 * sh1[1][1] * sh2[1][0] +
                    kSqrt06_05 * sh1[0][1] * sh2[2][0] -
                    kSqrt01_10 * (sh1[2][1] * sh2[0][0] - sh1[0][1] * sh2[4][0]),
                sh1[1][1] * sh2[1][1] +
                    kSqrt03_04 * sh1[0][1] * sh2[2][1] -
                    kSqrt01_16 * (sh1[2][1] * sh2[0][1] - sh1[0][1] * sh2[4][1]),
                kSqrt08_09 * sh1[1][1] * sh2[1][2] +
                    kSqrt02_03 * sh1[0][1] * sh2[2][2] -
                    kSqrt01_18 * (sh1[2][1] * sh2[0][2] - sh1[0][1] * sh2[4][2]),
                sh1[1][1] * sh2[1][3] +
                    kSqrt03_04 * sh1[0][1] * sh2[2][3] -
                    kSqrt01_16 * (sh1[2][1] * sh2[0][3] - sh1[0][1] * sh2[4][3]),
                kSqrt08_05 * sh1[1][1] * sh2[1][4] +
                    kSqrt06_05 * sh1[0][1] * sh2[2][4] -
                    kSqrt01_10 * (sh1[2][1] * sh2[0][4] - sh1[0][1] * sh2[4][4]),
                kSqrt04_15 * (sh1[1][2] * sh2[1][4] - sh1[1][0] * sh2[1][0]) +
                    kSqrt01_05 * (sh1[0][2] * sh2[2][4] - sh1[0][0] * sh2[2][0]) -
                    kSqrt01_60 *
                        (sh1[2][2] * sh2[0][4] -
                            sh1[2][0] * sh2[0][0] -
                            (sh1[0][2] * sh2[4][4] - sh1[0][0] * sh2[4][0]))
            ],
            [
                kSqrt03_10 * (sh1[1][2] * sh2[2][0] + sh1[1][0] * sh2[2][4]) -
                    kSqrt01_10 *
                        (sh1[2][2] * sh2[3][0] +
                            sh1[2][0] * sh2[3][4] +
                            (sh1[0][2] * sh2[1][0] + sh1[0][0] * sh2[1][4])),
                kSqrt09_05 * sh1[1][1] * sh2[2][0] - kSqrt03_05 * (sh1[2][1] * sh2[3][0] + sh1[0][1] * sh2[1][0]),
                kSqrt09_08 * sh1[1][1] * sh2[2][1] - kSqrt03_08 * (sh1[2][1] * sh2[3][1] + sh1[0][1] * sh2[1][1]),
                sh1[1][1] * sh2[2][2] - kSqrt01_03 * (sh1[2][1] * sh2[3][2] + sh1[0][1] * sh2[1][2]),
                kSqrt09_08 * sh1[1][1] * sh2[2][3] - kSqrt03_08 * (sh1[2][1] * sh2[3][3] + sh1[0][1] * sh2[1][3]),
                kSqrt09_05 * sh1[1][1] * sh2[2][4] - kSqrt03_05 * (sh1[2][1] * sh2[3][4] + sh1[0][1] * sh2[1][4]),
                kSqrt03_10 * (sh1[1][2] * sh2[2][4] - sh1[1][0] * sh2[2][0]) -
                    kSqrt01_10 *
                        (sh1[2][2] * sh2[3][4] -
                            sh1[2][0] * sh2[3][0] +
                            (sh1[0][2] * sh2[1][4] - sh1[0][0] * sh2[1][0]))
            ],
            [
                kSqrt04_15 * (sh1[1][2] * sh2[3][0] + sh1[1][0] * sh2[3][4]) +
                    kSqrt01_05 * (sh1[2][2] * sh2[2][0] + sh1[2][0] * sh2[2][4]) -
                    kSqrt01_60 *
                        (sh1[2][2] * sh2[4][0] +
                            sh1[2][0] * sh2[4][4] +
                            (sh1[0][2] * sh2[0][0] + sh1[0][0] * sh2[0][4])),
                kSqrt08_05 * sh1[1][1] * sh2[3][0] +
                    kSqrt06_05 * sh1[2][1] * sh2[2][0] -
                    kSqrt01_10 * (sh1[2][1] * sh2[4][0] + sh1[0][1] * sh2[0][0]),
                sh1[1][1] * sh2[3][1] +
                    kSqrt03_04 * sh1[2][1] * sh2[2][1] -
                    kSqrt01_16 * (sh1[2][1] * sh2[4][1] + sh1[0][1] * sh2[0][1]),
                kSqrt08_09 * sh1[1][1] * sh2[3][2] +
                    kSqrt02_03 * sh1[2][1] * sh2[2][2] -
                    kSqrt01_18 * (sh1[2][1] * sh2[4][2] + sh1[0][1] * sh2[0][2]),
                sh1[1][1] * sh2[3][3] +
                    kSqrt03_04 * sh1[2][1] * sh2[2][3] -
                    kSqrt01_16 * (sh1[2][1] * sh2[4][3] + sh1[0][1] * sh2[0][3]),
                kSqrt08_05 * sh1[1][1] * sh2[3][4] +
                    kSqrt06_05 * sh1[2][1] * sh2[2][4] -
                    kSqrt01_10 * (sh1[2][1] * sh2[4][4] + sh1[0][1] * sh2[0][4]),
                kSqrt04_15 * (sh1[1][2] * sh2[3][4] - sh1[1][0] * sh2[3][0]) +
                    kSqrt01_05 * (sh1[2][2] * sh2[2][4] - sh1[2][0] * sh2[2][0]) -
                    kSqrt01_60 *
                        (sh1[2][2] * sh2[4][4] -
                            sh1[2][0] * sh2[4][0] +
                            (sh1[0][2] * sh2[0][4] - sh1[0][0] * sh2[0][0]))
            ],
            [
                kSqrt01_06 * (sh1[1][2] * sh2[4][0] + sh1[1][0] * sh2[4][4]) +
                    kSqrt01_06 *
                        (sh1[2][2] * sh2[3][0] +
                            sh1[2][0] * sh2[3][4] -
                            (sh1[0][2] * sh2[1][0] + sh1[0][0] * sh2[1][4])),
                sh1[1][1] * sh2[4][0] + (sh1[2][1] * sh2[3][0] - sh1[0][1] * sh2[1][0]),
                kSqrt05_08 * sh1[1][1] * sh2[4][1] + kSqrt05_08 * (sh1[2][1] * sh2[3][1] - sh1[0][1] * sh2[1][1]),
                kSqrt05_09 * sh1[1][1] * sh2[4][2] + kSqrt05_09 * (sh1[2][1] * sh2[3][2] - sh1[0][1] * sh2[1][2]),
                kSqrt05_08 * sh1[1][1] * sh2[4][3] + kSqrt05_08 * (sh1[2][1] * sh2[3][3] - sh1[0][1] * sh2[1][3]),
                sh1[1][1] * sh2[4][4] + (sh1[2][1] * sh2[3][4] - sh1[0][1] * sh2[1][4]),
                kSqrt01_06 * (sh1[1][2] * sh2[4][4] - sh1[1][0] * sh2[4][0]) +
                    kSqrt01_06 *
                        (sh1[2][2] * sh2[3][4] -
                            sh1[2][0] * sh2[3][0] -
                            (sh1[0][2] * sh2[1][4] - sh1[0][0] * sh2[1][0]))
            ],
            [
                kSqrt01_04 *
                    (sh1[2][2] * sh2[4][0] + sh1[2][0] * sh2[4][4] - (sh1[0][2] * sh2[0][0] + sh1[0][0] * sh2[0][4])),
                kSqrt03_02 * (sh1[2][1] * sh2[4][0] - sh1[0][1] * sh2[0][0]),
                kSqrt15_16 * (sh1[2][1] * sh2[4][1] - sh1[0][1] * sh2[0][1]),
                kSqrt05_06 * (sh1[2][1] * sh2[4][2] - sh1[0][1] * sh2[0][2]),
                kSqrt15_16 * (sh1[2][1] * sh2[4][3] - sh1[0][1] * sh2[0][3]),
                kSqrt03_02 * (sh1[2][1] * sh2[4][4] - sh1[0][1] * sh2[0][4]),
                kSqrt01_04 *
                    (sh1[2][2] * sh2[4][4] - sh1[2][0] * sh2[4][0] - (sh1[0][2] * sh2[0][4] - sh1[0][0] * sh2[0][0]))
            ]
        ];

        const sh4 = band4Matrix(rot);

        if (isAxisAligned(rot)) {
            const { counts, indices, values } = buildSparse(sh1, sh2, sh3, sh4);

            this.apply = (result: Float32Array | number[], src?: Float32Array | number[]) => {
                if (!src || src === result) {
                    coeffsIn.set(result);
                    src = coeffsIn;
                }

                let vp = 0;

                if (result.length < 3) return;
                for (let i = 0; i < 3; i++) {
                    let sum = 0;
                    for (let k = 0; k < counts[i]; k++) {
                        sum += values[vp] * src[indices[vp]];
                        vp++;
                    }
                    result[i] = sum;
                }

                if (result.length < 8) return;
                for (let i = 0; i < 5; i++) {
                    let sum = 0;
                    for (let k = 0; k < counts[3 + i]; k++) {
                        sum += values[vp] * src[indices[vp]];
                        vp++;
                    }
                    result[3 + i] = sum;
                }

                if (result.length < 15) return;
                for (let i = 0; i < 7; i++) {
                    let sum = 0;
                    for (let k = 0; k < counts[8 + i]; k++) {
                        sum += values[vp] * src[indices[vp]];
                        vp++;
                    }
                    result[8 + i] = sum;
                }

                if (result.length < 24) return;
                for (let i = 0; i < 9; i++) {
                    let sum = 0;
                    for (let k = 0; k < counts[15 + i]; k++) {
                        sum += values[vp] * src[indices[vp]];
                        vp++;
                    }
                    result[15 + i] = sum;
                }
            };
        } else {
            this.apply = (result: Float32Array | number[], src?: Float32Array | number[]) => {
                if (!src || src === result) {
                    coeffsIn.set(result);
                    src = coeffsIn;
                }

                // band 1
                if (result.length < 3) {
                    return;
                }
                result[0] = dp(3, 0, src, sh1[0]);
                result[1] = dp(3, 0, src, sh1[1]);
                result[2] = dp(3, 0, src, sh1[2]);

                // band 2
                if (result.length < 8) {
                    return;
                }
                result[3] = dp(5, 3, src, sh2[0]);
                result[4] = dp(5, 3, src, sh2[1]);
                result[5] = dp(5, 3, src, sh2[2]);
                result[6] = dp(5, 3, src, sh2[3]);
                result[7] = dp(5, 3, src, sh2[4]);

                // band 3
                if (result.length < 15) {
                    return;
                }
                result[8] = dp(7, 8, src, sh3[0]);
                result[9] = dp(7, 8, src, sh3[1]);
                result[10] = dp(7, 8, src, sh3[2]);
                result[11] = dp(7, 8, src, sh3[3]);
                result[12] = dp(7, 8, src, sh3[4]);
                result[13] = dp(7, 8, src, sh3[5]);
                result[14] = dp(7, 8, src, sh3[6]);

                // band 4
                if (result.length < 24) {
                    return;
                }
                for (let i = 0; i < 9; i++) result[15 + i] = dp(9, 15, src, sh4[i]);
            };
        }
    }
}

export { RotateSH, band4Matrix };
