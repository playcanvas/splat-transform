/**
 * Tests for RotateSH, with focus on the axis-aligned (signed permutation) fast path.
 *
 * Axis-aligned rotations (multiples of 90° around coordinate axes) produce signed
 * permutation rotation matrices, enabling a fast path where each output SH coefficient
 * is ±1 × a single input coefficient instead of a full dot product.
 */

import { describe, it } from 'node:test';

import { Mat3, Quat } from 'playcanvas';

import { RotateSH } from '../src/lib/utils/rotate-sh.js';
import { assertClose } from './helpers/summary-compare.mjs';

const mat3FromEulers = (x, y, z) => {
    return new Mat3().setFromQuat(new Quat().setFromEulerAngles(x, y, z));
};

const testCoeffs = () => new Float32Array([
    0.5, -1.3, 2.1,
    -0.7, 1.8, -0.3, 0.9, -1.1,
    0.4, -0.6, 1.2, -1.5, 0.8, -0.2, 1.7
]);

const bandNorm = (c, start, count) => {
    let sum = 0;
    for (let i = start; i < start + count; i++) sum += c[i] * c[i];
    return sum;
};

describe('RotateSH axis-aligned fast path', () => {

    describe('band 1 known values', () => {
        it('90° around Z', () => {
            const r = new RotateSH(mat3FromEulers(0, 0, 90));
            const c = new Float32Array([1, 2, 3]);
            r.apply(c);
            assertClose(c[0], 3, 1e-6, 'c0');
            assertClose(c[1], 2, 1e-6, 'c1');
            assertClose(c[2], -1, 1e-6, 'c2');
        });

        it('90° around Y', () => {
            const r = new RotateSH(mat3FromEulers(0, 90, 0));
            const c = new Float32Array([1, 2, 3]);
            r.apply(c);
            assertClose(c[0], 1, 1e-6, 'c0');
            assertClose(c[1], 3, 1e-6, 'c1');
            assertClose(c[2], -2, 1e-6, 'c2');
        });

        it('90° around X', () => {
            const r = new RotateSH(mat3FromEulers(90, 0, 0));
            const c = new Float32Array([1, 2, 3]);
            r.apply(c);
            assertClose(c[0], 2, 1e-6, 'c0');
            assertClose(c[1], -1, 1e-6, 'c1');
            assertClose(c[2], 3, 1e-6, 'c2');
        });

        it('180° around Z (PLY convention)', () => {
            const r = new RotateSH(mat3FromEulers(0, 0, 180));
            const c = new Float32Array([1, 2, 3]);
            r.apply(c);
            assertClose(c[0], -1, 1e-6, 'c0');
            assertClose(c[1], 2, 1e-6, 'c1');
            assertClose(c[2], -3, 1e-6, 'c2');
        });
    });

    describe('round-trip (rotation then inverse)', () => {
        const rotations = [
            [90, 0, 0], [0, 90, 0], [0, 0, 90],
            [180, 0, 0], [0, 180, 0], [0, 0, 180],
            [270, 0, 0], [0, 270, 0], [0, 0, 270],
            [90, 90, 0], [90, 0, 90], [0, 90, 90],
            [90, 90, 90], [90, 180, 0], [180, 0, 90]
        ];

        for (const eulers of rotations) {
            it(`euler(${eulers}) round-trip preserves all bands`, () => {
                const q = new Quat().setFromEulerAngles(eulers[0], eulers[1], eulers[2]);
                const qInv = q.clone().invert();
                const r = new RotateSH(new Mat3().setFromQuat(q));
                const rInv = new RotateSH(new Mat3().setFromQuat(qInv));

                const original = testCoeffs();
                const c = new Float32Array(original);
                r.apply(c);
                rInv.apply(c);

                for (let i = 0; i < 15; i++) {
                    assertClose(c[i], original[i], 1e-5, `coeff ${i}`);
                }
            });
        }
    });

    describe('repeated application', () => {
        it('180° applied twice equals identity', () => {
            for (const eulers of [[180, 0, 0], [0, 180, 0], [0, 0, 180]]) {
                const r = new RotateSH(mat3FromEulers(eulers[0], eulers[1], eulers[2]));
                const original = testCoeffs();
                const c = new Float32Array(original);
                r.apply(c);
                r.apply(c);
                for (let i = 0; i < 15; i++) {
                    assertClose(c[i], original[i], 1e-5, `euler(${eulers}) coeff ${i}`);
                }
            }
        });

        it('90° applied four times equals identity', () => {
            for (const eulers of [[90, 0, 0], [0, 90, 0], [0, 0, 90]]) {
                const r = new RotateSH(mat3FromEulers(eulers[0], eulers[1], eulers[2]));
                const original = testCoeffs();
                const c = new Float32Array(original);
                r.apply(c);
                r.apply(c);
                r.apply(c);
                r.apply(c);
                for (let i = 0; i < 15; i++) {
                    assertClose(c[i], original[i], 1e-5, `euler(${eulers}) coeff ${i}`);
                }
            }
        });
    });

    describe('norm preservation', () => {
        for (const eulers of [[90, 0, 0], [0, 90, 0], [0, 0, 90], [90, 90, 0], [90, 90, 90]]) {
            it(`euler(${eulers}) preserves L2 norm of each band`, () => {
                const r = new RotateSH(mat3FromEulers(eulers[0], eulers[1], eulers[2]));
                const original = testCoeffs();
                const c = new Float32Array(original);
                r.apply(c);

                assertClose(bandNorm(c, 0, 3), bandNorm(original, 0, 3), 1e-5, 'band 1');
                assertClose(bandNorm(c, 3, 5), bandNorm(original, 3, 5), 1e-5, 'band 2');
                assertClose(bandNorm(c, 8, 7), bandNorm(original, 8, 7), 1e-5, 'band 3');
            });
        }
    });

    describe('composition', () => {
        it('sequential R(A) then R(B) equals single R(A*B)', () => {
            const qA = new Quat().setFromEulerAngles(90, 0, 0);
            const qB = new Quat().setFromEulerAngles(0, 90, 0);
            const qAB = new Quat().mul2(qA, qB);

            const rA = new RotateSH(new Mat3().setFromQuat(qA));
            const rB = new RotateSH(new Mat3().setFromQuat(qB));
            const rAB = new RotateSH(new Mat3().setFromQuat(qAB));

            const original = testCoeffs();

            const cSeq = new Float32Array(original);
            rB.apply(cSeq);
            rA.apply(cSeq);

            const cComp = new Float32Array(original);
            rAB.apply(cComp);

            for (let i = 0; i < 15; i++) {
                assertClose(cSeq[i], cComp[i], 1e-5, `coeff ${i}`);
            }
        });

        it('three composed 90° rotations match combined rotation', () => {
            const qX = new Quat().setFromEulerAngles(90, 0, 0);
            const qY = new Quat().setFromEulerAngles(0, 90, 0);
            const qZ = new Quat().setFromEulerAngles(0, 0, 90);
            const qAll = new Quat().mul2(qZ, new Quat().mul2(qY, qX));

            const rX = new RotateSH(new Mat3().setFromQuat(qX));
            const rY = new RotateSH(new Mat3().setFromQuat(qY));
            const rZ = new RotateSH(new Mat3().setFromQuat(qZ));
            const rAll = new RotateSH(new Mat3().setFromQuat(qAll));

            const original = testCoeffs();

            const cSeq = new Float32Array(original);
            rX.apply(cSeq);
            rY.apply(cSeq);
            rZ.apply(cSeq);

            const cComp = new Float32Array(original);
            rAll.apply(cComp);

            for (let i = 0; i < 15; i++) {
                assertClose(cSeq[i], cComp[i], 1e-5, `coeff ${i}`);
            }
        });
    });

    describe('partial bands', () => {
        it('correctly handles band-1-only input (3 coefficients)', () => {
            const r = new RotateSH(mat3FromEulers(0, 0, 90));
            const rInv = new RotateSH(mat3FromEulers(0, 0, -90));
            const original = new Float32Array([1, -2, 3]);
            const c = new Float32Array(original);
            r.apply(c);
            rInv.apply(c);
            for (let i = 0; i < 3; i++) {
                assertClose(c[i], original[i], 1e-6, `coeff ${i}`);
            }
        });

        it('correctly handles band-1+2 input (8 coefficients)', () => {
            const r = new RotateSH(mat3FromEulers(90, 0, 0));
            const rInv = new RotateSH(mat3FromEulers(-90, 0, 0));
            const original = new Float32Array([1, -2, 3, -4, 5, -6, 7, -8]);
            const c = new Float32Array(original);
            r.apply(c);
            rInv.apply(c);
            for (let i = 0; i < 8; i++) {
                assertClose(c[i], original[i], 1e-5, `coeff ${i}`);
            }
        });
    });

    describe('band 2 known values', () => {
        it('180° around Z negates xy and xz terms, preserves others', () => {
            const r = new RotateSH(mat3FromEulers(0, 0, 180));
            const original = testCoeffs();
            const c = new Float32Array(original);
            r.apply(c);

            // 180° around Z: x→-x, y→-y, z→z
            // Band 2 basis: xy, yz, (3z²-r²)/2, xz, (x²-y²)/2
            // xy → (-x)(-y) = xy → same          (index 3)
            // yz → (-y)(z) = -yz → negate         (index 4)
            // (3z²-r²)/2 → unchanged              (index 5)
            // xz → (-x)(z) = -xz → negate         (index 6)
            // (x²-y²)/2 → (x²-y²)/2 → same       (index 7)
            assertClose(c[3], original[3], 1e-6, 'xy unchanged');
            assertClose(c[4], -original[4], 1e-6, 'yz negated');
            assertClose(c[5], original[5], 1e-6, 'z² unchanged');
            assertClose(c[6], -original[6], 1e-6, 'xz negated');
            assertClose(c[7], original[7], 1e-6, 'x²-y² unchanged');
        });
    });
});

// Real SH basis exactly as the renderers evaluate it (engine gsplatEvalSH /
// 3DGS sh_utils), bands 1-4, for a unit direction.
const evalBasis = (x, y, z) => {
    const xx = x * x, yy = y * y, zz = z * z, xy = x * y, yz = y * z, xz = x * z;
    return [
        // band 1
        -0.4886025119029199 * y, 0.4886025119029199 * z, -0.4886025119029199 * x,
        // band 2
        1.0925484305920792 * xy, -1.0925484305920792 * yz, 0.31539156525252005 * (2 * zz - xx - yy),
        -1.0925484305920792 * xz, 0.5462742152960396 * (xx - yy),
        // band 3
        -0.5900435899266435 * y * (3 * xx - yy), 2.890611442640554 * xy * z,
        -0.4570457994644658 * y * (4 * zz - xx - yy), 0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy),
        -0.4570457994644658 * x * (4 * zz - xx - yy), 1.445305721320277 * z * (xx - yy),
        -0.5900435899266435 * x * (xx - 3 * yy),
        // band 4
        2.5033429417967046 * xy * (xx - yy), -1.7701307697799304 * yz * (3 * xx - yy),
        0.9461746957575601 * xy * (7 * zz - 1), -0.6690465435572892 * yz * (7 * zz - 3),
        0.10578554691520431 * (zz * (35 * zz - 30) + 3), -0.6690465435572892 * xz * (7 * zz - 3),
        0.47308734787878004 * (xx - yy) * (7 * zz - 1), -1.7701307697799304 * xz * (xx - 3 * yy),
        0.6258357354491761 * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
    ];
};

const BAND_RANGES = [[0, 3], [3, 8], [8, 15], [15, 24]];

// Evaluate one band of a 24-coefficient channel at direction d.
const evalBand = (c, band, d) => {
    const y = evalBasis(...d);
    const [a, b] = BAND_RANGES[band];
    let sum = 0;
    for (let i = a; i < b; i++) sum += c[i] * y[i];
    return sum;
};

describe('RotateSH band 4', () => {
    let seed = 7;
    const rand = () => {
        seed = (seed * 16807) % 2147483647;
        return seed / 2147483647 * 2 - 1;
    };
    const randDir = () => {
        const v = [rand(), rand(), rand()];
        const l = Math.hypot(...v);
        return v.map(c => c / l);
    };

    // f'(d) = f(R^T d): rotating the coefficients rotates the function.
    const checkRotation = (mat, label) => {
        const r = new RotateSH(mat);
        const c = Float64Array.from({ length: 24 }, rand);
        const out = new Float32Array(24);
        r.apply(out, c);
        const m = mat.data;
        for (let t = 0; t < 20; t++) {
            const d = randDir();
            // R^T d with column-major R
            const rd = [
                m[0] * d[0] + m[1] * d[1] + m[2] * d[2],
                m[3] * d[0] + m[4] * d[1] + m[5] * d[2],
                m[6] * d[0] + m[7] * d[1] + m[8] * d[2]
            ];
            for (let band = 0; band < 4; band++) {
                assertClose(evalBand(out, band, d), evalBand(c, band, rd), 1e-5, `${label} band ${band + 1}`);
            }
        }
    };

    it('rotates every band consistently with the evaluated basis (general rotations)', () => {
        for (let i = 0; i < 10; i++) {
            const q = new Quat(rand(), rand(), rand(), rand()).normalize();
            checkRotation(new Mat3().setFromQuat(q), `random ${i}`);
        }
    });

    it('rotates every band consistently on the axis-aligned fast path', () => {
        for (const [x, y, z] of [[90, 0, 0], [0, 90, 0], [0, 0, 90], [180, 0, 0], [90, 180, -90]]) {
            checkRotation(mat3FromEulers(x, y, z), `axis ${x},${y},${z}`);
        }
    });

    it('preserves band-4 energy and composes', () => {
        const q1 = new Quat().setFromEulerAngles(17, -40, 63);
        const q2 = new Quat().setFromEulerAngles(-5, 22, 141);
        const c = Float32Array.from({ length: 24 }, rand);
        const a = c.slice();
        new RotateSH(new Mat3().setFromQuat(q1)).apply(a);
        new RotateSH(new Mat3().setFromQuat(q2)).apply(a);
        const b = c.slice();
        new RotateSH(new Mat3().setFromQuat(new Quat().mul2(q2, q1))).apply(b);
        for (let i = 15; i < 24; i++) assertClose(a[i], b[i], 1e-5, `composed coeff ${i}`);
        assertClose(bandNorm(a, 15, 9), bandNorm(c, 15, 9), 1e-4, 'band-4 energy');
    });
});
