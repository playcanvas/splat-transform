import { TypedArray } from './data-table';
import { sigmoid } from '../utils';

/**
 * Pre-computed per-Gaussian inverse transform data for evaluating
 * Gaussian contribution at arbitrary 3D points.
 */
interface GaussianInverseTransform {
    /** Inverse rotation quaternion (w, x, y, z) with xyz negated */
    qw: number;
    qx: number;
    qy: number;
    qz: number;
    /** Inverse scale: exp(-log_scale) per axis */
    isx: number;
    isy: number;
    isz: number;
    /** Linear opacity (sigmoid of raw logit) */
    alpha: number;
}

/**
 * Compute the inverse transform for a single Gaussian.
 *
 * @param rotW - Quaternion w component.
 * @param rotX - Quaternion x component.
 * @param rotY - Quaternion y component.
 * @param rotZ - Quaternion z component.
 * @param logScaleX - Log scale x.
 * @param logScaleY - Log scale y.
 * @param logScaleZ - Log scale z.
 * @param opacityLogit - Raw opacity logit value.
 * @returns Inverse transform parameters.
 */
const computeGaussianInverse = (
    rotW: number, rotX: number, rotY: number, rotZ: number,
    logScaleX: number, logScaleY: number, logScaleZ: number,
    opacityLogit: number
): GaussianInverseTransform => {
    const qlen = Math.sqrt(rotW * rotW + rotX * rotX + rotY * rotY + rotZ * rotZ);
    const invLen = qlen > 0 ? 1 / qlen : 0;
    return {
        qw: rotW * invLen,
        qx: -rotX * invLen,
        qy: -rotY * invLen,
        qz: -rotZ * invLen,
        isx: Math.exp(-logScaleX),
        isy: Math.exp(-logScaleY),
        isz: Math.exp(-logScaleZ),
        alpha: sigmoid(opacityLogit)
    };
};

/**
 * Evaluate a Gaussian's opacity contribution at a 3D point.
 *
 * Uses the Rodrigues cross-product formula for inverse rotation,
 * then computes the Mahalanobis distance in the Gaussian's local frame.
 *
 * @param g - Pre-computed inverse transform of the Gaussian.
 * @param px - Gaussian center x.
 * @param py - Gaussian center y.
 * @param pz - Gaussian center z.
 * @param vx - Evaluation point x.
 * @param vy - Evaluation point y.
 * @param vz - Evaluation point z.
 * @returns Opacity contribution at the evaluation point.
 */
const evaluateGaussianAt = (
    g: GaussianInverseTransform,
    px: number, py: number, pz: number,
    vx: number, vy: number, vz: number
): number => {
    const dx = vx - px;
    const dy = vy - py;
    const dz = vz - pz;

    const tx = 2 * (g.qy * dz - g.qz * dy);
    const ty = 2 * (g.qz * dx - g.qx * dz);
    const tz = 2 * (g.qx * dy - g.qy * dx);

    const ldx = dx + g.qw * tx + (g.qy * tz - g.qz * ty);
    const ldy = dy + g.qw * ty + (g.qz * tx - g.qx * tz);
    const ldz = dz + g.qw * tz + (g.qx * ty - g.qy * tx);

    const sdx = ldx * g.isx;
    const sdy = ldy * g.isy;
    const sdz = ldz * g.isz;
    const d2 = sdx * sdx + sdy * sdy + sdz * sdz;

    return g.alpha * Math.exp(-0.5 * d2);
};

/**
 * Column arrays needed for Gaussian contribution evaluation.
 */
interface GaussianColumns {
    posX: TypedArray;
    posY: TypedArray;
    posZ: TypedArray;
    rotW: TypedArray;
    rotX: TypedArray;
    rotY: TypedArray;
    rotZ: TypedArray;
    scaleX: TypedArray;
    scaleY: TypedArray;
    scaleZ: TypedArray;
    opacity: TypedArray;
    extentX: TypedArray;
    extentY: TypedArray;
    extentZ: TypedArray;
}

export {
    evaluateGaussianAt,
    computeGaussianInverse,
    type GaussianColumns,
    type GaussianInverseTransform
};
