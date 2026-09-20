/**
 * 3D world-space covariance Σ = M·Mᵀ where M = R·diag(scale), then
 * rotated into camera space via V·Σ·Vᵀ.
 *
 * Reads:   r00..r22 (world rotation), lsX, lsY, lsZ (log scales),
 *          uniforms.rightX/Y/Z, uniforms.downX/Y/Z, uniforms.forwardX/Y/Z
 *          (camera basis rows = view-rotation matrix V)
 * Defines: c00, c01, c02, c11, c12, c22 (camera-space 3D covariance,
 *          symmetric — only the upper triangle is stored)
 *
 * The output covariance feeds the Jacobian chunks (pinhole / equirect)
 * to derive the 2D screen-space covariance via cov2D = J · cov3D · Jᵀ.
 */
const covariance3D = /* wgsl */`
    let sx = exp(lsX);
    let sy = exp(lsY);
    let sz = exp(lsZ);

    let m00 = r00 * sx; let m01 = r01 * sy; let m02 = r02 * sz;
    let m10 = r10 * sx; let m11 = r11 * sy; let m12 = r12 * sz;
    let m20 = r20 * sx; let m21 = r21 * sy; let m22 = r22 * sz;

    let sig00 = m00 * m00 + m01 * m01 + m02 * m02;
    let sig01 = m00 * m10 + m01 * m11 + m02 * m12;
    let sig02 = m00 * m20 + m01 * m21 + m02 * m22;
    let sig11 = m10 * m10 + m11 * m11 + m12 * m12;
    let sig12 = m10 * m20 + m11 * m21 + m12 * m22;
    let sig22 = m20 * m20 + m21 * m21 + m22 * m22;

    let v00 = uniforms.rightX; let v01 = uniforms.rightY; let v02 = uniforms.rightZ;
    let v10 = uniforms.downX; let v11 = uniforms.downY; let v12 = uniforms.downZ;
    let v20 = uniforms.forwardX; let v21 = uniforms.forwardY; let v22 = uniforms.forwardZ;

    let t00 = v00 * sig00 + v01 * sig01 + v02 * sig02;
    let t01 = v00 * sig01 + v01 * sig11 + v02 * sig12;
    let t02 = v00 * sig02 + v01 * sig12 + v02 * sig22;
    let t10 = v10 * sig00 + v11 * sig01 + v12 * sig02;
    let t11 = v10 * sig01 + v11 * sig11 + v12 * sig12;
    let t12 = v10 * sig02 + v11 * sig12 + v12 * sig22;
    let t20 = v20 * sig00 + v21 * sig01 + v22 * sig02;
    let t21 = v20 * sig01 + v21 * sig11 + v22 * sig12;
    let t22 = v20 * sig02 + v21 * sig12 + v22 * sig22;

    let c00 = t00 * v00 + t01 * v01 + t02 * v02;
    let c01 = t00 * v10 + t01 * v11 + t02 * v12;
    let c02 = t00 * v20 + t01 * v21 + t02 * v22;
    let c11 = t10 * v10 + t11 * v11 + t12 * v12;
    let c12 = t10 * v20 + t11 * v21 + t12 * v22;
    let c22 = t20 * v20 + t21 * v21 + t22 * v22;
`;

export { covariance3D };
