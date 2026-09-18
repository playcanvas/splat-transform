/**
 * Module-scope helpers for the 3D covariance: a symmetric 3×3 record and
 * the camera-space rotation V·Σ·Vᵀ as a function, so a shader can rotate
 * the same world covariance into more than one camera basis (the
 * motion-blur path needs the shutter-open and shutter-close poses).
 *
 * Include once at module scope, before `covariance3D`.
 */
const covariance3DFns = /* wgsl */`
struct Sym3 {
    c00: f32, c01: f32, c02: f32, c11: f32, c12: f32, c22: f32,
}

// Rotate a symmetric world covariance into the camera basis whose rows
// are (v0*, v1*, v2*) = (right, down, forward): returns V·Σ·Vᵀ.
fn camCov(
    v00: f32, v01: f32, v02: f32,
    v10: f32, v11: f32, v12: f32,
    v20: f32, v21: f32, v22: f32,
    sig00: f32, sig01: f32, sig02: f32, sig11: f32, sig12: f32, sig22: f32
) -> Sym3 {
    let t00 = v00 * sig00 + v01 * sig01 + v02 * sig02;
    let t01 = v00 * sig01 + v01 * sig11 + v02 * sig12;
    let t02 = v00 * sig02 + v01 * sig12 + v02 * sig22;
    let t10 = v10 * sig00 + v11 * sig01 + v12 * sig02;
    let t11 = v10 * sig01 + v11 * sig11 + v12 * sig12;
    let t12 = v10 * sig02 + v11 * sig12 + v12 * sig22;
    let t20 = v20 * sig00 + v21 * sig01 + v22 * sig02;
    let t21 = v20 * sig01 + v21 * sig11 + v22 * sig12;
    let t22 = v20 * sig02 + v21 * sig12 + v22 * sig22;

    var c: Sym3;
    c.c00 = t00 * v00 + t01 * v01 + t02 * v02;
    c.c01 = t00 * v10 + t01 * v11 + t02 * v12;
    c.c02 = t00 * v20 + t01 * v21 + t02 * v22;
    c.c11 = t10 * v10 + t11 * v11 + t12 * v12;
    c.c12 = t10 * v20 + t11 * v21 + t12 * v22;
    c.c22 = t20 * v20 + t21 * v21 + t22 * v22;
    return c;
}
`;

/**
 * 3D world-space covariance Σ = M·Mᵀ where M = R·diag(scale), then
 * rotated into camera space via V·Σ·Vᵀ.
 *
 * Reads:   r00..r22 (world rotation), lsX, lsY, lsZ (log scales),
 *          uniforms.rightX/Y/Z, uniforms.downX/Y/Z, uniforms.forwardX/Y/Z
 *          (camera basis rows = view-rotation matrix V)
 * Defines: sig00..sig22 (world-space covariance, upper triangle) and
 *          c00, c01, c02, c11, c12, c22 (camera-space 3D covariance,
 *          symmetric — only the upper triangle is stored)
 *
 * The output covariance feeds the Jacobian chunks (pinhole / equirect)
 * to derive the 2D screen-space covariance via cov2D = J · cov3D · Jᵀ.
 * Requires `covariance3DFns` at module scope.
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

    let cc = camCov(
        uniforms.rightX, uniforms.rightY, uniforms.rightZ,
        uniforms.downX, uniforms.downY, uniforms.downZ,
        uniforms.forwardX, uniforms.forwardY, uniforms.forwardZ,
        sig00, sig01, sig02, sig11, sig12, sig22
    );
    let c00 = cc.c00;
    let c01 = cc.c01;
    let c02 = cc.c02;
    let c11 = cc.c11;
    let c12 = cc.c12;
    let c22 = cc.c22;
`;

export { covariance3D, covariance3DFns };
