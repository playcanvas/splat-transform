/**
 * Equirect Jacobian + 2D EWA covariance.
 *
 * Reads:   cx, cy, cz, r2, rxzClamped,
 *          c00, c01, c02, c11, c12, c22 (camera-space 3D covariance)
 * Defines: cov00, cov01, cov11 (all var, so later dilation steps can add to them)
 * Requires: `jacobianEquirectFns` at module scope
 *
 * With longitude θ = atan2(cx, cz) and latitude φ = asin(cy/r) (cy is
 * the camera-down axis, so φ > 0 = below the horizon), the per-axis
 * screen derivatives multiplied by the pixel scales (kx, ky) =
 * (W/(2π), H/π) give the 2×3 Jacobian:
 *
 *   ∂screenX/∂(cx,cy,cz) = kx · ( cz/rxz²,  0,             -cx/rxz² )
 *   ∂screenY/∂(cx,cy,cz) = ky · (-cx·cy/(r²·rxz),  rxz/r²,  -cy·cz/(r²·rxz) )
 *
 * rxzClamped (>= POLE_EPS·r, set by the projection chunk) keeps every
 * denominator finite as a splat approaches the pole. j[0][1] = 0 but
 * j[1][0] != 0, so cov = J·Σ·Jᵀ carries the extra u00·jy0 / u10·jy0 /
 * u11·jy0 terms that the pinhole simplification dropped.
 */
const jacobianEquirect = /* wgsl */`
    let cov2 = cov2dEquirect(cx, cy, cz, r2, rxzClamped, c00, c01, c02, c11, c12, c22);
    var cov00 = cov2.x;
    var cov01 = cov2.y;
    var cov11 = cov2.z;
`;

/**
 * Module-scope function form of the equirect Jacobian + EWA projection,
 * returning `(cov00, cov01, cov11)`. The inline chunk above calls it for
 * the primary pose; the motion-blur path calls it again for the
 * shutter-close pose. Include once at module scope, before `jacobianEquirect`.
 */
const jacobianEquirectFns = /* wgsl */`
fn cov2dEquirect(
    cx: f32, cy: f32, cz: f32, r2: f32, rxzClamped: f32,
    c00: f32, c01: f32, c02: f32, c11: f32, c12: f32, c22: f32
) -> vec3<f32> {
    let kx = f32(uniforms.imageWidth) * 0.15915494309189535;
    let ky = f32(uniforms.imageHeight) * 0.3183098861837907;
    let invRxzC2 = 1.0 / (rxzClamped * rxzClamped);
    let invR2 = 1.0 / r2;
    let invR2Rxz = invR2 / rxzClamped;
    let jx0 =  kx * cz * invRxzC2;
    let jx2 = -kx * cx * invRxzC2;
    let jy0 = -ky * cx * cy * invR2Rxz;
    let jy1 =  ky * rxzClamped * invR2;
    let jy2 = -ky * cy * cz * invR2Rxz;

    let u00 = jx0 * c00 + jx2 * c02;
    let u01 = jx0 * c01 + jx2 * c12;
    let u02 = jx0 * c02 + jx2 * c22;
    let u10 = jy0 * c00 + jy1 * c01 + jy2 * c02;
    let u11 = jy0 * c01 + jy1 * c11 + jy2 * c12;
    let u12 = jy0 * c02 + jy1 * c12 + jy2 * c22;

    return vec3<f32>(
        u00 * jx0 + u02 * jx2,
        u00 * jy0 + u01 * jy1 + u02 * jy2,
        u10 * jy0 + u11 * jy1 + u12 * jy2
    );
}
`;

export { jacobianEquirect, jacobianEquirectFns };
