/**
 * Equirect Jacobian + 2D EWA covariance.
 *
 * Reads:   cx, cy, cz, r2, rxz, rxzClamped, imgWf, imgHf, invTwoPi, invPi,
 *          c00, c01, c02, c11, c12, c22 (camera-space 3D covariance)
 * Defines: cov00 (var), cov01 (let), cov11 (var)
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
    let kx = imgWf * invTwoPi;
    let ky = imgHf * invPi;
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

    var cov00 = u00 * jx0 + u02 * jx2;
    let cov01 = u00 * jy0 + u01 * jy1 + u02 * jy2;
    var cov11 = u10 * jy0 + u11 * jy1 + u12 * jy2;
`;

export { jacobianEquirect };
