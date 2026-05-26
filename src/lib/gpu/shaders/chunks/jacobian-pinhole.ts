/**
 * Pinhole Jacobian + 2D EWA covariance.
 *
 * Reads:   cx, cy, cz, invZ, uniforms.focalX, uniforms.focalY,
 *          uniforms.imageWidth, uniforms.imageHeight,
 *          c00, c01, c02, c11, c12, c22 (camera-space 3D covariance),
 *          JACOBIAN_LIMIT_FACTOR (from constants chunk)
 * Defines: cov00 (var), cov01 (let), cov11 (var)
 *
 * J is the 2×3 matrix
 *   [[jx0,   0, jx2],
 *    [  0, jy1, jy2]]
 * with x/z and y/z clamped to JACOBIAN_LIMIT_FACTOR · tan(half-FOV) so
 * splats outside the cone don't blow up the EWA approximation. The zero
 * entries (j[0][1] = 0, j[1][0] = 0) let us drop the u01·jy0 / u10 /
 * u11·jy0 terms in cov = J·Σ·Jᵀ that the equirect path retains.
 */
const jacobianPinhole = /* wgsl */`
    let limX = JACOBIAN_LIMIT_FACTOR * (f32(uniforms.imageWidth) * 0.5) / uniforms.focalX;
    let limY = JACOBIAN_LIMIT_FACTOR * (f32(uniforms.imageHeight) * 0.5) / uniforms.focalY;
    let txtz = clamp(cx * invZ, -limX, limX);
    let tytz = clamp(cy * invZ, -limY, limY);
    let jcx = txtz * cz;
    let jcy = tytz * cz;
    let jx0 = uniforms.focalX * invZ;
    let jx2 = -uniforms.focalX * jcx * invZ * invZ;
    let jy1 = uniforms.focalY * invZ;
    let jy2 = -uniforms.focalY * jcy * invZ * invZ;

    let u00 = jx0 * c00 + jx2 * c02;
    let u01 = jx0 * c01 + jx2 * c12;
    let u02 = jx0 * c02 + jx2 * c22;
    let u11 = jy1 * c11 + jy2 * c12;
    let u12 = jy1 * c12 + jy2 * c22;

    var cov00 = u00 * jx0 + u02 * jx2;
    let cov01 = u01 * jy1 + u02 * jy2;
    var cov11 = u11 * jy1 + u12 * jy2;
`;

export { jacobianPinhole };
