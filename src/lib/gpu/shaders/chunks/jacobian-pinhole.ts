/**
 * Pinhole Jacobian + 2D EWA covariance.
 *
 * Reads:   cx, cy, cz, invZ, uniforms.focalX, uniforms.focalY,
 *          uniforms.imageWidth, uniforms.imageHeight,
 *          c00, c01, c02, c11, c12, c22 (camera-space 3D covariance),
 *          JACOBIAN_LIMIT_FACTOR (from constants chunk)
 * Defines: cov00, cov01, cov11 (all var, so later dilation steps can add to them)
 * Requires: `jacobianPinholeFns` at module scope
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
    let cov2 = cov2dPinhole(cx, cy, cz, invZ, c00, c01, c02, c11, c12, c22);
    var cov00 = cov2.x;
    var cov01 = cov2.y;
    var cov11 = cov2.z;
`;

/**
 * Module-scope function form of the pinhole Jacobian + EWA projection,
 * returning `(cov00, cov01, cov11)`. The inline chunk above calls it for
 * the primary pose; the motion-blur path calls it again for the
 * shutter-close pose. Include once at module scope, before `jacobianPinhole`.
 */
const jacobianPinholeFns = /* wgsl */`
fn cov2dPinhole(
    cx: f32, cy: f32, cz: f32, invZ: f32,
    c00: f32, c01: f32, c02: f32, c11: f32, c12: f32, c22: f32
) -> vec3<f32> {
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

    return vec3<f32>(
        u00 * jx0 + u02 * jx2,
        u01 * jy1 + u02 * jy2,
        u11 * jy1 + u12 * jy2
    );
}
`;

export { jacobianPinhole, jacobianPinholeFns };
