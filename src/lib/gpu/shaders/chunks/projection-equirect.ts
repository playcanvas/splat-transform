/**
 * Equirect near-radius cull + spherical (atan2/asin) screen mapping +
 * pole-clamp setup used by the equirect Jacobian.
 *
 * Reads:   cx, cy, cz, uniforms.near, uniforms.imageWidth, uniforms.imageHeight,
 *          POLE_EPS (from constants chunk)
 * Defines: r2, r, rxz2, rxz, rxzClamped, invTwoPi, invPi, imgWf, imgHf,
 *          lon, sinLat, lat, screenX, screenY
 *
 * Splats inside the near sphere (r <= near) are written invalid and the
 * shader returns. The longitude atan2 is undefined at the camera origin
 * and degenerates near the poles where rxz → 0; rxzClamped (>= POLE_EPS·r)
 * keeps every denominator in the Jacobian chunk finite for splats
 * arbitrarily close to the zenith / nadir.
 *
 * Convention: cy is the camera-down axis, so cy > 0 = below horizon →
 * lat > 0 → screenY in the bottom half. screenY = 0 maps to the zenith
 * (above the camera).
 */
const projectionEquirect = /* wgsl */`
    let r2 = cx * cx + cy * cy + cz * cz;
    if (r2 <= uniforms.near * uniforms.near) { writeInvalid(i); return; }
    let r = sqrt(r2);
    let rxz2 = cx * cx + cz * cz;
    let rxz = sqrt(rxz2);
    let rxzClamped = max(rxz, POLE_EPS * r);

    let invTwoPi: f32 = 0.15915494309189535;
    let invPi:    f32 = 0.3183098861837907;
    let imgWf = f32(uniforms.imageWidth);
    let imgHf = f32(uniforms.imageHeight);
    let lon = atan2(cx, cz);
    let sinLat = clamp(cy / r, -1.0, 1.0);
    let lat = asin(sinLat);
    let screenX = (lon * invTwoPi + 0.5) * imgWf;
    let screenY = (lat * invPi + 0.5) * imgHf;
`;

export { projectionEquirect };
