/**
 * Pinhole near-plane cull + perspective screen mapping.
 *
 * Reads:   cx, cy, cz, uniforms.near, uniforms.focalX, uniforms.focalY,
 *          uniforms.imageWidth, uniforms.imageHeight
 * Defines: invZ, screenX, screenY
 *
 * Splats with cz <= near are written invalid and the shader returns.
 * `screenX`/`screenY` are `var` so the motion-blur branch can move the
 * footprint centre to the shutter midpoint.
 */
const projectionPinhole = /* wgsl */`
    if (cz <= uniforms.near) { writeInvalid(i); return; }

    let invZ = 1.0 / cz;
    var screenX = uniforms.focalX * cx * invZ + f32(uniforms.imageWidth) * 0.5;
    var screenY = uniforms.focalY * cy * invZ + f32(uniforms.imageHeight) * 0.5;
`;

export { projectionPinhole };
