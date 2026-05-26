/**
 * Pinhole near-plane cull + perspective screen mapping.
 *
 * Reads:   cx, cy, cz, uniforms.near, uniforms.focalX, uniforms.focalY,
 *          uniforms.imageWidth, uniforms.imageHeight
 * Defines: invZ, screenX, screenY
 *
 * Splats with cz <= near are written invalid and the shader returns.
 */
const projectionPinhole = /* wgsl */`
    if (cz <= uniforms.near) { writeInvalid(i); return; }

    let invZ = 1.0 / cz;
    let screenX = uniforms.focalX * cx * invZ + f32(uniforms.imageWidth) * 0.5;
    let screenY = uniforms.focalY * cy * invZ + f32(uniforms.imageHeight) * 0.5;
`;

export { projectionPinhole };
