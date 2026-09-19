/**
 * Quantize a colour to a packed RGBA8 u32 (bytes R, G, B, A from the low
 * end): clamp to [0, 1], scale to 255 and round half up. Shared by the
 * finalize and accumulate shaders so a single-slice frame and the last
 * slice of a blurred one quantize identically.
 */
const packRGBA8 = /* wgsl */`
fn packRGBA8(c: vec4<f32>) -> u32 {
    let q = vec4<u32>(clamp(c, vec4<f32>(0.0), vec4<f32>(1.0)) * 255.0 + 0.5);
    return q.r | (q.g << 8u) | (q.b << 16u) | (q.a << 24u);
}
`;

export { packRGBA8 };
