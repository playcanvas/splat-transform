/**
 * 24-bit quaternion quantization via the exponential map (SOG v3 rotations).
 *
 * A unit quaternion with `w >= 0` (negate otherwise: `q` and `-q` are the same
 * rotation) is mapped to `p = (2/π)·θ·v/|v|`, where `θ = atan2(|v|, w)` is half
 * the rotation angle and `v = (x, y, z)`. `p` lies in the unit ball, and each
 * component is stored in one byte on a mid-tread lattice `(b − 128)/127`, so
 * the identity rotation round-trips exactly. The three bytes are stored in the
 * RGB of `quats.webp`.
 *
 * Measured over 2M uniform random rotations: max error 1.20°, RMS 0.62° — better
 * than SOG v2's smallest-three as shipped (2.03° / 0.71°) in one byte less.
 * See https://marc-b-reynolds.github.io/quaternions/2017/05/02/QuatQuantPart1.html
 */

const HALF_PI = Math.PI / 2;

const toByte = (v: number) => Math.max(0, Math.min(255, Math.round(v * 127 + 128)));

/**
 * Encode a quaternion (any length, any sign) into three exp-map bytes.
 *
 * @param w - Scalar component.
 * @param x - Vector x component.
 * @param y - Vector y component.
 * @param z - Vector z component.
 * @param out - Destination for the three bytes.
 * @param offset - Index of the first byte in `out`.
 */
const encodeQuatExpMap = (w: number, x: number, y: number, z: number, out: Uint8Array, offset: number): void => {
    const len = Math.sqrt(w * w + x * x + y * y + z * z);
    if (!(len > 0)) {
        out[offset] = out[offset + 1] = out[offset + 2] = 128;
        return;
    }
    const s = w < 0 ? -1 / len : 1 / len;
    w *= s;
    x *= s;
    y *= s;
    z *= s;
    const sv = Math.sqrt(x * x + y * y + z * z);
    // θ/|v| → 1/w as |v| → 0, and w → 1 there
    const k = sv > 1e-12 ? Math.atan2(sv, w) / sv / HALF_PI : 1 / HALF_PI;
    out[offset] = toByte(x * k);
    out[offset + 1] = toByte(y * k);
    out[offset + 2] = toByte(z * k);
};

/**
 * Decode three exp-map bytes into a unit quaternion `[w, x, y, z]`.
 *
 * @param a - Byte for p.x.
 * @param b - Byte for p.y.
 * @param c - Byte for p.z.
 * @param out - Destination for `[w, x, y, z]`.
 * @param offset - Index of `w` in `out`.
 */
const decodeQuatExpMap = (a: number, b: number, c: number, out: Float32Array, offset: number): void => {
    const px = (a - 128) / 127;
    const py = (b - 128) / 127;
    const pz = (c - 128) / 127;
    const r = Math.sqrt(px * px + py * py + pz * pz);
    const t = HALF_PI * r;
    const k = r > 1e-12 ? Math.sin(t) / r : HALF_PI;
    out[offset] = Math.cos(t);
    out[offset + 1] = px * k;
    out[offset + 2] = py * k;
    out[offset + 3] = pz * k;
};

export { encodeQuatExpMap, decodeQuatExpMap };
