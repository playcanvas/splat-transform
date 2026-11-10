/**
 * Spherical-harmonics degree-4 evaluation (9 additional coefficients per
 * channel, indices [15..23]). Same basis as the engine's gsplatEvalSH and
 * gsplat's sh_utils; the direction is unit length, so the r² factors of the
 * basis are dropped.
 *
 * Reads:   dirX, dirY, dirZ, s (source splat index), shc(), COEFFS_PER_CHANNEL,
 *          SH_C4_0 .. SH_C4_8
 * Defines: (mutates) cR, cG, cB
 */
const shBand4 = /* wgsl */ `
    {
        let n = COEFFS_PER_CHANNEL;
        let xx4 = dirX * dirX;
        let yy4 = dirY * dirY;
        let zz4 = dirZ * dirZ;
        let xy4 = dirX * dirY;
        let yz4 = dirY * dirZ;
        let xz4 = dirX * dirZ;
        var b4 = array<f32, 9>(
            SH_C4_0 * xy4 * (xx4 - yy4),
            SH_C4_1 * yz4 * (3.0 * xx4 - yy4),
            SH_C4_2 * xy4 * (7.0 * zz4 - 1.0),
            SH_C4_3 * yz4 * (7.0 * zz4 - 3.0),
            SH_C4_4 * (zz4 * (35.0 * zz4 - 30.0) + 3.0),
            SH_C4_5 * xz4 * (7.0 * zz4 - 3.0),
            SH_C4_6 * (xx4 - yy4) * (7.0 * zz4 - 1.0),
            SH_C4_7 * xz4 * (xx4 - 3.0 * yy4),
            SH_C4_8 * (xx4 * (xx4 - 3.0 * yy4) - yy4 * (3.0 * xx4 - yy4))
        );
        for (var k = 0u; k < 9u; k = k + 1u) {
            let b = b4[k];
            cR = cR + b * shc(15u + k, s);
            cG = cG + b * shc(n + 15u + k, s);
            cB = cB + b * shc(2u * n + 15u + k, s);
        }
    }
`;

export { shBand4 };
