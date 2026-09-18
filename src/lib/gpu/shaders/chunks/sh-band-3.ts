/**
 * Spherical-harmonics degree-3 evaluation (7 additional coefficients per
 * channel, indices [8..14]).
 *
 * Reads:   dirX, dirY, dirZ, s (source splat index), shc(), COEFFS_PER_CHANNEL,
 *          SH_C3_0, SH_C3_1, SH_C3_2, SH_C3_3, SH_C3_4, SH_C3_5, SH_C3_6
 * Defines: (mutates) cR, cG, cB
 */
const shBand3 = /* wgsl */`
    {
        let n = COEFFS_PER_CHANNEL;
        let xx2 = dirX * dirX;
        let yy2 = dirY * dirY;
        let zz2 = dirZ * dirZ;
        let xy2 = dirX * dirY;
        let b8 = SH_C3_0 * dirY * (3.0 * xx2 - yy2);
        let b9 = SH_C3_1 * xy2 * dirZ;
        let b10 = SH_C3_2 * dirY * (4.0 * zz2 - xx2 - yy2);
        let b11 = SH_C3_3 * dirZ * (2.0 * zz2 - 3.0 * xx2 - 3.0 * yy2);
        let b12 = SH_C3_4 * dirX * (4.0 * zz2 - xx2 - yy2);
        let b13 = SH_C3_5 * dirZ * (xx2 - yy2);
        let b14 = SH_C3_6 * dirX * (xx2 - 3.0 * yy2);
        cR = cR + b8 * shc(8u, s) + b9 * shc(9u, s) + b10 * shc(10u, s) + b11 * shc(11u, s) + b12 * shc(12u, s) + b13 * shc(13u, s) + b14 * shc(14u, s);
        cG = cG + b8 * shc(n + 8u, s) + b9 * shc(n + 9u, s) + b10 * shc(n + 10u, s) + b11 * shc(n + 11u, s) + b12 * shc(n + 12u, s) + b13 * shc(n + 13u, s) + b14 * shc(n + 14u, s);
        cB = cB + b8 * shc(2u * n + 8u, s) + b9 * shc(2u * n + 9u, s) + b10 * shc(2u * n + 10u, s) + b11 * shc(2u * n + 11u, s) + b12 * shc(2u * n + 12u, s) + b13 * shc(2u * n + 13u, s) + b14 * shc(2u * n + 14u, s);
    }
`;

export { shBand3 };
