/**
 * Spherical-harmonics degree-3 evaluation (7 additional coefficients per
 * channel, indices [8..14]).
 *
 * Reads:   dirX, dirY, dirZ, base, splats, COEFFS_PER_CHANNEL,
 *          SH_C3_0, SH_C3_1, SH_C3_2, SH_C3_3, SH_C3_4, SH_C3_5, SH_C3_6
 * Defines: (mutates) cR, cG, cB
 */
const shBand3 = /* wgsl */`
    {
        let n = COEFFS_PER_CHANNEL;
        let shBase = base + 14u;
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
        cR = cR + b8 * splats[shBase + 8u] + b9 * splats[shBase + 9u] + b10 * splats[shBase + 10u] + b11 * splats[shBase + 11u] + b12 * splats[shBase + 12u] + b13 * splats[shBase + 13u] + b14 * splats[shBase + 14u];
        cG = cG + b8 * splats[shBase + n + 8u] + b9 * splats[shBase + n + 9u] + b10 * splats[shBase + n + 10u] + b11 * splats[shBase + n + 11u] + b12 * splats[shBase + n + 12u] + b13 * splats[shBase + n + 13u] + b14 * splats[shBase + n + 14u];
        cB = cB + b8 * splats[shBase + 2u * n + 8u] + b9 * splats[shBase + 2u * n + 9u] + b10 * splats[shBase + 2u * n + 10u] + b11 * splats[shBase + 2u * n + 11u] + b12 * splats[shBase + 2u * n + 12u] + b13 * splats[shBase + 2u * n + 13u] + b14 * splats[shBase + 2u * n + 14u];
    }
`;

export { shBand3 };
