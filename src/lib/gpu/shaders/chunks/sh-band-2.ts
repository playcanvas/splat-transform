/**
 * Spherical-harmonics degree-2 evaluation (5 additional coefficients per
 * channel, indices [3..7]).
 *
 * Reads:   dirX, dirY, dirZ, base, splats, COEFFS_PER_CHANNEL,
 *          SH_C2_0, SH_C2_1, SH_C2_2, SH_C2_3, SH_C2_4
 * Defines: (mutates) cR, cG, cB
 */
const shBand2 = /* wgsl */`
    {
        let n = COEFFS_PER_CHANNEL;
        let shBase = base + 14u;
        let xx2 = dirX * dirX;
        let yy2 = dirY * dirY;
        let zz2 = dirZ * dirZ;
        let xy2 = dirX * dirY;
        let yz2 = dirY * dirZ;
        let xz2 = dirX * dirZ;
        let b3 = SH_C2_0 * xy2;
        let b4 = SH_C2_1 * yz2;
        let b5 = SH_C2_2 * (2.0 * zz2 - xx2 - yy2);
        let b6 = SH_C2_3 * xz2;
        let b7 = SH_C2_4 * (xx2 - yy2);
        cR = cR + b3 * splats[shBase + 3u] + b4 * splats[shBase + 4u] + b5 * splats[shBase + 5u] + b6 * splats[shBase + 6u] + b7 * splats[shBase + 7u];
        cG = cG + b3 * splats[shBase + n + 3u] + b4 * splats[shBase + n + 4u] + b5 * splats[shBase + n + 5u] + b6 * splats[shBase + n + 6u] + b7 * splats[shBase + n + 7u];
        cB = cB + b3 * splats[shBase + 2u * n + 3u] + b4 * splats[shBase + 2u * n + 4u] + b5 * splats[shBase + 2u * n + 5u] + b6 * splats[shBase + 2u * n + 6u] + b7 * splats[shBase + 2u * n + 7u];
    }
`;

export { shBand2 };
