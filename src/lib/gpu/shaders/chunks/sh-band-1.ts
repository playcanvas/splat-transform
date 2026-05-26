/**
 * Spherical-harmonics degree-1 evaluation (3 coefficients per channel).
 *
 * Reads:   dirX, dirY, dirZ, base, splats, COEFFS_PER_CHANNEL,
 *          SH_C1
 * Defines: (mutates) cR, cG, cB
 *
 * Appends the band-1 contribution to the accumulating per-channel color
 * radiance. Channel-major SH layout: `f_rest_0..N-1` red, then green,
 * then blue.
 */
const shBand1 = /* wgsl */`
    {
        let n = COEFFS_PER_CHANNEL;
        let shBase = base + 14u;
        let b0 = -SH_C1 * dirY;
        let b1 = SH_C1 * dirZ;
        let b2 = -SH_C1 * dirX;
        cR = cR + b0 * splats[shBase + 0u] + b1 * splats[shBase + 1u] + b2 * splats[shBase + 2u];
        cG = cG + b0 * splats[shBase + n + 0u] + b1 * splats[shBase + n + 1u] + b2 * splats[shBase + n + 2u];
        cB = cB + b0 * splats[shBase + 2u * n + 0u] + b1 * splats[shBase + 2u * n + 1u] + b2 * splats[shBase + 2u * n + 2u];
    }
`;

export { shBand1 };
