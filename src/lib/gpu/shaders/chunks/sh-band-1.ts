/**
 * Spherical-harmonics degree-1 evaluation (3 coefficients per channel).
 *
 * Reads:   dirX, dirY, dirZ, s (source splat index), shc(), COEFFS_PER_CHANNEL,
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
        let b0 = -SH_C1 * dirY;
        let b1 = SH_C1 * dirZ;
        let b2 = -SH_C1 * dirX;
        cR = cR + b0 * shc(0u, s) + b1 * shc(1u, s) + b2 * shc(2u, s);
        cG = cG + b0 * shc(n + 0u, s) + b1 * shc(n + 1u, s) + b2 * shc(n + 2u, s);
        cB = cB + b0 * shc(2u * n + 0u, s) + b1 * shc(2u * n + 1u, s) + b2 * shc(2u * n + 2u, s);
    }
`;

export { shBand1 };
