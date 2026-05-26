/**
 * Equirect per-splat tile-coverage count.
 *
 * Reads:   screenX, screenY, radius, tsz, gox, goy,
 *          uniforms.groupTilesX, uniforms.groupTilesY
 * Defines: writes coverage[i]
 *
 * The X tile range can extend past the image edges into negative or
 * `> groupTilesX-1` indices — those represent the same splat seen
 * across the ±π longitude seam. Coverage is the raw X span (capped at
 * groupTilesX so a splat with radius > image_width doesn't emit
 * duplicate tile keys); emit-pairs walks [minTXraw .. maxTXraw] in
 * lock-step and applies a modular wrap when writing tile keys. Y is
 * clamped normally — equirect doesn't wrap across poles.
 *
 * @param maxCoveragePerSplat - Hard upper bound on per-splat tile count.
 * @returns WGSL source for the equirect tile-coverage block.
 */
const tileAabbEquirect = (maxCoveragePerSplat: number) => /* wgsl */`
    let minTXraw = i32(floor((screenX - radius - gox) / tsz));
    let maxTXraw = i32(floor((screenX + radius - gox) / tsz));
    let txCountRaw = maxTXraw - minTXraw + 1;
    let txCount = min(txCountRaw, i32(uniforms.groupTilesX));
    let minTY = max(0, i32(floor((screenY - radius - goy) / tsz)));
    let maxTY = min(i32(uniforms.groupTilesY) - 1, i32(floor((screenY + radius - goy) / tsz)));
    if (txCount <= 0 || maxTY < minTY) {
        coverage[i] = 0u;
    } else {
        let raw = u32(txCount) * u32(maxTY - minTY + 1);
        coverage[i] = min(raw, ${maxCoveragePerSplat}u);
    }
`;

export { tileAabbEquirect };
