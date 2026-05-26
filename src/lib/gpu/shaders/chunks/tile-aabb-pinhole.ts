/**
 * Pinhole per-splat tile-coverage count.
 *
 * Reads:   screenX, screenY, radius, tsz, gox, goy,
 *          uniforms.groupTilesX, uniforms.groupTilesY
 * Defines: writes coverage[i]
 *
 * Computes the splat's clamped tile bbox and stores the area
 * (`(maxTX - minTX + 1) · (maxTY - minTY + 1)`, capped at
 * MAX_COVERAGE_PER_SPLAT) into `coverage[i]`. The emit-pairs shader
 * later walks the same clamped bbox via its own pinhole tile-walk
 * chunk.
 *
 * Embeds the per-render MAX_COVERAGE_PER_SPLAT cap via JS-template
 * substitution because the value is fixed at shader-construction time
 * (it tracks the group's tile area).
 *
 * @param maxCoveragePerSplat - Hard upper bound on per-splat tile count.
 * @returns WGSL source for the pinhole tile-coverage block.
 */
const tileAabbPinhole = (maxCoveragePerSplat: number) => /* wgsl */`
    let minTX = max(0, i32(floor((screenX - radius - gox) / tsz)));
    let maxTX = min(i32(uniforms.groupTilesX) - 1, i32(floor((screenX + radius - gox) / tsz)));
    let minTY = max(0, i32(floor((screenY - radius - goy) / tsz)));
    let maxTY = min(i32(uniforms.groupTilesY) - 1, i32(floor((screenY + radius - goy) / tsz)));
    if (maxTX < minTX || maxTY < minTY) {
        coverage[i] = 0u;
    } else {
        let raw = u32(maxTX - minTX + 1) * u32(maxTY - minTY + 1);
        coverage[i] = min(raw, ${maxCoveragePerSplat}u);
    }
`;

export { tileAabbPinhole };
