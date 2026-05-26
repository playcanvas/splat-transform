/**
 * Equirect emit-pairs tile walk with ±π longitude seam wrap.
 *
 * Reads:   sX, sY, radius, tsz, gox, goy, cap, i,
 *          uniforms.groupTilesX, uniforms.groupTilesY
 * Defines: writes (tileKeys, splatValues) pairs for this splat's bbox
 * Returns: early if the bbox is empty
 *
 * Recomputes the raw X range (possibly wrapping past the seam) — must
 * match the project shader's `tile-aabb-equirect` coverage computation
 * exactly. Each emitted tx is wrapped into [0, groupTilesX-1] via
 * modular arithmetic. The rasterize-binned shader compensates by
 * wrapping its per-pixel dx into [-W/2, W/2], so a wrapped tile pulls
 * the splat's footprint from the correct copy across the seam.
 */
const tileWalkEquirect = /* wgsl */`
    let minTXraw = i32(floor((sX - radius - gox) / tsz));
    let maxTXraw = i32(floor((sX + radius - gox) / tsz));
    let txCountRaw = maxTXraw - minTXraw + 1;
    let groupTilesX_i = i32(uniforms.groupTilesX);
    let txCount = min(txCountRaw, groupTilesX_i);
    let minTY = max(0, i32(floor((sY - radius - goy) / tsz)));
    let maxTY = min(i32(uniforms.groupTilesY) - 1, i32(floor((sY + radius - goy) / tsz)));
    if (txCount <= 0 || maxTY < minTY) { return; }

    var slot = emitOffset[i];
    let end = slot + cap;
    for (var ty: i32 = minTY; ty <= maxTY; ty = ty + 1) {
        if (slot >= end) { break; }
        for (var k: i32 = 0; k < txCount; k = k + 1) {
            if (slot >= end) { break; }
            var tx = (minTXraw + k) % groupTilesX_i;
            if (tx < 0) { tx = tx + groupTilesX_i; }
            let t = u32(ty) * uniforms.groupTilesX + u32(tx);
            tileKeys[slot] = t;
            splatValues[slot] = i;
            slot = slot + 1u;
        }
    }
`;

export { tileWalkEquirect };
