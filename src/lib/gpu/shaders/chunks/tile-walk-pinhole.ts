/**
 * Pinhole emit-pairs tile walk.
 *
 * Reads:   sX, sY, radius, tsz, gox, goy, cap, i,
 *          uniforms.groupTilesX, uniforms.groupTilesY
 * Defines: writes (tileKeys, splatValues) pairs for this splat's bbox
 * Returns: early if the bbox is empty
 *
 * Walks the clamped per-splat tile bbox row-major and writes (tile,
 * splat) pair entries until either the bbox is exhausted or the
 * `coverage[i]` cap is hit. Mirrors the bbox computed by the pinhole
 * project shader's `tile-aabb-pinhole` chunk.
 */
const tileWalkPinhole = /* wgsl */`
    let minTX = max(0, i32(floor((sX - radius - gox) / tsz)));
    let maxTX = min(i32(uniforms.groupTilesX) - 1, i32(floor((sX + radius - gox) / tsz)));
    let minTY = max(0, i32(floor((sY - radius - goy) / tsz)));
    let maxTY = min(i32(uniforms.groupTilesY) - 1, i32(floor((sY + radius - goy) / tsz)));
    if (maxTX < minTX || maxTY < minTY) { return; }

    var slot = emitOffset[i];
    let end = slot + cap;
    for (var ty: i32 = minTY; ty <= maxTY; ty = ty + 1) {
        if (slot >= end) { break; }
        for (var tx: i32 = minTX; tx <= maxTX; tx = tx + 1) {
            if (slot >= end) { break; }
            let t = u32(ty) * uniforms.groupTilesX + u32(tx);
            tileKeys[slot] = t;
            splatValues[slot] = i;
            slot = slot + 1u;
        }
    }
`;

export { tileWalkPinhole };
