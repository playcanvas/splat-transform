import {
    type ChunkDataPool,
    type ChunkSource,
    type ChunkSourceMetadata,
    type ReadRequest,
    type SHBands,
    SH_REST_COUNTS,
    colorStride,
    colorFields
} from '../chunk';

/**
 * Reduce a source's SH band count (drop the higher-order coefficients), as a
 * lazy view — the streaming analog of `processDataTable`'s `filterBands`.
 *
 * Only the `color` layer changes: its stride shrinks and each record is repacked
 * to keep the DC term plus the low `outputBands` coefficients per channel;
 * `position`/`geometric`/`other` pass straight through. It is a band **drop**
 * only — if `outputBands >= src.meta.shBands` the source is returned unchanged
 * (matching the DataTable path, which can't add bands).
 *
 * SH rest is channel-major (`[R0..R(ic-1), G0.., B0..]`), so keeping the first
 * `oc` coefficients per channel is a strided copy, not a prefix; each record is
 * physically repacked (DC + kept coeffs) into the caller's reduced buffer. The
 * repack is a cheap per-chunk pass — no whole-scene materialization — and serves
 * both chunk and gather (indices) reads.
 *
 * @param src - Parent source.
 * @param outputBands - Target SH band count (0-3).
 * @param pool - Pool for the parent-band color temporaries; `chunkSize` must match the parent's.
 * @returns A derived source at the reduced band count (or `src` unchanged if it isn't a reduction).
 */
const reduceBandsSource = (src: ChunkSource, outputBands: SHBands, pool: ChunkDataPool): ChunkSource => {
    const inBands = src.meta.shBands;
    if (outputBands >= inBands) {
        return src; // band drop only — nothing to do
    }

    const inColorLayout = src.meta.layouts.color!;
    const outColorLayout = { stride: colorStride(outputBands), fields: colorFields(outputBands) };
    const meta: ChunkSourceMetadata = {
        ...src.meta,
        shBands: outputBands,
        layouts: { ...src.meta.layouts, color: outColorLayout }
    };

    const ic = SH_REST_COUNTS[inBands] / 3;      // input coefficients per channel
    const oc = SH_REST_COUNTS[outputBands] / 3;  // output coefficients per channel
    const inSw = inColorLayout.stride >>> 2;     // input color floats per record
    const outSw = outColorLayout.stride >>> 2;   // output color floats per record

    const read = async (request: ReadRequest): Promise<void> => {
        const outColor = request.color;
        if (!outColor) {
            // No color requested — forward verbatim (position/geometric/other).
            await src.read(request);
            return;
        }

        // Fill the other requested layers directly into the caller's buffers, and
        // the parent's full-band color into a temp; then repack temp -> reduced.
        const count = outColor.count;
        const tmp = pool.acquire('color', inColorLayout, count);
        await src.read({ ...request, color: tmp });

        const srcF = new Float32Array(tmp.data);
        const dstF = new Float32Array(outColor.data);
        for (let i = 0; i < count; i++) {
            const so = i * inSw;
            const dof = i * outSw;
            dstF[dof] = srcF[so];         // DC
            dstF[dof + 1] = srcF[so + 1];
            dstF[dof + 2] = srcF[so + 2];
            for (let c = 0; c < 3; c++) { // kept rest: first `oc` coeffs per channel (channel-major)
                for (let k = 0; k < oc; k++) {
                    dstF[dof + 3 + c * oc + k] = srcF[so + 3 + c * ic + k];
                }
            }
        }
        tmp.release();
    };

    return { meta, read, close: () => src.close() };
};

export { reduceBandsSource };
