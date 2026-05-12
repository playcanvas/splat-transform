import { Column, DataTable } from '../data-table';
import { join, type ReadFileSystem } from '../io/read';
import { logger, Transform, WebPCodec } from '../utils';

// V1 (legacy) SOG meta layout. Quantization is a per-channel linear lerp
// between mins/maxs, with no codebook. The engine's SOG parser still loads
// these (with a deprecation warning), so we mirror its decoding so older
// published assets keep working through splat-transform.
//
// The current (V2) format is handled in read-sog.ts; the public readSog
// dispatcher there falls back to readSogV1 when it sees a meta without
// `version: 2`. Keeping V1 isolated here lets both code paths stay free of
// version branching, and makes deleting V1 support trivial when the legacy
// data is no longer in circulation.
type MetaV1 = {
    means: {
        shape: [number, number];
        mins: number[];
        maxs: number[];
        files: string[];
    };
    scales: { mins: number[]; maxs: number[]; files: string[] };
    quats: { files: string[] };
    sh0: { mins: number[]; maxs: number[]; files: string[] };
    shN?: {
        shape?: [number, number];
        mins: number;
        maxs: number;
        quantization?: number;
        files: string[];
    };
};

const decodeMeans = (lo: Uint8Array, hi: Uint8Array, count: number) => {
    const xs = new Uint16Array(count);
    const ys = new Uint16Array(count);
    const zs = new Uint16Array(count);
    for (let i = 0; i < count; i++) {
        const o = i * 4;
        xs[i] = lo[o + 0] | (hi[o + 0] << 8);
        ys[i] = lo[o + 1] | (hi[o + 1] << 8);
        zs[i] = lo[o + 2] | (hi[o + 2] << 8);
    }
    return { xs, ys, zs };
};

// Inverse of logTransform(x) = sign(x) * ln(|x| + 1)
const invLogTransform = (v: number) => {
    const a = Math.abs(v);
    const e = Math.exp(a) - 1; // |x|
    return v < 0 ? -e : e;
};

const unpackQuat = (px: number, py: number, pz: number, tag: number): [number, number, number, number] => {
    const maxComp = tag - 252;
    const a = px / 255 * 2 - 1;
    const b = py / 255 * 2 - 1;
    const c = pz / 255 * 2 - 1;
    const sqrt2 = Math.sqrt(2);
    const comps = [0, 0, 0, 0];
    const idx = [
        [1, 2, 3],
        [0, 2, 3],
        [0, 1, 3],
        [0, 1, 2]
    ][maxComp];
    comps[idx[0]] = a / sqrt2;
    comps[idx[1]] = b / sqrt2;
    comps[idx[2]] = c / sqrt2;
    // reconstruct max component to make unit length with positive sign
    const t = 1 - (comps[0] * comps[0] + comps[1] * comps[1] + comps[2] * comps[2] + comps[3] * comps[3]);
    comps[maxComp] = Math.sqrt(Math.max(0, t));
    return comps as [number, number, number, number];
};

// Centroids texture width -> SH band count. The palette packs 64 entries per
// row, each `shCoeffs` pixels wide, so width = 64 * shCoeffs. V1 has no
// `bands` field in meta, so we infer from the texture geometry. Mapping
// mirrors the engine's GSplatSogData.calcBands.
const v1ShBandsWidths: Record<number, number> = { 192: 1, 512: 2, 960: 3 };

/**
 * Read a legacy V1 SOG file from a ReadFileSystem.
 *
 * Called by readSog (in read-sog.ts) after it detects a V1 meta payload.
 * Receives the already-parsed meta and the directory it was loaded from so
 * we don't re-fetch the JSON.
 *
 * @param fileSystem - The file system to read texture files from
 * @param baseDir - Directory containing the SOG textures (relative paths in
 * meta are resolved from here)
 * @param meta - The parsed V1 meta.json payload
 * @returns DataTable with Gaussian splat data
 * @ignore
 */
const readSogV1 = async (fileSystem: ReadFileSystem, baseDir: string, meta: MetaV1): Promise<DataTable> => {
    const decoder = await WebPCodec.create();
    const count = meta.means.shape[0];

    const load = async (name: string): Promise<Uint8Array> => {
        const src = await fileSystem.createSource(baseDir ? join(baseDir, name) : name);
        try {
            return await src.read().readAll();
        } finally {
            src.close();
        }
    };

    const columns: Column[] = [
        new Column('x', new Float32Array(count)),
        new Column('y', new Float32Array(count)),
        new Column('z', new Float32Array(count)),
        new Column('scale_0', new Float32Array(count)),
        new Column('scale_1', new Float32Array(count)),
        new Column('scale_2', new Float32Array(count)),
        new Column('f_dc_0', new Float32Array(count)),
        new Column('f_dc_1', new Float32Array(count)),
        new Column('f_dc_2', new Float32Array(count)),
        new Column('opacity', new Float32Array(count)),
        new Column('rot_0', new Float32Array(count)),
        new Column('rot_1', new Float32Array(count)),
        new Column('rot_2', new Float32Array(count)),
        new Column('rot_3', new Float32Array(count))
    ];

    const numPasses = 4 + (meta.shN ? 1 : 0);
    const bar = logger.bar('decoding', numPasses * count);
    let passesDone = 0;
    const tickPass = () => {
        bar.update(++passesDone * count);
    };

    // means: two textures means_l (low byte) + means_u (high byte) packed as
    // a 16-bit lerp between mins/maxs of the logTransform'd positions.
    const meansLoWebp = await load(meta.means.files[0]);
    const meansHiWebp = await load(meta.means.files[1]);
    const { rgba: lo, width, height } = decoder.decodeRGBA(meansLoWebp);
    const { rgba: hi } = decoder.decodeRGBA(meansHiWebp);
    if (width * height < count) throw new Error('SOG means texture too small for count');
    const { mins, maxs } = meta.means;
    const { xs, ys, zs } = decodeMeans(lo, hi, count);
    const xCol = columns[0].data as Float32Array;
    const yCol = columns[1].data as Float32Array;
    const zCol = columns[2].data as Float32Array;
    const xMin = mins[0], xScale = (maxs[0] - mins[0]) || 1;
    const yMin = mins[1], yScale = (maxs[1] - mins[1]) || 1;
    const zMin = mins[2], zScale = (maxs[2] - mins[2]) || 1;
    for (let i = 0; i < count; i++) {
        xCol[i] = invLogTransform(xMin + xScale * (xs[i] / 65535));
        yCol[i] = invLogTransform(yMin + yScale * (ys[i] / 65535));
        zCol[i] = invLogTransform(zMin + zScale * (zs[i] / 65535));
    }
    tickPass();

    // quats: 4 bytes per splat, last byte is the "largest component" tag
    // (252-255 -> w/x/y/z is largest); other 3 encode the smaller components
    // scaled by sqrt(2). Identical to V2.
    const quatsWebp = await load(meta.quats.files[0]);
    const { rgba: qr, width: qw, height: qh } = decoder.decodeRGBA(quatsWebp);
    if (qw * qh < count) throw new Error('SOG quats texture too small for count');
    const r0 = columns[10].data as Float32Array;
    const r1 = columns[11].data as Float32Array;
    const r2 = columns[12].data as Float32Array;
    const r3 = columns[13].data as Float32Array;
    for (let i = 0; i < count; i++) {
        const o = i * 4;
        const tag = qr[o + 3];
        if (tag < 252 || tag > 255) { // invalid tag, default to identity (rot_0 = w)
            r0[i] = 1; r1[i] = 0; r2[i] = 0; r3[i] = 0;
            continue;
        }
        // unpackQuat returns components in (w, x, y, z) order; rot_0..rot_3 map to (w, x, y, z).
        const [w, x, y, z] = unpackQuat(qr[o], qr[o + 1], qr[o + 2], tag);
        r0[i] = w; r1[i] = x; r2[i] = y; r3[i] = z;
    }
    tickPass();

    // scales: per-axis 8-bit linear lerp between mins/maxs (log-space).
    const scalesWebp = await load(meta.scales.files[0]);
    const { rgba: sl, width: sw, height: sh } = decoder.decodeRGBA(scalesWebp);
    if (sw * sh < count) throw new Error('SOG scales texture too small for count');
    const sMins = meta.scales.mins;
    const sMaxs = meta.scales.maxs;
    const s0 = columns[3].data as Float32Array;
    const s1 = columns[4].data as Float32Array;
    const s2 = columns[5].data as Float32Array;
    for (let i = 0; i < count; i++) {
        const o = i * 4;
        s0[i] = sMins[0] + (sMaxs[0] - sMins[0]) * (sl[o + 0] / 255);
        s1[i] = sMins[1] + (sMaxs[1] - sMins[1]) * (sl[o + 1] / 255);
        s2[i] = sMins[2] + (sMaxs[2] - sMins[2]) * (sl[o + 2] / 255);
    }
    tickPass();

    // sh0: 3 color channels lerped between mins[0..2]/maxs[0..2]; opacity is
    // a *pre-sigmoid logit* lerped between mins[3]/maxs[3] and written
    // straight into the opacity column (no sigmoid round-trip needed).
    const sh0Webp = await load(meta.sh0.files[0]);
    const { rgba: c0, width: cw, height: ch } = decoder.decodeRGBA(sh0Webp);
    if (cw * ch < count) throw new Error('SOG sh0 texture too small for count');
    const cMins = meta.sh0.mins;
    const cMaxs = meta.sh0.maxs;
    const dc0 = columns[6].data as Float32Array;
    const dc1 = columns[7].data as Float32Array;
    const dc2 = columns[8].data as Float32Array;
    const opCol = columns[9].data as Float32Array;
    for (let i = 0; i < count; i++) {
        const o = i * 4;
        dc0[i] = cMins[0] + (cMaxs[0] - cMins[0]) * (c0[o + 0] / 255);
        dc1[i] = cMins[1] + (cMaxs[1] - cMins[1]) * (c0[o + 1] / 255);
        dc2[i] = cMins[2] + (cMaxs[2] - cMins[2]) * (c0[o + 2] / 255);
        opCol[i] = cMins[3] + (cMaxs[3] - cMins[3]) * (c0[o + 3] / 255);
    }
    tickPass();

    // shN: indirect lookup via labels -> centroids palette, with each centroid
    // pixel byte lerped between scalar mins/maxs that span all SH coeffs.
    if (meta.shN) {
        const centroidsWebp = await load(meta.shN.files[0]);
        const labelsWebp = await load(meta.shN.files[1]);
        const { rgba: centroidsRGBA, width: cW, height: cH } = decoder.decodeRGBA(centroidsWebp);
        const { rgba: labelsRGBA, width: lW, height: lH } = decoder.decodeRGBA(labelsWebp);

        // Validate label texture size up-front: out-of-bounds typed-array reads
        // coerce to 0 via the bitwise ops below, which would silently map many
        // splats to centroid 0 instead of failing.
        if (lW * lH < count) throw new Error('SOG shN labels texture too small for count');

        // Centroids width determines the band count (palette packs 64 entries
        // per row, each shCoeffs pixels wide). An unrecognized width means the
        // shN payload is malformed; throw rather than silently skipping.
        const bands = v1ShBandsWidths[cW] ?? 0;
        const shCoeffs = [0, 3, 8, 15][bands];
        if (bands === 0) throw new Error(`SOG shN centroids texture has unrecognized width ${cW}, expected one of 192 / 512 / 960`);

        if (shCoeffs > 0) {
            const shMin = meta.shN.mins;
            const shSpan = meta.shN.maxs - meta.shN.mins;
            const dequant = (b: number) => shMin + shSpan * (b / 255);

            // Upper-bound palette guard: V1 has no explicit count, so use the
            // texture geometry (`floor(cW / shCoeffs) * cH` palette slots).
            const paletteCount = Math.floor(cW / shCoeffs) * cH;

            const baseIdx = columns.length;
            for (let i = 0; i < shCoeffs * 3; i++) {
                columns.push(new Column(`f_rest_${i}`, new Float32Array(count)));
            }

            const stride = 4;
            const getCentroidPixel = (centroidIndex: number, coeff: number) => {
                const cx = (centroidIndex % 64) * shCoeffs + coeff;
                const cy = Math.floor(centroidIndex / 64);
                if (cx >= cW || cy >= cH) return [0, 0, 0] as [number, number, number];
                const idx = (cy * cW + cx) * stride;
                return [centroidsRGBA[idx], centroidsRGBA[idx + 1], centroidsRGBA[idx + 2]] as [number, number, number];
            };

            for (let i = 0; i < count; i++) {
                const o = i * 4;
                const label = labelsRGBA[o] | (labelsRGBA[o + 1] << 8); // 16-bit palette index
                if (label >= paletteCount) continue; // safety
                for (let j = 0; j < shCoeffs; j++) {
                    const [lr, lg, lb] = getCentroidPixel(label, j);
                    (columns[baseIdx + j + shCoeffs * 0].data as Float32Array)[i] = dequant(lr);
                    (columns[baseIdx + j + shCoeffs * 1].data as Float32Array)[i] = dequant(lg);
                    (columns[baseIdx + j + shCoeffs * 2].data as Float32Array)[i] = dequant(lb);
                }
            }
        }
        tickPass();
    }

    // Close the bar only on success: leaving it open on the error path lets
    // `logger.error() -> unwindAll(true)` mark it as failed instead of
    // finalizing it as a successful bar first.
    bar.end();

    return new DataTable(columns, Transform.PLY);
};

export { readSogV1, type MetaV1 };
