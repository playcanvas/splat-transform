import { Column, DataTable } from '../data-table';
import { basename, dirname, join, type ReadFileSystem, readFile } from '../io/read';
import { logger, Transform, WebPCodec } from '../utils';
import { readSogV1, type MetaV1 } from './read-sog-v1';

// V2 (current) SOG meta layout - codebook-based quantization for scales /
// sh0 / shN. Legacy V1 uses a different per-channel mins/maxs scheme and is
// handled separately in read-sog-v1.ts.
type MetaV2 = {
    version: 2;
    count: number;
    means: { mins: number[]; maxs: number[]; files: string[] };
    scales: { codebook: number[]; files: string[] };
    quats: { files: string[] };
    sh0: { codebook: number[]; files: string[] };
    shN?: { count: number; bands: number; codebook: number[]; files: string[] };
};

type ReadSogOptions = {
    // Controls how readSog reports its own progress (mirrors writeSog's
    // `logging` option, reduced to the modes a bar-only reader needs):
    //   'own'    — open readSog's internal 'decoding' bar (default)
    //   'silent' — no bar; the caller drives its own progress (e.g. readLcc2
    //              decoding many chunks concurrently — logger bars are strictly
    //              LIFO, so concurrent per-chunk bars would corrupt each other)
    logging?: 'own' | 'silent';
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

const sigmoidInv = (y: number) => {
    const e = Math.min(1 - 1e-6, Math.max(1e-6, y));
    return Math.log(e / (1 - e));
};

/**
 * Read a SOG file from a ReadFileSystem.
 *
 * The current (V2) format is decoded inline. Legacy V1 files (no `version`
 * field, per-channel mins/maxs instead of codebooks) are detected here and
 * forwarded to {@link readSogV1} in read-sog-v1.ts.
 *
 * @param fileSystem - The file system to read from
 * @param filename - Path to meta.json (relative paths resolved from its directory).
 * The basename is used verbatim for the initial meta fetch so
 * any URL querystring/fragment (e.g. presigned `?token=...`)
 * is preserved.
 * @param options - Options controlling progress reporting.
 * @returns DataTable with Gaussian splat data
 * @ignore
 */
const readSog = async (fileSystem: ReadFileSystem, filename: string, options: ReadSogOptions = {}): Promise<DataTable> => {
    const baseDir = dirname(filename);
    const metaName = basename(filename);
    const resolve = (name: string) => (baseDir ? join(baseDir, name) : name);

    const metaBytes = await readFile(fileSystem, resolve(metaName));
    const rawMeta = JSON.parse(new TextDecoder().decode(metaBytes)) as MetaV2 | (MetaV1 & { version?: number });

    // Dispatch:
    //   - V1 (legacy) has no `version` field        -> readSogV1
    //   - V2 (current) has `version: 2`             -> handled inline below
    //   - any other (future/unknown) version        -> hard error rather than
    //     silently mis-routing to V1, which would surface as confusing
    //     downstream failures (missing `means.shape`, etc.).
    const version = rawMeta.version;
    if (version === undefined) {
        return readSogV1(fileSystem, baseDir, rawMeta as MetaV1, options);
    }
    if (version !== 2) {
        throw new Error(`Unsupported SOG meta version: ${version}`);
    }
    const meta = rawMeta as MetaV2;

    const decoder = await WebPCodec.create();
    const count = meta.count;

    const load = async (name: string): Promise<Uint8Array> => {
        const src = await fileSystem.createSource(resolve(name));
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

    // One bar across all per-gaussian decode passes. Total = passes * count
    // (means, quats, scales, sh0, plus an optional shN pass). Each pass ticks
    // with `count` once it has finished writing into the output columns.
    const numPasses = 4 + (meta.shN ? 1 : 0);
    const bar = options.logging === 'silent' ? null : logger.bar('decoding', numPasses * count);
    let passesDone = 0;
    const tickPass = () => {
        bar?.update(++passesDone * count);
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
    // scaled by sqrt(2).
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

    // scales: each byte indexes the shared 256-entry codebook.
    const scalesWebp = await load(meta.scales.files[0]);
    const { rgba: sl, width: sw, height: sh } = decoder.decodeRGBA(scalesWebp);
    if (sw * sh < count) throw new Error('SOG scales texture too small for count');
    const sCode = new Float32Array(meta.scales.codebook);
    const s0 = columns[3].data as Float32Array;
    const s1 = columns[4].data as Float32Array;
    const s2 = columns[5].data as Float32Array;
    for (let i = 0; i < count; i++) {
        const o = i * 4;
        s0[i] = sCode[sl[o]];
        s1[i] = sCode[sl[o + 1]];
        s2[i] = sCode[sl[o + 2]];
    }
    tickPass();

    // sh0: 3 color bytes index the shared codebook; opacity byte is a sigmoid
    // value, decoded back to a logit via sigmoidInv.
    const sh0Webp = await load(meta.sh0.files[0]);
    const { rgba: c0, width: cw, height: ch } = decoder.decodeRGBA(sh0Webp);
    if (cw * ch < count) throw new Error('SOG sh0 texture too small for count');
    const cCode = new Float32Array(meta.sh0.codebook);
    const dc0 = columns[6].data as Float32Array;
    const dc1 = columns[7].data as Float32Array;
    const dc2 = columns[8].data as Float32Array;
    const opCol = columns[9].data as Float32Array;
    for (let i = 0; i < count; i++) {
        const o = i * 4;
        dc0[i] = cCode[c0[o + 0]];
        dc1[i] = cCode[c0[o + 1]];
        dc2[i] = cCode[c0[o + 2]];
        opCol[i] = sigmoidInv(c0[o + 3] / 255);
    }
    tickPass();

    // shN (optional): indirect lookup via labels -> centroids palette, with
    // each centroid pixel byte indexing the shared codebook.
    if (meta.shN) {
        const { bands, count: paletteCount } = meta.shN;
        const shCoeffs = [0, 3, 8, 15][bands];
        if (shCoeffs > 0) {
            const codebook = new Float32Array(meta.shN.codebook);
            const centroidsWebp = await load(meta.shN.files[0]);
            const labelsWebp = await load(meta.shN.files[1]);
            const { rgba: centroidsRGBA, width: cW, height: cH } = decoder.decodeRGBA(centroidsWebp);
            const { rgba: labelsRGBA, width: lW, height: lH } = decoder.decodeRGBA(labelsWebp);

            // Validate texture geometry up-front: missing guards would let
            // truncated textures silently produce zeros (out-of-bounds typed-
            // array reads coerce to 0 via the bitwise ops below) instead of
            // failing like the means/quats/scales/sh0 passes do.
            if (lW * lH < count) throw new Error('SOG shN labels texture too small for count');
            if (cW !== 64 * shCoeffs) throw new Error(`SOG shN centroids texture width ${cW} does not match expected ${64 * shCoeffs} for ${bands}-band palette`);

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
                    (columns[baseIdx + j + shCoeffs * 0].data as Float32Array)[i] = codebook[lr] ?? 0;
                    (columns[baseIdx + j + shCoeffs * 1].data as Float32Array)[i] = codebook[lg] ?? 0;
                    (columns[baseIdx + j + shCoeffs * 2].data as Float32Array)[i] = codebook[lb] ?? 0;
                }
            }
        }
        tickPass();
    }

    // Close the bar only on success: leaving it open on the error path lets
    // `logger.error() -> unwindAll(true)` mark it as failed instead of
    // finalizing it as a successful bar first.
    bar?.end();

    return new DataTable(columns, Transform.PLY);
};

export { readSog };
export type { ReadSogOptions };
