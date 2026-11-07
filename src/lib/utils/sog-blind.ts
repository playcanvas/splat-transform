import type { ChunkSource, ExtraColumn } from '../chunk';

import { logger } from './logger';

/**
 * SOG v3 blind data encodings: how one source column is stored in the bytes
 * of a blind texture. Each blind texture holds up to 4 bytes per gaussian
 * (R, G, B, A), filled by its fields in order.
 *
 * - `u8` `u16` `u24` `u32`: unsigned integer, little-endian. The column must
 *   hold integers in range (float columns are accepted when they do).
 * - `f16` `f32`: the IEEE half / single float bits, little-endian.
 * - `unorm8` `unorm16`: a float quantized linearly from [min, max] (default
 *   [0, 1]) onto 0..255 / 0..65535, rounded to the nearest step.
 */
type SogBlindEncoding = 'u8' | 'u16' | 'u24' | 'u32' | 'f16' | 'f32' | 'unorm8' | 'unorm16';

/** One source column stored in a SOG blind texture. */
type SogBlindField = {
    /** Source column (an extra column of the source). */
    column: string;
    /** How the column is stored. Default: the fewest-byte unsigned integer encoding that holds it. */
    encoding?: SogBlindEncoding;
    /** [min, max] mapped onto the full unorm range. unorm encodings only. Default: [0, 1]. */
    range?: [number, number];
};

/** One SOG blind texture: fields filling its 4 bytes (R, G, B, A) in order. */
type SogBlindTexture = SogBlindField[];

/**
 * How a SOG file stored an extra column as blind data: its texture (the index
 * in the file's `blind` list), encoding and unorm range. The SOG reader tags
 * the columns it reads with it, so a SOG write stores them the same way by
 * default.
 */
type SogBlindOrigin = {
    readonly texture: number;
    readonly encoding: SogBlindEncoding;
    readonly range?: readonly [number, number];
};

/** Bytes each encoding occupies. */
const BLIND_ENCODING_BYTES: Readonly<Record<SogBlindEncoding, number>> = {
    u8: 1,
    u16: 2,
    u24: 3,
    u32: 4,
    f16: 2,
    f32: 4,
    unorm8: 1,
    unorm16: 2
};

const isBlindEncoding = (s: string): s is SogBlindEncoding => Object.hasOwn(BLIND_ENCODING_BYTES, s);

/** Whether the encoding decodes to an unsigned integer (else a float). */
const isIntegerEncoding = (e: SogBlindEncoding): boolean => e[0] === 'u' && !e.startsWith('unorm');

/**
 * The shader identifier suffix the engine derives from a field name (it reads
 * the field through `getBlind_<id>()`): characters outside [A-Za-z0-9_] become
 * '_', runs of '_' collapse to one and leading / trailing '_' are dropped.
 *
 * @param name - The field name.
 * @returns The identifier suffix, empty when the name has nothing to keep.
 */
const blindFieldId = (name: string): string => name.replace(/\W/g, '_').replace(/_+/g, '_').replace(/^_|_$/g, '');

/** The unsigned integer encoding with the fewest bytes that holds `max`. */
const integerEncodingFor = (max: number): SogBlindEncoding => {
    return max < 0x100 ? 'u8' : max < 0x10000 ? 'u16' : max < 0x1000000 ? 'u24' : 'u32';
};

const f32Scratch = new Float32Array(1);
const u32Scratch = new Uint32Array(f32Scratch.buffer);

/**
 * IEEE half-float bits of `v`, rounded to nearest even. Finite values beyond
 * the half range become infinity; NaN stays NaN.
 *
 * @param v - The value.
 * @returns The 16-bit pattern.
 */
const floatToHalfBits = (v: number): number => {
    f32Scratch[0] = v;
    const x = u32Scratch[0];
    const sign = (x >>> 16) & 0x8000;
    const exp = (x >>> 23) & 0xff;
    let mant = x & 0x7fffff;
    if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
    const e = exp - 127 + 15;
    if (e >= 0x1f) return sign | 0x7c00;
    if (e <= 0) {
        // subnormal half (or zero)
        if (e < -10) return sign;
        mant |= 0x800000;
        const shift = 14 - e;
        let half = mant >>> shift;
        const rem = mant & ((1 << shift) - 1);
        const mid = 1 << (shift - 1);
        if (rem > mid || (rem === mid && half & 1)) half++;
        return sign | half;
    }
    let half = (e << 10) | (mant >>> 13);
    const rem = mant & 0x1fff;
    if (rem > 0x1000 || (rem === 0x1000 && half & 1)) half++;
    return sign | half;
};

/**
 * The float a half-float bit pattern holds.
 *
 * @param h - The 16-bit pattern.
 * @returns The value.
 */
const halfBitsToFloat = (h: number): number => {
    const sign = h & 0x8000 ? -1 : 1;
    const exp = (h >>> 10) & 0x1f;
    const mant = h & 0x3ff;
    if (exp === 0) return sign * mant * 2 ** -24;
    if (exp === 0x1f) return mant ? NaN : sign * Infinity;
    return sign * (1 + mant / 1024) * 2 ** (exp - 15);
};

/**
 * Encode one value as the unsigned integer the encoding stores.
 *
 * @param v - The column value.
 * @param encoding - The encoding.
 * @param range - [min, max] for unorm encodings.
 * @param column - Column name, for errors.
 * @returns The stored bits (at most 32).
 */
const encodeBlindValue = (v: number, encoding: SogBlindEncoding, range: [number, number], column: string): number => {
    switch (encoding) {
        case 'f32':
            f32Scratch[0] = v;
            return u32Scratch[0];
        case 'f16':
            if (Number.isFinite(v) && Math.abs(v) > 65504) {
                throw new Error(`SOG blind data: column '${column}' holds ${v}, outside the f16 range (±65504)`);
            }
            return floatToHalfBits(v);
        case 'unorm8':
        case 'unorm16': {
            const [min, max] = range;
            // columns hold f32, so an endpoint value may be stored just outside the range
            // (0.1 as 0.10000000149011612): accept the f32 endpoints too, then clamp
            if (!(v >= Math.min(min, Math.fround(min)) && v <= Math.max(max, Math.fround(max)))) {
                throw new Error(
                    `SOG blind data: column '${column}' holds ${v}, outside its ${encoding} range [${min}, ${max}]`
                );
            }
            const steps = encoding === 'unorm8' ? 255 : 65535;
            const t = (Math.min(Math.max(v, min), max) - min) / (max - min);
            return Math.round(t * steps);
        }
        default: {
            const limit = 2 ** (8 * BLIND_ENCODING_BYTES[encoding]) - 1;
            if (!(Number.isInteger(v) && v >= 0 && v <= limit)) {
                throw new Error(
                    `SOG blind data: column '${column}' holds ${v}; ${encoding} stores unsigned integers 0 to ${limit}`
                );
            }
            return v;
        }
    }
};

/**
 * Decode stored bits back to the column value.
 *
 * @param bits - The stored bits.
 * @param encoding - The encoding.
 * @param range - [min, max] for unorm encodings.
 * @returns The value.
 */
const decodeBlindValue = (bits: number, encoding: SogBlindEncoding, range: [number, number]): number => {
    switch (encoding) {
        case 'f32':
            u32Scratch[0] = bits;
            return f32Scratch[0];
        case 'f16':
            return halfBitsToFloat(bits);
        case 'unorm8':
            return range[0] + (bits / 255) * (range[1] - range[0]);
        case 'unorm16':
            return range[0] + (bits / 65535) * (range[1] - range[0]);
        default:
            return bits >>> 0;
    }
};

const isUnorm = (e: SogBlindEncoding): boolean => e.startsWith('unorm');

/**
 * The blind textures that store the tagged extra columns as the SOG file
 * they came from did: grouped by texture, fields in column order.
 *
 * @param columns - The source's extra columns.
 * @returns The blind textures; empty when no column is tagged.
 */
const blindTexturesOf = (columns: readonly ExtraColumn[]): SogBlindTexture[] => {
    const textures = new Map<number, SogBlindTexture>();
    for (const { name, sogBlind } of columns) {
        if (!sogBlind) continue;
        const { texture, encoding, range } = sogBlind;
        let fields = textures.get(texture);
        if (!fields) textures.set(texture, (fields = []));
        fields.push({ column: name, encoding, ...(range ? { range: [range[0], range[1]] } : {}) });
    }
    return [...textures.keys()].sort((a, b) => a - b).map((t) => textures.get(t)!);
};

/**
 * The extra columns of a combined scene (whose inputs share column names and
 * types), with each column's blind origin merged: unorm ranges widen to hold
 * every input's values, and where inputs stored a column with different
 * encodings or textures the first input's is kept, with a warning.
 *
 * @param lists - Each input's extra columns.
 * @returns The combined scene's extra columns.
 */
const combineBlindOrigins = (lists: readonly (readonly ExtraColumn[])[]): readonly ExtraColumn[] => {
    let conflict = false;
    const combined = lists[0].map((column, i) => {
        let origin: SogBlindOrigin | undefined;
        for (const list of lists) {
            const o = list[i].sogBlind;
            if (!o) continue;
            if (!origin) {
                origin = o;
                continue;
            }
            if (o.texture !== origin.texture || o.encoding !== origin.encoding) conflict = true;
            if (isUnorm(o.encoding) && isUnorm(origin.encoding)) {
                const [min, max] = origin.range ?? [0, 1];
                const [oMin, oMax] = o.range ?? [0, 1];
                if (oMin < min || oMax > max) {
                    origin = { ...origin, range: [Math.min(min, oMin), Math.max(max, oMax)] };
                }
            }
        }
        return origin === column.sogBlind ? column : { ...column, sogBlind: origin };
    });
    if (conflict) {
        logger.warn("inputs store blind data with different encodings; the combined scene keeps the first input's");
    }
    return combined;
};

/**
 * The same source with its extra columns' blind origins taken from `columns`
 * (matched by name and type), for a source rebuilt from a DataTable, which
 * cannot carry them. Reads are untouched.
 *
 * @param source - The source.
 * @param columns - The extra columns carrying the origins.
 * @returns A source whose extra columns carry the origins.
 */
const withBlindOrigins = (source: ChunkSource, columns: readonly ExtraColumn[]): ChunkSource => {
    if (!columns.some((c) => c.sogBlind)) return source;
    const extraColumns = source.meta.extraColumns.map((column) => {
        const sogBlind = columns.find((c) => c.name === column.name && c.type === column.type)?.sogBlind;
        return sogBlind ? { ...column, sogBlind } : column;
    });
    // delegate rather than spread: sources may be class instances
    return {
        meta: { ...source.meta, extraColumns },
        read: (request) => source.read(request),
        close: () => source.close()
    };
};

export {
    BLIND_ENCODING_BYTES,
    blindTexturesOf,
    combineBlindOrigins,
    withBlindOrigins,
    blindFieldId,
    decodeBlindValue,
    encodeBlindValue,
    floatToHalfBits,
    halfBitsToFloat,
    integerEncodingFor,
    isBlindEncoding,
    isIntegerEncoding,
    type SogBlindEncoding,
    type SogBlindField,
    type SogBlindOrigin,
    type SogBlindTexture
};
