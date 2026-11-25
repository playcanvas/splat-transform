import type { ChunkSource, ExtraColumn } from '../chunk';

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
 * they came from did: grouped by texture, fields in column order. A group
 * that no longer fits one texture's 4 bytes (a combined scene widened an
 * encoding) continues in the next.
 *
 * @param columns - The source's extra columns.
 * @returns The blind textures; empty when no column is tagged.
 */
const blindTexturesOf = (columns: readonly ExtraColumn[]): SogBlindTexture[] => {
    const groups = new Map<number, ExtraColumn[]>();
    for (const column of columns) {
        if (!column.sogBlind) continue;
        const group = groups.get(column.sogBlind.texture);
        if (group) group.push(column);
        else groups.set(column.sogBlind.texture, [column]);
    }
    const textures: SogBlindTexture[] = [];
    for (const t of [...groups.keys()].sort((a, b) => a - b)) {
        let fields: SogBlindTexture = [];
        let bytes = 0;
        for (const { name, sogBlind } of groups.get(t)!) {
            const { encoding, range } = sogBlind!;
            if (bytes + BLIND_ENCODING_BYTES[encoding] > 4) {
                textures.push(fields);
                fields = [];
                bytes = 0;
            }
            fields.push({ column: name, encoding, ...(range ? { range: [range[0], range[1]] } : {}) });
            bytes += BLIND_ENCODING_BYTES[encoding];
        }
        textures.push(fields);
    }
    return textures;
};

/**
 * The narrowest encoding that holds every value of both: the wider unsigned
 * integer, unorm16 for two unorms, else f32, which holds any decoded float.
 */
const widerEncoding = (a: SogBlindEncoding, b: SogBlindEncoding): SogBlindEncoding => {
    if (a === b) return a;
    if (isIntegerEncoding(a) && isIntegerEncoding(b)) return BLIND_ENCODING_BYTES[a] >= BLIND_ENCODING_BYTES[b] ? a : b;
    if (isUnorm(a) && isUnorm(b)) return 'unorm16';
    return 'f32';
};

/** The merge of two blind origins: holds both's values, in the first's texture. */
const mergeOrigins = (a: SogBlindOrigin, b: SogBlindOrigin): SogBlindOrigin => {
    const encoding = widerEncoding(a.encoding, b.encoding);
    let range: readonly [number, number] | undefined;
    if (isUnorm(encoding)) {
        const [min, max] = a.range ?? [0, 1];
        const [bMin, bMax] = b.range ?? [0, 1];
        range = [Math.min(min, bMin), Math.max(max, bMax)];
        if (range[0] === 0 && range[1] === 1) range = undefined;
    }
    if (encoding === a.encoding && range?.[0] === a.range?.[0] && range?.[1] === a.range?.[1]) return a;
    return { texture: a.texture, encoding, ...(range ? { range } : {}) };
};

/**
 * A combined scene's extra columns, each with the blind origins of the inputs
 * merged (matched by name and type) to hold every input's values: encodings
 * widen (see {@link widerEncoding}), unorm ranges widen to their union, and
 * the first tagged input's texture is kept. An input whose column is untagged
 * or missing may hold any value of its type, so a column some input lacks a
 * tag for widens to u32 or f32. A column no input tags stays untagged.
 *
 * @param lists - Each input's extra columns.
 * @param columns - The combined scene's extra columns. Default: the first input's.
 * @returns The combined scene's extra columns, with merged origins.
 */
const combineBlindOrigins = (
    lists: readonly (readonly ExtraColumn[])[],
    columns: readonly ExtraColumn[] = lists[0]
): readonly ExtraColumn[] => {
    return columns.map((column) => {
        let origin: SogBlindOrigin | undefined;
        let untagged = false;
        for (const list of lists) {
            const o = list.find((c) => c.name === column.name && c.type === column.type)?.sogBlind;
            if (!o) untagged = true;
            else origin = origin ? mergeOrigins(origin, o) : o;
        }
        if (origin && untagged) {
            origin = mergeOrigins(origin, {
                texture: origin.texture,
                encoding: column.type === 'uint32' ? 'u32' : 'f32'
            });
        }
        if (origin === column.sogBlind) return column;
        const { sogBlind: _, ...rest } = column;
        return origin ? { ...rest, sogBlind: origin } : rest;
    });
};

/**
 * Extra columns with the blind origins of `from` (matched by name and type)
 * attached, for columns rebuilt where origins cannot travel (a DataTable, a
 * PLY spill).
 *
 * @param columns - The rebuilt columns.
 * @param from - The columns carrying the origins.
 * @returns The rebuilt columns with origins.
 */
const attachBlindOrigins = (columns: readonly ExtraColumn[], from: readonly ExtraColumn[]): readonly ExtraColumn[] => {
    if (!from.some((c) => c.sogBlind)) return columns;
    return columns.map((column) => {
        const sogBlind = from.find((c) => c.name === column.name && c.type === column.type)?.sogBlind;
        return sogBlind ? { ...column, sogBlind } : column;
    });
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
    // delegate rather than spread: sources may be class instances
    return {
        meta: { ...source.meta, extraColumns: attachBlindOrigins(source.meta.extraColumns, columns) },
        read: (request) => source.read(request),
        close: () => source.close()
    };
};

export {
    BLIND_ENCODING_BYTES,
    blindTexturesOf,
    attachBlindOrigins,
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
