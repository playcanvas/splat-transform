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
            if (!(v >= min && v <= max)) {
                throw new Error(
                    `SOG blind data: column '${column}' holds ${v}, outside its ${encoding} range [${min}, ${max}]`
                );
            }
            const steps = encoding === 'unorm8' ? 255 : 65535;
            return Math.round(((v - min) / (max - min)) * steps);
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

export {
    BLIND_ENCODING_BYTES,
    decodeBlindValue,
    encodeBlindValue,
    floatToHalfBits,
    halfBitsToFloat,
    integerEncodingFor,
    isBlindEncoding,
    isIntegerEncoding,
    type SogBlindEncoding,
    type SogBlindField,
    type SogBlindTexture
};
