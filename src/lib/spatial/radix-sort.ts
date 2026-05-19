/**
 * Reusable scratch buffers for `radixSortIndicesByFloat`. One instance is
 * shared across many sort calls to avoid allocating multi-MB typed arrays on
 * every call. Buffers grow on demand and never shrink.
 */
class RadixSortScratch {
    /** Working u32 keys (monotonic-encoded). */
    keysWork: Uint32Array;
    /** Ping-pong destination for u32 keys. */
    keysAlt: Uint32Array;
    /** Ping-pong destination for indices. */
    indicesAlt: Uint32Array;
    /** Per-byte histogram, reused across passes. */
    counts: Uint32Array;

    constructor() {
        this.keysWork = new Uint32Array(0);
        this.keysAlt = new Uint32Array(0);
        this.indicesAlt = new Uint32Array(0);
        this.counts = new Uint32Array(256);
    }

    ensure(count: number): void {
        if (count > this.keysWork.length) {
            this.keysWork = new Uint32Array(count);
            this.keysAlt = new Uint32Array(count);
            this.indicesAlt = new Uint32Array(count);
        }
    }
}

/**
 * Encode a Float32 bit pattern as a sortable u32: positive floats get their
 * sign bit set (so they sort above negatives), negative floats get all bits
 * inverted (so larger-magnitude negatives sort lower).
 *
 * @param bits - Float32 bit pattern.
 * @returns A u32 such that radix-sorting these values yields the same order
 *   as comparator-sorting the original Float32s ascending.
 */
const encodeFloatKey = (bits: number): number => {
    return (bits & 0x80000000) !== 0 ? (~bits >>> 0) : ((bits ^ 0x80000000) >>> 0);
};

/**
 * 4-pass LSD radix sort of `indices[0..count)` ascending by the Float32 value
 * at the parallel position in `keys`. Mutates `indices` in place.
 *
 * `keys[i]` is the sort key for `indices[i]` — the two arrays are parallel.
 * Callers compute both before calling and the sort permutes them together so
 * the parallel relationship is preserved on output.
 *
 * `keys` is read as Float32 and the IEEE-754 sign-bit dance is applied
 * internally to give a monotonic u32 encoding; the original Float32 buffer
 * is not mutated.
 *
 * NaN / Inf entries are sorted to one extreme based on their bit pattern.
 * Callers that need to reject them should filter before calling. (At
 * ~250K-150M elements this is ~20x faster than a JavaScript comparator
 * sort.)
 *
 * @param indices - Integer payload, mutated in place.
 * @param keys - Float32 sort keys, parallel to `indices`. Not mutated.
 * @param count - Number of valid entries (length of the parallel prefix).
 * @param scratch - Reusable scratch buffers; grown on demand.
 */
const radixSortIndicesByFloat = (
    indices: Uint32Array,
    keys: Float32Array,
    count: number,
    scratch: RadixSortScratch
): void => {
    if (count < 2) return;
    scratch.ensure(count);

    const { keysWork, keysAlt, indicesAlt, counts } = scratch;
    const keyBits = new Uint32Array(keys.buffer, keys.byteOffset, keys.length);

    // Encode keys (in indices order — keysWork[i] is the sort key for indices[i]).
    for (let i = 0; i < count; i++) {
        keysWork[i] = encodeFloatKey(keyBits[i]);
    }

    // 4-pass LSD radix sort, 8 bits per pass. After an even number of passes
    // (4) the sorted output is back in the originally-supplied indices buffer.
    let kIn = keysWork, iIn = indices;
    let kOut = keysAlt, iOut = indicesAlt;

    for (let shift = 0; shift < 32; shift += 8) {
        counts.fill(0);
        for (let i = 0; i < count; i++) {
            counts[(kIn[i] >>> shift) & 0xff]++;
        }
        // Prefix sum → starting offsets per bucket.
        let sum = 0;
        for (let b = 0; b < 256; b++) {
            const c = counts[b];
            counts[b] = sum;
            sum += c;
        }
        // Scatter.
        for (let i = 0; i < count; i++) {
            const b = (kIn[i] >>> shift) & 0xff;
            const pos = counts[b]++;
            kOut[pos] = kIn[i];
            iOut[pos] = iIn[i];
        }
        // Swap input/output.
        const tk = kIn; kIn = kOut; kOut = tk;
        const ti = iIn; iIn = iOut; iOut = ti;
    }
};

/**
 * Test whether a Float32 bit pattern represents NaN or ±Inf.
 * Useful for callers that want to filter non-finite keys out before sorting.
 *
 * @param bits - Float32 bit pattern.
 * @returns true if the value is NaN or infinite.
 */
const isFloatBitsNonFinite = (bits: number): boolean => {
    return (bits & 0x7F800000) === 0x7F800000;
};

export { RadixSortScratch, radixSortIndicesByFloat, isFloatBitsNonFinite };
