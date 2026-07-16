const EMPTY_KEY = -1;

/** Returned by `get` for a block index that was never inserted (empty block). */
const ABSENT = -1;

/** Stored/returned value marking a fully-solid block. */
const SOLID = -2;

/**
 * Build-once, query-many open-addressing hash from a linear block index to a
 * small state value, backed by typed arrays. Unlike a JS `Map`/`Set` it has no
 * 2^24-entry ceiling. Capacity is limited to the largest power-of-two table
 * whose slots can still be addressed by the 32-bit hash arithmetic.
 *
 * Values encode block state for the voxel cleanup / query passes:
 *   - `SOLID` (-2): the block is fully solid.
 *   - `>= 0`: the block is mixed; the value is its index into the caller's
 *     parallel masks array (mask pair at `masks[v * 2]`, `masks[v * 2 + 1]`).
 * `get` returns `ABSENT` (-1) for indices that were never inserted.
 *
 * Keys are held in a `Float64Array` so block indices beyond 2^32 (very large
 * grids) are stored and compared exactly; the hash mixes both 32-bit halves.
 * The table is sized once from a capacity hint and never resizes, so callers
 * must not insert more than the hinted number of entries.
 */
class BlockIndexMap {
    private keys: Float64Array;
    private vals: Int32Array;
    private mask: number;

    /**
     * @param capacityHint - Exact (or upper-bound) number of entries that will
     * be inserted. The table is sized to the next power of two above
     * `capacityHint / 0.7` so the load factor stays <= 0.7 and it never resizes.
     */
    constructor(capacityHint: number) {
        const need = Math.max(16, Math.ceil(capacityHint / 0.7));
        const cap = 2 ** Math.ceil(Math.log2(need));
        if (!Number.isSafeInteger(capacityHint) || capacityHint < 0 || cap > 0x80000000) {
            throw new Error(`BlockIndexMap capacity ${capacityHint} exceeds the typed hash-table limit`);
        }
        this.mask = cap - 1;
        this.keys = new Float64Array(cap).fill(EMPTY_KEY);
        this.vals = new Int32Array(cap);
    }

    set(key: number, val: number): void {
        const keys = this.keys;
        const mask = this.mask;
        let i = ((Math.imul(key >>> 0, 0x9E3779B9) ^ Math.imul((key / 4294967296) >>> 0, 0x85EBCA77)) >>> 0) & mask;
        while (keys[i] !== EMPTY_KEY && keys[i] !== key) {
            i = (i + 1) & mask;
        }
        keys[i] = key;
        this.vals[i] = val;
    }

    get(key: number): number {
        const keys = this.keys;
        const mask = this.mask;
        let i = ((Math.imul(key >>> 0, 0x9E3779B9) ^ Math.imul((key / 4294967296) >>> 0, 0x85EBCA77)) >>> 0) & mask;
        while (true) {
            const k = keys[i];
            if (k === key) return this.vals[i];
            if (k === EMPTY_KEY) return ABSENT;
            i = (i + 1) & mask;
        }
    }

    releaseStorage(): void {
        this.keys = new Float64Array(0);
        this.vals = new Int32Array(0);
        this.mask = 0;
    }
}

export { BlockIndexMap, ABSENT, SOLID };
