const EMPTY = -1;

/**
 * Open-addressing hash map for voxel block masks backed by typed arrays.
 * Replaces Map<number, [number, number]> to eliminate V8 Map overhead and
 * per-entry tuple allocations that cause GC pressure.
 *
 * Keys are non-negative block indices. Values are lo/hi uint32 mask pairs.
 * Uses Fibonacci hashing with linear probing and backward-shift deletion.
 */
class MaskStore {
    keys: Int32Array;
    lo: Uint32Array;
    hi: Uint32Array;
    private _size: number;
    private _capacity: number;
    private _mask: number;

    constructor(initialCapacity = 4096) {
        const cap = Math.max(16, initialCapacity);
        this._capacity = cap;
        this._mask = cap - 1;
        this._size = 0;
        this.keys = new Int32Array(cap).fill(EMPTY);
        this.lo = new Uint32Array(cap);
        this.hi = new Uint32Array(cap);
    }

    get size(): number {
        return this._size;
    }

    private _hash(key: number): number {
        return (Math.imul(key, 0x9E3779B9) >>> 0) & this._mask;
    }

    /**
     * Find the slot for a key. If the key exists, keys[slot] === key.
     * If the key doesn't exist, keys[slot] === -1 (empty slot for insertion).
     *
     * @param key - Block index to look up.
     * @returns Slot index into keys/lo/hi arrays.
     */
    slot(key: number): number {
        const mask = this._mask;
        let i = (Math.imul(key, 0x9E3779B9) >>> 0) & mask;
        while (true) {
            const k = this.keys[i];
            if (k === key || k === EMPTY) return i;
            i = (i + 1) & mask;
        }
    }

    has(key: number): boolean {
        return this.keys[this.slot(key)] !== EMPTY;
    }

    /**
     * Insert at a slot known to be empty (keys[slot] === -1).
     * Caller must have obtained slot via slot() and verified it is empty.
     *
     * @param slot - Empty slot index.
     * @param key - Block index to insert.
     * @param loVal - Lower 32 bits of voxel mask.
     * @param hiVal - Upper 32 bits of voxel mask.
     */
    insertAt(slot: number, key: number, loVal: number, hiVal: number): void {
        this.keys[slot] = key;
        this.lo[slot] = loVal;
        this.hi[slot] = hiVal;
        this._size++;
        if (this._size > ((this._capacity * 0.7) | 0)) {
            this._grow();
        }
    }

    /**
     * Set key to lo/hi values. Inserts if key doesn't exist, updates if it does.
     *
     * @param key - Block index.
     * @param loVal - Lower 32 bits of voxel mask.
     * @param hiVal - Upper 32 bits of voxel mask.
     */
    set(key: number, loVal: number, hiVal: number): void {
        let s = this.slot(key);
        if (this.keys[s] === EMPTY) {
            this.keys[s] = key;
            this._size++;
            if (this._size > ((this._capacity * 0.7) | 0)) {
                this._grow();
                s = this.slot(key);
            }
        }
        this.lo[s] = loVal;
        this.hi[s] = hiVal;
    }

    /**
     * Remove entry at slot using backward-shift deletion.
     * Maintains probe chain integrity without tombstones.
     *
     * @param slot - Slot index of the entry to remove.
     */
    removeAt(slot: number): void {
        this._size--;
        const mask = this._mask;
        let i = slot;
        let j = slot;
        while (true) {
            j = (j + 1) & mask;
            if (this.keys[j] === EMPTY) break;
            const k = ((Math.imul(this.keys[j], 0x9E3779B9) >>> 0) & mask);
            if ((i < j) ? (k <= i || k > j) : (k <= i && k > j)) {
                this.keys[i] = this.keys[j];
                this.lo[i] = this.lo[j];
                this.hi[i] = this.hi[j];
                i = j;
            }
        }
        this.keys[i] = EMPTY;
    }

    delete(key: number): void {
        const s = this.slot(key);
        if (this.keys[s] !== EMPTY) {
            this.removeAt(s);
        }
    }

    clear(): void {
        this.keys.fill(EMPTY);
        this._size = 0;
    }

    forEach(fn: (key: number, lo: number, hi: number) => void): void {
        const keys = this.keys;
        const cap = this._capacity;
        for (let i = 0; i < cap; i++) {
            if (keys[i] !== EMPTY) {
                fn(keys[i], this.lo[i], this.hi[i]);
            }
        }
    }

    clone(): MaskStore {
        const c = new MaskStore(this._capacity);
        c.keys.set(this.keys);
        c.lo.set(this.lo);
        c.hi.set(this.hi);
        c._size = this._size;
        return c;
    }

    private _grow(): void {
        const oldKeys = this.keys;
        const oldLo = this.lo;
        const oldHi = this.hi;
        const oldCap = this._capacity;

        this._capacity *= 2;
        this._mask = this._capacity - 1;
        this.keys = new Int32Array(this._capacity).fill(EMPTY);
        this.lo = new Uint32Array(this._capacity);
        this.hi = new Uint32Array(this._capacity);
        this._size = 0;

        for (let i = 0; i < oldCap; i++) {
            if (oldKeys[i] !== EMPTY) {
                const s = this.slot(oldKeys[i]);
                this.keys[s] = oldKeys[i];
                this.lo[s] = oldLo[i];
                this.hi[s] = oldHi[i];
                this._size++;
            }
        }
    }
}

export { MaskStore };
