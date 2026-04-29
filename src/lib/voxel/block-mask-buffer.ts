import { isSolid, isEmpty } from './morton';

const INITIAL_CAPACITY = 1024;

const growFloat64 = (src: Float64Array, newCap: number): Float64Array => {
    const grown = new Float64Array(newCap);
    grown.set(src);
    return grown;
};

const growUint32 = (src: Uint32Array, newCap: number): Uint32Array => {
    const grown = new Uint32Array(newCap);
    grown.set(src);
    return grown;
};

/**
 * Append-only buffer for streaming voxelization results.
 * Stores block masks using Morton codes for efficient octree construction.
 *
 * Backed by typed arrays that grow geometrically. Morton codes use
 * Float64Array (51 bits of precision needed; exceeds Smi range), and
 * voxel masks use Uint32Array. This raises the per-buffer capacity well
 * above V8's regular-array backing-store limit so very large grids can
 * round-trip without throwing `RangeError: Invalid array length`.
 */
class BlockMaskBuffer {
    /** Morton codes for solid blocks (mask is implicitly all 1s) */
    private _solidMorton: Float64Array = new Float64Array(INITIAL_CAPACITY);
    private _solidCount = 0;
    private _solidCap = INITIAL_CAPACITY;

    /** Morton codes for mixed blocks */
    private _mixedMorton: Float64Array = new Float64Array(INITIAL_CAPACITY);
    private _mixedCount = 0;
    private _mixedCap = INITIAL_CAPACITY;

    /** Interleaved voxel masks for mixed blocks: [lo0, hi0, lo1, hi1, ...] */
    private _mixedMasks: Uint32Array = new Uint32Array(INITIAL_CAPACITY * 2);

    /**
     * Add a non-empty block to the buffer.
     * Automatically classifies as solid or mixed based on mask values.
     *
     * @param morton - Morton code encoding block position
     * @param lo - Lower 32 bits of voxel mask
     * @param hi - Upper 32 bits of voxel mask
     */
    addBlock(morton: number, lo: number, hi: number): void {
        if (isEmpty(lo, hi)) {
            return;
        }

        if (isSolid(lo, hi)) {
            if (this._solidCount === this._solidCap) {
                this._solidCap *= 2;
                this._solidMorton = growFloat64(this._solidMorton, this._solidCap);
            }
            this._solidMorton[this._solidCount++] = morton;
        } else {
            if (this._mixedCount === this._mixedCap) {
                this._mixedCap *= 2;
                this._mixedMorton = growFloat64(this._mixedMorton, this._mixedCap);
                this._mixedMasks = growUint32(this._mixedMasks, this._mixedCap * 2);
            }
            this._mixedMorton[this._mixedCount] = morton;
            this._mixedMasks[this._mixedCount * 2] = lo;
            this._mixedMasks[this._mixedCount * 2 + 1] = hi;
            this._mixedCount++;
        }
    }

    /**
     * Get all mixed blocks as views into the underlying buffers.
     * Index `i` of `morton` corresponds to mask pair `(masks[i*2], masks[i*2+1])`.
     *
     * @returns Object with morton codes and interleaved masks
     */
    getMixedBlocks(): { morton: Float64Array; masks: Uint32Array } {
        return {
            morton: this._mixedMorton.subarray(0, this._mixedCount),
            masks: this._mixedMasks.subarray(0, this._mixedCount * 2)
        };
    }

    /**
     * Get all solid blocks as a view into the underlying buffer.
     *
     * @returns Array of Morton codes
     */
    getSolidBlocks(): Float64Array {
        return this._solidMorton.subarray(0, this._solidCount);
    }

    /**
     * Get total number of blocks stored.
     *
     * @returns Count of mixed + solid blocks
     */
    get count(): number {
        return this._mixedCount + this._solidCount;
    }

    /**
     * Get number of mixed blocks.
     *
     * @returns Count of mixed blocks
     */
    get mixedCount(): number {
        return this._mixedCount;
    }

    /**
     * Get number of solid blocks.
     *
     * @returns Count of solid blocks
     */
    get solidCount(): number {
        return this._solidCount;
    }

    /**
     * Clear all buffered blocks. Releases the underlying buffers so a cleared
     * instance does not retain peak memory.
     */
    clear(): void {
        this._solidMorton = new Float64Array(INITIAL_CAPACITY);
        this._solidCount = 0;
        this._solidCap = INITIAL_CAPACITY;

        this._mixedMorton = new Float64Array(INITIAL_CAPACITY);
        this._mixedMasks = new Uint32Array(INITIAL_CAPACITY * 2);
        this._mixedCount = 0;
        this._mixedCap = INITIAL_CAPACITY;
    }
}

export { BlockMaskBuffer };
