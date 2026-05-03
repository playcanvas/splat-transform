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
 * Stores (linear blockIdx, voxel mask) pairs for non-empty 4x4x4 blocks.
 *
 * Block keys are linear block indices `bx + by*nbx + bz*nbx*nby` in the
 * producer's grid coordinate system. Producers and consumers must agree
 * on the grid dimensions; the buffer itself is dimension-agnostic.
 *
 * Backed by typed arrays that grow geometrically. Keys use Float64Array so
 * the per-buffer capacity exceeds V8's regular-array backing-store limit
 * (large grids exceed Smi range and would throw `RangeError: Invalid array
 * length` with a regular array).
 */
class BlockMaskBuffer {
    /** Linear block indices for solid blocks (mask is implicitly all 1s) */
    private _solidIdx: Float64Array = new Float64Array(0);
    private _solidCount = 0;
    private _solidCap = 0;

    /** Linear block indices for mixed blocks */
    private _mixedIdx: Float64Array = new Float64Array(0);
    private _mixedCount = 0;
    private _mixedCap = 0;

    /** Interleaved voxel masks for mixed blocks: [lo0, hi0, lo1, hi1, ...] */
    private _mixedMasks: Uint32Array = new Uint32Array(0);

    /**
     * Add a non-empty block to the buffer.
     * Automatically classifies as solid or mixed based on mask values.
     *
     * @param blockIdx - Linear block index (`bx + by*nbx + bz*nbx*nby`)
     * @param lo - Lower 32 bits of voxel mask
     * @param hi - Upper 32 bits of voxel mask
     */
    addBlock(blockIdx: number, lo: number, hi: number): void {
        if (isEmpty(lo, hi)) {
            return;
        }

        if (isSolid(lo, hi)) {
            if (this._solidCount === this._solidCap) {
                // First grow: 0 → INITIAL_CAPACITY. Subsequent: double.
                this._solidCap = this._solidCap === 0 ? INITIAL_CAPACITY : this._solidCap * 2;
                this._solidIdx = growFloat64(this._solidIdx, this._solidCap);
            }
            this._solidIdx[this._solidCount++] = blockIdx;
        } else {
            if (this._mixedCount === this._mixedCap) {
                this._mixedCap = this._mixedCap === 0 ? INITIAL_CAPACITY : this._mixedCap * 2;
                this._mixedIdx = growFloat64(this._mixedIdx, this._mixedCap);
                this._mixedMasks = growUint32(this._mixedMasks, this._mixedCap * 2);
            }
            this._mixedIdx[this._mixedCount] = blockIdx;
            this._mixedMasks[this._mixedCount * 2] = lo;
            this._mixedMasks[this._mixedCount * 2 + 1] = hi;
            this._mixedCount++;
        }
    }

    /**
     * Get all mixed blocks as views into the underlying buffers.
     * Index `i` of `blockIdx` corresponds to mask pair `(masks[i*2], masks[i*2+1])`.
     *
     * @returns Object with linear block indices and interleaved masks
     */
    getMixedBlocks(): { blockIdx: Float64Array; masks: Uint32Array } {
        return {
            blockIdx: this._mixedIdx.subarray(0, this._mixedCount),
            masks: this._mixedMasks.subarray(0, this._mixedCount * 2)
        };
    }

    /**
     * Get all solid blocks as a view into the underlying buffer.
     *
     * @returns Array of linear block indices
     */
    getSolidBlocks(): Float64Array {
        return this._solidIdx.subarray(0, this._solidCount);
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
     * instance does not retain peak memory; the next `addBlock` re-allocates
     * to `INITIAL_CAPACITY`.
     */
    clear(): void {
        this._solidIdx = new Float64Array(0);
        this._solidCount = 0;
        this._solidCap = 0;

        this._mixedIdx = new Float64Array(0);
        this._mixedMasks = new Uint32Array(0);
        this._mixedCount = 0;
        this._mixedCap = 0;
    }
}

export { BlockMaskBuffer };
