import { isSolid, isEmpty } from './morton';

/**
 * Append-only buffer for streaming voxelization results.
 * Stores block masks using Morton codes for efficient octree construction.
 */
class BlockMaskBuffer {
    /** Morton codes for mixed blocks */
    private _mixedMorton: number[] = [];

    /** Interleaved voxel masks for mixed blocks: [lo0, hi0, lo1, hi1, ...] */
    private _mixedMasks: number[] = [];

    /** Morton codes for solid blocks (mask is implicitly all 1s) */
    private _solidMorton: number[] = [];

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
            this._solidMorton.push(morton);
        } else {
            this._mixedMorton.push(morton);
            this._mixedMasks.push(lo, hi);
        }
    }

    /**
     * Get all mixed blocks.
     *
     * @returns Object with morton codes and interleaved masks
     */
    getMixedBlocks(): { morton: number[]; masks: number[] } {
        return {
            morton: this._mixedMorton,
            masks: this._mixedMasks
        };
    }

    /**
     * Get all solid blocks.
     *
     * @returns Array of Morton codes
     */
    getSolidBlocks(): number[] {
        return this._solidMorton;
    }

    /**
     * Get total number of blocks stored.
     *
     * @returns Count of mixed + solid blocks
     */
    get count(): number {
        return this._mixedMorton.length + this._solidMorton.length;
    }

    /**
     * Get number of mixed blocks.
     *
     * @returns Count of mixed blocks
     */
    get mixedCount(): number {
        return this._mixedMorton.length;
    }

    /**
     * Get number of solid blocks.
     *
     * @returns Count of solid blocks
     */
    get solidCount(): number {
        return this._solidMorton.length;
    }

    /**
     * Clear all buffered blocks.
     */
    clear(): void {
        this._mixedMorton.length = 0;
        this._mixedMasks.length = 0;
        this._solidMorton.length = 0;
    }
}

export { BlockMaskBuffer };
