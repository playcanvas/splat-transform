import { BlockMaskBuffer } from './block-mask-buffer';
import { BlockMaskMap } from './block-mask-map';
import { mortonToXYZ, xyzToMorton } from './morton';

const SOLID_LO = 0xFFFFFFFF >>> 0;
const SOLID_HI = 0xFFFFFFFF >>> 0;

const BLOCK_EMPTY = 0;
const BLOCK_SOLID = 1;
const BLOCK_MIXED = 2;

// Face bitmasks for the 6 faces of a 4x4x4 block.
// bitIdx layout: lx + (ly << 2) + (lz << 4), lo = bits 0-31, hi = bits 32-63.
// Each pair [lo, hi] selects the 16 voxels on one face.
const FACE_MASKS_LO = [
    0x11111111 >>> 0, // -X: lx=0
    0x88888888 >>> 0, // +X: lx=3
    0x000F000F >>> 0, // -Y: ly=0
    0xF000F000 >>> 0, // +Y: ly=3
    0x0000FFFF >>> 0, // -Z: lz=0
    0x00000000 >>> 0  // +Z: lz=3
];
const FACE_MASKS_HI = [
    0x11111111 >>> 0,
    0x88888888 >>> 0,
    0x000F000F >>> 0,
    0xF000F000 >>> 0,
    0x00000000 >>> 0,
    0xFFFF0000 >>> 0
];

// ============================================================================
// SparseVoxelGrid (packed 2-bit types)
//
// Stores voxel data at 4x4x4 block granularity with exact voxel-level
// precision. The per-block type {EMPTY=0, SOLID=1, MIXED=2} is packed into
// a single Uint32Array at 2 bits per block (16 blocks per word):
//
//   block i  -> word index = i >>> 4, bit shift = (i & 15) << 1
//   read     -> (types[w] >>> shift) & 0x3
//   write    -> types[w] = (types[w] & ~(0x3 << shift)) | (value << shift)
//
// Per-word patterns:
//   0x55555555 = all 16 lanes SOLID  (0b01 in each pair)
//   0xAAAAAAAA = all 16 lanes MIXED  (0b10 in each pair)
//
// "Find non-empty blocks" iteration uses the trick:
//   nonEmpty = (word & 0x55555555) | ((word >>> 1) & 0x55555555);
// which sets bit 2k in `nonEmpty` whenever lane k is non-zero. Iterating
// the set bits with `clz32(nonEmpty & -nonEmpty)` then yields lane indices
// the same way the old occupancy bitfield did.
//
// MIXED blocks store their lo/hi voxel mask in `masks` keyed on the
// global block index (bx + by*nbx + bz*bStride).
//
// Memory: 2 bits per block, vs 1 byte (blockType) + 1 bit (occupancy)
// = 9 bits per block in the previous design. ~4.5x reduction on this
// component.
// ============================================================================

const TYPE_BITS_PER_BLOCK = 2;
const BLOCKS_PER_WORD = 32 / TYPE_BITS_PER_BLOCK; // 16
const TYPE_MASK = (1 << TYPE_BITS_PER_BLOCK) - 1; // 0b11
const SOLID_WORD = 0x55555555 >>> 0;             // 16 lanes, each = SOLID
// const MIXED_WORD = 0xAAAAAAAA >>> 0;          // 16 lanes, each = MIXED (unused so far)
const EVEN_BITS = 0x55555555 >>> 0;              // mask for even bit positions

class SparseVoxelGrid {
    readonly nx: number;
    readonly ny: number;
    readonly nz: number;
    readonly nbx: number;
    readonly nby: number;
    readonly nbz: number;
    readonly bStride: number;

    /**
     * Packed block types: 2 bits per block, 16 blocks per Uint32 word.
     * Length = ceil(totalBlocks / 16).
     */
    types: Uint32Array;

    /** Voxel masks for MIXED blocks, keyed on global block index. */
    masks: BlockMaskMap;

    constructor(nx: number, ny: number, nz: number) {
        this.nx = nx;
        this.ny = ny;
        this.nz = nz;
        this.nbx = nx >> 2;
        this.nby = ny >> 2;
        this.nbz = nz >> 2;
        this.bStride = this.nbx * this.nby;
        const totalBlocks = this.nbx * this.nby * this.nbz;
        this.types = new Uint32Array((totalBlocks + BLOCKS_PER_WORD - 1) >>> 4);
        this.masks = new BlockMaskMap();
    }

    /**
     * Read the 2-bit block type at the given block index.
     *
     * @param blockIdx - Linear block index (`bx + by*nbx + bz*bStride`).
     * @returns Block type: `BLOCK_EMPTY`, `BLOCK_SOLID`, or `BLOCK_MIXED`.
     */
    getBlockType(blockIdx: number): number {
        return (this.types[blockIdx >>> 4] >>> ((blockIdx & 15) << 1)) & TYPE_MASK;
    }

    /**
     * Write the 2-bit block type at the given block index. Caller is
     * responsible for keeping `masks` consistent (only `MIXED` blocks
     * should have a mask entry; `EMPTY`/`SOLID` should not).
     *
     * @param blockIdx - Linear block index.
     * @param value - Block type to set.
     */
    setBlockType(blockIdx: number, value: number): void {
        const w = blockIdx >>> 4;
        const shift = (blockIdx & 15) << 1;
        this.types[w] = ((this.types[w] & ~(TYPE_MASK << shift)) | ((value & TYPE_MASK) << shift)) >>> 0;
    }

    getVoxel(ix: number, iy: number, iz: number): number {
        const blockIdx = (ix >> 2) + (iy >> 2) * this.nbx + (iz >> 2) * this.bStride;
        const bt = this.getBlockType(blockIdx);
        if (bt === BLOCK_EMPTY) return 0;
        if (bt === BLOCK_SOLID) return 1;
        const s = this.masks.slot(blockIdx);
        const bitIdx = (ix & 3) + ((iy & 3) << 2) + ((iz & 3) << 4);
        return bitIdx < 32 ? (this.masks.lo[s] >>> bitIdx) & 1 : (this.masks.hi[s] >>> (bitIdx - 32)) & 1;
    }

    setVoxel(ix: number, iy: number, iz: number): void {
        const blockIdx = (ix >> 2) + (iy >> 2) * this.nbx + (iz >> 2) * this.bStride;
        const bt = this.getBlockType(blockIdx);
        if (bt === BLOCK_SOLID) return;
        const bitIdx = (ix & 3) + ((iy & 3) << 2) + ((iz & 3) << 4);
        if (bt === BLOCK_MIXED) {
            const s = this.masks.slot(blockIdx);
            if (bitIdx < 32) this.masks.lo[s] = (this.masks.lo[s] | (1 << bitIdx)) >>> 0;
            else this.masks.hi[s] = (this.masks.hi[s] | (1 << (bitIdx - 32))) >>> 0;
            if (this.masks.lo[s] === SOLID_LO && this.masks.hi[s] === SOLID_HI) {
                this.masks.removeAt(s);
                this.setBlockType(blockIdx, BLOCK_SOLID);
            }
        } else {
            this.setBlockType(blockIdx, BLOCK_MIXED);
            this.masks.set(blockIdx,
                bitIdx < 32 ? (1 << bitIdx) >>> 0 : 0,
                bitIdx >= 32 ? (1 << (bitIdx - 32)) >>> 0 : 0
            );
        }
    }

    orBlock(blockIdx: number, lo: number, hi: number): void {
        if (lo === 0 && hi === 0) return;
        const bt = this.getBlockType(blockIdx);
        if (bt === BLOCK_SOLID) return;
        if (bt === BLOCK_MIXED) {
            const s = this.masks.slot(blockIdx);
            this.masks.lo[s] = (this.masks.lo[s] | lo) >>> 0;
            this.masks.hi[s] = (this.masks.hi[s] | hi) >>> 0;
            if (this.masks.lo[s] === SOLID_LO && this.masks.hi[s] === SOLID_HI) {
                this.masks.removeAt(s);
                this.setBlockType(blockIdx, BLOCK_SOLID);
            }
        } else {
            if ((lo >>> 0) === SOLID_LO && (hi >>> 0) === SOLID_HI) {
                this.setBlockType(blockIdx, BLOCK_SOLID);
            } else {
                this.setBlockType(blockIdx, BLOCK_MIXED);
                this.masks.set(blockIdx, lo >>> 0, hi >>> 0);
            }
        }
    }

    clear(): void {
        this.types.fill(0);
        this.masks.clear();
    }

    clone(): SparseVoxelGrid {
        const g = new SparseVoxelGrid(this.nx, this.ny, this.nz);
        g.types.set(this.types);
        g.masks = this.masks.clone();
        return g;
    }

    static fromBuffer(acc: BlockMaskBuffer, nx: number, ny: number, nz: number): SparseVoxelGrid {
        const g = new SparseVoxelGrid(nx, ny, nz);
        const solidMortons = acc.getSolidBlocks();
        for (let i = 0; i < solidMortons.length; i++) {
            const [bx, by, bz] = mortonToXYZ(solidMortons[i]);
            const blockIdx = bx + by * g.nbx + bz * g.bStride;
            g.setBlockType(blockIdx, BLOCK_SOLID);
        }
        const mixed = acc.getMixedBlocks();
        for (let i = 0; i < mixed.morton.length; i++) {
            const [bx, by, bz] = mortonToXYZ(mixed.morton[i]);
            const blockIdx = bx + by * g.nbx + bz * g.bStride;
            g.setBlockType(blockIdx, BLOCK_MIXED);
            g.masks.set(blockIdx, mixed.masks[i * 2], mixed.masks[i * 2 + 1]);
        }
        return g;
    }

    toBuffer(
        cropMinBx: number, cropMinBy: number, cropMinBz: number,
        cropMaxBx: number, cropMaxBy: number, cropMaxBz: number,
        defaultSolid = false
    ): BlockMaskBuffer {
        const acc = new BlockMaskBuffer();
        for (let bz = cropMinBz; bz < cropMaxBz; bz++) {
            for (let by = cropMinBy; by < cropMaxBy; by++) {
                for (let bx = cropMinBx; bx < cropMaxBx; bx++) {
                    const blockIdx = bx + by * this.nbx + bz * this.bStride;
                    const bt = this.getBlockType(blockIdx);
                    let lo: number, hi: number;
                    if (bt === BLOCK_SOLID) {
                        lo = SOLID_LO;
                        hi = SOLID_HI;
                    } else if (bt === BLOCK_MIXED) {
                        const s = this.masks.slot(blockIdx);
                        lo = this.masks.lo[s];
                        hi = this.masks.hi[s];
                    } else if (defaultSolid) {
                        lo = SOLID_LO;
                        hi = SOLID_HI;
                    } else {
                        continue;
                    }
                    if (lo || hi) {
                        acc.addBlock(
                            xyzToMorton(bx - cropMinBx, by - cropMinBy, bz - cropMinBz),
                            lo, hi
                        );
                    }
                }
            }
        }
        return acc;
    }

    toBufferInverted(
        cropMinBx: number, cropMinBy: number, cropMinBz: number,
        cropMaxBx: number, cropMaxBy: number, cropMaxBz: number
    ): BlockMaskBuffer {
        const acc = new BlockMaskBuffer();
        for (let bz = cropMinBz; bz < cropMaxBz; bz++) {
            for (let by = cropMinBy; by < cropMaxBy; by++) {
                for (let bx = cropMinBx; bx < cropMaxBx; bx++) {
                    const blockIdx = bx + by * this.nbx + bz * this.bStride;
                    const bt = this.getBlockType(blockIdx);
                    let lo: number, hi: number;
                    if (bt === BLOCK_SOLID) {
                        continue;
                    } else if (bt === BLOCK_MIXED) {
                        const s = this.masks.slot(blockIdx);
                        lo = (~this.masks.lo[s]) >>> 0;
                        hi = (~this.masks.hi[s]) >>> 0;
                    } else {
                        lo = SOLID_LO;
                        hi = SOLID_HI;
                    }
                    if (lo || hi) {
                        acc.addBlock(
                            xyzToMorton(bx - cropMinBx, by - cropMinBy, bz - cropMinBz),
                            lo, hi
                        );
                    }
                }
            }
        }
        return acc;
    }

    /**
     * Get the bounding box of occupied blocks.
     *
     * @returns Block coordinate bounds, or null if no blocks are occupied.
     */
    getOccupiedBlockBounds(): {
        minBx: number; minBy: number; minBz: number;
        maxBx: number; maxBy: number; maxBz: number;
    } | null {
        const { nbx, nby } = this;
        const totalBlocks = this.nbx * this.nby * this.nbz;
        let minBx = nbx, minBy = nby, minBz = this.nbz;
        let maxBx = 0, maxBy = 0, maxBz = 0;
        let found = false;
        for (let w = 0; w < this.types.length; w++) {
            const word = this.types[w];
            if (word === 0) continue;
            // Set bit at even position 2k iff lane k is non-empty.
            let nonEmpty = ((word & EVEN_BITS) | ((word >>> 1) & EVEN_BITS)) >>> 0;
            while (nonEmpty) {
                const bp = 31 - Math.clz32(nonEmpty & -nonEmpty);
                const lane = bp >>> 1;
                const blockIdx = w * BLOCKS_PER_WORD + lane;
                if (blockIdx >= totalBlocks) {
                    nonEmpty = 0;
                    break;
                }
                const bx = blockIdx % nbx;
                const byBz = (blockIdx / nbx) | 0;
                const by = byBz % nby;
                const bz = (byBz / nby) | 0;
                if (bx < minBx) minBx = bx;
                if (bx > maxBx) maxBx = bx;
                if (by < minBy) minBy = by;
                if (by > maxBy) maxBy = by;
                if (bz < minBz) minBz = bz;
                if (bz > maxBz) maxBz = bz;
                found = true;
                nonEmpty &= nonEmpty - 1;
            }
        }
        return found ? { minBx, minBy, minBz, maxBx, maxBy, maxBz } : null;
    }

    /**
     * Get the bounding box of navigable (non-fully-solid) blocks. A block is
     * navigable if it is EMPTY or MIXED (i.e. contains at least one empty
     * voxel). Useful when the runtime treats out-of-grid as solid, so fully
     * solid blocks beyond the navigable region can be cropped away.
     *
     * @returns Block coordinate bounds, or null if every block is solid.
     */
    getNavigableBlockBounds(): {
        minBx: number; minBy: number; minBz: number;
        maxBx: number; maxBy: number; maxBz: number;
    } | null {
        const { nbx, nby } = this;
        const totalBlocks = this.nbx * this.nby * this.nbz;
        // Mask of valid lanes in the last word (in terms of even bit positions).
        const lastWordIdx = this.types.length - 1;
        const lastLanes = totalBlocks - lastWordIdx * BLOCKS_PER_WORD;
        // navMask in the last word: only the first `lastLanes` lanes are valid;
        // each lane occupies 2 bits, so the valid even-bit mask spans
        // 2 * lastLanes bits.
        const lastNonEmptyMask = lastLanes >= BLOCKS_PER_WORD ?
            EVEN_BITS :
            (((1 << (lastLanes * 2)) - 1) >>> 0) & EVEN_BITS;

        let minBx = nbx, minBy = nby, minBz = this.nbz;
        let maxBx = -1, maxBy = 0, maxBz = 0;

        for (let w = 0; w < this.types.length; w++) {
            const word = this.types[w];
            // Lane is non-SOLID iff its 2 bits aren't (1, 0) i.e. iff
            // word ^ SOLID_WORD has any bit set in that pair.
            const flipped = (word ^ SOLID_WORD) >>> 0;
            let navMask = ((flipped & EVEN_BITS) | ((flipped >>> 1) & EVEN_BITS)) >>> 0;
            // For the final (possibly partial) word, drop lanes past totalBlocks.
            if (w === lastWordIdx) navMask &= lastNonEmptyMask;

            while (navMask) {
                const bp = 31 - Math.clz32(navMask & -navMask);
                const lane = bp >>> 1;
                const blockIdx = w * BLOCKS_PER_WORD + lane;
                const bx = blockIdx % nbx;
                const byBz = (blockIdx / nbx) | 0;
                const by = byBz % nby;
                const bz = (byBz / nby) | 0;
                if (bx < minBx) minBx = bx;
                if (bx > maxBx) maxBx = bx;
                if (by < minBy) minBy = by;
                if (by > maxBy) maxBy = by;
                if (bz < minBz) minBz = bz;
                if (bz > maxBz) maxBz = bz;
                navMask &= navMask - 1;
            }
        }

        return maxBx >= 0 ? { minBx, minBy, minBz, maxBx, maxBy, maxBz } : null;
    }

    /**
     * Find the nearest free (unblocked) voxel to a seed position using
     * expanding cube shells.
     *
     * @param blocked - Grid to search for free cells in.
     * @param seedIx - Seed voxel X coordinate.
     * @param seedIy - Seed voxel Y coordinate.
     * @param seedIz - Seed voxel Z coordinate.
     * @param maxRadius - Maximum search radius in voxels.
     * @returns Coordinates of the nearest free voxel, or null.
     */
    static findNearestFreeCell(
        blocked: SparseVoxelGrid,
        seedIx: number, seedIy: number, seedIz: number,
        maxRadius: number
    ): { ix: number; iy: number; iz: number } | null {
        const { nx, ny, nz } = blocked;
        for (let r = 1; r <= maxRadius; r++) {
            for (let dz = -r; dz <= r; dz++) {
                for (let dy = -r; dy <= r; dy++) {
                    for (let dx = -r; dx <= r; dx++) {
                        if (Math.abs(dx) !== r && Math.abs(dy) !== r && Math.abs(dz) !== r) continue;
                        const ix = seedIx + dx;
                        const iy = seedIy + dy;
                        const iz = seedIz + dz;
                        if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) continue;
                        if (!blocked.getVoxel(ix, iy, iz)) return { ix, iy, iz };
                    }
                }
            }
        }
        return null;
    }
}

export {
    BLOCK_EMPTY,
    BLOCK_MIXED,
    BLOCK_SOLID,
    BLOCKS_PER_WORD,
    EVEN_BITS,
    FACE_MASKS_HI,
    FACE_MASKS_LO,
    SOLID_HI,
    SOLID_LO,
    SOLID_WORD,
    SparseVoxelGrid,
    TYPE_MASK
};
