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
// SparseVoxelGrid
//
// Stores voxel data at 4x4x4 block granularity with exact voxel-level
// precision. Memory is proportional to the number of non-empty blocks.
//
// Each block has a type stored in blockType[blockIdx]:
//   0 (EMPTY):  no voxels set
//   1 (SOLID):  all 64 voxels set, no mask entry
//   2 (MIXED):  partial voxels, lo/hi mask in BlockMaskMap
//
// An occupancy bitfield is maintained in parallel for fast scanning
// of occupied blocks (used by active-pair computation in dilation).
// ============================================================================

class SparseVoxelGrid {
    readonly nx: number;
    readonly ny: number;
    readonly nz: number;
    readonly nbx: number;
    readonly nby: number;
    readonly nbz: number;
    readonly bStride: number;

    blockType: Uint8Array;
    occupancy: Uint32Array;
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
        this.blockType = new Uint8Array(totalBlocks);
        this.occupancy = new Uint32Array(((totalBlocks) + 31) >>> 5);
        this.masks = new BlockMaskMap();
    }

    getVoxel(ix: number, iy: number, iz: number): number {
        const blockIdx = (ix >> 2) + (iy >> 2) * this.nbx + (iz >> 2) * this.bStride;
        const bt = this.blockType[blockIdx];
        if (bt === BLOCK_EMPTY) return 0;
        if (bt === BLOCK_SOLID) return 1;
        const s = this.masks.slot(blockIdx);
        const bitIdx = (ix & 3) + ((iy & 3) << 2) + ((iz & 3) << 4);
        return bitIdx < 32 ? (this.masks.lo[s] >>> bitIdx) & 1 : (this.masks.hi[s] >>> (bitIdx - 32)) & 1;
    }

    setVoxel(ix: number, iy: number, iz: number): void {
        const blockIdx = (ix >> 2) + (iy >> 2) * this.nbx + (iz >> 2) * this.bStride;
        const bt = this.blockType[blockIdx];
        if (bt === BLOCK_SOLID) return;
        const bitIdx = (ix & 3) + ((iy & 3) << 2) + ((iz & 3) << 4);
        if (bt === BLOCK_MIXED) {
            const s = this.masks.slot(blockIdx);
            if (bitIdx < 32) this.masks.lo[s] = (this.masks.lo[s] | (1 << bitIdx)) >>> 0;
            else this.masks.hi[s] = (this.masks.hi[s] | (1 << (bitIdx - 32))) >>> 0;
            if (this.masks.lo[s] === SOLID_LO && this.masks.hi[s] === SOLID_HI) {
                this.masks.removeAt(s);
                this.blockType[blockIdx] = BLOCK_SOLID;
            }
        } else {
            this.blockType[blockIdx] = BLOCK_MIXED;
            this.occupancy[blockIdx >>> 5] |= (1 << (blockIdx & 31));
            this.masks.set(blockIdx,
                bitIdx < 32 ? (1 << bitIdx) >>> 0 : 0,
                bitIdx >= 32 ? (1 << (bitIdx - 32)) >>> 0 : 0
            );
        }
    }

    orBlock(blockIdx: number, lo: number, hi: number): void {
        if (lo === 0 && hi === 0) return;
        const bt = this.blockType[blockIdx];
        if (bt === BLOCK_SOLID) return;
        if (bt === BLOCK_MIXED) {
            const s = this.masks.slot(blockIdx);
            this.masks.lo[s] = (this.masks.lo[s] | lo) >>> 0;
            this.masks.hi[s] = (this.masks.hi[s] | hi) >>> 0;
            if (this.masks.lo[s] === SOLID_LO && this.masks.hi[s] === SOLID_HI) {
                this.masks.removeAt(s);
                this.blockType[blockIdx] = BLOCK_SOLID;
            }
        } else {
            this.occupancy[blockIdx >>> 5] |= (1 << (blockIdx & 31));
            if ((lo >>> 0) === SOLID_LO && (hi >>> 0) === SOLID_HI) {
                this.blockType[blockIdx] = BLOCK_SOLID;
            } else {
                this.blockType[blockIdx] = BLOCK_MIXED;
                this.masks.set(blockIdx, lo >>> 0, hi >>> 0);
            }
        }
    }

    clear(): void {
        this.blockType.fill(0);
        this.occupancy.fill(0);
        this.masks.clear();
    }

    clone(): SparseVoxelGrid {
        const g = new SparseVoxelGrid(this.nx, this.ny, this.nz);
        g.blockType.set(this.blockType);
        g.occupancy.set(this.occupancy);
        g.masks = this.masks.clone();
        return g;
    }

    static fromAccumulator(acc: BlockMaskBuffer, nx: number, ny: number, nz: number): SparseVoxelGrid {
        const g = new SparseVoxelGrid(nx, ny, nz);
        const solidMortons = acc.getSolidBlocks();
        for (let i = 0; i < solidMortons.length; i++) {
            const [bx, by, bz] = mortonToXYZ(solidMortons[i]);
            const blockIdx = bx + by * g.nbx + bz * g.bStride;
            g.blockType[blockIdx] = BLOCK_SOLID;
            g.occupancy[blockIdx >>> 5] |= (1 << (blockIdx & 31));
        }
        const mixed = acc.getMixedBlocks();
        for (let i = 0; i < mixed.morton.length; i++) {
            const [bx, by, bz] = mortonToXYZ(mixed.morton[i]);
            const blockIdx = bx + by * g.nbx + bz * g.bStride;
            g.blockType[blockIdx] = BLOCK_MIXED;
            g.occupancy[blockIdx >>> 5] |= (1 << (blockIdx & 31));
            g.masks.set(blockIdx, mixed.masks[i * 2], mixed.masks[i * 2 + 1]);
        }
        return g;
    }

    toAccumulator(
        cropMinBx: number, cropMinBy: number, cropMinBz: number,
        cropMaxBx: number, cropMaxBy: number, cropMaxBz: number,
        defaultSolid = false
    ): BlockMaskBuffer {
        const acc = new BlockMaskBuffer();
        for (let bz = cropMinBz; bz < cropMaxBz; bz++) {
            for (let by = cropMinBy; by < cropMaxBy; by++) {
                for (let bx = cropMinBx; bx < cropMaxBx; bx++) {
                    const blockIdx = bx + by * this.nbx + bz * this.bStride;
                    const bt = this.blockType[blockIdx];
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

    toAccumulatorInverted(
        cropMinBx: number, cropMinBy: number, cropMinBz: number,
        cropMaxBx: number, cropMaxBy: number, cropMaxBz: number
    ): BlockMaskBuffer {
        const acc = new BlockMaskBuffer();
        for (let bz = cropMinBz; bz < cropMaxBz; bz++) {
            for (let by = cropMinBy; by < cropMaxBy; by++) {
                for (let bx = cropMinBx; bx < cropMaxBx; bx++) {
                    const blockIdx = bx + by * this.nbx + bz * this.bStride;
                    const bt = this.blockType[blockIdx];
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
        for (let w = 0; w < this.occupancy.length; w++) {
            let bits = this.occupancy[w];
            while (bits) {
                const bitPos = 31 - Math.clz32(bits & -bits);
                const blockIdx = w * 32 + bitPos;
                if (blockIdx >= totalBlocks) break;
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
                bits &= bits - 1;
            }
        }
        return minBx <= maxBx ? { minBx, minBy, minBz, maxBx, maxBy, maxBz } : null;
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
    FACE_MASKS_HI,
    FACE_MASKS_LO,
    SOLID_HI,
    SOLID_LO,
    SparseVoxelGrid
};
