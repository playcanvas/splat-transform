import {
    BlockAccumulator,
    mortonToXYZ,
    xyzToMorton
} from './sparse-octree';

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
//   1 (SOLID):  all 64 voxels set, no Map entry
//   2 (MIXED):  partial voxels, [lo, hi] mask in masks Map
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
    masks: Map<number, [number, number]>;

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
        this.masks = new Map();
    }

    getVoxel(ix: number, iy: number, iz: number): number {
        const blockIdx = (ix >> 2) + (iy >> 2) * this.nbx + (iz >> 2) * this.bStride;
        const bt = this.blockType[blockIdx];
        if (bt === BLOCK_EMPTY) return 0;
        if (bt === BLOCK_SOLID) return 1;
        const mask = this.masks.get(blockIdx)!;
        const bitIdx = (ix & 3) + ((iy & 3) << 2) + ((iz & 3) << 4);
        return bitIdx < 32 ? (mask[0] >>> bitIdx) & 1 : (mask[1] >>> (bitIdx - 32)) & 1;
    }

    setVoxel(ix: number, iy: number, iz: number): void {
        const blockIdx = (ix >> 2) + (iy >> 2) * this.nbx + (iz >> 2) * this.bStride;
        const bt = this.blockType[blockIdx];
        if (bt === BLOCK_SOLID) return;
        const bitIdx = (ix & 3) + ((iy & 3) << 2) + ((iz & 3) << 4);
        if (bt === BLOCK_MIXED) {
            const mask = this.masks.get(blockIdx)!;
            if (bitIdx < 32) mask[0] = (mask[0] | (1 << bitIdx)) >>> 0;
            else mask[1] = (mask[1] | (1 << (bitIdx - 32))) >>> 0;
            if (mask[0] === SOLID_LO && mask[1] === SOLID_HI) {
                this.masks.delete(blockIdx);
                this.blockType[blockIdx] = BLOCK_SOLID;
            }
        } else {
            this.blockType[blockIdx] = BLOCK_MIXED;
            this.occupancy[blockIdx >>> 5] |= (1 << (blockIdx & 31));
            this.masks.set(blockIdx, [
                bitIdx < 32 ? (1 << bitIdx) >>> 0 : 0,
                bitIdx >= 32 ? (1 << (bitIdx - 32)) >>> 0 : 0
            ]);
        }
    }

    orBlock(blockIdx: number, lo: number, hi: number): void {
        if (lo === 0 && hi === 0) return;
        const bt = this.blockType[blockIdx];
        if (bt === BLOCK_SOLID) return;
        if (bt === BLOCK_MIXED) {
            const mask = this.masks.get(blockIdx)!;
            mask[0] = (mask[0] | lo) >>> 0;
            mask[1] = (mask[1] | hi) >>> 0;
            if (mask[0] === SOLID_LO && mask[1] === SOLID_HI) {
                this.masks.delete(blockIdx);
                this.blockType[blockIdx] = BLOCK_SOLID;
            }
        } else {
            this.occupancy[blockIdx >>> 5] |= (1 << (blockIdx & 31));
            if ((lo >>> 0) === SOLID_LO && (hi >>> 0) === SOLID_HI) {
                this.blockType[blockIdx] = BLOCK_SOLID;
            } else {
                this.blockType[blockIdx] = BLOCK_MIXED;
                this.masks.set(blockIdx, [lo >>> 0, hi >>> 0]);
            }
        }
    }

    getBlockMask(blockIdx: number): [number, number] | null {
        const bt = this.blockType[blockIdx];
        if (bt === BLOCK_EMPTY) return null;
        if (bt === BLOCK_SOLID) return [SOLID_LO, SOLID_HI];
        return this.masks.get(blockIdx)!;
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
        for (const [k, v] of this.masks) {
            g.masks.set(k, [v[0], v[1]]);
        }
        return g;
    }

    static fromAccumulator(acc: BlockAccumulator, nx: number, ny: number, nz: number): SparseVoxelGrid {
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
            g.masks.set(blockIdx, [mixed.masks[i * 2], mixed.masks[i * 2 + 1]]);
        }
        return g;
    }

    toAccumulator(
        cropMinBx: number, cropMinBy: number, cropMinBz: number,
        cropMaxBx: number, cropMaxBy: number, cropMaxBz: number,
        defaultSolid = false
    ): BlockAccumulator {
        const acc = new BlockAccumulator();
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
                        const mask = this.masks.get(blockIdx)!;
                        lo = mask[0];
                        hi = mask[1];
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
    ): BlockAccumulator {
        const acc = new BlockAccumulator();
        for (let bz = cropMinBz; bz < cropMaxBz; bz++) {
            for (let by = cropMinBy; by < cropMaxBy; by++) {
                for (let bx = cropMinBx; bx < cropMaxBx; bx++) {
                    const blockIdx = bx + by * this.nbx + bz * this.bStride;
                    const bt = this.blockType[blockIdx];
                    let lo: number, hi: number;
                    if (bt === BLOCK_SOLID) {
                        continue;
                    } else if (bt === BLOCK_MIXED) {
                        const mask = this.masks.get(blockIdx)!;
                        lo = (~mask[0]) >>> 0;
                        hi = (~mask[1]) >>> 0;
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
