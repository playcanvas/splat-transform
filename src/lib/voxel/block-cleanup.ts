import { BlockMaskBuffer } from './block-mask-buffer';
import { popcount } from './morton';
import { sortKeyMaskPairs } from './sort-key-mask';
import { logger } from '../utils';

// ============================================================================
// Edge mask constants for 4x4x4 voxel blocks
// ============================================================================
// Bit layout: bitIdx = lx + ly*4 + lz*16
// lo = bits 0-31 (lz=0: 0-15, lz=1: 16-31)
// hi = bits 32-63 (lz=2: 0-15, lz=3: 16-31)

/** lx=0 positions in each 32-bit word */
const FACE_X0 = 0x11111111;
/** lx=3 positions in each 32-bit word */
const FACE_X3 = 0x88888888;
/** ly=0 positions in each 32-bit word */
const FACE_Y0 = 0x000F000F;
/** ly=3 positions in each 32-bit word */
const FACE_Y3 = 0xF000F000;
/** lz=0 positions: lo bits 0-15 */
const FACE_Z0_LO = 0x0000FFFF;
/** lz=3 positions: hi bits 16-31 */
const FACE_Z3_HI = 0xFFFF0000 >>> 0;

const SOLID_MASK = 0xFFFFFFFF >>> 0;

// ============================================================================
// Main function
// ============================================================================

/**
 * Remove isolated voxels and fill isolated empty voxels within mixed blocks.
 *
 * For each mixed block, computes 6 per-direction occupancy masks (in-block via
 * bit shifts + cross-block via adjacent block lookups), then:
 *   - Remove: keeps only voxels with at least one occupied 6-connected neighbor
 *   - Fill: fills empty voxels where all 6 neighbors are occupied
 *
 * Blocks that become empty or solid as a consequence are handled automatically.
 *
 * @param buffer - BlockMaskBuffer with voxelization results (linear-keyed). Entry order may be changed.
 * @param nbx - Grid block dimension X (used to decode block indices).
 * @param nby - Grid block dimension Y (used to decode block indices).
 * @param nbz - Grid block dimension Z (used for neighbor bounds checks).
 * @param maxMixedBlocks - Optional output limit; filtering stops once exceeded.
 * @returns New BlockMaskBuffer with filtered/filled data.
 */
function filterAndFillBlocks(
    buffer: BlockMaskBuffer,
    nbx: number,
    nby: number,
    nbz: number,
    maxMixedBlocks: number = Number.POSITIVE_INFINITY
): BlockMaskBuffer {
    const mixed = buffer.getMixedBlocks();
    const solid = buffer.getSolidBlocks();
    const masks = mixed.masks;
    const bStride = nbx * nby;
    const totalBlocks = bStride * nbz;

    // Sort the source arrays in place. Cross-block neighbor queries below are
    // resolved as monotonic merges over fixed-size mixed-block chunks, avoiding
    // a full-scene hash table and its power-of-two capacity overhead.
    solid.sort();
    sortKeyMaskPairs(mixed.blockIdx, masks, mixed.blockIdx.length);

    // New masks array (snapshot: cross-block lookups always read the original masks)
    const newMasks = new Uint32Array(masks.length);
    let voxelsRemoved = 0;
    let voxelsFilled = 0;
    let mixedBlocks = 0;
    let promotedSolidBlocks = 0;
    const CHUNK_SIZE = 1 << 20;
    const work = new Uint32Array(Math.min(CHUNK_SIZE, mixed.blockIdx.length) * 12);

    const lowerBound = (keys: Float64Array, target: number): number => {
        let lo = 0;
        let hi = keys.length;
        while (lo < hi) {
            const mid = (lo + hi) >>> 1;
            if (keys[mid] < target) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    };

    for (let chunkStart = 0; chunkStart < mixed.blockIdx.length; chunkStart += CHUNK_SIZE) {
        const chunkLength = Math.min(CHUNK_SIZE, mixed.blockIdx.length - chunkStart);

        for (let local = 0; local < chunkLength; local++) {
            const i = chunkStart + local;
            const origLo = masks[i * 2];
            const origHi = masks[i * 2 + 1];
            const w = local * 12;
            work[w] = (origLo >>> 1) & ~FACE_X3;
            work[w + 1] = (origHi >>> 1) & ~FACE_X3;
            work[w + 2] = (origLo << 1) & ~FACE_X0;
            work[w + 3] = (origHi << 1) & ~FACE_X0;
            work[w + 4] = (origLo >>> 4) & ~FACE_Y3;
            work[w + 5] = (origHi >>> 4) & ~FACE_Y3;
            work[w + 6] = (origLo << 4) & ~FACE_Y0;
            work[w + 7] = (origHi << 4) & ~FACE_Y0;
            work[w + 8] = (origLo >>> 16) | (origHi << 16);
            work[w + 9] = origHi >>> 16;
            work[w + 10] = origLo << 16;
            work[w + 11] = (origHi << 16) | (origLo >>> 16);
        }

        for (let direction = 0; direction < 6; direction++) {
            let solidPos = -1;
            let mixedPos = -1;
            for (let local = 0; local < chunkLength; local++) {
                const i = chunkStart + local;
                const idx = mixed.blockIdx[i];
                const bx = idx % nbx;
                const by = Math.floor(idx / nbx) % nby;
                let target = -1;
                if (direction === 0 && bx < nbx - 1) target = idx + 1;
                else if (direction === 1 && bx > 0) target = idx - 1;
                else if (direction === 2 && by < nby - 1) target = idx + nbx;
                else if (direction === 3 && by > 0) target = idx - nbx;
                else if (direction === 4 && idx + bStride < totalBlocks) target = idx + bStride;
                else if (direction === 5 && idx >= bStride) target = idx - bStride;
                if (target < 0) continue;

                if (solidPos < 0) {
                    solidPos = lowerBound(solid, target);
                    mixedPos = lowerBound(mixed.blockIdx, target);
                } else {
                    while (solidPos < solid.length && solid[solidPos] < target) solidPos++;
                    while (mixedPos < mixed.blockIdx.length && mixed.blockIdx[mixedPos] < target) mixedPos++;
                }

                let adjLo = 0;
                let adjHi = 0;
                if (solidPos < solid.length && solid[solidPos] === target) {
                    adjLo = SOLID_MASK;
                    adjHi = SOLID_MASK;
                } else if (mixedPos < mixed.blockIdx.length && mixed.blockIdx[mixedPos] === target) {
                    adjLo = masks[mixedPos * 2];
                    adjHi = masks[mixedPos * 2 + 1];
                } else {
                    continue;
                }

                const w = local * 12 + direction * 2;
                if (direction === 0) {
                    work[w] |= (adjLo & FACE_X0) << 3;
                    work[w + 1] |= (adjHi & FACE_X0) << 3;
                } else if (direction === 1) {
                    work[w] |= (adjLo & FACE_X3) >>> 3;
                    work[w + 1] |= (adjHi & FACE_X3) >>> 3;
                } else if (direction === 2) {
                    work[w] |= (adjLo & FACE_Y0) << 12;
                    work[w + 1] |= (adjHi & FACE_Y0) << 12;
                } else if (direction === 3) {
                    work[w] |= (adjLo & FACE_Y3) >>> 12;
                    work[w + 1] |= (adjHi & FACE_Y3) >>> 12;
                } else if (direction === 4) {
                    work[w + 1] |= (adjLo & FACE_Z0_LO) << 16;
                } else {
                    work[w] |= (adjHi & FACE_Z3_HI) >>> 16;
                }
            }
        }

        for (let local = 0; local < chunkLength; local++) {
            const i = chunkStart + local;
            const origLo = masks[i * 2];
            const origHi = masks[i * 2 + 1];
            const w = local * 12;
            const pxLo = work[w], pxHi = work[w + 1];
            const mxLo = work[w + 2], mxHi = work[w + 3];
            const pyLo = work[w + 4], pyHi = work[w + 5];
            const myLo = work[w + 6], myHi = work[w + 7];
            const pzLo = work[w + 8], pzHi = work[w + 9];
            const mzLo = work[w + 10], mzHi = work[w + 11];
            const neighborLo = pxLo | mxLo | pyLo | myLo | pzLo | mzLo;
            const neighborHi = pxHi | mxHi | pyHi | myHi | pzHi | mzHi;
            let lo = origLo & neighborLo;
            let hi = origHi & neighborHi;
            lo |= ~lo & pxLo & mxLo & pyLo & myLo & pzLo & mzLo;
            hi |= ~hi & pxHi & mxHi & pyHi & myHi & pzHi & mzHi;

            voxelsRemoved += popcount(origLo & ~lo) + popcount(origHi & ~hi);
            voxelsFilled += popcount(lo & ~origLo) + popcount(hi & ~origHi);
            newMasks[i * 2] = lo;
            newMasks[i * 2 + 1] = hi;

            if ((lo >>> 0) === SOLID_MASK && (hi >>> 0) === SOLID_MASK) {
                promotedSolidBlocks++;
            } else if (lo !== 0 || hi !== 0) {
                mixedBlocks++;
                if (mixedBlocks > maxMixedBlocks) {
                    throw new Error(`Voxel output has more than ${maxMixedBlocks} mixed blocks. Use a coarser voxel resolution.`);
                }
            }
        }
    }

    // Rebuild buffer with state transitions (mixed->empty, mixed->solid)
    const result = new BlockMaskBuffer(solid.length + promotedSolidBlocks, mixedBlocks);

    for (let i = 0; i < mixed.blockIdx.length; i++) {
        const lo = newMasks[i * 2];
        const hi = newMasks[i * 2 + 1];
        result.addBlock(mixed.blockIdx[i], lo, hi);
    }

    for (let i = 0; i < solid.length; i++) {
        result.addBlock(solid[i], SOLID_MASK, SOLID_MASK);
    }

    logger.debug(`block cleanup: ${voxelsRemoved} voxels removed, ${voxelsFilled} voxels filled`);

    return result;
}

export { filterAndFillBlocks };
