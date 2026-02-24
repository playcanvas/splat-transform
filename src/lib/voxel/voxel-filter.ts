import {
    BlockAccumulator,
    xyzToMorton,
    mortonToXYZ,
    popcount
} from './sparse-octree';
import { logger } from '../utils/logger';

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
 * @param accumulator - BlockAccumulator with voxelization results
 * @returns New BlockAccumulator with filtered/filled data
 */
function filterAndFillBlocks(accumulator: BlockAccumulator): BlockAccumulator {
    const mixed = accumulator.getMixedBlocks();
    const solid = accumulator.getSolidBlocks();
    const masks = mixed.masks;

    // Build lookup structures from original (unmodified) data
    const solidSet = new Set<number>();
    for (let i = 0; i < solid.length; i++) {
        solidSet.add(solid[i]);
    }

    const mixedMap = new Map<number, number>();
    for (let i = 0; i < mixed.morton.length; i++) {
        mixedMap.set(mixed.morton[i], i);
    }

    // New masks array (snapshot: cross-block lookups always read the original masks)
    const newMasks = new Array<number>(masks.length);
    let voxelsRemoved = 0;
    let voxelsFilled = 0;

    for (let i = 0; i < mixed.morton.length; i++) {
        const morton = mixed.morton[i];
        const origLo = masks[i * 2];
        const origHi = masks[i * 2 + 1];

        const [bx, by, bz] = mortonToXYZ(morton);

        // --- In-block per-direction occupancy masks ---
        // Each gives: bit p = 1 iff position p's neighbor in that direction is occupied

        // +X: result[p] = mask[p+1], valid for lx < 3
        let pxLo = (origLo >>> 1) & ~FACE_X3;
        let pxHi = (origHi >>> 1) & ~FACE_X3;

        // -X: result[p] = mask[p-1], valid for lx > 0
        let mxLo = (origLo << 1) & ~FACE_X0;
        let mxHi = (origHi << 1) & ~FACE_X0;

        // +Y: result[p] = mask[p+4], valid for ly < 3
        let pyLo = (origLo >>> 4) & ~FACE_Y3;
        let pyHi = (origHi >>> 4) & ~FACE_Y3;

        // -Y: result[p] = mask[p-4], valid for ly > 0
        let myLo = (origLo << 4) & ~FACE_Y0;
        let myHi = (origHi << 4) & ~FACE_Y0;

        // +Z: result[p] = mask[p+16], crosses lo/hi at lz=1->lz=2
        let pzLo = (origLo >>> 16) | (origHi << 16);
        let pzHi = origHi >>> 16;

        // -Z: result[p] = mask[p-16], crosses lo/hi at lz=2->lz=1
        let mzLo = origLo << 16;
        let mzHi = (origHi << 16) | (origLo >>> 16);

        // --- Cross-block contributions ---
        // For each face direction: look up the adjacent block, extract opposite-face
        // bits, shift to align with our face positions, OR into the direction mask.

        // +X: our lx=3 face <- adjacent's lx=0 face (shifted left by 3)
        addCrossFace(bx + 1, by, bz, solidSet, mixedMap, masks,
            FACE_X3, FACE_X0, 3, true, pxLo, pxHi,
            (lo, hi) => {
                pxLo = lo; pxHi = hi;
            });

        // -X: our lx=0 face <- adjacent's lx=3 face (shifted right by 3)
        addCrossFace(bx - 1, by, bz, solidSet, mixedMap, masks,
            FACE_X0, FACE_X3, 3, false, mxLo, mxHi,
            (lo, hi) => {
                mxLo = lo; mxHi = hi;
            });

        // +Y: our ly=3 face <- adjacent's ly=0 face (shifted left by 12)
        addCrossFace(bx, by + 1, bz, solidSet, mixedMap, masks,
            FACE_Y3, FACE_Y0, 12, true, pyLo, pyHi,
            (lo, hi) => {
                pyLo = lo; pyHi = hi;
            });

        // -Y: our ly=0 face <- adjacent's ly=3 face (shifted right by 12)
        addCrossFace(bx, by - 1, bz, solidSet, mixedMap, masks,
            FACE_Y0, FACE_Y3, 12, false, myLo, myHi,
            (lo, hi) => {
                myLo = lo; myHi = hi;
            });

        // +Z: our lz=3 face (hi bits 16-31) <- adjacent's lz=0 face (lo bits 0-15)
        addCrossFaceZ(bx, by, bz + 1, solidSet, mixedMap, masks, true, pzLo, pzHi,
            (lo, hi) => {
                pzLo = lo; pzHi = hi;
            });

        // -Z: our lz=0 face (lo bits 0-15) <- adjacent's lz=3 face (hi bits 16-31)
        addCrossFaceZ(bx, by, bz - 1, solidSet, mixedMap, masks, false, mzLo, mzHi,
            (lo, hi) => {
                mzLo = lo; mzHi = hi;
            });

        // --- Apply operations ---

        // Remove isolated voxels: keep only those with at least one occupied neighbor
        const neighborLo = pxLo | mxLo | pyLo | myLo | pzLo | mzLo;
        const neighborHi = pxHi | mxHi | pyHi | myHi | pzHi | mzHi;
        let lo = origLo & neighborLo;
        let hi = origHi & neighborHi;

        // Fill isolated empties: fill where all 6 neighbors are occupied
        const fillLo = ~lo & pxLo & mxLo & pyLo & myLo & pzLo & mzLo;
        const fillHi = ~hi & pxHi & mxHi & pyHi & myHi & pzHi & mzHi;
        lo |= fillLo;
        hi |= fillHi;

        voxelsRemoved += popcount(origLo & ~lo) + popcount(origHi & ~hi);
        voxelsFilled += popcount(lo & ~origLo) + popcount(hi & ~origHi);

        newMasks[i * 2] = lo;
        newMasks[i * 2 + 1] = hi;
    }

    // Rebuild accumulator with state transitions (mixed->empty, mixed->solid)
    const result = new BlockAccumulator();

    for (let i = 0; i < mixed.morton.length; i++) {
        const lo = newMasks[i * 2];
        const hi = newMasks[i * 2 + 1];
        // addBlock handles empty (discarded) and solid (stored without mask)
        result.addBlock(mixed.morton[i], lo, hi);
    }

    for (let i = 0; i < solid.length; i++) {
        result.addBlock(solid[i], SOLID_MASK, SOLID_MASK);
    }

    logger.log(
        `voxel filter: ${voxelsRemoved} voxels removed, ${voxelsFilled} voxels filled`
    );

    return result;
}

// ============================================================================
// Cross-block face helpers
// ============================================================================

/**
 * Add cross-block face contribution for X/Y directions (shift stays within lo/hi words).
 *
 * @param nx - Adjacent block X coordinate.
 * @param ny - Adjacent block Y coordinate.
 * @param nz - Adjacent block Z coordinate.
 * @param solidSet - Set of Morton codes for solid blocks.
 * @param mixedMap - Map from Morton code to index in the mixed masks array.
 * @param masks - Interleaved voxel masks for mixed blocks [lo0, hi0, lo1, hi1, ...].
 * @param ourFaceMask - Bit mask selecting our block's face positions.
 * @param adjFaceMask - Bit mask selecting the adjacent block's opposite face positions.
 * @param shiftAmount - Number of bits to shift the adjacent face into our face positions.
 * @param shiftLeft - True to shift left, false to shift right.
 * @param curLo - Current low 32 bits of the direction mask.
 * @param curHi - Current high 32 bits of the direction mask.
 * @param write - Callback to write the updated (lo, hi) direction mask.
 */
function addCrossFace(
    nx: number, ny: number, nz: number,
    solidSet: Set<number>,
    mixedMap: Map<number, number>,
    masks: number[],
    ourFaceMask: number,
    adjFaceMask: number,
    shiftAmount: number,
    shiftLeft: boolean,
    curLo: number, curHi: number,
    write: (lo: number, hi: number) => void
): void {
    const adjMorton = xyzToMorton(nx, ny, nz);

    if (solidSet.has(adjMorton)) {
        write(curLo | ourFaceMask, curHi | ourFaceMask);
        return;
    }

    const adjIdx = mixedMap.get(adjMorton);
    if (adjIdx === undefined) {
        write(curLo, curHi);
        return;
    }

    const adjLo = masks[adjIdx * 2];
    const adjHi = masks[adjIdx * 2 + 1];
    const faceLo = adjLo & adjFaceMask;
    const faceHi = adjHi & adjFaceMask;

    if (shiftLeft) {
        write(curLo | (faceLo << shiftAmount), curHi | (faceHi << shiftAmount));
    } else {
        write(curLo | (faceLo >>> shiftAmount), curHi | (faceHi >>> shiftAmount));
    }
}

/**
 * Add cross-block face contribution for Z direction (crosses lo/hi boundary).
 *
 * +Z: our lz=3 (hi bits 16-31) <- adjacent's lz=0 (lo bits 0-15), shift left by 16
 * -Z: our lz=0 (lo bits 0-15) <- adjacent's lz=3 (hi bits 16-31), shift right by 16
 *
 * @param nx - Adjacent block X coordinate.
 * @param ny - Adjacent block Y coordinate.
 * @param nz - Adjacent block Z coordinate.
 * @param solidSet - Set of Morton codes for solid blocks.
 * @param mixedMap - Map from Morton code to index in the mixed masks array.
 * @param masks - Interleaved voxel masks for mixed blocks [lo0, hi0, lo1, hi1, ...].
 * @param plusZ - True for +Z direction, false for -Z.
 * @param curLo - Current low 32 bits of the direction mask.
 * @param curHi - Current high 32 bits of the direction mask.
 * @param write - Callback to write the updated (lo, hi) direction mask.
 */
function addCrossFaceZ(
    nx: number, ny: number, nz: number,
    solidSet: Set<number>,
    mixedMap: Map<number, number>,
    masks: number[],
    plusZ: boolean,
    curLo: number, curHi: number,
    write: (lo: number, hi: number) => void
): void {
    const adjMorton = xyzToMorton(nx, ny, nz);

    if (solidSet.has(adjMorton)) {
        if (plusZ) {
            write(curLo, curHi | FACE_Z3_HI);
        } else {
            write(curLo | FACE_Z0_LO, curHi);
        }
        return;
    }

    const adjIdx = mixedMap.get(adjMorton);
    if (adjIdx === undefined) {
        write(curLo, curHi);
        return;
    }

    const adjLo = masks[adjIdx * 2];
    const adjHi = masks[adjIdx * 2 + 1];

    if (plusZ) {
        write(curLo, curHi | ((adjLo & FACE_Z0_LO) << 16));
    } else {
        write(curLo | ((adjHi & FACE_Z3_HI) >>> 16), curHi);
    }
}

export { filterAndFillBlocks };
