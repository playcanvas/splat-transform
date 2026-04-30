import { Vec3 } from 'playcanvas';

import type { Bounds } from '../data-table';
import { logger } from '../utils';
import { BlockMaskBuffer } from '../voxel/block-mask-buffer';
import { xyzToMorton, mortonToXYZ, popcount, getChildOffset } from '../voxel/morton';

/**
 * Solid leaf node marker: childMask = 0xFF, baseOffset = 0.
 * This is unambiguous because BFS layout guarantees children always come after
 * their parent, so baseOffset = 0 is never valid for an interior node.
 */
const SOLID_LEAF_MARKER = 0xFF000000 >>> 0;

// ============================================================================
// Sparse Octree Types
// ============================================================================

/**
 * Sparse voxel octree using Laine-Karras node format.
 */
interface SparseOctree {
    /** Grid bounds aligned to 4x4x4 block boundaries */
    gridBounds: Bounds;

    /** Original Gaussian scene bounds */
    sceneBounds: Bounds;

    /** Size of each voxel in world units */
    voxelResolution: number;

    /** Voxels per leaf dimension (always 4) */
    leafSize: number;

    /** Maximum tree depth */
    treeDepth: number;

    /** Number of interior nodes */
    numInteriorNodes: number;

    /** Number of mixed leaf nodes */
    numMixedLeaves: number;

    /** All nodes in Laine-Karras format (interior + leaves) */
    nodes: Uint32Array;

    /** Voxel masks for mixed leaves: pairs of u32 (lo, hi) */
    leafData: Uint32Array;
}

// ============================================================================
// Octree Node Types (during construction)
// ============================================================================

/** Block type enumeration */
const enum BlockType {
    Empty = 0,
    Solid = 1,
    Mixed = 2
}

/**
 * Per-level data stored during bottom-up construction.
 * Uses Structure-of-Arrays layout to avoid per-node object allocation.
 */
interface LevelData {
    /** Sorted Morton codes for nodes at this level */
    mortons: Float64Array | number[];
    /** Block type for each node (Solid or Mixed) */
    types: Uint8Array | number[];
    /** For level-0 Mixed nodes: index into mixed.masks. Otherwise -1. */
    maskIndices: Int32Array | number[];
    /** For interior nodes (Mixed at level > 0): 8-bit child presence mask. Null for leaf level. */
    childMasks: number[] | null;
}

// ============================================================================
// Parallel Sort
// ============================================================================

/**
 * Iterative quicksort with median-of-three pivot and an insertion-sort fallback
 * for small partitions. Sorts three parallel typed arrays in place, ordering
 * entries by `mortons` ascending.
 *
 * Avoids `TypedArray.prototype.sort(comparefn)` so we are not bounded by
 * FixedArray::kMaxLength (~134M elements with V8 pointer compression).
 *
 * @param mortons - Morton codes; sort key.
 * @param types - Block type per entry; permuted alongside mortons.
 * @param maskIndices - Mask index per entry; permuted alongside mortons.
 * @param n - Number of valid entries to sort (from index 0).
 */
function sortParallelByMorton(
    mortons: Float64Array,
    types: Uint8Array,
    maskIndices: Int32Array,
    n: number
): void {
    if (n < 2) return;

    const swap = (a: number, b: number): void => {
        const tm = mortons[a]; mortons[a] = mortons[b]; mortons[b] = tm;
        const tt = types[a]; types[a] = types[b]; types[b] = tt;
        const ti = maskIndices[a]; maskIndices[a] = maskIndices[b]; maskIndices[b] = ti;
    };

    // Stack holds [lo, hi] pairs. Always pushing the larger partition first
    // bounds depth to O(log n); 64 entries (32 frames) covers n up to 2^32.
    const stack = new Int32Array(64);
    let sp = 0;
    stack[sp++] = 0;
    stack[sp++] = n - 1;

    while (sp > 0) {
        const hi = stack[--sp];
        const lo = stack[--sp];

        if (hi - lo < 16) {
            // Insertion sort
            for (let i = lo + 1; i <= hi; i++) {
                const km = mortons[i], kt = types[i], ki = maskIndices[i];
                let j = i - 1;
                while (j >= lo && mortons[j] > km) {
                    mortons[j + 1] = mortons[j];
                    types[j + 1] = types[j];
                    maskIndices[j + 1] = maskIndices[j];
                    j--;
                }
                mortons[j + 1] = km;
                types[j + 1] = kt;
                maskIndices[j + 1] = ki;
            }
            continue;
        }

        // Median-of-three: order mortons[lo] <= mortons[mid] <= mortons[hi],
        // then stash the pivot at hi-1 so the inner loop has sentinels at both ends.
        const mid = (lo + hi) >>> 1;
        if (mortons[lo] > mortons[mid]) swap(lo, mid);
        if (mortons[lo] > mortons[hi]) swap(lo, hi);
        if (mortons[mid] > mortons[hi]) swap(mid, hi);
        swap(mid, hi - 1);
        const pivot = mortons[hi - 1];

        let i = lo;
        let j = hi - 1;
        while (true) {
            while (mortons[++i] < pivot) { /* sentinel at hi */ }
            while (mortons[--j] > pivot) { /* sentinel at lo */ }
            if (i >= j) break;
            swap(i, j);
        }
        swap(i, hi - 1);

        // Push larger partition first so the smaller is processed next.
        const leftSize = i - 1 - lo;
        const rightSize = hi - (i + 1);
        if (leftSize > rightSize) {
            stack[sp++] = lo;
            stack[sp++] = i - 1;
            if (rightSize > 0) {
                stack[sp++] = i + 1;
                stack[sp++] = hi;
            }
        } else {
            if (rightSize > 0) {
                stack[sp++] = i + 1;
                stack[sp++] = hi;
            }
            if (leftSize > 0) {
                stack[sp++] = lo;
                stack[sp++] = i - 1;
            }
        }
    }
}

/**
 * Iterative quicksort that sorts mixed-block mortons in place, permuting the
 * paired interleaved [lo, hi] mask pairs alongside. Same shape as
 * `sortParallelByMorton`, but tailored to the (Float64 morton, Uint32×2 mask)
 * layout used by `BlockMaskBuffer` so the buffer's typed arrays can serve as
 * the sorted data directly — no SoA copy required.
 *
 * @param mortons - Morton codes; sort key.
 * @param masks - Interleaved [lo, hi] mask pairs (length = 2 * n).
 * @param n - Number of mixed entries to sort.
 */
function sortMixedByMorton(
    mortons: Float64Array,
    masks: Uint32Array,
    n: number
): void {
    if (n < 2) return;

    const swap = (a: number, b: number): void => {
        const tm = mortons[a]; mortons[a] = mortons[b]; mortons[b] = tm;
        const a2 = a * 2, b2 = b * 2;
        const tlo = masks[a2]; masks[a2] = masks[b2]; masks[b2] = tlo;
        const thi = masks[a2 + 1]; masks[a2 + 1] = masks[b2 + 1]; masks[b2 + 1] = thi;
    };

    const stack = new Int32Array(64);
    let sp = 0;
    stack[sp++] = 0;
    stack[sp++] = n - 1;

    while (sp > 0) {
        const hi = stack[--sp];
        const lo = stack[--sp];

        if (hi - lo < 16) {
            // Insertion sort
            for (let i = lo + 1; i <= hi; i++) {
                const km = mortons[i];
                const kl = masks[i * 2];
                const kh = masks[i * 2 + 1];
                let j = i - 1;
                while (j >= lo && mortons[j] > km) {
                    mortons[j + 1] = mortons[j];
                    masks[(j + 1) * 2] = masks[j * 2];
                    masks[(j + 1) * 2 + 1] = masks[j * 2 + 1];
                    j--;
                }
                mortons[j + 1] = km;
                masks[(j + 1) * 2] = kl;
                masks[(j + 1) * 2 + 1] = kh;
            }
            continue;
        }

        const mid = (lo + hi) >>> 1;
        if (mortons[lo] > mortons[mid]) swap(lo, mid);
        if (mortons[lo] > mortons[hi]) swap(lo, hi);
        if (mortons[mid] > mortons[hi]) swap(mid, hi);
        swap(mid, hi - 1);
        const pivot = mortons[hi - 1];

        let i = lo;
        let j = hi - 1;
        while (true) {
            while (mortons[++i] < pivot) { /* sentinel at hi */ }
            while (mortons[--j] > pivot) { /* sentinel at lo */ }
            if (i >= j) break;
            swap(i, j);
        }
        swap(i, hi - 1);

        const leftSize = i - 1 - lo;
        const rightSize = hi - (i + 1);
        if (leftSize > rightSize) {
            stack[sp++] = lo;
            stack[sp++] = i - 1;
            if (rightSize > 0) {
                stack[sp++] = i + 1;
                stack[sp++] = hi;
            }
        } else {
            if (rightSize > 0) {
                stack[sp++] = i + 1;
                stack[sp++] = hi;
            }
            if (leftSize > 0) {
                stack[sp++] = lo;
                stack[sp++] = i - 1;
            }
        }
    }
}

/**
 * Binary-search lowest index i in `arr[0..n)` such that `arr[i] >= target`.
 * Returns `n` if all entries are less than `target`.
 *
 * @param arr - Sorted Float64Array.
 * @param target - Value to search for.
 * @param n - Upper bound of the search range (exclusive).
 * @returns Smallest index `i` with `arr[i] >= target`, or `n`.
 */
function lowerBoundF64(arr: Float64Array, target: number, n: number): number {
    let lo = 0;
    let hi = n;
    while (lo < hi) {
        const mid = (lo + hi) >>> 1;
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// ============================================================================
// Octree Construction
// ============================================================================

/**
 * Build a sparse octree from accumulated voxelization blocks.
 *
 * Uses Structure-of-Arrays (SoA) representation and linear scans on sorted
 * Morton codes instead of Maps and per-node objects for performance.
 *
 * @param buffer - BlockMaskBuffer containing voxelized blocks
 * @param gridBounds - Grid bounds aligned to block boundaries
 * @param sceneBounds - Original scene bounds
 * @param voxelResolution - Size of each voxel in world units
 * @returns Sparse octree structure
 */
function buildSparseOctree(
    buffer: BlockMaskBuffer,
    gridBounds: Bounds,
    sceneBounds: Bounds,
    voxelResolution: number
): SparseOctree {
    const tProfile = performance.now();

    const mixed = buffer.getMixedBlocks();
    const solid = buffer.getSolidBlocks();
    const totalBlocks = mixed.morton.length + solid.length;

    // --- Phase 1: Sort the buffer's existing typed arrays in place ---
    // Level 0 (leaves) is represented as TWO sorted streams — solid mortons
    // and mixed mortons (with paired masks). Sorting in place avoids
    // allocating SoA copies (Float64Array+Uint8Array+Int32Array, all sized to
    // totalBlocks); for a buffer with N_solid solid + N_mixed mixed entries,
    // that's ~13 * (N_solid + N_mixed) bytes saved.
    //
    // We avoid `comparefn` typed-array sort because V8 routes it through a
    // regular-array path bounded by FixedArray::kMaxLength (~134M with
    // pointer compression) and throws past it. Instead:
    //   - Native `Float64Array.sort()` (no comparefn) on the solid stream:
    //     numeric sort, no fallback path, no size limit.
    //   - Custom `sortMixedByMorton` for the mixed stream: same iterative
    //     quicksort as `sortParallelByMorton` but tailored to the
    //     (Float64 morton, Uint32×2 mask) layout.

    const solidStream = solid;
    const mixedStream = mixed.morton;
    const mixedMasks = mixed.masks;
    const nSolid = solid.length;
    const nMixed = mixed.morton.length;

    if (nSolid > 1) solidStream.sort();
    if (nMixed > 1) sortMixedByMorton(mixedStream, mixedMasks, nMixed);

    const tSort = performance.now();

    // --- Phase 2: Build tree bottom-up level by level using linear scan ---
    // Level 0 lives in `solidStream` + `mixedStream` (dual streams, sorted).
    // We build level 1 by a two-pointer merge of these streams; subsequent
    // levels (2, 3, ...) are built from JS-array level data via the same
    // single-array linear scan as before.
    //
    // `levels[]` holds INTERIOR levels only (no level 0). `levels[0]` is the
    // first interior level (= original "level 1"); `levels[length-1]` is the
    // root. Phase 3 special-cases level 0 access via a wave-entry sentinel
    // (`li === -1`).

    const bar = logger.bar('Building tree', 10);
    let octreeStep = 0;

    // Calculate tree depth based on grid size
    const gridSize = new Vec3(
        gridBounds.max.x - gridBounds.min.x,
        gridBounds.max.y - gridBounds.min.y,
        gridBounds.max.z - gridBounds.min.z
    );
    const blockSize = voxelResolution * 4;
    const blocksPerAxis = Math.max(
        Math.ceil(gridSize.x / blockSize),
        Math.ceil(gridSize.y / blockSize),
        Math.ceil(gridSize.z / blockSize)
    );
    const treeDepth = Math.max(1, Math.ceil(Math.log2(blocksPerAxis)));

    const levels: LevelData[] = [];

    // === Build level 1 from dual-stream level 0 ===
    // Two-pointer merge: at each step, the smaller-morton stream is the next
    // child to process. Group children with the same `floor(morton/8)` parent.
    // childMask records which octants are present; allSolid tracks whether we
    // can collapse this parent into a SOLID leaf at level 1.
    let curMortons: number[] = [];
    let curTypes: number[] = [];
    let curMaskIndices: number[] = [];
    let curChildMasks: number[] | null = [];

    {
        let sI = 0;
        let mI = 0;
        const nextChildMasks = curChildMasks!;
        while (sI < nSolid || mI < nMixed) {
            const sM0 = sI < nSolid ? solidStream[sI] : Number.POSITIVE_INFINITY;
            const mM0 = mI < nMixed ? mixedStream[mI] : Number.POSITIVE_INFINITY;
            const minMorton = sM0 < mM0 ? sM0 : mM0;
            const parentMorton = Math.floor(minMorton / 8);
            let childMask = 0;
            let allSolid = true;
            let childCount = 0;

            while (true) {
                const sM = sI < nSolid ? solidStream[sI] : Number.POSITIVE_INFINITY;
                const mM = mI < nMixed ? mixedStream[mI] : Number.POSITIVE_INFINITY;
                const cur = sM < mM ? sM : mM;
                if (!isFinite(cur) || Math.floor(cur / 8) !== parentMorton) break;
                childMask |= 1 << (cur % 8);
                childCount++;
                if (sM < mM) {
                    sI++;
                } else {
                    allSolid = false;
                    mI++;
                }
            }

            if (allSolid && childCount === 8) {
                curMortons.push(parentMorton);
                curTypes.push(BlockType.Solid);
                curMaskIndices.push(-1);
                nextChildMasks.push(0);
            } else {
                curMortons.push(parentMorton);
                curTypes.push(BlockType.Mixed);
                curMaskIndices.push(-1);
                nextChildMasks.push(childMask);
            }
        }
    }

    // 1 step for init / level-1 build
    bar.tick();
    octreeStep++;

    let actualDepth = treeDepth;
    const levelSteps = 8;

    // === Build levels 2..treeDepth from level 1 upward ===
    // Same logic as before — single-array linear scan. We push each level
    // before building the next, then push the final root after the loop.
    if (curMortons.length === 0) {
        // Empty input: nothing to push. levels stays empty.
        actualDepth = 1;
    } else if (curMortons.length === 1 && curMortons[0] === 0) {
        // Level 1 IS the root.
        actualDepth = 1;
        levels.push({
            mortons: curMortons,
            types: curTypes,
            maskIndices: curMaskIndices,
            childMasks: curChildMasks
        });
    } else {
        for (let level = 1; level < treeDepth; level++) {
            const targetStep = 1 + Math.min(levelSteps, Math.floor((level + 1) / treeDepth * levelSteps));
            while (octreeStep < targetStep) {
                bar.tick();
                octreeStep++;
            }

            // Push current level before building one above it
            levels.push({
                mortons: curMortons,
                types: curTypes,
                maskIndices: curMaskIndices,
                childMasks: curChildMasks
            });

            // Build next level using linear scan on sorted data.
            const n = curMortons.length;
            const nextMortons: number[] = [];
            const nextTypes: number[] = [];
            const nextMaskIndices: number[] = [];
            const nextChildMasks: number[] = [];

            let i = 0;
            while (i < n) {
                const parentMorton = Math.floor(curMortons[i] / 8);
                let childMask = 0;
                let allSolid = true;
                let childCount = 0;

                while (i < n && Math.floor(curMortons[i] / 8) === parentMorton) {
                    const octant = curMortons[i] % 8;
                    childMask |= (1 << octant);
                    if (curTypes[i] !== BlockType.Solid) {
                        allSolid = false;
                    }
                    childCount++;
                    i++;
                }

                if (allSolid && childCount === 8) {
                    nextMortons.push(parentMorton);
                    nextTypes.push(BlockType.Solid);
                    nextMaskIndices.push(-1);
                    nextChildMasks.push(0);
                } else {
                    nextMortons.push(parentMorton);
                    nextTypes.push(BlockType.Mixed);
                    nextMaskIndices.push(-1);
                    nextChildMasks.push(childMask);
                }
            }

            curMortons = nextMortons;
            curTypes = nextTypes;
            curMaskIndices = nextMaskIndices;
            curChildMasks = nextChildMasks;

            if (curMortons.length === 0 ||
                (curMortons.length === 1 && curMortons[0] === 0)) {
                actualDepth = level + 1;
                break;
            }
        }

        // Save root
        if (curMortons.length > 0) {
            levels.push({
                mortons: curMortons,
                types: curTypes,
                maskIndices: curMaskIndices,
                childMasks: curChildMasks
            });
        }
    }

    while (octreeStep < 9) {
        bar.tick();
        octreeStep++;
    }

    const tBuild = performance.now();

    // --- Phase 3: Flatten tree to Laine-Karras format ---
    const result = flattenTreeFromLevels(
        levels, solidStream, mixedStream, mixedMasks, nSolid, nMixed,
        gridBounds, sceneBounds, voxelResolution, actualDepth
    );

    const tFlatten = performance.now();

    bar.tick();
    bar.end();

    return result;
}

/**
 * Flatten the level-based tree into Laine-Karras format arrays using
 * wave-based BFS traversal from root down through levels.
 *
 * Level 0 (leaves) is represented as TWO sorted streams: `solidStream` (one
 * Float64 morton per solid leaf, no per-leaf data) and `mixedStream` (one
 * Float64 morton per mixed leaf, paired with `mixedMasks` at the same
 * index). Wave entries with `li === -1` refer to leaves: `ii < nMixed`
 * indicates a mixed leaf at `mixedStream[ii]`, while `ii >= nMixed`
 * indicates a solid leaf at `solidStream[ii - nMixed]`.
 *
 * `levels[i]` (for `i >= 0`) holds the i-th INTERIOR level (= original
 * "level i+1"), built bottom-up. `levels[length-1]` is the root.
 *
 * @param levels - Interior level data (index 0 = first interior level
 * above leaves, last = root).
 * @param solidStream - Sorted Float64 mortons for solid leaves.
 * @param mixedStream - Sorted Float64 mortons for mixed leaves (paired with
 * `mixedMasks` at the same index).
 * @param mixedMasks - Interleaved voxel masks for mixed leaves.
 * @param nSolid - Count of valid entries in `solidStream`.
 * @param nMixed - Count of valid entries in `mixedStream` (and pairs in
 * `mixedMasks`).
 * @param gridBounds - Grid bounds aligned to block boundaries.
 * @param sceneBounds - Original Gaussian scene bounds.
 * @param voxelResolution - Size of each voxel in world units.
 * @param treeDepth - Maximum tree depth.
 * @returns Sparse octree structure in Laine-Karras format.
 */
function flattenTreeFromLevels(
    levels: LevelData[],
    solidStream: Float64Array,
    mixedStream: Float64Array,
    mixedMasks: Uint32Array,
    nSolid: number,
    nMixed: number,
    gridBounds: Bounds,
    sceneBounds: Bounds,
    voxelResolution: number,
    treeDepth: number
): SparseOctree {
    if (levels.length === 0) {
        // Empty tree (no leaves and no interior levels).
        return {
            gridBounds,
            sceneBounds,
            voxelResolution,
            leafSize: 4,
            treeDepth,
            numInteriorNodes: 0,
            numMixedLeaves: 0,
            nodes: new Uint32Array(0),
            leafData: new Uint32Array(0)
        };
    }

    const rootLevel = levels[levels.length - 1];

    // Upper bound on total nodes (not all may be reachable if solids collapsed)
    let maxNodes = nSolid + nMixed;
    for (let l = 0; l < levels.length; l++) {
        maxNodes += levels[l].mortons.length;
    }

    const nodes = new Uint32Array(maxNodes);
    // Pre-size leafData to its upper bound (2 entries per mixed leaf).
    const leafData = new Uint32Array(nMixed * 2);
    let leafDataLen = 0;
    let numInteriorNodes = 0;
    let numMixedLeaves = 0;
    let emitPos = 0;

    // BFS wave as parallel arrays (avoids object allocation per queue entry)
    let waveLi: number[] = [];
    let waveIi: number[] = [];

    // Initialize wave with root level entries.
    const rootLi = levels.length - 1;
    for (let i = 0; i < rootLevel.mortons.length; i++) {
        waveLi.push(rootLi);
        waveIi.push(i);
    }

    // Reusable arrays for tracking interior nodes within each wave
    const intPos: number[] = [];
    const intLi: number[] = [];
    const intIi: number[] = [];
    const intMask: number[] = [];

    while (waveLi.length > 0) {
        intPos.length = 0;
        intLi.length = 0;
        intIi.length = 0;
        intMask.length = 0;

        // --- Emit all nodes in this wave ---
        for (let w = 0; w < waveLi.length; w++) {
            const li = waveLi[w];
            const ii = waveIi[w];

            if (li === -1) {
                // Level-0 leaf via dual-stream encoding.
                if (ii < nMixed) {
                    // Mixed leaf
                    const leafDataIndex = leafDataLen >> 1;
                    leafData[leafDataLen++] = mixedMasks[ii * 2];
                    leafData[leafDataLen++] = mixedMasks[ii * 2 + 1];
                    nodes[emitPos] = leafDataIndex & 0x00FFFFFF;
                    numMixedLeaves++;
                } else {
                    // Solid leaf
                    nodes[emitPos] = SOLID_LEAF_MARKER;
                }
                emitPos++;
                continue;
            }

            const level = levels[li];
            const type = level.types[ii];

            // A node is a leaf if it's Solid (at any level, collapsed or original).
            // (Originally was also "or li === 0"; with the dual-stream split,
            // level 0 leaves are handled via the li === -1 sentinel above.)
            const isLeaf = type === BlockType.Solid;

            if (isLeaf) {
                nodes[emitPos] = SOLID_LEAF_MARKER;
            } else {
                // Interior node — record position for backfill after wave
                intPos.push(emitPos);
                intLi.push(li);
                intIi.push(ii);
                intMask.push(level.childMasks![ii]);
                numInteriorNodes++;
                nodes[emitPos] = 0;
            }
            emitPos++;
        }

        // --- Build next wave from children of interior nodes ---
        const nextWaveLi: number[] = [];
        const nextWaveIi: number[] = [];
        let nextChildStart = emitPos;

        for (let j = 0; j < intPos.length; j++) {
            const childMask = intMask[j];
            const childCount = popcount(childMask);

            nodes[intPos[j]] = ((childMask & 0xFF) << 24) | (nextChildStart & 0x00FFFFFF);

            const myLi = intLi[j];
            const myMorton = levels[myLi].mortons[intIi[j]];
            const childMortonBase = myMorton * 8;
            const childMortonEnd = childMortonBase + 8;

            if (myLi === 0) {
                // Children are at level 0 → dual-stream binary search.
                // Walk both streams from their respective lower bounds in
                // morton order, emitting wave entries with `li === -1` and
                // `ii` encoded as: ii < nMixed for mixed leaves, ii >= nMixed
                // (= nMixed + solidIdx) for solid leaves.
                let sIdx = lowerBoundF64(solidStream, childMortonBase, nSolid);
                let mIdx = lowerBoundF64(mixedStream, childMortonBase, nMixed);
                while (true) {
                    const sM = sIdx < nSolid && solidStream[sIdx] < childMortonEnd ?
                        solidStream[sIdx] : Number.POSITIVE_INFINITY;
                    const mM = mIdx < nMixed && mixedStream[mIdx] < childMortonEnd ?
                        mixedStream[mIdx] : Number.POSITIVE_INFINITY;
                    if (!isFinite(sM) && !isFinite(mM)) break;
                    if (sM < mM) {
                        nextWaveLi.push(-1);
                        nextWaveIi.push(nMixed + sIdx);
                        sIdx++;
                    } else {
                        nextWaveLi.push(-1);
                        nextWaveIi.push(mIdx);
                        mIdx++;
                    }
                }
            } else {
                // Children are at an interior level → single-array binary search.
                const childLi = myLi - 1;
                const childLevel = levels[childLi];
                const childMortons = childLevel.mortons;

                let lo = 0;
                let hi = childMortons.length;
                while (lo < hi) {
                    const mid = (lo + hi) >> 1;
                    if (childMortons[mid] < childMortonBase) lo = mid + 1;
                    else hi = mid;
                }

                while (lo < childMortons.length && childMortons[lo] < childMortonEnd) {
                    nextWaveLi.push(childLi);
                    nextWaveIi.push(lo);
                    lo++;
                }
            }

            nextChildStart += childCount;
        }

        waveLi = nextWaveLi;
        waveIi = nextWaveIi;
    }

    return {
        gridBounds,
        sceneBounds,
        voxelResolution,
        leafSize: 4,
        treeDepth,
        numInteriorNodes,
        numMixedLeaves,
        nodes: emitPos === maxNodes ? nodes : nodes.slice(0, emitPos),
        leafData: leafDataLen === leafData.length ? leafData : leafData.slice(0, leafDataLen)
    };
}

export {
    buildSparseOctree,
    SOLID_LEAF_MARKER
};

export type { SparseOctree };
