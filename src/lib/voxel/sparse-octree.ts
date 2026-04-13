import { Vec3 } from 'playcanvas';

import { BlockMaskBuffer } from './block-mask-buffer';
import { xyzToMorton, mortonToXYZ, popcount } from './morton';
import type { Bounds } from '../data-table/gaussian-aabb';
import { logger } from '../utils/logger';

/**
 * Solid leaf node marker: childMask = 0xFF, baseOffset = 0.
 * This is unambiguous because BFS layout guarantees children always come after
 * their parent, so baseOffset = 0 is never valid for an interior node.
 */
const SOLID_LEAF_MARKER = 0xFF000000 >>> 0;

/**
 * Get the offset to a child node given a parent's child mask and octant.
 * Uses popcount to count how many children come before this octant.
 *
 * @param mask - 8-bit child mask from parent node
 * @param octant - Octant index (0-7)
 * @returns Offset from base child pointer
 */
function getChildOffset(mask: number, octant: number): number {
    const prefix = (1 << octant) - 1;
    return popcount(mask & prefix);
}

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

    // --- Phase 1: Combine blocks into SoA arrays and sort by Morton code ---
    // Uses typed arrays to reduce per-element memory overhead. The block scope
    // allows the unsorted intermediates (mortons, types, maskIndices, sortOrder)
    // to be GC'd before Phase 2 begins.

    let sortedMortons!: Float64Array;
    let sortedTypes!: Uint8Array;
    let sortedMaskIndices!: Int32Array;
    {
        const mortons = new Float64Array(totalBlocks);
        const types = new Uint8Array(totalBlocks);
        const maskIndices = new Int32Array(totalBlocks);

        let idx = 0;
        for (let i = 0; i < mixed.morton.length; i++) {
            mortons[idx] = mixed.morton[i];
            types[idx] = BlockType.Mixed;
            maskIndices[idx] = i;
            idx++;
        }
        for (let i = 0; i < solid.length; i++) {
            mortons[idx] = solid[i];
            types[idx] = BlockType.Solid;
            maskIndices[idx] = -1;
            idx++;
        }

        const sortOrder = new Uint32Array(totalBlocks);
        for (let i = 0; i < totalBlocks; i++) sortOrder[i] = i;
        sortOrder.sort((a: number, b: number) => mortons[a] - mortons[b]);

        sortedMortons = new Float64Array(totalBlocks);
        sortedTypes = new Uint8Array(totalBlocks);
        sortedMaskIndices = new Int32Array(totalBlocks);
        for (let i = 0; i < totalBlocks; i++) {
            const si = sortOrder[i];
            sortedMortons[i] = mortons[si];
            sortedTypes[i] = types[si];
            sortedMaskIndices[i] = maskIndices[si];
        }
    }

    const tSort = performance.now();

    // --- Phase 2: Build tree bottom-up level by level using linear scan ---
    // Instead of Map<number, BuildNode> per level, we use sorted parallel
    // arrays and exploit the fact that sorted Morton codes make parent
    // grouping a simple linear scan (consecutive entries with the same
    // floor(morton/8) share a parent).

    // Inner progress: 10 anonymous steps
    logger.progress.begin(10);
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

    // Store level data for each tree level (level 0 = leaves, higher = toward root)
    const levels: LevelData[] = [];

    // Current level data starts as the sorted leaf blocks
    let curMortons: Float64Array | number[] = sortedMortons;
    let curTypes: Uint8Array | number[] = sortedTypes;
    let curMaskIndices: Int32Array | number[] = sortedMaskIndices;
    let curChildMasks: number[] | null = null;

    // 1 step for init
    logger.progress.step();
    octreeStep++;

    // Build up level by level
    let actualDepth = treeDepth;
    const levelSteps = 8;

    for (let level = 0; level < treeDepth; level++) {
        // Report inner progress scaled to levelSteps
        const targetStep = 1 + Math.min(levelSteps, Math.floor((level + 1) / treeDepth * levelSteps));
        while (octreeStep < targetStep) {
            logger.progress.step();
            octreeStep++;
        }

        // Save current level before building the next one above
        levels.push({
            mortons: curMortons,
            types: curTypes,
            maskIndices: curMaskIndices,
            childMasks: curChildMasks
        });

        // Build next level using linear scan on sorted data.
        // Since curMortons is sorted, entries sharing the same parent
        // (floor(morton/8)) are contiguous — no Map needed.
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

            // Scan all consecutive entries that share this parent
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
                // All 8 children are solid — collapse to solid parent
                nextMortons.push(parentMorton);
                nextTypes.push(BlockType.Solid);
                nextMaskIndices.push(-1);
                nextChildMasks.push(0);
            } else {
                // Interior node with sparse children
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

        // Break when the tree is empty or has converged to a single root at Morton 0.
        // We must NOT break early if the single remaining node has a non-zero Morton,
        // because the reader reconstructs Morton codes starting from root Morton 0.
        if (curMortons.length === 0 ||
            (curMortons.length === 1 && curMortons[0] === 0)) {
            actualDepth = level + 1;
            break;
        }
    }

    // Save the root level
    levels.push({
        mortons: curMortons,
        types: curTypes,
        maskIndices: curMaskIndices,
        childMasks: curChildMasks
    });

    // Flush remaining level steps
    while (octreeStep < 9) {
        logger.progress.step();
        octreeStep++;
    }

    const tBuild = performance.now();

    // --- Phase 3: Flatten tree to Laine-Karras format ---
    // Uses wave-based BFS on level arrays, avoiding BuildNode objects
    // and the O(n²) queue.shift() of the original approach.
    const result = flattenTreeFromLevels(
        levels, mixed.masks, gridBounds, sceneBounds, voxelResolution, actualDepth
    );

    const tFlatten = performance.now();

    // Final step (10th)
    logger.progress.step();

    return result;
}

/**
 * Flatten the level-based tree into Laine-Karras format arrays using
 * wave-based BFS traversal from root down through levels.
 *
 * Uses parallel arrays for BFS waves (no per-node object allocation)
 * and binary search on sorted level mortons to locate children.
 *
 * @param levels - Array of per-level SoA data (index 0 = leaves, last = root).
 * @param mixedMasks - Interleaved voxel masks for mixed leaf blocks.
 * @param gridBounds - Grid bounds aligned to block boundaries.
 * @param sceneBounds - Original Gaussian scene bounds.
 * @param voxelResolution - Size of each voxel in world units.
 * @param treeDepth - Maximum tree depth.
 * @returns Sparse octree structure in Laine-Karras format.
 */
function flattenTreeFromLevels(
    levels: LevelData[],
    mixedMasks: number[],
    gridBounds: Bounds,
    sceneBounds: Bounds,
    voxelResolution: number,
    treeDepth: number
): SparseOctree {
    const rootLevel = levels[levels.length - 1];

    if (rootLevel.mortons.length === 0) {
        // Empty tree
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

    // Upper bound on total nodes (not all may be reachable if solids collapsed)
    let maxNodes = 0;
    for (let l = 0; l < levels.length; l++) {
        maxNodes += levels[l].mortons.length;
    }

    const nodes = new Uint32Array(maxNodes);
    const leafDataList: number[] = [];
    let numInteriorNodes = 0;
    let numMixedLeaves = 0;
    let emitPos = 0;

    // BFS wave as parallel arrays (avoids object allocation per queue entry)
    let waveLi: number[] = [];
    let waveIi: number[] = [];

    // Initialize wave with root level entries
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
        // Clear interior tracking arrays
        intPos.length = 0;
        intLi.length = 0;
        intIi.length = 0;
        intMask.length = 0;

        // Emit all nodes in this wave
        for (let w = 0; w < waveLi.length; w++) {
            const li = waveLi[w];
            const ii = waveIi[w];
            const level = levels[li];
            const type = level.types[ii];

            // A node is a leaf if it's Solid (at any level, collapsed or original)
            // or if it's at level 0 (the leaf block level).
            const isLeaf = (type === BlockType.Solid) || (li === 0);

            if (isLeaf) {
                if (type === BlockType.Solid) {
                    nodes[emitPos] = SOLID_LEAF_MARKER;
                } else {
                    // Mixed leaf — store index into leafData
                    const maskIdx = level.maskIndices[ii];
                    const leafDataIndex = leafDataList.length >> 1;
                    leafDataList.push(mixedMasks[maskIdx * 2]);
                    leafDataList.push(mixedMasks[maskIdx * 2 + 1]);
                    nodes[emitPos] = leafDataIndex & 0x00FFFFFF;
                    numMixedLeaves++;
                }
            } else {
                // Interior node — record position for backfill after wave
                intPos.push(emitPos);
                intLi.push(li);
                intIi.push(ii);
                intMask.push(level.childMasks![ii]);
                numInteriorNodes++;
                // Placeholder (will be filled below)
                nodes[emitPos] = 0;
            }
            emitPos++;
        }

        // Build next wave from children of interior nodes.
        // Backfill interior node encodings with correct baseOffset.
        const nextWaveLi: number[] = [];
        const nextWaveIi: number[] = [];
        let nextChildStart = emitPos;

        for (let j = 0; j < intPos.length; j++) {
            const childMask = intMask[j];
            const childCount = popcount(childMask);

            // Encode interior node: mask in high byte, baseOffset in low 24 bits
            nodes[intPos[j]] = ((childMask & 0xFF) << 24) | (nextChildStart & 0x00FFFFFF);

            // Find children in the level below using binary search.
            // Since each level's mortons are sorted, this is O(log n) per lookup.
            const childLi = intLi[j] - 1;
            const childLevel = levels[childLi];
            const myMorton = levels[intLi[j]].mortons[intIi[j]];
            const childMortonBase = myMorton * 8;
            const childMortonEnd = childMortonBase + 8;
            const childMortons = childLevel.mortons;

            // Binary search for first child with morton >= childMortonBase
            let lo = 0;
            let hi = childMortons.length;
            while (lo < hi) {
                const mid = (lo + hi) >> 1;
                if (childMortons[mid] < childMortonBase) lo = mid + 1;
                else hi = mid;
            }

            // Collect all children in morton order (they are contiguous in sorted array)
            while (lo < childMortons.length && childMortons[lo] < childMortonEnd) {
                nextWaveLi.push(childLi);
                nextWaveIi.push(lo);
                lo++;
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
        leafData: new Uint32Array(leafDataList)
    };
}

/**
 * Align bounds to 4x4x4 block boundaries.
 *
 * @param minX - Scene minimum X
 * @param minY - Scene minimum Y
 * @param minZ - Scene minimum Z
 * @param maxX - Scene maximum X
 * @param maxY - Scene maximum Y
 * @param maxZ - Scene maximum Z
 * @param voxelResolution - Size of each voxel
 * @returns Aligned bounds
 */
function alignGridBounds(
    minX: number, minY: number, minZ: number,
    maxX: number, maxY: number, maxZ: number,
    voxelResolution: number
): Bounds {
    const blockSize = 4 * voxelResolution;
    return {
        min: new Vec3(
            Math.floor(minX / blockSize) * blockSize,
            Math.floor(minY / blockSize) * blockSize,
            Math.floor(minZ / blockSize) * blockSize
        ),
        max: new Vec3(
            Math.ceil(maxX / blockSize) * blockSize,
            Math.ceil(maxY / blockSize) * blockSize,
            Math.ceil(maxZ / blockSize) * blockSize
        )
    };
}

// ============================================================================
// Exports
// ============================================================================

export {
    // Octree construction
    buildSparseOctree,
    alignGridBounds,

    // Constants
    SOLID_LEAF_MARKER,
    getChildOffset
};

export type { SparseOctree, Bounds };
