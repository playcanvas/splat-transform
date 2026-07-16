import { Vec3 } from 'playcanvas';

import type { Bounds } from '../data-table';
import { logger } from '../utils';
import { BlockMaskBuffer } from '../voxel/block-mask-buffer';
import { popcount, xyzToMorton } from '../voxel/morton';
import { sortKeyMaskPairs } from '../voxel/sort-key-mask';
import {
    BLOCK_EMPTY,
    BLOCK_MIXED,
    BLOCK_SOLID,
    BLOCKS_PER_WORD,
    EVEN_BITS,
    SparseVoxelGrid,
    TYPE_MASK,
    readBlockType,
    writeBlockType
} from '../voxel/sparse-voxel-grid';

/**
 * Solid leaf node marker: childMask = 0xFF, baseOffset = 0.
 * This is unambiguous because BFS layout guarantees children always come after
 * their parent, so baseOffset = 0 is never valid for an interior node.
 */
const SOLID_LEAF_MARKER = 0xFF000000 >>> 0;

/**
 * Maximum value encodable in the low 24 bits of a Laine-Karras node word.
 * Both the interior `baseOffset` (child node index) and the mixed-leaf
 * `leafDataIndex` share this ceiling: indices range 0..MAX_24BIT_OFFSET
 * (= 0xFFFFFF = 16,777,215), giving up to 16,777,216 distinct encodable
 * values. (Interior nodes additionally reserve baseOffset=0 for
 * `SOLID_LEAF_MARKER`, so the practical interior cap is 16,777,215.)
 */
const MAX_24BIT_OFFSET = 0x00FFFFFF;
const MAX_V1_MIXED_LEAVES = MAX_24BIT_OFFSET + 1;

const DENSE_SOLID_STREAM_THRESHOLD = 8_000_000;

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

interface BuildSparseOctreeOptions {
    /** Release the input grid's backing storage after the octree has copied the data it needs. */
    consumeGrid?: boolean;
    /** Force dense-mip construction; intended for tests and benchmarks. */
    dense?: boolean;
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
 *
 * After the dual-stream refactor, level 0 is held outside `interiorLevels`
 * (in `solidStream` + `mixedStream` + `mixedMasks`), so every entry stored
 * here is an interior node and `childMasks` is always non-null.
 */
interface LevelData {
    /** Sorted Morton codes for nodes at this level */
    mortons: Float64Array;
    /** Block type for each node (Solid or Mixed) */
    types: Uint8Array;
    /** 8-bit child presence mask for each node */
    childMasks: Uint8Array;
}

/**
 * Interior nodes waiting for child emission during BFS flattening.
 * Children are written immediately, so this wave never contains leaves.
 */
interface InteriorWave {
    pos: Uint32Array;
    li: Int32Array;
    ii: Uint32Array;
    length: number;
}

interface DenseLevel {
    types: Uint32Array;
    nbx: number;
    nby: number;
    nbz: number;
    nonEmptyCount: number;
}

// ============================================================================
// Mixed-stream Sort
// ============================================================================

/**
 * Iterative quicksort that sorts mixed-block mortons in place, permuting the
 * paired interleaved [lo, hi] mask pairs alongside. Median-of-three pivot,
 * insertion-sort fallback for small partitions.
 *
 * Avoids `TypedArray.prototype.sort(comparefn)` so we are not bounded by
 * FixedArray::kMaxLength (~134M elements with V8 pointer compression).
 *
 * Tailored to the (Float64 morton, Uint32×2 mask) layout used by
 * `BlockMaskBuffer` so the buffer's typed arrays can serve as the sorted
 * data directly — no SoA copy required. The solid stream uses the native
 * `Float64Array.sort()` (no comparefn) and doesn't need a custom path.
 *
 * @param mortons - Morton codes; sort key.
 * @param masks - Interleaved [lo, hi] mask pairs (length = 2 * n).
 * @param n - Number of mixed entries to sort.
 */
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

/**
 * Create an interior BFS wave buffer.
 *
 * @param initialCapacity - Initial entry capacity.
 * @returns Empty interior wave.
 */
function createInteriorWave(initialCapacity: number): InteriorWave {
    const cap = Math.max(16, initialCapacity);
    return {
        pos: new Uint32Array(cap),
        li: new Int32Array(cap),
        ii: new Uint32Array(cap),
        length: 0
    };
}

/**
 * Append an interior node to a BFS wave, growing the typed arrays if needed.
 *
 * @param wave - Target wave.
 * @param pos - Already-emitted node position to backfill.
 * @param li - Interior level index.
 * @param ii - Node index within the level.
 */
function pushInteriorWave(
    wave: InteriorWave,
    pos: number,
    li: number,
    ii: number
): void {
    if (wave.length === wave.pos.length) {
        const cap = wave.pos.length * 2;
        const grownPos = new Uint32Array(cap);
        const grownLi = new Int32Array(cap);
        const grownIi = new Uint32Array(cap);
        grownPos.set(wave.pos);
        grownLi.set(wave.li);
        grownIi.set(wave.ii);
        wave.pos = grownPos;
        wave.li = grownLi;
        wave.ii = grownIi;
    }
    const i = wave.length++;
    wave.pos[i] = pos;
    wave.li[i] = li;
    wave.ii[i] = ii;
}

/**
 * Tree depth needed for the occupied grid dimensions.
 *
 * @param gridBounds - Grid bounds aligned to block boundaries.
 * @param voxelResolution - Size of each voxel in world units.
 * @returns Octree depth in 4x4x4 block levels.
 */
function calculateTreeDepth(gridBounds: Bounds, voxelResolution: number): number {
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
    return Math.max(1, Math.ceil(Math.log2(blocksPerAxis)));
}

/**
 * Dense-mip octree construction is better when the grid contains many solid
 * blocks that will collapse. The sorted solid-leaf stream would otherwise be
 * enormous even though those leaves mostly disappear from the final tree.
 *
 * @param totalBlocks - Total 4x4x4 blocks in the grid.
 * @param nSolid - Number of solid blocks.
 * @param nMixed - Number of mixed blocks.
 * @returns True when dense construction is likely to use less memory.
 */
function shouldUseDenseMipBuild(totalBlocks: number, nSolid: number, nMixed: number): boolean {
    return nSolid >= DENSE_SOLID_STREAM_THRESHOLD &&
        nSolid > nMixed * 4 &&
        nSolid > totalBlocks * 0.25;
}

/**
 * Build dense 2-bit type mips from a SparseVoxelGrid. Level 0 is the input
 * block grid; higher levels aggregate 2x2x2 children.
 *
 * @param grid - Source sparse voxel grid.
 * @param maxDepth - Maximum octree depth to build.
 * @returns Dense type levels from leaf blocks to root.
 */
function buildDenseTypeLevels(grid: SparseVoxelGrid, maxDepth: number): DenseLevel[] {
    const levels: DenseLevel[] = [{
        types: grid.types,
        nbx: grid.nbx,
        nby: grid.nby,
        nbz: grid.nbz,
        nonEmptyCount: 0
    }];

    for (let li = 1; li <= maxDepth; li++) {
        const prev = levels[li - 1];
        const nbx = Math.max(1, Math.ceil(prev.nbx / 2));
        const nby = Math.max(1, Math.ceil(prev.nby / 2));
        const nbz = Math.max(1, Math.ceil(prev.nbz / 2));
        const total = nbx * nby * nbz;
        const types = new Uint32Array((total + BLOCKS_PER_WORD - 1) >>> 4);
        const prevStride = prev.nbx * prev.nby;
        const stride = nbx * nby;
        let nonEmptyCount = 0;

        for (let pz = 0; pz < nbz; pz++) {
            const childZ0 = pz << 1;
            for (let py = 0; py < nby; py++) {
                const childY0 = py << 1;
                for (let px = 0; px < nbx; px++) {
                    const childX0 = px << 1;
                    let childMask = 0;
                    let allSolid = true;
                    let childCount = 0;

                    for (let oct = 0; oct < 8; oct++) {
                        const cx = childX0 + (oct & 1);
                        const cy = childY0 + ((oct >> 1) & 1);
                        const cz = childZ0 + ((oct >> 2) & 1);
                        if (cx >= prev.nbx || cy >= prev.nby || cz >= prev.nbz) continue;
                        const childIdx = cx + cy * prev.nbx + cz * prevStride;
                        const bt = readBlockType(prev.types, childIdx);
                        if (bt === BLOCK_EMPTY) continue;
                        childMask |= 1 << oct;
                        childCount++;
                        if (bt !== BLOCK_SOLID) {
                            allSolid = false;
                        }
                    }

                    if (childMask !== 0) {
                        const parentIdx = px + py * nbx + pz * stride;
                        writeBlockType(types, parentIdx, allSolid && childCount === 8 ? BLOCK_SOLID : BLOCK_MIXED);
                        nonEmptyCount++;
                    }
                }
            }
        }

        levels.push({ types, nbx, nby, nbz, nonEmptyCount });

        if (nonEmptyCount === 0) break;
        if (nonEmptyCount === 1 && readBlockType(types, 0) !== BLOCK_EMPTY) break;
    }

    return levels;
}

/**
 * Flatten dense type mips into Laine-Karras node and leaf data arrays.
 *
 * @param levels - Dense type levels from leaf blocks to root.
 * @param grid - Original grid, used for mixed leaf masks.
 * @param gridBounds - Grid bounds aligned to block boundaries.
 * @param sceneBounds - Original Gaussian scene bounds.
 * @param voxelResolution - Size of each voxel in world units.
 * @returns Sparse octree structure.
 */
function flattenDenseLevels(
    levels: Array<DenseLevel | null>,
    grid: SparseVoxelGrid,
    gridBounds: Bounds,
    sceneBounds: Bounds,
    voxelResolution: number
): SparseOctree {
    if (grid.masks.size > MAX_V1_MIXED_LEAVES) {
        throw new Error(
            `Sparse octree mixed-leaf count (${grid.masks.size}) exceeds the ` +
            `Laine-Karras 24-bit baseOffset limit (${MAX_V1_MIXED_LEAVES}). ` +
            'Use a coarser voxel resolution.'
        );
    }
    const treeDepth = Math.max(1, levels.length - 1);
    const rootLi = levels.length - 1;
    const rootLevel = levels[rootLi]!;
    const rootType = readBlockType(rootLevel.types, 0);

    if (rootType === BLOCK_EMPTY) {
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

    let nodes = new Uint32Array(Math.max(1024, Math.min(MAX_24BIT_OFFSET + 1, grid.masks.size * 3)));
    let nodeLen = 0;
    let leafData = new Uint32Array(Math.max(1024, grid.masks.size * 2));
    let leafDataLen = 0;
    let numInteriorNodes = 0;
    let numMixedLeaves = 0;

    const appendNode = (value: number): number => {
        if (nodeLen >= MAX_24BIT_OFFSET + 1) {
            throw new Error(
                `Sparse octree node count (${nodeLen + 1}) exceeds the ` +
                `Laine-Karras 24-bit baseOffset limit (${MAX_24BIT_OFFSET + 1}). ` +
                'Use a coarser voxel resolution.'
            );
        }
        if (nodeLen === nodes.length) {
            const grown = new Uint32Array(Math.min(MAX_24BIT_OFFSET + 1, nodes.length * 2));
            grown.set(nodes);
            nodes = grown;
        }
        nodes[nodeLen] = value >>> 0;
        return nodeLen++;
    };

    const appendMixedLeaf = (blockIdx: number): void => {
        const leafDataIndex = leafDataLen >> 1;
        if (leafDataIndex > MAX_24BIT_OFFSET) {
            throw new Error(
                `Sparse octree mixed-leaf count (${leafDataIndex + 1}) exceeds the ` +
                `Laine-Karras 24-bit baseOffset limit (${MAX_24BIT_OFFSET + 1}). ` +
                'Use a coarser voxel resolution.'
            );
        }
        if (leafDataLen + 2 > leafData.length) {
            const grown = new Uint32Array(leafData.length * 2);
            grown.set(leafData);
            leafData = grown;
        }
        const s = grid.masks.slot(blockIdx);
        leafData[leafDataLen++] = grid.masks.lo[s];
        leafData[leafDataLen++] = grid.masks.hi[s];
        appendNode(leafDataIndex);
        numMixedLeaves++;
    };

    let curWave = createInteriorWave(1);
    let nextWave = createInteriorWave(1024);

    const appendDenseNode = (li: number, idx: number, wave: InteriorWave): void => {
        const level = levels[li]!;
        const bt = readBlockType(level.types, idx);
        if (bt === BLOCK_SOLID) {
            appendNode(SOLID_LEAF_MARKER);
        } else if (bt === BLOCK_MIXED) {
            const pos = appendNode(0);
            pushInteriorWave(wave, pos, li, idx);
            numInteriorNodes++;
        }
    };

    appendDenseNode(rootLi, 0, curWave);

    while (curWave.length > 0) {
        nextWave.length = 0;
        const currentLi = curWave.li[0];

        for (let w = 0; w < curWave.length; w++) {
            const li = curWave.li[w];
            const parentLevel = levels[li]!;
            const childLevel = levels[li - 1]!;
            const parentIdx = curWave.ii[w];
            const px = parentIdx % parentLevel.nbx;
            const pyBz = (parentIdx / parentLevel.nbx) | 0;
            const py = pyBz % parentLevel.nby;
            const pz = (pyBz / parentLevel.nby) | 0;
            const childX0 = px << 1;
            const childY0 = py << 1;
            const childZ0 = pz << 1;
            const childStride = childLevel.nbx * childLevel.nby;
            const childStart = nodeLen;
            let childMask = 0;

            if (childStart > MAX_24BIT_OFFSET) {
                throw new Error(
                    `Sparse octree node count (${childStart + 1}) exceeds the ` +
                    `Laine-Karras 24-bit baseOffset limit (${MAX_24BIT_OFFSET + 1}). ` +
                    'Use a coarser voxel resolution.'
                );
            }

            for (let oct = 0; oct < 8; oct++) {
                const cx = childX0 + (oct & 1);
                const cy = childY0 + ((oct >> 1) & 1);
                const cz = childZ0 + ((oct >> 2) & 1);
                if (cx >= childLevel.nbx || cy >= childLevel.nby || cz >= childLevel.nbz) continue;
                const childIdx = cx + cy * childLevel.nbx + cz * childStride;
                const bt = readBlockType(childLevel.types, childIdx);
                if (bt === BLOCK_EMPTY) continue;

                childMask |= 1 << oct;
                if (li === 1) {
                    if (bt === BLOCK_SOLID) {
                        appendNode(SOLID_LEAF_MARKER);
                    } else {
                        appendMixedLeaf(childIdx);
                    }
                } else {
                    appendDenseNode(li - 1, childIdx, nextWave);
                }
            }

            nodes[curWave.pos[w]] = ((childMask & 0xFF) << 24) | childStart;
        }

        levels[currentLi] = null;
        const tmp = curWave;
        curWave = nextWave;
        nextWave = tmp;
    }

    return {
        gridBounds,
        sceneBounds,
        voxelResolution,
        leafSize: 4,
        treeDepth,
        numInteriorNodes,
        numMixedLeaves,
        nodes: nodes.slice(0, nodeLen),
        leafData: leafData.slice(0, leafDataLen)
    };
}

/**
 * Build a sparse octree from dense type mips. This avoids the sorted solid
 * Morton stream for grids dominated by solid blocks.
 *
 * @param grid - Source grid.
 * @param gridBounds - Grid bounds aligned to block boundaries.
 * @param sceneBounds - Original Gaussian scene bounds.
 * @param voxelResolution - Size of each voxel in world units.
 * @param maxDepth - Maximum tree depth.
 * @param consumeGrid - Release grid storage after flattening.
 * @returns Sparse octree structure.
 */
function buildSparseOctreeDense(
    grid: SparseVoxelGrid,
    gridBounds: Bounds,
    sceneBounds: Bounds,
    voxelResolution: number,
    maxDepth: number,
    consumeGrid: boolean
): SparseOctree {
    const bar = logger.bar('Building tree', 10);
    const levels = buildDenseTypeLevels(grid, maxDepth);
    for (let i = 0; i < 8; i++) {
        bar.tick();
    }
    const result = flattenDenseLevels(levels, grid, gridBounds, sceneBounds, voxelResolution);
    bar.tick();
    if (consumeGrid) {
        grid.releaseStorage();
    }
    bar.tick();
    bar.end();
    return result;
}

// ============================================================================
// Octree Construction
// ============================================================================

/**
 * Build a sparse octree from a SparseVoxelGrid.
 *
 * Walks the grid's `types` array word-by-word (skipping empty words), counts
 * solid + mixed blocks, then emits Morton-keyed (solidStream, mixedStream,
 * mixedMasks) typed arrays sized exactly. The streams are then sorted; this
 * is the only place Morton encoding is paid in the post-voxelization pipeline.
 *
 * @param grid - SparseVoxelGrid containing voxelized blocks.
 * @param gridBounds - Grid bounds aligned to block boundaries
 * @param sceneBounds - Original scene bounds
 * @param voxelResolution - Size of each voxel in world units
 * @param options - Build options.
 * @returns Sparse octree structure
 */
function buildSparseOctree(
    grid: SparseVoxelGrid,
    gridBounds: Bounds,
    sceneBounds: Bounds,
    voxelResolution: number,
    options: BuildSparseOctreeOptions = {}
): SparseOctree {
    // --- Phase 1: Walk grid → emit Morton streams + sort ---
    // Level 0 (leaves) is represented as TWO sorted streams — solid mortons
    // and mixed mortons (with paired masks). Two passes:
    //   1. Pre-count solid and mixed blocks (word-skip empty words).
    //   2. Allocate and fill streams (Morton encode per non-empty block).
    // Then sort both streams.
    //
    // We avoid `comparefn` typed-array sort because V8 routes it through a
    // regular-array path bounded by FixedArray::kMaxLength (~134M with
    // pointer compression) and throws past it. Instead:
    //   - Native `Float64Array.sort()` (no comparefn) on the solid stream,
    //     avoiding V8's FixedArray comparefn path.
    //   - Custom `sortKeyMaskPairs` for the mixed stream: iterative
    //     quicksort tailored to the (Float64 morton, Uint32×2 mask) layout.

    const { nbx, nby, nbz, types: gridTypes, masks: gridMasks } = grid;
    const totalBlocks = nbx * nby * nbz;
    const treeDepth = calculateTreeDepth(gridBounds, voxelResolution);
    const lastWordIdx = gridTypes.length - 1;
    const lastLanes = totalBlocks - lastWordIdx * BLOCKS_PER_WORD;
    const lastValidWordMask = lastLanes >= BLOCKS_PER_WORD ?
        0xFFFFFFFF >>> 0 :
        ((1 << (lastLanes * 2)) - 1) >>> 0;

    // Pass 1: count.
    let nSolid = 0;
    let nMixed = 0;
    for (let w = 0; w < gridTypes.length; w++) {
        let word = gridTypes[w];
        if (w === lastWordIdx) word = (word & lastValidWordMask) >>> 0;
        if (word === 0) continue;
        // Solid lanes: 0b01 = 1; Mixed lanes: 0b10 = 2.
        const solidMask = (word & EVEN_BITS) & ~((word >>> 1) & EVEN_BITS);
        const mixedMask = ((word >>> 1) & EVEN_BITS) & ~(word & EVEN_BITS);
        nSolid += popcount(solidMask >>> 0);
        nMixed += popcount(mixedMask >>> 0);
    }

    if (options.dense || shouldUseDenseMipBuild(totalBlocks, nSolid, nMixed)) {
        return buildSparseOctreeDense(
            grid, gridBounds, sceneBounds, voxelResolution, treeDepth, !!options.consumeGrid
        );
    }

    const solidStream = new Float64Array(nSolid);
    const mixedStream = new Float64Array(nMixed);
    const mixedMasks = new Uint32Array(nMixed * 2);

    // Pass 2: fill.
    let solidWriteIdx = 0;
    let mixedWriteIdx = 0;
    for (let w = 0; w < gridTypes.length; w++) {
        let word = gridTypes[w];
        if (w === lastWordIdx) word = (word & lastValidWordMask) >>> 0;
        if (word === 0) continue;
        let nonEmpty = ((word & EVEN_BITS) | ((word >>> 1) & EVEN_BITS)) >>> 0;
        const baseIdx = w * BLOCKS_PER_WORD;
        while (nonEmpty) {
            const bp = 31 - Math.clz32(nonEmpty & -nonEmpty);
            const lane = bp >>> 1;
            nonEmpty &= nonEmpty - 1;
            const blockIdx = baseIdx + lane;
            if (blockIdx >= totalBlocks) break;
            const bx = blockIdx % nbx;
            const byBz = Math.floor(blockIdx / nbx);
            const by = byBz % nby;
            const bz = Math.floor(byBz / nby);
            const morton = xyzToMorton(bx, by, bz);
            const bt = (word >>> (lane << 1)) & TYPE_MASK;
            if (bt === BLOCK_SOLID) {
                solidStream[solidWriteIdx++] = morton;
            } else if (bt === BLOCK_MIXED) {
                mixedStream[mixedWriteIdx] = morton;
                const s = gridMasks.slot(blockIdx);
                mixedMasks[mixedWriteIdx * 2] = gridMasks.lo[s];
                mixedMasks[mixedWriteIdx * 2 + 1] = gridMasks.hi[s];
                mixedWriteIdx++;
            }
        }
    }

    if (options.consumeGrid) {
        grid.releaseStorage();
    }

    if (nSolid > 1) solidStream.sort();
    if (nMixed > 1) sortKeyMaskPairs(mixedStream, mixedMasks, nMixed);

    return buildSparseOctreeFromStreams(
        solidStream, mixedStream, mixedMasks,
        gridBounds, sceneBounds, voxelResolution, treeDepth
    );
}

interface BlockBufferRegion {
    minBx: number;
    minBy: number;
    minBz: number;
}

/**
 * Build a sparse octree directly from a voxelization buffer. The block-index
 * arrays are converted to Morton codes and sorted in place, so the buffer is
 * consumed and must not be reused after this call.
 *
 * @param buffer - Filtered voxelization output.
 * @param sourceNbx - X block count used to linearize buffer indices.
 * @param sourceNby - Y block count used to linearize buffer indices.
 * @param sourceNbz - Z block count used to linearize buffer indices.
 * @param gridBounds - Output bounds, optionally cropped to `region`.
 * @param sceneBounds - Original scene bounds.
 * @param voxelResolution - Size of each voxel in world units.
 * @param region - Source block origin corresponding to `gridBounds`.
 * @returns Sparse octree structure.
 */
function buildSparseOctreeFromBuffer(
    buffer: BlockMaskBuffer,
    sourceNbx: number,
    sourceNby: number,
    sourceNbz: number,
    gridBounds: Bounds,
    sceneBounds: Bounds,
    voxelResolution: number,
    region: BlockBufferRegion = { minBx: 0, minBy: 0, minBz: 0 }
): SparseOctree {
    const solidStream = buffer.getSolidBlocks();
    const mixed = buffer.getMixedBlocks();
    const mixedStream = mixed.blockIdx;
    if (mixedStream.length > MAX_V1_MIXED_LEAVES) {
        throw new Error(
            `Sparse octree mixed-leaf count (${mixedStream.length}) exceeds the ` +
            `Laine-Karras 24-bit baseOffset limit (${MAX_V1_MIXED_LEAVES}). ` +
            'Use a coarser voxel resolution.'
        );
    }
    const sourceStride = sourceNbx * sourceNby;
    const totalBlocks = sourceStride * sourceNbz;
    if (!Number.isSafeInteger(totalBlocks)) {
        throw new Error(
            `Voxel source grid ${sourceNbx}x${sourceNby}x${sourceNbz} exceeds the safe-integer block-index limit. ` +
            'Use a coarser voxel resolution.'
        );
    }

    const convertToMorton = (stream: Float64Array): void => {
        for (let i = 0; i < stream.length; i++) {
            const blockIdx = stream[i];
            if (!Number.isSafeInteger(blockIdx) || blockIdx < 0 || blockIdx >= totalBlocks) {
                throw new Error(`Voxel block index ${blockIdx} is outside the ${sourceNbx}x${sourceNby}x${sourceNbz} grid`);
            }
            const bx = blockIdx % sourceNbx;
            const byBz = Math.floor(blockIdx / sourceNbx);
            const by = byBz % sourceNby;
            const bz = Math.floor(byBz / sourceNby);
            const x = bx - region.minBx;
            const y = by - region.minBy;
            const z = bz - region.minBz;
            if (x < 0 || y < 0 || z < 0) {
                throw new Error(`Voxel block index ${blockIdx} lies outside the cropped output region`);
            }
            stream[i] = xyzToMorton(x, y, z);
        }
    };

    convertToMorton(solidStream);
    convertToMorton(mixedStream);
    if (solidStream.length > 1) solidStream.sort();
    if (mixedStream.length > 1) sortKeyMaskPairs(mixedStream, mixed.masks, mixedStream.length);

    const treeDepth = calculateTreeDepth(gridBounds, voxelResolution);
    return buildSparseOctreeFromStreams(
        solidStream, mixedStream, mixed.masks,
        gridBounds, sceneBounds, voxelResolution, treeDepth
    );
}

/**
 * Build and flatten an octree from sorted leaf streams.
 *
 * @param solidStream - Solid leaf Morton codes.
 * @param mixedStream - Mixed leaf Morton codes.
 * @param mixedMasks - Interleaved masks paired with `mixedStream`.
 * @param gridBounds - Grid bounds aligned to block boundaries.
 * @param sceneBounds - Original scene bounds.
 * @param voxelResolution - Size of each voxel in world units.
 * @param treeDepth - Tree depth in block levels.
 * @returns Sparse octree structure.
 */
function buildSparseOctreeFromStreams(
    solidStream: Float64Array,
    mixedStream: Float64Array,
    mixedMasks: Uint32Array,
    gridBounds: Bounds,
    sceneBounds: Bounds,
    voxelResolution: number,
    treeDepth: number
): SparseOctree {
    const nSolid = solidStream.length;
    const nMixed = mixedStream.length;
    if (treeDepth > 17) {
        throw new Error(`Sparse octree depth ${treeDepth} exceeds the 17-level Morton encoding limit. Use a coarser voxel resolution.`);
    }
    if (nMixed > MAX_V1_MIXED_LEAVES) {
        throw new Error(
            `Sparse octree mixed-leaf count (${nMixed}) exceeds the ` +
            `Laine-Karras 24-bit baseOffset limit (${MAX_V1_MIXED_LEAVES}). ` +
            'Use a coarser voxel resolution.'
        );
    }

    // --- Phase 2: Build tree bottom-up level by level using linear scan ---
    // Level 0 lives in `solidStream` + `mixedStream` (dual streams, sorted).
    // We build level 1 by a two-pointer merge of these streams; subsequent
    // levels (2, 3, ...) use exactly-sized typed arrays.
    //
    // `interiorLevels[]` holds INTERIOR levels only (no level 0).
    // `interiorLevels[0]` is the first interior level (= original "level 1");
    // `interiorLevels[length-1]` is the root. Phase 3 special-cases level 0
    // access via a wave-entry sentinel (`li === -1`).

    const bar = logger.bar('Building tree', 10);
    let octreeStep = 0;

    const interiorLevels: LevelData[] = [];

    // === Build level 1 from dual-stream level 0 ===
    // Two-pointer merge: at each step, the smaller-morton stream is the next
    // child to process. Group children with the same `floor(morton/8)` parent.
    // childMask records which octants are present; allSolid tracks whether we
    // can collapse this parent into a SOLID leaf at level 1.
    let parentCount = 0;
    {
        let sI = 0;
        let mI = 0;
        while (sI < nSolid || mI < nMixed) {
            const sM0 = sI < nSolid ? solidStream[sI] : Number.POSITIVE_INFINITY;
            const mM0 = mI < nMixed ? mixedStream[mI] : Number.POSITIVE_INFINITY;
            const parentMorton = Math.floor(Math.min(sM0, mM0) / 8);
            parentCount++;
            while (sI < nSolid && Math.floor(solidStream[sI] / 8) === parentMorton) sI++;
            while (mI < nMixed && Math.floor(mixedStream[mI] / 8) === parentMorton) mI++;
        }
    }

    let curMortons = new Float64Array(parentCount);
    let curTypes = new Uint8Array(parentCount);
    let curChildMasks = new Uint8Array(parentCount);

    {
        let sI = 0;
        let mI = 0;
        let writeIdx = 0;
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

            curMortons[writeIdx] = parentMorton;
            if (allSolid && childCount === 8) {
                curTypes[writeIdx] = BlockType.Solid;
            } else {
                curTypes[writeIdx] = BlockType.Mixed;
                curChildMasks[writeIdx] = childMask;
            }
            writeIdx++;
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
        // Empty input: nothing to push. interiorLevels stays empty.
        actualDepth = 1;
    } else if (curMortons.length === 1 && curMortons[0] === 0) {
        // Level 1 IS the root.
        actualDepth = 1;
        interiorLevels.push({
            mortons: curMortons,
            types: curTypes,
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
            interiorLevels.push({
                mortons: curMortons,
                types: curTypes,
                childMasks: curChildMasks
            });

            // Build next level using linear scan on sorted data.
            const n = curMortons.length;
            let nextCount = 0;
            for (let i = 0; i < n;) {
                const parentMorton = Math.floor(curMortons[i] / 8);
                nextCount++;
                do {
                    i++;
                } while (i < n && Math.floor(curMortons[i] / 8) === parentMorton);
            }
            const nextMortons = new Float64Array(nextCount);
            const nextTypes = new Uint8Array(nextCount);
            const nextChildMasks = new Uint8Array(nextCount);

            let i = 0;
            let writeIdx = 0;
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

                nextMortons[writeIdx] = parentMorton;
                if (allSolid && childCount === 8) {
                    nextTypes[writeIdx] = BlockType.Solid;
                } else {
                    nextTypes[writeIdx] = BlockType.Mixed;
                    nextChildMasks[writeIdx] = childMask;
                }
                writeIdx++;
            }

            curMortons = nextMortons;
            curTypes = nextTypes;
            curChildMasks = nextChildMasks;

            // Each iteration consumes n >= 1 entries and produces ceil(n/8) >= 1
            // parents, so curMortons.length stays >= 1 here; we only need to
            // check for convergence to a single root at morton 0.
            if (curMortons.length === 1 && curMortons[0] === 0) {
                actualDepth = level + 1;
                break;
            }
        }

        // Save root. By the invariant above, curMortons.length >= 1 always.
        interiorLevels.push({
            mortons: curMortons,
            types: curTypes,
            childMasks: curChildMasks
        });
    }

    while (octreeStep < 9) {
        bar.tick();
        octreeStep++;
    }

    // --- Phase 3: Flatten tree to Laine-Karras format ---
    const result = flattenTreeFromLevels(
        interiorLevels, solidStream, mixedStream, mixedMasks, nSolid, nMixed,
        gridBounds, sceneBounds, voxelResolution, actualDepth
    );

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
 * `interiorLevels[i]` (for `i >= 0`) holds the i-th INTERIOR level (= original
 * "level i+1"), built bottom-up. `interiorLevels[length-1]` is the root.
 *
 * @param interiorLevels - Interior level data (index 0 = first interior level
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
    interiorLevels: LevelData[],
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
    if (interiorLevels.length === 0) {
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

    const rootLevel = interiorLevels[interiorLevels.length - 1];

    // Upper bound on total nodes (not all may be reachable if solids collapsed)
    let maxNodes = nSolid + nMixed;
    for (let l = 0; l < interiorLevels.length; l++) {
        maxNodes += interiorLevels[l].mortons.length;
    }

    const nodes = new Uint32Array(Math.min(maxNodes, MAX_24BIT_OFFSET + 1));
    // Pre-size leafData to its upper bound (2 entries per mixed leaf).
    const leafData = new Uint32Array(nMixed * 2);
    let leafDataLen = 0;
    let numInteriorNodes = 0;
    let numMixedLeaves = 0;
    let emitPos = 0;

    // BFS waves use typed parallel arrays. `li === -1` identifies leaf entries.
    const rootLi = interiorLevels.length - 1;
    let wave = createInteriorWave(rootLevel.mortons.length);
    for (let i = 0; i < rootLevel.mortons.length; i++) {
        pushInteriorWave(wave, 0, rootLi, i);
    }

    while (wave.length > 0) {
        const interiors = createInteriorWave(wave.length);

        // --- Emit all nodes in this wave ---
        for (let w = 0; w < wave.length; w++) {
            const li = wave.li[w];
            const ii = wave.ii[w];

            if (li === -1) {
                // Level-0 leaf via dual-stream encoding.
                if (ii < nMixed) {
                    // Mixed leaf
                    const leafDataIndex = leafDataLen >> 1;
                    if (leafDataIndex > MAX_24BIT_OFFSET) {
                        throw new Error(
                            `Sparse octree mixed-leaf count (${leafDataIndex + 1}) exceeds the ` +
                            `Laine-Karras 24-bit baseOffset limit (${MAX_24BIT_OFFSET + 1}). ` +
                            'Use a coarser voxel resolution.'
                        );
                    }
                    leafData[leafDataLen++] = mixedMasks[ii * 2];
                    leafData[leafDataLen++] = mixedMasks[ii * 2 + 1];
                    nodes[emitPos] = leafDataIndex;
                    numMixedLeaves++;
                } else {
                    // Solid leaf
                    nodes[emitPos] = SOLID_LEAF_MARKER;
                }
                emitPos++;
                continue;
            }

            const level = interiorLevels[li];
            const type = level.types[ii];

            // Level-0 leaves are emitted via the `li === -1` sentinel above;
            // any solid interior-level node here is a collapsed solid leaf.
            const isLeaf = type === BlockType.Solid;

            if (isLeaf) {
                nodes[emitPos] = SOLID_LEAF_MARKER;
            } else {
                // Interior node — record position for backfill after wave
                pushInteriorWave(interiors, emitPos, li, ii);
                numInteriorNodes++;
                nodes[emitPos] = 0;
            }
            emitPos++;
        }

        // --- Build next wave from children of interior nodes ---
        let nextLength = 0;
        for (let j = 0; j < interiors.length; j++) {
            nextLength += popcount(interiorLevels[interiors.li[j]].childMasks[interiors.ii[j]]);
        }
        if (emitPos + nextLength > MAX_24BIT_OFFSET + 1) {
            throw new Error(
                `Sparse octree node count (${emitPos + nextLength}) exceeds the ` +
                `Laine-Karras 24-bit baseOffset limit (${MAX_24BIT_OFFSET + 1}). ` +
                'Use a coarser voxel resolution.'
            );
        }
        const nextWave = createInteriorWave(nextLength);
        let nextChildStart = emitPos;

        for (let j = 0; j < interiors.length; j++) {
            const myLi = interiors.li[j];
            const myIi = interiors.ii[j];
            const childMask = interiorLevels[myLi].childMasks[myIi];
            const childCount = popcount(childMask);

            nodes[interiors.pos[j]] = ((childMask & 0xFF) << 24) | nextChildStart;

            const myMorton = interiorLevels[myLi].mortons[myIi];
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
                        pushInteriorWave(nextWave, 0, -1, nMixed + sIdx);
                        sIdx++;
                    } else {
                        pushInteriorWave(nextWave, 0, -1, mIdx);
                        mIdx++;
                    }
                }
            } else {
                // Children are at an interior level → single-array binary search.
                const childLi = myLi - 1;
                const childLevel = interiorLevels[childLi];
                const childMortons = childLevel.mortons;

                let lo = 0;
                let hi = childMortons.length;
                while (lo < hi) {
                    const mid = (lo + hi) >> 1;
                    if (childMortons[mid] < childMortonBase) lo = mid + 1;
                    else hi = mid;
                }

                while (lo < childMortons.length && childMortons[lo] < childMortonEnd) {
                    pushInteriorWave(nextWave, 0, childLi, lo);
                    lo++;
                }
            }

            nextChildStart += childCount;
        }

        wave = nextWave;
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
    buildSparseOctreeFromBuffer,
    MAX_V1_MIXED_LEAVES,
    SOLID_LEAF_MARKER
};

export type { SparseOctree };
