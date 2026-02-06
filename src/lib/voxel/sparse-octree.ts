import { Vec3 } from 'playcanvas';

// ============================================================================
// Constants
// ============================================================================

/** All 64 bits set (as unsigned 32-bit) */
const SOLID_MASK = 0xFFFFFFFF >>> 0;

/** Solid leaf node marker: high byte = 0x00 (no children), lower bits all zero */
const SOLID_LEAF_MARKER = 0x00000000 >>> 0;

/** Mixed leaf node marker: high byte = 0x00, bit 23 set, lower 23 bits = leafData index */
const MIXED_LEAF_MARKER = 0x00800000 >>> 0;

// ============================================================================
// Morton Code Functions
// ============================================================================

/**
 * Encode block coordinates to Morton code (17 bits per axis = 51 bits total).
 * Supports up to 131,072 blocks per axis.
 *
 * @param x - Block X coordinate
 * @param y - Block Y coordinate
 * @param z - Block Z coordinate
 * @returns Morton code with interleaved bits: ...z2y2x2 z1y1x1 z0y0x0
 */
function xyzToMorton(x: number, y: number, z: number): number {
    // Use lookup table approach for better performance
    // Split into low and high parts to avoid 32-bit overflow issues
    let result = 0;
    for (let i = 0; i < 17; i++) {
        const xBit = (x >>> i) & 1;
        const yBit = (y >>> i) & 1;
        const zBit = (z >>> i) & 1;
        // Position: i*3 for x, i*3+1 for y, i*3+2 for z
        const shift = i * 3;
        result += xBit * Math.pow(2, shift);
        result += yBit * Math.pow(2, shift + 1);
        result += zBit * Math.pow(2, shift + 2);
    }
    return result;
}

/**
 * Decode Morton code to block coordinates.
 *
 * @param m - Morton code
 * @returns Tuple of [x, y, z] block coordinates
 */
function mortonToXYZ(m: number): [number, number, number] {
    let x = 0, y = 0, z = 0;
    for (let i = 0; i < 17; i++) {
        const shift = i * 3;
        x |= (Math.floor(m / Math.pow(2, shift)) & 1) << i;
        y |= (Math.floor(m / Math.pow(2, shift + 1)) & 1) << i;
        z |= (Math.floor(m / Math.pow(2, shift + 2)) & 1) << i;
    }
    return [x, y, z];
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Count the number of set bits in a 32-bit integer.
 *
 * @param n - 32-bit integer
 * @returns Number of bits set to 1
 */
function popcount(n: number): number {
    n = n >>> 0; // Ensure unsigned
    n = n - ((n >>> 1) & 0x55555555);
    n = (n & 0x33333333) + ((n >>> 2) & 0x33333333);
    return (((n + (n >>> 4)) & 0x0F0F0F0F) * 0x01010101) >>> 24;
}

/**
 * Check if a voxel mask represents a solid block (all 64 bits set).
 *
 * @param lo - Lower 32 bits of mask
 * @param hi - Upper 32 bits of mask
 * @returns True if all 64 voxels are solid
 */
function isSolid(lo: number, hi: number): boolean {
    return (lo >>> 0) === SOLID_MASK && (hi >>> 0) === SOLID_MASK;
}

/**
 * Check if a voxel mask represents an empty block (no bits set).
 *
 * @param lo - Lower 32 bits of mask
 * @param hi - Upper 32 bits of mask
 * @returns True if all 64 voxels are empty
 */
function isEmpty(lo: number, hi: number): boolean {
    return lo === 0 && hi === 0;
}

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
// Block Accumulator
// ============================================================================

/**
 * Accumulator for streaming voxelization results.
 * Stores blocks using Morton codes for efficient octree construction.
 */
class BlockAccumulator {
    /** Morton codes for mixed blocks */
    private _mixedMorton: number[] = [];

    /** Interleaved voxel masks for mixed blocks: [lo0, hi0, lo1, hi1, ...] */
    private _mixedMasks: number[] = [];

    /** Morton codes for solid blocks (mask is implicitly all 1s) */
    private _solidMorton: number[] = [];

    /**
     * Add a non-empty block to the accumulator.
     * Automatically classifies as solid or mixed based on mask values.
     *
     * @param morton - Morton code encoding block position
     * @param lo - Lower 32 bits of voxel mask
     * @param hi - Upper 32 bits of voxel mask
     */
    addBlock(morton: number, lo: number, hi: number): void {
        if (isEmpty(lo, hi)) {
            // Empty blocks are discarded
            return;
        }

        if (isSolid(lo, hi)) {
            // Solid blocks only need Morton code
            this._solidMorton.push(morton);
        } else {
            // Mixed blocks need Morton code + mask
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
     * Clear all accumulated blocks.
     */
    clear(): void {
        this._mixedMorton.length = 0;
        this._mixedMasks.length = 0;
        this._solidMorton.length = 0;
    }
}

// ============================================================================
// Sparse Octree Types
// ============================================================================

/**
 * Bounds specification with min/max Vec3.
 */
interface Bounds {
    min: Vec3;
    max: Vec3;
}

/**
 * Sparse voxel octree using Laine-Karras node format.
 */
interface SparseOctree {
    /** Grid bounds aligned to 4x4x4 block boundaries */
    gridBounds: Bounds;

    /** Original Gaussian scene bounds */
    gaussianBounds: Bounds;

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

/** Intermediate node during construction */
interface BuildNode {
    /** Node type */
    type: BlockType;

    /** Morton code of this node (at its level) */
    morton: number;

    /** For mixed leaves: index into mask data */
    maskIndex?: number;

    /** For interior nodes: child nodes (sparse, indexed by octant 0-7) */
    children?: (BuildNode | null)[];
}

// ============================================================================
// Octree Construction
// ============================================================================

/**
 * Build a sparse octree from accumulated voxelization blocks.
 *
 * @param accumulator - BlockAccumulator containing voxelized blocks
 * @param gridBounds - Grid bounds aligned to block boundaries
 * @param gaussianBounds - Original scene bounds
 * @param voxelResolution - Size of each voxel in world units
 * @returns Sparse octree structure
 */
function buildSparseOctree(
    accumulator: BlockAccumulator,
    gridBounds: Bounds,
    gaussianBounds: Bounds,
    voxelResolution: number
): SparseOctree {
    const mixed = accumulator.getMixedBlocks();
    const solid = accumulator.getSolidBlocks();

    // Combine all blocks into sorted array with type information
    interface BlockEntry {
        morton: number;
        type: BlockType;
        maskIndex: number; // Index into mixed.masks (only valid for mixed blocks)
    }

    const blocks: BlockEntry[] = [];

    // Add mixed blocks
    for (let i = 0; i < mixed.morton.length; i++) {
        blocks.push({
            morton: mixed.morton[i],
            type: BlockType.Mixed,
            maskIndex: i
        });
    }

    // Add solid blocks
    for (let i = 0; i < solid.length; i++) {
        blocks.push({
            morton: solid[i],
            type: BlockType.Solid,
            maskIndex: -1
        });
    }

    // Sort by Morton code
    blocks.sort((a, b) => a.morton - b.morton);

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

    // Build tree bottom-up
    // Start with leaf nodes, then merge upward level by level

    // Level 0 = leaf blocks (4x4x4 voxels each)
    // Each level up groups 8 children into 1 parent

    // Map from Morton code to node at each level
    let currentLevel = new Map<number, BuildNode>();

    // Initialize leaf level from blocks
    for (const block of blocks) {
        currentLevel.set(block.morton, {
            type: block.type,
            morton: block.morton,
            maskIndex: block.type === BlockType.Mixed ? block.maskIndex : undefined
        });
    }

    // Build up level by level
    for (let level = 0; level < treeDepth; level++) {
        const nextLevel = new Map<number, BuildNode>();

        // Group nodes by parent Morton code
        const parentGroups = new Map<number, BuildNode[]>();

        for (const [morton, node] of currentLevel) {
            const parentMorton = Math.floor(morton / 8); // morton >>> 3
            if (!parentGroups.has(parentMorton)) {
                parentGroups.set(parentMorton, []);
            }
            parentGroups.get(parentMorton)!.push(node);
        }

        // Create parent nodes
        for (const [parentMorton, children] of parentGroups) {
            // Check if all 8 children are solid
            let allSolid = true;
            let childMask = 0;
            const childNodes: (BuildNode | null)[] = [null, null, null, null, null, null, null, null];

            for (const child of children) {
                const octant = child.morton % 8; // child.morton & 7
                childMask |= (1 << octant);
                childNodes[octant] = child;

                if (child.type !== BlockType.Solid) {
                    allSolid = false;
                }
            }

            // If all 8 children exist and are solid, collapse to solid parent
            if (allSolid && children.length === 8) {
                nextLevel.set(parentMorton, {
                    type: BlockType.Solid,
                    morton: parentMorton
                });
            } else {
                // Create interior node
                nextLevel.set(parentMorton, {
                    type: BlockType.Mixed, // Interior nodes are treated as "mixed" (have children)
                    morton: parentMorton,
                    children: childNodes
                });
            }
        }

        currentLevel = nextLevel;

        // Break when the tree is empty or has converged to a single root at Morton 0.
        // We must NOT break early if the single remaining node has a non-zero Morton,
        // because the reader reconstructs Morton codes starting from root Morton 0.
        if (currentLevel.size === 0 ||
            (currentLevel.size === 1 && currentLevel.has(0))) {
            break;
        }
    }

    // Flatten tree to arrays
    return flattenTree(currentLevel, mixed.masks, gridBounds, gaussianBounds, voxelResolution, treeDepth);
}

/**
 * Flatten the constructed tree into Laine-Karras format arrays.
 */
function flattenTree(
    rootLevel: Map<number, BuildNode>,
    mixedMasks: number[],
    gridBounds: Bounds,
    gaussianBounds: Bounds,
    voxelResolution: number,
    treeDepth: number
): SparseOctree {
    // Collect all nodes in breadth-first order
    const nodeList: BuildNode[] = [];
    const leafDataList: number[] = [];

    // Get root node
    const rootNodes = Array.from(rootLevel.values());
    if (rootNodes.length === 0) {
        // Empty tree
        return {
            gridBounds,
            gaussianBounds,
            voxelResolution,
            leafSize: 4,
            treeDepth,
            numInteriorNodes: 0,
            numMixedLeaves: 0,
            nodes: new Uint32Array(0),
            leafData: new Uint32Array(0)
        };
    }

    // BFS to collect all nodes and assign indices
    const queue: BuildNode[] = [...rootNodes];
    const nodeIndices = new Map<BuildNode, number>();

    while (queue.length > 0) {
        const node = queue.shift()!;
        nodeIndices.set(node, nodeList.length);
        nodeList.push(node);

        // Queue children if this is an interior node
        if (node.children) {
            for (let octant = 0; octant < 8; octant++) {
                const child = node.children[octant];
                if (child && !nodeIndices.has(child)) {
                    queue.push(child);
                }
            }
        }
    }

    // Now encode nodes to Laine-Karras format
    const nodes = new Uint32Array(nodeList.length);
    let numInteriorNodes = 0;
    let numMixedLeaves = 0;

    // First pass: count children and assign child base offsets
    // Children of each interior node are allocated contiguously
    let nextChildOffset = rootNodes.length; // Children start after root(s)

    for (let i = 0; i < nodeList.length; i++) {
        const node = nodeList[i];

        if (node.children) {
            // Interior node
            numInteriorNodes++;

            // Count existing children and build child mask
            let childMask = 0;
            let childCount = 0;
            for (let octant = 0; octant < 8; octant++) {
                if (node.children[octant]) {
                    childMask |= (1 << octant);
                    childCount++;
                }
            }

            // Encode: mask in high byte, offset in low 24 bits
            const baseOffset = nextChildOffset;
            nodes[i] = ((childMask & 0xFF) << 24) | (baseOffset & 0x00FFFFFF);

            nextChildOffset += childCount;
        } else if (node.type === BlockType.Solid) {
            // Solid leaf
            nodes[i] = SOLID_LEAF_MARKER;
        } else if (node.type === BlockType.Mixed && node.maskIndex !== undefined) {
            // Mixed leaf - store index into leafData
            const leafDataIndex = leafDataList.length / 2;
            leafDataList.push(mixedMasks[node.maskIndex * 2]);     // lo
            leafDataList.push(mixedMasks[node.maskIndex * 2 + 1]); // hi
            nodes[i] = MIXED_LEAF_MARKER | (leafDataIndex & 0x007FFFFF);
            numMixedLeaves++;
        }
    }

    return {
        gridBounds,
        gaussianBounds,
        voxelResolution,
        leafSize: 4,
        treeDepth,
        numInteriorNodes,
        numMixedLeaves,
        nodes,
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
    // Morton code functions
    xyzToMorton,
    mortonToXYZ,

    // Utility functions
    popcount,
    isSolid,
    isEmpty,
    getChildOffset,

    // Accumulator
    BlockAccumulator,

    // Octree construction
    buildSparseOctree,
    alignGridBounds,

    // Constants
    SOLID_LEAF_MARKER,
    MIXED_LEAF_MARKER
};

export type { SparseOctree, Bounds };
