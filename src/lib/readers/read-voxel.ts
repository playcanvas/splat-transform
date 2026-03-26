import { basename, dirname, join } from 'pathe';

import { Column, DataTable } from '../data-table/data-table';
import { ReadFileSystem, readFile } from '../io/read';
import { getChildOffset, SOLID_LEAF_MARKER } from '../voxel/sparse-octree';

/** SH coefficient for color conversion */
const C0 = 0.28209479177387814;

/**
 * Metadata from a .voxel.json file.
 */
interface VoxelMetadata {
    version: string;
    gridBounds: { min: number[]; max: number[] };
    sceneBounds: { min: number[]; max: number[] };
    voxelResolution: number;
    leafSize: number;
    treeDepth: number;
    numInteriorNodes: number;
    numMixedLeaves: number;
    nodeCount: number;
    leafDataCount: number;
}

/**
 * Collected leaf blocks stored as parallel flat arrays to minimize GC pressure.
 */
interface LeafArrays {
    /** Morton codes of each leaf block */
    morton: number[];
    /** Whether each leaf block is solid (true) or mixed (false) */
    isSolid: Uint8Array;
    /** Number of leaf blocks */
    count: number;
}

/**
 * Iteratively expand a collapsed solid node into leaf-level solid blocks.
 * A collapsed solid at depth d with Morton m represents 8^(treeDepth-d) leaf blocks.
 * Uses an explicit stack instead of recursion for better performance.
 *
 * @param morton - Morton code of the solid node
 * @param depth - Current depth in the tree
 * @param treeDepth - Target leaf depth
 * @param outMorton - Output array to push leaf Morton codes into
 * @param outSolid - Output Uint8Array to mark leaves as solid
 * @param outCount - Current count of leaves (used as write index)
 * @param stack - Reusable stack array (pairs of [morton, depth])
 * @returns Updated leaf count after expansion
 */
const expandSolid = (
    morton: number,
    depth: number,
    treeDepth: number,
    outMorton: number[],
    outSolid: Uint8Array,
    outCount: number,
    stack: number[]
): number => {
    stack.length = 0;
    stack.push(morton, depth);

    while (stack.length > 0) {
        const d = stack.pop()!;
        const m = stack.pop()!;

        if (d === treeDepth) {
            outMorton.push(m);
            outSolid[outCount++] = 1;
        } else {
            // Push children in reverse order so octant 0 is processed first
            for (let octant = 7; octant >= 0; octant--) {
                stack.push(m * 8 + octant, d + 1);
            }
        }
    }

    return outCount;
};

/**
 * Traverse the octree and collect all leaf nodes with their Morton codes.
 * Collapsed solid parents are expanded back to leaf-level blocks.
 * Uses parallel flat arrays and index-based BFS for performance.
 *
 * @param nodes - Laine-Karras nodes array (Uint32)
 * @param _leafData - Leaf voxel masks (reserved for future use)
 * @param treeDepth - Maximum tree depth (leaf level)
 * @returns Parallel arrays of morton codes and solid flags
 */
const collectLeafBlocks = (
    nodes: Uint32Array,
    _leafData: Uint32Array,
    treeDepth: number
): LeafArrays => {
    // Find root nodes: nodes that are never referenced as children.
    const isChild = new Set<number>();
    for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i] >>> 0;
        if (node === SOLID_LEAF_MARKER) continue;
        const highByte = (node >> 24) & 0xFF;
        if (highByte !== 0x00) {
            const childMask = highByte;
            const baseOffset = node & 0x00FFFFFF;
            for (let octant = 0; octant < 8; octant++) {
                if (childMask & (1 << octant)) {
                    const offset = getChildOffset(childMask, octant);
                    isChild.add(baseOffset + offset);
                }
            }
        }
    }

    // BFS using parallel arrays and index-based dequeue (O(1) per dequeue)
    const qNodeIdx: number[] = [];
    const qMorton: number[] = [];
    const qDepth: number[] = [];

    let rootMorton = 0;
    for (let i = 0; i < nodes.length; i++) {
        if (!isChild.has(i)) {
            qNodeIdx.push(i);
            qMorton.push(rootMorton);
            qDepth.push(0);
            rootMorton++;
        }
    }

    // Output parallel arrays
    const leafMorton: number[] = [];
    // Pre-allocate a generous Uint8Array; will be trimmed at the end
    let leafSolidCapacity = nodes.length;
    let leafSolid = new Uint8Array(leafSolidCapacity);
    let leafCount = 0;

    // Reusable stack for expandSolid
    const expandStack: number[] = [];

    const ensureSolidCapacity = (needed: number) => {
        if (needed > leafSolidCapacity) {
            leafSolidCapacity = Math.max(leafSolidCapacity * 2, needed);
            const newArr = new Uint8Array(leafSolidCapacity);
            newArr.set(leafSolid);
            leafSolid = newArr;
        }
    };

    let head = 0;
    while (head < qNodeIdx.length) {
        const nodeIdx = qNodeIdx[head];
        const morton = qMorton[head];
        const depth = qDepth[head];
        head++;

        const node = nodes[nodeIdx] >>> 0;

        if (node === SOLID_LEAF_MARKER) {
            // Solid leaf - may be a collapsed parent if depth < treeDepth
            const levelsToExpand = treeDepth - depth;
            if (levelsToExpand === 0) {
                // Already at leaf level, no expansion needed
                leafMorton.push(morton);
                ensureSolidCapacity(leafCount + 1);
                leafSolid[leafCount++] = 1;
            } else {
                const expansionSize = 8 ** levelsToExpand;
                ensureSolidCapacity(leafCount + expansionSize);
                leafCount = expandSolid(morton, depth, treeDepth, leafMorton, leafSolid, leafCount, expandStack);
            }
        } else {
            const highByte = (node >> 24) & 0xFF;
            if (highByte === 0x00) {
                // Mixed leaf
                leafMorton.push(morton);
                ensureSolidCapacity(leafCount + 1);
                leafSolid[leafCount++] = 0;
            } else {
                // Interior node - queue children
                const childMask = highByte;
                const baseOffset = node & 0x00FFFFFF;
                for (let octant = 0; octant < 8; octant++) {
                    if (childMask & (1 << octant)) {
                        const offset = getChildOffset(childMask, octant);
                        qNodeIdx.push(baseOffset + offset);
                        qMorton.push(morton * 8 + octant);
                        qDepth.push(depth + 1);
                    }
                }
            }
        }
    }

    return {
        morton: leafMorton,
        isSolid: leafSolid.subarray(0, leafCount),
        count: leafCount
    };
};

/**
 * Read a .voxel.json file and convert to DataTable (finest/leaf LOD).
 *
 * Loads the voxel octree from .voxel.json + .voxel.bin, traverses to the leaf level,
 * and outputs a DataTable in the same Gaussian splat format as voxel-octree-node.mjs
 * at the leaf level. Users can then save to PLY, CSV, or any other format.
 *
 * @param fileSystem - File system for reading files
 * @param filename - Path to .voxel.json (the .voxel.bin must exist alongside it)
 * @returns DataTable with voxel block centers as Gaussian splats
 */
const readVoxel = async (
    fileSystem: ReadFileSystem,
    filename: string
): Promise<DataTable> => {
    const baseDir = dirname(filename);
    const load = (name: string) => readFile(fileSystem, baseDir ? join(baseDir, name) : name);

    // Load and parse JSON metadata
    const jsonBytes = await load(basename(filename));
    const metadata = JSON.parse(new TextDecoder().decode(jsonBytes)) as VoxelMetadata;

    if (metadata.version !== '1.0' && metadata.version !== '1.1') {
        throw new Error(`Unsupported voxel format version: ${metadata.version}`);
    }

    // Load binary data
    const binFilename = basename(filename).replace(/\.voxel\.json$/i, '.voxel.bin');
    let binBytes: Uint8Array;
    try {
        binBytes = await load(binFilename);
    } catch (e) {
        throw new Error(
            `Failed to load voxel binary file '${binFilename}'. ` +
            `Ensure ${binFilename} exists alongside ${basename(filename)}.`
        );
    }

    const nodeCount = metadata.nodeCount;
    const leafDataCount = metadata.leafDataCount;
    const expectedSize = (nodeCount + leafDataCount) * 4;
    if (binBytes.length < expectedSize) {
        throw new Error(
            `Voxel binary file truncated: expected ${expectedSize} bytes, got ${binBytes.length}`
        );
    }

    const nodes = new Uint32Array(binBytes.buffer, binBytes.byteOffset, nodeCount);
    const leafData = new Uint32Array(
        binBytes.buffer,
        binBytes.byteOffset + nodeCount * 4,
        leafDataCount
    );

    const leaves = collectLeafBlocks(nodes, leafData, metadata.treeDepth);

    if (leaves.count === 0) {
        return new DataTable([
            new Column('x', new Float32Array(0)),
            new Column('y', new Float32Array(0)),
            new Column('z', new Float32Array(0)),
            new Column('scale_0', new Float32Array(0)),
            new Column('scale_1', new Float32Array(0)),
            new Column('scale_2', new Float32Array(0)),
            new Column('rot_0', new Float32Array(0)),
            new Column('rot_1', new Float32Array(0)),
            new Column('rot_2', new Float32Array(0)),
            new Column('rot_3', new Float32Array(0)),
            new Column('f_dc_0', new Float32Array(0)),
            new Column('f_dc_1', new Float32Array(0)),
            new Column('f_dc_2', new Float32Array(0)),
            new Column('opacity', new Float32Array(0))
        ]);
    }

    const gridMin = metadata.gridBounds.min;
    const voxelResolution = metadata.voxelResolution;
    const blockSize = 4 * voxelResolution;
    const splatScale = Math.log(blockSize * 0.4);

    const numBlocks = leaves.count;
    const leavesMorton = leaves.morton;
    const leavesSolid = leaves.isSolid;

    const xArr = new Float32Array(numBlocks);
    const yArr = new Float32Array(numBlocks);
    const zArr = new Float32Array(numBlocks);
    const scale0 = new Float32Array(numBlocks);
    const scale1 = new Float32Array(numBlocks);
    const scale2 = new Float32Array(numBlocks);
    const rot0 = new Float32Array(numBlocks);
    const rot1 = new Float32Array(numBlocks);
    const rot2 = new Float32Array(numBlocks);
    const rot3 = new Float32Array(numBlocks);
    const fdc0 = new Float32Array(numBlocks);
    const fdc1 = new Float32Array(numBlocks);
    const fdc2 = new Float32Array(numBlocks);
    const opacityArr = new Float32Array(numBlocks);

    // Pre-compute constant SH values for solid blocks
    const solidR = (0.9 - 0.5) / C0;
    const solidG = (0.1 - 0.5) / C0;
    const solidB = (0.1 - 0.5) / C0;

    for (let i = 0; i < numBlocks; i++) {
        const morton = leavesMorton[i];

        // Inline Morton decode: extract x, y, z from interleaved 3-bit groups
        // Avoids function call overhead and tuple allocation
        let bx = 0, by = 0, bz = 0;
        let m = morton;
        let bit = 1;
        while (m > 0) {
            const triplet = m % 8;
            if (triplet & 1) bx |= bit;
            if (triplet & 2) by |= bit;
            if (triplet & 4) bz |= bit;
            bit <<= 1;
            m = Math.trunc(m / 8);
        }

        xArr[i] = gridMin[0] + (bx + 0.5) * blockSize;
        yArr[i] = gridMin[1] + (by + 0.5) * blockSize;
        zArr[i] = gridMin[2] + (bz + 0.5) * blockSize;

        scale0[i] = splatScale;
        scale1[i] = splatScale;
        scale2[i] = splatScale;

        rot0[i] = 1.0;
        // rot1, rot2, rot3 default to 0.0 from Float32Array initialization

        if (leavesSolid[i]) {
            fdc0[i] = solidR;
            fdc1[i] = solidG;
            fdc2[i] = solidB;
        } else {
            const gray = 0.3 + ((morton * 0.618033988749895) % 1.0) * 0.5;
            fdc0[i] = (gray - 0.5) / C0;
            fdc1[i] = (gray - 0.5) / C0;
            fdc2[i] = (gray - 0.5) / C0;
        }

        opacityArr[i] = 5.0;
    }

    return new DataTable([
        new Column('x', xArr),
        new Column('y', yArr),
        new Column('z', zArr),
        new Column('scale_0', scale0),
        new Column('scale_1', scale1),
        new Column('scale_2', scale2),
        new Column('rot_0', rot0),
        new Column('rot_1', rot1),
        new Column('rot_2', rot2),
        new Column('rot_3', rot3),
        new Column('f_dc_0', fdc0),
        new Column('f_dc_1', fdc1),
        new Column('f_dc_2', fdc2),
        new Column('opacity', opacityArr)
    ]);
};

export { readVoxel };
