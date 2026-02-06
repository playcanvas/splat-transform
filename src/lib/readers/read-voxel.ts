import { basename, dirname, join } from 'pathe';

import { Column, DataTable } from '../data-table/data-table';
import { ReadFileSystem, readFile } from '../io/read';
import { getChildOffset, mortonToXYZ } from '../voxel/index';

/** SH coefficient for color conversion */
const C0 = 0.28209479177387814;

/**
 * Metadata from a .voxel.json file.
 */
interface VoxelMetadata {
    version: string;
    gridBounds: { min: number[]; max: number[] };
    gaussianBounds: { min: number[]; max: number[] };
    voxelResolution: number;
    leafSize: number;
    treeDepth: number;
    numInteriorNodes: number;
    numMixedLeaves: number;
    nodeCount: number;
    leafDataCount: number;
}

interface LeafBlock {
    morton: number;
    isSolid: boolean;
    leafMorton: number;
}

/**
 * Recursively expand a collapsed solid node into leaf-level solid blocks.
 * A collapsed solid at depth d with Morton m represents 8^(treeDepth-d) leaf blocks.
 *
 * @param morton - Morton code of the solid node
 * @param depth - Current depth in the tree
 * @param treeDepth - Target leaf depth
 * @param leaves - Output array to push leaf blocks into
 */
const expandSolid = (
    morton: number,
    depth: number,
    treeDepth: number,
    leaves: LeafBlock[]
): void => {
    if (depth === treeDepth) {
        // At leaf level - emit as a genuine leaf block
        leaves.push({ morton, isSolid: true, leafMorton: morton });
        return;
    }
    // Not at leaf level - expand into 8 children
    for (let octant = 0; octant < 8; octant++) {
        expandSolid(morton * 8 + octant, depth + 1, treeDepth, leaves);
    }
};

/**
 * Traverse the octree and collect all leaf nodes with their Morton codes.
 * Collapsed solid parents are expanded back to leaf-level blocks.
 *
 * @param nodes - Laine-Karras nodes array (Uint32)
 * @param _leafData - Leaf voxel masks (reserved for future use)
 * @param treeDepth - Maximum tree depth (leaf level)
 * @returns Array of leaf blocks with morton, isSolid, leafMorton
 */
const collectLeafBlocks = (
    nodes: Uint32Array,
    _leafData: Uint32Array,
    treeDepth: number
): LeafBlock[] => {
    const leaves: LeafBlock[] = [];

    // Find root nodes: nodes that are never referenced as children.
    // Interior nodes have highByte > 0 (childMask 0x01-0xFF).
    // Leaf nodes have highByte === 0 (solid: 0x00000000, mixed: 0x00800000 | index).
    const isChild = new Set<number>();
    for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i] >>> 0;
        const highByte = (node >> 24) & 0xFF;
        if (highByte !== 0x00) {
            // Interior node
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

    // BFS from roots with Morton code and depth tracking
    const queue: { nodeIdx: number; morton: number; depth: number }[] = [];
    let rootMorton = 0;
    for (let i = 0; i < nodes.length; i++) {
        if (!isChild.has(i)) {
            queue.push({ nodeIdx: i, morton: rootMorton, depth: 0 });
            rootMorton++;
        }
    }

    while (queue.length > 0) {
        const { nodeIdx, morton, depth } = queue.shift()!;
        const node = nodes[nodeIdx] >>> 0;
        const highByte = (node >> 24) & 0xFF;

        if (highByte === 0x00) {
            // Leaf node (highByte 0 = no children)
            if (node & 0x00800000) {
                // Mixed leaf (bit 23 set) - always at leaf level
                leaves.push({ morton, isSolid: false, leafMorton: morton });
            } else {
                // Solid leaf - may be a collapsed parent if depth < treeDepth
                expandSolid(morton, depth, treeDepth, leaves);
            }
        } else {
            // Interior node - queue children
            const childMask = highByte;
            const baseOffset = node & 0x00FFFFFF;
            for (let octant = 0; octant < 8; octant++) {
                if (childMask & (1 << octant)) {
                    const offset = getChildOffset(childMask, octant);
                    const childIdx = baseOffset + offset;
                    const childMorton = morton * 8 + octant;
                    queue.push({ nodeIdx: childIdx, morton: childMorton, depth: depth + 1 });
                }
            }
        }
    }

    return leaves;
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

    if (metadata.version !== '1.0') {
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

    if (leaves.length === 0) {
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

    const numBlocks = leaves.length;
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

    for (let i = 0; i < numBlocks; i++) {
        const { morton, isSolid, leafMorton } = leaves[i];
        const [bx, by, bz] = mortonToXYZ(morton);

        xArr[i] = gridMin[0] + (bx + 0.5) * blockSize;
        yArr[i] = gridMin[1] + (by + 0.5) * blockSize;
        zArr[i] = gridMin[2] + (bz + 0.5) * blockSize;

        scale0[i] = splatScale;
        scale1[i] = splatScale;
        scale2[i] = splatScale;

        rot0[i] = 1.0;
        rot1[i] = 0.0;
        rot2[i] = 0.0;
        rot3[i] = 0.0;

        let r: number, g: number, b: number;
        if (isSolid) {
            r = 0.9;
            g = 0.1;
            b = 0.1;
        } else {
            const gray = 0.3 + ((leafMorton * 0.618033988749895) % 1.0) * 0.5;
            r = gray;
            g = gray;
            b = gray;
        }

        fdc0[i] = (r - 0.5) / C0;
        fdc1[i] = (g - 0.5) / C0;
        fdc2[i] = (b - 0.5) / C0;

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
