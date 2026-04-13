import { BlockMaskBuffer } from './block-mask-buffer';
import { mortonToXYZ, xyzToMorton } from './morton';
import type { Bounds } from './sparse-octree';

/**
 * Result of marching cubes surface extraction.
 */
interface MarchingCubesMesh {
    /** Vertex positions (3 floats per vertex) */
    positions: Float32Array;

    /** Triangle indices (3 indices per triangle) */
    indices: Uint32Array;
}

// ============================================================================
// Voxel bit helpers
// ============================================================================

// Bit layout within a 4x4x4 block: bitIdx = lx + ly*4 + lz*16
// lo = bits 0-31 (lz 0-1), hi = bits 32-63 (lz 2-3)

/**
 * Test whether a voxel is occupied within a block's bitmask.
 *
 * @param lo - Lower 32 bits of the block mask
 * @param hi - Upper 32 bits of the block mask
 * @param lx - Local x coordinate (0-3)
 * @param ly - Local y coordinate (0-3)
 * @param lz - Local z coordinate (0-3)
 * @returns True if the voxel is occupied
 */
function isVoxelSet(lo: number, hi: number, lx: number, ly: number, lz: number): boolean {
    const bitIdx = lx + ly * 4 + lz * 16;
    if (bitIdx < 32) {
        return (lo & (1 << bitIdx)) !== 0;
    }
    return (hi & (1 << (bitIdx - 32))) !== 0;
}

// ============================================================================
// Occupancy grid from BlockMaskBuffer
// ============================================================================

/**
 * Build a fast-lookup occupancy structure from a BlockMaskBuffer.
 * Returns a function that queries whether a voxel at global coordinates
 * (vx, vy, vz) is occupied.
 *
 * @param accumulator - Block data
 * @returns Lookup function (vx, vy, vz) => boolean
 */
function buildOccupancyLookup(accumulator: BlockMaskBuffer): (vx: number, vy: number, vz: number) => boolean {
    // Map from "bx,by,bz" encoded as single number to {lo,hi} or solid flag
    // Block key = bx + by * stride + bz * stride^2 where stride is large enough
    const mixed = accumulator.getMixedBlocks();
    const solid = accumulator.getSolidBlocks();

    // Use a Map<number, number> where value encodes index into mask arrays.
    // For solid blocks, store -1 as sentinel.
    const blockMap = new Map<number, number>();

    for (let i = 0; i < mixed.morton.length; i++) {
        blockMap.set(mixed.morton[i], i);
    }
    for (let i = 0; i < solid.length; i++) {
        blockMap.set(solid[i], -1);
    }

    const masks = mixed.masks;

    return (vx: number, vy: number, vz: number): boolean => {
        if (vx < 0 || vy < 0 || vz < 0) return false;

        const bx = vx >> 2;
        const by = vy >> 2;
        const bz = vz >> 2;

        const entry = blockMap.get(xyzToMorton(bx, by, bz));
        if (entry === undefined) return false;
        if (entry === -1) return true; // solid block

        const lo = masks[entry * 2];
        const hi = masks[entry * 2 + 1];
        return isVoxelSet(lo, hi, vx & 3, vy & 3, vz & 3);
    };
}

// ============================================================================
// Marching Cubes
// ============================================================================

/**
 * Extract a triangle mesh from a BlockMaskBuffer using marching cubes.
 *
 * Each voxel is treated as a cell in the marching cubes grid. Corner values
 * are binary (0 = empty, 1 = occupied) with a 0.5 threshold. Vertices are
 * placed at edge midpoints, producing a mesh that follows voxel boundaries.
 *
 * @param accumulator - Voxel block data after filtering
 * @param gridBounds - Grid bounds aligned to block boundaries
 * @param voxelResolution - Size of each voxel in world units
 * @returns Mesh with positions and indices
 */
function marchingCubes(
    accumulator: BlockMaskBuffer,
    gridBounds: Bounds,
    voxelResolution: number
): MarchingCubesMesh {
    const isOccupied = buildOccupancyLookup(accumulator);

    // Collect all voxel coordinates that need processing.
    // We need to check every cell where at least one corner differs from
    // the others, which means we need to check occupied voxels and their
    // immediate neighbors.
    const mixed = accumulator.getMixedBlocks();
    const solid = accumulator.getSolidBlocks();

    // Collect set of all block coordinates that exist
    const blockSet = new Set<number>();
    for (let i = 0; i < mixed.morton.length; i++) {
        blockSet.add(mixed.morton[i]);
    }
    for (let i = 0; i < solid.length; i++) {
        blockSet.add(solid[i]);
    }

    // For each block, we process a 4x4x4 region of marching cubes cells.
    // Each cell at (vx, vy, vz) has corners at (vx..vx+1, vy..vy+1, vz..vz+1).
    // We only generate triangles for cells that have mixed corners (not all same).

    // Vertex deduplication: edge ID -> vertex index
    // Edge ID encodes the voxel coordinate and edge direction
    const vertexMap = new Map<number, number>();
    const positions: number[] = [];
    const indices: number[] = [];

    const originX = gridBounds.min.x;
    const originY = gridBounds.min.y;
    const originZ = gridBounds.min.z;

    // Compute strides from actual grid dimensions (+3 for the -1 boundary
    // extension, the far edge +1, and one extra for safety).
    const strideX = Math.round((gridBounds.max.x - gridBounds.min.x) / voxelResolution) + 3;
    const strideXY = strideX * (Math.round((gridBounds.max.y - gridBounds.min.y) / voxelResolution) + 3);

    // Get or create a vertex at the midpoint of an edge.
    // Edge is identified by the lower corner voxel coordinate and axis (0=x, 1=y, 2=z).
    const getVertex = (vx: number, vy: number, vz: number, axis: number): number => {
        // Pack (vx, vy, vz, axis) into a single key. Offset by 1 so that
        // vx = -1 (from the boundary extension) maps to 0, keeping keys non-negative.
        const key = ((vx + 1) + (vy + 1) * strideX + (vz + 1) * strideXY) * 3 + axis;
        let idx = vertexMap.get(key);
        if (idx !== undefined) return idx;

        idx = positions.length / 3;
        let px = originX + vx * voxelResolution;
        let py = originY + vy * voxelResolution;
        let pz = originZ + vz * voxelResolution;

        // Place vertex at edge midpoint (binary field -> always at 0.5)
        if (axis === 0) px += voxelResolution * 0.5;
        else if (axis === 1) py += voxelResolution * 0.5;
        else pz += voxelResolution * 0.5;

        positions.push(px, py, pz);
        vertexMap.set(key, idx);
        return idx;
    };

    // Track processed orphan cells to avoid duplicate triangles.
    // When a cell's owner block doesn't exist, multiple neighboring blocks
    // can reach it via the -1 boundary extension. The Set ensures each
    // orphan cell is only processed once.
    const processedOrphans = new Set<number>();

    // Process all blocks and their boundary neighbors
    const allMortons: number[] = [];
    blockSet.forEach(m => allMortons.push(m));

    for (let bi = 0; bi < allMortons.length; bi++) {
        const morton = allMortons[bi];
        const [bx, by, bz] = mortonToXYZ(morton);

        // Iterate -1..3 to include the boundary layer in the negative
        // direction. Cells at lx/ly/lz = -1 straddle the block edge and
        // are needed to close the surface where no neighboring block exists.
        for (let lz = -1; lz < 4; lz++) {
            for (let ly = -1; ly < 4; ly++) {
                for (let lx = -1; lx < 4; lx++) {
                    const vx = bx * 4 + lx;
                    const vy = by * 4 + ly;
                    const vz = bz * 4 + lz;

                    // Determine which block owns this cell
                    const ownerBx = vx >> 2;
                    const ownerBy = vy >> 2;
                    const ownerBz = vz >> 2;

                    if (ownerBx !== bx || ownerBy !== by || ownerBz !== bz) {
                        // Cell belongs to a different block — skip if that
                        // block exists (it will process the cell itself).
                        // Guard negative coords: xyzToMorton assumes non-negative inputs.
                        if (ownerBx >= 0 && ownerBy >= 0 && ownerBz >= 0 &&
                            blockSet.has(xyzToMorton(ownerBx, ownerBy, ownerBz))) continue;

                        // Owner block doesn't exist — deduplicate so only the
                        // first neighboring block to reach this cell emits triangles.
                        const cellKey = (vx + 1) + (vy + 1) * strideX + (vz + 1) * strideXY;
                        if (processedOrphans.has(cellKey)) continue;
                        processedOrphans.add(cellKey);
                    }

                    // Get corner values for this cell (8 corners)
                    // Corners: (vx,vy,vz), (vx+1,vy,vz), (vx+1,vy+1,vz), (vx,vy+1,vz),
                    //          (vx,vy,vz+1), (vx+1,vy,vz+1), (vx+1,vy+1,vz+1), (vx,vy+1,vz+1)
                    const c0 = isOccupied(vx, vy, vz) ? 1 : 0;
                    const c1 = isOccupied(vx + 1, vy, vz) ? 1 : 0;
                    const c2 = isOccupied(vx + 1, vy + 1, vz) ? 1 : 0;
                    const c3 = isOccupied(vx, vy + 1, vz) ? 1 : 0;
                    const c4 = isOccupied(vx, vy, vz + 1) ? 1 : 0;
                    const c5 = isOccupied(vx + 1, vy, vz + 1) ? 1 : 0;
                    const c6 = isOccupied(vx + 1, vy + 1, vz + 1) ? 1 : 0;
                    const c7 = isOccupied(vx, vy + 1, vz + 1) ? 1 : 0;

                    const cubeIndex = c0 | (c1 << 1) | (c2 << 2) | (c3 << 3) |
                                      (c4 << 4) | (c5 << 5) | (c6 << 6) | (c7 << 7);

                    if (cubeIndex === 0 || cubeIndex === 255) continue;

                    const edges = EDGE_TABLE[cubeIndex]; // eslint-disable-line no-use-before-define
                    if (edges === 0) continue;

                    // Compute vertices on active edges
                    // Edge -> vertex mapping using getVertex with (corner voxel coord, axis)
                    const edgeVerts: number[] = new Array(12);

                    if (edges & 1)    edgeVerts[0]  = getVertex(vx, vy, vz, 0);       // edge 0: x-axis at (vx, vy, vz)
                    if (edges & 2)    edgeVerts[1]  = getVertex(vx + 1, vy, vz, 1);   // edge 1: y-axis at (vx+1, vy, vz)
                    if (edges & 4)    edgeVerts[2]  = getVertex(vx, vy + 1, vz, 0);   // edge 2: x-axis at (vx, vy+1, vz)
                    if (edges & 8)    edgeVerts[3]  = getVertex(vx, vy, vz, 1);       // edge 3: y-axis at (vx, vy, vz)
                    if (edges & 16)   edgeVerts[4]  = getVertex(vx, vy, vz + 1, 0);   // edge 4: x-axis at (vx, vy, vz+1)
                    if (edges & 32)   edgeVerts[5]  = getVertex(vx + 1, vy, vz + 1, 1); // edge 5: y-axis at (vx+1, vy, vz+1)
                    if (edges & 64)   edgeVerts[6]  = getVertex(vx, vy + 1, vz + 1, 0); // edge 6: x-axis at (vx, vy+1, vz+1)
                    if (edges & 128)  edgeVerts[7]  = getVertex(vx, vy, vz + 1, 1);   // edge 7: y-axis at (vx, vy, vz+1)
                    if (edges & 256)  edgeVerts[8]  = getVertex(vx, vy, vz, 2);       // edge 8: z-axis at (vx, vy, vz)
                    if (edges & 512)  edgeVerts[9]  = getVertex(vx + 1, vy, vz, 2);   // edge 9: z-axis at (vx+1, vy, vz)
                    if (edges & 1024) edgeVerts[10] = getVertex(vx + 1, vy + 1, vz, 2); // edge 10: z-axis at (vx+1, vy+1, vz)
                    if (edges & 2048) edgeVerts[11] = getVertex(vx, vy + 1, vz, 2);   // edge 11: z-axis at (vx, vy+1, vz)

                    // Emit triangles (reversed winding to face outward)
                    const triRow = TRI_TABLE[cubeIndex]; // eslint-disable-line no-use-before-define
                    for (let t = 0; t < triRow.length; t += 3) {
                        indices.push(
                            edgeVerts[triRow[t]],
                            edgeVerts[triRow[t + 2]],
                            edgeVerts[triRow[t + 1]]
                        );
                    }
                }
            }
        }
    }

    return {
        positions: new Float32Array(positions),
        indices: new Uint32Array(indices)
    };
}

// ============================================================================
// Marching Cubes Lookup Tables
// ============================================================================
// Standard tables from Paul Bourke's polygonising a scalar field.
// EDGE_TABLE: 256 entries, each a 12-bit mask of which edges are intersected.
// TRI_TABLE: 256 entries, each an array of edge indices forming triangles.

const EDGE_TABLE: number[] = [
    0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000
];

const TRI_TABLE: number[][] = [
    [],
    [0, 8, 3],
    [0, 1, 9],
    [1, 8, 3, 9, 8, 1],
    [1, 2, 10],
    [0, 8, 3, 1, 2, 10],
    [9, 2, 10, 0, 2, 9],
    [2, 8, 3, 2, 10, 8, 10, 9, 8],
    [3, 11, 2],
    [0, 11, 2, 8, 11, 0],
    [1, 9, 0, 2, 3, 11],
    [1, 11, 2, 1, 9, 11, 9, 8, 11],
    [3, 10, 1, 11, 10, 3],
    [0, 10, 1, 0, 8, 10, 8, 11, 10],
    [3, 9, 0, 3, 11, 9, 11, 10, 9],
    [9, 8, 10, 10, 8, 11],
    [4, 7, 8],
    [4, 3, 0, 7, 3, 4],
    [0, 1, 9, 8, 4, 7],
    [4, 1, 9, 4, 7, 1, 7, 3, 1],
    [1, 2, 10, 8, 4, 7],
    [3, 4, 7, 3, 0, 4, 1, 2, 10],
    [9, 2, 10, 9, 0, 2, 8, 4, 7],
    [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4],
    [8, 4, 7, 3, 11, 2],
    [11, 4, 7, 11, 2, 4, 2, 0, 4],
    [9, 0, 1, 8, 4, 7, 2, 3, 11],
    [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1],
    [3, 10, 1, 3, 11, 10, 7, 8, 4],
    [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4],
    [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3],
    [4, 7, 11, 4, 11, 9, 9, 11, 10],
    [9, 5, 4],
    [9, 5, 4, 0, 8, 3],
    [0, 5, 4, 1, 5, 0],
    [8, 5, 4, 8, 3, 5, 3, 1, 5],
    [1, 2, 10, 9, 5, 4],
    [3, 0, 8, 1, 2, 10, 4, 9, 5],
    [5, 2, 10, 5, 4, 2, 4, 0, 2],
    [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8],
    [9, 5, 4, 2, 3, 11],
    [0, 11, 2, 0, 8, 11, 4, 9, 5],
    [0, 5, 4, 0, 1, 5, 2, 3, 11],
    [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5],
    [10, 3, 11, 10, 1, 3, 9, 5, 4],
    [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10],
    [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3],
    [5, 4, 8, 5, 8, 10, 10, 8, 11],
    [9, 7, 8, 5, 7, 9],
    [9, 3, 0, 9, 5, 3, 5, 7, 3],
    [0, 7, 8, 0, 1, 7, 1, 5, 7],
    [1, 5, 3, 3, 5, 7],
    [9, 7, 8, 9, 5, 7, 10, 1, 2],
    [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3],
    [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2],
    [2, 10, 5, 2, 5, 3, 3, 5, 7],
    [7, 9, 5, 7, 8, 9, 3, 11, 2],
    [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11],
    [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7],
    [11, 2, 1, 11, 1, 7, 7, 1, 5],
    [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11],
    [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0],
    [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0],
    [11, 10, 5, 7, 11, 5],
    [10, 6, 5],
    [0, 8, 3, 5, 10, 6],
    [9, 0, 1, 5, 10, 6],
    [1, 8, 3, 1, 9, 8, 5, 10, 6],
    [1, 6, 5, 2, 6, 1],
    [1, 6, 5, 1, 2, 6, 3, 0, 8],
    [9, 6, 5, 9, 0, 6, 0, 2, 6],
    [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8],
    [2, 3, 11, 10, 6, 5],
    [11, 0, 8, 11, 2, 0, 10, 6, 5],
    [0, 1, 9, 2, 3, 11, 5, 10, 6],
    [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11],
    [6, 3, 11, 6, 5, 3, 5, 1, 3],
    [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6],
    [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9],
    [6, 5, 9, 6, 9, 11, 11, 9, 8],
    [5, 10, 6, 4, 7, 8],
    [4, 3, 0, 4, 7, 3, 6, 5, 10],
    [1, 9, 0, 5, 10, 6, 8, 4, 7],
    [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4],
    [6, 1, 2, 6, 5, 1, 4, 7, 8],
    [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7],
    [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6],
    [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9],
    [3, 11, 2, 7, 8, 4, 10, 6, 5],
    [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11],
    [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6],
    [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6],
    [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6],
    [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11],
    [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7],
    [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9],
    [10, 4, 9, 6, 4, 10],
    [4, 10, 6, 4, 9, 10, 0, 8, 3],
    [10, 0, 1, 10, 6, 0, 6, 4, 0],
    [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10],
    [1, 4, 9, 1, 2, 4, 2, 6, 4],
    [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4],
    [0, 2, 4, 4, 2, 6],
    [8, 3, 2, 8, 2, 4, 4, 2, 6],
    [10, 4, 9, 10, 6, 4, 11, 2, 3],
    [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6],
    [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10],
    [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1],
    [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3],
    [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1],
    [3, 11, 6, 3, 6, 0, 0, 6, 4],
    [6, 4, 8, 11, 6, 8],
    [7, 10, 6, 7, 8, 10, 8, 9, 10],
    [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10],
    [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0],
    [10, 6, 7, 10, 7, 1, 1, 7, 3],
    [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7],
    [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9],
    [7, 8, 0, 7, 0, 6, 6, 0, 2],
    [7, 3, 2, 6, 7, 2],
    [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7],
    [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7],
    [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11],
    [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1],
    [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6],
    [0, 9, 1, 11, 6, 7],
    [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0],
    [7, 11, 6],
    [7, 6, 11],
    [3, 0, 8, 11, 7, 6],
    [0, 1, 9, 11, 7, 6],
    [8, 1, 9, 8, 3, 1, 11, 7, 6],
    [10, 1, 2, 6, 11, 7],
    [1, 2, 10, 3, 0, 8, 6, 11, 7],
    [2, 9, 0, 2, 10, 9, 6, 11, 7],
    [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8],
    [7, 2, 3, 6, 2, 7],
    [7, 0, 8, 7, 6, 0, 6, 2, 0],
    [2, 7, 6, 2, 3, 7, 0, 1, 9],
    [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6],
    [10, 7, 6, 10, 1, 7, 1, 3, 7],
    [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8],
    [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7],
    [7, 6, 10, 7, 10, 8, 8, 10, 9],
    [6, 8, 4, 11, 8, 6],
    [3, 6, 11, 3, 0, 6, 0, 4, 6],
    [8, 6, 11, 8, 4, 6, 9, 0, 1],
    [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6],
    [6, 8, 4, 6, 11, 8, 2, 10, 1],
    [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6],
    [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9],
    [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3],
    [8, 2, 3, 8, 4, 2, 4, 6, 2],
    [0, 4, 2, 4, 6, 2],
    [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8],
    [1, 9, 4, 1, 4, 2, 2, 4, 6],
    [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1],
    [10, 1, 0, 10, 0, 6, 6, 0, 4],
    [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3],
    [10, 9, 4, 6, 10, 4],
    [4, 9, 5, 7, 6, 11],
    [0, 8, 3, 4, 9, 5, 11, 7, 6],
    [5, 0, 1, 5, 4, 0, 7, 6, 11],
    [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5],
    [9, 5, 4, 10, 1, 2, 7, 6, 11],
    [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5],
    [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2],
    [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6],
    [7, 2, 3, 7, 6, 2, 5, 4, 9],
    [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7],
    [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0],
    [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8],
    [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7],
    [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4],
    [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10],
    [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10],
    [6, 9, 5, 6, 11, 9, 11, 8, 9],
    [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5],
    [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11],
    [6, 11, 3, 6, 3, 5, 5, 3, 1],
    [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6],
    [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10],
    [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5],
    [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3],
    [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2],
    [9, 5, 6, 9, 6, 0, 0, 6, 2],
    [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8],
    [1, 5, 6, 2, 1, 6],
    [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6],
    [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0],
    [0, 3, 8, 5, 6, 10],
    [10, 5, 6],
    [11, 5, 10, 7, 5, 11],
    [11, 5, 10, 11, 7, 5, 8, 3, 0],
    [5, 11, 7, 5, 10, 11, 1, 9, 0],
    [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1],
    [11, 1, 2, 11, 7, 1, 7, 5, 1],
    [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11],
    [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7],
    [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2],
    [2, 5, 10, 2, 3, 5, 3, 7, 5],
    [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5],
    [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2],
    [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2],
    [1, 3, 5, 3, 7, 5],
    [0, 8, 7, 0, 7, 1, 1, 7, 5],
    [9, 0, 3, 9, 3, 5, 5, 3, 7],
    [9, 8, 7, 5, 9, 7],
    [5, 8, 4, 5, 10, 8, 10, 11, 8],
    [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0],
    [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5],
    [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4],
    [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8],
    [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11],
    [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5],
    [9, 4, 5, 2, 11, 3],
    [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4],
    [5, 10, 2, 5, 2, 4, 4, 2, 0],
    [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9],
    [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2],
    [8, 4, 5, 8, 5, 3, 3, 5, 1],
    [0, 4, 5, 1, 0, 5],
    [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5],
    [9, 4, 5],
    [4, 11, 7, 4, 9, 11, 9, 10, 11],
    [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11],
    [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11],
    [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4],
    [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2],
    [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3],
    [11, 7, 4, 11, 4, 2, 2, 4, 0],
    [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4],
    [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9],
    [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7],
    [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10],
    [1, 10, 2, 8, 7, 4],
    [4, 9, 1, 4, 1, 7, 7, 1, 3],
    [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1],
    [4, 0, 3, 7, 4, 3],
    [4, 8, 7],
    [9, 10, 8, 10, 11, 8],
    [3, 0, 9, 3, 9, 11, 11, 9, 10],
    [0, 1, 10, 0, 10, 8, 8, 10, 11],
    [3, 1, 10, 11, 3, 10],
    [1, 2, 11, 1, 11, 9, 9, 11, 8],
    [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9],
    [0, 2, 11, 8, 0, 11],
    [3, 2, 11],
    [2, 3, 8, 2, 8, 10, 10, 8, 9],
    [9, 10, 2, 0, 9, 2],
    [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8],
    [1, 10, 2],
    [1, 3, 8, 9, 1, 8],
    [0, 9, 1],
    [0, 3, 8],
    []
];

export { marchingCubes };
export type { MarchingCubesMesh };
