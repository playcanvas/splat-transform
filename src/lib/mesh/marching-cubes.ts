import type { Bounds } from '../data-table';
import {
    BLOCK_EMPTY,
    BLOCK_SOLID,
    BLOCKS_PER_WORD,
    EVEN_BITS,
    SparseVoxelGrid,
    readBlockType
} from '../voxel/sparse-voxel-grid';

/**
 * A simple triangle mesh with positions and indices.
 */
interface Mesh {
    /** Vertex positions (3 floats per vertex) */
    positions: Float32Array;

    /** Triangle indices (3 indices per triangle) */
    indices: Uint32Array;
}

/**
 * Result of marching cubes surface extraction.
 */
type MarchingCubesMesh = Mesh;

/**
 * Options for marching cubes extraction.
 */
interface MarchingCubesOptions {
    /**
     * Pre-merge exact full-face cells on flat axis-aligned regions before
     * creating the mesh. Ambiguous and bevel cases still use normal marching
     * cubes, so coplanarMerge can apply the final lossless optimization.
     */
    mergeFlatFaces?: boolean;
}

interface LocalTriPrim {
    e0: number;
    e1: number;
    e2: number;
    nx: number;
    ny: number;
    nz: number;
    d: number;
    orientation: number;
    merge: boolean;
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
// Marching Cubes
// ============================================================================

// Sentinel values for the per-block 3x3x3 neighbor table.
const NEIGHBOR_EMPTY = -2;
const NEIGHBOR_SOLID = -1;

const EDGE_VERTEX_Q = new Int8Array([
    1, 0, 0,
    2, 1, 0,
    1, 2, 0,
    0, 1, 0,
    1, 0, 2,
    2, 1, 2,
    1, 2, 2,
    0, 1, 2,
    0, 0, 1,
    2, 0, 1,
    2, 2, 1,
    0, 2, 1
]);

const HASH_A = 0x9E3779B9;
const HASH_B = 0x85EBCA6B;
const HASH_C = 0xC2B2AE35;

/**
 * Extract a triangle mesh from a SparseVoxelGrid using marching cubes.
 *
 * Each voxel is treated as a cell in the marching cubes grid. Corner values
 * are binary (0 = empty, 1 = occupied) with a 0.5 threshold. Vertices are
 * placed at edge midpoints, producing a mesh that follows voxel boundaries.
 *
 * @param grid - Voxel grid (after filtering / nav phases)
 * @param gridBounds - Grid bounds aligned to block boundaries
 * @param voxelResolution - Size of each voxel in world units
 * @param options - Optional extraction settings
 * @returns Mesh with positions and indices
 */
function marchingCubes(
    grid: SparseVoxelGrid,
    gridBounds: Bounds,
    voxelResolution: number,
    options: MarchingCubesOptions = {}
): MarchingCubesMesh {
    const { nbx, nby, nbz, bStride, types, masks } = grid;
    const totalBlocks = nbx * nby * nbz;
    const mergeFlatFaces = options.mergeFlatFaces === true;

    // Vertex deduplication: edge ID -> vertex index. Open-addressed typed-
    // array hash table rather than `Map<number, number>` because (1) a single
    // V8 Map is capped at ~2^24 entries (large carved scenes blow past this),
    // and (2) per-entry overhead in Map is ~50 bytes vs 12 bytes (Float64 key
    // + Uint32 value) here, which matters when the table holds tens of
    // millions of vertices.
    //
    // Empty slots are marked with `key === -1` (real keys are non-negative).
    // Hash uses Fibonacci constant on the lower 32 bits of the key. The same
    // structure is used for orphan-cell deduplication, where `vVals` is unused.
    let vCap = 1 << 14;
    let vMask = vCap - 1;
    let vSize = 0;
    let vKeys = new Float64Array(vCap).fill(-1);
    let vVals = new Uint32Array(vCap);

    const vGrow = (): void => {
        const oldKeys = vKeys;
        const oldVals = vVals;
        const oldCap = vCap;
        vCap *= 2;
        vMask = vCap - 1;
        vKeys = new Float64Array(vCap).fill(-1);
        vVals = new Uint32Array(vCap);
        for (let j = 0; j < oldCap; j++) {
            const k = oldKeys[j];
            if (k === -1) continue;
            let i = (Math.imul(k | 0, 0x9E3779B9) >>> 0) & vMask;
            while (vKeys[i] !== -1) i = (i + 1) & vMask;
            vKeys[i] = k;
            vVals[i] = oldVals[j];
        }
    };

    let oCap = 1 << 14;
    let oMask = oCap - 1;
    let oSize = 0;
    let oKeys = new Float64Array(oCap).fill(-1);

    const oGrow = (): void => {
        const oldKeys = oKeys;
        const oldCap = oCap;
        oCap *= 2;
        oMask = oCap - 1;
        oKeys = new Float64Array(oCap).fill(-1);
        for (let j = 0; j < oldCap; j++) {
            const k = oldKeys[j];
            if (k === -1) continue;
            let i = (Math.imul(k | 0, 0x9E3779B9) >>> 0) & oMask;
            while (oKeys[i] !== -1) i = (i + 1) & oMask;
            oKeys[i] = k;
        }
    };

    // Growable typed-array buffers. Capacity doubles on demand to avoid
    // the GC churn of pushing into JS number[] for huge meshes.
    let posCap = 1024;
    let posLen = 0;
    let positions = new Float32Array(posCap);
    let idxCap = 1024;
    let idxLen = 0;
    let indices = new Uint32Array(idxCap);

    const originX = gridBounds.min.x;
    const originY = gridBounds.min.y;
    const originZ = gridBounds.min.z;

    const gridNx = Math.round((gridBounds.max.x - gridBounds.min.x) / voxelResolution);
    const gridNy = Math.round((gridBounds.max.y - gridBounds.min.y) / voxelResolution);
    const gridNz = Math.round((gridBounds.max.z - gridBounds.min.z) / voxelResolution);

    // Compute strides from actual grid dimensions (+3 for the -1 boundary
    // extension, the far edge +1, and one extra for safety).
    const strideX = gridNx + 3;
    const strideXY = strideX * (gridNy + 3);

    // Per-block 3x3x3 neighbor lookup table populated once per processed block.
    // Index = (dx+1) + (dy+1)*3 + (dz+1)*9, dx/dy/dz in {-1, 0, 1}.
    // neighborEntry: NEIGHBOR_EMPTY, NEIGHBOR_SOLID, or NEIGHBOR_MIXED (mask in neighborMasks).
    const NEIGHBOR_MIXED = 0;
    const neighborEntry = new Int32Array(27);
    const neighborMasks = new Uint32Array(54);

    // Reused scratch for emitted edge vertex indices.
    const edgeVerts = new Int32Array(12);

    // Block coordinate of the block currently being processed. Captured by
    // isOccupiedLocal so it can fold the per-corner block lookup into a
    // direct typed-array index instead of a hash lookup.
    let bx = 0, by = 0, bz = 0;

    const isOccupiedLocal = (cx: number, cy: number, cz: number): boolean => {
        if (cx < 0 || cy < 0 || cz < 0) return false;
        const idx = ((cx >> 2) - bx + 1) + ((cy >> 2) - by + 1) * 3 + ((cz >> 2) - bz + 1) * 9;
        const entry = neighborEntry[idx];
        if (entry === NEIGHBOR_EMPTY) return false;
        if (entry === NEIGHBOR_SOLID) return true;
        const lo = neighborMasks[idx * 2];
        const hi = neighborMasks[idx * 2 + 1];
        return isVoxelSet(lo, hi, cx & 3, cy & 3, cz & 3);
    };

    const addPosition = (px: number, py: number, pz: number): number => {
        if (posLen + 3 > posCap) {
            posCap *= 2;
            const grown = new Float32Array(posCap);
            grown.set(positions);
            positions = grown;
        }

        const idx = posLen / 3;
        positions[posLen++] = px;
        positions[posLen++] = py;
        positions[posLen++] = pz;
        return idx;
    };

    const ensureIndexCapacity = (additional: number): void => {
        if (idxLen + additional <= idxCap) return;
        while (idxLen + additional > idxCap) {
            idxCap *= 2;
        }
        const grown = new Uint32Array(idxCap);
        grown.set(indices);
        indices = grown;
    };

    const appendTri = (a: number, b: number, c: number): void => {
        ensureIndexCapacity(3);
        indices[idxLen++] = a;
        indices[idxLen++] = b;
        indices[idxLen++] = c;
    };

    // Get or create a vertex at the midpoint of an edge.
    // Edge is identified by the lower corner voxel coordinate and axis (0=x, 1=y, 2=z).
    const getVertex = (vx: number, vy: number, vz: number, axis: number): number => {
        // Pack (vx, vy, vz, axis) into a single key. Offset by 1 so that
        // vx = -1 (from the boundary extension) maps to 0, keeping keys non-negative.
        const key = ((vx + 1) + (vy + 1) * strideX + (vz + 1) * strideXY) * 3 + axis;

        // Probe for either the matching slot or the next empty one.
        let i = (Math.imul(key | 0, 0x9E3779B9) >>> 0) & vMask;
        while (true) {
            const k = vKeys[i];
            if (k === key) return vVals[i];
            if (k === -1) break;
            i = (i + 1) & vMask;
        }

        let px = originX + vx * voxelResolution;
        let py = originY + vy * voxelResolution;
        let pz = originZ + vz * voxelResolution;

        // Place vertex at edge midpoint (binary field -> always at 0.5)
        if (axis === 0) px += voxelResolution * 0.5;
        else if (axis === 1) py += voxelResolution * 0.5;
        else pz += voxelResolution * 0.5;

        const idx = addPosition(px, py, pz);
        vKeys[i] = key;
        vVals[i] = idx;
        vSize++;
        if (vSize > ((vCap * 0.7) | 0)) vGrow();
        return idx;
    };

    // Full-face MC cases can be merged before vertex creation. Encode each
    // unit face cell as a sortable integer in Float64: bucket / plane / u / v,
    // where bucket = axis*2 + positiveNormalBit and coordinates are offset by
    // +1 to cover the -1 boundary extension.
    const faceCoordStride = Math.max(gridNx, gridNy, gridNz) + 3;
    let faceCellCap = 0;
    let faceCellLen = 0;
    let faceCellKeys = new Float64Array(0);

    const addFaceCell = (bucket: number, p: number, u: number, v: number): void => {
        if (faceCellLen === faceCellCap) {
            faceCellCap = faceCellCap === 0 ? 1024 : faceCellCap * 2;
            const grown = new Float64Array(faceCellCap);
            grown.set(faceCellKeys);
            faceCellKeys = grown;
        }
        faceCellKeys[faceCellLen++] =
            (((bucket * faceCoordStride + (p + 1)) * faceCoordStride + (u + 1)) *
                faceCoordStride + (v + 1));
    };

    const collectFlatFace = (cubeIndex: number, vx: number, vy: number, vz: number): boolean => {
        if (!mergeFlatFaces) return false;

        switch (cubeIndex) {
            case 153: // low-X corners occupied, high-X corners empty => +X normal
                addFaceCell(1, vx, vy, vz);
                return true;
            case 102: // high-X occupied => -X normal
                addFaceCell(0, vx, vy, vz);
                return true;
            case 51: // low-Y occupied => +Y normal
                addFaceCell(3, vy, vx, vz);
                return true;
            case 204: // high-Y occupied => -Y normal
                addFaceCell(2, vy, vx, vz);
                return true;
            case 15: // low-Z occupied => +Z normal
                addFaceCell(5, vz, vx, vy);
                return true;
            case 240: // high-Z occupied => -Z normal
                addFaceCell(4, vz, vx, vy);
                return true;
            default:
                return false;
        }
    };

    const getFaceVertex = (axis: number, p: number, u: number, v: number): number => {
        if (axis === 0) return getVertex(p, u, v, 0);
        if (axis === 1) return getVertex(u, p, v, 1);
        return getVertex(u, v, p, 2);
    };

    const addFaceCenter = (axis: number, p: number, u: number, v: number): number => {
        if (axis === 0) {
            return addPosition(
                originX + (p + 0.5) * voxelResolution,
                originY + u * voxelResolution,
                originZ + v * voxelResolution
            );
        }
        if (axis === 1) {
            return addPosition(
                originX + u * voxelResolution,
                originY + (p + 0.5) * voxelResolution,
                originZ + v * voxelResolution
            );
        }
        return addPosition(
            originX + u * voxelResolution,
            originY + v * voxelResolution,
            originZ + (p + 0.5) * voxelResolution
        );
    };

    const emitFaceQuad = (axis: number, positive: boolean, p: number, u: number, v: number): void => {
        const a = getFaceVertex(axis, p, u, v);
        const b = getFaceVertex(axis, p, u + 1, v);
        const c = getFaceVertex(axis, p, u + 1, v + 1);
        const d = getFaceVertex(axis, p, u, v + 1);
        const localCcwIsPositive = axis !== 1;
        if (positive === localCcwIsPositive) {
            appendTri(a, b, c);
            appendTri(a, c, d);
        } else {
            appendTri(a, c, b);
            appendTri(a, d, c);
        }
    };

    const emitFaceRectangle = (
        bucket: number,
        p: number,
        u0: number,
        v0: number,
        width: number,
        height: number
    ): void => {
        const rawTriCount = width * height * 2;
        const fanTriCount = (width + height) * 2;
        const axis = bucket >> 1;
        const positive = (bucket & 1) === 1;

        if (fanTriCount >= rawTriCount) {
            for (let dv = 0; dv < height; dv++) {
                for (let du = 0; du < width; du++) {
                    emitFaceQuad(axis, positive, p, u0 + du, v0 + dv);
                }
            }
            return;
        }

        const u1 = u0 + width;
        const v1 = v0 + height;
        const center = addFaceCenter(axis, p, u0 + width * 0.5, v0 + height * 0.5);
        const localCcwIsPositive = axis !== 1;
        const useLocalCcw = positive === localCcwIsPositive;
        const perimeterCount = fanTriCount;
        const perimeter = new Uint32Array(perimeterCount);
        let n = 0;

        for (let u = u0; u <= u1; u++) {
            perimeter[n++] = getFaceVertex(axis, p, u, v0);
        }
        for (let v = v0 + 1; v <= v1; v++) {
            perimeter[n++] = getFaceVertex(axis, p, u1, v);
        }
        for (let u = u1 - 1; u >= u0; u--) {
            perimeter[n++] = getFaceVertex(axis, p, u, v1);
        }
        for (let v = v1 - 1; v > v0; v--) {
            perimeter[n++] = getFaceVertex(axis, p, u0, v);
        }

        ensureIndexCapacity(perimeterCount * 3);
        if (useLocalCcw) {
            for (let i = 0; i < perimeterCount; i++) {
                indices[idxLen++] = center;
                indices[idxLen++] = perimeter[i];
                indices[idxLen++] = perimeter[(i + 1) % perimeterCount];
            }
        } else {
            for (let i = perimeterCount - 1; i >= 0; i--) {
                indices[idxLen++] = center;
                indices[idxLen++] = perimeter[i];
                indices[idxLen++] = perimeter[(i - 1 + perimeterCount) % perimeterCount];
            }
        }
    };

    const flushFaceCells = (): void => {
        if (faceCellLen === 0) return;

        const keys = faceCellKeys.slice(0, faceCellLen);
        keys.sort();

        const decodeGroup = (key: number): { bucket: number; pOff: number } => {
            let q = Math.floor(key / faceCoordStride);
            q = Math.floor(q / faceCoordStride);
            const pOff = q % faceCoordStride;
            const bucket = Math.floor(q / faceCoordStride);
            return { bucket, pOff };
        };

        const decodeUvKey = (key: number): number => {
            const vOff = key % faceCoordStride;
            const q = Math.floor(key / faceCoordStride);
            const uOff = q % faceCoordStride;
            return uOff * faceCoordStride + vOff;
        };

        let start = 0;
        while (start < keys.length) {
            const { bucket, pOff } = decodeGroup(keys[start]);
            let end = start + 1;
            while (end < keys.length) {
                const g = decodeGroup(keys[end]);
                if (g.bucket !== bucket || g.pOff !== pOff) break;
                end++;
            }

            const count = end - start;
            let hCap = 1;
            while (hCap < count / 0.7) hCap *= 2;
            const hMask = hCap - 1;
            const hKeys = new Float64Array(hCap).fill(-1);
            const hVals = new Int32Array(hCap);

            const hash = (key: number): number => {
                const hi = (key / 0x100000000) | 0;
                return (Math.imul((key | 0) ^ hi, 0x9E3779B9) >>> 0) & hMask;
            };

            for (let i = 0; i < count; i++) {
                const uvKey = decodeUvKey(keys[start + i]);
                let h = hash(uvKey);
                while (hKeys[h] !== -1) h = (h + 1) & hMask;
                hKeys[h] = uvKey;
                hVals[h] = i;
            }

            const lookup = (uvKey: number): number => {
                let h = hash(uvKey);
                while (true) {
                    const k = hKeys[h];
                    if (k === uvKey) return hVals[h];
                    if (k === -1) return -1;
                    h = (h + 1) & hMask;
                }
            };

            const visited = new Uint8Array(count);
            const uvKeyOf = (uOff: number, vOff: number): number => uOff * faceCoordStride + vOff;
            const p = pOff - 1;

            for (let i = 0; i < count; i++) {
                if (visited[i]) continue;
                const uvKey = decodeUvKey(keys[start + i]);
                const uOff = Math.floor(uvKey / faceCoordStride);
                const vOff = uvKey % faceCoordStride;

                let width = 1;
                while (true) {
                    const idx = lookup(uvKeyOf(uOff + width, vOff));
                    if (idx === -1 || visited[idx]) break;
                    width++;
                }

                let height = 1;
                while (true) {
                    let canGrow = true;
                    for (let du = 0; du < width; du++) {
                        const idx = lookup(uvKeyOf(uOff + du, vOff + height));
                        if (idx === -1 || visited[idx]) {
                            canGrow = false;
                            break;
                        }
                    }
                    if (!canGrow) break;
                    height++;
                }

                for (let dv = 0; dv < height; dv++) {
                    for (let du = 0; du < width; du++) {
                        const idx = lookup(uvKeyOf(uOff + du, vOff + dv));
                        visited[idx] = 1;
                    }
                }

                emitFaceRectangle(bucket, p, uOff - 1, vOff - 1, width, height);
            }

            start = end;
        }
    };

    // Track processed orphan cells to avoid duplicate triangles. When a cell's
    // owner block doesn't exist, multiple neighboring blocks can reach it via
    // the -1 boundary extension. The hash table ensures each orphan cell is
    // only processed once. Same typed-array structure as the vertex hash —
    // see `oKeys` / `oGrow` above. Stored separately because keys collide with
    // vertex keys (same encoding, different namespace).

    // Iterate non-empty blocks of the grid via word-level skipping.
    // Each block is processed once; the inner loop also handles boundary cells
    // that straddle into neighboring blocks (so we don't need a separate pass
    // for orphan boundary cells along the negative grid edges).
    for (let w = 0; w < types.length; w++) {
        const word = types[w];
        if (word === 0) continue;
        let nonEmpty = ((word & EVEN_BITS) | ((word >>> 1) & EVEN_BITS)) >>> 0;
        const baseBlockIdx = w * BLOCKS_PER_WORD;
        while (nonEmpty) {
            const bp = 31 - Math.clz32(nonEmpty & -nonEmpty);
            const lane = bp >>> 1;
            const blockIdx = baseBlockIdx + lane;
            nonEmpty &= nonEmpty - 1;
            if (blockIdx >= totalBlocks) break;

            // Decode block coordinates.
            bx = blockIdx % nbx;
            const byBz = (blockIdx / nbx) | 0;
            by = byBz % nby;
            bz = (byBz / nby) | 0;

            // Populate the 3x3x3 neighbor table for this block. After this loop,
            // every per-cell occupancy query is a direct typed-array index.
            //
            // Track `allNeighborsSolid` so we can skip the entire cell loop
            // for blocks deep inside an obstruction, where every cubeIndex is
            // 255 and no triangles are emitted. On large carved scenes this
            // is the bulk of SOLID blocks and dominates marching-cubes runtime.
            let currentBlockIsSolid = false;
            let allNeighborsSolid = true;
            for (let dz = -1; dz <= 1; dz++) {
                const nbZ = bz + dz;
                for (let dy = -1; dy <= 1; dy++) {
                    const nbY = by + dy;
                    for (let dx = -1; dx <= 1; dx++) {
                        const nbX = bx + dx;
                        const slot = (dx + 1) + (dy + 1) * 3 + (dz + 1) * 9;
                        if (nbX < 0 || nbY < 0 || nbZ < 0 ||
                            nbX >= nbx || nbY >= nby || nbZ >= nbz) {
                            neighborEntry[slot] = NEIGHBOR_EMPTY;
                            allNeighborsSolid = false;
                            continue;
                        }
                        const nbIdx = nbX + nbY * nbx + nbZ * bStride;
                        const bt = readBlockType(types, nbIdx);
                        if (bt === BLOCK_EMPTY) {
                            neighborEntry[slot] = NEIGHBOR_EMPTY;
                            allNeighborsSolid = false;
                        } else if (bt === BLOCK_SOLID) {
                            neighborEntry[slot] = NEIGHBOR_SOLID;
                            if (dx === 0 && dy === 0 && dz === 0) currentBlockIsSolid = true;
                        } else {
                            neighborEntry[slot] = NEIGHBOR_MIXED;
                            allNeighborsSolid = false;
                            const ms = masks.slot(nbIdx);
                            neighborMasks[slot * 2] = masks.lo[ms];
                            neighborMasks[slot * 2 + 1] = masks.hi[ms];
                        }
                    }
                }
            }

            // Block is fully interior to a solid region — every cubeIndex is
            // 255, every cell would emit 0 triangles. Skip the cell loop.
            if (currentBlockIsSolid && allNeighborsSolid) continue;

            // Iterate cell origins from -1 through 3 on each axis. The -1 and
            // 3 layers straddle block edges and close surfaces where no
            // neighboring block exists.
            for (let lz = -1; lz < 4; lz++) {
                const lzInside = lz >= 0 && lz <= 2;
                for (let ly = -1; ly < 4; ly++) {
                    const lyInside = ly >= 0 && ly <= 2;
                    for (let lx = -1; lx < 4; lx++) {
                    // For solid blocks, the 27 cells with all axes in 0..2
                    // are fully inside the block. All 8 corners are 1 so
                    // cubeIndex == 255 and no triangles are emitted -- skip.
                        if (currentBlockIsSolid && lzInside && lyInside && lx >= 0 && lx <= 2) continue;

                        const vx = bx * 4 + lx;
                        const vy = by * 4 + ly;
                        const vz = bz * 4 + lz;

                        // Determine which block owns this cell
                        const ownerBx = vx >> 2;
                        const ownerBy = vy >> 2;
                        const ownerBz = vz >> 2;

                        if (ownerBx !== bx || ownerBy !== by || ownerBz !== bz) {
                        // Cell belongs to a different block — skip if that
                        // block is non-empty (it will process the cell itself).
                            if (ownerBx >= 0 && ownerBy >= 0 && ownerBz >= 0 &&
                            ownerBx < nbx && ownerBy < nby && ownerBz < nbz) {
                                const ownerIdx = ownerBx + ownerBy * nbx + ownerBz * bStride;
                                if (readBlockType(types, ownerIdx) !== BLOCK_EMPTY) continue;
                            }

                            // Owner block doesn't exist or is out-of-bounds —
                            // deduplicate so only the first neighboring block to
                            // reach this cell emits triangles.
                            const cellKey = (vx + 1) + (vy + 1) * strideX + (vz + 1) * strideXY;
                            let oi = (Math.imul(cellKey | 0, 0x9E3779B9) >>> 0) & oMask;
                            let oFound = false;
                            while (true) {
                                const ok = oKeys[oi];
                                if (ok === cellKey) {
                                    oFound = true;
                                    break;
                                }
                                if (ok === -1) break;
                                oi = (oi + 1) & oMask;
                            }
                            if (oFound) continue;
                            oKeys[oi] = cellKey;
                            oSize++;
                            if (oSize > ((oCap * 0.7) | 0)) oGrow();
                        }

                        // Get corner values for this cell (8 corners)
                        // Corners: (vx,vy,vz), (vx+1,vy,vz), (vx+1,vy+1,vz), (vx,vy+1,vz),
                        //          (vx,vy,vz+1), (vx+1,vy,vz+1), (vx+1,vy+1,vz+1), (vx,vy+1,vz+1)
                        const c0 = isOccupiedLocal(vx, vy, vz) ? 1 : 0;
                        const c1 = isOccupiedLocal(vx + 1, vy, vz) ? 1 : 0;
                        const c2 = isOccupiedLocal(vx + 1, vy + 1, vz) ? 1 : 0;
                        const c3 = isOccupiedLocal(vx, vy + 1, vz) ? 1 : 0;
                        const c4 = isOccupiedLocal(vx, vy, vz + 1) ? 1 : 0;
                        const c5 = isOccupiedLocal(vx + 1, vy, vz + 1) ? 1 : 0;
                        const c6 = isOccupiedLocal(vx + 1, vy + 1, vz + 1) ? 1 : 0;
                        const c7 = isOccupiedLocal(vx, vy + 1, vz + 1) ? 1 : 0;

                        const cubeIndex = c0 | (c1 << 1) | (c2 << 2) | (c3 << 3) |
                                      (c4 << 4) | (c5 << 5) | (c6 << 6) | (c7 << 7);

                        if (cubeIndex === 0 || cubeIndex === 255) continue;

                        if (collectFlatFace(cubeIndex, vx, vy, vz)) continue;

                        const edges = EDGE_TABLE[cubeIndex]; // eslint-disable-line no-use-before-define
                        if (edges === 0) continue;

                        // Compute vertices on active edges
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
                        const triLen = triRow.length;
                        ensureIndexCapacity(triLen);
                        for (let t = 0; t < triLen; t += 3) {
                            indices[idxLen++] = edgeVerts[triRow[t]];
                            indices[idxLen++] = edgeVerts[triRow[t + 2]];
                            indices[idxLen++] = edgeVerts[triRow[t + 1]];
                        }
                    }
                }
            }
        }
    }

    flushFaceCells();

    return {
        positions: positions.slice(0, posLen),
        indices: indices.slice(0, idxLen)
    };
}

let triPrimCache: LocalTriPrim[][] | null = null;
const orientationIds = new Map<string, number>();
const orientationNx: number[] = [];
const orientationNy: number[] = [];
const orientationNz: number[] = [];

const gcd2 = (a: number, b: number): number => {
    let x = Math.abs(a);
    let y = Math.abs(b);
    while (y !== 0) {
        const t = x % y;
        x = y;
        y = t;
    }
    return x;
};

const getOrientationId = (nx: number, ny: number, nz: number): number => {
    const key = `${nx},${ny},${nz}`;
    let id = orientationIds.get(key);
    if (id !== undefined) return id;
    id = orientationIds.size;
    orientationIds.set(key, id);
    orientationNx[id] = nx;
    orientationNy[id] = ny;
    orientationNz[id] = nz;
    return id;
};

const buildTriPrimCache = (): LocalTriPrim[][] => {
    if (triPrimCache) return triPrimCache;

    triPrimCache = new Array<LocalTriPrim[]>(256);

    for (let cubeIndex = 0; cubeIndex < 256; cubeIndex++) {
        const row = TRI_TABLE[cubeIndex]; // eslint-disable-line no-use-before-define
        const prims: LocalTriPrim[] = [];
        for (let i = 0; i < row.length; i += 3) {
            // Match the runtime marchingCubes winding, which reverses the
            // lookup-table b/c order to face outward.
            const e0 = row[i];
            const e1 = row[i + 2];
            const e2 = row[i + 1];
            const a = e0 * 3;
            const b = e1 * 3;
            const c = e2 * 3;
            const ax = EDGE_VERTEX_Q[a];
            const ay = EDGE_VERTEX_Q[a + 1];
            const az = EDGE_VERTEX_Q[a + 2];
            const bx = EDGE_VERTEX_Q[b];
            const by = EDGE_VERTEX_Q[b + 1];
            const bz = EDGE_VERTEX_Q[b + 2];
            const cx = EDGE_VERTEX_Q[c];
            const cy = EDGE_VERTEX_Q[c + 1];
            const cz = EDGE_VERTEX_Q[c + 2];

            const ux = bx - ax;
            const uy = by - ay;
            const uz = bz - az;
            const vx = cx - ax;
            const vy = cy - ay;
            const vz = cz - az;
            let nx = uy * vz - uz * vy;
            let ny = uz * vx - ux * vz;
            let nz = ux * vy - uy * vx;
            const g = gcd2(gcd2(nx, ny), nz);
            nx /= g;
            ny /= g;
            nz /= g;

            prims.push({
                e0,
                e1,
                e2,
                nx,
                ny,
                nz,
                d: nx * ax + ny * ay + nz * az,
                orientation: getOrientationId(nx, ny, nz),
                merge: false
            });
        }

        const groups = new Map<string, number[]>();
        for (let i = 0; i < prims.length; i++) {
            const prim = prims[i];
            const normalL1 = Math.abs(prim.nx) + Math.abs(prim.ny) + Math.abs(prim.nz);
            if (normalL1 > 2) continue;
            const key = `${prim.nx},${prim.ny},${prim.nz},${prim.d}`;
            let group = groups.get(key);
            if (!group) {
                group = [];
                groups.set(key, group);
            }
            group.push(i);
        }

        for (const group of groups.values()) {
            if (group.length !== 2) continue;
            if (prims.length !== 2) continue;

            const vertices = new Set<number>();
            const edges = new Map<string, number>();
            for (const primIdx of group) {
                const prim = prims[primIdx];
                vertices.add(prim.e0);
                vertices.add(prim.e1);
                vertices.add(prim.e2);
                const addEdge = (a: number, b: number): void => {
                    const key = a < b ? `${a},${b}` : `${b},${a}`;
                    edges.set(key, (edges.get(key) ?? 0) + 1);
                };
                addEdge(prim.e0, prim.e1);
                addEdge(prim.e1, prim.e2);
                addEdge(prim.e2, prim.e0);
            }
            let boundaryEdges = 0;
            let sharedEdges = 0;
            for (const count of edges.values()) {
                if (count === 1) boundaryEdges++;
                else if (count === 2) sharedEdges++;
            }
            if (vertices.size === 4 && boundaryEdges === 4 && sharedEdges === 1) {
                prims[group[0]].merge = true;
                prims[group[1]].merge = true;
            }
        }
        triPrimCache[cubeIndex] = prims;
    }

    return triPrimCache;
};

/**
 * Extract a marching-cubes mesh while merging coplanar regions during
 * generation.
 *
 * The raw MC triangles are streamed into exact quantized plane buckets. Edges
 * shared by coplanar triangles cancel immediately, leaving only patch
 * boundaries. Those boundaries are then triangulated with globally shared
 * quantized vertices. This preserves the MC surface geometry while avoiding
 * the huge raw mesh and the generic post-process collapse pass.
 *
 * @param grid - Voxel grid (after filtering / nav phases)
 * @param gridBounds - Grid bounds aligned to block boundaries
 * @param voxelResolution - Size of each voxel in world units
 * @returns Mesh with positions and indices
 */
function marchingCubesMerged(
    grid: SparseVoxelGrid,
    gridBounds: Bounds,
    voxelResolution: number
): MarchingCubesMesh {
    const { nbx, nby, nbz, bStride, types, masks } = grid;
    const totalBlocks = nbx * nby * nbz;
    const triPrims = buildTriPrimCache();

    const originX = gridBounds.min.x;
    const originY = gridBounds.min.y;
    const originZ = gridBounds.min.z;
    const halfResolution = voxelResolution * 0.5;

    const gridNx = Math.round((gridBounds.max.x - gridBounds.min.x) / voxelResolution);
    const gridNy = Math.round((gridBounds.max.y - gridBounds.min.y) / voxelResolution);
    const gridNz = Math.round((gridBounds.max.z - gridBounds.min.z) / voxelResolution);
    const maxGrid = Math.max(gridNx, gridNy, gridNz);
    const qStride = maxGrid * 2 + 5;
    const qStride2 = qStride * qStride;
    const cellStrideX = gridNx + 3;
    const cellStrideXY = cellStrideX * (gridNy + 3);

    let maxNormalL1 = 1;
    for (let i = 0; i < orientationNx.length; i++) {
        maxNormalL1 = Math.max(
            maxNormalL1,
            Math.abs(orientationNx[i]) + Math.abs(orientationNy[i]) + Math.abs(orientationNz[i])
        );
    }
    const planeDOffset = maxNormalL1 * (maxGrid * 2 + 4);
    const planeDStride = planeDOffset * 2 + 1;

    let edgeCap = 1 << 14;
    let edgeMask = edgeCap - 1;
    let edgeSize = 0;
    let edgeUsed = 0;
    let edgeState = new Uint8Array(edgeCap);
    let edgePlane = new Float64Array(edgeCap);
    let edgeA = new Float64Array(edgeCap);
    let edgeB = new Float64Array(edgeCap);
    let edgeStart = new Float64Array(edgeCap);
    let edgeEnd = new Float64Array(edgeCap);
    let rawTriCap = 1024;
    let rawTriLen = 0;
    let rawTriVerts = new Float64Array(rawTriCap * 3);

    const hashFloat = (value: number, mul: number): number => {
        return Math.imul((value | 0) ^ ((value / 0x100000000) | 0), mul);
    };

    const hashEdge = (plane: number, a: number, b: number): number => {
        return (
            hashFloat(plane, HASH_A) ^
            hashFloat(a, HASH_B) ^
            hashFloat(b, HASH_C)
        ) >>> 0;
    };

    const rehashEdges = (newCap: number): void => {
        const oldState = edgeState;
        const oldPlane = edgePlane;
        const oldA = edgeA;
        const oldB = edgeB;
        const oldStart = edgeStart;
        const oldEnd = edgeEnd;
        const oldCap = edgeCap;

        edgeCap = newCap;
        edgeMask = edgeCap - 1;
        edgeSize = 0;
        edgeUsed = 0;
        edgeState = new Uint8Array(edgeCap);
        edgePlane = new Float64Array(edgeCap);
        edgeA = new Float64Array(edgeCap);
        edgeB = new Float64Array(edgeCap);
        edgeStart = new Float64Array(edgeCap);
        edgeEnd = new Float64Array(edgeCap);

        for (let i = 0; i < oldCap; i++) {
            if (oldState[i] !== 1) continue;
            let h = hashEdge(oldPlane[i], oldA[i], oldB[i]) & edgeMask;
            while (edgeState[h] === 1) h = (h + 1) & edgeMask;
            edgeState[h] = 1;
            edgePlane[h] = oldPlane[i];
            edgeA[h] = oldA[i];
            edgeB[h] = oldB[i];
            edgeStart[h] = oldStart[i];
            edgeEnd[h] = oldEnd[i];
            edgeSize++;
            edgeUsed++;
        }
    };

    const addPlaneEdge = (plane: number, start: number, end: number): void => {
        if (edgeUsed > ((edgeCap * 0.7) | 0)) {
            rehashEdges(edgeSize > edgeCap * 0.35 ? edgeCap * 2 : edgeCap);
        }

        const a = start < end ? start : end;
        const b = start < end ? end : start;
        let h = hashEdge(plane, a, b) & edgeMask;
        let firstDeleted = -1;

        while (true) {
            const state = edgeState[h];
            if (state === 0) {
                const slot = firstDeleted === -1 ? h : firstDeleted;
                if (firstDeleted === -1) edgeUsed++;
                edgeState[slot] = 1;
                edgePlane[slot] = plane;
                edgeA[slot] = a;
                edgeB[slot] = b;
                edgeStart[slot] = start;
                edgeEnd[slot] = end;
                edgeSize++;
                return;
            }
            if (state === 2) {
                if (firstDeleted === -1) firstDeleted = h;
            } else if (edgePlane[h] === plane && edgeA[h] === a && edgeB[h] === b) {
                edgeState[h] = 2;
                edgeSize--;
                return;
            }
            h = (h + 1) & edgeMask;
        }
    };

    const addRawTri = (a: number, b: number, c: number): void => {
        if (rawTriLen === rawTriCap) {
            rawTriCap *= 2;
            const grown = new Float64Array(rawTriCap * 3);
            grown.set(rawTriVerts);
            rawTriVerts = grown;
        }
        const i = rawTriLen * 3;
        rawTriVerts[i] = a;
        rawTriVerts[i + 1] = b;
        rawTriVerts[i + 2] = c;
        rawTriLen++;
    };

    const vertexKey = (qx: number, qy: number, qz: number): number => {
        return (qx + 2) + (qy + 2) * qStride + (qz + 2) * qStride2;
    };

    const edgeVertexKey = (edge: number, baseQx: number, baseQy: number, baseQz: number): number => {
        const e = edge * 3;
        return vertexKey(
            baseQx + EDGE_VERTEX_Q[e],
            baseQy + EDGE_VERTEX_Q[e + 1],
            baseQz + EDGE_VERTEX_Q[e + 2]
        );
    };

    const NEIGHBOR_MIXED = 0;
    const neighborEntry = new Int32Array(27);
    const neighborMasks = new Uint32Array(54);
    let bx = 0, by = 0, bz = 0;

    const isOccupiedLocal = (cx: number, cy: number, cz: number): boolean => {
        if (cx < 0 || cy < 0 || cz < 0) return false;
        const idx = ((cx >> 2) - bx + 1) + ((cy >> 2) - by + 1) * 3 + ((cz >> 2) - bz + 1) * 9;
        const entry = neighborEntry[idx];
        if (entry === NEIGHBOR_EMPTY) return false;
        if (entry === NEIGHBOR_SOLID) return true;
        const lo = neighborMasks[idx * 2];
        const hi = neighborMasks[idx * 2 + 1];
        return isVoxelSet(lo, hi, cx & 3, cy & 3, cz & 3);
    };

    let orphanCap = 1 << 14;
    let orphanMask = orphanCap - 1;
    let orphanSize = 0;
    let orphanKeys = new Float64Array(orphanCap).fill(-1);

    const growOrphans = (): void => {
        const oldKeys = orphanKeys;
        const oldCap = orphanCap;
        orphanCap *= 2;
        orphanMask = orphanCap - 1;
        orphanKeys = new Float64Array(orphanCap).fill(-1);
        for (let i = 0; i < oldCap; i++) {
            const key = oldKeys[i];
            if (key === -1) continue;
            let h = hashFloat(key, HASH_A) & orphanMask;
            while (orphanKeys[h] !== -1) h = (h + 1) & orphanMask;
            orphanKeys[h] = key;
        }
    };

    for (let w = 0; w < types.length; w++) {
        const word = types[w];
        if (word === 0) continue;
        let nonEmpty = ((word & EVEN_BITS) | ((word >>> 1) & EVEN_BITS)) >>> 0;
        const baseBlockIdx = w * BLOCKS_PER_WORD;
        while (nonEmpty) {
            const bp = 31 - Math.clz32(nonEmpty & -nonEmpty);
            const lane = bp >>> 1;
            const blockIdx = baseBlockIdx + lane;
            nonEmpty &= nonEmpty - 1;
            if (blockIdx >= totalBlocks) break;

            bx = blockIdx % nbx;
            const byBz = (blockIdx / nbx) | 0;
            by = byBz % nby;
            bz = (byBz / nby) | 0;

            let currentBlockIsSolid = false;
            let allNeighborsSolid = true;
            for (let dz = -1; dz <= 1; dz++) {
                const nbZ = bz + dz;
                for (let dy = -1; dy <= 1; dy++) {
                    const nbY = by + dy;
                    for (let dx = -1; dx <= 1; dx++) {
                        const nbX = bx + dx;
                        const slot = (dx + 1) + (dy + 1) * 3 + (dz + 1) * 9;
                        if (nbX < 0 || nbY < 0 || nbZ < 0 ||
                            nbX >= nbx || nbY >= nby || nbZ >= nbz) {
                            neighborEntry[slot] = NEIGHBOR_EMPTY;
                            allNeighborsSolid = false;
                            continue;
                        }
                        const nbIdx = nbX + nbY * nbx + nbZ * bStride;
                        const bt = readBlockType(types, nbIdx);
                        if (bt === BLOCK_EMPTY) {
                            neighborEntry[slot] = NEIGHBOR_EMPTY;
                            allNeighborsSolid = false;
                        } else if (bt === BLOCK_SOLID) {
                            neighborEntry[slot] = NEIGHBOR_SOLID;
                            if (dx === 0 && dy === 0 && dz === 0) currentBlockIsSolid = true;
                        } else {
                            neighborEntry[slot] = NEIGHBOR_MIXED;
                            allNeighborsSolid = false;
                            const ms = masks.slot(nbIdx);
                            neighborMasks[slot * 2] = masks.lo[ms];
                            neighborMasks[slot * 2 + 1] = masks.hi[ms];
                        }
                    }
                }
            }

            if (currentBlockIsSolid && allNeighborsSolid) continue;

            for (let lz = -1; lz < 4; lz++) {
                const lzInside = lz >= 0 && lz <= 2;
                for (let ly = -1; ly < 4; ly++) {
                    const lyInside = ly >= 0 && ly <= 2;
                    for (let lx = -1; lx < 4; lx++) {
                        if (currentBlockIsSolid && lzInside && lyInside && lx >= 0 && lx <= 2) continue;

                        const vx = bx * 4 + lx;
                        const vy = by * 4 + ly;
                        const vz = bz * 4 + lz;
                        const ownerBx = vx >> 2;
                        const ownerBy = vy >> 2;
                        const ownerBz = vz >> 2;

                        if (ownerBx !== bx || ownerBy !== by || ownerBz !== bz) {
                            if (ownerBx >= 0 && ownerBy >= 0 && ownerBz >= 0 &&
                                ownerBx < nbx && ownerBy < nby && ownerBz < nbz) {
                                const ownerIdx = ownerBx + ownerBy * nbx + ownerBz * bStride;
                                if (readBlockType(types, ownerIdx) !== BLOCK_EMPTY) continue;
                            }

                            const cellKey = (vx + 1) + (vy + 1) * cellStrideX + (vz + 1) * cellStrideXY;
                            let h = hashFloat(cellKey, HASH_A) & orphanMask;
                            let found = false;
                            while (true) {
                                const key = orphanKeys[h];
                                if (key === cellKey) {
                                    found = true;
                                    break;
                                }
                                if (key === -1) break;
                                h = (h + 1) & orphanMask;
                            }
                            if (found) continue;
                            orphanKeys[h] = cellKey;
                            orphanSize++;
                            if (orphanSize > ((orphanCap * 0.7) | 0)) growOrphans();
                        }

                        const c0 = isOccupiedLocal(vx, vy, vz) ? 1 : 0;
                        const c1 = isOccupiedLocal(vx + 1, vy, vz) ? 1 : 0;
                        const c2 = isOccupiedLocal(vx + 1, vy + 1, vz) ? 1 : 0;
                        const c3 = isOccupiedLocal(vx, vy + 1, vz) ? 1 : 0;
                        const c4 = isOccupiedLocal(vx, vy, vz + 1) ? 1 : 0;
                        const c5 = isOccupiedLocal(vx + 1, vy, vz + 1) ? 1 : 0;
                        const c6 = isOccupiedLocal(vx + 1, vy + 1, vz + 1) ? 1 : 0;
                        const c7 = isOccupiedLocal(vx, vy + 1, vz + 1) ? 1 : 0;
                        const cubeIndex = c0 | (c1 << 1) | (c2 << 2) | (c3 << 3) |
                            (c4 << 4) | (c5 << 5) | (c6 << 6) | (c7 << 7);

                        if (cubeIndex === 0 || cubeIndex === 255) continue;

                        const baseQx = vx * 2;
                        const baseQy = vy * 2;
                        const baseQz = vz * 2;
                        const prims = triPrims[cubeIndex];

                        for (let i = 0; i < prims.length; i++) {
                            const prim = prims[i];
                            const planeD = prim.d + prim.nx * baseQx + prim.ny * baseQy + prim.nz * baseQz;
                            const planeKey = prim.orientation * planeDStride + planeD + planeDOffset;
                            const a = edgeVertexKey(prim.e0, baseQx, baseQy, baseQz);
                            const b = edgeVertexKey(prim.e1, baseQx, baseQy, baseQz);
                            const c = edgeVertexKey(prim.e2, baseQx, baseQy, baseQz);
                            if (prim.merge) {
                                addPlaneEdge(planeKey, a, b);
                                addPlaneEdge(planeKey, b, c);
                                addPlaneEdge(planeKey, c, a);
                            } else {
                                addRawTri(a, b, c);
                            }
                        }
                    }
                }
            }
        }
    }

    if (edgeSize === 0 && rawTriLen === 0) {
        return { positions: new Float32Array(0), indices: new Uint32Array(0) };
    }

    let posCap = 1024;
    let posLen = 0;
    let positions = new Float32Array(posCap);
    let idxCap = 1024;
    let idxLen = 0;
    let indices = new Uint32Array(idxCap);
    const vertexMap = new Map<number, number>();

    const decodeVertexKey = (key: number): [number, number, number] => {
        const zOff = Math.floor(key / qStride2);
        const rem = key - zOff * qStride2;
        const yOff = Math.floor(rem / qStride);
        const xOff = rem - yOff * qStride;
        return [xOff - 2, yOff - 2, zOff - 2];
    };

    const addPosition = (qx: number, qy: number, qz: number): number => {
        if (posLen + 3 > posCap) {
            posCap *= 2;
            const grown = new Float32Array(posCap);
            grown.set(positions);
            positions = grown;
        }
        const idx = posLen / 3;
        positions[posLen++] = originX + qx * halfResolution;
        positions[posLen++] = originY + qy * halfResolution;
        positions[posLen++] = originZ + qz * halfResolution;
        return idx;
    };

    const getVertex = (key: number): number => {
        const existing = vertexMap.get(key);
        if (existing !== undefined) return existing;
        const [qx, qy, qz] = decodeVertexKey(key);
        const idx = addPosition(qx, qy, qz);
        vertexMap.set(key, idx);
        return idx;
    };

    const ensureIndexCapacity = (additional: number): void => {
        if (idxLen + additional <= idxCap) return;
        while (idxLen + additional > idxCap) idxCap *= 2;
        const grown = new Uint32Array(idxCap);
        grown.set(indices);
        indices = grown;
    };

    const appendTri = (a: number, b: number, c: number): void => {
        ensureIndexCapacity(3);
        indices[idxLen++] = a;
        indices[idxLen++] = b;
        indices[idxLen++] = c;
    };

    const edgeCount = edgeSize;
    const recordPlane = new Float64Array(edgeCount);
    const recordStart = new Float64Array(edgeCount);
    const recordEnd = new Float64Array(edgeCount);
    let recordLen = 0;
    for (let i = 0; i < edgeCap; i++) {
        if (edgeState[i] !== 1) continue;
        recordPlane[recordLen] = edgePlane[i];
        recordStart[recordLen] = edgeStart[i];
        recordEnd[recordLen] = edgeEnd[i];
        recordLen++;
    }

    let recordStride = 1;
    while (recordStride <= recordLen) recordStride *= 2;
    const sortedRecords = new Float64Array(recordLen);
    for (let i = 0; i < recordLen; i++) {
        sortedRecords[i] = recordPlane[i] * recordStride + i;
    }
    sortedRecords.sort();

    const signedArea = (loop: number[], u: Float64Array, v: Float64Array): number => {
        let area = 0;
        for (let i = 0; i < loop.length; i++) {
            const j = (i + 1) % loop.length;
            area += u[loop[i]] * v[loop[j]] - u[loop[j]] * v[loop[i]];
        }
        return area;
    };

    const pointInTri = (
        px: number,
        py: number,
        ax: number,
        ay: number,
        bx: number,
        by: number,
        cx: number,
        cy: number
    ): boolean => {
        const ab = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
        const bc = (cx - bx) * (py - by) - (cy - by) * (px - bx);
        const ca = (ax - cx) * (py - cy) - (ay - cy) * (px - cx);
        return ab > 1e-12 && bc > 1e-12 && ca > 1e-12;
    };

    const pointOnSegment = (
        px: number,
        py: number,
        ax: number,
        ay: number,
        bx: number,
        by: number
    ): boolean => {
        const cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
        if (Math.abs(cross) > 1e-12) return false;
        const dot = (px - ax) * (px - bx) + (py - ay) * (py - by);
        return dot < -1e-12;
    };

    const triangulateLoop = (
        loop: number[],
        u: Float64Array,
        v: Float64Array,
        localToGlobal: Float64Array,
        reverse: boolean
    ): void => {
        const n = loop.length;
        if (n < 3) return;

        const prev = new Int32Array(n);
        const next = new Int32Array(n);
        for (let i = 0; i < n; i++) {
            prev[i] = i === 0 ? n - 1 : i - 1;
            next[i] = i === n - 1 ? 0 : i + 1;
        }

        const triArea = (a: number, b: number, c: number): number => {
            const ia = loop[a];
            const ib = loop[b];
            const ic = loop[c];
            return (
                (u[ib] - u[ia]) * (v[ic] - v[ia]) -
                (v[ib] - v[ia]) * (u[ic] - u[ia])
            );
        };

        const appendLoopTri = (a: number, b: number, c: number): void => {
            const ia = getVertex(localToGlobal[loop[a]]);
            const ib = getVertex(localToGlobal[loop[b]]);
            const ic = getVertex(localToGlobal[loop[c]]);
            if (reverse) appendTri(ia, ic, ib);
            else appendTri(ia, ib, ic);
        };

        let remaining = n;
        let current = 0;
        let attempts = 0;

        while (remaining > 3 && attempts < remaining) {
            const a = prev[current];
            const b = current;
            const c = next[current];
            const area = triArea(a, b, c);
            const c2 = next[c];
            const keepsArea = remaining !== 4 || triArea(a, c, c2) > 0;

            if (area > 0 && keepsArea) {
                let containsPoint = false;
                const ia = loop[a];
                const ib = loop[b];
                const ic = loop[c];
                let p = next[c];
                while (p !== a) {
                    const ip = loop[p];
                    if (
                        pointOnSegment(u[ip], v[ip], u[ia], v[ia], u[ic], v[ic]) ||
                        pointInTri(u[ip], v[ip], u[ia], v[ia], u[ib], v[ib], u[ic], v[ic])
                    ) {
                        containsPoint = true;
                        break;
                    }
                    p = next[p];
                }
                if (!containsPoint) {
                    appendLoopTri(a, b, c);
                    next[a] = c;
                    prev[c] = a;
                    current = c;
                    remaining--;
                    attempts = 0;
                    continue;
                }
            }
            current = next[current];
            attempts++;
        }

        if (remaining === 3) {
            const a = current;
            const b = next[a];
            const c = next[b];
            if (triArea(a, b, c) > 0) {
                appendLoopTri(a, b, c);
            }
        }
    };

    const processPlane = (start: number, end: number, planeKey: number): void => {
        const groupCount = end - start;
        const localMap = new Map<number, number>();
        const localToGlobal = new Float64Array(groupCount * 2);
        const edgeLocalStart = new Int32Array(groupCount);
        const edgeLocalEnd = new Int32Array(groupCount);
        let localCount = 0;

        const getLocal = (key: number): number => {
            const existing = localMap.get(key);
            if (existing !== undefined) return existing;
            const idx = localCount++;
            localMap.set(key, idx);
            localToGlobal[idx] = key;
            return idx;
        };

        for (let i = start; i < end; i++) {
            const record = sortedRecords[i] % recordStride;
            edgeLocalStart[i - start] = getLocal(recordStart[record]);
            edgeLocalEnd[i - start] = getLocal(recordEnd[record]);
        }

        const orientation = Math.floor(planeKey / planeDStride);
        const nx = orientationNx[orientation];
        const ny = orientationNy[orientation];
        const nz = orientationNz[orientation];
        const ax = Math.abs(nx);
        const ay = Math.abs(ny);
        const az = Math.abs(nz);
        const dropAxis = ax >= ay && ax >= az ? 0 : (ay >= az ? 1 : 2);
        const flip = (dropAxis === 0 ? nx : (dropAxis === 1 ? ny : nz)) < 0;
        const u = new Float64Array(localCount);
        const v = new Float64Array(localCount);

        for (let i = 0; i < localCount; i++) {
            const [qx, qy, qz] = decodeVertexKey(localToGlobal[i]);
            if (dropAxis === 0) {
                u[i] = qy;
                v[i] = flip ? -qz : qz;
            } else if (dropAxis === 1) {
                u[i] = qx;
                v[i] = flip ? -qz : qz;
            } else {
                u[i] = qx;
                v[i] = flip ? -qy : qy;
            }
        }

        const head = new Int32Array(localCount).fill(-1);
        const nextEdge = new Int32Array(groupCount);
        const used = new Uint8Array(groupCount);

        for (let i = 0; i < groupCount; i++) {
            const s = edgeLocalStart[i];
            nextEdge[i] = head[s];
            head[s] = i;
        }

        const chooseNextEdge = (from: number, prevVertex: number): number => {
            let best = -1;
            let candidateCount = 0;
            for (let e = head[from]; e !== -1; e = nextEdge[e]) {
                if (used[e]) continue;
                candidateCount++;
                if (best === -1) best = e;
            }
            if (candidateCount <= 1 || prevVertex === -1) return best;

            const inX = u[from] - u[prevVertex];
            const inY = v[from] - v[prevVertex];
            let bestTurn = Infinity;
            for (let e = head[from]; e !== -1; e = nextEdge[e]) {
                if (used[e]) continue;
                const to = edgeLocalEnd[e];
                const outX = u[to] - u[from];
                const outY = v[to] - v[from];
                const cross = inX * outY - inY * outX;
                const dot = inX * outX + inY * outY;
                let turn = Math.atan2(cross, dot);
                if (turn <= 0) turn += Math.PI * 2;
                if (turn < bestTurn) {
                    bestTurn = turn;
                    best = e;
                }
            }
            return best;
        };

        for (let i = 0; i < groupCount; i++) {
            if (used[i]) continue;

            const loop: number[] = [];
            const firstEdge = i;
            let edge = firstEdge;
            let prevVertex = -1;

            while (edge !== -1 && !used[edge]) {
                used[edge] = 1;
                const from = edgeLocalStart[edge];
                const to = edgeLocalEnd[edge];
                if (loop.length === 0 || loop[loop.length - 1] !== from) loop.push(from);
                prevVertex = from;
                edge = chooseNextEdge(to, prevVertex);
                if (to === edgeLocalStart[firstEdge] && edge === firstEdge) break;
            }

            if (loop.length < 3) continue;
            const area = signedArea(loop, u, v);
            if (Math.abs(area) < 1e-12) continue;
            if (area > 0) {
                triangulateLoop(loop, u, v, localToGlobal, flip);
            } else {
                loop.reverse();
                triangulateLoop(loop, u, v, localToGlobal, !flip);
            }
        }
    };

    let groupStart = 0;
    while (groupStart < recordLen) {
        const planeKey = Math.floor(sortedRecords[groupStart] / recordStride);
        let groupEnd = groupStart + 1;
        while (groupEnd < recordLen && Math.floor(sortedRecords[groupEnd] / recordStride) === planeKey) {
            groupEnd++;
        }
        processPlane(groupStart, groupEnd, planeKey);
        groupStart = groupEnd;
    }

    for (let i = 0; i < rawTriLen; i++) {
        const t = i * 3;
        appendTri(
            getVertex(rawTriVerts[t]),
            getVertex(rawTriVerts[t + 1]),
            getVertex(rawTriVerts[t + 2])
        );
    }

    return {
        positions: positions.slice(0, posLen),
        indices: indices.slice(0, idxLen)
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

export { marchingCubes, marchingCubesMerged };
export type { Mesh, MarchingCubesMesh, MarchingCubesOptions };
