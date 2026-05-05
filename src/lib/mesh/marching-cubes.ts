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

/**
 * Extract a triangle mesh from a SparseVoxelGrid using marching cubes.
 *
 * Each voxel is treated as a cell in the marching cubes grid. Corner values
 * are binary (0 = empty, 1 = occupied) with a 0.5 threshold. Vertices are
 * placed at edge midpoints, producing the binary-field isosurface between
 * occupied and empty samples.
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
    const scaledCoordStride = (Math.max(gridNx, gridNy, gridNz) + 3) * 2 + 5;
    const scaledCoordOffset = 3;

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

    // When flat MC face cells are merged into large rectangles, rectangle
    // boundaries must still be split at any neighbouring raw-MC vertex that
    // lies along the same edge. Otherwise the pre-merged mesh can contain
    // T-junctions before the final coplanarMerge pass. Coordinates here use
    // the exact binary-MC half-grid: voxel corners are even, MC edge
    // midpoints are odd on the crossed edge axis.
    const splitLinePoints = mergeFlatFaces ? new Map<number, number[]>() : undefined;
    let collectSplitPoints = mergeFlatFaces;

    const splitLineKey = (varAxis: number, x2: number, y2: number, z2: number): number => {
        const x = x2 + scaledCoordOffset;
        const y = y2 + scaledCoordOffset;
        const z = z2 + scaledCoordOffset;
        if (varAxis === 0) return (y * scaledCoordStride + z) * 3;
        if (varAxis === 1) return (x * scaledCoordStride + z) * 3 + 1;
        return (x * scaledCoordStride + y) * 3 + 2;
    };

    const addSplitLinePoint = (key: number, value: number): void => {
        if (!splitLinePoints) return;
        let points = splitLinePoints.get(key);
        if (!points) {
            points = [];
            splitLinePoints.set(key, points);
        }
        points.push(value);
    };

    const addSplitPointForVertex = (vx: number, vy: number, vz: number, axis: number): void => {
        if (!splitLinePoints || !collectSplitPoints) return;
        const x2 = vx * 2 + (axis === 0 ? 1 : 0);
        const y2 = vy * 2 + (axis === 1 ? 1 : 0);
        const z2 = vz * 2 + (axis === 2 ? 1 : 0);

        addSplitLinePoint(splitLineKey(0, x2, y2, z2), x2);
        addSplitLinePoint(splitLineKey(1, x2, y2, z2), y2);
        addSplitLinePoint(splitLineKey(2, x2, y2, z2), z2);
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
        addSplitPointForVertex(vx, vy, vz, axis);
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
    const diagCoordStride = (Math.max(gridNx, gridNy, gridNz) + 3) * 8 + 9;
    const diagCoordOffset = Math.floor(diagCoordStride / 2);
    let diagCellCap = 0;
    let diagCellLen = 0;
    let diagCellKeys = new Float64Array(0);

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

    const addDiagCell = (bucket: number, plane: number, u: number, e: number): void => {
        if (diagCellLen === diagCellCap) {
            diagCellCap = diagCellCap === 0 ? 1024 : diagCellCap * 2;
            const grown = new Float64Array(diagCellCap);
            grown.set(diagCellKeys);
            diagCellKeys = grown;
        }
        diagCellKeys[diagCellLen++] =
            (((bucket * diagCoordStride + (plane + diagCoordOffset)) * diagCoordStride +
                (u + diagCoordOffset)) * diagCoordStride + (e + diagCoordOffset));
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

    const scaledFacePoint = (axis: number, p: number, u: number, v: number): [number, number, number] => {
        if (axis === 0) return [p * 2 + 1, u * 2, v * 2];
        if (axis === 1) return [u * 2, p * 2 + 1, v * 2];
        return [u * 2, v * 2, p * 2 + 1];
    };

    const diagBucket = (axisA: number, axisB: number, signA: number, signB: number): number => {
        const pair = axisA === 0 ? (axisB === 1 ? 0 : 1) : 2;
        const signBits = (signA > 0 ? 1 : 0) | (signB > 0 ? 2 : 0);
        return pair * 4 + signBits;
    };

    const decodeDiagBucket = (bucket: number): {
        axisA: number;
        axisB: number;
        axisE: number;
        signA: number;
        signB: number;
    } => {
        const pair = (bucket / 4) | 0;
        const signBits = bucket & 3;
        const signA = (signBits & 1) !== 0 ? 1 : -1;
        const signB = (signBits & 2) !== 0 ? 1 : -1;
        if (pair === 0) return { axisA: 0, axisB: 1, axisE: 2, signA, signB };
        if (pair === 1) return { axisA: 0, axisB: 2, axisE: 1, signA, signB };
        return { axisA: 1, axisB: 2, axisE: 0, signA, signB };
    };

    const coordByAxis = (x: number, y: number, z: number, axis: number): number => {
        if (axis === 0) return x;
        if (axis === 1) return y;
        return z;
    };

    const diagPoint = (bucket: number, plane: number, u: number, e: number): [number, number, number] => {
        const { axisA, axisB, axisE, signA, signB } = decodeDiagBucket(bucket);
        const out = [0, 0, 0];
        out[axisA] = signA * ((plane + u) / 2);
        out[axisB] = signB * ((plane - u) / 2);
        out[axisE] = e;
        return [out[0], out[1], out[2]];
    };

    const edgeScaledPoint = (
        edge: number,
        vx: number,
        vy: number,
        vz: number,
        out: Int32Array,
        offset: number
    ): void => {
        const x = vx * 2;
        const y = vy * 2;
        const z = vz * 2;
        switch (edge) {
            case 0: out[offset] = x + 1; out[offset + 1] = y; out[offset + 2] = z; break;
            case 1: out[offset] = x + 2; out[offset + 1] = y + 1; out[offset + 2] = z; break;
            case 2: out[offset] = x + 1; out[offset + 1] = y + 2; out[offset + 2] = z; break;
            case 3: out[offset] = x; out[offset + 1] = y + 1; out[offset + 2] = z; break;
            case 4: out[offset] = x + 1; out[offset + 1] = y; out[offset + 2] = z + 2; break;
            case 5: out[offset] = x + 2; out[offset + 1] = y + 1; out[offset + 2] = z + 2; break;
            case 6: out[offset] = x + 1; out[offset + 1] = y + 2; out[offset + 2] = z + 2; break;
            case 7: out[offset] = x; out[offset + 1] = y + 1; out[offset + 2] = z + 2; break;
            case 8: out[offset] = x; out[offset + 1] = y; out[offset + 2] = z + 1; break;
            case 9: out[offset] = x + 2; out[offset + 1] = y; out[offset + 2] = z + 1; break;
            case 10: out[offset] = x + 2; out[offset + 1] = y + 2; out[offset + 2] = z + 1; break;
            default: out[offset] = x; out[offset + 1] = y + 2; out[offset + 2] = z + 1; break;
        }
    };

    const pairVerts = new Int32Array(18);
    const uniqueVerts = new Int32Array(12);

    const samePoint = (
        src: Int32Array,
        a: number,
        b: number
    ): boolean => src[a] === src[b] && src[a + 1] === src[b + 1] && src[a + 2] === src[b + 2];

    const pointInUnique = (x: number, y: number, z: number, uniqueCount: number): boolean => {
        for (let i = 0; i < uniqueCount; i++) {
            const o = i * 3;
            if (uniqueVerts[o] === x && uniqueVerts[o + 1] === y && uniqueVerts[o + 2] === z) {
                return true;
            }
        }
        return false;
    };

    const collectDiagPair = (
        triRow: number[],
        triA: number,
        triB: number,
        vx: number,
        vy: number,
        vz: number
    ): boolean => {
        const edgesA = triA * 3;
        const edgesB = triB * 3;
        edgeScaledPoint(triRow[edgesA], vx, vy, vz, pairVerts, 0);
        edgeScaledPoint(triRow[edgesA + 2], vx, vy, vz, pairVerts, 3);
        edgeScaledPoint(triRow[edgesA + 1], vx, vy, vz, pairVerts, 6);
        edgeScaledPoint(triRow[edgesB], vx, vy, vz, pairVerts, 9);
        edgeScaledPoint(triRow[edgesB + 2], vx, vy, vz, pairVerts, 12);
        edgeScaledPoint(triRow[edgesB + 1], vx, vy, vz, pairVerts, 15);

        let uniqueCount = 0;
        for (let i = 0; i < 6; i++) {
            const src = i * 3;
            let found = false;
            for (let j = 0; j < uniqueCount; j++) {
                const dst = j * 3;
                if (pairVerts[src] === uniqueVerts[dst] &&
                    pairVerts[src + 1] === uniqueVerts[dst + 1] &&
                    pairVerts[src + 2] === uniqueVerts[dst + 2]) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                if (uniqueCount === 4) return false;
                const dst = uniqueCount * 3;
                uniqueVerts[dst] = pairVerts[src];
                uniqueVerts[dst + 1] = pairVerts[src + 1];
                uniqueVerts[dst + 2] = pairVerts[src + 2];
                uniqueCount++;
            }
        }
        if (uniqueCount !== 4) return false;

        const ax = pairVerts[3] - pairVerts[0];
        const ay = pairVerts[4] - pairVerts[1];
        const az = pairVerts[5] - pairVerts[2];
        const bx = pairVerts[6] - pairVerts[0];
        const by = pairVerts[7] - pairVerts[1];
        const bz = pairVerts[8] - pairVerts[2];
        const normal = [
            ay * bz - az * by,
            az * bx - ax * bz,
            ax * by - ay * bx
        ];
        const absNormal = [Math.abs(normal[0]), Math.abs(normal[1]), Math.abs(normal[2])];
        let axisE = -1;
        let axisA = -1;
        let axisB = -1;
        for (let i = 0; i < 3; i++) {
            if (absNormal[i] === 0) axisE = i;
            else if (axisA === -1) axisA = i;
            else axisB = i;
        }
        if (axisE === -1 || axisA === -1 || axisB === -1) return false;
        if (absNormal[axisA] !== absNormal[axisB]) return false;

        const signA = normal[axisA] > 0 ? 1 : -1;
        const signB = normal[axisB] > 0 ? 1 : -1;
        const bucket = diagBucket(axisA, axisB, signA, signB);
        const plane = signA * coordByAxis(uniqueVerts[0], uniqueVerts[1], uniqueVerts[2], axisA) +
            signB * coordByAxis(uniqueVerts[0], uniqueVerts[1], uniqueVerts[2], axisB);
        let minU = Infinity;
        let maxU = -Infinity;
        let minE = Infinity;
        let maxE = -Infinity;
        for (let i = 0; i < uniqueCount; i++) {
            const o = i * 3;
            const a = coordByAxis(uniqueVerts[o], uniqueVerts[o + 1], uniqueVerts[o + 2], axisA);
            const b = coordByAxis(uniqueVerts[o], uniqueVerts[o + 1], uniqueVerts[o + 2], axisB);
            const e = coordByAxis(uniqueVerts[o], uniqueVerts[o + 1], uniqueVerts[o + 2], axisE);
            if (signA * a + signB * b !== plane) return false;
            const u = signA * a - signB * b;
            if (u < minU) minU = u;
            if (u > maxU) maxU = u;
            if (e < minE) minE = e;
            if (e > maxE) maxE = e;
        }
        if (maxU - minU !== 2 || maxE - minE !== 2) return false;

        const p00 = diagPoint(bucket, plane, minU, minE);
        const p10 = diagPoint(bucket, plane, maxU, minE);
        const p11 = diagPoint(bucket, plane, maxU, maxE);
        const p01 = diagPoint(bucket, plane, minU, maxE);
        if (!pointInUnique(p00[0], p00[1], p00[2], uniqueCount) ||
            !pointInUnique(p10[0], p10[1], p10[2], uniqueCount) ||
            !pointInUnique(p11[0], p11[1], p11[2], uniqueCount) ||
            !pointInUnique(p01[0], p01[1], p01[2], uniqueCount)) {
            return false;
        }

        let sharedCount = 0;
        let sharedU0 = 0;
        let sharedE0 = 0;
        let sharedU1 = 0;
        let sharedE1 = 0;
        for (let i = 0; i < 3; i++) {
            const oi = i * 3;
            for (let j = 3; j < 6; j++) {
                const oj = j * 3;
                if (!samePoint(pairVerts, oi, oj)) continue;
                const a = coordByAxis(pairVerts[oi], pairVerts[oi + 1], pairVerts[oi + 2], axisA);
                const b = coordByAxis(pairVerts[oi], pairVerts[oi + 1], pairVerts[oi + 2], axisB);
                const e = coordByAxis(pairVerts[oi], pairVerts[oi + 1], pairVerts[oi + 2], axisE);
                if (sharedCount === 0) {
                    sharedU0 = signA * a - signB * b;
                    sharedE0 = e;
                } else {
                    sharedU1 = signA * a - signB * b;
                    sharedE1 = e;
                }
                sharedCount++;
            }
        }
        if (sharedCount !== 2 || sharedU0 === sharedU1 || sharedE0 === sharedE1) return false;

        addDiagCell(bucket, plane, minU, minE);
        return true;
    };

    const collectDiagFaces = (triRow: number[], vx: number, vy: number, vz: number): number => {
        if (!mergeFlatFaces) return 0;
        const triCount = (triRow.length / 3) | 0;
        let usedMask = 0;
        for (let i = 0; i < triCount; i++) {
            if ((usedMask & (1 << i)) !== 0) continue;
            for (let j = i + 1; j < triCount; j++) {
                if ((usedMask & (1 << j)) !== 0) continue;
                if (collectDiagPair(triRow, i, j, vx, vy, vz)) {
                    usedMask |= (1 << i) | (1 << j);
                    break;
                }
            }
        }
        return usedMask;
    };

    const getScaledVertex = (x2: number, y2: number, z2: number): number => {
        if ((x2 & 1) !== 0) return getVertex((x2 - 1) / 2, y2 / 2, z2 / 2, 0);
        if ((y2 & 1) !== 0) return getVertex(x2 / 2, (y2 - 1) / 2, z2 / 2, 1);
        return getVertex(x2 / 2, y2 / 2, (z2 - 1) / 2, 2);
    };

    let perimeterScratch = new Uint32Array(16);
    let perimeterU = new Int32Array(16);
    let perimeterV = new Int32Array(16);
    let perimeterLen = 0;

    const localFaceUv = (axis: number, x2: number, y2: number, z2: number): [number, number] => {
        if (axis === 0) return [y2, z2];
        if (axis === 1) return [x2, z2];
        return [x2, y2];
    };

    const addPerimeterVertex = (vertex: number, u: number, v: number): void => {
        if (perimeterLen > 0 && perimeterScratch[perimeterLen - 1] === vertex) return;
        if (perimeterLen === perimeterScratch.length) {
            const grown = new Uint32Array(perimeterScratch.length * 2);
            grown.set(perimeterScratch);
            perimeterScratch = grown;
            const grownU = new Int32Array(perimeterU.length * 2);
            grownU.set(perimeterU);
            perimeterU = grownU;
            const grownV = new Int32Array(perimeterV.length * 2);
            grownV.set(perimeterV);
            perimeterV = grownV;
        }
        perimeterScratch[perimeterLen] = vertex;
        perimeterU[perimeterLen] = u;
        perimeterV[perimeterLen] = v;
        perimeterLen++;
    };

    const addPerimeterPoint = (axis: number, x2: number, y2: number, z2: number): void => {
        const [u, v] = localFaceUv(axis, x2, y2, z2);
        addPerimeterVertex(getScaledVertex(x2, y2, z2), u, v);
    };

    const addDiagPerimeterPoint = (bucket: number, x2: number, y2: number, z2: number): void => {
        const { axisA, axisB, axisE, signA, signB } = decodeDiagBucket(bucket);
        const a = coordByAxis(x2, y2, z2, axisA);
        const b = coordByAxis(x2, y2, z2, axisB);
        const e = coordByAxis(x2, y2, z2, axisE);
        addPerimeterVertex(getScaledVertex(x2, y2, z2), signA * a - signB * b, e);
    };

    const addSplitSegment = (
        x0: number, y0: number, z0: number,
        x1: number, y1: number, z1: number
    ): void => {
        const changes = (x0 !== x1 ? 1 : 0) + (y0 !== y1 ? 1 : 0) + (z0 !== z1 ? 1 : 0);
        if (changes !== 1) return;
        if (x0 !== x1) {
            const key = splitLineKey(0, x0, y0, z0);
            addSplitLinePoint(key, x0);
            addSplitLinePoint(key, x1);
        } else if (y0 !== y1) {
            const key = splitLineKey(1, x0, y0, z0);
            addSplitLinePoint(key, y0);
            addSplitLinePoint(key, y1);
        } else {
            const key = splitLineKey(2, x0, y0, z0);
            addSplitLinePoint(key, z0);
            addSplitLinePoint(key, z1);
        }
    };

    const addSplitEdgeVertices = (
        axis: number,
        x0: number, y0: number, z0: number,
        x1: number, y1: number, z1: number,
        addPoint: (x2: number, y2: number, z2: number) => void = (px, py, pz) => {
            addPerimeterPoint(axis, px, py, pz);
        }
    ): void => {
        let varAxis: number;
        let start: number;
        let end: number;
        if (x0 !== x1) {
            varAxis = 0;
            start = x0;
            end = x1;
        } else if (y0 !== y1) {
            varAxis = 1;
            start = y0;
            end = y1;
        } else {
            varAxis = 2;
            start = z0;
            end = z1;
        }

        const points = splitLinePoints?.get(splitLineKey(varAxis, x0, y0, z0));
        if (!points) return;

        const lo = Math.min(start, end);
        const hi = Math.max(start, end);
        const forward = start <= end;

        const emitPoint = (t: number): void => {
            if (varAxis === 0) addPoint(t, y0, z0);
            else if (varAxis === 1) addPoint(x0, t, z0);
            else addPoint(x0, y0, t);
        };

        if (forward) {
            for (let i = 0; i < points.length; i++) {
                const t = points[i];
                if (t < lo) continue;
                if (t > hi) break;
                emitPoint(t);
            }
        } else {
            for (let i = points.length - 1; i >= 0; i--) {
                const t = points[i];
                if (t > hi) continue;
                if (t < lo) break;
                emitPoint(t);
            }
        }
    };

    const appendOrientedTri = (a: number, b: number, c: number, useLocalCcw: boolean): void => {
        if (useLocalCcw) appendTri(a, b, c);
        else appendTri(a, c, b);
    };

    const appendPerimeterTri = (a: number, b: number, c: number, useLocalCcw: boolean): void => {
        const abx = perimeterU[b] - perimeterU[a];
        const aby = perimeterV[b] - perimeterV[a];
        const acx = perimeterU[c] - perimeterU[a];
        const acy = perimeterV[c] - perimeterV[a];
        if (abx * acy - aby * acx <= 0) return;
        appendOrientedTri(perimeterScratch[a], perimeterScratch[b], perimeterScratch[c], useLocalCcw);
    };

    const triangulateTwoSideChain = (
        chainAStart: number,
        chainAEnd: number,
        chainBStart: number,
        chainBEnd: number,
        useLocalCcw: boolean
    ): void => {
        const chainALen = chainAEnd - chainAStart;
        const chainBLen = chainBEnd - chainBStart;
        if (chainALen < 2 || chainBLen < 2) return;

        const v2 = chainBEnd - 1;
        for (let i = chainAStart; i < chainAEnd - 2; i++) {
            appendPerimeterTri(v2, i, i + 1, useLocalCcw);
        }

        const pivot = chainALen > 1 ? chainAEnd - 2 : chainAStart;
        for (let i = chainBStart; i < chainBEnd - 1; i++) {
            appendPerimeterTri(pivot, i, i + 1, useLocalCcw);
        }
    };

    const emitFaceRectangle = (
        bucket: number,
        p: number,
        u0: number,
        v0: number,
        u1: number,
        v1: number
    ): void => {
        const axis = bucket >> 1;
        const positive = (bucket & 1) === 1;
        const a = scaledFacePoint(axis, p, u0, v0);
        const b = scaledFacePoint(axis, p, u1, v0);
        const c = scaledFacePoint(axis, p, u1, v1);
        const d = scaledFacePoint(axis, p, u0, v1);

        perimeterLen = 0;
        const side0Start = 0;
        addSplitEdgeVertices(axis, a[0], a[1], a[2], b[0], b[1], b[2]);
        const side0End = perimeterLen;
        const side1Start = side0End - 1;
        addSplitEdgeVertices(axis, b[0], b[1], b[2], c[0], c[1], c[2]);
        const side1End = perimeterLen;
        const side2Start = side1End - 1;
        addSplitEdgeVertices(axis, c[0], c[1], c[2], d[0], d[1], d[2]);
        const side2End = perimeterLen;
        const side3Start = side2End - 1;
        addSplitEdgeVertices(axis, d[0], d[1], d[2], a[0], a[1], a[2]);
        const side3End = perimeterLen;

        const localCcwIsPositive = axis !== 1;
        const useLocalCcw = positive === localCcwIsPositive;
        triangulateTwoSideChain(side0Start, side0End, side1Start, side1End, useLocalCcw);
        triangulateTwoSideChain(side2Start, side2End, side3Start, side3End, useLocalCcw);
    };

    const emitDiagRectangle = (
        bucket: number,
        plane: number,
        u0: number,
        e0: number,
        u1: number,
        e1: number
    ): void => {
        const a = diagPoint(bucket, plane, u0, e0);
        const b = diagPoint(bucket, plane, u1, e0);
        const c = diagPoint(bucket, plane, u1, e1);
        const d = diagPoint(bucket, plane, u0, e1);
        const addPoint = (x2: number, y2: number, z2: number): void => {
            addDiagPerimeterPoint(bucket, x2, y2, z2);
        };

        perimeterLen = 0;
        const side0Start = 0;
        addPoint(a[0], a[1], a[2]);
        addPoint(b[0], b[1], b[2]);
        const side0End = perimeterLen;
        const side1Start = side0End - 1;
        addSplitEdgeVertices(0, b[0], b[1], b[2], c[0], c[1], c[2], addPoint);
        const side1End = perimeterLen;
        const side2Start = side1End - 1;
        addPoint(d[0], d[1], d[2]);
        const side2End = perimeterLen;
        const side3Start = side2End - 1;
        addSplitEdgeVertices(0, d[0], d[1], d[2], a[0], a[1], a[2], addPoint);
        const side3End = perimeterLen;

        const { axisA, axisB, signA, signB } = decodeDiagBucket(bucket);
        const abx = b[0] - a[0];
        const aby = b[1] - a[1];
        const abz = b[2] - a[2];
        const bcx = c[0] - b[0];
        const bcy = c[1] - b[1];
        const bcz = c[2] - b[2];
        const nx = aby * bcz - abz * bcy;
        const ny = abz * bcx - abx * bcz;
        const nz = abx * bcy - aby * bcx;
        const normal = [0, 0, 0];
        normal[axisA] = signA;
        normal[axisB] = signB;
        const useLocalCcw = nx * normal[0] + ny * normal[1] + nz * normal[2] > 0;

        triangulateTwoSideChain(side0Start, side0End, side1Start, side1End, useLocalCcw);
        triangulateTwoSideChain(side2Start, side2End, side3Start, side3End, useLocalCcw);
    };

    const flushFaceCells = (): void => {
        if (faceCellLen === 0 && diagCellLen === 0) return;

        const keys = faceCellKeys.subarray(0, faceCellLen);
        faceCellKeys = new Float64Array(0);
        keys.sort();

        let rectCap = 1024;
        let rectLen = 0;
        let rectBucket = new Int32Array(rectCap);
        let rectP = new Int32Array(rectCap);
        let rectU0 = new Int32Array(rectCap);
        let rectV0 = new Int32Array(rectCap);
        let rectU1 = new Int32Array(rectCap);
        let rectV1 = new Int32Array(rectCap);
        let diagRectCap = 1024;
        let diagRectLen = 0;
        let diagRectBucket = new Int32Array(diagRectCap);
        let diagRectPlane = new Int32Array(diagRectCap);
        let diagRectU0 = new Int32Array(diagRectCap);
        let diagRectE0 = new Int32Array(diagRectCap);
        let diagRectU1 = new Int32Array(diagRectCap);
        let diagRectE1 = new Int32Array(diagRectCap);

        const addRect = (bucket: number, p: number, u0: number, v0: number, u1: number, v1: number): void => {
            if (rectLen === rectCap) {
                rectCap *= 2;
                const grow = (src: Int32Array<ArrayBuffer>): Int32Array<ArrayBuffer> => {
                    const out = new Int32Array(rectCap);
                    out.set(src);
                    return out;
                };
                rectBucket = grow(rectBucket);
                rectP = grow(rectP);
                rectU0 = grow(rectU0);
                rectV0 = grow(rectV0);
                rectU1 = grow(rectU1);
                rectV1 = grow(rectV1);
            }
            rectBucket[rectLen] = bucket;
            rectP[rectLen] = p;
            rectU0[rectLen] = u0;
            rectV0[rectLen] = v0;
            rectU1[rectLen] = u1;
            rectV1[rectLen] = v1;
            rectLen++;
        };

        const addDiagRect = (bucket: number, plane: number, u0: number, e0: number, u1: number, e1: number): void => {
            if (diagRectLen === diagRectCap) {
                diagRectCap *= 2;
                const grow = (src: Int32Array<ArrayBuffer>): Int32Array<ArrayBuffer> => {
                    const out = new Int32Array(diagRectCap);
                    out.set(src);
                    return out;
                };
                diagRectBucket = grow(diagRectBucket);
                diagRectPlane = grow(diagRectPlane);
                diagRectU0 = grow(diagRectU0);
                diagRectE0 = grow(diagRectE0);
                diagRectU1 = grow(diagRectU1);
                diagRectE1 = grow(diagRectE1);
            }
            diagRectBucket[diagRectLen] = bucket;
            diagRectPlane[diagRectLen] = plane;
            diagRectU0[diagRectLen] = u0;
            diagRectE0[diagRectLen] = e0;
            diagRectU1[diagRectLen] = u1;
            diagRectE1[diagRectLen] = e1;
            diagRectLen++;
        };

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

                addRect(bucket, p, uOff - 1, vOff - 1, uOff - 1 + width, vOff - 1 + height);
            }

            start = end;
        }

        if (diagCellLen > 0) {
            const diagKeys = diagCellKeys.slice(0, diagCellLen);
            diagCellKeys = new Float64Array(0);
            diagKeys.sort();

            const decodeDiagKey = (key: number): {
                bucket: number;
                plane: number;
                u: number;
                e: number;
            } => {
                const eOff = key % diagCoordStride;
                let q = Math.floor(key / diagCoordStride);
                const uOff = q % diagCoordStride;
                q = Math.floor(q / diagCoordStride);
                const planeOff = q % diagCoordStride;
                const bucket = Math.floor(q / diagCoordStride);
                return {
                    bucket,
                    plane: planeOff - diagCoordOffset,
                    u: uOff - diagCoordOffset,
                    e: eOff - diagCoordOffset
                };
            };

            let diagStart = 0;
            while (diagStart < diagKeys.length) {
                const first = decodeDiagKey(diagKeys[diagStart]);
                let diagEnd = diagStart + 1;
                while (diagEnd < diagKeys.length) {
                    const next = decodeDiagKey(diagKeys[diagEnd]);
                    if (next.bucket !== first.bucket || next.plane !== first.plane || next.u !== first.u) {
                        break;
                    }
                    diagEnd++;
                }

                let i = diagStart;
                while (i < diagEnd) {
                    const run = decodeDiagKey(diagKeys[i]);
                    let e1 = run.e + 2;
                    i++;
                    while (i < diagEnd) {
                        const next = decodeDiagKey(diagKeys[i]);
                        if (next.e !== e1) break;
                        e1 += 2;
                        i++;
                    }
                    addDiagRect(run.bucket, run.plane, run.u, run.e, run.u + 2, e1);
                }

                diagStart = diagEnd;
            }
        }

        for (let r = 0; r < rectLen; r++) {
            const axis = rectBucket[r] >> 1;
            const p = rectP[r];
            const u0 = rectU0[r];
            const v0 = rectV0[r];
            const u1 = rectU1[r];
            const v1 = rectV1[r];
            const a = scaledFacePoint(axis, p, u0, v0);
            const b = scaledFacePoint(axis, p, u1, v0);
            const c = scaledFacePoint(axis, p, u1, v1);
            const d = scaledFacePoint(axis, p, u0, v1);
            addSplitSegment(a[0], a[1], a[2], b[0], b[1], b[2]);
            addSplitSegment(b[0], b[1], b[2], c[0], c[1], c[2]);
            addSplitSegment(c[0], c[1], c[2], d[0], d[1], d[2]);
            addSplitSegment(d[0], d[1], d[2], a[0], a[1], a[2]);
        }

        for (let r = 0; r < diagRectLen; r++) {
            const bucket = diagRectBucket[r];
            const plane = diagRectPlane[r];
            const u0 = diagRectU0[r];
            const e0 = diagRectE0[r];
            const u1 = diagRectU1[r];
            const e1 = diagRectE1[r];
            const a = diagPoint(bucket, plane, u0, e0);
            const b = diagPoint(bucket, plane, u1, e0);
            const c = diagPoint(bucket, plane, u1, e1);
            const d = diagPoint(bucket, plane, u0, e1);
            addSplitSegment(a[0], a[1], a[2], b[0], b[1], b[2]);
            addSplitSegment(b[0], b[1], b[2], c[0], c[1], c[2]);
            addSplitSegment(c[0], c[1], c[2], d[0], d[1], d[2]);
            addSplitSegment(d[0], d[1], d[2], a[0], a[1], a[2]);
        }

        if (splitLinePoints) {
            for (const points of splitLinePoints.values()) {
                points.sort((a, b) => a - b);
                let write = 0;
                for (let i = 0; i < points.length; i++) {
                    if (i === 0 || points[i] !== points[i - 1]) {
                        points[write++] = points[i];
                    }
                }
                points.length = write;
            }
        }

        collectSplitPoints = false;
        for (let r = 0; r < rectLen; r++) {
            emitFaceRectangle(
                rectBucket[r],
                rectP[r],
                rectU0[r],
                rectV0[r],
                rectU1[r],
                rectV1[r]
            );
        }
        for (let r = 0; r < diagRectLen; r++) {
            emitDiagRectangle(
                diagRectBucket[r],
                diagRectPlane[r],
                diagRectU0[r],
                diagRectE0[r],
                diagRectU1[r],
                diagRectE1[r]
            );
        }
        splitLinePoints?.clear();
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

                        const triRow = TRI_TABLE[cubeIndex]; // eslint-disable-line no-use-before-define
                        const triLen = triRow.length;
                        const usedMask = collectDiagFaces(triRow, vx, vy, vz);
                        let neededEdges = 0;
                        let emitTriLen = 0;
                        for (let t = 0; t < triLen; t += 3) {
                            const triIdx = (t / 3) | 0;
                            if ((usedMask & (1 << triIdx)) !== 0) continue;
                            neededEdges |= (1 << triRow[t]) | (1 << triRow[t + 1]) | (1 << triRow[t + 2]);
                            emitTriLen += 3;
                        }
                        if (neededEdges === 0) continue;

                        // Compute vertices on edges used by triangles that
                        // were not absorbed into a merged binary-MC rectangle.
                        if (neededEdges & 1)    edgeVerts[0]  = getVertex(vx, vy, vz, 0);       // edge 0: x-axis at (vx, vy, vz)
                        if (neededEdges & 2)    edgeVerts[1]  = getVertex(vx + 1, vy, vz, 1);   // edge 1: y-axis at (vx+1, vy, vz)
                        if (neededEdges & 4)    edgeVerts[2]  = getVertex(vx, vy + 1, vz, 0);   // edge 2: x-axis at (vx, vy+1, vz)
                        if (neededEdges & 8)    edgeVerts[3]  = getVertex(vx, vy, vz, 1);       // edge 3: y-axis at (vx, vy, vz)
                        if (neededEdges & 16)   edgeVerts[4]  = getVertex(vx, vy, vz + 1, 0);   // edge 4: x-axis at (vx, vy, vz+1)
                        if (neededEdges & 32)   edgeVerts[5]  = getVertex(vx + 1, vy, vz + 1, 1); // edge 5: y-axis at (vx+1, vy, vz+1)
                        if (neededEdges & 64)   edgeVerts[6]  = getVertex(vx, vy + 1, vz + 1, 0); // edge 6: x-axis at (vx, vy+1, vz+1)
                        if (neededEdges & 128)  edgeVerts[7]  = getVertex(vx, vy, vz + 1, 1);   // edge 7: y-axis at (vx, vy, vz+1)
                        if (neededEdges & 256)  edgeVerts[8]  = getVertex(vx, vy, vz, 2);       // edge 8: z-axis at (vx, vy, vz)
                        if (neededEdges & 512)  edgeVerts[9]  = getVertex(vx + 1, vy, vz, 2);   // edge 9: z-axis at (vx+1, vy, vz)
                        if (neededEdges & 1024) edgeVerts[10] = getVertex(vx + 1, vy + 1, vz, 2); // edge 10: z-axis at (vx+1, vy+1, vz)
                        if (neededEdges & 2048) edgeVerts[11] = getVertex(vx, vy + 1, vz, 2);   // edge 11: z-axis at (vx, vy+1, vz)

                        // Emit triangles (reversed winding to face outward)
                        ensureIndexCapacity(emitTriLen);
                        for (let t = 0; t < triLen; t += 3) {
                            const triIdx = (t / 3) | 0;
                            if ((usedMask & (1 << triIdx)) !== 0) continue;
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
export type { Mesh, MarchingCubesMesh, MarchingCubesOptions };
