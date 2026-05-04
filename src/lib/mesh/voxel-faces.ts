import type { Bounds } from '../data-table';
import type { Mesh } from './marching-cubes';
import {
    BLOCK_EMPTY,
    BLOCK_SOLID,
    BLOCKS_PER_WORD,
    EVEN_BITS,
    SparseVoxelGrid,
    readBlockType
} from '../voxel/sparse-voxel-grid';

const HASH_MUL = 0x9E3779B9;

/**
 * Extract a watertight voxel-boundary mesh from a SparseVoxelGrid.
 *
 * Exposed voxel faces are first greedily merged into axis-aligned rectangles.
 * Rectangle boundaries are then split at every collinear rectangle corner
 * before triangulation, so adjacent rectangles share matching edges instead
 * of producing T-junctions.
 *
 * @param grid - Voxel grid after filtering / nav phases.
 * @param gridBounds - Grid bounds aligned to block boundaries.
 * @param voxelResolution - Size of each voxel in world units.
 * @returns Mesh with positions and indices.
 */
const voxelFaces = (
    grid: SparseVoxelGrid,
    gridBounds: Bounds,
    voxelResolution: number
): Mesh => {
    const { nbx, nby, nbz, bStride, types, masks, nx, ny, nz } = grid;
    const totalBlocks = nbx * nby * nbz;
    const coordStride = Math.max(nx, ny, nz) + 1;

    let faceCap = 1024;
    let faceLen = 0;
    let faceKeys = new Float64Array(faceCap);

    const addFace = (bucket: number, p: number, u: number, v: number): void => {
        if (faceLen === faceCap) {
            faceCap *= 2;
            const grown = new Float64Array(faceCap);
            grown.set(faceKeys);
            faceKeys = grown;
        }
        faceKeys[faceLen++] =
            (((bucket * coordStride + p) * coordStride + u) * coordStride + v);
    };

    const blockTypeAt = (bx: number, by: number, bz: number): number => {
        if (bx < 0 || by < 0 || bz < 0 || bx >= nbx || by >= nby || bz >= nbz) {
            return BLOCK_EMPTY;
        }
        return readBlockType(types, bx + by * nbx + bz * bStride);
    };

    const isVoxelSetLocal = (lo: number, hi: number, lx: number, ly: number, lz: number): boolean => {
        const bitIdx = lx + (ly << 2) + (lz << 4);
        return bitIdx < 32 ?
            ((lo >>> bitIdx) & 1) !== 0 :
            ((hi >>> (bitIdx - 32)) & 1) !== 0;
    };

    const isVoxelSetGlobal = (ix: number, iy: number, iz: number): boolean => {
        if (ix < 0 || iy < 0 || iz < 0 || ix >= nx || iy >= ny || iz >= nz) {
            return false;
        }
        const blockIdx = (ix >> 2) + (iy >> 2) * nbx + (iz >> 2) * bStride;
        const bt = readBlockType(types, blockIdx);
        if (bt === BLOCK_EMPTY) return false;
        if (bt === BLOCK_SOLID) return true;
        const s = masks.slot(blockIdx);
        return isVoxelSetLocal(masks.lo[s], masks.hi[s], ix & 3, iy & 3, iz & 3);
    };

    const addVoxelFace = (ix: number, iy: number, iz: number, bucket: number): void => {
        switch (bucket) {
            case 0: addFace(0, ix, iy, iz); break;         // -X
            case 1: addFace(1, ix + 1, iy, iz); break;     // +X
            case 2: addFace(2, iy, ix, iz); break;         // -Y
            case 3: addFace(3, iy + 1, ix, iz); break;     // +Y
            case 4: addFace(4, iz, ix, iy); break;         // -Z
            default: addFace(5, iz + 1, ix, iy); break;    // +Z
        }
    };

    const processSolidBlock = (bx: number, by: number, bz: number): void => {
        const x0 = bx << 2;
        const y0 = by << 2;
        const z0 = bz << 2;

        const emitX = (bucket: number, neighborBlockType: number, ix: number, nx2: number): void => {
            if (neighborBlockType === BLOCK_SOLID) return;
            for (let lz = 0; lz < 4; lz++) {
                const iz = z0 + lz;
                for (let ly = 0; ly < 4; ly++) {
                    const iy = y0 + ly;
                    if (neighborBlockType === BLOCK_EMPTY || !isVoxelSetGlobal(nx2, iy, iz)) {
                        addVoxelFace(ix, iy, iz, bucket);
                    }
                }
            }
        };

        const emitY = (bucket: number, neighborBlockType: number, iy: number, ny2: number): void => {
            if (neighborBlockType === BLOCK_SOLID) return;
            for (let lz = 0; lz < 4; lz++) {
                const iz = z0 + lz;
                for (let lx = 0; lx < 4; lx++) {
                    const ix = x0 + lx;
                    if (neighborBlockType === BLOCK_EMPTY || !isVoxelSetGlobal(ix, ny2, iz)) {
                        addVoxelFace(ix, iy, iz, bucket);
                    }
                }
            }
        };

        const emitZ = (bucket: number, neighborBlockType: number, iz: number, nz2: number): void => {
            if (neighborBlockType === BLOCK_SOLID) return;
            for (let ly = 0; ly < 4; ly++) {
                const iy = y0 + ly;
                for (let lx = 0; lx < 4; lx++) {
                    const ix = x0 + lx;
                    if (neighborBlockType === BLOCK_EMPTY || !isVoxelSetGlobal(ix, iy, nz2)) {
                        addVoxelFace(ix, iy, iz, bucket);
                    }
                }
            }
        };

        emitX(0, blockTypeAt(bx - 1, by, bz), x0, x0 - 1);
        emitX(1, blockTypeAt(bx + 1, by, bz), x0 + 3, x0 + 4);
        emitY(2, blockTypeAt(bx, by - 1, bz), y0, y0 - 1);
        emitY(3, blockTypeAt(bx, by + 1, bz), y0 + 3, y0 + 4);
        emitZ(4, blockTypeAt(bx, by, bz - 1), z0, z0 - 1);
        emitZ(5, blockTypeAt(bx, by, bz + 1), z0 + 3, z0 + 4);
    };

    const processMixedBlock = (blockIdx: number, bx: number, by: number, bz: number): void => {
        const s = masks.slot(blockIdx);
        const lo = masks.lo[s];
        const hi = masks.hi[s];
        const x0 = bx << 2;
        const y0 = by << 2;
        const z0 = bz << 2;

        for (let lz = 0; lz < 4; lz++) {
            const iz = z0 + lz;
            for (let ly = 0; ly < 4; ly++) {
                const iy = y0 + ly;
                for (let lx = 0; lx < 4; lx++) {
                    if (!isVoxelSetLocal(lo, hi, lx, ly, lz)) continue;
                    const ix = x0 + lx;
                    if (!isVoxelSetGlobal(ix - 1, iy, iz)) addVoxelFace(ix, iy, iz, 0);
                    if (!isVoxelSetGlobal(ix + 1, iy, iz)) addVoxelFace(ix, iy, iz, 1);
                    if (!isVoxelSetGlobal(ix, iy - 1, iz)) addVoxelFace(ix, iy, iz, 2);
                    if (!isVoxelSetGlobal(ix, iy + 1, iz)) addVoxelFace(ix, iy, iz, 3);
                    if (!isVoxelSetGlobal(ix, iy, iz - 1)) addVoxelFace(ix, iy, iz, 4);
                    if (!isVoxelSetGlobal(ix, iy, iz + 1)) addVoxelFace(ix, iy, iz, 5);
                }
            }
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

            const bx = blockIdx % nbx;
            const byBz = (blockIdx / nbx) | 0;
            const by = byBz % nby;
            const bz = (byBz / nby) | 0;
            const bt = (word >>> (lane << 1)) & 3;
            if (bt === BLOCK_SOLID) {
                processSolidBlock(bx, by, bz);
            } else {
                processMixedBlock(blockIdx, bx, by, bz);
            }
        }
    }

    if (faceLen === 0) {
        return { positions: new Float32Array(0), indices: new Uint32Array(0) };
    }

    let rectCap = 1024;
    let rectLen = 0;
    let rectBucket = new Int32Array(rectCap);
    let rectP = new Int32Array(rectCap);
    let rectU0 = new Int32Array(rectCap);
    let rectV0 = new Int32Array(rectCap);
    let rectU1 = new Int32Array(rectCap);
    let rectV1 = new Int32Array(rectCap);

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

    const keys = faceKeys.slice(0, faceLen);
    faceKeys = new Float64Array(0);
    keys.sort();

    const decodeGroup = (key: number): { bucket: number; p: number } => {
        let q = Math.floor(key / coordStride);
        q = Math.floor(q / coordStride);
        const p = q % coordStride;
        const bucket = Math.floor(q / coordStride);
        return { bucket, p };
    };

    const decodeUvKey = (key: number): number => {
        const v = key % coordStride;
        const q = Math.floor(key / coordStride);
        const u = q % coordStride;
        return u * coordStride + v;
    };

    let groupStart = 0;
    while (groupStart < keys.length) {
        const { bucket, p } = decodeGroup(keys[groupStart]);
        let groupEnd = groupStart + 1;
        while (groupEnd < keys.length) {
            const g = decodeGroup(keys[groupEnd]);
            if (g.bucket !== bucket || g.p !== p) break;
            groupEnd++;
        }

        const count = groupEnd - groupStart;
        let hCap = 1;
        while (hCap < count / 0.7) hCap *= 2;
        const hMask = hCap - 1;
        const hKeys = new Float64Array(hCap).fill(-1);
        const hVals = new Int32Array(hCap);

        const hash = (key: number): number => {
            const hi = (key / 0x100000000) | 0;
            return (Math.imul((key | 0) ^ hi, HASH_MUL) >>> 0) & hMask;
        };

        for (let i = 0; i < count; i++) {
            const uvKey = decodeUvKey(keys[groupStart + i]);
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
        const uvKeyOf = (u: number, v: number): number => u * coordStride + v;

        for (let i = 0; i < count; i++) {
            if (visited[i]) continue;
            const uvKey = decodeUvKey(keys[groupStart + i]);
            const u0 = Math.floor(uvKey / coordStride);
            const v0 = uvKey % coordStride;

            let width = 1;
            while (true) {
                const idx = lookup(uvKeyOf(u0 + width, v0));
                if (idx === -1 || visited[idx]) break;
                width++;
            }

            let height = 1;
            while (true) {
                let canGrow = true;
                for (let du = 0; du < width; du++) {
                    const idx = lookup(uvKeyOf(u0 + du, v0 + height));
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
                    visited[lookup(uvKeyOf(u0 + du, v0 + dv))] = 1;
                }
            }

            addRect(bucket, p, u0, v0, u0 + width, v0 + height);
        }

        groupStart = groupEnd;
    }

    const globalPoint = (
        axis: number,
        p: number,
        u: number,
        v: number
    ): [number, number, number] => {
        if (axis === 0) return [p, u, v];
        if (axis === 1) return [u, p, v];
        return [u, v, p];
    };

    const lineKey = (varAxis: number, x: number, y: number, z: number): number => {
        if (varAxis === 0) return (y + z * coordStride) * 3;
        if (varAxis === 1) return (x + z * coordStride) * 3 + 1;
        return (x + y * coordStride) * 3 + 2;
    };

    const linePoints = new Map<number, number[]>();

    const addLinePoint = (key: number, value: number): void => {
        let points = linePoints.get(key);
        if (!points) {
            points = [];
            linePoints.set(key, points);
        }
        points.push(value);
    };

    const addLineSegment = (
        x0: number, y0: number, z0: number,
        x1: number, y1: number, z1: number
    ): void => {
        if (x0 !== x1) {
            const key = lineKey(0, x0, y0, z0);
            addLinePoint(key, x0);
            addLinePoint(key, x1);
        } else if (y0 !== y1) {
            const key = lineKey(1, x0, y0, z0);
            addLinePoint(key, y0);
            addLinePoint(key, y1);
        } else {
            const key = lineKey(2, x0, y0, z0);
            addLinePoint(key, z0);
            addLinePoint(key, z1);
        }
    };

    for (let r = 0; r < rectLen; r++) {
        const axis = rectBucket[r] >> 1;
        const p = rectP[r];
        const a = globalPoint(axis, p, rectU0[r], rectV0[r]);
        const b = globalPoint(axis, p, rectU1[r], rectV0[r]);
        const c = globalPoint(axis, p, rectU1[r], rectV1[r]);
        const d = globalPoint(axis, p, rectU0[r], rectV1[r]);
        addLineSegment(a[0], a[1], a[2], b[0], b[1], b[2]);
        addLineSegment(b[0], b[1], b[2], c[0], c[1], c[2]);
        addLineSegment(c[0], c[1], c[2], d[0], d[1], d[2]);
        addLineSegment(d[0], d[1], d[2], a[0], a[1], a[2]);
    }

    for (const points of linePoints.values()) {
        points.sort((a, b) => a - b);
        let write = 0;
        for (let i = 0; i < points.length; i++) {
            if (i === 0 || points[i] !== points[i - 1]) {
                points[write++] = points[i];
            }
        }
        points.length = write;
    }

    let posCap = 1024;
    let posLen = 0;
    let positions = new Float32Array(posCap);
    let idxCap = 1024;
    let idxLen = 0;
    let indices = new Uint32Array(idxCap);
    const vertexMap = new Map<number, number>();
    let perimeterScratch = new Uint32Array(16);
    let perimeterLen = 0;

    const addPosition = (x: number, y: number, z: number): number => {
        if (posLen + 3 > posCap) {
            posCap *= 2;
            const grown = new Float32Array(posCap);
            grown.set(positions);
            positions = grown;
        }
        const idx = posLen / 3;
        positions[posLen++] = gridBounds.min.x + x * voxelResolution;
        positions[posLen++] = gridBounds.min.y + y * voxelResolution;
        positions[posLen++] = gridBounds.min.z + z * voxelResolution;
        return idx;
    };

    const vertexKey = (x: number, y: number, z: number): number => {
        return (x + y * coordStride + z * coordStride * coordStride);
    };

    const getVertex = (x: number, y: number, z: number): number => {
        const key = vertexKey(x, y, z);
        const existing = vertexMap.get(key);
        if (existing !== undefined) return existing;
        const idx = addPosition(x, y, z);
        vertexMap.set(key, idx);
        return idx;
    };

    const addCenterVertex = (axis: number, p: number, u: number, v: number): number => {
        const [x, y, z] = globalPoint(axis, p, u, v);
        return addPosition(x, y, z);
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

    const resetPerimeter = (): void => {
        perimeterLen = 0;
    };

    const addPerimeterVertex = (v: number): void => {
        if (perimeterLen > 0 && perimeterScratch[perimeterLen - 1] === v) return;
        if (perimeterLen === perimeterScratch.length) {
            const grown = new Uint32Array(perimeterScratch.length * 2);
            grown.set(perimeterScratch);
            perimeterScratch = grown;
        }
        perimeterScratch[perimeterLen++] = v;
    };

    const addEdgeVertices = (
        x0: number, y0: number, z0: number,
        x1: number, y1: number, z1: number
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

        const key = lineKey(varAxis, x0, y0, z0);
        const points = linePoints.get(key);
        if (!points) return;

        const lo = Math.min(start, end);
        const hi = Math.max(start, end);
        const forward = start <= end;

        const emitPoint = (t: number): void => {
            if (varAxis === 0) addPerimeterVertex(getVertex(t, y0, z0));
            else if (varAxis === 1) addPerimeterVertex(getVertex(x0, t, z0));
            else addPerimeterVertex(getVertex(x0, y0, t));
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

    for (let r = 0; r < rectLen; r++) {
        const bucket = rectBucket[r];
        const axis = bucket >> 1;
        const positive = (bucket & 1) === 1;
        const p = rectP[r];
        const u0 = rectU0[r];
        const v0 = rectV0[r];
        const u1 = rectU1[r];
        const v1 = rectV1[r];
        const a = globalPoint(axis, p, u0, v0);
        const b = globalPoint(axis, p, u1, v0);
        const c = globalPoint(axis, p, u1, v1);
        const d = globalPoint(axis, p, u0, v1);

        resetPerimeter();
        addEdgeVertices(a[0], a[1], a[2], b[0], b[1], b[2]);
        addEdgeVertices(b[0], b[1], b[2], c[0], c[1], c[2]);
        addEdgeVertices(c[0], c[1], c[2], d[0], d[1], d[2]);
        addEdgeVertices(d[0], d[1], d[2], a[0], a[1], a[2]);
        if (perimeterLen > 1 && perimeterScratch[0] === perimeterScratch[perimeterLen - 1]) {
            perimeterLen--;
        }
        if (perimeterLen < 3) continue;

        const localCcwIsPositive = axis !== 1;
        const useLocalCcw = positive === localCcwIsPositive;

        if (perimeterLen === 4) {
            const p0 = perimeterScratch[0];
            const p1 = perimeterScratch[1];
            const p2 = perimeterScratch[2];
            const p3 = perimeterScratch[3];
            if (useLocalCcw) {
                appendTri(p0, p1, p2);
                appendTri(p0, p2, p3);
            } else {
                appendTri(p0, p2, p1);
                appendTri(p0, p3, p2);
            }
        } else {
            const center = addCenterVertex(axis, p, (u0 + u1) * 0.5, (v0 + v1) * 0.5);
            for (let i = 0; i < perimeterLen; i++) {
                const j = (i + 1) % perimeterLen;
                if (useLocalCcw) appendTri(center, perimeterScratch[i], perimeterScratch[j]);
                else appendTri(center, perimeterScratch[j], perimeterScratch[i]);
            }
        }
    }

    return {
        positions: positions.slice(0, posLen),
        indices: indices.slice(0, idxLen)
    };
};

export { voxelFaces };
