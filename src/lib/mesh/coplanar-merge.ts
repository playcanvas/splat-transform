import type { Mesh } from './marching-cubes';

const NORMAL_EPS = 1e-3;
const ENCODING_STRIDE = 1 << 21;
const ENCODING_BIAS = 1 << 20;
const ENCODING_MASK = ENCODING_STRIDE - 1;

interface Bucket {
    cells: Set<number>;
    minU: number;
    maxU: number;
    minV: number;
    maxV: number;
}

/**
 * Pack two signed cell coordinates into a single non-negative number suitable
 * for storage in a {@link Set}. Each axis is biased and packed into 21 bits,
 * so the addressable cell range is ±1M cells per axis.
 *
 * @param cu - Signed cell coordinate along the bucket's first in-plane axis.
 * @param cv - Signed cell coordinate along the bucket's second in-plane axis.
 * @returns A non-negative number uniquely identifying the (cu, cv) cell.
 */
const encodeCell = (cu: number, cv: number): number => {
    return (cu + ENCODING_BIAS) * ENCODING_STRIDE + (cv + ENCODING_BIAS);
};

const decodeU = (e: number): number => Math.floor(e / ENCODING_STRIDE) - ENCODING_BIAS;
const decodeV = (e: number): number => (e % ENCODING_STRIDE) - ENCODING_BIAS;

/**
 * Losslessly fuse coplanar axis-aligned triangle islands in a marching-cubes
 * mesh into greedy-style quads. Bevel triangles (those whose normal is not
 * cardinally aligned) pass through unchanged.
 *
 * Background: marching cubes on binary occupancy emits the interior of every
 * exposed voxel face as 1 or 2 tiny coplanar triangles whose face normal is
 * exactly ±X / ±Y / ±Z. The corner-cutting "bevel" triangles between
 * differently-oriented faces have non-axis-aligned normals. The flat
 * interiors dominate the triangle count by orders of magnitude on real
 * scenes; replacing them with maximal axis-aligned rectangles produces an
 * identical surface with vastly fewer triangles, and lets us skip a costly
 * generic simplification pass entirely.
 *
 * Algorithm:
 *
 * 1. Single pass over the input triangles. Compute each face normal and
 *    classify the triangle:
 *    - If one normal component is ~±1 the triangle is axis-aligned. Bucket
 *      it by `(axis, sign, planeIndex)` and record which unit voxel-face
 *      cell it covers in the bucket's 2D cell set.
 *    - Otherwise it's a bevel; queue its triangle index for verbatim
 *      emission later.
 * 2. Per bucket, allocate a {@link Uint8Array} mask sized to the bucket's
 *    2D AABB and run the same row-then-column greedy rectangle pass used
 *    by {@link greedyVoxelMesh}. Emit each rectangle as two triangles with
 *    the winding required for the bucket's outward sign.
 * 3. Emit the queued bevel triangles unchanged.
 * 4. All emitted vertices are routed through a position-keyed welder so
 *    fused quads, neighbouring fused quads, and bevel triangles share
 *    indices wherever they share a position.
 *
 * Crack analysis: every MC vertex sits on a half-voxel grid, and bevel
 * triangles only ever share a single point with an axis-aligned plane (the
 * bevel's two non-coplanar vertices guarantee no shared edge with the
 * fused rectangles), so the fused output is crack-free for valid MC input.
 *
 * @param mesh - Input triangle mesh from {@link marchingCubes}.
 * @param voxelResolution - Size of one voxel in world units. Used as the cell stride for plane bucketing and vertex welding.
 * @returns A new mesh with the same surface geometry and far fewer triangles.
 */
const coplanarMerge = (mesh: Mesh, voxelResolution: number): Mesh => {
    const { positions, indices } = mesh;
    const triCount = (indices.length / 3) | 0;

    if (triCount === 0) {
        return { positions: new Float32Array(0), indices: new Uint32Array(0) };
    }

    const planeStep = voxelResolution * 0.5;
    const invPlaneStep = 1 / planeStep;
    const invVoxelRes = 1 / voxelResolution;

    const buckets = new Map<string, Bucket>();
    const bevelTris: number[] = [];

    // Pass 1: classify triangles, bucket axis-aligned cells, queue bevels.
    for (let t = 0; t < triCount; t++) {
        const ia = indices[t * 3] * 3;
        const ib = indices[t * 3 + 1] * 3;
        const ic = indices[t * 3 + 2] * 3;

        const ax = positions[ia], ay = positions[ia + 1], az = positions[ia + 2];
        const bx = positions[ib], by = positions[ib + 1], bz = positions[ib + 2];
        const cx = positions[ic], cy = positions[ic + 1], cz = positions[ic + 2];

        const ex = bx - ax, ey = by - ay, ez = bz - az;
        const fx = cx - ax, fy = cy - ay, fz = cz - az;
        const nx = ey * fz - ez * fy;
        const ny = ez * fx - ex * fz;
        const nz = ex * fy - ey * fx;
        const nLen = Math.sqrt(nx * nx + ny * ny + nz * nz);
        if (nLen < NORMAL_EPS) continue;

        const inx = nx / nLen;
        const iny = ny / nLen;
        const inz = nz / nLen;

        const aax = Math.abs(inx);
        const aay = Math.abs(iny);
        const aaz = Math.abs(inz);

        let axis = -1;
        let sign = 0;
        if (aax > 1 - NORMAL_EPS) {
            axis = 0;
            sign = inx > 0 ? 1 : 0;
        } else if (aay > 1 - NORMAL_EPS) {
            axis = 1;
            sign = iny > 0 ? 1 : 0;
        } else if (aaz > 1 - NORMAL_EPS) {
            axis = 2;
            sign = inz > 0 ? 1 : 0;
        }

        if (axis === -1) {
            bevelTris.push(t);
            continue;
        }

        const planeCoord = axis === 0 ? ax : axis === 1 ? ay : az;
        const planeIdx = Math.round(planeCoord * invPlaneStep);

        let u0: number, v0: number, u1: number, v1: number, u2: number, v2: number;
        if (axis === 0) {
            u0 = ay; v0 = az; u1 = by; v1 = bz; u2 = cy; v2 = cz;
        } else if (axis === 1) {
            u0 = ax; v0 = az; u1 = bx; v1 = bz; u2 = cx; v2 = cz;
        } else {
            u0 = ax; v0 = ay; u1 = bx; v1 = by; u2 = cx; v2 = cy;
        }

        const minU = Math.min(u0, u1, u2);
        const maxU = Math.max(u0, u1, u2);
        const minV = Math.min(v0, v1, v2);
        const maxV = Math.max(v0, v1, v2);
        const cellU0 = Math.round(minU * invVoxelRes);
        const cellU1 = Math.round(maxU * invVoxelRes);
        const cellV0 = Math.round(minV * invVoxelRes);
        const cellV1 = Math.round(maxV * invVoxelRes);

        // For canonical MC binary input, axis-aligned tris cover exactly
        // one unit cell (cellU1 - cellU0 === 1 and likewise for V). The
        // loop is defensive for non-canonical inputs that span multiple
        // cells: every covered cell is marked occupied.
        const uHi = Math.max(cellU1, cellU0 + 1);
        const vHi = Math.max(cellV1, cellV0 + 1);

        const key = `${axis}:${sign}:${planeIdx}`;
        let bucket = buckets.get(key);
        if (!bucket) {
            bucket = {
                cells: new Set(),
                minU: cellU0,
                maxU: uHi - 1,
                minV: cellV0,
                maxV: vHi - 1
            };
            buckets.set(key, bucket);
        } else {
            if (cellU0 < bucket.minU) bucket.minU = cellU0;
            if (uHi - 1 > bucket.maxU) bucket.maxU = uHi - 1;
            if (cellV0 < bucket.minV) bucket.minV = cellV0;
            if (vHi - 1 > bucket.maxV) bucket.maxV = vHi - 1;
        }

        for (let cu = cellU0; cu < uHi; cu++) {
            for (let cv = cellV0; cv < vHi; cv++) {
                bucket.cells.add(encodeCell(cu, cv));
            }
        }
    }

    // Output buffers (capacity-doubling typed arrays) plus a position-keyed
    // welder. Quantising to the half-voxel grid is exactly the resolution
    // at which both fused-quad corners and bevel vertices are placed.
    let posCap = 1024;
    let posLen = 0;
    let outPositions = new Float32Array(posCap);
    let idxCap = 1024;
    let idxLen = 0;
    let outIndices = new Uint32Array(idxCap);

    const ensurePos = (need: number) => {
        if (posLen + need <= posCap) return;
        while (posLen + need > posCap) posCap *= 2;
        const grown = new Float32Array(posCap);
        grown.set(outPositions);
        outPositions = grown;
    };
    const ensureIdx = (need: number) => {
        if (idxLen + need <= idxCap) return;
        while (idxLen + need > idxCap) idxCap *= 2;
        const grown = new Uint32Array(idxCap);
        grown.set(outIndices);
        outIndices = grown;
    };

    const vertexMap = new Map<string, number>();
    const getOrAddVertex = (x: number, y: number, z: number): number => {
        const ix = Math.round(x * invPlaneStep);
        const iy = Math.round(y * invPlaneStep);
        const iz = Math.round(z * invPlaneStep);
        const key = `${ix}_${iy}_${iz}`;
        const existing = vertexMap.get(key);
        if (existing !== undefined) return existing;
        ensurePos(3);
        const idx = posLen / 3;
        outPositions[posLen++] = x;
        outPositions[posLen++] = y;
        outPositions[posLen++] = z;
        vertexMap.set(key, idx);
        return idx;
    };

    const emitQuad = (
        x0: number, y0: number, z0: number,
        x1: number, y1: number, z1: number,
        x2: number, y2: number, z2: number,
        x3: number, y3: number, z3: number
    ) => {
        const i0 = getOrAddVertex(x0, y0, z0);
        const i1 = getOrAddVertex(x1, y1, z1);
        const i2 = getOrAddVertex(x2, y2, z2);
        const i3 = getOrAddVertex(x3, y3, z3);
        ensureIdx(6);
        outIndices[idxLen++] = i0;
        outIndices[idxLen++] = i1;
        outIndices[idxLen++] = i2;
        outIndices[idxLen++] = i0;
        outIndices[idxLen++] = i2;
        outIndices[idxLen++] = i3;
    };

    // Pass 2: greedy-mesh each bucket and emit fused quads.
    buckets.forEach((bucket, key) => {
        const parts = key.split(':');
        const axis = parseInt(parts[0], 10);
        const sign = parseInt(parts[1], 10);
        const planeIdx = parseInt(parts[2], 10);
        const planeCoord = planeIdx * planeStep;

        const minU = bucket.minU;
        const minV = bucket.minV;
        const U = bucket.maxU - bucket.minU + 1;
        const V = bucket.maxV - bucket.minV + 1;
        const mask = new Uint8Array(U * V);
        bucket.cells.forEach((c) => {
            const cu = decodeU(c);
            const cv = decodeV(c);
            mask[(cu - minU) + (cv - minV) * U] = 1;
        });

        for (let v = 0; v < V; v++) {
            for (let u = 0; u < U; u++) {
                if (mask[u + v * U] !== 1) continue;

                let w = 1;
                while (u + w < U && mask[(u + w) + v * U] === 1) w++;

                let h = 1;
                while (v + h < V) {
                    let rowFull = true;
                    for (let du = 0; du < w; du++) {
                        if (mask[(u + du) + (v + h) * U] !== 1) {
                            rowFull = false;
                            break;
                        }
                    }
                    if (!rowFull) break;
                    h++;
                }

                for (let dv = 0; dv < h; dv++) {
                    for (let du = 0; du < w; du++) {
                        mask[(u + du) + (v + dv) * U] = 2;
                    }
                }

                const cu0 = u + minU;
                const cv0 = v + minV;
                const cu1 = cu0 + w;
                const cv1 = cv0 + h;
                const u0w = cu0 * voxelResolution;
                const u1w = cu1 * voxelResolution;
                const v0w = cv0 * voxelResolution;
                const v1w = cv1 * voxelResolution;

                if (axis === 0) {
                    if (sign === 1) {
                        emitQuad(
                            planeCoord, u0w, v0w,
                            planeCoord, u1w, v0w,
                            planeCoord, u1w, v1w,
                            planeCoord, u0w, v1w
                        );
                    } else {
                        emitQuad(
                            planeCoord, u0w, v0w,
                            planeCoord, u0w, v1w,
                            planeCoord, u1w, v1w,
                            planeCoord, u1w, v0w
                        );
                    }
                } else if (axis === 1) {
                    if (sign === 1) {
                        emitQuad(
                            u0w, planeCoord, v0w,
                            u0w, planeCoord, v1w,
                            u1w, planeCoord, v1w,
                            u1w, planeCoord, v0w
                        );
                    } else {
                        emitQuad(
                            u0w, planeCoord, v0w,
                            u1w, planeCoord, v0w,
                            u1w, planeCoord, v1w,
                            u0w, planeCoord, v1w
                        );
                    }
                } else if (sign === 1) {
                    emitQuad(
                        u0w, v0w, planeCoord,
                        u1w, v0w, planeCoord,
                        u1w, v1w, planeCoord,
                        u0w, v1w, planeCoord
                    );
                } else {
                    emitQuad(
                        u0w, v0w, planeCoord,
                        u0w, v1w, planeCoord,
                        u1w, v1w, planeCoord,
                        u1w, v0w, planeCoord
                    );
                }
            }
        }
    });

    // Pass 3: bevel triangles pass through verbatim, with vertices welded.
    for (let i = 0; i < bevelTris.length; i++) {
        const t = bevelTris[i];
        const ia = indices[t * 3] * 3;
        const ib = indices[t * 3 + 1] * 3;
        const ic = indices[t * 3 + 2] * 3;
        const i0 = getOrAddVertex(positions[ia], positions[ia + 1], positions[ia + 2]);
        const i1 = getOrAddVertex(positions[ib], positions[ib + 1], positions[ib + 2]);
        const i2 = getOrAddVertex(positions[ic], positions[ic + 1], positions[ic + 2]);
        ensureIdx(3);
        outIndices[idxLen++] = i0;
        outIndices[idxLen++] = i1;
        outIndices[idxLen++] = i2;
    }

    return {
        positions: outPositions.subarray(0, posLen),
        indices: outIndices.subarray(0, idxLen)
    };
};

export { coplanarMerge };
