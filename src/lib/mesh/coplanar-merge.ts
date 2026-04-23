import type { Mesh } from './marching-cubes';

const NORMAL_EPS = 1e-3;
const DIAG_COMP = Math.SQRT1_2;
const ENCODING_STRIDE = 1 << 21;
const ENCODING_BIAS = 1 << 20;
const ENCODING_MASK = ENCODING_STRIDE - 1;

interface AxisCellTri {
    tri: number;      // triangle index in the input mesh
    mask: number;     // 4-bit corner-coverage mask (bit 0=u_lo,v_lo, 1=u_hi,v_lo,
                      // 2=u_lo,v_hi, 3=u_hi,v_hi)
}

interface AxisCellEntry {
    tris: AxisCellTri[]; // per-tri records contributed to this (bucket, cell)
}

interface AxisBucket {
    cells: Map<number, AxisCellEntry>; // encoded cell -> per-tri records
    fusedCells: Set<number>;           // populated post-validation
    minU: number;
    maxU: number;
    minV: number;
    maxV: number;
}

interface DiagCellTri {
    tri: number;      // triangle index in the input mesh
    mask: number;     // 4-bit corner-coverage mask for this tri (A_lo|A_hi|B_lo|B_hi)
}

interface DiagCellEntry {
    tris: DiagCellTri[]; // per-tri records contributed to this (bucket, cellE)
}

interface DiagBucket {
    pair: number;     // 0 = XY (edge Z), 1 = YZ (edge X), 2 = XZ (edge Y)
    su: number;       // -1 or +1, sign along the bucket's first in-plane axis
    sv: number;       // -1 or +1, sign along the bucket's second in-plane axis
    cellU: number;    // cell index along the first in-plane cardinal axis
    cellV: number;    // cell index along the second in-plane cardinal axis
    cells: Map<number, DiagCellEntry>; // cellE -> tri indices and corner coverage
    edgeCells: Set<number>; // populated post-validation; cells with all 4 corners covered
}

/**
 * Pack two signed cell coordinates into a single non-negative number suitable
 * for storage in a {@link Set}. Each axis is biased and packed into 21 bits,
 * so the addressable cell range is +/-1M cells per axis.
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
 * Losslessly fuse coplanar marching-cubes triangle islands into greedy-style
 * quads. Two families are merged:
 *
 * 1. Axis-aligned face triangles (normal +/-X, +/-Y, +/-Z): bucketed by
 *    `(axis, sign, planeIndex)`, then meshed with a 2D row-then-column
 *    greedy rectangle pass over the per-bucket occupancy mask.
 * 2. Axis-diagonal "edge bevel" triangles whose normal lies in one of the 12
 *    families `(s_u, 0, s_v)`, `(s_u, s_v, 0)` or `(0, s_u, s_v)` with
 *    `s_u, s_v in {-1, +1}`: bucketed by `(pair, signPair, cellU, cellV)`,
 *    then run-length compressed along the edge axis (the cardinal axis with
 *    zero normal component).
 *
 * Anything else (corner-cap bevels with normals `(+/-1, +/-1, +/-1)/sqrt(3)`
 * or non-canonical inputs) falls through to a verbatim emission queue.
 *
 * Crack analysis:
 *
 * - Axis-aligned quad corners live on the integer voxel grid.
 * - Diagonal quad corners and bevel-triangle vertices live on a half-voxel
 *   grid (edge midpoints).
 * - The two grids never collide under the welder's `round(coord * 2 / r)`
 *   key, so axis-aligned quads cannot accidentally weld with bevel/diagonal
 *   vertices, and the surface stays watertight.
 * - Diagonal runs only fuse along a single cardinal axis (the third axis is
 *   per-bucket constant). MC bevel cells on the same plane but in different
 *   `(cellU, cellV)` columns are geometrically disconnected, so they end up
 *   in different buckets and are never merged across a gap.
 * - A few MC cube configurations (e.g. cubeIndex 29) emit triangles whose
 *   normal is axis-diagonal but whose vertices do NOT match the canonical
 *   2-tri wedge layout `{A_lo, A_hi, B_lo, B_hi}` for any cell, or supply
 *   only one half of the expected 2-tri wedge (the other half is suppressed
 *   by the cube's other corners), or supply two tris that collectively
 *   cover all 4 corners but share a SIDE of the wedge parallelogram rather
 *   than a diagonal (a "butterfly" arrangement that covers a different
 *   planar region than the canonical wedge). To prevent the fuser from
 *   emitting a quad shifted by half a voxel from the original surface,
 *   each diagonal-classified triangle is validated in two stages:
 *     1. Per-tri: all 3 vertices must lie on the bucket+cell's expected
 *        4-corner set (within a 0.25*r tolerance). Rogue tris fail here.
 *     2. Per-cell: the cell must contain exactly 2 tris whose corner
 *        masks union to 0xF and intersect in 0x9 (A_lo|B_hi) or 0x6
 *        (A_hi|B_lo) - i.e. share a wedge diagonal. Half-wedges and
 *        butterflies fail here.
 *   Anything that fails either check is routed to the verbatim queue.
 * - The exact same hazard exists for axis-aligned face triangles: configs
 *   like cubeIndex 31 emit a single triangle covering 3 of the 4 cell
 *   corners on a half-voxel plane (the surface bends into an adjacent
 *   axis-diagonal wedge over the missing corner). Naively marking the cell
 *   occupied would emit a full 1x1 quad - fabricating the missing corner
 *   vertex and doubling the surface area. Axis-aligned tris therefore go
 *   through the same two-stage validation as diagonals: per-tri vertex
 *   check against the cell's 4 corners (bits 0=u_lo,v_lo, 1=u_hi,v_lo,
 *   2=u_lo,v_hi, 3=u_hi,v_hi), and per-cell requirement of exactly 2 tris
 *   with combined mask 0xF and intersection 0x9 or 0x6 (shared cell
 *   diagonal). Half-cell tris and same-side pairs fall through to the
 *   verbatim queue.
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

    const r = voxelResolution;
    const planeStep = r * 0.5;
    const invPlaneStep = 1 / planeStep;
    const invVoxelRes = 1 / r;
    const cellEps = 1e-6;

    const axisBuckets = new Map<string, AxisBucket>();
    const diagBuckets = new Map<string, DiagBucket>();
    const passThroughTris: number[] = [];

    // Pass 1: classify triangles, bucket axis-aligned and axis-diagonal cells,
    // queue everything else for verbatim emission.
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

        // Axis-aligned classification.
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

        if (axis !== -1) {
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

            // Per-tri vertex validation: a canonical face-quad half-tri has
            // all 3 vertices at distinct corners of one unit cell. Tris
            // spanning multiple cells (non-canonical input) or with vertices
            // off the cell-corner grid can't be safely fused; route them
            // verbatim so they print at exact MC positions.
            const tol = r * 0.25;
            if (cellU1 - cellU0 !== 1 || cellV1 - cellV0 !== 1) {
                passThroughTris.push(t);
                continue;
            }
            const uLoW = cellU0 * r;
            const uHiW = (cellU0 + 1) * r;
            const vLoW = cellV0 * r;
            const vHiW = (cellV0 + 1) * r;
            // Bit layout: 0=u_lo,v_lo  1=u_hi,v_lo  2=u_lo,v_hi  3=u_hi,v_hi
            const cornerBitAxis = (u: number, v: number): number => {
                const onULo = Math.abs(u - uLoW) < tol;
                const onUHi = Math.abs(u - uHiW) < tol;
                const onVLo = Math.abs(v - vLoW) < tol;
                const onVHi = Math.abs(v - vHiW) < tol;
                if (onULo && onVLo) return 0x1;
                if (onUHi && onVLo) return 0x2;
                if (onULo && onVHi) return 0x4;
                if (onUHi && onVHi) return 0x8;
                return 0;
            };
            const bitU = cornerBitAxis(u0, v0);
            const bitV = cornerBitAxis(u1, v1);
            const bitW = cornerBitAxis(u2, v2);
            if (bitU === 0 || bitV === 0 || bitW === 0) {
                passThroughTris.push(t);
                continue;
            }

            const key = `${axis}:${sign}:${planeIdx}`;
            let bucket = axisBuckets.get(key);
            if (!bucket) {
                bucket = {
                    cells: new Map(),
                    fusedCells: new Set(),
                    minU: cellU0,
                    maxU: cellU0,
                    minV: cellV0,
                    maxV: cellV0
                };
                axisBuckets.set(key, bucket);
            } else {
                if (cellU0 < bucket.minU) bucket.minU = cellU0;
                if (cellU0 > bucket.maxU) bucket.maxU = cellU0;
                if (cellV0 < bucket.minV) bucket.minV = cellV0;
                if (cellV0 > bucket.maxV) bucket.maxV = cellV0;
            }

            const cellEncoded = encodeCell(cellU0, cellV0);
            let entry = bucket.cells.get(cellEncoded);
            if (!entry) {
                entry = { tris: [] };
                bucket.cells.set(cellEncoded, entry);
            }
            entry.tris.push({ tri: t, mask: bitU | bitV | bitW });
            continue;
        }

        // Axis-diagonal classification: one component near zero, the other
        // two near +/- 1/sqrt(2). Identifies the "edge bevel" family.
        let pair = -1;
        if (aax < NORMAL_EPS &&
            Math.abs(aay - DIAG_COMP) < NORMAL_EPS &&
            Math.abs(aaz - DIAG_COMP) < NORMAL_EPS) {
            pair = 1; // YZ, edge axis = X
        } else if (aay < NORMAL_EPS &&
                   Math.abs(aax - DIAG_COMP) < NORMAL_EPS &&
                   Math.abs(aaz - DIAG_COMP) < NORMAL_EPS) {
            pair = 2; // XZ, edge axis = Y
        } else if (aaz < NORMAL_EPS &&
                   Math.abs(aax - DIAG_COMP) < NORMAL_EPS &&
                   Math.abs(aay - DIAG_COMP) < NORMAL_EPS) {
            pair = 0; // XY, edge axis = Z
        }

        if (pair === -1) {
            passThroughTris.push(t);
            continue;
        }

        let su = 0, sv = 0;
        let uA = 0, uB = 0, uC = 0;
        let vA = 0, vB = 0, vC = 0;
        let eA = 0, eB = 0, eC = 0;
        if (pair === 0) {
            su = inx > 0 ? 1 : -1;
            sv = iny > 0 ? 1 : -1;
            uA = ax; uB = bx; uC = cx;
            vA = ay; vB = by; vC = cy;
            eA = az; eB = bz; eC = cz;
        } else if (pair === 1) {
            su = iny > 0 ? 1 : -1;
            sv = inz > 0 ? 1 : -1;
            uA = ay; uB = by; uC = cy;
            vA = az; vB = bz; vC = cz;
            eA = ax; eB = bx; eC = cx;
        } else {
            su = inx > 0 ? 1 : -1;
            sv = inz > 0 ? 1 : -1;
            uA = ax; uB = bx; uC = cx;
            vA = az; vB = bz; vC = cz;
            eA = ay; eB = by; eC = cy;
        }

        // Cell coords are derived from the minimum vertex coord in each
        // in-plane axis. For canonical MC bevels, the per-axis vertex range
        // is exactly half a cell, so floor(min * invVoxelRes) recovers the
        // owning cell index for both negative and positive sign cases.
        const minU = Math.min(uA, uB, uC);
        const minV = Math.min(vA, vB, vC);
        const minE = Math.min(eA, eB, eC);
        const cellU = Math.floor(minU * invVoxelRes + cellEps);
        const cellV = Math.floor(minV * invVoxelRes + cellEps);
        const cellE = Math.round(minE * invVoxelRes);

        // Wedge-membership validation (per-tri half). The fuser assumes the
        // triangle is one half of a 2-tri wedge in cell `(cellU, cellV)`
        // along the edge axis at `cellE`, with vertices drawn from the
        // 4-point set
        //   { (Au, Av, e_lo), (Au, Av, e_hi), (Bu, Bv, e_lo), (Bu, Bv, e_hi) }
        // where Au/Av/Bu/Bv are the canonical bevel-vertex offsets for the
        // bucket's signs. Some MC configurations (cubeIndex 29 and friends)
        // emit triangles with a diagonal normal whose vertices do NOT lie on
        // that set; if we bucketed them they would fuse into a quad offset
        // by half a voxel. Reject any tri whose vertices don't all match.
        // Bit layout of cornerMask: 0=A_lo, 1=A_hi, 2=B_lo, 3=B_hi.
        const aV = sv === -1 ? 1 : 0;
        const bU = su === -1 ? 1 : 0;
        const expAu = (cellU + 0.5) * r;
        const expAv = (cellV + aV)  * r;
        const expBu = (cellU + bU)  * r;
        const expBv = (cellV + 0.5) * r;
        const expELo = cellE       * r;
        const expEHi = (cellE + 1) * r;
        const tol = r * 0.25;
        const cornerBit = (u: number, v: number, e: number): number => {
            const onA = Math.abs(u - expAu) < tol && Math.abs(v - expAv) < tol;
            const onB = Math.abs(u - expBu) < tol && Math.abs(v - expBv) < tol;
            if (!onA && !onB) return 0;
            const lo = Math.abs(e - expELo) < tol;
            const hi = Math.abs(e - expEHi) < tol;
            if (!lo && !hi) return 0;
            if (onA) return lo ? 0x1 : 0x2;
            return lo ? 0x4 : 0x8;
        };
        const bitA = cornerBit(uA, vA, eA);
        const bitB = cornerBit(uB, vB, eB);
        const bitC = cornerBit(uC, vC, eC);
        if (bitA === 0 || bitB === 0 || bitC === 0) {
            passThroughTris.push(t);
            continue;
        }

        const dKey = `${pair}:${su}:${sv}:${cellU}:${cellV}`;
        let dBucket = diagBuckets.get(dKey);
        if (!dBucket) {
            dBucket = {
                pair,
                su,
                sv,
                cellU,
                cellV,
                cells: new Map(),
                edgeCells: new Set()
            };
            diagBuckets.set(dKey, dBucket);
        }
        let entry = dBucket.cells.get(cellE);
        if (!entry) {
            entry = { tris: [] };
            dBucket.cells.set(cellE, entry);
        }
        entry.tris.push({ tri: t, mask: bitA | bitB | bitC });
    }

    // Pass 1.4: per-cell axis-aligned face validation. A canonical face
    // quad is exactly 2 tris covering all 4 cell corners and sharing a cell
    // diagonal (intersection mask 0x9 = u_lo,v_lo|u_hi,v_hi or 0x6 =
    // u_hi,v_lo|u_lo,v_hi). Cells that don't match (single half-cell tri,
    // butterfly pairs sharing a side, etc.) get all their tris routed to
    // the verbatim queue so the greedy mesher only operates on cells where
    // emitting a full quad reproduces the original surface.
    for (const bucket of axisBuckets.values()) {
        for (const [cellEncoded, entry] of bucket.cells) {
            const tris = entry.tris;
            let isCanonicalFace = false;
            if (tris.length === 2) {
                const m0 = tris[0].mask;
                const m1 = tris[1].mask;
                if ((m0 | m1) === 0xF) {
                    const shared = m0 & m1;
                    if (shared === 0x9 || shared === 0x6) {
                        isCanonicalFace = true;
                    }
                }
            }
            if (isCanonicalFace) {
                bucket.fusedCells.add(cellEncoded);
            } else {
                for (const t of tris) {
                    passThroughTris.push(t.tri);
                }
            }
        }
    }

    // Pass 1.5: per-cell wedge validation. A canonical edge-bevel wedge is
    // exactly 2 tris that cover all 4 corners {A_lo, A_hi, B_lo, B_hi} and
    // share a diagonal of the wedge parallelogram (A_lo-B_hi or A_hi-B_lo).
    // Cells that fail any of:
    //   - exactly 2 tris
    //   - combined corner mask == 0xF (full coverage, half-wedges fail here)
    //   - intersection mask == 0x9 (A_lo|B_hi) or 0x6 (A_hi|B_lo) (shared
    //     diagonal, "butterfly" pairs sharing a side fail here)
    // get all their tris routed to the verbatim queue. Without the diagonal
    // check butterflies would still fuse into the canonical 4-corner quad,
    // covering a different planar region than the original tris and leaving
    // visible internal-diagonal artifacts on bevel surfaces.
    for (const dBucket of diagBuckets.values()) {
        for (const [cellE, entry] of dBucket.cells) {
            const tris = entry.tris;
            let isCanonicalWedge = false;
            if (tris.length === 2) {
                const m0 = tris[0].mask;
                const m1 = tris[1].mask;
                if ((m0 | m1) === 0xF) {
                    const shared = m0 & m1;
                    if (shared === 0x9 || shared === 0x6) {
                        isCanonicalWedge = true;
                    }
                }
            }
            if (isCanonicalWedge) {
                dBucket.edgeCells.add(cellE);
            } else {
                for (const t of tris) {
                    passThroughTris.push(t.tri);
                }
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

    // Pass 2a: greedy-mesh each axis-aligned bucket and emit fused quads.
    axisBuckets.forEach((bucket, key) => {
        const parts = key.split(':');
        const axis = parseInt(parts[0], 10);
        const sign = parseInt(parts[1], 10);
        const planeIdx = parseInt(parts[2], 10);
        const planeCoord = planeIdx * planeStep;

        if (bucket.fusedCells.size === 0) return;

        const minU = bucket.minU;
        const minV = bucket.minV;
        const U = bucket.maxU - bucket.minU + 1;
        const V = bucket.maxV - bucket.minV + 1;
        const mask = new Uint8Array(U * V);
        bucket.fusedCells.forEach((c) => {
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
                const u0w = cu0 * r;
                const u1w = cu1 * r;
                const v0w = cv0 * r;
                const v1w = cv1 * r;

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

    // Pass 2b: 1D run-length fuse each axis-diagonal bucket and emit one
    // fused quad per maximal contiguous run along the bucket's edge axis.
    //
    // Per-cell bevel geometry. A bevel with outward normal `(s_u, s_v)` in
    // its (u, v) plane sits on a cell whose only solid corner (in that
    // plane projection) is in the `(-s_u, -s_v)` direction. The two MC
    // edge midpoints adjacent to that solid corner are:
    //   A = (cellU + 0.5,                 cellV + (s_v == -1 ? 1 : 0))
    //   B = (cellU + (s_u == -1 ? 1 : 0), cellV + 0.5)
    // a "U-edge" and "V-edge" midpoint respectively. A run from edge cell
    // [e0, e1) emits one quad with corners {A_e0, B_e0, B_e1, A_e1}
    // (or its reverse, depending on handedness; see the AB/BA selector).
    //
    // Handedness selector: cross(B - A, +e_axis_unit) gives a normal of
    // (s_v, s_u, ...) up to a per-pair handedness sign. For the
    // right-handed pairs XY (e=Z) and YZ (e=X) the cross product matches
    // the outward normal `(s_u, s_v, 0)` when s_u == s_v (so AB winding
    // wins). For the left-handed pair XZ (e=Y) the cross product matches
    // when s_u != s_v. Packed as `(handedness * s_u * s_v) > 0`.
    diagBuckets.forEach((bucket) => {
        const { pair, su, sv, cellU, cellV, edgeCells } = bucket;

        const sortedCells = [...edgeCells].sort((a, b) => a - b);
        const handedness = pair === 2 ? -1 : 1;
        const useAB = (handedness * su * sv) > 0;

        // Cell-relative half-voxel offsets for the two characteristic
        // vertex positions in the (u, v) plane.
        const aU = 0.5;
        const aV = sv === -1 ? 1 : 0;
        const bU = su === -1 ? 1 : 0;
        const bV = 0.5;

        const Au = (cellU + aU) * r;
        const Av = (cellV + aV) * r;
        const Bu = (cellU + bU) * r;
        const Bv = (cellV + bV) * r;

        let i = 0;
        while (i < sortedCells.length) {
            let j = i + 1;
            while (j < sortedCells.length && sortedCells[j] === sortedCells[j - 1] + 1) {
                j++;
            }

            const e0 = sortedCells[i] * r;
            const e1 = (sortedCells[j - 1] + 1) * r;

            // Resolve (u, v, e) coords back to (x, y, z) for the chosen pair.
            // Inlined per-pair to avoid an allocation in the inner loop.
            let x0: number, y0: number, z0: number;
            let x1: number, y1: number, z1: number;
            let x2: number, y2: number, z2: number;
            let x3: number, y3: number, z3: number;

            if (useAB) {
                if (pair === 0) {
                    x0 = Au; y0 = Av; z0 = e0;
                    x1 = Bu; y1 = Bv; z1 = e0;
                    x2 = Bu; y2 = Bv; z2 = e1;
                    x3 = Au; y3 = Av; z3 = e1;
                } else if (pair === 1) {
                    x0 = e0; y0 = Au; z0 = Av;
                    x1 = e0; y1 = Bu; z1 = Bv;
                    x2 = e1; y2 = Bu; z2 = Bv;
                    x3 = e1; y3 = Au; z3 = Av;
                } else {
                    x0 = Au; y0 = e0; z0 = Av;
                    x1 = Bu; y1 = e0; z1 = Bv;
                    x2 = Bu; y2 = e1; z2 = Bv;
                    x3 = Au; y3 = e1; z3 = Av;
                }
            } else if (pair === 0) {
                x0 = Bu; y0 = Bv; z0 = e0;
                x1 = Au; y1 = Av; z1 = e0;
                x2 = Au; y2 = Av; z2 = e1;
                x3 = Bu; y3 = Bv; z3 = e1;
            } else if (pair === 1) {
                x0 = e0; y0 = Bu; z0 = Bv;
                x1 = e0; y1 = Au; z1 = Av;
                x2 = e1; y2 = Au; z2 = Av;
                x3 = e1; y3 = Bu; z3 = Bv;
            } else {
                x0 = Bu; y0 = e0; z0 = Bv;
                x1 = Au; y1 = e0; z1 = Av;
                x2 = Au; y2 = e1; z2 = Av;
                x3 = Bu; y3 = e1; z3 = Bv;
            }

            emitQuad(
                x0, y0, z0,
                x1, y1, z1,
                x2, y2, z2,
                x3, y3, z3
            );

            i = j;
        }
    });

    // Pass 3: pass-through triangles (corner caps and any non-canonical
    // input) emitted verbatim, with vertices welded.
    for (let i = 0; i < passThroughTris.length; i++) {
        const t = passThroughTris[i];
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
