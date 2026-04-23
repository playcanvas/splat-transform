import type { Mesh } from './marching-cubes';

const NORMAL_EPS = 1e-3;
const PLANE_REL_EPS = 1e-3;
const COLLINEAR_REL_EPS = 1e-3;

/**
 * Losslessly reduce coplanar regions of a marching-cubes mesh by
 * topology-preserving vertex removal.
 *
 * For a closed manifold MC mesh, a vertex `v` is "lossless-removable" iff
 * its incident-tri fan, walked in cyclic order, falls into one of:
 *
 * 1. K=1 coplanar fan. Every triangle in v's fan lies on the same plane
 *    (same unit normal and same plane offset, within tolerance). Removing
 *    v is the inverse of vertex split: re-triangulate the boundary
 *    polygon in the same plane.
 *
 * 2. K=2 collinear seam. The fan splits into exactly two contiguous
 *    coplanar arcs (different planes). The two crease vertices `a` and
 *    `b` (the boundary points where the plane changes around v) are
 *    collinear with v in 3D, with v between them. Removing v collapses
 *    the two crease edges (v-a, v-b) into a single straight edge (a-b)
 *    that lies in both planes; each arc's polygon re-triangulates
 *    without v.
 *
 * Vertices with K >= 3 (multi-way corners) are kept.
 *
 * Removing a removable v is exact-lossless: the surface footprint is
 * identical, no vertex moves and none are created. The transformation
 * is the inverse of vertex split, so it is topology-preserving by
 * construction:
 *
 * - No T-junctions. Every old vertex on the polygon boundary remains a
 *   vertex of every triangle that previously touched it. Adjacent fused
 *   regions and verbatim regions stay coupled at every shared vertex.
 * - Watertight. The closed manifold structure is preserved across
 *   removal. Both the K=1 and K=2 cases preserve the K=2 seam edge as
 *   a single shared edge between the two plane groups.
 * - Bit-exact. Every output position is a verbatim copy of an input
 *   position; no vertex is fabricated.
 *
 * Algorithm:
 *
 * 1. Build per-vertex incident-tri lists and per-tri normalized normals
 *    and plane offsets.
 * 2. Process vertices via a dirty-flag worklist. Initially queue every
 *    vertex; after a successful removal, re-queue the ring neighbours
 *    so chains of K=1 / K=2 vertices collapse in one run.
 * 3. For each dequeued vertex `v`:
 *      a. Walk the fan to extract the cyclic ring vertices and the
 *         cyclic ordered tris (each tri (v, ring[i], ring[(i+1)%k]) is
 *         the i-th tri in fan order).
 *      b. Decide K. If all tris share a plane: K=1. Otherwise count
 *         transitions in cyclic order; K=2 if exactly two arcs.
 *      c. K>=3: skip. K=2: verify ring[i1], v, ring[i2] are collinear.
 *      d. For each arc, project its polygon to 2D using the arc's
 *         plane basis, ear-clip, and append the new tris.
 *      e. Mark v's old tris dead, register the new tris in each polygon
 *         vertex's incident list, and re-queue the ring.
 * 4. Compact: drop dead tris and unused vertices, remap indices.
 *
 * @param mesh - Input triangle mesh from {@link marchingCubes}.
 * @param voxelResolution - Size of one voxel in world units. Used to scale the plane-offset tolerance. (The K=2 collinearity check is purely angular and has no voxel-scaled term.)
 * @returns A new mesh with the same surface geometry, no T-junctions, and far fewer triangles.
 */
const coplanarMerge = (mesh: Mesh, voxelResolution: number): Mesh => {
    const { positions, indices } = mesh;
    const inputTriCount = (indices.length / 3) | 0;
    const vertCount = (positions.length / 3) | 0;

    if (inputTriCount === 0) {
        return { positions: new Float32Array(0), indices: new Uint32Array(0) };
    }

    // Plane-offset tolerance for the coplanarity test. An absolute floor
    // (voxelResolution * PLANE_REL_EPS) handles near-origin planes, while
    // a relative term scaled by max(|da|, |db|) handles large-offset
    // planes where Float32 position precision in d = n . p causes
    // relative error proportional to |d|. Without the relative term,
    // coplanar fans far from the origin would be seen as distinct planes.
    const planeEps = (da: number, db: number): number => {
        const absA = da < 0 ? -da : da;
        const absB = db < 0 ? -db : db;
        return PLANE_REL_EPS * (voxelResolution + (absA > absB ? absA : absB));
    };

    // Mutable triangle table. Flat typed arrays so we can append new tris
    // (from ear-clipping) without per-tri allocations and without hitting
    // V8's regular-Array backing-store size cap on large meshes.
    // `triVertsArr` indexes are 3*t, everything else is t.
    //
    // Normals are stored as Float32 (unit vectors; ample precision for the
    // dot-product coplanarity test against `1 - NORMAL_EPS`). The plane
    // offset stays Float64 since it's an absolute world-space scalar that
    // can be large for distant scenes.
    let triCap = inputTriCount;
    let triVertsArr: Uint32Array = new Uint32Array(triCap * 3);
    let triNxArr: Float32Array = new Float32Array(triCap);
    let triNyArr: Float32Array = new Float32Array(triCap);
    let triNzArr: Float32Array = new Float32Array(triCap);
    let triDArr: Float64Array = new Float64Array(triCap);
    let triAliveArr: Uint8Array = new Uint8Array(triCap);
    let triCount = inputTriCount;

    // Capacity-doubling appenders for new tris generated by ear-clipping.
    const ensureTriCap = () => {
        if (triCount < triCap) return;
        const newCap = triCap * 2;
        const growF32 = (src: Float32Array): Float32Array => {
            const out = new Float32Array(newCap);
            out.set(src);
            return out;
        };
        const growF64 = (src: Float64Array): Float64Array => {
            const out = new Float64Array(newCap);
            out.set(src);
            return out;
        };
        const newVerts = new Uint32Array(newCap * 3);
        newVerts.set(triVertsArr);
        triVertsArr = newVerts;
        triNxArr = growF32(triNxArr);
        triNyArr = growF32(triNyArr);
        triNzArr = growF32(triNzArr);
        triDArr = growF64(triDArr);
        const aliveOut = new Uint8Array(newCap);
        aliveOut.set(triAliveArr);
        triAliveArr = aliveOut;
        triCap = newCap;
    };

    // Pass 0: compute per-tri normalized normal and plane offset.
    for (let t = 0; t < inputTriCount; t++) {
        const ia = indices[t * 3];
        const ib = indices[t * 3 + 1];
        const ic = indices[t * 3 + 2];
        triVertsArr[t * 3] = ia;
        triVertsArr[t * 3 + 1] = ib;
        triVertsArr[t * 3 + 2] = ic;

        const ax = positions[ia * 3];
        const ay = positions[ia * 3 + 1];
        const az = positions[ia * 3 + 2];
        const bx = positions[ib * 3];
        const by = positions[ib * 3 + 1];
        const bz = positions[ib * 3 + 2];
        const cx = positions[ic * 3];
        const cy = positions[ic * 3 + 1];
        const cz = positions[ic * 3 + 2];

        const ex = bx - ax, ey = by - ay, ez = bz - az;
        const fx = cx - ax, fy = cy - ay, fz = cz - az;
        let nx = ey * fz - ez * fy;
        let ny = ez * fx - ex * fz;
        let nz = ex * fy - ey * fx;
        const nLen = Math.sqrt(nx * nx + ny * ny + nz * nz);
        if (nLen < 1e-12) {
            // Degenerate input tri; drop it from the active set.
            triAliveArr[t] = 0;
            continue;
        }
        const inv = 1 / nLen;
        nx *= inv; ny *= inv; nz *= inv;
        triNxArr[t] = nx;
        triNyArr[t] = ny;
        triNzArr[t] = nz;
        triDArr[t] = nx * ax + ny * ay + nz * az;
        triAliveArr[t] = 1;
    }

    // Per-vertex incident-tri lists, stored as a singly-linked free-listed
    // node pool backed by typed arrays. Each pool node `n` holds:
    //   poolTri[n]  - the incident tri index
    //   poolNext[n] - next node in the same vertex's list (or -1 = end)
    // `vertHead[v]` is the head node for vertex v (or -1 if v has no
    // incident tris). Removed nodes are pushed onto the `freeHead` chain
    // for reuse, so the pool's max occupancy is the initial fan-mention
    // count (3 * inputTriCount): the worklist's K=1/K=2 collapses are
    // monotonically tri-reducing, so the free list serves all subsequent
    // ear-clip allocations without ever growing the backing arrays.
    //
    // Footprint: 8 B/node * 3 * inputTriCount + 4 B/vert * vertCount.
    // For 25.7M raw tris / 12.9M raw verts that's ~615 MB pool +
    // ~52 MB heads, vs ~1.4 GB for the prior `number[][]` adjacency
    // (V8 Array headers, FixedArray headers, hidden classes per inner
    // array all inflate the boxed-SMI payload).
    const poolCap = inputTriCount * 3;
    const poolTri = new Int32Array(poolCap);
    const poolNext = new Int32Array(poolCap);
    let poolLen = 0;
    let freeHead = -1;
    const vertHead = new Int32Array(vertCount).fill(-1);

    const allocNode = (): number => {
        if (freeHead !== -1) {
            const n = freeHead;
            freeHead = poolNext[n];
            return n;
        }
        return poolLen++;
    };

    const freeNode = (n: number) => {
        poolNext[n] = freeHead;
        freeHead = n;
    };

    const addTriToVert = (v: number, tri: number) => {
        const n = allocNode();
        poolTri[n] = tri;
        poolNext[n] = vertHead[v];
        vertHead[v] = n;
    };

    // O(degree) removal of `tri` from `v`'s incident list. Returns true
    // if found and removed.
    const removeTriFromVert = (v: number, tri: number): boolean => {
        let prev = -1;
        let cur = vertHead[v];
        while (cur !== -1) {
            if (poolTri[cur] === tri) {
                const nxt = poolNext[cur];
                if (prev === -1) vertHead[v] = nxt;
                else poolNext[prev] = nxt;
                freeNode(cur);
                return true;
            }
            prev = cur;
            cur = poolNext[cur];
        }
        return false;
    };

    for (let t = 0; t < inputTriCount; t++) {
        if (triAliveArr[t] === 0) continue;
        addTriToVert(triVertsArr[t * 3], t);
        addTriToVert(triVertsArr[t * 3 + 1], t);
        addTriToVert(triVertsArr[t * 3 + 2], t);
    }

    // Build a right-handed tangent / bitangent basis (t, b) on the plane
    // with normal n, picking the cardinal axis least aligned with n to
    // avoid precision loss in the cross product. By construction
    // t x b = n, so a polygon traced CCW around +n in 3D projects to a
    // CCW polygon in (t, b) coordinates (positive 2D signed area).
    const buildBasis = (nx: number, ny: number, nz: number): [
        number, number, number, number, number, number
    ] => {
        const ax = Math.abs(nx);
        const ay = Math.abs(ny);
        const az = Math.abs(nz);
        let tx: number, ty: number, tz: number;
        if (ax <= ay && ax <= az) {
            // X axis least aligned: t = (1, 0, 0) x n
            tx = 0; ty = -nz; tz = ny;
        } else if (ay <= az) {
            // Y axis least aligned: t = (0, 1, 0) x n
            tx = nz; ty = 0; tz = -nx;
        } else {
            // Z axis least aligned: t = (0, 0, 1) x n
            tx = -ny; ty = nx; tz = 0;
        }
        const tlen = Math.sqrt(tx * tx + ty * ty + tz * tz);
        const inv = 1 / tlen;
        tx *= inv; ty *= inv; tz *= inv;
        // bitangent = n x t (unit length because n and t are unit and
        // perpendicular).
        const bx = ny * tz - nz * ty;
        const by = nz * tx - nx * tz;
        const bz = nx * ty - ny * tx;
        return [tx, ty, tz, bx, by, bz];
    };

    // Reusable scratch buffers for extractFan. With typical MC fan sizes
    // (k <= ~12), a linear scan over parallel typed arrays is cheaper than
    // the Map-of-Map-of-Set allocations the previous version paid per
    // vertex, and saves ~3 allocations per worklist iteration on meshes
    // with tens of millions of vertices.
    let fanScratchCap = 16;
    let fanFromScratch = new Int32Array(fanScratchCap);
    let fanToScratch = new Int32Array(fanScratchCap);
    let fanTriScratch = new Int32Array(fanScratchCap);
    let fanRingScratch = new Int32Array(fanScratchCap);
    let fanTrisScratch = new Int32Array(fanScratchCap);
    const growFanScratch = (need: number) => {
        if (need <= fanScratchCap) return;
        let c = fanScratchCap;
        while (c < need) c *= 2;
        fanFromScratch = new Int32Array(c);
        fanToScratch = new Int32Array(c);
        fanTriScratch = new Int32Array(c);
        fanRingScratch = new Int32Array(c);
        fanTrisScratch = new Int32Array(c);
        fanScratchCap = c;
    };

    // Walk v's fan to extract its cyclic boundary polygon AND the matching
    // cyclic ordered tris. ring[i] is the "from" vertex of the i-th tri
    // (in fan order), and tris[i] = (v, ring[i], ring[(i+1) % k]) is the
    // tri whose two non-v vertices are ring[i] and ring[(i+1) % k].
    // Returns null when the fan is non-manifold (duplicate from-vertex,
    // closes prematurely, or fails to close).
    //
    // Output aliases the module-level `fanRingScratch` / `fanTrisScratch`
    // buffers; the caller must consume them before the next extractFan
    // call. `k` is returned explicitly since the scratch buffers may be
    // oversized.
    const extractFan = (v: number): number => {
        let k = 0;
        for (let n = vertHead[v]; n !== -1; n = poolNext[n]) k++;
        if (k < 3) return -1;
        growFanScratch(k);

        // Collect (from, to, tri) triples into parallel scratch arrays.
        // The O(k^2) duplicate-from scan here and the O(k^2) cyclic walk
        // below are cheap for small k and avoid per-vertex Map allocs.
        let i = 0;
        for (let n = vertHead[v]; n !== -1; n = poolNext[n]) {
            const t = poolTri[n];
            const a = triVertsArr[t * 3];
            const b = triVertsArr[t * 3 + 1];
            const c = triVertsArr[t * 3 + 2];
            let from: number, to: number;
            if (a === v) {
                from = b; to = c;
            } else if (b === v) {
                from = c; to = a;
            } else {
                from = a; to = b;
            }
            for (let j = 0; j < i; j++) {
                if (fanFromScratch[j] === from) return -1;
            }
            fanFromScratch[i] = from;
            fanToScratch[i] = to;
            fanTriScratch[i] = t;
            i++;
        }

        const start = fanFromScratch[0];
        let cur = start;
        for (let step = 0; step < k; step++) {
            fanRingScratch[step] = cur;
            let found = -1;
            for (let j = 0; j < k; j++) {
                if (fanFromScratch[j] === cur) {
                    found = j;
                    break;
                }
            }
            if (found === -1) return -1;
            fanTrisScratch[step] = fanTriScratch[found];
            const next = fanToScratch[found];
            // Premature cycle close => fan is non-manifold (multi-component).
            if (next === start && step < k - 1) return -1;
            cur = next;
        }
        if (cur !== start) return -1;
        return k;
    };

    // Ear-clip a planar simple polygon. `px, py` are the projected 2D
    // coordinates of the polygon vertices (in CCW order). Returns a flat
    // array of (k - 2) * 3 vertex indices into the polygon, or null if
    // the polygon is degenerate / not simple.
    //
    // Strictly-collinear interior vertices (cross product == 0) are not
    // considered convex ear apices and so produce slivers as side
    // vertices of neighbouring ears. These are transient: any such
    // vertex is itself K=1 in the same plane and the worklist removes
    // it in a subsequent iteration, replacing the slivers with a clean
    // re-triangulation of its updated fan. An in-earClip pre-pass that
    // drops the vertex from this polygon would create a T-junction
    // with the vertex's other incident tris (which still reference it),
    // so we leave the cleanup to the worklist.
    const earClip = (px: Float64Array, py: Float64Array): Int32Array | null => {
        const n = px.length;
        if (n < 3) return null;
        if (n === 3) {
            const tri = new Int32Array(3);
            tri[0] = 0; tri[1] = 1; tri[2] = 2;
            return tri;
        }

        // Verify CCW orientation. By construction (right-handed basis)
        // valid input polygons are CCW with positive signed area.
        let area2 = 0;
        for (let i = 0; i < n; i++) {
            const j = (i + 1) % n;
            area2 += px[i] * py[j] - px[j] * py[i];
        }
        if (area2 <= 0) return null;

        const prev = new Int32Array(n);
        const next = new Int32Array(n);
        for (let i = 0; i < n; i++) {
            prev[i] = (i - 1 + n) % n;
            next[i] = (i + 1) % n;
        }

        const isConvex = (a: number, b: number, c: number): boolean => {
            return (px[b] - px[a]) * (py[c] - py[a]) -
                   (py[b] - py[a]) * (px[c] - px[a]) > 0;
        };

        const inTri = (p: number, a: number, b: number, c: number): boolean => {
            const x = px[p], y = py[p];
            const d1 = (x - px[b]) * (py[a] - py[b]) - (px[a] - px[b]) * (y - py[b]);
            const d2 = (x - px[c]) * (py[b] - py[c]) - (px[b] - px[c]) * (y - py[c]);
            const d3 = (x - px[a]) * (py[c] - py[a]) - (px[c] - px[a]) * (y - py[a]);
            const hasNeg = d1 < 0 || d2 < 0 || d3 < 0;
            const hasPos = d1 > 0 || d2 > 0 || d3 > 0;
            return !(hasNeg && hasPos);
        };

        const isEar = (a: number, b: number, c: number): boolean => {
            if (!isConvex(a, b, c)) return false;
            let p = next[c];
            while (p !== a) {
                if (inTri(p, a, b, c)) return false;
                p = next[p];
            }
            return true;
        };

        const result = new Int32Array((n - 2) * 3);
        let resultLen = 0;
        let count = n;
        let i = 0;
        let stalls = 0;

        while (count > 3) {
            if (stalls > count) return null;
            const p = prev[i];
            const nxt = next[i];
            if (isEar(p, i, nxt)) {
                result[resultLen++] = p;
                result[resultLen++] = i;
                result[resultLen++] = nxt;
                next[p] = nxt;
                prev[nxt] = p;
                count--;
                i = nxt;
                stalls = 0;
            } else {
                i = next[i];
                stalls++;
            }
        }
        result[resultLen++] = prev[i];
        result[resultLen++] = i;
        result[resultLen++] = next[i];
        return result;
    };

    // Worklist: iterative dirty-flag scheduler. Initially queue every
    // vertex; on each successful removal, re-queue the ring neighbours so
    // chains of K=1 / K=2 vertices collapse in a single run.
    const inQueue = new Uint8Array(vertCount);
    let queue = new Int32Array(Math.max(vertCount, 16));
    let queueLen = 0;
    let queueHead = 0;
    const pushQueue = (u: number) => {
        if (inQueue[u]) return;
        inQueue[u] = 1;
        if (queueLen >= queue.length) {
            const grown = new Int32Array(queue.length * 2);
            grown.set(queue);
            queue = grown;
        }
        queue[queueLen++] = u;
    };
    const compactQueue = () => {
        // Reclaim consumed prefix when slack exceeds 50% (and is large
        // enough to be worth the copy). Bounded by O(total pushes).
        if (queueHead > 4096 && queueHead * 2 > queueLen) {
            queue.copyWithin(0, queueHead, queueLen);
            queueLen -= queueHead;
            queueHead = 0;
        }
    };
    for (let v = 0; v < vertCount; v++) {
        inQueue[v] = 1;
        queue[queueLen++] = v;
    }

    // Tolerance for K=2 seam collinearity: cosine of the angle between
    // (v -> a) and (v -> b) must be <= -(1 - COLLINEAR_REL_EPS), i.e.
    // the two seam edges through v are nearly antiparallel (v lies on
    // the segment from a to b in 3D).
    const cosineMax = -1 + COLLINEAR_REL_EPS;

    // Reusable scratch for the per-arc plane descriptor.
    const arcStartIdx = new Int32Array(2);
    const arcPolySize = new Int32Array(2);
    const arcPlaneT = new Int32Array(2);

    while (queueHead < queueLen) {
        const v = queue[queueHead++];
        inQueue[v] = 0;
        compactQueue();

        // Cheap "fan size < 3" early-out without traversing the full
        // linked list.
        const h0 = vertHead[v];
        if (h0 === -1) continue;
        const h1 = poolNext[h0];
        if (h1 === -1) continue;
        if (poolNext[h1] === -1) continue;

        const k = extractFan(v);
        if (k === -1) continue;
        const ring = fanRingScratch;
        const fanTris = fanTrisScratch;

        // Decide K. First check if all tris are coplanar with fanTris[0]
        // (K=1 fast path).
        const t0 = fanTris[0];
        const n0x = triNxArr[t0];
        const n0y = triNyArr[t0];
        const n0z = triNzArr[t0];
        const d0 = triDArr[t0];
        let allCoplanar = true;
        for (let i = 1; i < k; i++) {
            const t = fanTris[i];
            const dT = triDArr[t];
            const dotN = triNxArr[t] * n0x + triNyArr[t] * n0y + triNzArr[t] * n0z;
            if (dotN < 1 - NORMAL_EPS || Math.abs(dT - d0) > planeEps(dT, d0)) {
                allCoplanar = false;
                break;
            }
        }

        let arcCount = 0;
        if (allCoplanar) {
            // K=1: single arc covering the whole ring (k vertices).
            arcStartIdx[0] = 0;
            arcPolySize[0] = k;
            arcPlaneT[0] = t0;
            arcCount = 1;
        } else {
            // Walk cyclically; mark transitions where plane changes vs.
            // the previous tri in fan order. Exactly 2 transitions =>
            // K=2 candidate.
            let nTransitions = 0;
            let i1 = -1;
            let i2 = -1;
            for (let i = 0; i < k; i++) {
                const t = fanTris[i];
                const prevT = fanTris[(i - 1 + k) % k];
                const nx1 = triNxArr[t];
                const ny1 = triNyArr[t];
                const nz1 = triNzArr[t];
                const d1 = triDArr[t];
                const nx0 = triNxArr[prevT];
                const ny0 = triNyArr[prevT];
                const nz0 = triNzArr[prevT];
                const dPrev = triDArr[prevT];
                const dotN = nx1 * nx0 + ny1 * ny0 + nz1 * nz0;
                if (dotN < 1 - NORMAL_EPS || Math.abs(d1 - dPrev) > planeEps(d1, dPrev)) {
                    nTransitions++;
                    if (nTransitions === 1) i1 = i;
                    else if (nTransitions === 2) i2 = i;
                    else break;
                }
            }
            if (nTransitions !== 2) continue;

            // K=2 collinearity: ring[i1], v, ring[i2] must be collinear
            // with v between (cosine of va-vb angle near -1).
            const a = ring[i1];
            const b = ring[i2];
            const ax = positions[a * 3] - positions[v * 3];
            const ay = positions[a * 3 + 1] - positions[v * 3 + 1];
            const az = positions[a * 3 + 2] - positions[v * 3 + 2];
            const bx = positions[b * 3] - positions[v * 3];
            const by = positions[b * 3 + 1] - positions[v * 3 + 1];
            const bz = positions[b * 3 + 2] - positions[v * 3 + 2];
            const lenA = Math.sqrt(ax * ax + ay * ay + az * az);
            const lenB = Math.sqrt(bx * bx + by * by + bz * bz);
            if (lenA < 1e-12 || lenB < 1e-12) continue;
            const cosine = (ax * bx + ay * by + az * bz) / (lenA * lenB);
            if (cosine > cosineMax) continue;

            // Two arcs: arc 0 covers tris[i1..i2-1] (mA tris, mA+1 polygon
            // verts ring[i1..i2]); arc 1 covers tris[i2..i1-1] cyclically
            // (mB tris, mB+1 polygon verts).
            const mA = i2 - i1;
            const mB = k - mA;
            arcStartIdx[0] = i1; arcPolySize[0] = mA + 1; arcPlaneT[0] = fanTris[i1];
            arcStartIdx[1] = i2; arcPolySize[1] = mB + 1; arcPlaneT[1] = fanTris[i2];
            arcCount = 2;
        }

        // Build polygons and triangulations. Bail (without committing)
        // if any arc fails to triangulate.
        const arcPolys: Int32Array[] = new Array(arcCount);
        const arcEars: Int32Array[] = new Array(arcCount);
        let allArcsOk = true;
        for (let g = 0; g < arcCount; g++) {
            const polySize = arcPolySize[g];
            const startIdx = arcStartIdx[g];
            const planeT = arcPlaneT[g];
            const poly = new Int32Array(polySize);
            for (let j = 0; j < polySize; j++) {
                poly[j] = ring[(startIdx + j) % k];
            }

            const nx = triNxArr[planeT];
            const ny = triNyArr[planeT];
            const nz = triNzArr[planeT];
            const [tx, ty, tz, bx, by, bz] = buildBasis(nx, ny, nz);
            const px = new Float64Array(polySize);
            const py = new Float64Array(polySize);
            for (let j = 0; j < polySize; j++) {
                const u = poly[j];
                const x = positions[u * 3];
                const y = positions[u * 3 + 1];
                const z = positions[u * 3 + 2];
                px[j] = x * tx + y * ty + z * tz;
                py[j] = x * bx + y * by + z * bz;
            }

            const earIdx = earClip(px, py);
            if (earIdx === null || earIdx.length !== (polySize - 2) * 3) {
                allArcsOk = false;
                break;
            }
            arcPolys[g] = poly;
            arcEars[g] = earIdx;
        }
        if (!allArcsOk) continue;

        // Commit: walk v's linked-list of incident tris in place. For each
        // tri, mark it dead, unlink it from each non-v vertex's list, and
        // free v's own node. Iteration is safe because we only mutate
        // OTHER vertices' lists during the walk.
        let cur = vertHead[v];
        while (cur !== -1) {
            const t = poolTri[cur];
            triAliveArr[t] = 0;
            for (let j = 0; j < 3; j++) {
                const u = triVertsArr[t * 3 + j];
                if (u === v) continue;
                removeTriFromVert(u, t);
            }
            const nxt = poolNext[cur];
            freeNode(cur);
            cur = nxt;
        }
        vertHead[v] = -1;

        // Append new tris per arc, each carrying its arc's plane.
        for (let g = 0; g < arcCount; g++) {
            const planeT = arcPlaneT[g];
            const nx = triNxArr[planeT];
            const ny = triNyArr[planeT];
            const nz = triNzArr[planeT];
            const d = triDArr[planeT];
            const earIdx = arcEars[g];
            const poly = arcPolys[g];
            for (let i = 0; i < earIdx.length; i += 3) {
                const ua = poly[earIdx[i]];
                const ub = poly[earIdx[i + 1]];
                const uc = poly[earIdx[i + 2]];
                ensureTriCap();
                const newT = triCount++;
                triVertsArr[newT * 3] = ua;
                triVertsArr[newT * 3 + 1] = ub;
                triVertsArr[newT * 3 + 2] = uc;
                triNxArr[newT] = nx;
                triNyArr[newT] = ny;
                triNzArr[newT] = nz;
                triDArr[newT] = d;
                triAliveArr[newT] = 1;
                addTriToVert(ua, newT);
                addTriToVert(ub, newT);
                addTriToVert(uc, newT);
            }
        }

        // Re-queue every ring vertex: each may now satisfy K=1 or K=2 in
        // its updated fan, allowing the collapse to propagate along long
        // collinear chains in a single sweep.
        for (let i = 0; i < k; i++) {
            pushQueue(ring[i]);
        }
    }

    // Compact: drop dead tris and unused vertices, remap indices.
    const usedVerts = new Uint8Array(vertCount);
    let outTriCount = 0;
    for (let t = 0; t < triCount; t++) {
        if (triAliveArr[t] === 0) continue;
        outTriCount++;
        usedVerts[triVertsArr[t * 3]] = 1;
        usedVerts[triVertsArr[t * 3 + 1]] = 1;
        usedVerts[triVertsArr[t * 3 + 2]] = 1;
    }

    let outVertCount = 0;
    const vertRemap = new Int32Array(vertCount);
    for (let v = 0; v < vertCount; v++) {
        if (usedVerts[v] === 1) {
            vertRemap[v] = outVertCount++;
        } else {
            vertRemap[v] = -1;
        }
    }

    const outPositions = new Float32Array(outVertCount * 3);
    for (let v = 0; v < vertCount; v++) {
        if (usedVerts[v] === 0) continue;
        const o = vertRemap[v] * 3;
        outPositions[o] = positions[v * 3];
        outPositions[o + 1] = positions[v * 3 + 1];
        outPositions[o + 2] = positions[v * 3 + 2];
    }

    const outIndices = new Uint32Array(outTriCount * 3);
    let oi = 0;
    for (let t = 0; t < triCount; t++) {
        if (triAliveArr[t] === 0) continue;
        outIndices[oi++] = vertRemap[triVertsArr[t * 3]];
        outIndices[oi++] = vertRemap[triVertsArr[t * 3 + 1]];
        outIndices[oi++] = vertRemap[triVertsArr[t * 3 + 2]];
    }

    return { positions: outPositions, indices: outIndices };
};

export { coplanarMerge };
