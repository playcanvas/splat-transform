import { type FlatKdTree } from '../spatial/kd-tree';

/** Marks an unfilled neighbour slot (fewer than k non-self points available). */
const KNN_SENTINEL = 0xFFFFFFFF;

/** A forest part: flat KD subtree (GLOBAL splat ids) plus its point AABB. */
type ForestPart = FlatKdTree & {
    /** [minx, miny, minz, maxx, maxy, maxz] over the part's points. */
    aabb: Float32Array;
};

/**
 * Exact k-nearest-neighbours of query points against a forest of flat KD
 * subtrees (each part's `nodeSplatIdx` holds GLOBAL splat ids; together the
 * parts cover the whole scene).
 *
 * The top-K state carries across parts: a part whose AABB is farther than
 * the carried worst is skipped outright (the near-path descent is otherwise
 * unconditional, so without this cull every part costs a full descent per
 * query); within a traversed part the standard "skip the far subtree" DFS
 * runs with the accumulated bound. The final top-K equals a single tree's
 * over the union — exact by construction, no halos, no verification.
 *
 * Engine-free (imported by worker tasks); mirrors the GpuKnn WGSL traversal.
 *
 * @param parts - The forest (global splat ids in `nodeSplatIdx`).
 * @param queryPos - Interleaved xyz query positions.
 * @param queryIds - Global splat id per query (self-exclusion).
 * @param count - Query count.
 * @param k - Neighbours per query.
 * @param out - Destination, `count * k` global ids (UNSORTED within a row),
 * {@link KNN_SENTINEL} filling surplus slots.
 */
const knnForestQuery = (
    parts: ForestPart[],
    queryPos: Float32Array,
    queryIds: Uint32Array,
    count: number,
    k: number,
    out: Uint32Array
): void => {
    const topDist = new Float32Array(k);
    const topIdx = new Uint32Array(k);
    const stack = new Uint32Array(48);

    for (let q = 0; q < count; q++) {
        const qx = queryPos[q * 3];
        const qy = queryPos[q * 3 + 1];
        const qz = queryPos[q * 3 + 2];
        const qid = queryIds[q];

        topDist.fill(Infinity);
        topIdx.fill(KNN_SENTINEL);
        let worst = Infinity;
        let worstIdx = 0;

        for (const part of parts) {
            const { nodeSplatIdx, nodePositions, nodeChildren, aabb } = part;

            // Part-entry cull: skip when the AABB cannot improve the top-K.
            const ddx = Math.max(aabb[0] - qx, qx - aabb[3], 0);
            const ddy = Math.max(aabb[1] - qy, qy - aabb[4], 0);
            const ddz = Math.max(aabb[2] - qz, qz - aabb[5], 0);
            if (ddx * ddx + ddy * ddy + ddz * ddz >= worst) continue;
            // Stack packs axis (top 2 bits) with the node index, mirroring
            // the GPU kernel: axis cycles x→y→z with depth.
            let sp = 0;
            stack[sp++] = part.rootIdx;
            while (sp > 0) {
                const packed = stack[--sp];
                const nodeIdx = packed & 0x3FFFFFFF;
                const axis = packed >>> 30;

                const np = nodeIdx * 3;
                const nx = nodePositions[np];
                const ny = nodePositions[np + 1];
                const nz = nodePositions[np + 2];
                const splatId = nodeSplatIdx[nodeIdx];

                if (splatId !== qid) {
                    const dx = nx - qx, dy = ny - qy, dz = nz - qz;
                    const d2 = dx * dx + dy * dy + dz * dz;
                    if (d2 < worst) {
                        topDist[worstIdx] = d2;
                        topIdx[worstIdx] = splatId;
                        let w = topDist[0];
                        let wi = 0;
                        for (let i = 1; i < k; i++) {
                            if (topDist[i] > w) {
                                w = topDist[i]; wi = i;
                            }
                        }
                        worst = w;
                        worstIdx = wi;
                    }
                }

                const qAxisVal = axis === 0 ? qx : (axis === 1 ? qy : qz);
                const nAxisVal = axis === 0 ? nx : (axis === 1 ? ny : nz);
                const delta = qAxisVal - nAxisVal;
                const nextAxis = axis + 1 >= 3 ? 0 : axis + 1;
                const nextAxisPacked = nextAxis << 30;

                const nc = nodeIdx * 2;
                const left = nodeChildren[nc];
                const right = nodeChildren[nc + 1];
                const near = delta < 0 ? left : right;
                const far = delta < 0 ? right : left;

                if (far !== KNN_SENTINEL && delta * delta < worst) {
                    stack[sp++] = far | nextAxisPacked;
                }
                if (near !== KNN_SENTINEL) {
                    stack[sp++] = near | nextAxisPacked;
                }
            }
        }

        out.set(topIdx, q * k);
    }
};

export { knnForestQuery, KNN_SENTINEL, type ForestPart };
