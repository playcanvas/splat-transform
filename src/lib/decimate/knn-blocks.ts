import { knnQueryBlock, KNN_SENTINEL } from './knn-core';
import { type BlockRange, type ResidentPositions } from './partition';

/**
 * A block's local point set: owned gaussians first (in the block's sorted
 * owned order), then halo members from neighbouring blocks. `ids` maps local
 * index → global gaussian index; `positions` is interleaved xyz; `h` is the
 * halo radius the set was collected with.
 */
type BlockLocals = {
    ids: Uint32Array;
    ownedCount: number;
    positions: Float32Array;
    h: number;
};

/** Local-slot marker: neighbour was fixed by verification; resolve via its global id. */
const KNN_FIXED = 0xFFFFFFFE;

// Density-based halo radius: haloFactor × the Poisson estimate of the k-NN
// radius from the block's AABB volume and count. Degenerate blocks (planar /
// coincident points) drive the estimate toward 0 — that's fine, the
// verification pass is the correctness backstop; h is only an efficiency hint.
const haloRadius = (block: BlockRange, k: number, haloFactor: number): number => {
    const nOwned = block.end - block.start;
    if (nOwned === 0) return 0;
    const ex = Math.max(block.aabb[3] - block.aabb[0], 1e-12);
    const ey = Math.max(block.aabb[4] - block.aabb[1], 1e-12);
    const ez = Math.max(block.aabb[5] - block.aabb[2], 1e-12);
    const lambda = nOwned / (ex * ey * ez);
    const rk = Math.cbrt((k * 3) / (4 * Math.PI * lambda));
    return haloFactor * rk;
};

// Squared distance from a point to an AABB ([minx..z, maxx..z]).
const pointAabbDist2 = (px: number, py: number, pz: number, aabb: Float32Array): number => {
    const dx = Math.max(0, aabb[0] - px, px - aabb[3]);
    const dy = Math.max(0, aabb[1] - py, py - aabb[4]);
    const dz = Math.max(0, aabb[2] - pz, pz - aabb[5]);
    return dx * dx + dy * dy + dz * dz;
};

// Squared distance between two AABBs (0 when overlapping).
const aabbAabbDist2 = (a: Float32Array, b: Float32Array): number => {
    let d2 = 0;
    for (let c = 0; c < 3; c++) {
        const gap = Math.max(0, b[c] - a[3 + c], a[c] - b[3 + c]);
        d2 += gap * gap;
    }
    return d2;
};

/**
 * Collect a block's local point set: its owned gaussians plus a halo of
 * points from neighbouring blocks within `h` of the block AABB.
 *
 * @param pos - Resident positions.
 * @param order - Partition index array.
 * @param blocks - All block ranges.
 * @param blockIdx - Which block to collect.
 * @param k - Neighbours per query (drives the halo radius estimate).
 * @param haloFactor - Multiplier on the k-NN radius estimate.
 * @param haloCap - Maximum halo size as a multiple of the owned count (buffer-sizing bound; `h` shrinks until the full halo fits the cap — members are never dropped).
 * @returns The block's locals.
 */
const collectBlock = (
    pos: ResidentPositions,
    order: Uint32Array,
    blocks: BlockRange[],
    blockIdx: number,
    k: number,
    haloFactor: number,
    haloCap = 1
): BlockLocals => {
    const block = blocks[blockIdx];
    const nOwned = block.end - block.start;
    const maxHalo = Math.ceil(nOwned * haloCap);

    const countHalo = (h2: number): number => {
        let count = 0;
        for (let b2 = 0; b2 < blocks.length; b2++) {
            if (b2 === blockIdx) continue;
            const other = blocks[b2];
            if (aabbAabbDist2(block.aabb, other.aabb) > h2) continue;
            for (let i = other.start; i < other.end; i++) {
                const g = order[i];
                if (pointAabbDist2(pos.x[g], pos.y[g], pos.z[g], block.aabb) <= h2) count++;
            }
        }
        return count;
    };

    // The verification rule (`d_k ≤ depth + h`) is only sound when the halo
    // FULLY covers AABB ⊕ h — so the size cap must never truncate members.
    // Instead, shrink h until the full halo fits the cap; points beyond the
    // reduced h are then legitimately outside the covered region and boundary
    // queries fall through to the brute-force requery.
    let h = haloRadius(block, k, haloFactor);
    for (let iter = 0; iter < 40 && h > 0; iter++) {
        if (countHalo(h * h) <= maxHalo) break;
        h *= 0.7;
        if (h < 1e-12) h = 0;
    }
    const h2 = h * h;

    // Halo membership: points of other blocks within h of this block's AABB.
    const haloIds: number[] = [];
    for (let b2 = 0; b2 < blocks.length; b2++) {
        if (b2 === blockIdx) continue;
        const other = blocks[b2];
        if (aabbAabbDist2(block.aabb, other.aabb) > h2) continue;
        for (let i = other.start; i < other.end; i++) {
            const g = order[i];
            if (pointAabbDist2(pos.x[g], pos.y[g], pos.z[g], block.aabb) <= h2) {
                haloIds.push(g);
            }
        }
    }

    const n = nOwned + haloIds.length;
    const ids = new Uint32Array(n);
    const positions = new Float32Array(n * 3);
    for (let i = 0; i < nOwned; i++) {
        const g = order[block.start + i];
        ids[i] = g;
        positions[i * 3] = pos.x[g];
        positions[i * 3 + 1] = pos.y[g];
        positions[i * 3 + 2] = pos.z[g];
    }
    for (let i = 0; i < haloIds.length; i++) {
        const g = haloIds[i];
        const l = nOwned + i;
        ids[l] = g;
        positions[l * 3] = pos.x[g];
        positions[l * 3 + 1] = pos.y[g];
        positions[l * 3 + 2] = pos.z[g];
    }
    return { ids, ownedCount: nOwned, positions, h };
};

/**
 * CPU block KNN: exact k-NN of the owned points within block ∪ halo, as
 * LOCAL indices (see {@link knnQueryBlock}).
 * @param locals - The block's local point set.
 * @param k - Neighbours per query.
 * @returns Local neighbour indices, `ownedCount * k` long.
 */
const knnBlockCpu = (locals: BlockLocals, k: number): Uint32Array => {
    return knnQueryBlock(locals.positions, locals.ownedCount, k);
};

/**
 * Map local neighbour indices to global gaussian indices (sentinels pass
 * through).
 * @param locals - The block's local point set.
 * @param nbLocal - Local neighbour indices.
 * @returns A new array of global neighbour indices.
 */
const toGlobalNeighbors = (locals: BlockLocals, nbLocal: Uint32Array): Uint32Array => {
    const out = new Uint32Array(nbLocal.length);
    for (let s = 0; s < nbLocal.length; s++) {
        out[s] = nbLocal[s] === KNN_SENTINEL ? KNN_SENTINEL : locals.ids[nbLocal[s]];
    }
    return out;
};

// Insert (g, d2) into the k-best arrays (ascending by distance).
const kBestInsert = (bestIds: Uint32Array, bestD2: Float64Array, size: number, k: number, g: number, d2: number): number => {
    if (size === k && d2 >= bestD2[k - 1]) return size;
    let at = size < k ? size : k - 1;
    while (at > 0 && bestD2[at - 1] > d2) {
        bestD2[at] = bestD2[at - 1];
        bestIds[at] = bestIds[at - 1];
        at--;
    }
    bestD2[at] = d2;
    bestIds[at] = g;
    return Math.min(size + 1, k);
};

/**
 * Exactness backstop for block KNN. A query's result is guaranteed correct
 * when its k-th neighbour distance fits inside the halo-covered region
 * (`d_k ≤ depth(q) + h`). Queries that fail — or that carry sentinel slots
 * despite the scene having ≥ k other points — are re-queried by brute force
 * against every block within the k-th distance, making the block KNN
 * globally exact.
 *
 * Fixed entries are written into `nbGlobal`; the matching `nbLocal` slots
 * (when provided) are marked {@link KNN_FIXED} so callers resolve those
 * neighbours by global id.
 *
 * @param pos - Resident positions.
 * @param order - Partition index array.
 * @param blocks - All block ranges.
 * @param blockIdx - Which block was queried.
 * @param locals - The block's local point set.
 * @param k - Neighbours per query.
 * @param nbGlobal - Global neighbour indices (fixed in place).
 * @param nbLocal - Optional parallel local indices to mark.
 * @returns The number of re-queried gaussians.
 */
const verifyAndFixKnn = (
    pos: ResidentPositions,
    order: Uint32Array,
    blocks: BlockRange[],
    blockIdx: number,
    locals: BlockLocals,
    k: number,
    nbGlobal: Uint32Array,
    nbLocal?: Uint32Array
): number => {
    const block = blocks[blockIdx];
    const N = pos.x.length;
    const { h } = locals;
    let fixed = 0;

    const bestIds = new Uint32Array(k);
    const bestD2 = new Float64Array(k);

    for (let qi = 0; qi < locals.ownedCount; qi++) {
        const g = locals.ids[qi];
        const qx = pos.x[g], qy = pos.y[g], qz = pos.z[g];

        let dkSq = 0;
        let sentinels = false;
        for (let s = 0; s < k; s++) {
            const nb = nbGlobal[qi * k + s];
            if (nb === KNN_SENTINEL) {
                sentinels = true;
                continue;
            }
            const dx = pos.x[nb] - qx, dy = pos.y[nb] - qy, dz = pos.z[nb] - qz;
            const d2 = dx * dx + dy * dy + dz * dz;
            if (d2 > dkSq) dkSq = d2;
        }

        // Depth of q inside the block AABB (0 at/outside the boundary).
        const depth = Math.max(0, Math.min(
            qx - block.aabb[0], block.aabb[3] - qx,
            qy - block.aabb[1], block.aabb[4] - qy,
            qz - block.aabb[2], block.aabb[5] - qz
        ));

        const needFix = (sentinels && N - 1 >= k) || Math.sqrt(dkSq) > depth + h;
        if (!needFix) continue;

        // Brute re-query against every block within reach of the current
        // k-th distance (everything, when sentinels made the bound unknown).
        const r2 = sentinels ? Infinity : dkSq;
        let size = 0;
        for (let b2 = 0; b2 < blocks.length; b2++) {
            const other = blocks[b2];
            if (r2 !== Infinity && pointAabbDist2(qx, qy, qz, other.aabb) > r2) continue;
            for (let i = other.start; i < other.end; i++) {
                const cand = order[i];
                if (cand === g) continue;
                const dx = pos.x[cand] - qx, dy = pos.y[cand] - qy, dz = pos.z[cand] - qz;
                const d2 = dx * dx + dy * dy + dz * dz;
                if (size < k || d2 < bestD2[size - 1]) {
                    size = kBestInsert(bestIds, bestD2, size, k, cand, d2);
                }
            }
        }
        for (let s = 0; s < k; s++) {
            nbGlobal[qi * k + s] = s < size ? bestIds[s] : KNN_SENTINEL;
            if (nbLocal) nbLocal[qi * k + s] = s < size ? KNN_FIXED : KNN_SENTINEL;
        }
        fixed++;
    }
    return fixed;
};

export {
    collectBlock,
    knnBlockCpu,
    toGlobalNeighbors,
    verifyAndFixKnn,
    haloRadius,
    KNN_FIXED,
    type BlockLocals
};
