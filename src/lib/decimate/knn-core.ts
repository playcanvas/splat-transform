import { KdTree } from '../spatial/kd-tree';

/** Marks an unfilled neighbour slot (fewer than k non-self points available). */
const KNN_SENTINEL = 0xFFFFFFFF;

/**
 * Exact k-nearest-neighbours for the owned prefix of a local point set.
 *
 * Engine-free (imported by worker tasks). Builds a {@link KdTree} over
 * all `n` local points (owned first, then halo) and queries the first
 * `ownedCount`. Output `out[q * k + s]` is a LOCAL index into `positions`,
 * sorted ascending by distance, excluding the query itself, with
 * {@link KNN_SENTINEL} filling surplus slots — the same contract as the
 * legacy CPU KNN loop.
 *
 * @param positions - Interleaved xyz for all local points (owned + halo).
 * @param ownedCount - Number of owned points at the front; only these are queried.
 * @param k - Neighbours per query.
 * @returns Local neighbour indices, `ownedCount * k` long.
 */
const knnQueryBlock = (positions: Float32Array, ownedCount: number, k: number): Uint32Array => {
    const n = positions.length / 3;
    const x = new Float32Array(n);
    const y = new Float32Array(n);
    const z = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        x[i] = positions[i * 3];
        y[i] = positions[i * 3 + 1];
        z[i] = positions[i * 3 + 2];
    }
    const tree = new KdTree([x, y, z]);
    const out = new Uint32Array(ownedCount * k).fill(KNN_SENTINEL);
    const q = new Float32Array(3);
    for (let i = 0; i < ownedCount; i++) {
        q[0] = x[i];
        q[1] = y[i];
        q[2] = z[i];
        // Request k+1 because the tree returns the query itself (distance 0).
        const res = tree.findKNearest(q, k + 1);
        let outPos = 0;
        for (let m = 0; m < res.indices.length && outPos < k; m++) {
            const j = res.indices[m];
            if (j === i) continue;
            out[i * k + outPos] = j;
            outPos++;
        }
    }
    return out;
};

export { knnQueryBlock, KNN_SENTINEL };
