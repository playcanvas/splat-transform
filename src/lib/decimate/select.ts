/**
 * Global merge selection over the resident candidate arrays.
 *
 * PROTOTYPE — cost-ordered agglomeration (replaces the 50%-matching + chain
 * closure). Candidate edges are walked cheapest-first and unioned (union-find,
 * capped at {@link MAX_GROUP} members) until the generation's target removal
 * count is reached. Because a single-linkage cluster can absorb many members,
 * cheap (redundant) regions collapse deeply while expensive (large / distinct)
 * gaussians stay as singletons — so removal follows local error instead of a
 * uniform 50% everywhere. Combined with the L2 field cost, large distant
 * gaussians (sky) survive far longer than tiny detail.
 *
 * Cost ordering is log-scaled bucketed (the L2 cost spans many orders of
 * magnitude); ties within a bucket resolve in arbitrary order.
 *
 * Engine-free; pure resident-array computation, no IO.
 */

import { type CandidateArrays } from './priority';

/** Log-scaled cost buckets for the ordered agglomeration walk. */
const COST_BUCKETS = 4096;

/** Max gaussians merged into one group in a single generation. */
const MAX_GROUP = 4;

const NO_CANDIDATE = 0xFFFFFFFF;

/**
 * The selected merge groups.
 *
 * Groups are CSR-packed: group g's members are
 * `groupMembers[groupOffsets[g] .. groupOffsets[g+1])` (global gaussian
 * ids). `memberGroup[i]` is gaussian i's group or -1 (survivor).
 * `groupMin[g]` is the group's minimum member id — the merge stream emits
 * each group exactly once, at that member's position in block order.
 * `removed` is the gaussian-count reduction achieved (Σ size-1); callers
 * compare it to `mergesNeeded` and decide stall vs next generation.
 */
type SelectionResult = {
    groupOffsets: Uint32Array;
    groupMembers: Uint32Array;
    memberGroup: Int32Array;
    groupMin: Uint32Array;
    mergedGroups: number;
    removed: number;
};

/**
 * Select merges from the candidate arrays.
 *
 * @param cand - Per-gaussian best-K candidates from the priority pass.
 * @param N - Gaussian count.
 * @param K - Candidates per gaussian.
 * @param mergesNeeded - Target removal count for this generation.
 * @returns The selection.
 */
const selectMerges = (cand: CandidateArrays, N: number, K: number, mergesNeeded: number): SelectionResult => {
    const E = N * K;

    // Finite candidate range (costs are ≥ 0; log-scale the positive ones so
    // cost ordering keeps resolution across the metric's huge dynamic range).
    let hi = 0, loPos = Infinity, anyFinite = false;
    for (let e = 0; e < E; e++) {
        if (cand.idx[e] === NO_CANDIDATE) continue;
        const c = cand.cost[e];
        if (!Number.isFinite(c)) continue;
        anyFinite = true;
        if (c > hi) hi = c;
        if (c > 0 && c < loPos) loPos = c;
    }
    const useLog = anyFinite && Number.isFinite(loPos) && loPos < hi;
    const logLo = useLog ? Math.log(loPos) : 0;
    const logSpan = useLog ? Math.log(hi) - logLo : 1;
    const bucketOf = (c: number): number => {
        if (!(c > 0) || !useLog) return 0;
        const b = 1 + Math.floor(((Math.log(c) - logLo) / logSpan) * (COST_BUCKETS - 2));
        return b < 0 ? 0 : (b >= COST_BUCKETS ? COST_BUCKETS - 1 : b);
    };

    // Counting sort of finite candidate edges by bucket (cheapest first).
    const counts = new Uint32Array(COST_BUCKETS + 1);
    for (let e = 0; e < E; e++) {
        if (cand.idx[e] === NO_CANDIDATE) continue;
        const c = cand.cost[e];
        if (Number.isFinite(c)) counts[bucketOf(c) + 1]++;
    }
    for (let b = 0; b < COST_BUCKETS; b++) counts[b + 1] += counts[b];
    const orderE = new Uint32Array(counts[COST_BUCKETS]);
    const cursor = counts.slice(0, COST_BUCKETS);
    for (let e = 0; e < E; e++) {
        if (cand.idx[e] === NO_CANDIDATE) continue;
        const c = cand.cost[e];
        if (Number.isFinite(c)) orderE[cursor[bucketOf(c)]++] = e;
    }

    // Union-find (union by size + path halving).
    const parent = new Uint32Array(N);
    for (let i = 0; i < N; i++) parent[i] = i;
    const size = new Uint32Array(N).fill(1);
    const find = (x: number): number => {
        while (parent[x] !== x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };

    // Cost-ordered agglomeration: union cheapest edges first, capping group
    // size, until the target removal count is reached.
    let removed = 0;
    for (let t = 0; t < orderE.length && removed < mergesNeeded; t++) {
        const e = orderE[t];
        const j = cand.idx[e];
        if (j === NO_CANDIDATE) continue;
        const i = (e / K) | 0;
        let ri = find(i), rj = find(j);
        if (ri === rj) continue;
        if (size[ri] + size[rj] > MAX_GROUP) continue;
        if (size[ri] < size[rj]) {
            const tmp = ri; ri = rj; rj = tmp;
        }
        parent[rj] = ri;
        size[ri] += size[rj];
        removed++;
    }

    // Assemble CSR groups from the forest (roots with size > 1).
    const memberGroup = new Int32Array(N).fill(-1);
    const rootGroup = new Int32Array(N).fill(-1);
    let G = 0;
    for (let i = 0; i < N; i++) {
        const r = find(i);
        if (size[r] > 1) {
            if (rootGroup[r] < 0) rootGroup[r] = G++;
            memberGroup[i] = rootGroup[r];
        }
    }

    const groupOffsets = new Uint32Array(G + 1);
    for (let i = 0; i < N; i++) {
        const g = memberGroup[i];
        if (g >= 0) groupOffsets[g + 1]++;
    }
    for (let g = 0; g < G; g++) groupOffsets[g + 1] += groupOffsets[g];
    const groupMembers = new Uint32Array(groupOffsets[G]);
    const fill = groupOffsets.slice(0, G);
    for (let i = 0; i < N; i++) {
        const g = memberGroup[i];
        if (g >= 0) groupMembers[fill[g]++] = i;
    }

    // Members were appended in ascending id order, so each group's first slot
    // is its minimum id.
    const groupMin = new Uint32Array(G);
    for (let g = 0; g < G; g++) groupMin[g] = groupMembers[groupOffsets[g]];

    return { groupOffsets, groupMembers, memberGroup, groupMin, mergedGroups: G, removed };
};

export { selectMerges, MAX_GROUP, type SelectionResult };
