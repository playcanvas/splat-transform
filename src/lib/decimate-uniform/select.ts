/**
 * Global merge selection over the resident candidate arrays: bucketed greedy
 * disjoint matching in cost order, plus chain closure that attaches
 * still-unmatched gaussians to a candidate's group (≤3, relief cap 4) so a
 * 50% target completes in one generation instead of a mop-up pass.
 *
 * Bucket walk ≈ exact cost-sorted greedy up to 1/SELECT_BUCKETS-of-range
 * quantization — the selection-semantics match with the legacy algorithm.
 *
 * Engine-free; pure resident-array computation, no IO.
 */

import { type CandidateArrays } from './priority';

/** Cost-histogram buckets for the ordered greedy walk. */
const SELECT_BUCKETS = 1024;

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

    // Pass 1: finite cost range.
    let lo = Infinity, hi = -Infinity;
    for (let e = 0; e < E; e++) {
        const c = cand.cost[e];
        if (Number.isFinite(c)) {
            if (c < lo) lo = c;
            if (c > hi) hi = c;
        }
    }
    const span = hi > lo ? hi - lo : 1;
    const bucketOf = (c: number) => Math.min(SELECT_BUCKETS - 1, Math.floor(((c - lo) / span) * SELECT_BUCKETS));

    // Counting sort of finite candidate entries by bucket.
    const counts = new Uint32Array(SELECT_BUCKETS + 1);
    for (let e = 0; e < E; e++) {
        if (Number.isFinite(cand.cost[e])) counts[bucketOf(cand.cost[e]) + 1]++;
    }
    for (let b = 0; b < SELECT_BUCKETS; b++) counts[b + 1] += counts[b];
    const orderE = new Uint32Array(counts[SELECT_BUCKETS]);
    const cursor = counts.slice(0, SELECT_BUCKETS);
    for (let e = 0; e < E; e++) {
        if (Number.isFinite(cand.cost[e])) orderE[cursor[bucketOf(cand.cost[e])]++] = e;
    }

    const memberGroup = new Int32Array(N).fill(-1);

    // Primary greedy: pair free endpoints, cheapest bucket first.
    const maxPairs = Math.max(0, mergesNeeded);
    const pairA = new Uint32Array(maxPairs);
    const pairB = new Uint32Array(maxPairs);
    let pairs = 0;
    let removed = 0;
    for (let t = 0; t < orderE.length && removed < mergesNeeded; t++) {
        const e = orderE[t];
        const j = cand.idx[e];
        if (j === NO_CANDIDATE) continue;
        const i = (e / K) | 0;
        if (memberGroup[i] !== -1 || memberGroup[j] !== -1) continue;
        memberGroup[i] = pairs;
        memberGroup[j] = pairs;
        pairA[pairs] = i;
        pairB[pairs] = j;
        pairs++;
        removed++;
    }

    // Chain closure: attach unmatched gaussians to a candidate's group,
    // cheapest first, cap 3 — then a relief walk at cap 4. After the primary
    // walk every free gaussian's candidates are all matched (else the pair
    // would have been taken), so closure can almost always attach.
    const groupSize = new Uint32Array(pairs).fill(2);
    const joinMember = new Uint32Array(Math.max(0, mergesNeeded - removed));
    const joinGroup = new Uint32Array(joinMember.length);
    let joins = 0;
    for (const cap of [3, 4]) {
        if (removed >= mergesNeeded) break;
        for (let t = 0; t < orderE.length && removed < mergesNeeded; t++) {
            const e = orderE[t];
            const j = cand.idx[e];
            if (j === NO_CANDIDATE) continue;
            const i = (e / K) | 0;
            if (memberGroup[i] !== -1) continue;
            const g = memberGroup[j];
            if (g === -1 || groupSize[g] >= cap) continue;
            memberGroup[i] = g;
            groupSize[g]++;
            joinMember[joins] = i;
            joinGroup[joins] = g;
            joins++;
            removed++;
        }
    }

    // CSR assembly.
    const G = pairs;
    const groupOffsets = new Uint32Array(G + 1);
    for (let g = 0; g < G; g++) groupOffsets[g + 1] = groupOffsets[g] + groupSize[g];
    const groupMembers = new Uint32Array(groupOffsets[G]);
    const fill = new Uint32Array(G);
    for (let g = 0; g < G; g++) {
        const o = groupOffsets[g];
        groupMembers[o] = pairA[g];
        groupMembers[o + 1] = pairB[g];
        fill[g] = 2;
    }
    for (let t = 0; t < joins; t++) {
        const g = joinGroup[t];
        groupMembers[groupOffsets[g] + fill[g]] = joinMember[t];
        fill[g]++;
    }
    const groupMin = new Uint32Array(G);
    for (let g = 0; g < G; g++) {
        let min = groupMembers[groupOffsets[g]];
        for (let m = groupOffsets[g] + 1; m < groupOffsets[g + 1]; m++) {
            if (groupMembers[m] < min) min = groupMembers[m];
        }
        groupMin[g] = min;
    }

    return { groupOffsets, groupMembers, memberGroup, groupMin, mergedGroups: G, removed };
};

export { selectMerges, SELECT_BUCKETS, type SelectionResult };
