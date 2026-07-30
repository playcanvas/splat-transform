import { type GraphicsDevice } from 'playcanvas';

import {
    bestEdgesForPartition,
    partitionBestOut,
    type RecostState
} from './recost-core';
import { MAX_GROUP, type SelectionResult } from './select';
import { GpuRecost, COMMIT_LOG_STRIDE } from '../gpu/gpu-recost';

const NIL = 0xFFFFFFFF;

/** Quality-critical maximum validated commits between refreshes. */
const WAVE = 4096;

type BlockPlan = {
    /** Interleaved local core roots `(a, b)` in commit order. */
    pairs: Uint32Array;
    /** Marginal costs in commit order. */
    costs: Float32Array;
    frozen: number;
    unfrozen: number;
    diagnostics?: BlockPlanDiagnostics;
};

type BlockPlanDiagnostics = {
    waves: number;
    refreshes: number;
    reverseInvalidations: number;
    heapPops: number;
    staleHeapPops: number;
};

type BlockPlanInputs = {
    splatCache: Float32Array;
    neighbors: Uint32Array;
    D: number;
    coreCount: number;
    totalCount: number;
    device?: GraphicsDevice;
    /** Production multi-block planning sets this: CPU is reference/test only. */
    requireGpu?: boolean;
};

/**
 * Plan every legal core-core commit for one immutable core-plus-halo view.
 * Halo rows participate in eligibility only: if their best cost is no worse
 * than the best core continuation the root is frozen, and no halo row is
 * ever unioned. Reverse adjacency makes that decision dynamic when any
 * referenced core member changes ownership.
 *
 * @param inputs - Local cache/KNN and core boundary.
 * @returns Compact commit sequence and freeze statistics.
 */
const planBlockMerges = async (inputs: BlockPlanInputs): Promise<BlockPlan> => {
    const { splatCache: SC, neighbors, D, coreCount, totalCount: N, device, requireGpu = false } = inputs;
    if (neighbors.length !== N * D) throw new Error('block plan: malformed neighbour rows');
    if (coreCount < 1 || coreCount > N) throw new Error('block plan: invalid core count');

    const parent = new Uint32Array(N);
    const size = new Uint32Array(N);
    const version = new Uint32Array(N);
    const mHead = new Uint32Array(N);
    const mNext = new Uint32Array(N);
    const mTail = new Uint32Array(N);
    const lastSeq = new Uint32Array(coreCount);
    const frozenState = new Uint8Array(coreCount);
    size.fill(1);
    version.fill(1);
    mNext.fill(NIL);
    for (let i = 0; i < N; i++) {
        parent[i] = i;
        mHead[i] = i;
        mTail[i] = i;
    }

    const st: RecostState = {
        SC,
        cands: neighbors,
        D,
        N,
        maxGroup: MAX_GROUP,
        parent,
        size,
        version,
        mHead,
        mNext
    };

    const find = (x0: number): number => {
        let x = x0;
        while (parent[x] !== x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };

    // Fixed reverse adjacency of the generation-input KNN rows. Only core
    // query rows exist; halo rows are immutable and never need refresh roots.
    const reverseOffsets = new Uint32Array(N + 1);
    for (let q = 0; q < coreCount; q++) {
        for (let s = 0; s < D; s++) {
            const c = neighbors[q * D + s];
            if (c !== NIL) reverseOffsets[c + 1]++;
        }
    }
    for (let i = 0; i < N; i++) reverseOffsets[i + 1] += reverseOffsets[i];
    const reverseRows = new Uint32Array(reverseOffsets[N]);
    const reverseFill = reverseOffsets.slice(0, N);
    for (let q = 0; q < coreCount; q++) {
        for (let s = 0; s < D; s++) {
            const c = neighbors[q * D + s];
            if (c !== NIL) reverseRows[reverseFill[c]++] = q;
        }
    }

    let heapCap = Math.ceil(coreCount * 1.25) + 16;
    let hCost = new Float32Array(heapCap);
    let hA = new Uint32Array(heapCap);
    let hB = new Uint32Array(heapCap);
    let hSeq = new Uint32Array(heapCap);
    let hVb = new Uint32Array(heapCap);
    const heapIndex = new Int32Array(coreCount).fill(-1);
    let heapSize = 0;
    let seqCounter = 0;

    const swap = (i: number, j: number): void => {
        let t = hCost[i]; hCost[i] = hCost[j]; hCost[j] = t;
        t = hA[i]; hA[i] = hA[j]; hA[j] = t;
        t = hB[i]; hB[i] = hB[j]; hB[j] = t;
        t = hSeq[i]; hSeq[i] = hSeq[j]; hSeq[j] = t;
        t = hVb[i]; hVb[i] = hVb[j]; hVb[j] = t;
        heapIndex[hA[i]] = i;
        heapIndex[hA[j]] = j;
    };
    const less = (i: number, j: number): boolean => hCost[i] < hCost[j] ||
        (hCost[i] === hCost[j] && (hA[i] < hA[j] || (hA[i] === hA[j] && hB[i] < hB[j])));
    const heapRemoveAt = (at: number): void => {
        heapIndex[hA[at]] = -1;
        heapSize--;
        if (at === heapSize) return;
        hCost[at] = hCost[heapSize]; hA[at] = hA[heapSize]; hB[at] = hB[heapSize];
        hSeq[at] = hSeq[heapSize]; hVb[at] = hVb[heapSize];
        heapIndex[hA[at]] = at;
        let i = at;
        if (i > 0 && less(i, (i - 1) >> 1)) {
            while (i > 0) {
                const p = (i - 1) >> 1;
                if (!less(i, p)) break;
                swap(i, p);
                i = p;
            }
        } else {
            for (;;) {
                const l = i * 2 + 1;
                const r = l + 1;
                let m = i;
                if (l < heapSize && less(l, m)) m = l;
                if (r < heapSize && less(r, m)) m = r;
                if (m === i) break;
                swap(i, m);
                i = m;
            }
        }
    };
    const heapPush = (cost: number, a: number, b: number, seq: number, vb: number): void => {
        const existing = heapIndex[a];
        if (existing >= 0) heapRemoveAt(existing);
        if (heapSize === heapCap) {
            const next = heapCap * 2;
            const nextCost = new Float32Array(next); nextCost.set(hCost); hCost = nextCost;
            const nextA = new Uint32Array(next); nextA.set(hA); hA = nextA;
            const nextB = new Uint32Array(next); nextB.set(hB); hB = nextB;
            const nextSeq = new Uint32Array(next); nextSeq.set(hSeq); hSeq = nextSeq;
            const nextVb = new Uint32Array(next); nextVb.set(hVb); hVb = nextVb;
            heapCap = next;
        }
        let i = heapSize++;
        hCost[i] = cost; hA[i] = a; hB[i] = b; hSeq[i] = seq; hVb[i] = vb;
        heapIndex[a] = i;
        while (i > 0) {
            const p = (i - 1) >> 1;
            if (!less(i, p)) break;
            swap(i, p);
            i = p;
        }
    };
    const popped = { cost: 0, a: 0, b: 0, seq: 0, vb: 0 };
    const heapPop = (): boolean => {
        if (heapSize === 0) return false;
        popped.cost = hCost[0]; popped.a = hA[0]; popped.b = hB[0]; popped.seq = hSeq[0]; popped.vb = hVb[0];
        heapRemoveAt(0);
        return true;
    };

    const pending = new Uint32Array(coreCount);
    const queuedRound = new Uint32Array(coreCount);
    let pendingCount = coreCount;
    let round = 1;
    const queueRefresh = (root0: number): void => {
        const root = find(root0);
        if (root >= coreCount) return;
        if (queuedRound[root] === round) return;
        lastSeq[root] = ++seqCounter;
        const existing = heapIndex[root];
        if (existing >= 0) heapRemoveAt(existing);
        queuedRound[root] = round;
        pending[pendingCount++] = root;
    };
    for (let i = 0; i < coreCount; i++) {
        pending[i] = i;
        queuedRound[i] = round;
        lastSeq[i] = ++seqCounter;
    }

    const gpuFits = !!device && D * MAX_GROUP === 64 && GpuRecost.fits(device, N, D, WAVE, coreCount, true);
    if (requireGpu && !gpuFits) {
        throw new Error(
            'multi-block adaptive decimation requires a WebGPU block working set that fits the adapter limits ' +
            `(core ${coreCount}, halo ${N - coreCount})`
        );
    }
    const gpu = gpuFits ? new GpuRecost(device!, N, D, MAX_GROUP, WAVE, coreCount) : undefined;
    const commitLog = gpu ? new Uint32Array(WAVE * COMMIT_LOG_STRIDE) : undefined;
    const outBest = gpu ? new Uint32Array(coreCount * gpu.outputStride) : undefined;
    const outCost = gpu ? new Float32Array(outBest!.buffer) : undefined;

    const planPairs: number[] = [];
    const planCosts: number[] = [];
    let frozen = 0;
    let unfrozen = 0;
    let waves = 0;
    let refreshes = 0;
    let reverseInvalidations = 0;
    let heapPops = 0;
    let staleHeapPops = 0;

    const refreshResult = (root: number, corePartner: number, coreCost: number, haloPartner: number, haloCost: number): void => {
        const isFrozen = haloPartner !== NIL && haloCost <= coreCost;
        if (isFrozen) {
            if (frozenState[root] === 0) frozen++;
            frozenState[root] = 1;
            return;
        }
        if (frozenState[root] !== 0) unfrozen++;
        frozenState[root] = 0;
        if (corePartner !== NIL && Number.isFinite(coreCost)) {
            heapPush(coreCost, root, corePartner, lastSeq[root], version[corePartner]);
        }
    };

    try {
        gpu?.init(SC, neighbors);
        for (;;) {
            let wave = 0;
            while (wave < WAVE && heapPop()) {
                heapPops++;
                const a = popped.a;
                if (parent[a] !== a || popped.seq !== lastSeq[a]) {
                    staleHeapPops++;
                    continue;
                }
                const b = popped.b;
                if (b === a || b >= coreCount || parent[b] !== b ||
                    version[b] !== popped.vb || size[a] + size[b] > MAX_GROUP) {
                    staleHeapPops++;
                    queueRefresh(a);
                    continue;
                }

                const keep = size[a] >= size[b] ? a : b;
                const lose = keep === a ? b : a;
                const loseHeap = heapIndex[lose];
                if (loseHeap >= 0) heapRemoveAt(loseHeap);
                if (commitLog) {
                    const o = wave * COMMIT_LOG_STRIDE;
                    commitLog[o] = lose;
                    commitLog[o + 1] = keep;
                    commitLog[o + 2] = mTail[keep];
                    commitLog[o + 3] = mHead[lose];
                    commitLog[o + 4] = size[keep] + size[lose];
                }
                mNext[mTail[keep]] = mHead[lose];
                mTail[keep] = mTail[lose];
                parent[lose] = keep;
                size[keep] += size[lose];
                version[keep]++;
                planPairs.push(a, b);
                planCosts.push(popped.cost);
                wave++;

                queueRefresh(keep);
                // Any root whose fixed pool references a changed member may
                // have a different core/halo minimum, including frozen roots.
                for (let m = mHead[keep]; m !== NIL; m = mNext[m]) {
                    for (let r = reverseOffsets[m]; r < reverseOffsets[m + 1]; r++) {
                        reverseInvalidations++;
                        queueRefresh(reverseRows[r]);
                    }
                }
            }

            if (pendingCount === 0 && heapSize === 0) break;
            if (pendingCount > 0) {
                waves++;
                refreshes += pendingCount;
            }
            if (gpu && pendingCount > 0) {
                await gpu.wave(commitLog!, wave, pending, pendingCount, outBest!);
                for (let p = 0; p < pendingCount; p++) {
                    const root = pending[p];
                    if (parent[root] !== root) continue;
                    const o = p * gpu.outputStride;
                    refreshResult(
                        root,
                        outBest![o],
                        outCost![o + 1],
                        gpu.outputStride === 4 ? outBest![o + 2] : NIL,
                        gpu.outputStride === 4 ? outCost![o + 3] : Infinity
                    );
                }
            } else {
                for (let p = 0; p < pendingCount; p++) {
                    const root = pending[p];
                    if (parent[root] !== root) continue;
                    bestEdgesForPartition(st, root, coreCount);
                    refreshResult(
                        root,
                        partitionBestOut.corePartner < 0 ? NIL : partitionBestOut.corePartner,
                        partitionBestOut.coreCost,
                        partitionBestOut.haloPartner < 0 ? NIL : partitionBestOut.haloPartner,
                        partitionBestOut.haloCost
                    );
                }
            }
            pendingCount = 0;
            round++;
        }
    } finally {
        gpu?.destroy();
    }

    return {
        pairs: Uint32Array.from(planPairs),
        costs: Float32Array.from(planCosts),
        frozen,
        unfrozen,
        diagnostics: {
            waves,
            refreshes,
            reverseInvalidations,
            heapPops,
            staleHeapPops
        }
    };
};

/**
 * Replay a selected prefix into the standard CSR selection shape, local to
 * one core. Only this block-sized structure exists during output.
 *
 * @param coreCount - Core row count.
 * @param plan - Selected plan prefix.
 * @returns Local selection.
 */
const replayBlockPlan = (coreCount: number, plan: BlockPlan): SelectionResult => {
    const parent = new Uint32Array(coreCount);
    const size = new Uint32Array(coreCount).fill(1);
    for (let i = 0; i < coreCount; i++) parent[i] = i;
    const find = (x0: number): number => {
        let x = x0;
        while (parent[x] !== x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    for (let i = 0; i < plan.costs.length; i++) {
        const a = find(plan.pairs[i * 2]);
        const b = find(plan.pairs[i * 2 + 1]);
        if (a === b || size[a] + size[b] > MAX_GROUP) throw new Error('invalid decimation block plan replay');
        const keep = size[a] >= size[b] ? a : b;
        const lose = keep === a ? b : a;
        parent[lose] = keep;
        size[keep] += size[lose];
    }

    const memberGroup = new Int32Array(coreCount).fill(-1);
    const rootGroup = new Int32Array(coreCount).fill(-1);
    let groups = 0;
    for (let i = 0; i < coreCount; i++) {
        const root = find(i);
        if (size[root] > 1) {
            if (rootGroup[root] < 0) rootGroup[root] = groups++;
            memberGroup[i] = rootGroup[root];
        }
    }
    const groupOffsets = new Uint32Array(groups + 1);
    for (let i = 0; i < coreCount; i++) {
        const group = memberGroup[i];
        if (group >= 0) groupOffsets[group + 1]++;
    }
    for (let g = 0; g < groups; g++) groupOffsets[g + 1] += groupOffsets[g];
    const groupMembers = new Uint32Array(groupOffsets[groups]);
    const fill = groupOffsets.slice(0, groups);
    for (let i = 0; i < coreCount; i++) {
        const group = memberGroup[i];
        if (group >= 0) groupMembers[fill[group]++] = i;
    }
    const groupMin = new Uint32Array(groups);
    for (let g = 0; g < groups; g++) groupMin[g] = groupMembers[groupOffsets[g]];
    return {
        groupOffsets,
        groupMembers,
        memberGroup,
        groupMin,
        mergedGroups: groups,
        removed: plan.costs.length
    };
};

export {
    planBlockMerges,
    replayBlockPlan,
    WAVE,
    type BlockPlan,
    type BlockPlanDiagnostics,
    type BlockPlanInputs
};
