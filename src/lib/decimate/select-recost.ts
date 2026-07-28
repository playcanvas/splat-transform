/**
 * Re-costed merge selection: exact greedy agglomeration within a generation,
 * parallelized as commit/refresh rounds.
 *
 * Where {@link selectMerges} consumes the priority pass's pairwise costs
 * one-shot (costs go stale as groups form), this selection re-evaluates a
 * cluster's candidates after it changes, so merges always execute at costs
 * evaluated against current state. To make the (dominant) evaluation work
 * parallel, refreshes are batched into rounds:
 *
 *   1. Drain the heap, committing every entry that is still valid; clusters
 *      that changed (merged) or whose best partner changed are queued.
 *   2. Bulk-evaluate all queued clusters' best edges — on the worker pool
 *      when the state is SharedArrayBuffer-backed, inline otherwise — and
 *      push the results.
 *   3. Repeat until the generation target is reached or no merges remain.
 *
 * Relative to strictly-eager greedy, refreshed clusters re-enter the heap at
 * round boundaries instead of immediately; commits still only execute at
 * costs validated against the exact current state (version checks), so the
 * deviation is confined to near-tie ordering (re-certified on the evaluation
 * harness).
 *
 * Cost definition and evaluation kernel live in {@link recost-core} (shared
 * with the worker task). Seeds come from the priority pass's candidate
 * arrays; refresh candidate pools come from the persisted neighbour graph.
 *
 * decimate-source gates this path by memory budget (~330 B per gaussian
 * resident) and falls back to {@link selectMerges} when over.
 *
 * Engine-free; pure resident-array computation, no IO.
 */

import { type CandidateArrays } from './priority';
import { evalMergeCore, bestEdgeFor, evalOut, bestOut, NO_CANDIDATE, CACHE_STRIDE, type RecostState } from './recost-core';
import { MAX_GROUP, type SelectionResult } from './select';
import { WorkerQueue } from '../workers';

const NIL = 0xFFFFFFFF;

/** Inputs for {@link selectMergesRecosted}. */
type RecostInputs = {
    /** Per-gaussian best-K candidates from the priority pass (seed costs include the colour term). */
    cand: CandidateArrays;
    /** Candidates per gaussian. */
    K: number;
    /**
     * Resident per-splat cache, {@link CACHE_STRIDE} floats per splat
     * (packGpuCache row layout), persisted by the priority pass.
     * SharedArrayBuffer-backed for the parallel path.
     */
    splatCache: Float32Array;
    /**
     * Global neighbour ids per splat (D per row, sentinel padded) — the union
     * of a cluster's members' rows forms its refresh candidate pool (the full
     * neighbour graph, not just the seed candidates: top-K lists collapse onto
     * shared hubs in degenerate/coincident regions and would starve refreshes).
     * SharedArrayBuffer-backed for the parallel path.
     */
    neighbors: Uint32Array;
    /** Neighbours per splat. */
    D: number;
    /** Gaussian count. */
    N: number;
    /** Target removal count for this generation. */
    mergesNeeded: number;
};

/**
 * Allocate a typed array, on shared memory when requested.
 *
 * @param ctor - Typed-array constructor.
 * @param bytes - Buffer size in bytes.
 * @param shared - Back with a SharedArrayBuffer (worker-visible state).
 * @returns The array view.
 */
const alloc = <T extends Float64Array | Uint32Array>(
    ctor: new (buffer: ArrayBufferLike) => T, bytes: number, shared: boolean
): T => {
    return new ctor(shared ? new SharedArrayBuffer(bytes) : new ArrayBuffer(bytes));
};

/**
 * Select merges with within-generation re-costing (round-parallel).
 *
 * @param inputs - See {@link RecostInputs}.
 * @returns The selection (same contract as {@link selectMerges}).
 */
const selectMergesRecosted = async (inputs: RecostInputs): Promise<SelectionResult> => {
    const { cand, K, splatCache: SC, neighbors, D, N, mergesNeeded } = inputs;

    const shared = typeof SharedArrayBuffer !== 'undefined' &&
        SC.buffer instanceof SharedArrayBuffer &&
        neighbors.buffer instanceof SharedArrayBuffer &&
        !WorkerQueue.isInline;

    // ---- Cluster state (indexed by union-find root).
    const parent = alloc(Uint32Array, N * 4, shared);
    const size = alloc(Uint32Array, N * 4, shared);
    const W = alloc(Float64Array, N * 8, shared);
    const mx = alloc(Float64Array, N * 8, shared);
    const my = alloc(Float64Array, N * 8, shared);
    const mz = alloc(Float64Array, N * 8, shared);
    const M2 = alloc(Float64Array, N * 6 * 8, shared);
    const baseW = alloc(Float64Array, N * 3 * 8, shared);
    const Sself = alloc(Float64Array, N * 8, shared);
    const Err = alloc(Float64Array, N * 8, shared);
    const version = alloc(Uint32Array, N * 4, shared);
    const mHead = alloc(Uint32Array, N * 4, shared);
    const mNext = alloc(Uint32Array, N * 4, shared);
    // Main-thread-only state.
    const mTail = new Uint32Array(N);
    const lastSeq = new Uint32Array(N);

    const st: RecostState = {
        SC,
        cands: neighbors,
        D,
        N,
        maxGroup: MAX_GROUP,
        parent,
        size,
        W,
        mx,
        my,
        mz,
        M2,
        baseW,
        Sself,
        Err,
        version,
        mHead,
        mNext
    };

    size.fill(1);
    version.fill(1);
    mNext.fill(NIL);
    for (let i = 0; i < N; i++) {
        parent[i] = i; mHead[i] = i; mTail[i] = i;
        const o = i * CACHE_STRIDE, i6 = i * 6, i3 = i * 3;
        const mass = SC[o + 11];
        W[i] = mass;
        mx[i] = SC[o]; my[i] = SC[o + 1]; mz[i] = SC[o + 2];
        // The cache Σ is unregularized; moments accumulate it directly
        // (mergeGroup member math). EPS enters once, on the merged covariance.
        M2[i6] = mass * SC[o + 3];
        M2[i6 + 1] = mass * SC[o + 4];
        M2[i6 + 2] = mass * SC[o + 5];
        M2[i6 + 3] = mass * SC[o + 6];
        M2[i6 + 4] = mass * SC[o + 7];
        M2[i6 + 5] = mass * SC[o + 8];
        baseW[i3] = mass * SC[o + 12];
        baseW[i3 + 1] = mass * SC[o + 13];
        baseW[i3 + 2] = mass * SC[o + 14];
        Sself[i] = SC[o + 10] * SC[o + 10] * SC[o + 15] * (Math.PI ** 1.5) * SC[o + 9];
        Err[i] = 0;
    }

    const find = (x: number): number => {
        while (parent[x] !== x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };

    // ---- Min-heap of candidate edges. Entries carry the winning
    // evaluation's E/Scross (NaN E = seed entry, computed at commit).
    let heapCap = Math.ceil(N * 1.25) + 16;
    let hCost = new Float64Array(heapCap);
    let hA = new Uint32Array(heapCap), hB = new Uint32Array(heapCap);
    let hSeq = new Uint32Array(heapCap), hVb = new Uint32Array(heapCap);
    let hE = new Float64Array(heapCap), hS = new Float64Array(heapCap);
    let heapSize = 0;
    let seqCounter = 0;

    const swap = (i: number, j: number): void => {
        let t;
        t = hCost[i]; hCost[i] = hCost[j]; hCost[j] = t;
        t = hA[i]; hA[i] = hA[j]; hA[j] = t;
        t = hB[i]; hB[i] = hB[j]; hB[j] = t;
        t = hSeq[i]; hSeq[i] = hSeq[j]; hSeq[j] = t;
        t = hVb[i]; hVb[i] = hVb[j]; hVb[j] = t;
        t = hE[i]; hE[i] = hE[j]; hE[j] = t;
        t = hS[i]; hS[i] = hS[j]; hS[j] = t;
    };

    const heapPush = (cost: number, a: number, b: number, seq: number, vb: number, e: number, s: number): void => {
        if (heapSize === heapCap) {
            const nc = heapCap * 2;
            const gf = (old: Float64Array) => {
                const x = new Float64Array(nc); x.set(old); return x;
            };
            const g = (old: Uint32Array) => {
                const x = new Uint32Array(nc); x.set(old); return x;
            };
            hCost = gf(hCost); hE = gf(hE); hS = gf(hS);
            hA = g(hA); hB = g(hB); hSeq = g(hSeq); hVb = g(hVb);
            heapCap = nc;
        }
        let i = heapSize++;
        hCost[i] = cost; hA[i] = a; hB[i] = b; hSeq[i] = seq; hVb[i] = vb; hE[i] = e; hS[i] = s;
        while (i > 0) {
            const p = (i - 1) >> 1;
            if (hCost[p] <= hCost[i]) break;
            swap(i, p); i = p;
        }
    };

    const popOut = { cost: 0, a: 0, b: 0, seq: 0, vb: 0, E: 0, S: 0 };
    const heapPop = (): boolean => {
        if (heapSize === 0) return false;
        popOut.cost = hCost[0]; popOut.a = hA[0]; popOut.b = hB[0]; popOut.seq = hSeq[0]; popOut.vb = hVb[0];
        popOut.E = hE[0]; popOut.S = hS[0];
        heapSize--;
        if (heapSize > 0) {
            hCost[0] = hCost[heapSize]; hA[0] = hA[heapSize]; hB[0] = hB[heapSize];
            hSeq[0] = hSeq[heapSize]; hVb[0] = hVb[heapSize];
            hE[0] = hE[heapSize]; hS[0] = hS[heapSize];
            let i = 0;
            for (;;) {
                const l = 2 * i + 1, r = l + 1;
                let m = i;
                if (l < heapSize && hCost[l] < hCost[m]) m = l;
                if (r < heapSize && hCost[r] < hCost[m]) m = r;
                if (m === i) break;
                swap(i, m); i = m;
            }
        }
        return true;
    };

    // Refresh queue: clusters whose best edge must be re-evaluated. Queuing
    // bumps lastSeq so stale heap entries for the cluster discard on pop;
    // queuedRound dedupes within a round.
    let pending = new Uint32Array(1 << 16);
    let pendingCount = 0;
    const queuedRound = new Uint32Array(N);
    let round = 1;
    const queueRefresh = (root: number): void => {
        lastSeq[root] = ++seqCounter;
        if (queuedRound[root] === round) return;
        queuedRound[root] = round;
        if (pendingCount === pending.length) {
            const g = new Uint32Array(pending.length * 2);
            g.set(pending); pending = g;
        }
        pending[pendingCount++] = root;
    };

    // Seed: the priority pass's cheapest candidate per gaussian.
    for (let i = 0; i < N; i++) {
        const j = cand.idx[i * K];
        const c = cand.cost[i * K];
        lastSeq[i] = ++seqCounter;
        if (j !== NO_CANDIDATE && Number.isFinite(c)) {
            heapPush(c, i, j, lastSeq[i], version[j], NaN, 0);
        }
    }

    // ---- Round loop. Each round commits at most WAVE merges before the
    // bulk refresh, so refreshed edges (notably cheap continuation merges in
    // redundant regions — the concentration behaviour the quality depends on)
    // re-enter the heap at most one wave late. An unbounded drain would spend
    // the budget up the cost curve before any refresh returns.
    const WAVE = 4096;
    let removed = 0;
    while (removed < mergesNeeded) {
        // Drain: commit still-valid entries, up to the wave budget.
        let wave = 0;
        while (removed < mergesNeeded && wave < WAVE && heapPop()) {
            const a = popOut.a;
            if (parent[a] !== a || popOut.seq !== lastSeq[a]) continue;
            const b = popOut.b;
            if (parent[b] !== b || version[b] !== popOut.vb || size[a] + size[b] > MAX_GROUP) {
                queueRefresh(a);
                continue;
            }

            let E = popOut.E, scross = popOut.S;
            if (Number.isNaN(E)) {
                evalMergeCore(st, a, b);
                E = evalOut.E; scross = evalOut.Scross;
            }
            const keep = size[a] >= size[b] ? a : b;
            const lose = keep === a ? b : a;

            const WA = W[a], WB = W[b], WC = WA + WB;
            const iw = 1 / WC;
            const mcx = (WA * mx[a] + WB * mx[b]) * iw;
            const mcy = (WA * my[a] + WB * my[b]) * iw;
            const mcz = (WA * mz[a] + WB * mz[b]) * iw;
            const dax = mx[a] - mcx, day = my[a] - mcy, daz = mz[a] - mcz;
            const dbx = mx[b] - mcx, dby = my[b] - mcy, dbz = mz[b] - mcz;
            const a6 = a * 6, b6 = b * 6, k6 = keep * 6;
            const n0 = M2[a6] + M2[b6] + WA * dax * dax + WB * dbx * dbx;
            const n1 = M2[a6 + 1] + M2[b6 + 1] + WA * dax * day + WB * dbx * dby;
            const n2 = M2[a6 + 2] + M2[b6 + 2] + WA * dax * daz + WB * dbx * dbz;
            const n3 = M2[a6 + 3] + M2[b6 + 3] + WA * day * day + WB * dby * dby;
            const n4 = M2[a6 + 4] + M2[b6 + 4] + WA * day * daz + WB * dby * dbz;
            const n5 = M2[a6 + 5] + M2[b6 + 5] + WA * daz * daz + WB * dbz * dbz;
            M2[k6] = n0; M2[k6 + 1] = n1; M2[k6 + 2] = n2; M2[k6 + 3] = n3; M2[k6 + 4] = n4; M2[k6 + 5] = n5;
            W[keep] = WC; mx[keep] = mcx; my[keep] = mcy; mz[keep] = mcz;
            const ka = keep * 3, aa = a * 3, bb = b * 3;
            const bw0 = baseW[aa] + baseW[bb], bw1 = baseW[aa + 1] + baseW[bb + 1], bw2 = baseW[aa + 2] + baseW[bb + 2];
            baseW[ka] = bw0; baseW[ka + 1] = bw1; baseW[ka + 2] = bw2;
            Sself[keep] = Sself[a] + Sself[b] + 2 * scross;
            Err[keep] = E;
            size[keep] += size[lose];
            mNext[mTail[keep]] = mHead[lose];
            mTail[keep] = mTail[lose];
            parent[lose] = keep;
            version[keep]++;
            removed++;
            wave++;

            queueRefresh(keep);
        }
        if (removed >= mergesNeeded) break;
        if (pendingCount === 0 && heapSize === 0) break;

        // Bulk refresh of queued clusters (parallel when shared).
        if (shared) {
            const workers = Math.max(1, Math.min(8, WorkerQueue.maxWorkers ?? 4));
            const chunk = Math.ceil(pendingCount / workers);
            const jobs: Promise<Float64Array>[] = [];
            for (let off = 0; off < pendingCount; off += chunk) {
                const roots = pending.slice(off, Math.min(off + chunk, pendingCount));
                jobs.push(WorkerQueue.run('recostBestEdges', {
                    sc: SC,
                    cands: neighbors,
                    d: D,
                    n: N,
                    maxGroup: MAX_GROUP,
                    parent,
                    size,
                    w: W,
                    mx,
                    my,
                    mz,
                    m2: M2,
                    baseW,
                    sself: Sself,
                    err: Err,
                    version,
                    mHead,
                    mNext,
                    roots
                }, [roots.buffer as ArrayBuffer]));
            }
            const results = await Promise.all(jobs);
            for (const res of results) {
                for (let i = 0; i < res.length; i += 6) {
                    const root = res[i];
                    const partner = res[i + 1];
                    if (partner < 0) continue;
                    heapPush(res[i + 3], root, partner, lastSeq[root], res[i + 2], res[i + 4], res[i + 5]);
                }
            }
        } else {
            for (let p = 0; p < pendingCount; p++) {
                const root = pending[p];
                if (bestEdgeFor(st, root)) {
                    heapPush(bestOut.cost, root, bestOut.partner, lastSeq[root], bestOut.vb, bestOut.E, bestOut.S);
                }
            }
        }
        pendingCount = 0;
        round++;
    }

    // ---- CSR assembly (identical contract to selectMerges).
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
    const groupMin = new Uint32Array(G);
    for (let g = 0; g < G; g++) groupMin[g] = groupMembers[groupOffsets[g]];

    return { groupOffsets, groupMembers, memberGroup, groupMin, mergedGroups: G, removed };
};

export { selectMergesRecosted, CACHE_STRIDE, type RecostInputs };
