/**
 * Re-costed merge selection: exact greedy agglomeration within a generation,
 * batched into commit/refresh rounds.
 *
 * Where {@link selectMerges} consumes the priority pass's pairwise costs
 * one-shot (costs go stale as groups form), this selection re-evaluates a
 * cluster's candidates after it changes, so merges always execute at costs
 * evaluated against current state:
 *
 *   1. Drain the heap, committing every entry that is still valid; clusters
 *      that changed (merged) or whose best partner changed are queued.
 *   2. Re-evaluate all queued clusters' best edges and push the results.
 *   3. Repeat until the generation target is reached or no merges remain.
 *
 * Relative to strictly-eager greedy, refreshed clusters re-enter the heap at
 * round boundaries instead of immediately; commits still only execute at
 * costs validated against the exact current state (version checks), so the
 * deviation is confined to near-tie ordering (re-certified on the evaluation
 * harness).
 *
 * The evaluation kernel ({@link recost-core}) is stateless over members:
 * cluster moments are recomputed from the ≤ MAX_GROUP member cache rows on
 * every evaluation, so this module keeps no per-root float state at all —
 * a commit is a chain splice plus union-find/size/version updates, and the
 * heap carries only (cost, a, b, seq, versionB). Seeding is wave 0: every
 * gaussian starts queued, so the first refresh evaluates each singleton's
 * best edge over its neighbour rows — no candidate arrays are consumed.
 *
 * With a device, refreshes run on the GPU ({@link GpuRecost}): the CPU
 * drains commits into a log, the GPU replays it against its own structure
 * copy and bulk-evaluates the queued roots, and the CPU pushes the returned
 * (partner, cost) pairs with validation tokens from its own mirrors. Waves
 * are serial by construction (each drain consumes the previous refresh's
 * entries), so there is nothing to double-buffer. Without a device — or
 * when the buffers don't fit the adapter's binding limits — the refresh
 * loop runs inline with the same eval in f64.
 *
 * decimate-source gates this path by memory budget and falls back to
 * {@link selectMerges} when over.
 */

import { type GraphicsDevice } from 'playcanvas';

import { bestEdgeFor, bestOut, CACHE_STRIDE, type RecostState } from './recost-core';
import { MAX_GROUP, type SelectionResult } from './select';
import { GpuRecost, COMMIT_LOG_STRIDE } from '../gpu/gpu-recost';

const NIL = 0xFFFFFFFF;

/** Inputs for {@link selectMergesRecosted}. */
type RecostInputs = {
    /**
     * Resident per-splat cache, {@link CACHE_STRIDE} floats per splat
     * (buildSplatCache row layout), persisted by the priority pass.
     */
    splatCache: Float32Array;
    /**
     * Global neighbour ids per splat (D per row, sentinel padded) — the union
     * of a cluster's members' rows forms its refresh candidate pool (the full
     * neighbour graph, not just the seed candidates: top-K lists collapse onto
     * shared hubs in degenerate/coincident regions and would starve refreshes).
     */
    neighbors: Uint32Array;
    /** Neighbours per splat. */
    D: number;
    /** Gaussian count. */
    N: number;
    /** Target removal count for this generation. */
    mergesNeeded: number;
    /** Optional GPU device: bulk refreshes run on the wave engine when the buffers fit. */
    device?: GraphicsDevice;
};

// Max merges committed between refreshes. LOAD-BEARING: an unbounded drain
// runs up the cost curve before cheap continuation merges re-enter (measured
// −9.5 dB); refreshing more often is quality-safe, less often is not.
const WAVE = 4096;

/**
 * Select merges with within-generation re-costing (round-batched).
 *
 * @param inputs - See {@link RecostInputs}.
 * @returns The selection (same contract as {@link selectMerges}).
 */
const selectMergesRecosted = async (inputs: RecostInputs): Promise<SelectionResult> => {
    const { splatCache: SC, neighbors, D, N, mergesNeeded, device } = inputs;

    // ---- Cluster structure (indexed by union-find root) — integers only.
    const parent = new Uint32Array(N);
    const size = new Uint32Array(N);
    const version = new Uint32Array(N);
    const mHead = new Uint32Array(N);
    const mNext = new Uint32Array(N);
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
        version,
        mHead,
        mNext
    };

    size.fill(1);
    version.fill(1);
    mNext.fill(NIL);
    for (let i = 0; i < N; i++) {
        parent[i] = i; mHead[i] = i; mTail[i] = i;
    }

    const find = (x: number): number => {
        while (parent[x] !== x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };

    // ---- Min-heap of candidate edges: one live entry per cluster, its
    // current best edge. Costs are f32 — ordering-only (ties resolve
    // arbitrarily by design), nothing accumulates.
    let heapCap = Math.ceil(N * 1.25) + 16;
    let hCost = new Float32Array(heapCap);
    let hA = new Uint32Array(heapCap), hB = new Uint32Array(heapCap);
    let hSeq = new Uint32Array(heapCap), hVb = new Uint32Array(heapCap);
    let heapSize = 0;
    let seqCounter = 0;

    const swap = (i: number, j: number): void => {
        let t;
        t = hCost[i]; hCost[i] = hCost[j]; hCost[j] = t;
        t = hA[i]; hA[i] = hA[j]; hA[j] = t;
        t = hB[i]; hB[i] = hB[j]; hB[j] = t;
        t = hSeq[i]; hSeq[i] = hSeq[j]; hSeq[j] = t;
        t = hVb[i]; hVb[i] = hVb[j]; hVb[j] = t;
    };

    const heapPush = (cost: number, a: number, b: number, seq: number, vb: number): void => {
        if (heapSize === heapCap) {
            const nc = heapCap * 2;
            const gf = (old: Float32Array) => {
                const x = new Float32Array(nc); x.set(old); return x;
            };
            const g = (old: Uint32Array) => {
                const x = new Uint32Array(nc); x.set(old); return x;
            };
            hCost = gf(hCost);
            hA = g(hA); hB = g(hB); hSeq = g(hSeq); hVb = g(hVb);
            heapCap = nc;
        }
        let i = heapSize++;
        hCost[i] = cost; hA[i] = a; hB[i] = b; hSeq[i] = seq; hVb[i] = vb;
        while (i > 0) {
            const p = (i - 1) >> 1;
            if (hCost[p] <= hCost[i]) break;
            swap(i, p); i = p;
        }
    };

    const popOut = { cost: 0, a: 0, b: 0, seq: 0, vb: 0 };
    const heapPop = (): boolean => {
        if (heapSize === 0) return false;
        popOut.cost = hCost[0]; popOut.a = hA[0]; popOut.b = hB[0]; popOut.seq = hSeq[0]; popOut.vb = hVb[0];
        heapSize--;
        if (heapSize > 0) {
            hCost[0] = hCost[heapSize]; hA[0] = hA[heapSize]; hB[0] = hB[heapSize];
            hSeq[0] = hSeq[heapSize]; hVb[0] = hVb[heapSize];
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
    const pending = new Uint32Array(N);
    let pendingCount = 0;
    const queuedRound = new Uint32Array(N);
    let round = 1;
    const queueRefresh = (root: number): void => {
        lastSeq[root] = ++seqCounter;
        if (queuedRound[root] === round) return;
        queuedRound[root] = round;
        pending[pendingCount++] = root;
    };

    // Seed = wave 0: every gaussian starts queued, so the first refresh
    // evaluates each singleton's best edge over its neighbour rows (identical
    // work to a per-edge cost pass + argmin, and the same eval as every later
    // refresh). The first drain is a no-op on the empty heap.
    for (let i = 0; i < N; i++) {
        lastSeq[i] = ++seqCounter;
        queuedRound[i] = round;
        pending[i] = i;
    }
    pendingCount = N;

    // ---- GPU wave engine when the buffers fit (its workgroup shape needs
    // the production k·MAX_GROUP); inline refresh otherwise.
    const gpu = device && D * MAX_GROUP === 64 && GpuRecost.fits(device, N, D, WAVE) ?
        new GpuRecost(device, N, D, MAX_GROUP, WAVE) :
        undefined;
    const commitLog = gpu ? new Uint32Array(WAVE * COMMIT_LOG_STRIDE) : undefined;
    const outBest = gpu ? new Uint32Array(N * 2) : undefined;
    const outCost = gpu ? new Float32Array(outBest!.buffer) : undefined;

    // ---- Round loop. Each round commits at most WAVE merges before the
    // bulk refresh, so refreshed edges (notably cheap continuation merges in
    // redundant regions — the concentration behaviour the quality depends on)
    // re-enter the heap at most one wave late. An unbounded drain would spend
    // the budget up the cost curve before any refresh returns.
    let removed = 0;
    try {
        gpu?.init(SC, neighbors);

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

                // Commit: pure structure — splice the member chains, re-parent,
                // bump the version so stale partner entries invalidate. The GPU
                // replays the same splice from the log (values pre-resolved so
                // the replay does no reads).
                const keep = size[a] >= size[b] ? a : b;
                const lose = keep === a ? b : a;
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
                removed++;
                wave++;

                queueRefresh(keep);
            }
            if (removed >= mergesNeeded) break;
            if (pendingCount === 0 && heapSize === 0) break;

            // Refresh queued clusters' best edges.
            if (gpu && pendingCount > 0) {
                await gpu.wave(commitLog!, wave, pending, pendingCount, outBest!);
                for (let p = 0; p < pendingCount; p++) {
                    const partner = outBest![p * 2];
                    if (partner === NIL) continue;
                    const cost = outCost![p * 2 + 1];
                    if (Number.isNaN(cost)) continue;
                    const root = pending[p];
                    heapPush(cost, root, partner, lastSeq[root], version[partner]);
                }
            } else {
                for (let p = 0; p < pendingCount; p++) {
                    const root = pending[p];
                    if (bestEdgeFor(st, root)) {
                        heapPush(bestOut.cost, root, bestOut.partner, lastSeq[root], bestOut.vb);
                    }
                }
            }
            pendingCount = 0;
            round++;
        }
    } finally {
        gpu?.destroy();
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
