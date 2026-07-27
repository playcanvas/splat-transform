/**
 * Re-costed merge selection: exact greedy agglomeration within a generation.
 *
 * Where {@link selectMerges} consumes the priority pass's pairwise costs
 * one-shot (costs go stale as groups form), this selection re-evaluates a
 * cluster's candidates after every merge against its CURRENT moments, so the
 * cheapest-first order is always true. Validated against the reference
 * implementation (tools/decimate-exact.mjs) as the production form of the
 * study winner: within-generation exact re-costing closes the remaining
 * ~1–2 dB at fine levels vs one-shot selection.
 *
 * Cost of a cluster = exact field-L2 vs its generation-input members (their
 * parameters come from the resident splat cache emitted by the priority
 * pass), plus the scale-free DC colour term ({@link COLOR_WEIGHT}) — the same
 * cost definition as the pairwise kernel, so the precomputed candidate costs
 * seed the heap directly.
 *
 * Memory: ~200 B per gaussian resident (splat cache, neighbour ids, cluster
 * moments, heap). decimate-source gates this path by memory budget and falls
 * back to {@link selectMerges} when it does not fit.
 *
 * Engine-free; pure resident-array computation, no IO.
 */

import { COLOR_WEIGHT } from './edge-cost-cpu';
import { KNN_SENTINEL } from './knn-core';
import { EPS_COV, ellipsoidArea } from './moment-match';
import { type CandidateArrays } from './priority';
import { MAX_GROUP, type SelectionResult } from './select';

/** Floats per splat in the resident cache (packGpuCache layout). */
export const CACHE_STRIDE = 16;

const NO_CANDIDATE = 0xFFFFFFFF;
const NIL = 0xFFFFFFFF;

/** Skip Gaussian products whose exponent bound exceeds this (e^-60 ≈ 9e-27). */
const CULL_QUAD = 120;

const PI_1_5 = Math.PI ** 1.5;
const TWO_PI_1_5 = (2 * Math.PI) ** 1.5;

/** Inputs for {@link selectMergesRecosted}. */
type RecostInputs = {
    /** Per-gaussian best-K candidates from the priority pass (seed costs include the colour term). */
    cand: CandidateArrays;
    /** Candidates per gaussian. */
    K: number;
    /**
     * Resident per-splat cache, {@link CACHE_STRIDE} floats per splat
     * (pos 3, Σ 6 [xx xy xz yy yz zz], √|Σ|, α, mass, base colour 3, |base|²)
     * — the packGpuCache row layout, persisted by the priority pass.
     */
    splatCache: Float32Array;
    /** Global neighbour ids per splat (D per row, KNN_SENTINEL padded). */
    neighbors: Uint32Array;
    /** Neighbours per splat. */
    D: number;
    /** Gaussian count. */
    N: number;
    /** Target removal count for this generation. */
    mergesNeeded: number;
};

// ⟨G_a,G_b⟩ scaled by √|Σa|·√|Σb| for M = Σa+Σb (6 comps) and offset d.
const crossG = (
    sdAB: number,
    m0: number, m1: number, m2: number, m3: number, m4: number, m5: number,
    dx: number, dy: number, dz: number
): number => {
    const c00 = m3 * m5 - m4 * m4;
    const c01 = m2 * m4 - m1 * m5;
    const c02 = m1 * m4 - m2 * m3;
    const c11 = m0 * m5 - m2 * m2;
    const c12 = m1 * m2 - m0 * m4;
    const c22 = m0 * m3 - m1 * m1;
    const det = Math.max(m0 * c00 + m1 * c01 + m2 * c02, 1e-60);
    const quad = (c00 * dx * dx + c11 * dy * dy + c22 * dz * dz +
        2 * (c01 * dx * dy + c02 * dx * dz + c12 * dy * dz)) / det;
    if (!(quad < CULL_QUAD)) return 0;
    return TWO_PI_1_5 * sdAB / Math.sqrt(det) * Math.exp(-0.5 * quad);
};

// Smith closed-form eigenvalues of a symmetric 3×3 (6 comps), descending.
const eig3 = (m0: number, m1: number, m2: number, m3: number, m4: number, m5: number, out: Float64Array): void => {
    const q = (m0 + m3 + m5) / 3;
    const p1 = m1 * m1 + m2 * m2 + m4 * m4;
    if (p1 <= 1e-30) {
        out[0] = Math.max(m0, m3, m5);
        out[2] = Math.min(m0, m3, m5);
        out[1] = m0 + m3 + m5 - out[0] - out[2];
        return;
    }
    const p2 = (m0 - q) * (m0 - q) + (m3 - q) * (m3 - q) + (m5 - q) * (m5 - q) + 2 * p1;
    const p = Math.sqrt(p2 / 6);
    const ip = 1 / p;
    const b00 = (m0 - q) * ip, b11 = (m3 - q) * ip, b22 = (m5 - q) * ip;
    const b01 = m1 * ip, b02 = m2 * ip, b12 = m4 * ip;
    const detB = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02) + b02 * (b01 * b12 - b11 * b02);
    let r = detB / 2;
    r = r < -1 ? -1 : (r > 1 ? 1 : r);
    const phi = Math.acos(r) / 3;
    const e0 = q + 2 * p * Math.cos(phi);
    const e2 = q + 2 * p * Math.cos(phi + (2 * Math.PI) / 3);
    out[0] = e0; out[1] = 3 * q - e0 - e2; out[2] = e2;
};

/**
 * Select merges with exact within-generation re-costing.
 *
 * @param inputs - See {@link RecostInputs}.
 * @returns The selection (same contract as {@link selectMerges}).
 */
const selectMergesRecosted = (inputs: RecostInputs): SelectionResult => {
    const { cand, K, splatCache: SC, neighbors, D, N, mergesNeeded } = inputs;

    // ---- Cluster state (indexed by union-find root).
    const parent = new Uint32Array(N);
    const size = new Uint32Array(N).fill(1);
    const W = new Float64Array(N);
    const mx = new Float64Array(N), my = new Float64Array(N), mz = new Float64Array(N);
    const M2 = new Float64Array(N * 6);
    const baseW = new Float64Array(N * 3);
    const Sself = new Float64Array(N);
    const Err = new Float64Array(N);
    const version = new Uint32Array(N).fill(1);
    const lastSeq = new Uint32Array(N);
    const mHead = new Uint32Array(N), mTail = new Uint32Array(N);
    const mNext = new Uint32Array(N).fill(NIL);
    const stamp = new Uint32Array(N);
    let stampGen = 0;
    let seqCounter = 0;
    let liveCount = N;

    for (let i = 0; i < N; i++) {
        parent[i] = i; mHead[i] = i; mTail[i] = i;
        const o = i * CACHE_STRIDE, i6 = i * 6, i3 = i * 3;
        const mass = SC[o + 11];
        W[i] = mass;
        mx[i] = SC[o]; my[i] = SC[o + 1]; mz[i] = SC[o + 2];
        // The cache Σ is unregularized (buildCostCache uses raw variances), so
        // it feeds the moments directly — matching mergeGroup's member math.
        // EPS_COV enters once, on the merged covariance in evalMerge.
        M2[i6] = mass * SC[o + 3];
        M2[i6 + 1] = mass * SC[o + 4];
        M2[i6 + 2] = mass * SC[o + 5];
        M2[i6 + 3] = mass * SC[o + 6];
        M2[i6 + 4] = mass * SC[o + 7];
        M2[i6 + 5] = mass * SC[o + 8];
        baseW[i3] = mass * SC[o + 12];
        baseW[i3 + 1] = mass * SC[o + 13];
        baseW[i3 + 2] = mass * SC[o + 14];
        Sself[i] = SC[o + 10] * SC[o + 10] * SC[o + 15] * PI_1_5 * SC[o + 9];
        Err[i] = 0;
    }

    const find = (x: number): number => {
        while (parent[x] !== x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };

    // ---- Exact merge cost: ΔE of A∪B vs generation-input members + colour term.
    const eigOut = new Float64Array(3);
    let gatherBuf = new Uint32Array(1 << 12);
    const gatherMembers = (root: number): number => {
        let cnt = 0;
        for (let m = mHead[root]; m !== NIL; m = mNext[m]) {
            if (cnt === gatherBuf.length) {
                const g = new Uint32Array(gatherBuf.length * 2);
                g.set(gatherBuf); gatherBuf = g;
            }
            gatherBuf[cnt++] = m;
        }
        return cnt;
    };

    const evalOut = { E: 0, Scross: 0 };
    const evalMerge = (A: number, B: number): number => {
        const WA = W[A], WB = W[B], WC = WA + WB;
        const iw = 1 / WC;
        const mcx = (WA * mx[A] + WB * mx[B]) * iw;
        const mcy = (WA * my[A] + WB * my[B]) * iw;
        const mcz = (WA * mz[A] + WB * mz[B]) * iw;
        const dax = mx[A] - mcx, day = my[A] - mcy, daz = mz[A] - mcz;
        const dbx = mx[B] - mcx, dby = my[B] - mcy, dbz = mz[B] - mcz;
        const a6 = A * 6, b6 = B * 6;

        const sm0 = (M2[a6] + M2[b6] + WA * dax * dax + WB * dbx * dbx) * iw + EPS_COV;
        const sm1 = (M2[a6 + 1] + M2[b6 + 1] + WA * dax * day + WB * dbx * dby) * iw;
        const sm2 = (M2[a6 + 2] + M2[b6 + 2] + WA * dax * daz + WB * dbx * dbz) * iw;
        const sm3 = (M2[a6 + 3] + M2[b6 + 3] + WA * day * day + WB * dby * dby) * iw + EPS_COV;
        const sm4 = (M2[a6 + 4] + M2[b6 + 4] + WA * day * daz + WB * dby * dbz) * iw;
        const sm5 = (M2[a6 + 5] + M2[b6 + 5] + WA * daz * daz + WB * dbz * dbz) * iw + EPS_COV;

        const detm = Math.max(
            sm0 * (sm3 * sm5 - sm4 * sm4) - sm1 * (sm1 * sm5 - sm4 * sm2) + sm2 * (sm1 * sm4 - sm3 * sm2),
            1e-60
        );
        const sdC = Math.sqrt(detm);

        eig3(sm0, sm1, sm2, sm3, sm4, sm5, eigOut);
        const s0 = Math.sqrt(Math.max(eigOut[0], 1e-18));
        const s1 = Math.sqrt(Math.max(eigOut[1], 1e-18));
        const s2 = Math.sqrt(Math.max(eigOut[2], 1e-18));
        const alphaC = Math.min(1, WC / Math.max(ellipsoidArea(s0, s1, s2), 1e-30));

        const a3 = A * 3, b3 = B * 3;
        const bc0 = (baseW[a3] + baseW[b3]) * iw;
        const bc1 = (baseW[a3 + 1] + baseW[b3 + 1]) * iw;
        const bc2 = (baseW[a3 + 2] + baseW[b3 + 2]) * iw;
        const bn2C = bc0 * bc0 + bc1 * bc1 + bc2 * bc2;

        const selfM = alphaC * alphaC * bn2C * PI_1_5 * sdC;

        // ⟨Σ member fields, f_m⟩ over both chains.
        let memfm = 0;
        for (let pass = 0; pass < 2; pass++) {
            for (let m = pass === 0 ? mHead[A] : mHead[B]; m !== NIL; m = mNext[m]) {
                const o = m * CACHE_STRIDE;
                const wgt = SC[o + 10] * alphaC *
                    (SC[o + 12] * bc0 + SC[o + 13] * bc1 + SC[o + 14] * bc2);
                if (wgt === 0) continue;
                memfm += wgt * crossG(SC[o + 9] * sdC,
                    SC[o + 3] + sm0, SC[o + 4] + sm1, SC[o + 5] + sm2,
                    SC[o + 6] + sm3, SC[o + 7] + sm4, SC[o + 8] + sm5,
                    SC[o] - mcx, SC[o + 1] - mcy, SC[o + 2] - mcz);
            }
        }

        // Scross(A,B) = Σ_{a∈A,b∈B}⟨f_a,f_b⟩ with distance culling.
        const na = gatherMembers(A);
        let scross = 0;
        for (let b = mHead[B]; b !== NIL; b = mNext[b]) {
            const ob = b * CACHE_STRIDE;
            const bxp = SC[ob], byp = SC[ob + 1], bzp = SC[ob + 2];
            const trb = SC[ob + 3] + SC[ob + 6] + SC[ob + 8];
            const alb = SC[ob + 10], sdb = SC[ob + 9];
            const cb0 = SC[ob + 12], cb1 = SC[ob + 13], cb2 = SC[ob + 14];
            for (let t = 0; t < na; t++) {
                const a = gatherBuf[t];
                const oa = a * CACHE_STRIDE;
                const dx = SC[oa] - bxp, dy = SC[oa + 1] - byp, dz = SC[oa + 2] - bzp;
                const d2 = dx * dx + dy * dy + dz * dz;
                if (d2 > CULL_QUAD * (SC[oa + 3] + SC[oa + 6] + SC[oa + 8] + trb)) continue;
                const wgt = SC[oa + 10] * alb *
                    (SC[oa + 12] * cb0 + SC[oa + 13] * cb1 + SC[oa + 14] * cb2);
                if (wgt === 0) continue;
                scross += wgt * crossG(SC[oa + 9] * sdb,
                    SC[oa + 3] + SC[ob + 3], SC[oa + 4] + SC[ob + 4], SC[oa + 5] + SC[ob + 5],
                    SC[oa + 6] + SC[ob + 6], SC[oa + 7] + SC[ob + 7], SC[oa + 8] + SC[ob + 8],
                    dx, dy, dz);
            }
        }

        const E = Sself[A] + Sself[B] + 2 * scross - 2 * memfm + selfM;
        evalOut.E = E;
        evalOut.Scross = scross;

        // Scale-free colour term between the cluster mean base colours (same
        // definition as the pairwise kernel, so seed costs are consistent).
        const iwA = 1 / W[A], iwB = 1 / W[B];
        const d0 = baseW[a3] * iwA - baseW[b3] * iwB;
        const d1 = baseW[a3 + 1] * iwA - baseW[b3 + 1] * iwB;
        const d2c = baseW[a3 + 2] * iwA - baseW[b3 + 2] * iwB;
        return (E - Err[A] - Err[B]) + COLOR_WEIGHT * (d0 * d0 + d1 * d1 + d2c * d2c);
    };

    // ---- Lazy min-heap of candidate edges (one live entry per cluster).
    let heapCap = Math.ceil(N * 1.25) + 16;
    let hCost = new Float64Array(heapCap);
    let hA = new Uint32Array(heapCap), hB = new Uint32Array(heapCap);
    let hSeq = new Uint32Array(heapCap), hVb = new Uint32Array(heapCap);
    let heapSize = 0;

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
            const c2 = new Float64Array(nc); c2.set(hCost); hCost = c2;
            const g = (old: Uint32Array) => {
                const x = new Uint32Array(nc); x.set(old); return x;
            };
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

    // Candidate derivation: live clusters owning any neighbour of any member.
    let candBuf = new Uint32Array(1 << 12);
    const deriveCandidates = (root: number): number => {
        stampGen++;
        const gen = stampGen;
        let cnt = 0;
        for (let m = mHead[root]; m !== NIL; m = mNext[m]) {
            const base = m * D;
            for (let s = 0; s < D; s++) {
                const nb = neighbors[base + s];
                if (nb === KNN_SENTINEL) continue;
                const r = find(nb);
                if (r === root || stamp[r] === gen) continue;
                stamp[r] = gen;
                if (cnt === candBuf.length) {
                    const g = new Uint32Array(candBuf.length * 2);
                    g.set(candBuf); candBuf = g;
                }
                candBuf[cnt++] = r;
            }
        }
        return cnt;
    };

    const pushBestEdge = (root: number): void => {
        const cnt = deriveCandidates(root);
        lastSeq[root] = ++seqCounter;
        if (cnt === 0) return;
        const sz = size[root];
        let bc = Infinity, bp = -1, bv = 0;
        for (let t = 0; t < cnt; t++) {
            const c = candBuf[t];
            if (sz + size[c] > MAX_GROUP) continue;
            const d = evalMerge(root, c);
            if (d < bc) {
                bc = d; bp = c; bv = version[c];
            }
        }
        if (bp >= 0) heapPush(bc, root, bp, lastSeq[root], bv);
    };

    // Seed: the priority pass's cheapest candidate per gaussian (same cost
    // definition, already sorted ascending — no re-evaluation needed).
    for (let i = 0; i < N; i++) {
        const j = cand.idx[i * K];
        const c = cand.cost[i * K];
        lastSeq[i] = ++seqCounter;
        if (j !== NO_CANDIDATE && Number.isFinite(c)) {
            heapPush(c, i, j, lastSeq[i], version[j]);
        }
    }

    // ---- Greedy loop.
    let removed = 0;
    while (removed < mergesNeeded) {
        if (!heapPop()) break;
        const a = popOut.a;
        if (parent[a] !== a || popOut.seq !== lastSeq[a]) continue;
        const b = popOut.b;
        if (parent[b] !== b || version[b] !== popOut.vb || size[a] + size[b] > MAX_GROUP) {
            pushBestEdge(a);
            continue;
        }

        // Commit a+b (exact recompute for the error bookkeeping).
        evalMerge(a, b);
        const E = evalOut.E, scross = evalOut.Scross;
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
        liveCount--;
        removed++;

        pushBestEdge(keep);
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

export { selectMergesRecosted, type RecostInputs };
