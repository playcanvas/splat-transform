/**
 * Re-costed selection evaluation kernel, shared between the main thread and
 * worker threads (bulk refresh rounds). Operates on a plain view-of-state
 * object whose arrays may be backed by SharedArrayBuffers; workers treat the
 * state as read-only (find() does no path compression here).
 *
 * Cost definition matches the pairwise kernel: exact field-L2 of the merged
 * cluster vs its generation-input members (splat cache rows) plus the
 * scale-free DC colour term — see select-recost.ts for the orchestration.
 *
 * Engine-free; no allocation in the hot paths beyond small local scratch.
 */

import { COLOR_WEIGHT } from './edge-cost-cpu';
import { EPS_COV, ellipsoidArea } from './moment-match';

/** Floats per splat in the resident cache (packGpuCache layout). */
export const CACHE_STRIDE = 16;

const NO_CANDIDATE = 0xFFFFFFFF;
const NIL = 0xFFFFFFFF;

/** Skip Gaussian products whose exponent bound exceeds this (e^-60 ≈ 9e-27). */
const CULL_QUAD = 120;

const PI_1_5 = Math.PI ** 1.5;
const TWO_PI_1_5 = (2 * Math.PI) ** 1.5;

/**
 * The resident selection state (allocated by select-recost, possibly on
 * SharedArrayBuffers so bulk refreshes can run on worker threads).
 */
type RecostState = {
    /** Per-splat cache, {@link CACHE_STRIDE} floats per generation-input splat. */
    SC: Float32Array;
    /** Candidate ids per splat (D per row, NO_CANDIDATE padded) — the union of members' rows forms a cluster's candidate pool. */
    cands: Uint32Array;
    /** Candidate ids per splat. */
    D: number;
    /** Gaussian count. */
    N: number;
    /** Max original members per group. */
    maxGroup: number;
    // Union-find + cluster state (indexed by root).
    parent: Uint32Array;
    size: Uint32Array;
    W: Float64Array;
    mx: Float64Array; my: Float64Array; mz: Float64Array;
    M2: Float64Array;
    baseW: Float64Array;
    Sself: Float64Array;
    Err: Float64Array;
    version: Uint32Array;
    mHead: Uint32Array;
    mNext: Uint32Array;
};

/**
 * Read-only find (no path compression — safe on shared state in workers).
 *
 * @param parent - Union-find parent array.
 * @param x - Element to resolve.
 * @returns The set root.
 */
const findRO = (parent: Uint32Array, x: number): number => {
    while (parent[x] !== x) x = parent[x];
    return x;
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
const eigOut = new Float64Array(3);
const eig3 = (m0: number, m1: number, m2: number, m3: number, m4: number, m5: number): void => {
    const q = (m0 + m3 + m5) / 3;
    const p1 = m1 * m1 + m2 * m2 + m4 * m4;
    if (p1 <= 1e-30) {
        eigOut[0] = Math.max(m0, m3, m5);
        eigOut[2] = Math.min(m0, m3, m5);
        eigOut[1] = m0 + m3 + m5 - eigOut[0] - eigOut[2];
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
    eigOut[0] = e0; eigOut[1] = 3 * q - e0 - e2; eigOut[2] = e2;
};

let gatherBuf = new Uint32Array(1 << 12);
const gatherMembers = (st: RecostState, root: number): number => {
    const { mHead, mNext } = st;
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

/** Outputs of {@link evalMergeCore} beyond the cost (reused object). */
const evalOut = { E: 0, Scross: 0 };

/**
 * Marginal cost ΔE of merging clusters A and B (exact field-L2 vs the
 * generation-input members) plus the scale-free colour term. Fills
 * {@link evalOut} with E(A∪B) and Scross(A,B).
 *
 * @param st - Selection state.
 * @param A - First cluster root.
 * @param B - Second cluster root.
 * @returns The merge cost.
 */
const evalMergeCore = (st: RecostState, A: number, B: number): number => {
    const { SC, W, mx, my, mz, M2, baseW, Sself, Err, mHead, mNext } = st;
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

    eig3(sm0, sm1, sm2, sm3, sm4, sm5);
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
    const na = gatherMembers(st, A);
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

    const iwA = 1 / W[A], iwB = 1 / W[B];
    const d0 = baseW[a3] * iwA - baseW[b3] * iwB;
    const d1 = baseW[a3 + 1] * iwA - baseW[b3 + 1] * iwB;
    const d2c = baseW[a3 + 2] * iwA - baseW[b3 + 2] * iwB;
    return (E - Err[A] - Err[B]) + COLOR_WEIGHT * (d0 * d0 + d1 * d1 + d2c * d2c);
};

/** Result of {@link bestEdgeFor} (reused object). */
const bestOut = { partner: -1, vb: 0, cost: 0, E: 0, S: 0 };

/**
 * Compute the cheapest legal merge for `root`: candidates are the live
 * clusters owning any member's candidate ids, deduplicated linearly (pools
 * are small), respecting the group cap.
 *
 * @param st - Selection state.
 * @param root - Cluster root to refresh.
 * @returns True when a legal candidate exists (result in {@link bestOut}).
 */
const candScratch = new Uint32Array(256);

const bestEdgeFor = (st: RecostState, root: number): boolean => {
    const { cands, D, parent, size, version, mHead, mNext, maxGroup } = st;
    const sz = size[root];
    // Derive + dedup candidates (linear scan — pools are ≤ a few dozen).
    let cnt = 0;
    const cbuf = candScratch;
    for (let m = mHead[root]; m !== NIL; m = mNext[m]) {
        const base = m * D;
        for (let s = 0; s < D; s++) {
            const nb = cands[base + s];
            if (nb === NO_CANDIDATE) continue;
            const r = findRO(parent, nb);
            if (r === root) continue;
            let seen = false;
            for (let t = 0; t < cnt; t++) {
                if (cbuf[t] === r) {
                    seen = true; break;
                }
            }
            if (seen) continue;
            if (cnt < cbuf.length) cbuf[cnt++] = r;
        }
    }
    let bc = Infinity, bp = -1, bv = 0, bE = 0, bS = 0;
    for (let t = 0; t < cnt; t++) {
        const c = cbuf[t];
        if (sz + size[c] > maxGroup) continue;
        const d = evalMergeCore(st, root, c);
        if (d < bc) {
            bc = d; bp = c; bv = version[c]; bE = evalOut.E; bS = evalOut.Scross;
        }
    }
    if (bp < 0) return false;
    bestOut.partner = bp; bestOut.vb = bv; bestOut.cost = bc; bestOut.E = bE; bestOut.S = bS;
    return true;
};

export { evalMergeCore, bestEdgeFor, findRO, evalOut, bestOut, NO_CANDIDATE, type RecostState };
