/**
 * Re-costed selection evaluation kernel — stateless over members.
 *
 * Cost definition matches the pairwise kernel: exact field-L2 of the merged
 * cluster vs its generation-input members (splat cache rows) plus the
 * scale-free DC colour term. Because groups are capped at MAX_GROUP original
 * members, a cluster's moments (W, mean, M2, colour sums) are recomputed on
 * the fly from its ≤ maxGroup immutable cache rows — no per-root float state
 * is stored anywhere, and commits are pure integer structure updates.
 *
 * The marginal cost ΔE = E(A∪B) − E(A) − E(B) is evaluated in the cancelled
 * form (the within-cluster pairwise sums Sself drop out identically):
 *
 *   ΔE = 2·scross(A,B) − 2·memfm(A∪B) + selfM(A∪B)
 *      + (|A|≥2 ? 2·memfm(A) − selfM(A) : self_A)
 *      + (|B|≥2 ? 2·memfm(B) − selfM(B) : self_B)
 *
 * where memfm(C) = Σ_{k∈C} ⟨f_k, f_C⟩, selfM(C) = ⟨f_C, f_C⟩, scross(A,B) =
 * Σ_{a∈A,b∈B} ⟨f_a, f_b⟩, and self_x is a singleton's self product (its
 * E is exactly 0 by convention — the merged Gaussian of one splat is itself).
 * Beyond deleting all per-root aggregates, the cancelled form is also
 * better-conditioned: the mutually-cancelling Sself magnitudes never enter.
 *
 * Engine-free; single-threaded (module-scope scratch, no allocation in the
 * hot paths).
 */

import { CACHE_STRIDE, COLOR_WEIGHT } from './edge-cost-cpu';
import { EPS_COV, ellipsoidArea } from './moment-match';

export { CACHE_STRIDE };

const NO_CANDIDATE = 0xFFFFFFFF;
const NIL = 0xFFFFFFFF;

/** Skip Gaussian products whose exponent bound exceeds this (e^-60 ≈ 9e-27). */
const CULL_QUAD = 120;

const PI_1_5 = Math.PI ** 1.5;
const TWO_PI_1_5 = (2 * Math.PI) ** 1.5;

/**
 * The resident selection state (allocated by select-recost). Only integer
 * structure — the union-find, the member chains, and the immutable per-splat
 * cache/neighbour inputs from the priority pass.
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
    // Union-find + cluster structure (indexed by root).
    parent: Uint32Array;
    size: Uint32Array;
    version: Uint32Array;
    mHead: Uint32Array;
    mNext: Uint32Array;
};

/**
 * Read-only find (no path compression).
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

/**
 * A cluster's moments composed from its member cache rows: raw mass-scaled
 * aggregates (the parallel-axis form the old per-root state stored) plus the
 * finished merged-Gaussian quantities the field terms need.
 */
type Comp = {
    // Raw aggregates.
    W: number;
    mx: number; my: number; mz: number;   // mass-weighted mean
    M2: Float64Array;                      // Σ mass·(Σₖ + δδᵀ), 6 comps
    bw0: number; bw1: number; bw2: number; // Σ mass·base
    // Finished merged Gaussian (valid after finishComp).
    sm: Float64Array;                      // M2/W + EPS_COV·I, 6 comps
    sd: number;                            // √|Σ_C|
    alpha: number;                         // min(1, W / area)
    bc0: number; bc1: number; bc2: number; // mean base colour (bw/W)
    bn2: number;                           // |mean base|²
    selfM: number;                         // ⟨f_C, f_C⟩
};

const makeComp = (): Comp => ({
    W: 0,
    mx: 0,
    my: 0,
    mz: 0,
    M2: new Float64Array(6),
    bw0: 0,
    bw1: 0,
    bw2: 0,
    sm: new Float64Array(6),
    sd: 0,
    alpha: 0,
    bc0: 0,
    bc1: 0,
    bc2: 0,
    bn2: 0,
    selfM: 0
});

// Module-scope scratch — single-threaded by design.
const compA = makeComp();
const compB = makeComp();
const compAB = makeComp();

// Compose raw aggregates from member cache rows (two passes: mean, then
// central second moments — algebraically identical to the parallel-axis
// accumulation the formation tree would produce).
const composeRaw = (SC: Float32Array, members: Uint32Array, count: number, out: Comp): void => {
    let W = 0, sx = 0, sy = 0, sz = 0, b0 = 0, b1 = 0, b2 = 0;
    for (let t = 0; t < count; t++) {
        const o = members[t] * CACHE_STRIDE;
        const m = SC[o + 11];
        W += m;
        sx += m * SC[o]; sy += m * SC[o + 1]; sz += m * SC[o + 2];
        b0 += m * SC[o + 12]; b1 += m * SC[o + 13]; b2 += m * SC[o + 14];
    }
    const iw = 1 / W;
    const mx = sx * iw, my = sy * iw, mz = sz * iw;
    const M2 = out.M2;
    M2.fill(0);
    for (let t = 0; t < count; t++) {
        const o = members[t] * CACHE_STRIDE;
        const m = SC[o + 11];
        const dx = SC[o] - mx, dy = SC[o + 1] - my, dz = SC[o + 2] - mz;
        M2[0] += m * (SC[o + 3] + dx * dx);
        M2[1] += m * (SC[o + 4] + dx * dy);
        M2[2] += m * (SC[o + 5] + dx * dz);
        M2[3] += m * (SC[o + 6] + dy * dy);
        M2[4] += m * (SC[o + 7] + dy * dz);
        M2[5] += m * (SC[o + 8] + dz * dz);
    }
    out.W = W; out.mx = mx; out.my = my; out.mz = mz;
    out.bw0 = b0; out.bw1 = b1; out.bw2 = b2;
};

// Compose A∪B's raw aggregates from A's and B's (parallel-axis identity —
// the exact arithmetic the merged-covariance step has always used).
const composeUnion = (a: Comp, b: Comp, out: Comp): void => {
    const WA = a.W, WB = b.W, WC = WA + WB;
    const iw = 1 / WC;
    const mcx = (WA * a.mx + WB * b.mx) * iw;
    const mcy = (WA * a.my + WB * b.my) * iw;
    const mcz = (WA * a.mz + WB * b.mz) * iw;
    const dax = a.mx - mcx, day = a.my - mcy, daz = a.mz - mcz;
    const dbx = b.mx - mcx, dby = b.my - mcy, dbz = b.mz - mcz;
    const MA = a.M2, MB = b.M2, MC = out.M2;
    MC[0] = MA[0] + MB[0] + WA * dax * dax + WB * dbx * dbx;
    MC[1] = MA[1] + MB[1] + WA * dax * day + WB * dbx * dby;
    MC[2] = MA[2] + MB[2] + WA * dax * daz + WB * dbx * dbz;
    MC[3] = MA[3] + MB[3] + WA * day * day + WB * dby * dby;
    MC[4] = MA[4] + MB[4] + WA * day * daz + WB * dby * dbz;
    MC[5] = MA[5] + MB[5] + WA * daz * daz + WB * dbz * dbz;
    out.W = WC; out.mx = mcx; out.my = mcy; out.mz = mcz;
    out.bw0 = a.bw0 + b.bw0; out.bw1 = a.bw1 + b.bw1; out.bw2 = a.bw2 + b.bw2;
};

// Derive the merged Gaussian's field quantities from the raw aggregates.
const finishComp = (c: Comp): void => {
    const iw = 1 / c.W;
    const sm = c.sm;
    sm[0] = c.M2[0] * iw + EPS_COV;
    sm[1] = c.M2[1] * iw;
    sm[2] = c.M2[2] * iw;
    sm[3] = c.M2[3] * iw + EPS_COV;
    sm[4] = c.M2[4] * iw;
    sm[5] = c.M2[5] * iw + EPS_COV;

    const detm = Math.max(
        sm[0] * (sm[3] * sm[5] - sm[4] * sm[4]) - sm[1] * (sm[1] * sm[5] - sm[4] * sm[2]) + sm[2] * (sm[1] * sm[4] - sm[3] * sm[2]),
        1e-60
    );
    c.sd = Math.sqrt(detm);

    eig3(sm[0], sm[1], sm[2], sm[3], sm[4], sm[5]);
    const s0 = Math.sqrt(Math.max(eigOut[0], 1e-18));
    const s1 = Math.sqrt(Math.max(eigOut[1], 1e-18));
    const s2 = Math.sqrt(Math.max(eigOut[2], 1e-18));
    c.alpha = Math.min(1, c.W / Math.max(ellipsoidArea(s0, s1, s2), 1e-30));

    c.bc0 = c.bw0 * iw; c.bc1 = c.bw1 * iw; c.bc2 = c.bw2 * iw;
    c.bn2 = c.bc0 * c.bc0 + c.bc1 * c.bc1 + c.bc2 * c.bc2;
    c.selfM = c.alpha * c.alpha * c.bn2 * PI_1_5 * c.sd;
};

// memfm(C over `members`) = Σ ⟨f_k, f_C⟩ for the finished cluster `c`.
const memfm = (SC: Float32Array, members: Uint32Array, count: number, c: Comp): number => {
    const sm = c.sm;
    let acc = 0;
    for (let t = 0; t < count; t++) {
        const o = members[t] * CACHE_STRIDE;
        const wgt = SC[o + 10] * c.alpha *
            (SC[o + 12] * c.bc0 + SC[o + 13] * c.bc1 + SC[o + 14] * c.bc2);
        if (wgt === 0) continue;
        acc += wgt * crossG(SC[o + 9] * c.sd,
            SC[o + 3] + sm[0], SC[o + 4] + sm[1], SC[o + 5] + sm[2],
            SC[o + 6] + sm[3], SC[o + 7] + sm[4], SC[o + 8] + sm[5],
            SC[o] - c.mx, SC[o + 1] - c.my, SC[o + 2] - c.mz);
    }
    return acc;
};

// A singleton's self product ⟨f, f⟩ (its cluster error is exactly 0).
const selfRow = (SC: Float32Array, row: number): number => {
    const o = row * CACHE_STRIDE;
    return SC[o + 10] * SC[o + 10] * SC[o + 15] * PI_1_5 * SC[o + 9];
};

// Member buffers (grow-only; production maxGroup is 4).
let aBuf: Uint32Array = new Uint32Array(1 << 12);
let bBuf: Uint32Array = new Uint32Array(1 << 12);
const gatherChain = (st: RecostState, root: number, into: Uint32Array): { buf: Uint32Array, count: number } => {
    const { mHead, mNext } = st;
    let buf = into;
    let cnt = 0;
    for (let m = mHead[root]; m !== NIL; m = mNext[m]) {
        if (cnt === buf.length) {
            const g = new Uint32Array(buf.length * 2);
            g.set(buf); buf = g;
        }
        buf[cnt++] = m;
    }
    return { buf, count: cnt };
};

// One side's term of the cancelled cost form, given the side's raw
// aggregates (finishes `c` when the side is a real cluster).
const sideTerm = (SC: Float32Array, members: Uint32Array, count: number, c: Comp): number => {
    if (count < 2) return selfRow(SC, members[0]);
    finishComp(c);
    return 2 * memfm(SC, members, count, c) - c.selfM;
};

// scross(A,B) = Σ_{a∈A,b∈B} ⟨f_a, f_b⟩ with distance culling.
const scrossPairs = (
    SC: Float32Array,
    aMembers: Uint32Array, na: number,
    bMembers: Uint32Array, nb: number
): number => {
    let acc = 0;
    for (let u = 0; u < nb; u++) {
        const ob = bMembers[u] * CACHE_STRIDE;
        const bxp = SC[ob], byp = SC[ob + 1], bzp = SC[ob + 2];
        const trb = SC[ob + 3] + SC[ob + 6] + SC[ob + 8];
        const alb = SC[ob + 10], sdb = SC[ob + 9];
        const cb0 = SC[ob + 12], cb1 = SC[ob + 13], cb2 = SC[ob + 14];
        for (let t = 0; t < na; t++) {
            const oa = aMembers[t] * CACHE_STRIDE;
            const dx = SC[oa] - bxp, dy = SC[oa + 1] - byp, dz = SC[oa + 2] - bzp;
            const d2 = dx * dx + dy * dy + dz * dz;
            if (d2 > CULL_QUAD * (SC[oa + 3] + SC[oa + 6] + SC[oa + 8] + trb)) continue;
            const wgt = SC[oa + 10] * alb *
                (SC[oa + 12] * cb0 + SC[oa + 13] * cb1 + SC[oa + 14] * cb2);
            if (wgt === 0) continue;
            acc += wgt * crossG(SC[oa + 9] * sdb,
                SC[oa + 3] + SC[ob + 3], SC[oa + 4] + SC[ob + 4], SC[oa + 5] + SC[ob + 5],
                SC[oa + 6] + SC[ob + 6], SC[oa + 7] + SC[ob + 7], SC[oa + 8] + SC[ob + 8],
                dx, dy, dz);
        }
    }
    return acc;
};

// Evaluate ΔE for A (context already composed in compA/aBuf, side term
// `aTerm`) against cluster B, in the cancelled form.
const evalAgainst = (st: RecostState, na: number, aTerm: number, B: number): number => {
    const { SC } = st;
    const b = gatherChain(st, B, bBuf);
    bBuf = b.buf;
    const nb = b.count;
    composeRaw(SC, bBuf, nb, compB);
    const bTerm = sideTerm(SC, bBuf, nb, compB);

    composeUnion(compA, compB, compAB);
    finishComp(compAB);

    const memfmAB = memfm(SC, aBuf, na, compAB) + memfm(SC, bBuf, nb, compAB);
    const scross = scrossPairs(SC, aBuf, na, bBuf, nb);

    // Scale-free DC colour term between the clusters' mean base colours.
    const iwA = 1 / compA.W, iwB = 1 / compB.W;
    const d0 = compA.bw0 * iwA - compB.bw0 * iwB;
    const d1 = compA.bw1 * iwA - compB.bw1 * iwB;
    const d2c = compA.bw2 * iwA - compB.bw2 * iwB;

    return (2 * scross - 2 * memfmAB + compAB.selfM + aTerm + bTerm) +
        COLOR_WEIGHT * (d0 * d0 + d1 * d1 + d2c * d2c);
};

/**
 * Marginal cost ΔE of merging clusters A and B (exact field-L2 vs the
 * generation-input members) plus the scale-free colour term — recomputed
 * statelessly from the members' cache rows.
 *
 * @param st - Selection state.
 * @param A - First cluster root.
 * @param B - Second cluster root.
 * @returns The merge cost.
 */
const evalMergeCore = (st: RecostState, A: number, B: number): number => {
    const a = gatherChain(st, A, aBuf);
    aBuf = a.buf;
    const na = a.count;
    composeRaw(st.SC, aBuf, na, compA);
    const aTerm = sideTerm(st.SC, aBuf, na, compA);
    return evalAgainst(st, na, aTerm, B);
};

/** Result of {@link bestEdgeFor} (reused object). */
const bestOut = { partner: -1, vb: 0, cost: 0 };
const partitionBestOut = {
    corePartner: -1,
    coreVersion: 0,
    coreCost: Infinity,
    haloPartner: -1,
    haloCost: Infinity
};

const candScratch = new Uint32Array(256);

/**
 * Compute the cheapest legal merge for `root`: candidates are the live
 * clusters owning any member's candidate ids, deduplicated linearly (pools
 * are small), respecting the group cap. The root's composition and side
 * term are hoisted — computed once, reused for every candidate.
 *
 * @param st - Selection state.
 * @param root - Cluster root to refresh.
 * @param coreCount - Rows below this boundary are mutable core candidates.
 * @returns True when a legal candidate exists (result in {@link bestOut}).
 */
const bestEdgesForPartition = (st: RecostState, root: number, coreCount: number): boolean => {
    const { SC, cands, D, parent, size, version, maxGroup } = st;
    partitionBestOut.corePartner = -1;
    partitionBestOut.coreVersion = 0;
    partitionBestOut.coreCost = Infinity;
    partitionBestOut.haloPartner = -1;
    partitionBestOut.haloCost = Infinity;
    if (maxGroup * D > candScratch.length) {
        throw new Error(`recost: candidate pool bound ${maxGroup * D} exceeds scratch (${candScratch.length})`);
    }
    const sz = size[root];

    const a = gatherChain(st, root, aBuf);
    aBuf = a.buf;
    const na = a.count;

    // Derive + dedup candidates (linear scan — pools are ≤ maxGroup·D).
    let cnt = 0;
    const cbuf = candScratch;
    for (let t = 0; t < na; t++) {
        const base = aBuf[t] * D;
        for (let s = 0; s < D; s++) {
            const nb = cands[base + s];
            if (nb === NO_CANDIDATE) continue;
            const r = findRO(parent, nb);
            if (r === root) continue;
            let seen = false;
            for (let u = 0; u < cnt; u++) {
                if (cbuf[u] === r) {
                    seen = true; break;
                }
            }
            if (!seen) cbuf[cnt++] = r;
        }
    }
    if (cnt === 0) return false;

    composeRaw(SC, aBuf, na, compA);
    const aTerm = sideTerm(SC, aBuf, na, compA);

    let bc = Infinity, bp = -1, bv = 0;
    let hc = Infinity, hp = -1;
    for (let t = 0; t < cnt; t++) {
        const c = cbuf[t];
        if (sz + size[c] > maxGroup) continue;
        const d = evalAgainst(st, na, aTerm, c);
        if (c < coreCount) {
            if (d < bc || (d === bc && c < bp)) {
                bc = d; bp = c; bv = version[c];
            }
        } else if (d < hc || (d === hc && c < hp)) {
            hc = d; hp = c;
        }
    }
    partitionBestOut.corePartner = bp;
    partitionBestOut.coreVersion = bv;
    partitionBestOut.coreCost = bc;
    partitionBestOut.haloPartner = hp;
    partitionBestOut.haloCost = hc;
    return bp >= 0 || hp >= 0;
};

const bestEdgeFor = (st: RecostState, root: number): boolean => {
    if (!bestEdgesForPartition(st, root, st.N) || partitionBestOut.corePartner < 0) return false;
    bestOut.partner = partitionBestOut.corePartner;
    bestOut.vb = partitionBestOut.coreVersion;
    bestOut.cost = partitionBestOut.coreCost;
    return true;
};

export {
    evalMergeCore,
    bestEdgeFor,
    bestEdgesForPartition,
    bestOut,
    partitionBestOut,
    NO_CANDIDATE,
    type RecostState
};
