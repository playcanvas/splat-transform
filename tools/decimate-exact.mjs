#!/usr/bin/env node
/**
 * Reference "quality ceiling" decimator — exact greedy agglomeration.
 *
 * The algorithm the production pipeline approximates, run without the
 * approximations, to establish the best decimation quality achievable under
 * the merge/moment-match model:
 *
 *   - One merge at a time: pop the globally-cheapest candidate merge, commit
 *     it, re-evaluate the affected candidates, repeat. No batch selection, no
 *     stale costs — after every commit the merged cluster's candidates are
 *     re-costed against its *current* state.
 *   - Cost of a cluster C = exact L2 field error vs the ORIGINAL scene:
 *         E(C) = || Σ_{k∈C} f_k  −  f_m(C) ||²
 *     where f_k = α_k·c_k·G_k are the original member fields (DC colour) and
 *     f_m(C) is the single moment-matched Gaussian of ALL original members
 *     (same math as production mergeGroup). A candidate merge's cost is the
 *     marginal error ΔE = E(A∪B) − E(A) − E(B). Because E is always measured
 *     against the originals, chained/stretched clusters price in their full
 *     accumulated error — nothing is hidden by incremental approximations.
 *   - Nested LOD levels from ONE run: keep merging and snapshot a PLY at each
 *     target count. Every level approximates the ORIGINAL scene, not the
 *     previous level, so there is no compounding across levels.
 *
 * Deliberately not fast: everything resident, single-threaded greedy loop.
 * Performance work comes after the quality ceiling is confirmed.
 *
 * Usage:
 *   node --import tsx --max-old-space-size=49152 tools/decimate-exact.mjs \
 *     --input scenes/bad-sky.ply --out-prefix scenes/bad-sky.exact --halvings 3
 *   node --import tsx tools/decimate-exact.mjs --selftest
 *
 * Emits <out-prefix>1.ply, <out-prefix>2.ply, ... (level counts = successive
 * ceil(N/2), matching the production 50% cascade exactly).
 */

import { openSync, writeSync, closeSync } from 'node:fs';

import { knnForestQuery, KNN_SENTINEL } from '../src/lib/decimate/knn-core.js';
import {
    EPS_COV, sigmoid, logit, ellipsoidArea, quatToRotmat, sigmaFromRotVar,
    det3, eigenSymmetric3x3, rotmatToQuat, mergeGroup, createMergeScratch,
    makeGaussianSamples, gaussLogpdfDiagrot, logAddExp, LOG2PI
} from '../src/lib/decimate/moment-match.js';
import { buildSplatCache, computeEdgeCost, CACHE_STRIDE } from '../src/lib/decimate/edge-cost-cpu.js';
import { createChunkDataPool } from '../src/lib/index.js';
import { readPly } from '../src/lib/readers/read-ply.js';
import { buildFlatKdTree } from '../src/lib/spatial/kd-tree.js';
import { NodeReadFileSystem } from '../src/cli/node-file-system.js';

const C0 = 0.28209479177387814;
const PI_1_5 = Math.PI ** 1.5;
const TWO_PI_1_5 = (2 * Math.PI) ** 1.5;
const KNN_K = 16;
/** Skip Gaussian products whose exponent bound exceeds this (e^-60 ≈ 9e-27). */
const CULL_QUAD = 120;

const log = (msg) => console.log(`[${(performance.now() / 1000).toFixed(1)}s] ${msg}`);

// ---------------------------------------------------------------------------
// Engine state (module-level typed arrays, sized once by initEngine).
// Clusters are union-find sets over original indices; all per-cluster state is
// indexed by the set's root. Original (per-splat) caches never change.
// ---------------------------------------------------------------------------

let N = 0, colorDim = 3;

// Per-original caches (immutable after init).
let px, py, pz;              // f32 positions
let cs6;                     // f32 6N — Σ_k with EPS on the variance diagonal (product path)
let csd;                     // f32 N — √|Σ_k| (with-EPS)
let cal;                     // f32 N — α_k
let cmass;                   // f32 N — α·area + 1e-30 (merge weight)
let ctr;                     // f32 N — trace(Σ_k) for product culling
let cb;                      // f32 3N — DC base colour (0.5 + C0·f_dc)
let cbn2;                    // f32 N — |base|²
let knn;                     // u32 KNN_K·N — global neighbour ids (or KNN_SENTINEL)

// Per-cluster state (valid at union-find roots).
let parent, ufsize;          // u32
let W;                       // f64 — total mass
let mx, my, mz;              // f64 — mass-weighted mean
let M2;                      // f64 6N — central second moments Σ w(δδᵀ + Σ_noEPS)
let colorW;                  // f32 colorDim·N — mass-weighted raw colour sums
let Sself;                   // f64 — Σ_{k,l∈C}⟨f_k,f_l⟩
let Err;                     // f64 — E(C) vs originals
let version, lastSeq;        // u32
let mHead, mTail, mNext;     // u32 member chains (NIL = 0xFFFFFFFF)
const NIL = 0xFFFFFFFF;

let liveCount = 0;

/** Max original members per cluster (Infinity = uncapped reference). */
let maxGroup = Infinity;

/**
 * Size-normalization exponent p: merge cost = ΔE / σ_merged^p, with σ the
 * merged Gaussian's geometric-mean std (√|Σ|^{1/3}). p=0 is the pure field-L2
 * (volume, σ³-weighted); p=1 ≈ area (σ²) weighting; p=3 ≈ scale-free
 * (uniform-rate) behavior. Interpolates the absolute↔relative spectrum.
 */
let sizeExponent = 0;

/**
 * Viewing-kernel dilation δ (world units): the field-L2 metric is evaluated
 * after convolving both sides with N(0, δ²I) — every covariance in the
 * products gains +δ²I and amplitudes dilute by √(|Σ|/|Σ+δ²I|) (mip-style).
 * Prices merges as seen at resolution δ; output parameters are unaffected.
 */
let dilate = 0;

/** Cost mode: 'l2' = field-L2 vs originals; 'kl' = legacy pairwise KL between cluster reps. */
let costMode = 'l2';

/**
 * Scale-free colour dissimilarity weight λ: cost += λ·Σ_allCoeffs(Δc)² between
 * the two clusters' mass-weighted mean colours. Unlike the field-L2's own
 * colour sensitivity (which vanishes ∝σ³ for faint splats), this term keeps
 * light-vs-dark pairing selective at any scale (old's one good property).
 * Ordering-only: Err/E bookkeeping stays pure field-L2.
 */
let colorWeight = 0;

/** Restrict the λ colour term to the 3 DC coefficients (500M-residency probe). */
let colorDcOnly = false;

/**
 * Needle-chaining guard: a merge is forbidden (cost = ∞) when the result is
 * BOTH longer than either member (σmax > LEN_TOL × member max) AND still
 * needle-like (σmax/σmid ≥ NEEDLE_TOL × member max needleness). End-to-end
 * chaining of the scan's native thin splats (which otherwise compounds
 * 0.2m source needles into metre-long artifacts) fails both tests; legitimate
 * merges pass at least one: side-by-side joins thicken (needleness drops),
 * and flat pancakes (sky clouds) have σmax/σmid ≈ 1 throughout. Relative
 * everywhere — nothing is grandfathered, no scene-scale constants.
 */
let needleGuard = false;
const LEN_TOL = 1.15;
/** Absolute needle-shape threshold: σmax/σmid above this reads as a needle. */
const NEEDLE_ABS = 6;
let clen;    // f32[N] — max σmax over the cluster's ORIGINAL members (anchor, no ratchet)
let cneedle; // f32[N] — cluster rep σmax/σmid (diagnostics)

const initEngine = (n, dim) => {
    N = n; colorDim = dim;
    px = new Float32Array(N); py = new Float32Array(N); pz = new Float32Array(N);
    cs6 = new Float32Array(N * 6);
    csd = new Float32Array(N); cal = new Float32Array(N); cmass = new Float32Array(N);
    ctr = new Float32Array(N); cb = new Float32Array(N * 3); cbn2 = new Float32Array(N);
    knn = new Uint32Array(N * KNN_K).fill(KNN_SENTINEL);
    parent = new Uint32Array(N); ufsize = new Uint32Array(N).fill(1);
    W = new Float64Array(N);
    mx = new Float64Array(N); my = new Float64Array(N); mz = new Float64Array(N);
    M2 = new Float64Array(N * 6);
    colorW = new Float32Array(N * colorDim);
    Sself = new Float64Array(N); Err = new Float64Array(N);
    version = new Uint32Array(N).fill(1); lastSeq = new Uint32Array(N);
    clen = new Float32Array(N); cneedle = new Float32Array(N).fill(1);
    mHead = new Uint32Array(N); mTail = new Uint32Array(N); mNext = new Uint32Array(N).fill(NIL);
    for (let i = 0; i < N; i++) { parent[i] = i; mHead[i] = i; mTail[i] = i; }
    liveCount = N;
};

// Initialize original i (and its singleton cluster) from raw layer values:
// pos3 floats, geo8 = rot(4 wxyz) + log-scales(3) + logit-opacity, colour row.
const initRow = (i, x, y, z, geo, g8, color, cOff) => {
    px[i] = x; py[i] = y; pz[i] = z;

    let qw = geo[g8], qx = geo[g8 + 1], qy = geo[g8 + 2], qz = geo[g8 + 3];
    const invq = 1 / Math.max(Math.hypot(qw, qx, qy, qz), 1e-12);
    qw *= invq; qx *= invq; qy *= invq; qz *= invq;
    const sx = Math.max(Math.exp(geo[g8 + 4]), 1e-12);
    const sy = Math.max(Math.exp(geo[g8 + 5]), 1e-12);
    const sz = Math.max(Math.exp(geo[g8 + 6]), 1e-12);
    const alpha = sigmoid(geo[g8 + 7]);

    // Product-path caches carry the viewing-kernel dilation (Σ+δ²I with
    // amplitude dilution √(|Σ|/|Σ+δ²I|)); moments/emission stay undilated.
    const d2 = dilate * dilate;
    const vx = sx * sx + EPS_COV, vy = sy * sy + EPS_COV, vz = sz * sz + EPS_COV;
    const R = evR, S = evS; // scratch
    quatToRotmat(qw, qx, qy, qz, R, 0);
    sigmaFromRotVar(R, 0, vx + d2, vy + d2, vz + d2, S, 0);
    const i6 = i * 6;
    cs6[i6] = S[0]; cs6[i6 + 1] = S[1]; cs6[i6 + 2] = S[2];
    cs6[i6 + 3] = S[4]; cs6[i6 + 4] = S[5]; cs6[i6 + 5] = S[8];
    ctr[i] = S[0] + S[4] + S[8];
    const detU = vx * vy * vz;
    const detD = (vx + d2) * (vy + d2) * (vz + d2);
    csd[i] = Math.sqrt(Math.max(detD, 1e-60));
    cal[i] = alpha * Math.sqrt(detU / detD);
    const mass = alpha * ellipsoidArea(sx, sy, sz) + 1e-30;
    cmass[i] = mass;
    const sHi = Math.max(sx, sy, sz), sLo = Math.min(sx, sy, sz);
    const sMid = sx + sy + sz - sHi - sLo;
    clen[i] = sHi;
    cneedle[i] = sHi / Math.max(sMid, 1e-12);

    const b0 = 0.5 + C0 * color[cOff];
    const b1 = 0.5 + C0 * color[cOff + 1];
    const b2 = 0.5 + C0 * color[cOff + 2];
    cb[i * 3] = b0; cb[i * 3 + 1] = b1; cb[i * 3 + 2] = b2;
    cbn2[i] = b0 * b0 + b1 * b1 + b2 * b2;

    // Singleton cluster: exact moments of one member (undilated, Σ WITHOUT the
    // EPS used by the product path — matches production mergeGroup member math;
    // the dilated cache S carries EPS+δ² on its diagonal).
    W[i] = mass;
    mx[i] = x; my[i] = y; mz[i] = z;
    M2[i6] = mass * (S[0] - EPS_COV - d2);
    M2[i6 + 1] = mass * S[1];
    M2[i6 + 2] = mass * S[2];
    M2[i6 + 3] = mass * (S[4] - EPS_COV - d2);
    M2[i6 + 4] = mass * S[5];
    M2[i6 + 5] = mass * (S[8] - EPS_COV - d2);
    for (let c = 0; c < colorDim; c++) colorW[i * colorDim + c] = mass * color[cOff + c];
    Sself[i] = cal[i] * cal[i] * cbn2[i] * PI_1_5 * csd[i];
    Err[i] = 0;
};
const evR = new Float32Array(9), evS = new Float32Array(9);

const find = (x) => {
    while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
};

// ---------------------------------------------------------------------------
// Cost evaluation.
// ---------------------------------------------------------------------------

// ⟨G_a,G_b⟩ scaled by √|Σa|·√|Σb| for M = Σa+Σb (6 comps) and offset d.
const crossG = (sdAB, m0, m1, m2, m3, m4, m5, dx, dy, dz) => {
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

// Smith closed-form eigenvalues of a symmetric 3×3 (6 comps) — for merged area.
const eig3 = (m0, m1, m2, m3, m4, m5, out) => {
    const q = (m0 + m3 + m5) / 3;
    const p1 = m1 * m1 + m2 * m2 + m4 * m4;
    if (p1 <= 1e-30) { out[0] = m0; out[1] = m3; out[2] = m5; return; }
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
    const e2 = q + 2 * p * Math.cos(phi + 2 * Math.PI / 3);
    out[0] = e0; out[1] = 3 * q - e0 - e2; out[2] = e2;
};
const eigOut = new Float64Array(3);

// Member gather scratch (cluster A's members flattened for the Scross loop).
let gatherBuf = new Uint32Array(1 << 16);
const gatherMembers = (root) => {
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

// ---- Legacy-KL cost mode: pairwise cost between the two clusters' current
// representative gaussians (verbatim port of the legacy computeEdgeCost:
// KL-style geometric term with one MC sample + L2 over colour coefficients).
// O(1) per eval regardless of member count; incremental (not vs originals).
const klZ = makeGaussianSamples(1, 0)[0];
const klS = {
    SigA: new Float64Array(9), SigB: new Float64Array(9),
    eigA: new Float64Array(9), eigV: new Float64Array(9),
    RA: new Float64Array(9), RB: new Float64Array(9),
    a: new Float64Array(8), b: new Float64Array(8), // v3, invd3... packed below
    sigm: new Float64Array(9)
};
// Fill side params for cluster C: Sig (9), R (9), out = [vx,vy,vz,ldet,mass].
const klSide = (C, Sig, R, out) => {
    const i6 = C * 6, iw = 1 / W[C];
    Sig[0] = M2[i6] * iw + EPS_COV;
    Sig[1] = Sig[3] = M2[i6 + 1] * iw;
    Sig[2] = Sig[6] = M2[i6 + 2] * iw;
    Sig[4] = M2[i6 + 3] * iw + EPS_COV;
    Sig[5] = Sig[7] = M2[i6 + 4] * iw;
    Sig[8] = M2[i6 + 5] * iw + EPS_COV;
    eigenSymmetric3x3(Sig, klS.eigA, klS.eigV);
    R.set(klS.eigV);
    const v0 = Math.max(klS.eigA[0], 1e-30);
    const v1 = Math.max(klS.eigA[4], 1e-30);
    const v2 = Math.max(klS.eigA[8], 1e-30);
    out[0] = v0; out[1] = v1; out[2] = v2;
    out[3] = Math.log(v0) + Math.log(v1) + Math.log(v2);
    const s0 = Math.sqrt(v0), s1 = Math.sqrt(v1), s2 = Math.sqrt(v2);
    const area = ellipsoidArea(s0, s1, s2);
    const alphaM = Math.min(1, W[C] / Math.max(area, 1e-30));
    out[4] = alphaM * area + 1e-12;
};

const evalKl = (A, B) => {
    evalOut.E = 0; evalOut.Scross = 0;
    const { SigA, SigB, RA, RB, a, b, sigm } = klS;
    klSide(A, SigA, RA, a);
    klSide(B, SigB, RB, b);

    const mux = mx[A], muy = my[A], muz = mz[A];
    const mvx = mx[B], mvy = my[B], mvz = mz[B];
    const wi = a[4], wj = b[4];
    const Wsafe = wi + wj > 0 ? wi + wj : 1;
    let pi_ = wi / Wsafe;
    pi_ = Math.max(1e-12, Math.min(1 - 1e-12, pi_));
    const pj_ = 1 - pi_;
    const logPi = Math.log(pi_), logPj = Math.log(pj_);

    const mmx = pi_ * mux + pj_ * mvx;
    const mmy = pi_ * muy + pj_ * mvy;
    const mmz = pi_ * muz + pj_ * mvz;
    const dix = mux - mmx, diy = muy - mmy, diz = muz - mmz;
    const djx = mvx - mmx, djy = mvy - mmy, djz = mvz - mmz;

    for (let t = 0; t < 9; t++) sigm[t] = pi_ * SigA[t] + pj_ * SigB[t];
    sigm[0] += pi_ * dix * dix + pj_ * djx * djx + EPS_COV;
    sigm[1] += pi_ * dix * diy + pj_ * djx * djy;
    sigm[2] += pi_ * dix * diz + pj_ * djx * djz;
    sigm[3] = sigm[1];
    sigm[4] += pi_ * diy * diy + pj_ * djy * djy + EPS_COV;
    sigm[5] += pi_ * diy * diz + pj_ * djy * djz;
    sigm[6] = sigm[2]; sigm[7] = sigm[5];
    sigm[8] += pi_ * diz * diz + pj_ * djz * djz + EPS_COV;
    const detm = Math.max(det3(sigm, 0), 1e-30);
    const EpNegLogQ = 0.5 * (3 * LOG2PI + Math.log(detm) + 3);

    const z0 = klZ[0], z1 = klZ[1], z2 = klZ[2];
    const sia = Math.sqrt(a[0]), sib = Math.sqrt(a[1]), sic = Math.sqrt(a[2]);
    const sja = Math.sqrt(b[0]), sjb = Math.sqrt(b[1]), sjc = Math.sqrt(b[2]);
    const xix = mux + z0 * sia * RA[0] + z1 * sib * RA[1] + z2 * sic * RA[2];
    const xiy = muy + z0 * sia * RA[3] + z1 * sib * RA[4] + z2 * sic * RA[5];
    const xiz = muz + z0 * sia * RA[6] + z1 * sib * RA[7] + z2 * sic * RA[8];
    const xjx = mvx + z0 * sja * RB[0] + z1 * sjb * RB[1] + z2 * sjc * RB[2];
    const xjy = mvy + z0 * sja * RB[3] + z1 * sjb * RB[4] + z2 * sjc * RB[5];
    const xjz = mvz + z0 * sja * RB[6] + z1 * sjb * RB[7] + z2 * sjc * RB[8];

    const ia = 1 / a[0], ib = 1 / a[1], ic = 1 / a[2];
    const ja = 1 / b[0], jb = 1 / b[1], jc = 1 / b[2];
    const logNiOnI = gaussLogpdfDiagrot(xix, xiy, xiz, mux, muy, muz, RA, 0, ia, ib, ic, a[3]);
    const logNjOnI = gaussLogpdfDiagrot(xix, xiy, xiz, mvx, mvy, mvz, RB, 0, ja, jb, jc, b[3]);
    const logNiOnJ = gaussLogpdfDiagrot(xjx, xjy, xjz, mux, muy, muz, RA, 0, ia, ib, ic, a[3]);
    const logNjOnJ = gaussLogpdfDiagrot(xjx, xjy, xjz, mvx, mvy, mvz, RB, 0, ja, jb, jc, b[3]);
    const Ei = logAddExp(logPi + logNiOnI, logPj + logNjOnI);
    const Ej = logAddExp(logPi + logNiOnJ, logPj + logNjOnJ);
    const geo = pi_ * Ei + pj_ * Ej + EpNegLogQ;

    let cSh = 0;
    const cd = colorDim, iwA = 1 / W[A], iwB = 1 / W[B];
    for (let c = 0; c < cd; c++) {
        const d = colorW[A * cd + c] * iwA - colorW[B * cd + c] * iwB;
        cSh += d * d;
    }
    return geo + cSh;
};

// Marginal cost ΔE of merging clusters A and B (exact, vs originals).
// Fills evalOut with E(A∪B) and Scross(A,B) for reuse at commit.
const evalMerge = (A, B) => {
    if (costMode === 'kl') return evalKl(A, B);
    const WA = W[A], WB = W[B], WC = WA + WB;
    const iw = 1 / WC;
    const mcx = (WA * mx[A] + WB * mx[B]) * iw;
    const mcy = (WA * my[A] + WB * my[B]) * iw;
    const mcz = (WA * mz[A] + WB * mz[B]) * iw;
    const dax = mx[A] - mcx, day = my[A] - mcy, daz = mz[A] - mcz;
    const dbx = mx[B] - mcx, dby = my[B] - mcy, dbz = mz[B] - mcz;
    const a6 = A * 6, b6 = B * 6;

    // Merged covariance Σm = (M2_A + M2_B + shift terms)/W + EPS·I.
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

    // Needle-chaining guard: a needle-shaped result (σmax/σmid > NEEDLE_ABS)
    // may never be longer than LEN_TOL × the longest ORIGINAL member — the
    // original-length anchor kills the per-merge growth ratchet. Thickening is
    // always allowed (aspect ≤ NEEDLE_ABS passes unconditionally). Checked
    // before the expensive member loops.
    if (needleGuard) {
        const needleM = s0 / Math.max(s1, 1e-12);
        if (needleM > NEEDLE_ABS && s0 > LEN_TOL * Math.max(clen[A], clen[B])) return Infinity;
    }

    const alphaC = Math.min(1, WC / Math.max(ellipsoidArea(s0, s1, s2), 1e-30));

    const cd = colorDim;
    const bc0 = 0.5 + C0 * (colorW[A * cd] + colorW[B * cd]) * iw;
    const bc1 = 0.5 + C0 * (colorW[A * cd + 1] + colorW[B * cd + 1]) * iw;
    const bc2 = 0.5 + C0 * (colorW[A * cd + 2] + colorW[B * cd + 2]) * iw;
    const bn2C = bc0 * bc0 + bc1 * bc1 + bc2 * bc2;

    // Viewing-kernel dilation of the merged gaussian for the product terms
    // (member caches are already dilated); amplitude dilutes accordingly.
    const dd = dilate * dilate;
    const smD0 = sm0 + dd, smD3 = sm3 + dd, smD5 = sm5 + dd;
    const detmD = dd === 0 ? detm : Math.max(
        smD0 * (smD3 * smD5 - sm4 * sm4) - sm1 * (sm1 * smD5 - sm4 * sm2) + sm2 * (sm1 * sm4 - smD3 * sm2),
        1e-60
    );
    const sdCD = Math.sqrt(detmD);
    const aCe = dd === 0 ? alphaC : alphaC * Math.sqrt(detm / detmD);

    const selfM = aCe * aCe * bn2C * PI_1_5 * sdCD;

    // ⟨Σ member fields, f_m⟩ over both chains (dilated space).
    let memfm = 0;
    for (let pass = 0; pass < 2; pass++) {
        for (let m = pass === 0 ? mHead[A] : mHead[B]; m !== NIL; m = mNext[m]) {
            const m6 = m * 6;
            const wgt = cal[m] * aCe * (cb[m * 3] * bc0 + cb[m * 3 + 1] * bc1 + cb[m * 3 + 2] * bc2);
            if (wgt === 0) continue;
            memfm += wgt * crossG(csd[m] * sdCD,
                cs6[m6] + smD0, cs6[m6 + 1] + sm1, cs6[m6 + 2] + sm2,
                cs6[m6 + 3] + smD3, cs6[m6 + 4] + sm4, cs6[m6 + 5] + smD5,
                px[m] - mcx, py[m] - mcy, pz[m] - mcz);
        }
    }

    // Scross(A,B) = Σ_{a∈A,b∈B}⟨f_a,f_b⟩ with distance culling.
    const na = gatherMembers(A);
    let scross = 0;
    for (let b = mHead[B]; b !== NIL; b = mNext[b]) {
        const b6i = b * 6, b3 = b * 3;
        const bx = px[b], by = py[b], bz = pz[b];
        const trb = ctr[b], alb = cal[b], sdb = csd[b];
        const cb0 = cb[b3], cb1 = cb[b3 + 1], cb2 = cb[b3 + 2];
        for (let t = 0; t < na; t++) {
            const a = gatherBuf[t];
            const dx = px[a] - bx, dy = py[a] - by, dz = pz[a] - bz;
            const d2 = dx * dx + dy * dy + dz * dz;
            if (d2 > CULL_QUAD * (ctr[a] + trb)) continue;
            const a6i = a * 6, a3 = a * 3;
            const wgt = cal[a] * alb * (cb[a3] * cb0 + cb[a3 + 1] * cb1 + cb[a3 + 2] * cb2);
            if (wgt === 0) continue;
            scross += wgt * crossG(csd[a] * sdb,
                cs6[a6i] + cs6[b6i], cs6[a6i + 1] + cs6[b6i + 1], cs6[a6i + 2] + cs6[b6i + 2],
                cs6[a6i + 3] + cs6[b6i + 3], cs6[a6i + 4] + cs6[b6i + 4], cs6[a6i + 5] + cs6[b6i + 5],
                dx, dy, dz);
        }
    }

    const E = Sself[A] + Sself[B] + 2 * scross - 2 * memfm + selfM;
    evalOut.E = E;
    evalOut.Scross = scross;
    const dE = E - Err[A] - Err[B];
    // Size normalization: σ_gm = (√|Σm|)^{1/3}; cost = ΔE / σ_gm^p.
    let cost = sizeExponent === 0 ? dE : dE / Math.pow(sdC, sizeExponent / 3);
    if (colorWeight > 0) {
        const iwA = 1 / W[A], iwB = 1 / W[B];
        const lim = colorDcOnly ? Math.min(3, cd) : cd;
        let cd2 = 0;
        for (let c = 0; c < lim; c++) {
            const d = colorW[A * cd + c] * iwA - colorW[B * cd + c] * iwB;
            cd2 += d * d;
        }
        cost += colorWeight * cd2;
    }
    return cost;
};

// ---------------------------------------------------------------------------
// Candidate derivation: a cluster's candidates are the live clusters owning
// any original-KNN neighbour of any member. Reuses the static KNN graph.
// ---------------------------------------------------------------------------

let stamp, stampGen = 0;
let candBuf = new Uint32Array(1 << 12);

const deriveCandidates = (root) => {
    stampGen++;
    const gen = stampGen;
    let cnt = 0;
    for (let m = mHead[root]; m !== NIL; m = mNext[m]) {
        const base = m * KNN_K;
        for (let s = 0; s < KNN_K; s++) {
            const nb = knn[base + s];
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

// ---------------------------------------------------------------------------
// Binary min-heap of candidate edges (one live entry per cluster: its current
// best edge). Lazy invalidation via (seq, partner-version).
// ---------------------------------------------------------------------------

let hCost, hA, hB, hSeq, hVb, heapSize = 0, heapCap = 0;
let seqCounter = 0;

const heapInit = (cap) => {
    heapCap = cap;
    hCost = new Float64Array(cap);
    hA = new Uint32Array(cap); hB = new Uint32Array(cap);
    hSeq = new Uint32Array(cap); hVb = new Uint32Array(cap);
    heapSize = 0;
};

const heapPush = (cost, a, b, seq, vb) => {
    if (heapSize === heapCap) {
        const nc = heapCap * 2;
        const c2 = new Float64Array(nc); c2.set(hCost); hCost = c2;
        const g = (old) => { const x = new Uint32Array(nc); x.set(old); return x; };
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

const swap = (i, j) => {
    let t;
    t = hCost[i]; hCost[i] = hCost[j]; hCost[j] = t;
    t = hA[i]; hA[i] = hA[j]; hA[j] = t;
    t = hB[i]; hB[i] = hB[j]; hB[j] = t;
    t = hSeq[i]; hSeq[i] = hSeq[j]; hSeq[j] = t;
    t = hVb[i]; hVb[i] = hVb[j]; hVb[j] = t;
};

const popOut = { cost: 0, a: 0, b: 0, seq: 0, vb: 0 };
const heapPop = () => {
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

// Largest live cluster (diagnostic; O(N) but only called from throttled logs).
const maxLiveSize = () => {
    let mx = 0;
    for (let i = 0; i < N; i++) {
        if (parent[i] === i && ufsize[i] > mx) mx = ufsize[i];
    }
    return mx;
};

// Log2-bucketed live cluster-size histogram, e.g. "1:2041 2-3:511 4-7:88".
const sizeHistogram = () => {
    const buckets = new Map();
    for (let i = 0; i < N; i++) {
        if (parent[i] !== i) continue;
        const b = Math.floor(Math.log2(ufsize[i]));
        buckets.set(b, (buckets.get(b) ?? 0) + 1);
    }
    return [...buckets.entries()].sort((a, b) => a[0] - b[0])
        .map(([b, c]) => `${1 << b}${b > 0 ? `-${(1 << (b + 1)) - 1}` : ''}:${c}`).join(' ');
};

// Recompute a cluster's best edge and push it (bumps lastSeq — older entries
// for this cluster become stale).
const pushBestEdge = (root) => {
    const cnt = deriveCandidates(root);
    lastSeq[root] = ++seqCounter;
    if (cnt === 0) return;
    const sz = ufsize[root];
    let bc = Infinity, bp = -1, bv = 0;
    for (let t = 0; t < cnt; t++) {
        const cand = candBuf[t];
        if (sz + ufsize[cand] > maxGroup) continue;
        const d = evalMerge(root, cand);
        if (d < bc) { bc = d; bp = cand; bv = version[cand]; }
    }
    if (bp >= 0) heapPush(bc, root, bp, lastSeq[root], bv);
};

// Commit the merge of roots A and B. Returns the marginal error added.
const commitMerge = (A, B) => {
    const dE = evalMerge(A, B);          // exact recompute (E, Scross)
    const E = evalOut.E, scross = evalOut.Scross;

    const keep = ufsize[A] >= ufsize[B] ? A : B;
    const lose = keep === A ? B : A;

    // Moments (compute with locals before overwriting keep's slots).
    const WA = W[A], WB = W[B], WC = WA + WB;
    const iw = 1 / WC;
    const mcx = (WA * mx[A] + WB * mx[B]) * iw;
    const mcy = (WA * my[A] + WB * my[B]) * iw;
    const mcz = (WA * mz[A] + WB * mz[B]) * iw;
    const dax = mx[A] - mcx, day = my[A] - mcy, daz = mz[A] - mcz;
    const dbx = mx[B] - mcx, dby = my[B] - mcy, dbz = mz[B] - mcz;
    const a6 = A * 6, b6 = B * 6, k6 = keep * 6;
    const n0 = M2[a6] + M2[b6] + WA * dax * dax + WB * dbx * dbx;
    const n1 = M2[a6 + 1] + M2[b6 + 1] + WA * dax * day + WB * dbx * dby;
    const n2 = M2[a6 + 2] + M2[b6 + 2] + WA * dax * daz + WB * dbx * dbz;
    const n3 = M2[a6 + 3] + M2[b6 + 3] + WA * day * day + WB * dby * dby;
    const n4 = M2[a6 + 4] + M2[b6 + 4] + WA * day * daz + WB * dby * dbz;
    const n5 = M2[a6 + 5] + M2[b6 + 5] + WA * daz * daz + WB * dbz * dbz;
    M2[k6] = n0; M2[k6 + 1] = n1; M2[k6 + 2] = n2; M2[k6 + 3] = n3; M2[k6 + 4] = n4; M2[k6 + 5] = n5;
    W[keep] = WC; mx[keep] = mcx; my[keep] = mcy; mz[keep] = mcz;

    const cd = colorDim, kc = keep * cd, lc = lose * cd;
    for (let c = 0; c < cd; c++) colorW[kc + c] += colorW[lc + c];

    Sself[keep] = Sself[A] + Sself[B] + 2 * scross;
    Err[keep] = E;
    ufsize[keep] += ufsize[lose];
    mNext[mTail[keep]] = mHead[lose];
    mTail[keep] = mTail[lose];
    parent[lose] = keep;
    version[keep]++;
    liveCount--;

    // Maintain guard state: clen = max ORIGINAL member σmax (composes by max —
    // the anchor that prevents compounding growth); cneedle = current rep
    // needleness (diagnostics).
    clen[keep] = Math.max(clen[A], clen[B]);
    eig3(n0 / WC + EPS_COV, n1 / WC, n2 / WC, n3 / WC + EPS_COV, n4 / WC, n5 / WC + EPS_COV, eigOut);
    cneedle[keep] = Math.sqrt(Math.max(eigOut[0], 1e-18) / Math.max(eigOut[1], 1e-18));

    pushBestEdge(keep);
    return dE;
};

// ---------------------------------------------------------------------------
// Main greedy loop with nested snapshots.
// ---------------------------------------------------------------------------

const runGreedy = (targets, onSnapshot, progressEvery = 1_000_000) => {
    log(`initial best edges (${liveCount} clusters)…`);
    for (let i = 0; i < N; i++) {
        pushBestEdge(i);
        if ((i + 1) % 5_000_000 === 0) log(`  seeded ${i + 1}/${N}`);
    }
    log(`greedy loop → targets [${targets.join(', ')}]`);

    let ti = 0;
    let commits = 0, pops = 0, recomputes = 0;
    let totalDE = 0;
    const t0 = performance.now();
    let lastLogAt = t0;

    while (ti < targets.length && liveCount > targets[ti]) {
        if (!heapPop()) {
            log(`heap exhausted at ${liveCount} clusters (target ${targets[ti]}) — emitting partial level and stopping`);
            onSnapshot(ti, liveCount);
            ti++;
            break;
        }
        pops++;
        const a = popOut.a;
        // Stale checks: a must still be a live root and this must be its
        // latest entry (seq is globally monotonic, so no collisions); the
        // partner must still be a live root with an unchanged version —
        // find(b) is NOT sufficient (an absorbed b resolves to a different
        // cluster whose independent version counter can coincidentally match,
        // committing a stale cost and bypassing the cap).
        if (parent[a] !== a || popOut.seq !== lastSeq[a]) continue;
        const b = popOut.b;
        if (parent[b] !== b || version[b] !== popOut.vb ||
            ufsize[a] + ufsize[b] > maxGroup /* defensive: cap must hold */) {
            recomputes++;
            pushBestEdge(a);
            continue;
        }
        totalDE += commitMerge(a, b);
        commits++;

        if (commits % progressEvery === 0 || (commits & 0xFFF) === 0) {
            const now = performance.now();
            if (commits % progressEvery === 0 || now - lastLogAt > 30_000) {
                lastLogAt = now;
                const dt = (now - t0) / 1000;
                log(`  ${commits} merges (${(commits / dt).toFixed(0)}/s avg) · live ${liveCount} · heap ${heapSize} · pops ${pops} · recomputes ${recomputes} · maxSize ${maxLiveSize()} · ΣΔE ${totalDE.toExponential(3)}`);
            }
        }
        if (liveCount === targets[ti]) {
            log(`snapshot ${ti + 1}: ${liveCount} clusters · ΣΔE ${totalDE.toExponential(4)} · sizes ${sizeHistogram()}`);
            onSnapshot(ti, liveCount);
            ti++;
        }
    }
    const dt = (performance.now() - t0) / 1000;
    log(`greedy done: ${commits} merges in ${dt.toFixed(1)}s · ${recomputes} recomputes · ${pops} pops`);
};

// ---------------------------------------------------------------------------
// Emission: moment-match every live cluster (identical math to mergeGroup) and
// stream a binary-little-endian PLY.
// ---------------------------------------------------------------------------

const emitPly = (path) => {
    const props = ['x', 'y', 'z'];
    props.push('f_dc_0', 'f_dc_1', 'f_dc_2');
    for (let r = 0; r < colorDim - 3; r++) props.push(`f_rest_${r}`);
    props.push('opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3');

    const header = `ply\nformat binary_little_endian 1.0\nelement vertex ${liveCount}\n${props.map(p => `property float ${p}`).join('\n')}\nend_header\n`;
    const stride = props.length * 4;

    const fd = openSync(path, 'w');
    writeSync(fd, Buffer.from(header, 'ascii'));

    const ROWS = 65536;
    const buf = Buffer.allocUnsafe(ROWS * stride);
    const f32 = new Float32Array(buf.buffer, buf.byteOffset, ROWS * (stride >> 2));
    let rows = 0, written = 0;

    const Sig = new Float64Array(9), eigA = new Float64Array(9), eigV = new Float64Array(9);
    const Rm = new Float64Array(9), quat = new Float64Array(4);

    for (let i = 0; i < N; i++) {
        if (parent[i] !== i) continue;
        const i6 = i * 6, iw = 1 / W[i];
        // Σm = M2/W + EPS·I (same as mergeGroup's Σp(δδᵀ+Σ)+EPS).
        Sig[0] = M2[i6] * iw + EPS_COV;
        Sig[1] = Sig[3] = M2[i6 + 1] * iw;
        Sig[2] = Sig[6] = M2[i6 + 2] * iw;
        Sig[4] = M2[i6 + 3] * iw + EPS_COV;
        Sig[5] = Sig[7] = M2[i6 + 4] * iw;
        Sig[8] = M2[i6 + 5] * iw + EPS_COV;
        eigenSymmetric3x3(Sig, eigA, eigV);

        // Order eigenpairs descending (mergeGroup's o0/o1/o2 logic).
        const v0 = eigA[0], v1 = eigA[4], v2 = eigA[8];
        let o0, o1, o2;
        if (v0 >= v1) {
            if (v1 >= v2) { o0 = 0; o1 = 1; o2 = 2; } else if (v0 >= v2) { o0 = 0; o1 = 2; o2 = 1; } else { o0 = 2; o1 = 0; o2 = 1; }
        } else if (v0 >= v2) { o0 = 1; o1 = 0; o2 = 2; } else if (v1 >= v2) { o0 = 1; o1 = 2; o2 = 0; } else { o0 = 2; o1 = 1; o2 = 0; }
        const ev0 = Math.max(eigA[3 * o0 + o0], 1e-18);
        const ev1 = Math.max(eigA[3 * o1 + o1], 1e-18);
        const ev2 = Math.max(eigA[3 * o2 + o2], 1e-18);
        const s0 = Math.sqrt(ev0), s1 = Math.sqrt(ev1), s2 = Math.sqrt(ev2);
        const alphaM = Math.min(1, W[i] / Math.max(ellipsoidArea(s0, s1, s2), 1e-30));

        Rm[0] = eigV[o0]; Rm[1] = eigV[o1]; Rm[2] = eigV[o2];
        Rm[3] = eigV[3 + o0]; Rm[4] = eigV[3 + o1]; Rm[5] = eigV[3 + o2];
        Rm[6] = eigV[6 + o0]; Rm[7] = eigV[6 + o1]; Rm[8] = eigV[6 + o2];
        if (det3(Rm, 0) < 0) { Rm[2] *= -1; Rm[5] *= -1; Rm[8] *= -1; }
        rotmatToQuat(Rm, 0, quat, 0);

        const o = rows * (stride >> 2);
        f32[o] = mx[i]; f32[o + 1] = my[i]; f32[o + 2] = mz[i];
        const cbase = i * colorDim;
        for (let c = 0; c < colorDim; c++) f32[o + 3 + c] = colorW[cbase + c] * iw;
        const oo = o + 3 + colorDim;
        f32[oo] = logit(Math.max(0, Math.min(1, alphaM)));
        f32[oo + 1] = Math.log(s0); f32[oo + 2] = Math.log(s1); f32[oo + 3] = Math.log(s2);
        f32[oo + 4] = quat[0]; f32[oo + 5] = quat[1]; f32[oo + 6] = quat[2]; f32[oo + 7] = quat[3];

        if (++rows === ROWS) { writeSync(fd, buf, 0, rows * stride); written += rows; rows = 0; }
    }
    if (rows > 0) { writeSync(fd, buf, 0, rows * stride); written += rows; }
    closeSync(fd);
    if (written !== liveCount) throw new Error(`emitted ${written} rows, expected ${liveCount}`);
    log(`wrote ${path} (${written} gaussians)`);
};

// ---------------------------------------------------------------------------
// PLY load → engine init (single sequential pass; originals never stored raw).
// ---------------------------------------------------------------------------

const loadPly = async (filename) => {
    const pool = createChunkDataPool();
    const fs = new NodeReadFileSystem();
    const src = await readPly(await fs.createSource(filename), pool);
    const { meta } = src;
    if (meta.numLods !== 1) throw new Error('single-LOD input required');
    const dim = meta.layouts.color.stride >> 2;
    log(`loading ${filename}: ${meta.numGaussians} gaussians · colorDim ${dim}`);
    initEngine(meta.numGaussians, dim);
    stamp = new Uint32Array(N);

    let base = 0;
    const sigSample = [];
    for (let c = 0; c < meta.numChunks[0]; c++) {
        const count = Math.min(meta.chunkSize, meta.numGaussians - c * meta.chunkSize);
        const pcd = pool.acquire('position', meta.layouts.position, count);
        const gcd = pool.acquire('geometric', meta.layouts.geometric, count);
        const ccd = pool.acquire('color', meta.layouts.color, count);
        await src.read({ chunkIndex: c, position: pcd, geometric: gcd, color: ccd });
        const p = new Float32Array(pcd.data, 0, count * 3);
        const g = new Float32Array(gcd.data, 0, count * 8);
        const col = new Float32Array(ccd.data, 0, count * dim);
        for (let i = 0; i < count; i++) {
            initRow(base + i, p[i * 3], p[i * 3 + 1], p[i * 3 + 2], g, i * 8, col, i * dim);
            if ((base + i) % 991 === 0) {
                sigSample.push(Math.exp((g[i * 8 + 4] + g[i * 8 + 5] + g[i * 8 + 6]) / 3));
            }
        }
        base += count;
        pcd.release(); gcd.release(); ccd.release();
        if ((c + 1) % 200 === 0) log(`  loaded ${base}/${meta.numGaussians}`);
    }
    await src.close();
    sigSample.sort((x, y) => x - y);
    const q = (f) => sigSample[Math.min(sigSample.length - 1, (sigSample.length * f) | 0)];
    log(`load complete (${base} rows) · σ_gm p10 ${q(0.1).toExponential(2)} · median ${q(0.5).toExponential(2)} · p90 ${q(0.9).toExponential(2)}`);
};

// Exact global KNN via the production forest (single part at crop scale).
const buildKnn = async () => {
    const k = Math.min(KNN_K, Math.max(1, N - 1));
    log(`KNN: forest query (k=${k})`);

    const ids = new Uint32Array(N);
    const aabb = new Float32Array([Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity]);
    for (let i = 0; i < N; i++) {
        ids[i] = i;
        aabb[0] = Math.min(aabb[0], px[i]);
        aabb[1] = Math.min(aabb[1], py[i]);
        aabb[2] = Math.min(aabb[2], pz[i]);
        aabb[3] = Math.max(aabb[3], px[i]);
        aabb[4] = Math.max(aabb[4], py[i]);
        aabb[5] = Math.max(aabb[5], pz[i]);
    }
    const part = { ...buildFlatKdTree(px, py, pz), aabb };   // splat ids already global (identity)
    const queryPos = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
        queryPos[i * 3] = px[i];
        queryPos[i * 3 + 1] = py[i];
        queryPos[i * 3 + 2] = pz[i];
    }
    const out = new Uint32Array(N * k);
    knnForestQuery([part], queryPos, ids, N, k, out);
    for (let g = 0; g < N; g++) {
        for (let s = 0; s < k; s++) knn[g * KNN_K + s] = out[g * k + s];
    }
    log('  KNN done');
};

// ---------------------------------------------------------------------------
// Selftest: engine math vs the library (pair costs, moment composition,
// emission parameters) on synthetic data.
// ---------------------------------------------------------------------------

const selftest = () => {
    const n = 400, dim = 3;
    const rand = (() => { let t = 12345; return () => { t = (t * 1103515245 + 12345) & 0x7fffffff; return t / 0x7fffffff; }; })();

    const view = {
        pos: new Float32Array(n * 3),
        geo: new Float32Array(n * 8),
        color: new Float32Array(n * dim),
        colorDim: dim
    };
    for (let i = 0; i < n; i++) {
        view.pos[i * 3] = rand() * 4; view.pos[i * 3 + 1] = rand() * 4; view.pos[i * 3 + 2] = rand() * 4;
        const qw = rand() - 0.5, qx = rand() - 0.5, qy = rand() - 0.5, qz = rand() - 0.5;
        view.geo[i * 8] = qw; view.geo[i * 8 + 1] = qx; view.geo[i * 8 + 2] = qy; view.geo[i * 8 + 3] = qz;
        view.geo[i * 8 + 4] = -2.5 + rand() * 2;
        view.geo[i * 8 + 5] = -2.5 + rand() * 2;
        view.geo[i * 8 + 6] = -2.5 + rand() * 2;
        view.geo[i * 8 + 7] = -1 + rand() * 3;
        view.color[i * dim] = rand() * 2 - 1;
        view.color[i * dim + 1] = rand() * 2 - 1;
        view.color[i * dim + 2] = rand() * 2 - 1;
    }

    initEngine(n, dim);
    stamp = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
        initRow(i, view.pos[i * 3], view.pos[i * 3 + 1], view.pos[i * 3 + 2], view.geo, i * 8, view.color, i * dim);
    }
    // Brute-force KNN.
    for (let i = 0; i < n; i++) {
        const d2s = [];
        for (let j = 0; j < n; j++) {
            if (j === i) continue;
            const dx = px[i] - px[j], dy = py[i] - py[j], dz = pz[i] - pz[j];
            d2s.push([dx * dx + dy * dy + dz * dz, j]);
        }
        d2s.sort((a, b) => a[0] - b[0]);
        for (let s = 0; s < KNN_K; s++) knn[i * KNN_K + s] = d2s[s][1];
    }

    // 1. Singleton pair cost parity vs the library edge cost.
    const cache = new Float32Array(n * CACHE_STRIDE);
    buildSplatCache(view, cache);
    let worst = 0;
    for (let t = 0; t < 200; t++) {
        const i = (rand() * n) | 0;
        const j = knn[i * KNN_K + ((rand() * KNN_K) | 0)];
        const mine = evalMerge(i, j);
        const lib = computeEdgeCost(cache, i, j);
        const rel = Math.abs(mine - lib) / Math.max(1e-12, Math.abs(lib));
        if (rel > worst) worst = rel;
    }
    console.log(`selftest 1 — pair-cost parity vs lib: worst rel diff ${worst.toExponential(2)} ${worst < 1e-3 ? 'PASS' : 'FAIL'}`);
    if (worst >= 1e-3) process.exit(1);

    // 2. Greedy run to 50%, then compare cluster moments vs n-ary mergeGroup.
    heapInit(2 * n);
    runGreedy([n >> 1], () => {}, 1e9);
    const scratch = createMergeScratch();
    const out = { pos: new Float64Array(3), geo: new Float64Array(8), color: new Float64Array(dim) };
    let checked = 0, worstPos = 0, worstScale = 0, worstAlpha = 0, worstColor = 0, worstQuat = 1;
    for (let i = 0; i < n && checked < 40; i++) {
        if (parent[i] !== i || ufsize[i] < 2) continue;
        const cnt = gatherMembers(i);
        const members = Array.from(gatherBuf.subarray(0, cnt));
        mergeGroup(view, members, cnt, out, scratch);

        worstPos = Math.max(worstPos, Math.abs(out.pos[0] - mx[i]), Math.abs(out.pos[1] - my[i]), Math.abs(out.pos[2] - mz[i]));

        // Rebuild my emission params for this cluster.
        const i6 = i * 6, iw = 1 / W[i];
        const Sig = new Float64Array([
            M2[i6] * iw + EPS_COV, M2[i6 + 1] * iw, M2[i6 + 2] * iw,
            M2[i6 + 1] * iw, M2[i6 + 3] * iw + EPS_COV, M2[i6 + 4] * iw,
            M2[i6 + 2] * iw, M2[i6 + 4] * iw, M2[i6 + 5] * iw + EPS_COV
        ]);
        const eigA = new Float64Array(9), eigV = new Float64Array(9);
        eigenSymmetric3x3(Sig, eigA, eigV);
        const evs = [eigA[0], eigA[4], eigA[8]].sort((x, y) => y - x);
        const myScales = evs.map(v => Math.log(Math.sqrt(Math.max(v, 1e-18))));
        const libScales = [out.geo[4], out.geo[5], out.geo[6]];
        for (let s = 0; s < 3; s++) worstScale = Math.max(worstScale, Math.abs(myScales[s] - libScales[s]));

        const myAlpha = Math.min(1, W[i] / Math.max(ellipsoidArea(...evs.map(v => Math.sqrt(Math.max(v, 1e-18)))), 1e-30));
        worstAlpha = Math.max(worstAlpha, Math.abs(myAlpha - sigmoid(out.geo[7])));

        for (let c = 0; c < dim; c++) {
            worstColor = Math.max(worstColor, Math.abs(colorW[i * dim + c] * iw - out.color[c]));
        }
        checked++;
    }
    console.log(`selftest 2 — moment parity vs mergeGroup over ${checked} clusters: pos ${worstPos.toExponential(2)} scale ${worstScale.toExponential(2)} alpha ${worstAlpha.toExponential(2)} color ${worstColor.toExponential(2)}`);
    const ok2 = worstPos < 1e-5 && worstScale < 1e-4 && worstAlpha < 1e-4 && worstColor < 1e-4;
    console.log(ok2 ? 'PASS' : 'FAIL');
    if (!ok2) process.exit(1);

    // 3. Err invariants: every cluster error ≥ 0 (up to float noise), and
    //    liveCount bookkeeping is exact.
    let minErr = Infinity, roots = 0;
    for (let i = 0; i < n; i++) {
        if (parent[i] !== i) continue;
        roots++;
        if (Err[i] < minErr) minErr = Err[i];
    }
    console.log(`selftest 3 — ${roots} roots (expect ${n >> 1}) · min Err ${minErr.toExponential(2)} ${roots === n >> 1 && minErr > -1e-9 ? 'PASS' : 'FAIL'}`);
    if (roots !== n >> 1 || minErr <= -1e-9) process.exit(1);

    console.log('selftest OK');
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

const main = async () => {
    const argv = process.argv.slice(2);
    const argValue = (name, dflt) => {
        const i = argv.indexOf(name);
        return i >= 0 && i + 1 < argv.length ? argv[i + 1] : dflt;
    };

    if (argv.includes('--selftest')) {
        selftest();
        process.exit(0);
    }

    const input = argValue('--input');
    const outPrefix = argValue('--out-prefix');
    const halvings = parseInt(argValue('--halvings', '1'), 10);
    const mg = argValue('--max-group');
    if (mg) maxGroup = parseInt(mg, 10);
    const se = argValue('--size-exponent');
    if (se) sizeExponent = parseFloat(se);
    const dl = argValue('--dilate');
    if (dl) dilate = parseFloat(dl);
    const cm = argValue('--cost');
    if (cm) {
        if (cm !== 'l2' && cm !== 'kl') throw new Error(`--cost must be l2|kl (got ${cm})`);
        costMode = cm;
    }
    const cw = argValue('--color-weight');
    if (cw) colorWeight = parseFloat(cw);
    if (argv.includes('--needle-guard')) needleGuard = true;
    if (argv.includes('--color-dc-only')) colorDcOnly = true;
    if (!input || !outPrefix) {
        console.error('usage: decimate-exact.mjs --input <in.ply> --out-prefix <prefix> --halvings <n> [--max-group <n>] | --selftest');
        process.exit(1);
    }
    log(`max-group: ${maxGroup} · size-exponent: ${sizeExponent} · dilate: ${dilate} · cost: ${costMode} · color-weight: ${colorWeight} · needle-guard: ${needleGuard}`);

    await loadPly(input);
    await buildKnn();

    // Targets: successive ceil(count/2) — identical to the production cascade.
    const targets = [];
    let c = N;
    for (let h = 0; h < halvings; h++) {
        c = c - Math.floor(c / 2);
        targets.push(c);
    }
    heapInit(Math.ceil(N * 1.25));
    runGreedy(targets, (ti) => emitPly(`${outPrefix}${ti + 1}.ply`));

    log('done');
    process.exit(0);
};

main().catch((e) => {
    console.error(e);
    process.exit(1);
});
