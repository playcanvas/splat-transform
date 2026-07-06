/**
 * Merge/cost math for chunk-native decimation, generalized to n-ary groups.
 *
 * Engine-free: this module is imported by worker tasks, so it must not import
 * DataTable or playcanvas (see the note at the top of workers/tasks.ts).
 *
 * The n = 2 path of {@link mergeGroup} is arithmetic-identical to the legacy
 * `momentMatch` in the pre-3.0 `data-table/decimate.ts` (area·α weighted,
 * law-of-total-variance covariance, mass-conserving opacity capped at 1) —
 * enforced by test/moment-match.test.mjs against a verbatim reference copy.
 */

const LOG2PI = Math.log(2 * Math.PI);

/** Covariance diagonal regularizer, matching legacy EPS_COV. */
const EPS_COV = 1e-8;

// ---------- sigmoid / logit ----------

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

const logit = (p: number) => {
    p = Math.max(1e-7, Math.min(1 - 1e-7, p));
    return Math.log(p / (1 - p));
};

const logAddExp = (a: number, b: number) => {
    if (a === -Infinity) return b;
    if (b === -Infinity) return a;
    const m = Math.max(a, b);
    return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
};

// ---------- PRNG (MC samples for the edge cost; seed 0 matches legacy) ----------

const mulberry32 = (seed: number) => {
    return () => {
        let t = (seed += 0x6d2b79f5);
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
};

const makeGaussianSamples = (n: number, seed: number): Float64Array[] => {
    const rand = mulberry32(seed >>> 0);
    const out: Float64Array[] = [];
    while (out.length < n) {
        const u1 = Math.max(rand(), 1e-12);
        const u2 = rand();
        const u3 = Math.max(rand(), 1e-12);
        const u4 = rand();
        const r1 = Math.sqrt(-2 * Math.log(u1));
        const t1 = 2 * Math.PI * u2;
        const r2 = Math.sqrt(-2 * Math.log(u3));
        const t2 = 2 * Math.PI * u4;
        out.push(new Float64Array([r1 * Math.cos(t1), r1 * Math.sin(t1), r2 * Math.cos(t2)]));
    }
    return out;
};

// ---------- 3x3 matrix helpers (row-major, 9 floats) ----------

type Mat9 = Float32Array | Float64Array;

const quatToRotmat = (qw: number, qx: number, qy: number, qz: number, out: Mat9, o: number) => {
    const xx = qx * qx, yy = qy * qy, zz = qz * qz;
    const wx = qw * qx, wy = qw * qy, wz = qw * qz;
    const xy = qx * qy, xz = qx * qz, yz = qy * qz;
    out[o] = 1 - 2 * (yy + zz);
    out[o + 1] = 2 * (xy - wz);
    out[o + 2] = 2 * (xz + wy);
    out[o + 3] = 2 * (xy + wz);
    out[o + 4] = 1 - 2 * (xx + zz);
    out[o + 5] = 2 * (yz - wx);
    out[o + 6] = 2 * (xz - wy);
    out[o + 7] = 2 * (yz + wx);
    out[o + 8] = 1 - 2 * (xx + yy);
};

const sigmaFromRotVar = (R: Mat9, r: number, vx: number, vy: number, vz: number, out: Mat9, o: number) => {
    const r00 = R[r], r01 = R[r + 1], r02 = R[r + 2];
    const r10 = R[r + 3], r11 = R[r + 4], r12 = R[r + 5];
    const r20 = R[r + 6], r21 = R[r + 7], r22 = R[r + 8];
    out[o] = r00 * r00 * vx + r01 * r01 * vy + r02 * r02 * vz;
    out[o + 1] = r00 * r10 * vx + r01 * r11 * vy + r02 * r12 * vz;
    out[o + 2] = r00 * r20 * vx + r01 * r21 * vy + r02 * r22 * vz;
    out[o + 3] = out[o + 1];
    out[o + 4] = r10 * r10 * vx + r11 * r11 * vy + r12 * r12 * vz;
    out[o + 5] = r10 * r20 * vx + r11 * r21 * vy + r12 * r22 * vz;
    out[o + 6] = out[o + 2];
    out[o + 7] = out[o + 5];
    out[o + 8] = r20 * r20 * vx + r21 * r21 * vy + r22 * r22 * vz;
};

const det3 = (A: Mat9, o: number) => {
    return (
        A[o] * (A[o + 4] * A[o + 8] - A[o + 5] * A[o + 7]) -
        A[o + 1] * (A[o + 3] * A[o + 8] - A[o + 5] * A[o + 6]) +
        A[o + 2] * (A[o + 3] * A[o + 7] - A[o + 4] * A[o + 6])
    );
};

const gaussLogpdfDiagrot = (
    x: number, y: number, z: number,
    mx: number, my: number, mz: number,
    R: Mat9, ro: number,
    invx: number, invy: number, invz: number, logdet: number
) => {
    const dx = x - mx, dy = y - my, dz = z - mz;
    const y0 = dx * R[ro] + dy * R[ro + 3] + dz * R[ro + 6];
    const y1 = dx * R[ro + 1] + dy * R[ro + 4] + dz * R[ro + 7];
    const y2 = dx * R[ro + 2] + dy * R[ro + 5] + dz * R[ro + 8];
    const quad = y0 * y0 * invx + y1 * y1 * invy + y2 * y2 * invz;
    return -0.5 * (3 * LOG2PI + logdet + quad);
};

// Jacobi eigendecomposition for 3x3 symmetric matrix; caller-provided scratch,
// eigenvalues land on A's diagonal, eigenvectors in V's columns.
const eigenSymmetric3x3 = (Ain: Float64Array, A: Float64Array, V: Float64Array) => {
    A.set(Ain);
    V[0] = 1; V[1] = 0; V[2] = 0;
    V[3] = 0; V[4] = 1; V[5] = 0;
    V[6] = 0; V[7] = 0; V[8] = 1;

    for (let iter = 0; iter < 24; iter++) {
        let p = 0, q = 1;
        let maxAbs = Math.abs(A[1]);
        if (Math.abs(A[2]) > maxAbs) {
            p = 0; q = 2; maxAbs = Math.abs(A[2]);
        }
        if (Math.abs(A[5]) > maxAbs) {
            p = 1; q = 2; maxAbs = Math.abs(A[5]);
        }
        if (maxAbs < 1e-12) break;

        const pp = 3 * p + p, qq = 3 * q + q, pq = 3 * p + q;
        const app = A[pp], aqq = A[qq], apq = A[pq];
        const tau = (aqq - app) / (2 * apq);
        const t = Math.sign(tau) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = t * c;

        for (let k = 0; k < 3; k++) {
            if (k === p || k === q) continue;
            const kp = 3 * k + p, kq = 3 * k + q;
            const pk = 3 * p + k, qk = 3 * q + k;
            const akp = A[kp], akq = A[kq];
            A[kp] = c * akp - s * akq;
            A[pk] = A[kp];
            A[kq] = s * akp + c * akq;
            A[qk] = A[kq];
        }
        A[pp] = c * c * app - 2 * s * c * apq + s * s * aqq;
        A[qq] = s * s * app + 2 * s * c * apq + c * c * aqq;
        A[pq] = 0; A[3 * q + p] = 0;

        for (let k = 0; k < 3; k++) {
            const kp = 3 * k + p, kq = 3 * k + q;
            const vkp = V[kp], vkq = V[kq];
            V[kp] = c * vkp - s * vkq;
            V[kq] = s * vkp + c * vkq;
        }
    }
};

const rotmatToQuat = (R: Float64Array, o: number, out: Float64Array, oo: number) => {
    const m00 = R[o], m11 = R[o + 4], m22 = R[o + 8];
    const tr = m00 + m11 + m22;
    let qw: number, qx: number, qy: number, qz: number;

    if (tr > 0) {
        const S = Math.sqrt(tr + 1) * 2;
        qw = 0.25 * S;
        qx = (R[o + 7] - R[o + 5]) / S;
        qy = (R[o + 2] - R[o + 6]) / S;
        qz = (R[o + 3] - R[o + 1]) / S;
    } else if (R[o] > R[o + 4] && R[o] > R[o + 8]) {
        const S = Math.sqrt(1 + R[o] - R[o + 4] - R[o + 8]) * 2;
        qw = (R[o + 7] - R[o + 5]) / S;
        qx = 0.25 * S;
        qy = (R[o + 1] + R[o + 3]) / S;
        qz = (R[o + 2] + R[o + 6]) / S;
    } else if (R[o + 4] > R[o + 8]) {
        const S = Math.sqrt(1 + R[o + 4] - R[o] - R[o + 8]) * 2;
        qw = (R[o + 2] - R[o + 6]) / S;
        qx = (R[o + 1] + R[o + 3]) / S;
        qy = 0.25 * S;
        qz = (R[o + 5] + R[o + 7]) / S;
    } else {
        const S = Math.sqrt(1 + R[o + 8] - R[o] - R[o + 4]) * 2;
        qw = (R[o + 3] - R[o + 1]) / S;
        qx = (R[o + 2] + R[o + 6]) / S;
        qy = (R[o + 5] + R[o + 7]) / S;
        qz = 0.25 * S;
    }

    const n = Math.hypot(qw, qx, qy, qz);
    const inv = 1 / Math.max(n, 1e-12);
    out[oo] = qw * inv;
    out[oo + 1] = qx * inv;
    out[oo + 2] = qy * inv;
    out[oo + 3] = qz * inv;
};

// ---------- ellipsoid area (Knud Thomsen p=1.6075) ----------

const ELLIPSOID_P = 1.6075;
const ellipsoidArea = (sx: number, sy: number, sz: number): number => {
    const a = Math.pow(sx * sy, ELLIPSOID_P);
    const b = Math.pow(sx * sz, ELLIPSOID_P);
    const c = Math.pow(sy * sz, ELLIPSOID_P);
    return 4 * Math.PI * Math.pow((a + b + c) / 3, 1 / ELLIPSOID_P);
};

// ---------- splat views ----------

/**
 * Column-tight view over a batch of splats, the working representation for
 * block processing. `pos` is 3 f32/splat (x, y, z); `geo` is 8 f32/splat in
 * geometric-layer order (rot_0..3, scale_0..2 log-space, opacity logit);
 * `color` is `colorDim` f32/splat (dc0..2 then f_rest coefficients).
 */
type SplatView = {
    pos: Float32Array;
    geo: Float32Array;
    color: Float32Array;
    colorDim: number;
};

/** Merged-splat output: pos 3, geo 8 (same encoding as SplatView), color colorDim. */
type MergedOut = {
    pos: Float64Array;
    geo: Float64Array;
    color: Float64Array;
};

/**
 * Reusable scratch shared by {@link mergeGroup} and the CPU edge cost.
 * Allocated once per pass, not per call — the merge loop runs hundreds of
 * millions of times at scale.
 */
type MergeScratch = {
    sigm: Float64Array;
    sigI: Float64Array;
    sigJ: Float64Array;
    rI: Float64Array;
    rJ: Float64Array;
    sig: Float64Array;
    rM: Float64Array;
    eigA: Float64Array;
    eigV: Float64Array;
};

const createMergeScratch = (): MergeScratch => ({
    sigm: new Float64Array(9),
    sigI: new Float64Array(9),
    sigJ: new Float64Array(9),
    rI: new Float64Array(9),
    rJ: new Float64Array(9),
    sig: new Float64Array(9),
    rM: new Float64Array(9),
    eigA: new Float64Array(9),
    eigV: new Float64Array(9)
});

/**
 * Merge weight of one splat: area·α "ink" mass (+1e-30, the merge-path
 * epsilon; the cost-path cache uses +1e-12 — both match their legacy
 * counterparts exactly).
 * @param geo - Geometric-layer view (8 f32/splat).
 * @param i - Splat index.
 * @returns The merge weight.
 */
const splatMass = (geo: Float32Array, i: number): number => {
    const i8 = i * 8;
    const sx = Math.max(Math.exp(geo[i8 + 4]), 1e-12);
    const sy = Math.max(Math.exp(geo[i8 + 5]), 1e-12);
    const sz = Math.max(Math.exp(geo[i8 + 6]), 1e-12);
    return sigmoid(geo[i8 + 7]) * ellipsoidArea(sx, sy, sz) + 1e-30;
};

/**
 * n-ary moment match: merge `count` splats into one Gaussian. Weights are
 * area·α; merged covariance is the weighted sum of (δδᵀ + Σₖ) (law of total
 * variance); opacity is mass-conserving capped at 1; color/SH is the weighted
 * average. For n = 2 this is arithmetic-identical to the legacy pairwise
 * `momentMatch`.
 *
 * @param view - Splat columns.
 * @param members - Indices of the splats to merge.
 * @param count - Number of members.
 * @param out - Output splat (geo encoded ready-to-store: quat, log scales, logit opacity).
 * @param scratch - Reusable scratch from {@link createMergeScratch}.
 */
const mergeGroup = (
    view: SplatView,
    members: ArrayLike<number>,
    count: number,
    out: MergedOut,
    scratch: MergeScratch
): void => {
    const { pos, geo, color, colorDim } = view;

    let W = 0;
    for (let m = 0; m < count; m++) W += splatMass(geo, members[m]);

    // Merged mean (weighted).
    let mux = 0, muy = 0, muz = 0;
    for (let m = 0; m < count; m++) {
        const i = members[m];
        const p = splatMass(geo, i) / W;
        mux += p * pos[i * 3];
        muy += p * pos[i * 3 + 1];
        muz += p * pos[i * 3 + 2];
    }

    // Merged covariance: Σ pₖ (δₖδₖᵀ + Σₖ), accumulated member-by-member.
    const Sig = scratch.sig;
    Sig.fill(0);
    const SigI = scratch.sigI;
    const Ri = scratch.rI;
    for (let m = 0; m < count; m++) {
        const i = members[m];
        const i8 = i * 8;
        const p = splatMass(geo, i) / W;

        let qw = geo[i8], qx = geo[i8 + 1], qy = geo[i8 + 2], qz = geo[i8 + 3];
        const qn = 1 / Math.max(Math.hypot(qw, qx, qy, qz), 1e-12);
        qw *= qn; qx *= qn; qy *= qn; qz *= qn;
        const sx = Math.max(Math.exp(geo[i8 + 4]), 1e-12);
        const sy = Math.max(Math.exp(geo[i8 + 5]), 1e-12);
        const sz = Math.max(Math.exp(geo[i8 + 6]), 1e-12);

        quatToRotmat(qw, qx, qy, qz, Ri, 0);
        sigmaFromRotVar(Ri, 0, sx * sx, sy * sy, sz * sz, SigI, 0);

        const dx = pos[i * 3] - mux, dy = pos[i * 3 + 1] - muy, dz = pos[i * 3 + 2] - muz;
        Sig[0] += p * (dx * dx + SigI[0]);
        Sig[1] += p * (dx * dy + SigI[1]);
        Sig[2] += p * (dx * dz + SigI[2]);
        Sig[4] += p * (dy * dy + SigI[4]);
        Sig[5] += p * (dy * dz + SigI[5]);
        Sig[8] += p * (dz * dz + SigI[8]);
    }
    Sig[3] = Sig[1];
    Sig[6] = Sig[2];
    Sig[7] = Sig[5];
    Sig[0] += EPS_COV;
    Sig[4] += EPS_COV;
    Sig[8] += EPS_COV;

    // Eigendecompose → scales (√λ, descending) + right-handed rotation → quat.
    const eigA = scratch.eigA;
    const eigV = scratch.eigV;
    eigenSymmetric3x3(Sig, eigA, eigV);
    const vecs = eigV;

    const v0 = eigA[0], v1 = eigA[4], v2 = eigA[8];
    let o0: number, o1: number, o2: number;
    if (v0 >= v1) {
        if (v1 >= v2)      {
            o0 = 0; o1 = 1; o2 = 2;
        } else if (v0 >= v2) {
            o0 = 0; o1 = 2; o2 = 1;
        } else               {
            o0 = 2; o1 = 0; o2 = 1;
        }
    } else {
        if (v0 >= v2)      {
            o0 = 1; o1 = 0; o2 = 2;
        } else if (v1 >= v2) {
            o0 = 1; o1 = 2; o2 = 0;
        } else               {
            o0 = 2; o1 = 1; o2 = 0;
        }
    }
    const ev0 = Math.max(eigA[3 * o0 + o0], 1e-18);
    const ev1 = Math.max(eigA[3 * o1 + o1], 1e-18);
    const ev2 = Math.max(eigA[3 * o2 + o2], 1e-18);

    const s0 = Math.sqrt(ev0);
    const s1 = Math.sqrt(ev1);
    const s2 = Math.sqrt(ev2);

    // Mass-conserving opacity, capped at 1 (no scale inflation).
    const alphaM = Math.min(1, W / Math.max(ellipsoidArea(s0, s1, s2), 1e-30));

    const Rm = scratch.rM;
    Rm[0] = vecs[o0]; Rm[1] = vecs[o1]; Rm[2] = vecs[o2];
    Rm[3] = vecs[3 + o0]; Rm[4] = vecs[3 + o1]; Rm[5] = vecs[3 + o2];
    Rm[6] = vecs[6 + o0]; Rm[7] = vecs[6 + o1]; Rm[8] = vecs[6 + o2];
    if (det3(Rm, 0) < 0) {
        Rm[2] *= -1; Rm[5] *= -1; Rm[8] *= -1;
    }
    rotmatToQuat(Rm, 0, out.geo, 0);

    out.pos[0] = mux; out.pos[1] = muy; out.pos[2] = muz;
    out.geo[4] = Math.log(s0);
    out.geo[5] = Math.log(s1);
    out.geo[6] = Math.log(s2);
    out.geo[7] = logit(Math.max(0, Math.min(1, alphaM)));

    // Color: weight-normalized (area·α weighted) average.
    for (let c = 0; c < colorDim; c++) {
        let acc = 0;
        for (let m = 0; m < count; m++) {
            const i = members[m];
            acc += (splatMass(geo, i) / W) * color[i * colorDim + c];
        }
        out.color[c] = acc;
    }
};

export {
    EPS_COV,
    LOG2PI,
    sigmoid,
    logit,
    logAddExp,
    makeGaussianSamples,
    quatToRotmat,
    sigmaFromRotVar,
    det3,
    gaussLogpdfDiagrot,
    eigenSymmetric3x3,
    rotmatToQuat,
    ellipsoidArea,
    splatMass,
    mergeGroup,
    createMergeScratch,
    type SplatView,
    type MergedOut,
    type MergeScratch
};
