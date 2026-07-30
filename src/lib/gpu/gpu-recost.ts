import {
    BUFFERUSAGE_COPY_DST,
    BUFFERUSAGE_COPY_SRC,
    GraphicsDevice,
    StorageBuffer
} from 'playcanvas';

import { makeKernel, type Kernel } from './compute-kernel';
import { gaussianL2Wgsl } from './shaders/chunks/gaussian-l2';

/**
 * GPU engine for the re-costed selection's wave rounds.
 *
 * The CPU keeps the heap, the commit decisions, and integer mirrors of the
 * union-find/chains; the GPU holds its own copy of that structure plus the
 * immutable splat cache and neighbour graph, and does the bulk work: after
 * each drained wave the CPU uploads the wave's commit log, a `replay` kernel
 * applies it (three scattered writes per entry — commits within a wave touch
 * disjoint roots, so no atomics), and a `refresh` kernel re-evaluates each
 * queued root's best edge with one 64-lane workgroup per root (lane = one
 * (member, neighbour-slot) candidate; duplicate candidates cost nothing in
 * lockstep). Results come back as 8 B per queued root: (partner, cost f32).
 *
 * The evaluation mirrors decimate/recost-core.ts's stateless cancelled form
 * in f32 — keep them in lockstep. Costs order the heap only; nothing
 * accumulates, so f32 error stays per-eval (near-tie ordering class).
 *
 * The splat cache and neighbour rows are split across two buffers by row
 * range (a single binding would exceed the ~2 GiB per-binding ceiling near
 * 34M splats); `fits` pre-flights all binding sizes by arithmetic because a
 * device-side OOM escalates to a hard failure (node-device policy), not a
 * catchable fallback.
 */

const WG = 64;
const NONE = 0xFFFFFFFF;
const MAX_DIM = 65535;

// Shared structure access for the replay/refresh kernels: parentMeta[i] =
// (parent, size-at-root); chain[i] = (head-at-root, next).
const refreshWgsl = (
    k: number,
    maxGroup: number,
    splitN: number,
    coreCount: number,
    partitioned: boolean
) => /* wgsl */`
struct Uniforms {
    pendingCount: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
// Per-splat cache (CACHE_STRIDE 16 f32/row), split by row range at SPLIT.
@group(0) @binding(1) var<storage, read> cacheA: array<f32>;
@group(0) @binding(2) var<storage, read> cacheB: array<f32>;
// Neighbour ids (K u32/row, sentinel padded), same row split.
@group(0) @binding(3) var<storage, read> nbA: array<u32>;
@group(0) @binding(4) var<storage, read> nbB: array<u32>;
@group(0) @binding(5) var<storage, read> parentMeta: array<vec2u>;
@group(0) @binding(6) var<storage, read> chain: array<vec2u>;
@group(0) @binding(7) var<storage, read> pending: array<u32>;
// Per queued root: the global path writes (partner, cost); block-local mode
// additionally writes its best immutable-halo (partner, cost).
@group(0) @binding(8) var<storage, read_write> outBest: array<${partitioned ? 'vec4u' : 'vec2u'}>;

const K: u32 = ${k}u;
const MAXG: u32 = ${maxGroup}u;
const SPLIT: u32 = ${splitN}u;
const CORE_COUNT: u32 = ${coreCount}u;
const NONE: u32 = 0xFFFFFFFFu;
const NIL: u32 = 0xFFFFFFFFu;
const F32_MAX: f32 = 3.4028234663852886e+38;
const CULL_QUAD: f32 = 120.0;
${gaussianL2Wgsl}

fn cacheAt(row: u32, c: u32) -> f32 {
    if (row < SPLIT) { return cacheA[row * 16u + c]; }
    return cacheB[(row - SPLIT) * 16u + c];
}

fn nbAt(row: u32, s: u32) -> u32 {
    if (row < SPLIT) { return nbA[row * K + s]; }
    return nbB[(row - SPLIT) * K + s];
}

fn findRO(x0: u32) -> u32 {
    var x = x0;
    loop {
        let p = parentMeta[x].x;
        if (p == x) { return x; }
        x = p;
    }
}

// ⟨G_a,G_b⟩ scaled by √|Σa|·√|Σb| for M = Σa+Σb and offset d, with the
// recost kernel's exponent cull (mirrors recost-core crossG).
fn crossG(sdAB: f32, m: array<f32, 6>, d: vec3f) -> f32 {
    let c00 = m[3] * m[5] - m[4] * m[4];
    let c01 = m[2] * m[4] - m[1] * m[5];
    let c02 = m[1] * m[4] - m[2] * m[3];
    let c11 = m[0] * m[5] - m[2] * m[2];
    let c12 = m[1] * m[2] - m[0] * m[4];
    let c22 = m[0] * m[3] - m[1] * m[1];
    let det = max(m[0] * c00 + m[1] * c01 + m[2] * c02, 1e-30);
    let quad = (c00 * d.x * d.x + c11 * d.y * d.y + c22 * d.z * d.z +
        2.0 * (c01 * d.x * d.y + c02 * d.x * d.z + c12 * d.y * d.z)) / det;
    if (!(quad < CULL_QUAD)) { return 0.0; }
    return TWO_PI_1_5 * sdAB / sqrt(det) * exp(-0.5 * quad);
}

// Raw mass-scaled aggregates of a member set (mirrors composeRaw).
struct Raw {
    w: f32,
    mean: vec3f,
    m2: array<f32, 6>,
    bw: vec3f,
}

fn composeRaw(members: array<u32, ${maxGroup}>, count: u32) -> Raw {
    var w = 0.0;
    var s = vec3f(0.0);
    var b = vec3f(0.0);
    for (var t = 0u; t < count; t++) {
        let o = members[t];
        let m = cacheAt(o, 11u);
        w += m;
        s += m * vec3f(cacheAt(o, 0u), cacheAt(o, 1u), cacheAt(o, 2u));
        b += m * vec3f(cacheAt(o, 12u), cacheAt(o, 13u), cacheAt(o, 14u));
    }
    var r: Raw;
    r.w = w;
    r.mean = s / w;
    r.bw = b;
    for (var c = 0u; c < 6u; c++) { r.m2[c] = 0.0; }
    for (var t = 0u; t < count; t++) {
        let o = members[t];
        let m = cacheAt(o, 11u);
        let d = vec3f(cacheAt(o, 0u), cacheAt(o, 1u), cacheAt(o, 2u)) - r.mean;
        r.m2[0] += m * (cacheAt(o, 3u) + d.x * d.x);
        r.m2[1] += m * (cacheAt(o, 4u) + d.x * d.y);
        r.m2[2] += m * (cacheAt(o, 5u) + d.x * d.z);
        r.m2[3] += m * (cacheAt(o, 6u) + d.y * d.y);
        r.m2[4] += m * (cacheAt(o, 7u) + d.y * d.z);
        r.m2[5] += m * (cacheAt(o, 8u) + d.z * d.z);
    }
    return r;
}

// A∪B raw aggregates via the parallel-axis identity (mirrors composeUnion).
fn composeUnion(a: Raw, b: Raw) -> Raw {
    var o: Raw;
    o.w = a.w + b.w;
    let iw = 1.0 / o.w;
    o.mean = (a.w * a.mean + b.w * b.mean) * iw;
    let da = a.mean - o.mean;
    let db = b.mean - o.mean;
    o.m2[0] = a.m2[0] + b.m2[0] + a.w * da.x * da.x + b.w * db.x * db.x;
    o.m2[1] = a.m2[1] + b.m2[1] + a.w * da.x * da.y + b.w * db.x * db.y;
    o.m2[2] = a.m2[2] + b.m2[2] + a.w * da.x * da.z + b.w * db.x * db.z;
    o.m2[3] = a.m2[3] + b.m2[3] + a.w * da.y * da.y + b.w * db.y * db.y;
    o.m2[4] = a.m2[4] + b.m2[4] + a.w * da.y * da.z + b.w * db.y * db.z;
    o.m2[5] = a.m2[5] + b.m2[5] + a.w * da.z * da.z + b.w * db.z * db.z;
    o.bw = a.bw + b.bw;
    return o;
}

// Finished merged-Gaussian quantities (mirrors finishComp).
struct Fin {
    sm: array<f32, 6>,
    mean: vec3f,
    sd: f32,
    alpha: f32,
    bc: vec3f,
    selfM: f32,
}

fn finish(r: Raw) -> Fin {
    var f: Fin;
    let iw = 1.0 / r.w;
    f.sm[0] = r.m2[0] * iw + EPS_COV;
    f.sm[1] = r.m2[1] * iw;
    f.sm[2] = r.m2[2] * iw;
    f.sm[3] = r.m2[3] * iw + EPS_COV;
    f.sm[4] = r.m2[4] * iw;
    f.sm[5] = r.m2[5] * iw + EPS_COV;
    f.mean = r.mean;
    let detm = max(
        f.sm[0] * (f.sm[3] * f.sm[5] - f.sm[4] * f.sm[4]) - f.sm[1] * (f.sm[1] * f.sm[5] - f.sm[4] * f.sm[2]) + f.sm[2] * (f.sm[1] * f.sm[4] - f.sm[3] * f.sm[2]),
        1e-30
    );
    f.sd = sqrt(detm);
    let e = eig3(f.sm);
    let s0 = sqrt(max(e.x, 1e-18));
    let s1 = sqrt(max(e.y, 1e-18));
    let s2 = sqrt(max(e.z, 1e-18));
    f.alpha = min(1.0, r.w / max(ellipsoidArea(s0, s1, s2), 1e-30));
    f.bc = r.bw * iw;
    f.selfM = f.alpha * f.alpha * dot(f.bc, f.bc) * PI_1_5 * f.sd;
    return f;
}

// memfm(C over members) = Σ ⟨f_k, f_C⟩ (mirrors memfm).
fn memfmOf(members: array<u32, ${maxGroup}>, count: u32, f: Fin) -> f32 {
    var acc = 0.0;
    for (var t = 0u; t < count; t++) {
        let o = members[t];
        let wgt = cacheAt(o, 10u) * f.alpha *
            dot(vec3f(cacheAt(o, 12u), cacheAt(o, 13u), cacheAt(o, 14u)), f.bc);
        if (wgt == 0.0) { continue; }
        let m = array<f32, 6>(
            cacheAt(o, 3u) + f.sm[0], cacheAt(o, 4u) + f.sm[1], cacheAt(o, 5u) + f.sm[2],
            cacheAt(o, 6u) + f.sm[3], cacheAt(o, 7u) + f.sm[4], cacheAt(o, 8u) + f.sm[5]
        );
        let d = vec3f(cacheAt(o, 0u), cacheAt(o, 1u), cacheAt(o, 2u)) - f.mean;
        acc += wgt * crossG(cacheAt(o, 9u) * f.sd, m, d);
    }
    return acc;
}

// A singleton's self product ⟨f, f⟩ (mirrors selfRow).
fn selfRow(o: u32) -> f32 {
    let a = cacheAt(o, 10u);
    return a * a * cacheAt(o, 15u) * PI_1_5 * cacheAt(o, 9u);
}

// One side's term of the cancelled cost form (mirrors sideTerm).
fn sideTerm(members: array<u32, ${maxGroup}>, count: u32, r: Raw) -> f32 {
    if (count < 2u) { return selfRow(members[0]); }
    let f = finish(r);
    return 2.0 * memfmOf(members, count, f) - f.selfM;
}

// scross(A,B) = Σ_{a∈A,b∈B} ⟨f_a, f_b⟩ with distance culling (mirrors scrossPairs).
fn scrossPairs(am: array<u32, ${maxGroup}>, na: u32, bm: array<u32, ${maxGroup}>, nb: u32) -> f32 {
    var acc = 0.0;
    for (var u = 0u; u < nb; u++) {
        let ob = bm[u];
        let bp = vec3f(cacheAt(ob, 0u), cacheAt(ob, 1u), cacheAt(ob, 2u));
        let trb = cacheAt(ob, 3u) + cacheAt(ob, 6u) + cacheAt(ob, 8u);
        let alb = cacheAt(ob, 10u);
        let sdb = cacheAt(ob, 9u);
        let cb = vec3f(cacheAt(ob, 12u), cacheAt(ob, 13u), cacheAt(ob, 14u));
        for (var t = 0u; t < na; t++) {
            let oa = am[t];
            let d = vec3f(cacheAt(oa, 0u), cacheAt(oa, 1u), cacheAt(oa, 2u)) - bp;
            let d2 = dot(d, d);
            if (d2 > CULL_QUAD * (cacheAt(oa, 3u) + cacheAt(oa, 6u) + cacheAt(oa, 8u) + trb)) { continue; }
            let wgt = cacheAt(oa, 10u) * alb *
                dot(vec3f(cacheAt(oa, 12u), cacheAt(oa, 13u), cacheAt(oa, 14u)), cb);
            if (wgt == 0.0) { continue; }
            let m = array<f32, 6>(
                cacheAt(oa, 3u) + cacheAt(ob, 3u), cacheAt(oa, 4u) + cacheAt(ob, 4u), cacheAt(oa, 5u) + cacheAt(ob, 5u),
                cacheAt(oa, 6u) + cacheAt(ob, 6u), cacheAt(oa, 7u) + cacheAt(ob, 7u), cacheAt(oa, 8u) + cacheAt(ob, 8u)
            );
            acc += wgt * crossG(cacheAt(oa, 9u) * sdb, m, d);
        }
    }
    return acc;
}

var<workgroup> wgAbort: u32;
var<workgroup> wgRoot: u32;
var<workgroup> wgSize: u32;
var<workgroup> wgCount: u32;
var<workgroup> wgMembers: array<u32, ${maxGroup}>;
var<workgroup> wgRawW: f32;
var<workgroup> wgRawMean: vec3f;
var<workgroup> wgRawM2: array<f32, 6>;
var<workgroup> wgRawBw: vec3f;
var<workgroup> wgATerm: f32;
var<workgroup> redCost: array<f32, ${WG}>;
var<workgroup> redPartner: array<u32, ${WG}>;
var<workgroup> redHaloCost: array<f32, ${WG}>;
var<workgroup> redHaloPartner: array<u32, ${WG}>;

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wgid: vec3u, @builtin(local_invocation_id) lid3: vec3u) {
    let pIdx = wgid.y * ${MAX_DIM}u + wgid.x;
    let lid = lid3.x;

    // Lane 0 resolves the root and hoists the candidate-independent A side.
    if (lid == 0u) {
        if (pIdx >= uniforms.pendingCount) {
            wgAbort = 1u;
        } else {
            let root = pending[pIdx];
            if (parentMeta[root].x != root) {
                // Stale queued root (absorbed since queuing) — no result.
                outBest[pIdx] = ${partitioned ?
        'vec4u(NONE, bitcast<u32>(F32_MAX), NONE, bitcast<u32>(F32_MAX))' :
        'vec2u(NONE, bitcast<u32>(F32_MAX))'};
                wgAbort = 1u;
            } else {
                wgAbort = 0u;
                wgRoot = root;
                wgSize = parentMeta[root].y;
                var cnt = 0u;
                var m = chain[root].x;
                while (m != NIL && cnt < MAXG) {
                    wgMembers[cnt] = m;
                    cnt++;
                    m = chain[m].y;
                }
                wgCount = cnt;
                let raw = composeRaw(wgMembers, cnt);
                wgRawW = raw.w;
                wgRawMean = raw.mean;
                for (var c = 0u; c < 6u; c++) { wgRawM2[c] = raw.m2[c]; }
                wgRawBw = raw.bw;
                wgATerm = sideTerm(wgMembers, cnt, raw);
            }
        }
    }
    workgroupBarrier();
    // No early return — the reduction barriers below must stay in uniform
    // control flow, so aborted workgroups just run with every lane inactive.
    let aborted = wgAbort == 1u;

    // One lane per (member, neighbour-slot) candidate. Duplicate candidate
    // roots are evaluated redundantly (lockstep makes them free); dedup would
    // only shift tie order.
    var cost = F32_MAX;
    var partner = NONE;
    var haloCost = F32_MAX;
    var haloPartner = NONE;
    let mIdx = lid / K;
    let slot = lid % K;
    if (!aborted && mIdx < wgCount) {
        let cand = nbAt(wgMembers[mIdx], slot);
        if (cand != NONE) {
            let r = findRO(cand);
            if (r != wgRoot && wgSize + parentMeta[r].y <= MAXG) {
                // Gather B's members and rebuild A's raw aggregates locally.
                var bm: array<u32, ${maxGroup}>;
                var bCount = 0u;
                var bmm = chain[r].x;
                while (bmm != NIL && bCount < MAXG) {
                    bm[bCount] = bmm;
                    bCount++;
                    bmm = chain[bmm].y;
                }
                var aRaw: Raw;
                aRaw.w = wgRawW;
                aRaw.mean = wgRawMean;
                for (var c = 0u; c < 6u; c++) { aRaw.m2[c] = wgRawM2[c]; }
                aRaw.bw = wgRawBw;

                let bRaw = composeRaw(bm, bCount);
                let bTerm = sideTerm(bm, bCount, bRaw);
                let fAB = finish(composeUnion(aRaw, bRaw));
                let memfmAB = memfmOf(wgMembers, wgCount, fAB) + memfmOf(bm, bCount, fAB);
                let scross = scrossPairs(wgMembers, wgCount, bm, bCount);
                let dbc = aRaw.bw / aRaw.w - bRaw.bw / bRaw.w;
                let c = (2.0 * scross - 2.0 * memfmAB + fAB.selfM + wgATerm + bTerm) +
                    COLOR_WEIGHT * dot(dbc, dbc);
                // NaN loses every comparison → stays unselected (fail-loud:
                // an all-NaN scene produces no pushes and the caller throws).
                if (c < F32_MAX) {
                    if (r < CORE_COUNT) {
                        cost = c;
                        partner = r;
                    } else {
                        haloCost = c;
                        haloPartner = r;
                    }
                }
            }
        }
    }

    // Lexicographic (cost, partner) min-reduction — deterministic under any
    // lane scheduling.
    redCost[lid] = cost;
    redPartner[lid] = partner;
    redHaloCost[lid] = haloCost;
    redHaloPartner[lid] = haloPartner;
    workgroupBarrier();
    for (var s = ${WG >> 1}u; s > 0u; s >>= 1u) {
        if (lid < s) {
            let c2 = redCost[lid + s];
            let p2 = redPartner[lid + s];
            if (c2 < redCost[lid] || (c2 == redCost[lid] && p2 < redPartner[lid])) {
                redCost[lid] = c2;
                redPartner[lid] = p2;
            }
            let hc2 = redHaloCost[lid + s];
            let hp2 = redHaloPartner[lid + s];
            if (hc2 < redHaloCost[lid] || (hc2 == redHaloCost[lid] && hp2 < redHaloPartner[lid])) {
                redHaloCost[lid] = hc2;
                redHaloPartner[lid] = hp2;
            }
        }
        workgroupBarrier();
    }
    if (lid == 0u && !aborted) {
        var p = redPartner[0];
        if (redCost[0] == F32_MAX) { p = NONE; }
        var hp = redHaloPartner[0];
        if (redHaloCost[0] == F32_MAX) { hp = NONE; }
        outBest[pIdx] = ${partitioned ?
        'vec4u(p, bitcast<u32>(redCost[0]), hp, bitcast<u32>(redHaloCost[0]))' :
        'vec2u(p, bitcast<u32>(redCost[0]))'};
    }
}
`;

// Apply a wave's commit log: per entry (lose, keep, tailKeep, headLose,
// newSize) — three scattered writes, no reads. Race-free because validation
// guarantees each root commits at most once per wave (entries touching a
// committed root fail their version/seq/parent checks), so writes across
// entries target disjoint words.
const replayWgsl = () => /* wgsl */`
struct Uniforms {
    commitCount: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> commitLog: array<u32>;
@group(0) @binding(2) var<storage, read_write> parentMeta: array<vec2u>;
@group(0) @binding(3) var<storage, read_write> chain: array<vec2u>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let e = gid.x;
    if (e >= uniforms.commitCount) { return; }
    let o = e * 5u;
    let lose = commitLog[o];
    let keep = commitLog[o + 1u];
    let tailKeep = commitLog[o + 2u];
    let headLose = commitLog[o + 3u];
    let newSize = commitLog[o + 4u];
    chain[tailKeep].y = headLose;
    parentMeta[lose].x = keep;
    parentMeta[keep].y = newSize;
}
`;

// Initialize the structure buffers: every splat a singleton root.
const initWgsl = () => /* wgsl */`
struct Uniforms {
    count: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> parentMeta: array<vec2u>;
@group(0) @binding(2) var<storage, read_write> chain: array<vec2u>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.y * ${MAX_DIM * 256}u + gid.x;
    if (i >= uniforms.count) { return; }
    parentMeta[i] = vec2u(i, 1u);
    chain[i] = vec2u(i, 0xFFFFFFFFu);
}
`;

/** A wave's commit log entry width in u32s. */
const COMMIT_LOG_STRIDE = 5;

class GpuRecost {
    /** Number of u32 values written per refreshed root. */
    readonly outputStride: number;

    /**
     * Upload the immutable inputs and initialize the structure buffers.
     * Call once before the first wave.
     */
    init: (cache: Float32Array, neighbors: Uint32Array) => void;
    /**
     * Run one wave: replay `commitCount` log entries, refresh
     * `pendingCount` queued roots, and read back (partner, cost) pairs
     * into `outBest` (`outputStride` u32 per root; costs are bitcast f32).
     */
    wave: (
        commitLog: Uint32Array,
        commitCount: number,
        pending: Uint32Array,
        pendingCount: number,
        outBest: Uint32Array
    ) => Promise<void>;
    destroy: () => void;

    /**
     * Whether the wave engine's buffers fit the device's binding limits for
     * `n` splats (pre-flight by arithmetic — a device-side OOM is a hard
     * failure, not a catchable fallback).
     *
     * @param device - PlayCanvas GraphicsDevice (WebGPU).
     * @param n - Generation splat count.
     * @param k - Neighbours per splat.
     * @param wave - Max commits per wave.
     * @param coreCount - Mutable leading rows when sizing block-local output.
     * @param allowSmall - Permit small GPU-local blocks that the global path prefers to evaluate inline.
     * @returns True when all bindings fit.
     */
    static fits(
        device: GraphicsDevice,
        n: number,
        k: number,
        wave: number,
        coreCount = n,
        allowSmall = false
    ): boolean {
        if (n < 1024 && !allowSmall) return false;   // inline is instant below this on the global path
        const limits = (device as any).limits;
        const maxBinding = Math.min(
            typeof limits?.maxStorageBufferBindingSize === 'number' ? limits.maxStorageBufferBindingSize : 128 * 2 ** 20,
            typeof limits?.maxBufferSize === 'number' ? limits.maxBufferSize : 256 * 2 ** 20
        );
        const splitN = Math.ceil(n / 2);
        return splitN * 16 * 4 <= maxBinding &&      // cacheA/B
            splitN * k * 4 <= maxBinding &&           // nbA/B
            n * (coreCount < n ? 16 : 8) <= maxBinding && // outBest
            n * 8 <= maxBinding &&                    // parentMeta/chain
            wave * COMMIT_LOG_STRIDE * 4 <= maxBinding;
    }

    /**
     * @param device - PlayCanvas GraphicsDevice (WebGPU).
     * @param n - Generation splat count.
     * @param k - Neighbours per splat (the refresh workgroup is k·maxGroup lanes).
     * @param maxGroup - Group size cap.
     * @param wave - Max commits per wave (commit log capacity).
     * @param coreCount - Mutable leading rows; remaining rows are immutable halo.
     */
    constructor(device: GraphicsDevice, n: number, k: number, maxGroup: number, wave: number, coreCount = n) {
        if (k * maxGroup !== WG) {
            throw new Error(`GpuRecost: k·maxGroup must be ${WG} (got ${k}·${maxGroup})`);
        }
        const splitN = Math.ceil(n / 2);
        const partitioned = coreCount < n;
        this.outputStride = partitioned ? 4 : 2;
        const outputStrideBytes = this.outputStride * 4;

        const cacheABuf = new StorageBuffer(device, splitN * 16 * 4, BUFFERUSAGE_COPY_DST);
        const cacheBBuf = new StorageBuffer(device, Math.max(n - splitN, 1) * 16 * 4, BUFFERUSAGE_COPY_DST);
        const nbABuf = new StorageBuffer(device, splitN * k * 4, BUFFERUSAGE_COPY_DST);
        const nbBBuf = new StorageBuffer(device, Math.max(n - splitN, 1) * k * 4, BUFFERUSAGE_COPY_DST);
        const parentMetaBuf = new StorageBuffer(device, n * 8, BUFFERUSAGE_COPY_DST);
        const chainBuf = new StorageBuffer(device, n * 8, BUFFERUSAGE_COPY_DST);
        const pendingBuf = new StorageBuffer(device, n * 4, BUFFERUSAGE_COPY_DST);
        const outBestBuf = new StorageBuffer(device, n * outputStrideBytes, BUFFERUSAGE_COPY_SRC | BUFFERUSAGE_COPY_DST);
        const commitLogBuf = new StorageBuffer(device, wave * COMMIT_LOG_STRIDE * 4, BUFFERUSAGE_COPY_DST);

        const initKernel = makeKernel(device, 'recost-init', initWgsl(), ['count'], [
            ['parentMeta', false],
            ['chain', false]
        ]);
        initKernel.compute.setParameter('parentMeta', parentMetaBuf);
        initKernel.compute.setParameter('chain', chainBuf);

        const replayKernel = makeKernel(device, 'recost-replay', replayWgsl(), ['commitCount'], [
            ['commitLog', true],
            ['parentMeta', false],
            ['chain', false]
        ]);
        replayKernel.compute.setParameter('commitLog', commitLogBuf);
        replayKernel.compute.setParameter('parentMeta', parentMetaBuf);
        replayKernel.compute.setParameter('chain', chainBuf);

        const refreshKernel = makeKernel(
            device,
            'recost-refresh',
            refreshWgsl(k, maxGroup, splitN, coreCount, partitioned),
            ['pendingCount'],
            [
                ['cacheA', true],
                ['cacheB', true],
                ['nbA', true],
                ['nbB', true],
                ['parentMeta', true],
                ['chain', true],
                ['pending', true],
                ['outBest', false]
            ]
        );
        refreshKernel.compute.setParameter('cacheA', cacheABuf);
        refreshKernel.compute.setParameter('cacheB', cacheBBuf);
        refreshKernel.compute.setParameter('nbA', nbABuf);
        refreshKernel.compute.setParameter('nbB', nbBBuf);
        refreshKernel.compute.setParameter('parentMeta', parentMetaBuf);
        refreshKernel.compute.setParameter('chain', chainBuf);
        refreshKernel.compute.setParameter('pending', pendingBuf);
        refreshKernel.compute.setParameter('outBest', outBestBuf);

        // Chunked uploads keep Dawn's staging allocations bounded.
        const CHUNK = 1 << 24;
        const writeChunked = (buf: StorageBuffer, data: Float32Array | Uint32Array, srcBase: number, count: number) => {
            for (let off = 0; off < count; off += CHUNK) {
                const c = Math.min(CHUNK, count - off);
                buf.write(off * 4, data, srcBase + off, c);
            }
        };

        this.init = (cache: Float32Array, neighbors: Uint32Array) => {
            writeChunked(cacheABuf, cache, 0, splitN * 16);
            writeChunked(cacheBBuf, cache, splitN * 16, (n - splitN) * 16);
            writeChunked(nbABuf, neighbors, 0, splitN * k);
            writeChunked(nbBBuf, neighbors, splitN * k, (n - splitN) * k);

            const groups = Math.ceil(n / 256);
            initKernel.compute.setParameter('count', n);
            initKernel.compute.setupDispatch(Math.min(groups, MAX_DIM), Math.ceil(groups / MAX_DIM));
            device.computeDispatch([initKernel.compute], 'recost-init');
        };

        this.wave = async (
            commitLog: Uint32Array,
            commitCount: number,
            pending: Uint32Array,
            pendingCount: number,
            outBest: Uint32Array
        ) => {
            const computes = [];
            if (commitCount > 0) {
                commitLogBuf.write(0, commitLog, 0, commitCount * COMMIT_LOG_STRIDE);
                replayKernel.compute.setParameter('commitCount', commitCount);
                replayKernel.compute.setupDispatch(Math.ceil(commitCount / 64));
                computes.push(replayKernel.compute);
            }
            pendingBuf.write(0, pending, 0, pendingCount);
            refreshKernel.compute.setParameter('pendingCount', pendingCount);
            refreshKernel.compute.setupDispatch(
                Math.min(pendingCount, MAX_DIM),
                Math.ceil(pendingCount / MAX_DIM)
            );
            computes.push(refreshKernel.compute);
            device.computeDispatch(computes, 'recost-wave');

            // Blocking readback — also the wave's submit boundary.
            await outBestBuf.read(0, pendingCount * outputStrideBytes, outBest, true);
        };

        this.destroy = () => {
            cacheABuf.destroy();
            cacheBBuf.destroy();
            nbABuf.destroy();
            nbBBuf.destroy();
            parentMetaBuf.destroy();
            chainBuf.destroy();
            pendingBuf.destroy();
            outBestBuf.destroy();
            commitLogBuf.destroy();
            initKernel.destroy();
            replayKernel.destroy();
            refreshKernel.destroy();
        };
    }
}

export { GpuRecost, COMMIT_LOG_STRIDE };
