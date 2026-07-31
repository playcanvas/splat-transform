import { basename, dirname, resolve } from 'pathe';
import { BoundingBox, Mat4, Quat, Vec3 } from 'playcanvas';

import { logWrittenFile } from './utils';
import { writeSogSource } from './write-sog.js';
import { type ChunkDataPool, type ChunkSource, type ReadRequest, createChunkDataPool } from '../chunk';
import { Column, DataTable } from '../data-table';
import { det3, sigmoid, type SplatView } from '../decimate/moment-match';
import { buildCostCache, type CostCache } from '../decimate-uniform/edge-cost-cpu';
import { type FileSystem } from '../io/write';
import { bakeTransform, permuteSource, sortMortonColumns } from '../ops';
import { BTreeNode, BTree } from '../spatial';
import type { DeviceCreator } from '../types';
import { logger, Transform } from '../utils';
import { version } from '../version';

type Aabb = {
    min: number[],
    max: number[]
};

type MetaLod = {
    file: number;
    offset: number;
    count: number;
};

type MetaNode = {
    bound: Aabb;
    children?: MetaNode[];
    lods?: { [key: number]: MetaLod };
    errors?: number[];
};

type LodMeta = {
    version: number;
    asset: {
        generator: string;
    };
    count: number;
    counts: number[];
    lodLevels: number;
    environment?: string;
    filenames: string[];
    tree: MetaNode;
};

const boundUnion = (result: Aabb, a: Aabb, b: Aabb) => {
    const am = a.min;
    const aM = a.max;
    const bm = b.min;
    const bM = b.max;
    const rm = result.min;
    const rM = result.max;

    rm[0] = Math.min(am[0], bm[0]);
    rm[1] = Math.min(am[1], bm[1]);
    rm[2] = Math.min(am[2], bm[2]);
    rM[0] = Math.max(aM[0], bM[0]);
    rM[1] = Math.max(aM[1], bM[1]);
    rM[2] = Math.max(aM[2], bM[2]);
};

/**
 * The only per-gaussian column held resident for the partition: positions.
 * Everything else (rotation/scale for bounds, color/SH for encoding) is gathered
 * from the source on demand, so resident scales as ~12 B/gaussian regardless of
 * SH degree — the point of the streaming LOD writer for very large scenes.
 */
type SlimColumns = {
    x: Float32Array; y: Float32Array; z: Float32Array;
};

/**
 * Overlay a per-unit gathered source so position-layer reads are answered from
 * the resident slim columns (`flat[outputRow]` is the flat analysis index)
 * instead of re-reading the file — the LOD writer already holds every position
 * in memory, so the SOG writer's position phase costs no I/O. Requests carrying
 * other layers forward to `parent` with the position request stripped; requests
 * without a position layer pass through untouched.
 *
 * @param parent - The gathered unit source to overlay.
 * @param slim - The resident position columns, indexed by flat analysis index.
 * @param flat - Flat analysis index of each unit output row.
 * @returns The overlaid source.
 */
const positionsFromSlim = (parent: ChunkSource, slim: SlimColumns, flat: Uint32Array): ChunkSource => {
    const { chunkSize } = parent.meta;
    const read = async (request: ReadRequest): Promise<void> => {
        const pos = request.position;
        if (!pos) return parent.read(request);
        const out = new Float32Array(pos.data);
        if ('indices' in request) {
            const { indices, indexOffset, count } = request;
            for (let j = 0; j < count; j++) {
                const g = flat[indices[indexOffset + j]];
                out[j * 3] = slim.x[g]; out[j * 3 + 1] = slim.y[g]; out[j * 3 + 2] = slim.z[g];
            }
        } else {
            const base = request.chunkIndex * chunkSize;
            const count = Math.min(chunkSize, flat.length - base);
            for (let j = 0; j < count; j++) {
                const g = flat[base + j];
                out[j * 3] = slim.x[g]; out[j * 3 + 1] = slim.y[g]; out[j * 3 + 2] = slim.z[g];
            }
        }
        if (request.geometric || request.color || request.other) {
            await parent.read({ ...request, position: undefined });
        }
    };
    return { meta: parent.meta, read, close: () => parent.close() };
};

// Expand a batch of gathered (position, rotation, scale) records into ellipsoid
// AABBs and fold them into `min`/`max`. Pulled out of `calcBound` so the bounds
// pass can run over gathered batches; the math mirrors the legacy per-gaussian
// path exactly (quaternion order (rot_1, rot_2, rot_3, rot_0); scale = exp).
const accumulateBound = (
    min: number[], max: number[],
    pos: Float32Array, rot: Float32Array, scale: Float32Array, count: number
): void => {
    const p = new Vec3();
    const r = new Quat();
    const s = new Vec3();
    const mat4 = new Mat4();
    const a = new BoundingBox();
    const b = new BoundingBox();

    a.center.set(0, 0, 0);

    for (let i = 0; i < count; i++) {
        p.set(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
        r.set(rot[i * 4 + 1], rot[i * 4 + 2], rot[i * 4 + 3], rot[i * 4 + 0]).normalize();
        s.set(Math.exp(scale[i * 3]), Math.exp(scale[i * 3 + 1]), Math.exp(scale[i * 3 + 2]));
        mat4.setTRS(p, r, Vec3.ONE);

        a.halfExtents.set(s.x, s.y, s.z);
        b.setFromTransformedAabb(a, mat4);

        const m = b.getMin();
        const M = b.getMax();

        if (!isFinite(m.x) || !isFinite(m.y) || !isFinite(m.z) || !isFinite(M.x) || !isFinite(M.y) || !isFinite(M.z)) {
            logger.warn(`skipping invalid bounding box: min=(${m.x}, ${m.y}, ${m.z}) max=(${M.x}, ${M.y}, ${M.z})`);
            continue;
        }

        min[0] = Math.min(min[0], m.x);
        min[1] = Math.min(min[1], m.y);
        min[2] = Math.min(min[2], m.z);
        max[0] = Math.max(max[0], M.x);
        max[1] = Math.max(max[1], M.y);
        max[2] = Math.max(max[2], M.z);
    }
};

// Per-leaf ellipsoid AABB, computed per structural LOD. Positions are resident,
// but rotation/scale are gathered from the source by index so the geometric layer
// is never wholly resident — the bounds-pass analog of the per-unit heavy gather.
// `bins` maps LOD -> flat analysis indices; each is gathered from its own LOD
// (flat index `g` -> local row `g - cum[lod]`).
const calcBound = async (
    source: ChunkSource, pool: ChunkDataPool, bins: Map<number, Uint32Array>, cum: number[],
    tick?: (n: number) => void
): Promise<Aabb> => {
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];

    const batch = pool.chunkSize;
    const { layouts } = source.meta;

    for (const [lodValue, flat] of bins) {
        const base = cum[lodValue];
        const local = new Uint32Array(flat.length);
        for (let i = 0; i < flat.length; ++i) local[i] = flat[i] - base;

        for (let off = 0; off < local.length; off += batch) {
            const count = Math.min(batch, local.length - off);
            const pos = pool.acquire('position', layouts.position!, count);
            const geo = pool.acquire('geometric', layouts.geometric!, count);
            await source.read({ indices: local, indexOffset: off, count, lod: lodValue, position: pos, geometric: geo });
            accumulateBound(
                min, max,
                // position is full-stride packed xyz — read the pool buffer
                // in place rather than copying it out per batch
                new Float32Array(pos.data, 0, count * 3),
                geo.field('rotation') as Float32Array,
                geo.field('scale') as Float32Array,
                count
            );
            pos.release();
            geo.release();
            tick?.(count);
        }
    }

    return { min, max };
};

// Group the flat analysis indices under `parent` by their structural LOD
// (`lodOf(flatIndex)`). Two passes (count, then fill) so each LOD's indices land
// in a tight `Uint32Array` rather than a `number[]` — the indices are the
// dominant retained bookkeeping for a large scene, so keeping them off the V8
// heap (4 B each, no GC pressure) is what lets LOD export scale to hundreds of
// millions of splats.
const binIndices = (parent: BTreeNode, lodOf: (index: number) => number): Map<number, Uint32Array> => {
    const counts = new Map<number, number>();
    const tally = (node: BTreeNode) => {
        if (node.indices) {
            for (let i = 0; i < node.indices.length; ++i) {
                const lodValue = lodOf(node.indices[i]);
                counts.set(lodValue, (counts.get(lodValue) ?? 0) + 1);
            }
        } else {
            if (node.left) tally(node.left);
            if (node.right) tally(node.right);
        }
    };
    tally(parent);

    const result = new Map<number, Uint32Array>();
    const offset = new Map<number, number>();
    for (const [lodValue, count] of counts) {
        result.set(lodValue, new Uint32Array(count));
        offset.set(lodValue, 0);
    }

    const fill = (node: BTreeNode) => {
        if (node.indices) {
            for (let i = 0; i < node.indices.length; ++i) {
                const v = node.indices[i];
                const lodValue = lodOf(v);
                const o = offset.get(lodValue)!;
                result.get(lodValue)![o] = v;
                offset.set(lodValue, o + 1);
            }
        } else {
            if (node.left) fill(node.left);
            if (node.right) fill(node.right);
        }
    };
    fill(parent);

    return result;
};

const distanceToAabbSq = (node: BTreeNode, x: number, y: number, z: number): number => {
    const { min, max } = node.aabb;
    const dx = x < min[0] ? min[0] - x : x > max[0] ? x - max[0] : 0;
    const dy = y < min[1] ? min[1] - y : y > max[1] ? y - max[1] : 0;
    const dz = z < min[2] ? min[2] - z : z > max[2] ? z - max[2] : 0;
    return dx * dx + dy * dy + dz * dz;
};

const findNearest = (
    root: BTreeNode, slim: SlimColumns, lodOf: (index: number) => number,
    queries: Uint32Array, targetLod: number, k: number
): Int32Array => {
    const result = new Int32Array(queries.length * k);
    result.fill(-1);
    const distances = new Float64Array(k);

    for (let q = 0; q < queries.length; q++) {
        distances.fill(Infinity);
        const g = queries[q];
        const x = slim.x[g], y = slim.y[g], z = slim.z[g];
        const output = q * k;

        const visit = (node: BTreeNode): void => {
            if (distanceToAabbSq(node, x, y, z) > distances[k - 1]) return;

            if (node.indices) {
                for (let i = 0; i < node.indices.length; i++) {
                    const candidate = node.indices[i];
                    if (lodOf(candidate) !== targetLod) continue;
                    const dx = slim.x[candidate] - x;
                    const dy = slim.y[candidate] - y;
                    const dz = slim.z[candidate] - z;
                    const distance = dx * dx + dy * dy + dz * dz;
                    if (distance >= distances[k - 1]) continue;

                    let insert = k - 1;
                    while (insert > 0 && distance < distances[insert - 1]) {
                        distances[insert] = distances[insert - 1];
                        result[output + insert] = result[output + insert - 1];
                        insert--;
                    }
                    distances[insert] = distance;
                    result[output + insert] = candidate;
                }
                return;
            }

            const left = node.left!;
            const right = node.right!;
            if (distanceToAabbSq(left, x, y, z) <= distanceToAabbSq(right, x, y, z)) {
                visit(left);
                visit(right);
            } else {
                visit(right);
                visit(left);
            }
        };
        visit(root);
    }

    return result;
};

const uniqueGlobals = (queries: Uint32Array, neighbours: Int32Array): Uint32Array => {
    const combined = new Uint32Array(queries.length + neighbours.length);
    combined.set(queries);
    let count = queries.length;
    for (let i = 0; i < neighbours.length; i++) {
        if (neighbours[i] >= 0) combined[count++] = neighbours[i];
    }
    const sorted = combined.subarray(0, count);
    sorted.sort();
    let unique = 0;
    for (let i = 0; i < sorted.length; i++) {
        if (i === 0 || sorted[i] !== sorted[i - 1]) sorted[unique++] = sorted[i];
    }
    return sorted.slice(0, unique);
};

const indexOfSorted = (values: Uint32Array, value: number): number => {
    let lo = 0;
    let hi = values.length - 1;
    while (lo <= hi) {
        const mid = (lo + hi) >>> 1;
        const candidate = values[mid];
        if (candidate < value) lo = mid + 1;
        else if (candidate > value) hi = mid - 1;
        else return mid;
    }
    return -1;
};

const gatherView = async (
    source: ChunkSource, pool: ChunkDataPool, slim: SlimColumns,
    globals: Uint32Array, lod: number, base: number
): Promise<SplatView> => {
    const { layouts } = source.meta;
    const colorDim = layouts.color!.stride >> 2;
    const view: SplatView = {
        pos: new Float32Array(globals.length * 3),
        geo: new Float32Array(globals.length * 8),
        color: new Float32Array(globals.length * colorDim),
        colorDim
    };
    const local = new Uint32Array(globals.length);
    for (let i = 0; i < globals.length; i++) {
        const g = globals[i];
        local[i] = g - base;
        view.pos[i * 3] = slim.x[g];
        view.pos[i * 3 + 1] = slim.y[g];
        view.pos[i * 3 + 2] = slim.z[g];
    }

    for (let off = 0; off < local.length; off += pool.chunkSize) {
        const count = Math.min(pool.chunkSize, local.length - off);
        const geo = pool.acquire('geometric', layouts.geometric!, count);
        const color = pool.acquire('color', layouts.color!, count);
        await source.read({ indices: local, indexOffset: off, count, lod, geometric: geo, color });
        view.geo.set(new Float32Array(geo.data, 0, count * 8), off * 8);
        view.color.set(new Float32Array(color.data, 0, count * colorDim), off * colorDim);
        geo.release();
        color.release();
    }

    return view;
};

const combineViews = (a: SplatView, b: SplatView): SplatView => {
    const countA = a.pos.length / 3;
    const countB = b.pos.length / 3;
    const result: SplatView = {
        pos: new Float32Array((countA + countB) * 3),
        geo: new Float32Array((countA + countB) * 8),
        color: new Float32Array((countA + countB) * a.colorDim),
        colorDim: a.colorDim
    };
    result.pos.set(a.pos); result.pos.set(b.pos, a.pos.length);
    result.geo.set(a.geo); result.geo.set(b.geo, a.geo.length);
    result.color.set(a.color); result.color.set(b.color, a.color.length);
    return result;
};

const PI_1_5 = Math.PI ** 1.5;
const TWO_PI_1_5 = (2 * Math.PI) ** 1.5;

// Per-splat terms of the field-L2: the gaussian's opacity, sqrt(det sigma), and
// its self inner product <f, f> = alpha^2 * pi^1.5 * sqrt(det sigma).
type ErrorCache = {
    alpha: Float64Array;
    sqrtDet: Float64Array;
    self: Float64Array;
};

const buildErrorCache = (view: SplatView, cache: CostCache): ErrorCache => {
    const n = view.geo.length / 8;
    const alpha = new Float64Array(n);
    const sqrtDet = new Float64Array(n);
    const self = new Float64Array(n);

    for (let i = 0; i < n; i++) {
        const a = sigmoid(view.geo[i * 8 + 7]);
        const root = Math.sqrt(Math.max(det3(cache.sigma, i * 9), 1e-300));
        alpha[i] = a;
        sqrtDet[i] = root;
        self[i] = a * a * PI_1_5 * root;
    }

    return { alpha, sqrtDet, self };
};

const sumScratch = new Float64Array(9);

/**
 * Approximation error between two splats: the relative field-L2 of the density
 * each one paints plus the stored-SH L2.
 *
 * Writing f(x) = alpha * exp(-0.5 (x-mu)^T sigma^-1 (x-mu)), every inner product
 * <f_i, f_j> is closed form, so
 *
 *   ||f_i - f_j||^2 / (||f_i||^2 + ||f_j||^2)
 *
 * is exact: zero only for identical splats, monotone in their separation, and
 * bounded by 1 once they no longer overlap. The decimator's edge cost is not
 * usable here — as a one-sample MC estimate of a KL it carries ~1.2 nats of
 * noise, far more than the error a well-decimated leaf actually has, and it is
 * blind to opacity (alpha only sets the merge weights there).
 *
 * @param view - Splat columns.
 * @param cache - Per-splat cache from `buildCostCache` (supplies sigma).
 * @param errorCache - Per-splat cache from {@link buildErrorCache}.
 * @param i - First splat (view row).
 * @param j - Second splat (view row).
 * @returns The error, zero when the two splats are identical.
 */
const splatError = (
    view: SplatView, cache: CostCache, errorCache: ErrorCache, i: number, j: number
): number => {
    const s = sumScratch;
    const i9 = 9 * i, j9 = 9 * j;
    for (let a = 0; a < 9; a++) s[a] = cache.sigma[i9 + a] + cache.sigma[j9 + a];
    const detS = Math.max(det3(s, 0), 1e-300);

    // d^T (sigma_i + sigma_j)^-1 d, via the adjugate (symmetric, so no transpose)
    const { pos } = view;
    const dx = pos[3 * i] - pos[3 * j];
    const dy = pos[3 * i + 1] - pos[3 * j + 1];
    const dz = pos[3 * i + 2] - pos[3 * j + 2];
    const ax = (s[4] * s[8] - s[5] * s[7]) * dx + (s[2] * s[7] - s[1] * s[8]) * dy + (s[1] * s[5] - s[2] * s[4]) * dz;
    const ay = (s[5] * s[6] - s[3] * s[8]) * dx + (s[0] * s[8] - s[2] * s[6]) * dy + (s[2] * s[3] - s[0] * s[5]) * dz;
    const az = (s[3] * s[7] - s[4] * s[6]) * dx + (s[1] * s[6] - s[0] * s[7]) * dy + (s[0] * s[4] - s[1] * s[3]) * dz;
    const quad = (dx * ax + dy * ay + dz * az) / detS;

    const cross = errorCache.alpha[i] * errorCache.alpha[j] * TWO_PI_1_5 *
        errorCache.sqrtDet[i] * errorCache.sqrtDet[j] / Math.sqrt(detS) * Math.exp(-0.5 * quad);
    const total = errorCache.self[i] + errorCache.self[j];
    const geoError = Math.max(0, total - 2 * cross) / Math.max(total, 1e-300);

    const { color, colorDim } = view;
    let colorError = 0;
    for (let c = 0; c < colorDim; c++) {
        const d = color[i * colorDim + c] - color[j * colorDim + c];
        colorError += d * d;
    }

    return geoError + colorError;
};

const calcErrors = async (
    source: ChunkSource, pool: ChunkDataPool, slim: SlimColumns, root: BTreeNode,
    bins: Map<number, Uint32Array>, lodOf: (index: number) => number, cum: number[], numLods: number
): Promise<number[]> => {
    // Symmetric Chamfer-style error against the finest representation present in
    // this leaf, plus the alpha-mass a level fails to carry. Center-space KNN
    // supplies a small candidate set (including candidates across leaf
    // boundaries); the final match uses the closed-form {@link splatError}. This
    // compares arbitrary input LODs, so it does not depend on how those LODs
    // were produced.
    const errors = new Array(numLods).fill(0);
    const referenceLod = Math.min(...bins.keys());
    const referenceQueries = bins.get(referenceLod)!;
    const k = 4;

    for (const [lod, targetQueries] of bins) {
        if (lod === referenceLod) continue;

        const forward = findNearest(root, slim, lodOf, referenceQueries, lod, k);
        const reverse = findNearest(root, slim, lodOf, targetQueries, referenceLod, k);
        const referenceGlobals = uniqueGlobals(referenceQueries, reverse);
        const targetGlobals = uniqueGlobals(targetQueries, forward);
        const referenceView = await gatherView(source, pool, slim, referenceGlobals, referenceLod, cum[referenceLod]);
        const targetView = await gatherView(source, pool, slim, targetGlobals, lod, cum[lod]);
        const view = combineViews(referenceView, targetView);
        const cache = buildCostCache(view);
        const errorCache = buildErrorCache(view, cache);
        const targetOffset = referenceGlobals.length;

        // `mass` (alpha * ellipsoid area) is a splat's share of what the leaf
        // paints, so it weights both the per-splat mean and the coverage term —
        // a hair-thin splat and one spanning the whole leaf are not equals.
        const directional = (
            queries: Uint32Array, neighbours: Int32Array,
            queryGlobals: Uint32Array, neighbourGlobals: Uint32Array,
            queryOffset: number, neighbourOffset: number
        ): number => {
            let total = 0;
            let weight = 0;
            for (let i = 0; i < queries.length; i++) {
                const queryRow = queryOffset + indexOfSorted(queryGlobals, queries[i]);
                let best = Infinity;
                for (let j = 0; j < k; j++) {
                    const neighbour = neighbours[i * k + j];
                    if (neighbour < 0) continue;
                    const neighbourRow = neighbourOffset + indexOfSorted(neighbourGlobals, neighbour);
                    best = Math.min(best, splatError(view, cache, errorCache, queryRow, neighbourRow));
                }
                if (best === Infinity) continue;
                total += cache.mass[queryRow] * best;
                weight += cache.mass[queryRow];
            }
            return weight > 0 ? total / weight : 0;
        };

        const massOf = (queries: Uint32Array, globals: Uint32Array, offset: number): number => {
            let mass = 0;
            for (let i = 0; i < queries.length; i++) {
                mass += cache.mass[offset + indexOfSorted(globals, queries[i])];
            }
            return mass;
        };

        // Nearest-neighbour matching cannot see thinning: drop every second
        // splat of an overlapping group and each survivor still has a
        // near-identical partner, while the alpha the group accumulates halves.
        // Sky gaps are exactly that, so compare the mass the levels carry.
        const referenceMass = massOf(referenceQueries, referenceGlobals, 0);
        const targetMass = massOf(targetQueries, targetGlobals, targetOffset);
        const coverageError = Math.max(0, 1 - targetMass / Math.max(referenceMass, 1e-300));

        const forwardError = directional(referenceQueries, forward, referenceGlobals, targetGlobals, 0, targetOffset);
        const reverseError = directional(targetQueries, reverse, targetGlobals, referenceGlobals, targetOffset, 0);
        errors[lod] = 0.5 * (forwardError + reverseError) + coverageError;
    }

    // The engine's budget balancer drops a level from its frontier whenever a
    // coarser one claims less error, which pins the node to that coarser level
    // at any budget. Keep the table monotone so a level can only ever be
    // dropped for being genuinely no better.
    let previous = 0;
    for (const lod of [...bins.keys()].sort((a, b) => a - b)) {
        previous = errors[lod] = Math.max(errors[lod], previous);
    }

    return errors;
};

/**
 * Read positions out of a multi-LOD source into flat per-gaussian arrays — one
 * sequential pass across every structural LOD (LOD 0 first, then 1, …, laid out
 * contiguously). Nothing else is materialized here (for a fixed-stride file
 * source the rotation/scale/color/SH bytes are read-and-discarded); rotation/
 * scale are gathered per leaf for bounds, and the heavy layers per unit at encode
 * time. Flat gaussian `g` belongs to the LOD whose cumulative range contains it.
 *
 * @param source - The PLY-space scene source (one or more structural LODs).
 * @param pool - Pool for the temporary per-chunk read buffers.
 * @returns The flat position columns, indexed by gaussian across all LODs.
 */
const extractSlim = async (source: ChunkSource, pool: ChunkDataPool): Promise<SlimColumns> => {
    const { meta } = source;
    const N = meta.lodCounts.reduce((acc, c) => acc + c, 0);
    const cols: SlimColumns = {
        x: new Float32Array(N),
        y: new Float32Array(N),
        z: new Float32Array(N)
    };

    const { chunkSize } = meta;
    let base = 0;
    for (let lod = 0; lod < meta.numLods; lod++) {
        const lodCount = meta.lodCounts[lod];
        const numChunks = meta.numChunks[lod];
        for (let k = 0; k < numChunks; k++) {
            const count = Math.min(chunkSize, lodCount - k * chunkSize);
            const pos = pool.acquire('position', meta.layouts.position!, count);
            await source.read({ chunkIndex: k, lod, position: pos });

            // position is full-stride packed xyz — read the pool buffer in
            // place (this loop visits every gaussian of every LOD)
            const p = new Float32Array(pos.data, 0, count * 3);
            for (let i = 0; i < count; i++) {
                const di = base + i;
                cols.x[di] = p[i * 3]; cols.y[di] = p[i * 3 + 1]; cols.z[di] = p[i * 3 + 2];
            }
            base += count;
            pos.release();
        }
    }
    return cols;
};

type WriteLodSourceOptions = {
    filename: string;
    /**
     * The scene as a structural multi-LOD source: LOD `i` is output detail level
     * `i`. Streamed SOG/lcc/lcc2 expose this intrinsically; multi-PLY `--tag-lod` inputs are stacked
     * by tag via {@link stackLods}. No per-gaussian lod tag — LOD is structural.
     */
    mainSource: ChunkSource;
    envSource: ChunkSource | null;
    iterations: number;
    createDevice?: DeviceCreator;
    chunkCount: number;
    chunkExtent: number;
};

/**
 * Writes Gaussian splat data to multi-LOD format with spatial chunking. The main
 * source's pending coordinate-space transform is baked to PLY space up front, so
 * the spatial tree/bounds and the SOG payloads share one coordinate space.
 *
 * Creates a hierarchical structure with multiple LOD levels, each stored in
 * separate SOG files, plus a binary-tree spatial index for view-dependent
 * loading. The partition / per-leaf bounds / lod binning run over flat analysis
 * columns extracted from `mainSource`; each unit's gaussians are gathered lazily
 * from `mainSource` via {@link permuteSource} and encoded chunk-native.
 *
 * @param options - Options including filename, sources, and chunking parameters.
 * @param fs - File system for writing output files.
 * @ignore
 */
const writeLodSource = async (options: WriteLodSourceOptions, fs: FileSystem) => {
    const { filename, envSource, iterations, createDevice, chunkCount, chunkExtent } = options;

    // Bake the pending coordinate-space transform to PLY once, up front, so the
    // partition/bounds passes (extractSlim, calcBound, morton) and the per-unit
    // SOG payloads all read the same space (writeSogSource's internal bake then
    // sees identity). Mirrors the legacy writer's convert-to-PLY-before-tree
    // step; a PLY-space input hits bakeTransform's identity fast-path. The env
    // needs no bake here: no bounds are computed from it and writeSogSource
    // bakes it itself.
    const mainSource = bakeTransform(options.mainSource, Transform.PLY);

    // Pool for slim extraction read buffers and the chunk-native SOG encodes.
    const pool = createChunkDataPool();

    const slim = await extractSlim(mainSource, pool);
    const hasEnv = !!envSource && envSource.meta.numGaussians > 0;

    // LOD is structural: flat analysis gaussian `g` belongs to the LOD whose
    // cumulative range contains it. `cum[L]` is LOD L's flat base (and the offset
    // to convert a flat index back to a row local to that LOD for gathering).
    const { lodCounts, numLods } = mainSource.meta;
    const cum = [0];
    for (let l = 0; l < numLods; l++) cum.push(cum[l] + lodCounts[l]);
    const lodOf = (g: number): number => {
        for (let l = numLods - 1; l > 0; l--) {
            if (g >= cum[l]) return l;
        }
        return 0;
    };

    const outputDir = dirname(filename);

    // ensure top-level output folder exists
    await fs.mkdir(outputDir);

    // construct a kd-tree based on centroids from all lods
    const centroidsTable = new DataTable([
        new Column('x', slim.x),
        new Column('y', slim.y),
        new Column('z', slim.z)
    ]);

    let bTree: BTree | null = new BTree(centroidsTable);

    // approximate number of gaussians we'll place into file units
    const binSize = chunkCount * 1024;
    const binDim = chunkExtent;

    // map of lod -> file units -> subunits (each subunit a tight Uint32Array of
    // gaussian indices). This is the bulk retained bookkeeping; Uint32Array keeps
    // it off the V8 heap at 4 B/gaussian.
    const lodFiles: Map<number, Uint32Array[][]> = new Map();
    const filenames: string[] = [];
    let lodLevels = 0;

    // Every gaussian lands in exactly one leaf, so leaf-bounds batches tick the
    // bar to the total gaussian count across LODs.
    const chunkingBar = logger.bar('chunking', cum[numLods]);

    const build = async (node: BTreeNode): Promise<MetaNode> => {
        if (!node.indices && (node.count > binSize || (node.aabb && node.aabb.largestDim() > binDim))) {
            const children = [
                await build(node.left),
                await build(node.right)
            ];

            const bound = {
                min: [0, 0, 0],
                max: [0, 0, 0]
            };
            boundUnion(bound, children[0].bound, children[1].bound);

            return { bound, children };
        }

        const lods: { [key: number]: MetaLod } = { };
        const bins = binIndices(node, lodOf);

        for (const [lodValue, indices] of bins) {
            if (!lodFiles.has(lodValue)) {
                lodFiles.set(lodValue, [[]]);
            }
            const fileList = lodFiles.get(lodValue);
            const fileIndex = fileList.length - 1;
            const lastFile = fileList[fileIndex];
            const fileSize = lastFile.reduce((acc, curr) => acc + curr.length, 0);

            const filename = `${lodValue}_${fileIndex}/meta.json`;
            if (filenames.indexOf(filename) === -1) {
                filenames.push(filename);
            }

            lods[lodValue] = {
                file: filenames.indexOf(filename),
                offset: fileSize,
                count: indices.length
            };

            lastFile.push(indices);

            if (fileSize + indices.length > binSize) {
                fileList.push([]);
            }

            lodLevels = Math.max(lodLevels, lodValue + 1);
        }

        // Bound and approximation errors over the leaf's full structural LOD data.
        const bound = await calcBound(mainSource, pool, bins, cum, n => chunkingBar.tick(n));
        const errors = await calcErrors(mainSource, pool, slim, bTree!.root, bins, lodOf, cum, numLods);

        return { bound, lods, errors };
    };

    let tree: MetaNode;
    try {
        tree = await build(bTree.root);
    } finally {
        chunkingBar.end();
    }

    // The kd-tree is dead once the partition is built (lodFiles holds its own
    // index copies): release its N×4B index buffer and node AABBs before the
    // unit writes, where peak memory lives.
    bTree = null;

    // count splats per lod level
    const counts = new Array(lodLevels).fill(0);
    for (const [lodValue, fileUnits] of lodFiles) {
        for (const fileUnit of fileUnits) {
            counts[lodValue] += fileUnit.reduce((acc, curr) => acc + curr.length, 0);
        }
    }

    const meta: LodMeta = {
        version: 1,
        asset: {
            generator: `splat-transform v${version}`
        },
        count: counts.reduce((acc, curr) => acc + curr, 0),
        counts,
        lodLevels,
        ...(hasEnv ? { environment: 'env/meta.json' } : {}),
        filenames,
        tree
    };

    // write the meta file with float precision quantization (approx. 32-bit float => ~7 significant digits)
    const replacer = (_key: string, value: any) => {
        if (typeof value === 'number') {
            if (!Number.isFinite(value)) return value;
            return Number.isInteger(value) ? value : +value.toPrecision(7);
        }
        return value;
    };

    const writingGroup = logger.group('Writing');

    // count the total number of sog units we'll write so the per-sog groups
    // can render as a numbered series
    let sogTotal = 0;
    if (hasEnv) sogTotal += 1;
    for (const [, fileUnits] of lodFiles) {
        for (const fu of fileUnits) {
            if (fu.length > 0) sogTotal += 1;
        }
    }

    let sogIndex = 0;

    // write the environment sog
    if (hasEnv) {
        sogIndex++;
        const envGroup = logger.group('env', { index: sogIndex, total: sogTotal });
        try {
            const envPathname = resolve(outputDir, 'env/meta.json');

            // ensure output folder exists before any files are written
            await fs.mkdir(dirname(envPathname));

            await writeSogSource(
                envSource!,
                pool,
                { filename: envPathname, bundle: false, iterations, createDevice, logging: 'flat' },
                fs
            );
        } finally {
            envGroup.end();
        }
    }

    // write lod-meta.json
    const metaJson = (new TextEncoder()).encode(JSON.stringify(meta, replacer));
    const writer = await fs.createWriter(filename);
    await writer.write(metaJson);
    await writer.close();
    logWrittenFile(basename(filename), writer.bytesWritten);

    // write file units
    for (const [lodValue, fileUnits] of lodFiles) {
        for (let i = 0; i < fileUnits.length; ++i) {
            const fileUnit = fileUnits[i];

            if (fileUnit.length === 0) {
                continue;
            }

            const groupName = `${lodValue}_${i}`;
            sogIndex++;
            const unitGroup = logger.group(groupName, { index: sogIndex, total: sogTotal });

            try {
                // ensure output folder exists before any files are written
                const pathname = resolve(outputDir, `${lodValue}_${i}/meta.json`);
                await fs.mkdir(dirname(pathname));

                // Morton-order each subunit and concatenate into the unit's
                // global (flat) row order.
                const totalIndices = fileUnit.reduce((acc, curr) => acc + curr.length, 0);
                const orderedIndices = new Uint32Array(totalIndices);
                for (let j = 0, offset = 0; j < fileUnit.length; ++j) {
                    orderedIndices.set(fileUnit[j], offset);
                    sortMortonColumns(slim.x, slim.y, slim.z, orderedIndices.subarray(offset, offset + fileUnit[j].length));
                    offset += fileUnit[j].length;
                }

                // This file unit's flat indices all belong to LOD `lodValue`;
                // convert to rows local to that LOD for the gather.
                const base = cum[lodValue];
                const orderedLocal = new Uint32Array(totalIndices);
                for (let j = 0; j < totalIndices; ++j) orderedLocal[j] = orderedIndices[j] - base;

                // Gather the ordered subset lazily from LOD `lodValue` (no per-unit
                // copy) and encode via the chunk-native SOG writer. The rows are
                // already in write order, so pass an identity ordering to skip the
                // writer's own Morton pass.
                const unitSource = positionsFromSlim(
                    permuteSource(mainSource, orderedLocal, { lod: lodValue }),
                    slim, orderedIndices
                );
                const identity = new Uint32Array(totalIndices);
                for (let j = 0; j < totalIndices; ++j) identity[j] = j;

                await writeSogSource(unitSource, pool, {
                    filename: pathname,
                    bundle: false,
                    iterations,
                    createDevice,
                    indices: identity,
                    logging: 'flat'
                }, fs);
            } finally {
                unitGroup.end();
            }
        }
    }

    writingGroup.end();
};

export { positionsFromSlim, writeLodSource, type WriteLodSourceOptions };
