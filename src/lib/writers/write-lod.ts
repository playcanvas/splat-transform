import { basename, dirname, resolve } from 'pathe';
import { BoundingBox, Mat4, Quat, Vec3 } from 'playcanvas';

import { ATLAS_MAX_BATCH_GAUSSIANS, ErrorRenderer, type LeafLevels, leafViewSize } from './lod-error';
import { logWrittenFile } from './utils';
import { writeSogSource } from './write-sog.js';
import { type ChunkDataPool, type ChunkSource, type ReadRequest, createChunkDataPool } from '../chunk';
import { Column, DataTable } from '../data-table';
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
        /** Gaussians per file unit the partition aimed for (`--lod-chunk-count` × 1024). */
        chunkGaussians: number;
        /** Largest leaf extent the partition allowed, in world units (`--lod-chunk-extent`). */
        chunkExtent: number;
    };
    count: number;
    counts: number[];
    lodLevels: number;
    /**
     * Whether every leaf carries an `errors` table. Declared here so a consumer
     * can pick its LOD allocation strategy up front instead of searching the tree
     * for the field (the engine's budget balancer needs exactly this answer).
     */
    lodErrors: boolean;
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

const invalidGaussian = (lod: number, row: number, what: string): Error => new Error(
    `LOD ${lod} gaussian ${row} has ${what}; ` +
    'run --filter-nan to drop invalid gaussians before writing LODs'
);

/**
 * Reject a batch of gaussians whose geometry is not finite. The bounds pass reads
 * every gaussian's geometric record once, so this covers the whole scene:
 * everything downstream — the error pass above all, which renders every gaussian
 * — may then assume finite input rather than each stage carrying its own opinion
 * about invalid data. {@link assertFiniteColor} does the same for the layer this
 * pass does not read.
 *
 * The rules mirror `filterNaNRows`, including its two deliberate exceptions
 * (`scale_*` may be `-Infinity`, `opacity` may be `+Infinity`, both harmless
 * here), so anything `--filter-nan` keeps is accepted.
 *
 * @param pos - Packed xyz for the batch.
 * @param geo - Packed 8-float geometric records for the batch.
 * @param count - Gaussians in the batch.
 * @param rows - Rows local to `lod`, indexed from `offset`, for error messages.
 * @param offset - Index of the batch's first row within `rows`.
 * @param lod - Structural LOD the batch was read from.
 */
const assertFiniteGeometry = (
    pos: Float32Array, geo: Float32Array, count: number,
    rows: Uint32Array, offset: number, lod: number
): void => {
    const reject = (i: number, what: string) => {
        throw invalidGaussian(lod, rows[offset + i], what);
    };

    for (let i = 0; i < count; i++) {
        const p = i * 3;
        if (!isFinite(pos[p]) || !isFinite(pos[p + 1]) || !isFinite(pos[p + 2])) {
            reject(i, 'a non-finite position');
        }

        const o = i * 8;
        if (!isFinite(geo[o]) || !isFinite(geo[o + 1]) || !isFinite(geo[o + 2]) || !isFinite(geo[o + 3])) {
            reject(i, 'a non-finite rotation');
        }
        if (geo[o] === 0 && geo[o + 1] === 0 && geo[o + 2] === 0 && geo[o + 3] === 0) {
            reject(i, 'a zero-norm rotation');
        }
        for (let e = 4; e <= 6; e++) {
            const v = geo[o + e];
            if (!isFinite(v) && v !== -Infinity) reject(i, 'a non-finite scale');
        }
        const opacity = geo[o + 7];
        if (!isFinite(opacity) && opacity !== Infinity) reject(i, 'a non-finite opacity');
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
            // position is full-stride packed xyz — read the pool buffer in place
            // rather than copying it out per batch
            const posBatch = new Float32Array(pos.data, 0, count * 3);
            assertFiniteGeometry(
                posBatch, new Float32Array(geo.data, 0, count * 8), count, local, off, lodValue
            );
            accumulateBound(
                min, max,
                posBatch,
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

/**
 * Reject a batch of gaussians whose color or stored SH is not finite — the
 * companion to {@link assertFiniteGeometry} for the layer the bounds pass does not
 * read. A non-finite coefficient would paint NaN into the error renders and the
 * comparison would silently absorb it, so it is refused up front like invalid
 * geometry. Coverage is again the whole scene: every gaussian of a leaf is
 * gathered for its own level's render.
 *
 * @param color - Packed color/SH records for the batch.
 * @param count - Gaussians in the batch.
 * @param colorDim - Floats per gaussian (3 + stored SH).
 * @param rows - Rows local to `lod`, indexed from `offset`, for error messages.
 * @param offset - Index of the batch's first row within `rows`.
 * @param lod - Structural LOD the batch was read from.
 */
const assertFiniteColor = (
    color: Float32Array, count: number, colorDim: number,
    rows: Uint32Array, offset: number, lod: number
): void => {
    for (let i = 0; i < count; i++) {
        const base = i * colorDim;
        for (let c = 0; c < colorDim; c++) {
            if (!isFinite(color[base + c])) {
                throw invalidGaussian(lod, rows[offset + i], 'a non-finite color or SH coefficient');
            }
        }
    }
};

const GEOMETRIC_COLUMNS = ['rot_0', 'rot_1', 'rot_2', 'rot_3', 'scale_0', 'scale_1', 'scale_2', 'opacity'];

/**
 * Gather one structural level of a leaf into a resident {@link DataTable} for the
 * rasterizer. Positions come from the resident slim columns; rotation, scale,
 * opacity and color/SH are read from the source by index in pool-sized batches —
 * the same per-leaf gather the bounds pass does for its layers.
 *
 * The table stays in PLY space, tagged as such. The error pass renders scene and
 * camera in one space and only ever compares renders against each other, so the
 * space is immaterial as long as it is shared; PLY-space SH is evaluated against
 * PLY-space view directions, which is self-consistent.
 *
 * @param source - The PLY-space scene source.
 * @param pool - Pool for the temporary per-batch read buffers.
 * @param slim - Resident position columns.
 * @param indices - Flat analysis indices of the level's gaussians in this leaf.
 * @param lod - The structural LOD the indices belong to.
 * @param base - Flat base of `lod`, converting a flat index to a row local to it.
 * @returns The level's gaussians as a table in PLY space.
 */
const gatherLeafTable = async (
    source: ChunkSource, pool: ChunkDataPool, slim: SlimColumns,
    indices: Uint32Array, lod: number, base: number
): Promise<DataTable> => {
    const { layouts } = source.meta;
    const colorDim = layouts.color!.stride >> 2;
    const n = indices.length;

    const position = [new Float32Array(n), new Float32Array(n), new Float32Array(n)];
    const geometric = GEOMETRIC_COLUMNS.map(() => new Float32Array(n));
    const color = Array.from({ length: colorDim }, () => new Float32Array(n));

    const local = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
        const g = indices[i];
        local[i] = g - base;
        position[0][i] = slim.x[g];
        position[1][i] = slim.y[g];
        position[2][i] = slim.z[g];
    }

    for (let off = 0; off < n; off += pool.chunkSize) {
        const count = Math.min(pool.chunkSize, n - off);
        const geo = pool.acquire('geometric', layouts.geometric!, count);
        const col = pool.acquire('color', layouts.color!, count);
        await source.read({ indices: local, indexOffset: off, count, lod, geometric: geo, color: col });

        const geoBatch = new Float32Array(geo.data, 0, count * 8);
        const colorBatch = new Float32Array(col.data, 0, count * colorDim);
        assertFiniteColor(colorBatch, count, colorDim, local, off, lod);
        for (let i = 0; i < count; i++) {
            for (let k = 0; k < 8; k++) geometric[k][off + i] = geoBatch[i * 8 + k];
            for (let k = 0; k < colorDim; k++) color[k][off + i] = colorBatch[i * colorDim + k];
        }

        geo.release();
        col.release();
    }

    const columns = [
        new Column('x', position[0]),
        new Column('y', position[1]),
        new Column('z', position[2]),
        ...GEOMETRIC_COLUMNS.map((name, k) => new Column(name, geometric[k])),
        new Column('f_dc_0', color[0]),
        new Column('f_dc_1', color[1]),
        new Column('f_dc_2', color[2])
    ];
    for (let k = 3; k < colorDim; k++) columns.push(new Column(`f_rest_${k - 3}`, color[k]));

    return new DataTable(columns, Transform.PLY);
};

/** A leaf awaiting its error table: the node it fills in and the indices to gather. */
type LeafJob = {
    node: MetaNode;
    bins: Map<number, Uint32Array>;
};

/**
 * Make a leaf's raw error table monotone non-decreasing across its levels. The
 * engine ranks upgrades by error reduction per splat *across* leaves, so a coarser
 * level must never advertise less error than the finer one it stands in for.
 *
 * @param raw - Raw error per LOD.
 * @param levels - The leaf's levels, ascending.
 * @returns The clamped table, zero for levels absent from the leaf.
 */
const monotoneErrors = (raw: number[], levels: number[]): number[] => {
    const errors = new Array(raw.length).fill(0);
    let previous = 0;
    for (const lod of levels) previous = errors[lod] = Math.max(raw[lod], previous);
    return errors;
};

/**
 * Fill in every leaf's error table: gather its levels, then measure them with the
 * {@link ErrorRenderer} — small leaves batched into atlases, the rest alone.
 *
 * Comparing composited images rather than the gaussians themselves is what makes
 * the number track what the viewer will show: a merge of overlapping splats into
 * one that paints the same pixels costs nothing, while thinning, blur and colour
 * drift all register in proportion to how visible they are at the judged size
 * (see {@link leafViewSize}). Any per-splat comparison saturates once a splat no
 * longer overlaps its nearest match, which compresses a 128x decimation and a 2x
 * one into the same narrow band and leaves the engine's allocator ranking on
 * splat count alone.
 *
 * Each leaf is self-contained, so the pass needs no scene-wide search structure
 * and its cost is a fixed price per render times the number of leaves, levels
 * and views, with batching dividing the render count for small leaves.
 *
 * Leaves are visited in partition order so each atlas holds neighbours and the
 * source gathers stay local. An atlas is flushed when it fills, and every
 * partially filled one at the end.
 *
 * @param renderer - The renderer for the pass.
 * @param source - The PLY-space scene source.
 * @param pool - Pool for the per-batch read buffers.
 * @param slim - Resident position columns.
 * @param cum - Flat base of each structural LOD.
 * @param numLods - Number of structural LODs in the source.
 * @param jobs - The leaves, in partition order.
 */
const runErrorPass = async (
    renderer: ErrorRenderer, source: ChunkSource, pool: ChunkDataPool, slim: SlimColumns,
    cum: number[], numLods: number, jobs: LeafJob[]
): Promise<void> => {
    const bar = logger.bar('lod errors', jobs.length);
    const atlases = new Map<number, { leaf: LeafLevels; job: LeafJob }[]>();
    const atlasGaussians = new Map<number, number>();

    const assign = (job: LeafJob, levels: number[], raw: number[]) => {
        job.node.errors = monotoneErrors(raw, levels);
        bar.tick(1);
    };

    const flush = async (frame: number) => {
        const batch = atlases.get(frame);
        if (!batch?.length) return;
        atlases.set(frame, []);
        atlasGaussians.set(frame, 0);
        const raws = await renderer.atlasErrors(batch.map(b => b.leaf), frame, numLods);
        batch.forEach((b, i) => assign(b.job, [...b.leaf.levels.keys()].sort((x, y) => x - y), raws[i]));
    };

    try {
        for (const job of jobs) {
            const levels = [...job.bins.keys()].sort((a, b) => a - b);
            if (levels.length < 2) {
                assign(job, levels, new Array(numLods).fill(0));
                continue;
            }

            const tables = new Map<number, DataTable>();
            for (const lod of levels) {
                tables.set(lod, await gatherLeafTable(source, pool, slim, job.bins.get(lod)!, lod, cum[lod]));
            }
            const leaf: LeafLevels = { bound: job.node.bound, levels: tables };
            const reference = tables.get(levels[0])!;
            const frame = leafViewSize(leaf.bound, reference);

            if (ErrorRenderer.fitsAtlas(frame, leaf)) {
                // Batches are keyed by the frame rounded up to a power of two, so a
                // scene's leaves fall into a few well-filled batches rather than one
                // per frame size; a leaf is only ever judged at a frame at least its own.
                const key = 1 << Math.ceil(Math.log2(frame));
                const batch = atlases.get(key) ?? [];
                atlases.set(key, batch);
                batch.push({ leaf, job });
                const gaussians = (atlasGaussians.get(key) ?? 0) + reference.numRows;
                atlasGaussians.set(key, gaussians);
                if (batch.length === ErrorRenderer.atlasCapacity(key) || gaussians >= ATLAS_MAX_BATCH_GAUSSIANS) await flush(key);
            } else {
                assign(job, levels, await renderer.leafErrors(leaf, numLods));
            }
        }
        for (const frame of atlases.keys()) await flush(frame);
    } finally {
        bar.end();
    }
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
    /**
     * Supplies the GPU the per-leaf error tables are rendered on (and that SOG
     * encoding uses). Without one the manifest declares `lodErrors: false` and a
     * consumer falls back to deriving errors from splat counts.
     */
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

    // The error tables are rendered, so they need the GPU. One renderer serves the
    // whole pass; the partition below computes the tables leaf by leaf.
    const device = createDevice ? await createDevice() : null;
    const renderer = device ? new ErrorRenderer(device, mainSource.meta.shBands) : null;
    if (!renderer) {
        logger.info('No GPU device: LOD error tables are not written; the viewer will derive them from splat counts.');
    }
    const leafJobs: LeafJob[] = [];

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

        // Bound over the leaf's full structural LOD data; the error table is filled
        // in by the pass after the partition, which batches leaves.
        const bound = await calcBound(mainSource, pool, bins, cum, n => chunkingBar.tick(n));
        const leaf: MetaNode = { bound, lods };
        if (renderer) leafJobs.push({ node: leaf, bins });
        return leaf;
    };

    let tree: MetaNode;
    try {
        tree = await build(bTree.root);
    } finally {
        chunkingBar.end();
    }

    if (renderer) {
        const errorStart = performance.now();
        try {
            await runErrorPass(renderer, mainSource, pool, slim, cum, numLods, leafJobs);
        } finally {
            renderer.destroy();
        }
        logger.info(`LOD error pass: ${((performance.now() - errorStart) / 1000).toFixed(1)}s over ${leafJobs.length} leaves`);
    }

    const trimErrors = (node: MetaNode): void => {
        if (node.errors) node.errors.length = lodLevels;
        for (const child of node.children ?? []) trimErrors(child);
    };
    trimErrors(tree);

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
            generator: `splat-transform v${version}`,
            chunkGaussians: binSize,
            chunkExtent: binDim
        },
        count: counts.reduce((acc, curr) => acc + curr, 0),
        counts,
        lodLevels,
        lodErrors: renderer !== null,
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
