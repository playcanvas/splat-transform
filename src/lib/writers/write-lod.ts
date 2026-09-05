import { basename, dirname, resolve } from 'pathe';
import { BoundingBox, type GraphicsDevice, Mat4, Quat, Vec3 } from 'playcanvas';

import { logWrittenFile } from './utils';
import { writeSogSource } from './write-sog.js';
import { type ChunkDataPool, type ChunkSource, type ReadRequest, createChunkDataPool } from '../chunk';
import { Column, DataTable } from '../data-table';
import { type FileSystem } from '../io/write';
import { bakeTransform, permuteSource, sortMortonColumns } from '../ops';
import { type RenderCamera, renderSplats } from '../render';
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
 * about invalid data. {@link assertFiniteColor} does the
 * same for the layer this pass does not read.
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

/**
 * Pixels across each error render. This is the metric's one calibration: it fixes
 * the on-screen size at which a level is judged, since detail finer than a pixel
 * of the render is invisible to it. 512 across a leaf corresponds to a leaf
 * spanning roughly a quarter of a 2K-wide viewport — the band in which the
 * engine's budget allocator is actually trading levels off against each other.
 * The nearest leaves are bought first whatever their error says, and the far
 * field sits at its coarsest level regardless, so accuracy elsewhere buys little.
 */
const ERROR_VIEW_SIZE = 512;

/**
 * Camera distance from the leaf centre, in bounding-sphere radii. Near-orthographic
 * without being so far that the splats' projected footprints are lost to precision.
 */
const ERROR_VIEW_DISTANCE = 4;

/** The six axis-aligned directions the leaf is viewed from. */
const ERROR_VIEW_DIRECTIONS = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];

/**
 * Transparent black: the renderer then reports residual coverage in the alpha
 * channel and colour premultiplied by it in RGB, so thinning (lost coverage) and
 * colour drift both register as a difference.
 */
const ERROR_BACKGROUND = { r: 0, g: 0, b: 0, a: 0 };

/**
 * The six views of a leaf: pinhole cameras on each axis, far enough out to frame
 * the leaf's bounding sphere with a little margin.
 *
 * @param bound - The leaf's ellipsoid AABB (the same volume the engine derives its
 * screen coverage from).
 * @returns One camera per view direction.
 */
const leafCameras = (bound: Aabb): RenderCamera[] => {
    const { min, max } = bound;
    const cx = (min[0] + max[0]) / 2;
    const cy = (min[1] + max[1]) / 2;
    const cz = (min[2] + max[2]) / 2;
    const radius = Math.max(Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / 2, 1e-3);
    const distance = ERROR_VIEW_DISTANCE * radius;
    const fovY = 2 * Math.atan(1.05 * radius / distance);

    return ERROR_VIEW_DIRECTIONS.map(([dx, dy, dz]) => ({
        projection: 'pinhole',
        position: new Vec3(cx + dx * distance, cy + dy * distance, cz + dz * distance),
        target: new Vec3(cx, cy, cz),
        // any vector not parallel to the view direction
        up: dy !== 0 ? new Vec3(0, 0, 1) : new Vec3(0, 1, 0),
        fovY,
        width: ERROR_VIEW_SIZE,
        height: ERROR_VIEW_SIZE,
        // the leaf occupies [distance - radius, distance + radius] in depth
        near: radius
    }));
};

const sumSquared = (image: Uint8Array): number => {
    let total = 0;
    for (let i = 0; i < image.length; i++) {
        const v = image[i] / 255;
        total += v * v;
    }
    return total;
};

const sumSquaredDifference = (a: Uint8Array, b: Uint8Array): number => {
    let total = 0;
    for (let i = 0; i < a.length; i++) {
        const d = (a[i] - b[i]) / 255;
        total += d * d;
    }
    return total;
};

/**
 * Per-level approximation error of a leaf, measured on rendered images.
 *
 * Each level present in the leaf is rendered from the six {@link leafCameras}
 * views and compared against the finest level present, which is the reference
 * and carries no error. A level's error is the squared RGBA difference summed
 * over every pixel of every view, divided by the reference's summed squared RGBA
 * — a relative image error that is zero for an identical level, grows without
 * bound as a level departs from the reference, and is dimensionless so leaves of
 * any physical size compare on one scale.
 *
 * Comparing composited images rather than the gaussians themselves is what makes
 * the number track what the viewer will show: a merge of overlapping splats into
 * one that paints the same pixels costs nothing, while thinning, blur and colour
 * drift all register in proportion to how visible they are at the calibrated
 * size ({@link ERROR_VIEW_SIZE}). Any per-splat comparison saturates once a
 * splat no longer overlaps its nearest match, which compresses a 128x decimation
 * and a 2x one into the same narrow band and leaves the engine's allocator
 * ranking on splat count alone.
 *
 * The cost is linear in the leaf's gaussians (each is projected once per view
 * per level) and each leaf is self-contained, so the pass scales linearly with
 * the scene and needs no scene-wide search structure.
 *
 * The table is kept monotone non-decreasing across levels. The engine ranks
 * upgrades by error reduction per splat *across* leaves, so a coarser level must
 * never advertise less error than the finer one it stands in for.
 *
 * @param device - Graphics device the renders run on.
 * @param source - The PLY-space scene source.
 * @param pool - Pool for the per-batch read buffers.
 * @param slim - Resident position columns.
 * @param bound - The leaf's ellipsoid AABB, in PLY space.
 * @param bins - The leaf's flat analysis indices, by structural LOD.
 * @param cum - Flat base of each structural LOD.
 * @param numLods - Number of structural LODs in the source.
 * @returns One error per structural LOD; zero for the reference and for levels
 * absent from the leaf.
 */
const calcErrors = async (
    device: GraphicsDevice, source: ChunkSource, pool: ChunkDataPool, slim: SlimColumns,
    bound: Aabb, bins: Map<number, Uint32Array>, cum: number[], numLods: number
): Promise<number[]> => {
    const errors = new Array(numLods).fill(0);
    const levels = [...bins.keys()].sort((a, b) => a - b);
    if (levels.length < 2) return errors;

    const cameras = leafCameras(bound);

    const referenceLod = levels[0];
    const referenceTable = await gatherLeafTable(source, pool, slim, bins.get(referenceLod)!, referenceLod, cum[referenceLod]);
    const referenceImages: Uint8Array[] = [];
    let energy = 0;
    for (const camera of cameras) {
        const image = await renderSplats(device, referenceTable, camera, ERROR_BACKGROUND, { quiet: true });
        energy += sumSquared(image);
        referenceImages.push(image);
    }

    let previous = 0;
    for (let i = 1; i < levels.length; i++) {
        const lod = levels[i];
        const table = await gatherLeafTable(source, pool, slim, bins.get(lod)!, lod, cum[lod]);
        let difference = 0;
        for (let v = 0; v < cameras.length; v++) {
            const image = await renderSplats(device, table, cameras[v], ERROR_BACKGROUND, { quiet: true });
            difference += sumSquaredDifference(referenceImages[v], image);
        }
        previous = errors[lod] = Math.max(energy > 0 ? difference / energy : 0, previous);
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

    // The error tables are rendered, so they need the GPU. Acquired up front: the
    // partition pass below computes them leaf by leaf.
    const device = createDevice ? await createDevice() : null;
    if (!device) {
        logger.info('No GPU device: LOD error tables are not written; the viewer will derive them from splat counts.');
    }

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
        if (!device) return { bound, lods };

        const errors = await calcErrors(device, mainSource, pool, slim, bound, bins, cum, numLods);
        return { bound, lods, errors };
    };

    let tree: MetaNode;
    try {
        tree = await build(bTree.root);
    } finally {
        chunkingBar.end();
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
        lodErrors: device !== null,
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
