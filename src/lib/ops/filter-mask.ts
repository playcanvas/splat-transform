import { Vec3 } from 'playcanvas';

import {
    type ChunkData,
    type ChunkDataPool,
    type ChunkLayer,
    type ChunkSource,
    type ChunkSourceMetadata,
    SH_REST_COUNTS
} from '../source';
import { Transform } from '../utils';
import { inverseTransforms, isTransformColumn, rawColumnMap } from '../value-transforms';
import { bakeTransform } from './bake-transform';

/**
 * Row-selection passes over a {@link ChunkSource}: each scans the source (LOD 0)
 * chunk-by-chunk and returns an **ascending** `Uint32Array` of the gaussian
 * indices that survive, ready to hand to `filterSource`. These mirror the
 * per-row predicates of `processDataTable`'s filter actions exactly, so the
 * selection (and therefore the filtered output) is identical.
 */

const LAYERS: ChunkLayer[] = ['position', 'geometric', 'color', 'other'];

// Mutable form of a read request, built layer-by-layer before passing to a
// source's `read` (whose `ChunkReadRequest` fields are readonly).
type MutableReadRequest = { chunkIndex: number; lod: number } & { [L in ChunkLayer]?: ChunkData };

type Comparator = 'lt' | 'lte' | 'gt' | 'gte' | 'eq' | 'neq';

const comparators: Record<Comparator, (a: number, b: number) => boolean> = {
    lt: (a, b) => a < b,
    lte: (a, b) => a <= b,
    gt: (a, b) => a > b,
    gte: (a, b) => a >= b,
    eq: (a, b) => a === b,
    neq: (a, b) => a !== b
};

/**
 * Stream `src` (LOD 0) chunk-by-chunk, reading `layers`, and collect the global
 * indices of rows for which the per-chunk predicate returns true. The predicate
 * is built once per chunk from the raw layer `ArrayBuffer`s (so callers can view
 * them as f32/u32 as needed) and the chunk's row `count`.
 * @param src - The source to scan (LOD 0).
 * @param pool - Pool for the temporary layer read buffers.
 * @param layers - Layers to read for each chunk.
 * @param makePredicate - Builds the per-row keep test from a chunk's layer buffers and row count.
 * @returns Ascending global indices of the kept rows.
 */
const selectRows = async (
    src: ChunkSource,
    pool: ChunkDataPool,
    layers: ChunkLayer[],
    makePredicate: (buffers: Partial<Record<ChunkLayer, ArrayBuffer>>, count: number) => (r: number) => boolean
): Promise<Uint32Array> => {
    const { meta } = src;
    const N = meta.numGaussians;
    const S = meta.chunkSize;
    const numChunks = meta.numChunks[0] ?? 0;

    const kept = new Uint32Array(N);
    let k = 0;

    for (let c = 0; c < numChunks; c++) {
        const base = c * S;
        const count = Math.min(S, N - base);

        const acquired: ChunkData[] = [];
        const buffers: Partial<Record<ChunkLayer, ArrayBuffer>> = {};
        const req: MutableReadRequest = { chunkIndex: c, lod: 0 };
        for (const layer of layers) {
            const cd = pool.acquire(layer, meta.layouts[layer]!, count);
            req[layer] = cd;
            buffers[layer] = cd.data;
            acquired.push(cd);
        }
        await src.read(req);

        const predicate = makePredicate(buffers, count);
        for (let r = 0; r < count; r++) {
            if (predicate(r)) kept[k++] = base + r;
        }

        for (const cd of acquired) cd.release();
    }

    return kept.subarray(0, k);
};

/**
 * Remove gaussians containing NaN or Infinity. Mirrors `processDataTable`'s
 * `filterNaN`: any non-finite value drops the row, except `opacity` may be
 * `+Infinity` and `scale_*` may be `-Infinity`.
 * @param src - The source to filter (LOD 0).
 * @param pool - Pool for the temporary read buffers.
 * @returns Ascending indices of the surviving gaussians.
 */
const filterNaNRows = (src: ChunkSource, pool: ChunkDataPool): Promise<Uint32Array> => {
    const { meta } = src;
    const layers = LAYERS.filter(l => meta.availableLayers.has(l));
    const colStride32 = 3 + SH_REST_COUNTS[meta.shBands];
    const extras = meta.extraColumns;

    return selectRows(src, pool, layers, (buffers) => {
        const posF = buffers.position ? new Float32Array(buffers.position) : null;
        const geoF = buffers.geometric ? new Float32Array(buffers.geometric) : null;
        const colF = buffers.color ? new Float32Array(buffers.color) : null;
        const othF = buffers.other ? new Float32Array(buffers.other) : null;
        const extraCount = extras.length;

        return (r: number): boolean => {
            if (posF) {
                const o = r * 3;
                if (!isFinite(posF[o]) || !isFinite(posF[o + 1]) || !isFinite(posF[o + 2])) return false;
            }
            if (geoF) {
                const o = r * 8;
                if (!isFinite(geoF[o]) || !isFinite(geoF[o + 1]) || !isFinite(geoF[o + 2]) || !isFinite(geoF[o + 3])) {
                    return false; // rotation
                }
                for (let e = 4; e <= 6; e++) {
                    const v = geoF[o + e];
                    if (!isFinite(v) && v !== -Infinity) return false; // scale: -Inf ok
                }
                const op = geoF[o + 7];
                if (!isFinite(op) && op !== Infinity) return false; // opacity: +Inf ok
            }
            if (colF) {
                const o = r * colStride32;
                for (let e = 0; e < colStride32; e++) {
                    if (!isFinite(colF[o + e])) return false;
                }
            }
            if (othF && extraCount > 0) {
                const o = r * extraCount;
                for (let e = 0; e < extraCount; e++) {
                    // uint columns are always finite; only floats can hold NaN/Inf.
                    if (extras[e].type === 'float32' && !isFinite(othF[o + e])) return false;
                }
            }
            return true;
        };
    });
};

type ColumnLoc = { layer: ChunkLayer; elem: number; type: 'float32' | 'uint32' };

const POSITION_ELEMS: Record<string, number> = { x: 0, y: 1, z: 2 };
const GEOMETRIC_ELEMS: Record<string, number> = {
    rot_0: 0, rot_1: 1, rot_2: 2, rot_3: 3, scale_0: 4, scale_1: 5, scale_2: 6, opacity: 7
};
const DC_ELEMS: Record<string, number> = { f_dc_0: 0, f_dc_1: 1, f_dc_2: 2 };

// Resolve a legacy column name to its (layer, element-within-record, type), or
// null if the source can't serve it. f_rest beyond the source's bands is absent.
const columnLocation = (name: string, meta: ChunkSourceMetadata): ColumnLoc | null => {
    if (name in POSITION_ELEMS) return { layer: 'position', elem: POSITION_ELEMS[name], type: 'float32' };
    if (name in GEOMETRIC_ELEMS) return { layer: 'geometric', elem: GEOMETRIC_ELEMS[name], type: 'float32' };
    if (name in DC_ELEMS) return { layer: 'color', elem: DC_ELEMS[name], type: 'float32' };
    const rest = /^f_rest_(\d+)$/.exec(name);
    if (rest) {
        const k = parseInt(rest[1], 10);
        return k < SH_REST_COUNTS[meta.shBands] ? { layer: 'color', elem: 3 + k, type: 'float32' } : null;
    }
    const ei = meta.extraColumns.findIndex(e => e.name === name);
    if (ei >= 0) return { layer: 'other', elem: ei, type: meta.extraColumns[ei].type };
    return null;
};

type ByValueParams = { columnName: string; comparator: string; value: number };

/**
 * Keep gaussians matching a column comparison. Mirrors `processDataTable`'s
 * `filterByValue`: `*_raw` compares the stored value directly; otherwise an
 * `opacity`/`scale_*`/`f_dc_*` value is converted from user-friendly to raw
 * space first. If the source has a pending transform and the column is affected
 * by it, the column is baked (via {@link bakeTransform}) before comparison.
 * @param src - The source to filter (LOD 0).
 * @param pool - Pool for the temporary read buffers.
 * @param params - Column name, comparator and comparison value.
 * @returns Ascending indices of the surviving gaussians.
 */
const filterByValueRows = (src: ChunkSource, pool: ChunkDataPool, params: ByValueParams): Promise<Uint32Array> => {
    const { meta } = src;
    let columnName = params.columnName;
    let value = params.value;

    if (rawColumnMap[columnName]) {
        columnName = rawColumnMap[columnName];
    } else if (inverseTransforms[columnName]) {
        if (columnName === 'opacity' && (value <= 0 || value >= 1)) {
            throw new Error(`filterByValue: opacity value must be between 0 and 1 (exclusive), got ${value}`);
        }
        value = inverseTransforms[columnName](value);
    }

    const loc = columnLocation(columnName, meta);
    if (!loc || !meta.availableLayers.has(loc.layer)) {
        throw new Error(`filterByValue: column '${columnName}' not found`);
    }

    const compare = comparators[params.comparator as Comparator];
    if (!compare) {
        throw new Error(`filterByValue: unknown comparator '${params.comparator}', expected one of: ${Object.keys(comparators).join(', ')}`);
    }

    // Transform-affected columns must be compared in baked (world) space, exactly
    // as the DataTable path bakes the whole table to identity before comparing.
    const needBake = !meta.transform.isIdentity() && isTransformColumn(columnName);
    const readSrc = needBake ? bakeTransform(src, Transform.IDENTITY) : src;

    const stride32 = meta.layouts[loc.layer]!.stride >>> 2;

    return selectRows(readSrc, pool, [loc.layer], (buffers) => {
        const buf = buffers[loc.layer]!;
        if (loc.type === 'float32') {
            const f = new Float32Array(buf);
            return (r: number) => compare(f[r * stride32 + loc.elem], value);
        }
        const u = new Uint32Array(buf);
        return (r: number) => compare(u[r * stride32 + loc.elem], value);
    });
};

type BoxParams = { min: Vec3; max: Vec3 };

/**
 * Keep gaussians inside an axis-aligned box. Mirrors `processDataTable`'s
 * `filterBox`: with a pending transform the box is mapped into the source's raw
 * space (cheaper than baking every position) and raw positions are tested.
 * @param src - The source to filter (LOD 0).
 * @param pool - Pool for the temporary read buffers.
 * @param params - The box min/max corners (world space).
 * @returns Ascending indices of the surviving gaussians.
 */
const filterBoxRows = (src: ChunkSource, pool: ChunkDataPool, params: BoxParams): Promise<Uint32Array> => {
    const { min, max } = params;
    const transform = src.meta.transform;

    if (transform.isIdentity()) {
        return selectRows(src, pool, ['position'], (buffers) => {
            const p = new Float32Array(buffers.position!);
            return (r: number) => {
                const o = r * 3;
                const x = p[o], y = p[o + 1], z = p[o + 2];
                return x >= min.x && x <= max.x &&
                       y >= min.y && y <= max.y &&
                       z >= min.z && z <= max.z;
            };
        });
    }

    const { translation, scale } = transform;
    if (scale === 0) {
        throw new Error('Cannot apply filterBox with scale 0');
    }
    const invRot = transform.rotation.clone().invert();
    const axes = [new Vec3(1, 0, 0), new Vec3(0, 1, 0), new Vec3(0, 0, 1)];
    const rawAxes = axes.map((a) => {
        const r = new Vec3();
        invRot.transformVector(a, r);
        return r;
    });

    const minArr = [min.x, min.y, min.z];
    const maxArr = [max.x, max.y, max.z];
    const rawMin = new Array(3);
    const rawMax = new Array(3);
    for (let j = 0; j < 3; j++) {
        const dot = axes[j].dot(translation);
        rawMin[j] = (minArr[j] - dot) / scale;
        rawMax[j] = (maxArr[j] - dot) / scale;
    }

    return selectRows(src, pool, ['position'], (buffers) => {
        const p = new Float32Array(buffers.position!);
        return (r: number) => {
            const o = r * 3;
            const x = p[o], y = p[o + 1], z = p[o + 2];
            for (let j = 0; j < 3; j++) {
                const proj = rawAxes[j].x * x + rawAxes[j].y * y + rawAxes[j].z * z;
                if (proj < rawMin[j] || proj > rawMax[j]) return false;
            }
            return true;
        };
    });
};

type SphereParams = { center: Vec3; radius: number };

/**
 * Keep gaussians inside a sphere. Mirrors `processDataTable`'s `filterSphere`:
 * with a pending transform the center/radius are mapped into the source's raw
 * space and raw positions are tested.
 * @param src - The source to filter (LOD 0).
 * @param pool - Pool for the temporary read buffers.
 * @param params - The sphere center (world space) and radius.
 * @returns Ascending indices of the surviving gaussians.
 */
const filterSphereRows = (src: ChunkSource, pool: ChunkDataPool, params: SphereParams): Promise<Uint32Array> => {
    const transform = src.meta.transform;
    const rawCenter = params.center.clone();
    let rawRadius = params.radius;
    if (!transform.isIdentity()) {
        transform.clone().invert().transformPoint(rawCenter, rawCenter);
        rawRadius /= transform.scale;
    }
    const radiusSq = rawRadius * rawRadius;
    const cx = rawCenter.x, cy = rawCenter.y, cz = rawCenter.z;

    return selectRows(src, pool, ['position'], (buffers) => {
        const p = new Float32Array(buffers.position!);
        return (r: number) => {
            const o = r * 3;
            const dx = p[o] - cx, dy = p[o + 1] - cy, dz = p[o + 2] - cz;
            return dx * dx + dy * dy + dz * dz < radiusSq;
        };
    });
};

export { filterNaNRows, filterByValueRows, filterBoxRows, filterSphereRows };
