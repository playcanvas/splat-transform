import { Column, DataTable } from '../data-table';
import {
    createInMemoryChunkSource,
    InMemoryChunkSource,
    SH_REST_COUNTS,
    DEFAULT_CHUNK_SIZE,
    type ExtraColumn,
    type ChunkData,
    type ChunkDataPool,
    type ChunkSource,
    type ChunkLayer,
    type SHBands
} from '../source';
import { type Transform } from '../utils';

/**
 * The legacy `DataTable` <-> `ChunkSource` compatibility bridge.
 *
 * Both directions exist only for the 3.0 migration: `dataTableToChunkSource` lets
 * not-yet-ported readers upgrade their `DataTable` output to a source, and
 * `materializeToDataTable` lets not-yet-ported writers / process actions /
 * supersplat consume a source as a `DataTable`. This is the single module that
 * depends on both representations; the `source/` module stays DataTable-free.
 * When every consumer speaks `ChunkSource`, this bridge and the `DataTable`
 * class can be removed.
 */

// ---------------------------------------------------------------------------
// DataTable -> ChunkSource
// ---------------------------------------------------------------------------

/** Standard column names that map directly to the canonical layers. */
const POSITION_COLS = ['x', 'y', 'z'] as const;
const GEOMETRIC_COLS = [
    'rot_0', 'rot_1', 'rot_2', 'rot_3',
    'scale_0', 'scale_1', 'scale_2',
    'opacity'
] as const;
const COLOR_DC_COLS = ['f_dc_0', 'f_dc_1', 'f_dc_2'] as const;

const standardColumnSet = new Set<string>([
    ...POSITION_COLS,
    ...GEOMETRIC_COLS,
    ...COLOR_DC_COLS
]);

/**
 * Determine the SH band count from the highest `f_rest_*` index present.
 * @param dataTable - The table to inspect.
 * @returns The detected SH band count.
 */
const detectShBands = (dataTable: DataTable): SHBands => {
    let highestRest = -1;
    for (const c of dataTable.columns) {
        const m = c.name.match(/^f_rest_(\d+)$/);
        if (m) {
            const n = parseInt(m[1], 10);
            if (n > highestRest) highestRest = n;
        }
    }
    const count = highestRest + 1;
    if (count === 0) return 0;
    if (count === SH_REST_COUNTS[1]) return 1;
    if (count === SH_REST_COUNTS[2]) return 2;
    if (count === SH_REST_COUNTS[3]) return 3;
    throw new Error(`dataTableToChunkSource: unrecognized f_rest_* count: ${count}`);
};

const detectExtras = (dataTable: DataTable): ExtraColumn[] => {
    const extras: ExtraColumn[] = [];
    for (const c of dataTable.columns) {
        if (standardColumnSet.has(c.name)) continue;
        if (/^f_rest_\d+$/.test(c.name)) continue;
        const type: 'float32' | 'uint32' = (
            c.dataType === 'float32' || c.dataType === 'float64'
        ) ? 'float32' : 'uint32';
        extras.push({ name: c.name, type });
    }
    return extras;
};

/**
 * Split an interleaved typed array of N gaussians × `elemsPerRow` elements into
 * per-chunk `ArrayBuffer` blocks of `chunkSize` gaussians each (last may short).
 * @param interleaved - The interleaved source data (f32 or u32).
 * @param numGaussians - Total gaussian count.
 * @param elemsPerRow - Elements (4-byte) per gaussian.
 * @param chunkSize - Gaussians per chunk.
 * @returns One `ArrayBuffer` per chunk, each exactly `rows * elemsPerRow * 4` bytes.
 */
const splitToChunks = (
    interleaved: Float32Array | Uint32Array,
    numGaussians: number,
    elemsPerRow: number,
    chunkSize: number
): ArrayBuffer[] => {
    const out: ArrayBuffer[] = [];
    const isFloat = interleaved instanceof Float32Array;
    let rowsRemaining = numGaussians;
    let rowOffset = 0;
    while (rowsRemaining > 0) {
        const rows = Math.min(chunkSize, rowsRemaining);
        const slice = interleaved.subarray(
            rowOffset * elemsPerRow,
            (rowOffset + rows) * elemsPerRow
        );
        // Copy into a fresh ArrayBuffer so each chunk owns its bytes.
        const ab = new ArrayBuffer(rows * elemsPerRow * 4);
        if (isFloat) {
            new Float32Array(ab).set(slice as Float32Array);
        } else {
            new Uint32Array(ab).set(slice as Uint32Array);
        }
        out.push(ab);
        rowOffset += rows;
        rowsRemaining -= rows;
    }
    return out;
};

/**
 * Convert a legacy `DataTable` into a `ChunkSource` by repacking its
 * columnar data into the canonical per-layer interleaved layout.
 *
 * Detects SH band count from the highest `f_rest_*` index, identifies
 * non-standard columns as `other`-layer extras, and copies each gaussian's
 * fields into the appropriate per-layer buffer.
 *
 * Used during the 3.0 migration by readers that haven't yet been ported to
 * native chunked decoding — they call this at the end of their existing decode
 * to upgrade to the new return type.
 *
 * When `indices` is supplied, only those rows are repacked, in that order — a
 * direct ordered-subset gather (e.g. the LOD writer's per-unit gather), avoiding
 * a separate `DataTable.clone({ rows })` copy.
 * @param dataTable - The legacy table to convert.
 * @param chunkSize - Gaussians per chunk (default {@link DEFAULT_CHUNK_SIZE}).
 * @param indices - Optional ordered row indices to gather; output row `i` is `dataTable` row `indices[i]`.
 * @returns A CPU-resident `InMemoryChunkSource` over the repacked data.
 */
const dataTableToChunkSource = (
    dataTable: DataTable,
    chunkSize: number = DEFAULT_CHUNK_SIZE,
    indices?: Uint32Array
): InMemoryChunkSource => {
    const count = indices ? indices.length : dataTable.numRows;
    const shBands = detectShBands(dataTable);
    const numRest = SH_REST_COUNTS[shBands];
    const extras = detectExtras(dataTable);
    const transform: Transform = dataTable.transform;

    const hasPosition = POSITION_COLS.every(c => dataTable.hasColumn(c));
    const hasGeometric = GEOMETRIC_COLS.every(c => dataTable.hasColumn(c));
    const hasColor = COLOR_DC_COLS.every(c => dataTable.hasColumn(c));
    const hasOther = extras.length > 0;

    const col = (name: string): Float32Array => dataTable.getColumnByName(name)!.data as Float32Array;
    const srcRow = (i: number): number => (indices ? indices[i] : i);

    const positionChunks: ArrayBuffer[] | undefined = hasPosition ? (() => {
        const arr = new Float32Array(count * 3);
        const x = col('x'), y = col('y'), z = col('z');
        for (let i = 0; i < count; i++) {
            const s = srcRow(i);
            arr[i * 3 + 0] = x[s];
            arr[i * 3 + 1] = y[s];
            arr[i * 3 + 2] = z[s];
        }
        return splitToChunks(arr, count, 3, chunkSize);
    })() : undefined;

    const geometricChunks: ArrayBuffer[] | undefined = hasGeometric ? (() => {
        const arr = new Float32Array(count * 8);
        const r0 = col('rot_0'), r1 = col('rot_1'), r2 = col('rot_2'), r3 = col('rot_3');
        const s0 = col('scale_0'), s1 = col('scale_1'), s2 = col('scale_2');
        const op = col('opacity');
        for (let i = 0; i < count; i++) {
            const s = srcRow(i);
            const o = i * 8;
            arr[o + 0] = r0[s];
            arr[o + 1] = r1[s];
            arr[o + 2] = r2[s];
            arr[o + 3] = r3[s];
            arr[o + 4] = s0[s];
            arr[o + 5] = s1[s];
            arr[o + 6] = s2[s];
            arr[o + 7] = op[s];
        }
        return splitToChunks(arr, count, 8, chunkSize);
    })() : undefined;

    const colorChunks: ArrayBuffer[] | undefined = hasColor ? (() => {
        const elemsPerRow = 3 + numRest;
        const arr = new Float32Array(count * elemsPerRow);
        const dc0 = col('f_dc_0'), dc1 = col('f_dc_1'), dc2 = col('f_dc_2');
        const restCols: Float32Array[] = [];
        for (let r = 0; r < numRest; r++) restCols.push(col(`f_rest_${r}`));
        for (let i = 0; i < count; i++) {
            const s = srcRow(i);
            const o = i * elemsPerRow;
            arr[o + 0] = dc0[s];
            arr[o + 1] = dc1[s];
            arr[o + 2] = dc2[s];
            for (let r = 0; r < numRest; r++) arr[o + 3 + r] = restCols[r][s];
        }
        return splitToChunks(arr, count, elemsPerRow, chunkSize);
    })() : undefined;

    const otherChunks: ArrayBuffer[] | undefined = hasOther ? (() => {
        const elemsPerRow = extras.length;
        const arr = new Uint32Array(count * elemsPerRow);
        const f32View = new Float32Array(arr.buffer);
        const cols = extras.map(e => dataTable.getColumnByName(e.name)!.data);
        for (let i = 0; i < count; i++) {
            const s = srcRow(i);
            const o = i * elemsPerRow;
            for (let e = 0; e < elemsPerRow; e++) {
                if (extras[e].type === 'float32') {
                    f32View[o + e] = cols[e][s] as number;
                } else {
                    arr[o + e] = cols[e][s] as number;
                }
            }
        }
        return splitToChunks(arr, count, elemsPerRow, chunkSize);
    })() : undefined;

    return createInMemoryChunkSource({
        numGaussians: count,
        chunkSize,
        shBands,
        extraColumns: extras,
        transform,
        lodCounts: [count],
        position: positionChunks ? [positionChunks] : undefined,
        geometric: geometricChunks ? [geometricChunks] : undefined,
        color: colorChunks ? [colorChunks] : undefined,
        other: otherChunks ? [otherChunks] : undefined
    });
};

// ---------------------------------------------------------------------------
// ChunkSource -> DataTable
// ---------------------------------------------------------------------------

/**
 * Materialize a `ChunkSource` into the legacy columnar `DataTable`
 * representation.
 *
 * Each available layer is read chunk-by-chunk and scattered into the
 * appropriate named columns (`x, y, z, rot_*, scale_*, opacity, f_dc_*,
 * f_rest_*`, plus extras).
 * @param src - The source to materialize.
 * @param pool - The `ChunkData` pool used for the temporary read buffers; its `chunkSize` must be >= the source's.
 * @returns A `DataTable` holding the source's gaussians in canonical column form.
 */
const materializeToDataTable = async (
    src: ChunkSource,
    pool: ChunkDataPool
): Promise<DataTable> => {
    const { meta } = src;
    const N = meta.numGaussians;

    const x = new Float32Array(N);
    const y = new Float32Array(N);
    const z = new Float32Array(N);

    const rot0 = new Float32Array(N);
    const rot1 = new Float32Array(N);
    const rot2 = new Float32Array(N);
    const rot3 = new Float32Array(N);
    const scale0 = new Float32Array(N);
    const scale1 = new Float32Array(N);
    const scale2 = new Float32Array(N);
    const opacity = new Float32Array(N);

    const dc0 = new Float32Array(N);
    const dc1 = new Float32Array(N);
    const dc2 = new Float32Array(N);

    const numRest = SH_REST_COUNTS[meta.shBands];
    const restArrays: Float32Array[] = [];
    for (let i = 0; i < numRest; i++) restArrays.push(new Float32Array(N));

    const extraArrays = meta.extraColumns.map(e => ({
        name: e.name,
        type: e.type,
        data: e.type === 'float32' ? new Float32Array(N) : new Uint32Array(N)
    }));

    const wantsPosition = meta.availableLayers.has('position');
    const wantsGeometric = meta.availableLayers.has('geometric');
    const wantsColor = meta.availableLayers.has('color');
    const wantsOther = meta.availableLayers.has('other') && extraArrays.length > 0;

    const chunkSize = meta.chunkSize;
    const numChunks = meta.numChunks[0] ?? 0;

    for (let k = 0; k < numChunks; k++) {
        const rowStart = k * chunkSize;
        const count = Math.min(chunkSize, N - rowStart);

        const layouts = meta.layouts;
        const acquired: { layer: ChunkLayer; chunkData: ChunkData }[] = [];
        const req: {
            chunkIndex: number; lod: number;
            position?: ChunkData; geometric?: ChunkData; color?: ChunkData; other?: ChunkData;
        } = { chunkIndex: k, lod: 0 };

        if (wantsPosition) {
            const c = pool.acquire('position', layouts.position!, count);
            req.position = c;
            acquired.push({ layer: 'position', chunkData: c });
        }
        if (wantsGeometric) {
            const c = pool.acquire('geometric', layouts.geometric!, count);
            req.geometric = c;
            acquired.push({ layer: 'geometric', chunkData: c });
        }
        if (wantsColor) {
            const c = pool.acquire('color', layouts.color!, count);
            req.color = c;
            acquired.push({ layer: 'color', chunkData: c });
        }
        if (wantsOther) {
            const c = pool.acquire('other', layouts.other!, count);
            req.other = c;
            acquired.push({ layer: 'other', chunkData: c });
        }

        await src.read(req);

        for (const { layer, chunkData } of acquired) {
            const elemsPerRow = chunkData.stride >> 2;

            if (layer === 'position') {
                const f32 = new Float32Array(chunkData.data, 0, count * elemsPerRow);
                for (let i = 0; i < count; i++) {
                    const di = rowStart + i;
                    const si = i * 3;
                    x[di] = f32[si + 0];
                    y[di] = f32[si + 1];
                    z[di] = f32[si + 2];
                }
            } else if (layer === 'geometric') {
                const f32 = new Float32Array(chunkData.data, 0, count * elemsPerRow);
                for (let i = 0; i < count; i++) {
                    const di = rowStart + i;
                    const si = i * 8;
                    rot0[di] = f32[si + 0];
                    rot1[di] = f32[si + 1];
                    rot2[di] = f32[si + 2];
                    rot3[di] = f32[si + 3];
                    scale0[di] = f32[si + 4];
                    scale1[di] = f32[si + 5];
                    scale2[di] = f32[si + 6];
                    opacity[di] = f32[si + 7];
                }
            } else if (layer === 'color') {
                const f32 = new Float32Array(chunkData.data, 0, count * elemsPerRow);
                const stride = 3 + numRest;
                for (let i = 0; i < count; i++) {
                    const di = rowStart + i;
                    const si = i * stride;
                    dc0[di] = f32[si + 0];
                    dc1[di] = f32[si + 1];
                    dc2[di] = f32[si + 2];
                    for (let r = 0; r < numRest; r++) {
                        restArrays[r][di] = f32[si + 3 + r];
                    }
                }
            } else { // 'other'
                const f32 = new Float32Array(chunkData.data, 0, count * elemsPerRow);
                const u32 = new Uint32Array(chunkData.data, 0, count * elemsPerRow);
                const cols = extraArrays.length;
                for (let i = 0; i < count; i++) {
                    const di = rowStart + i;
                    for (let e = 0; e < cols; e++) {
                        if (extraArrays[e].type === 'float32') {
                            (extraArrays[e].data as Float32Array)[di] = f32[i * cols + e];
                        } else {
                            (extraArrays[e].data as Uint32Array)[di] = u32[i * cols + e];
                        }
                    }
                }
            }
        }

        for (const { chunkData } of acquired) chunkData.release();
    }

    const columns: Column[] = [];
    if (wantsPosition) {
        columns.push(new Column('x', x), new Column('y', y), new Column('z', z));
    }
    if (wantsGeometric) {
        columns.push(
            new Column('rot_0', rot0),
            new Column('rot_1', rot1),
            new Column('rot_2', rot2),
            new Column('rot_3', rot3),
            new Column('scale_0', scale0),
            new Column('scale_1', scale1),
            new Column('scale_2', scale2),
            new Column('opacity', opacity)
        );
    }
    if (wantsColor) {
        columns.push(
            new Column('f_dc_0', dc0),
            new Column('f_dc_1', dc1),
            new Column('f_dc_2', dc2)
        );
        for (let r = 0; r < numRest; r++) {
            columns.push(new Column(`f_rest_${r}`, restArrays[r]));
        }
    }
    if (wantsOther) {
        for (const e of extraArrays) {
            columns.push(new Column(e.name, e.data));
        }
    }

    return new DataTable(columns, meta.transform);
};

export { dataTableToChunkSource, materializeToDataTable };
