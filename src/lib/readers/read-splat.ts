import { fileChunkSource, readExact, sortGatherSlots, gatherRuns } from './reader-utils';
import {
    type ReadRequest,
    type ChunkSource,
    type ChunkSourceMetadata,
    type ChunkDataPool,
    type LayerLayout,
    type ChunkLayer,
    POSITION_STRIDE,
    GEOMETRIC_STRIDE,
    colorStride,
    positionFields,
    geometricFields,
    colorFields
} from '../chunk';
import { Column, DataTable } from '../data-table';
import { type ReadSource } from '../io/read';
import { Transform } from '../utils';

const SH_C0 = 0.28209479177387814;

// Each Antimatter15 .splat record is 32 bytes: pos(3×f32) scale(3×f32)
// rgba(4×u8) rot(4×u8).
const BYTES_PER_SPLAT = 32;

// Decode one .splat record (read from `dv`/`u8` at byte offset `o`) into the
// canonical layer buffers at gaussian slot `i`. Shared by the lazy reader and
// the eager oracle so they stay byte-identical.
const decodePosition = (dv: DataView, o: number, pos: Float32Array, i: number): void => {
    const p = i * 3;
    pos[p] = dv.getFloat32(o, true);
    pos[p + 1] = dv.getFloat32(o + 4, true);
    pos[p + 2] = dv.getFloat32(o + 8, true);
};

const decodeGeometric = (dv: DataView, u8: Uint8Array, o: number, geo: Float32Array, i: number): void => {
    const g = i * 8;
    // rotation quaternion: u8 [0,255] -> [-1,1], normalized (identity if degenerate)
    const r0 = (u8[o + 28] / 255) * 2 - 1;
    const r1 = (u8[o + 29] / 255) * 2 - 1;
    const r2 = (u8[o + 30] / 255) * 2 - 1;
    const r3 = (u8[o + 31] / 255) * 2 - 1;
    const len = Math.sqrt(r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3);
    if (len > 0) {
        geo[g] = r0 / len;
        geo[g + 1] = r1 / len;
        geo[g + 2] = r2 / len;
        geo[g + 3] = r3 / len;
    } else {
        geo[g] = 1; geo[g + 1] = 0; geo[g + 2] = 0; geo[g + 3] = 0;
    }
    // scale: linear in .splat -> log space
    geo[g + 4] = Math.log(dv.getFloat32(o + 12, true));
    geo[g + 5] = Math.log(dv.getFloat32(o + 16, true));
    geo[g + 6] = Math.log(dv.getFloat32(o + 20, true));
    // opacity: u8 -> inverse sigmoid (logit), clamped off the asymptotes
    const eps = 1e-6;
    const a = Math.max(eps, Math.min(1 - eps, u8[o + 27] / 255));
    geo[g + 7] = Math.log(a / (1 - a));
};

const decodeColor = (u8: Uint8Array, o: number, col: Float32Array, i: number): void => {
    const c = i * 3;
    col[c] = (u8[o + 24] / 255 - 0.5) / SH_C0;
    col[c + 1] = (u8[o + 25] / 255 - 0.5) / SH_C0;
    col[c + 2] = (u8[o + 26] / 255 - 0.5) / SH_C0;
};

/**
 * Read an Antimatter15 `.splat` file as a lazy {@link ChunkSource}.
 *
 * The format is a flat array of 32-byte records (no header), so reads seek
 * directly to `chunkIndex * chunkSize * 32` and decode only the requested
 * layers — the whole scene is never resident. `.splat` carries no SH, so the
 * source exposes `position`, `geometric` and a DC-only `color` layer.
 *
 * Falls back to a single resident read when the source size is unknown (e.g. a
 * URL with no content-length), so non-seekable sources still work.
 *
 * @param source - The read source for the `.splat` file.
 * @param pool - Pool whose `chunkSize` sets the gaussians-per-chunk granularity.
 * @returns A lazy source over the file's gaussians (in `Transform.PLY` space).
 * @ignore
 */
const readSplat = async (source: ReadSource, pool: ChunkDataPool): Promise<ChunkSource> => {
    // Size is needed up front (gaussian count). If unknown, load resident once.
    let resident: Uint8Array | null = null;
    let size = source.size;
    if (size === undefined) {
        resident = await source.read().readAll();
        size = resident.length;
    }

    if (size % BYTES_PER_SPLAT !== 0) {
        throw new Error('Invalid .splat file: file size is not a multiple of 32 bytes');
    }
    const numGaussians = size / BYTES_PER_SPLAT;
    if (numGaussians === 0) {
        throw new Error('Invalid .splat file: file is empty');
    }

    const chunkSize = pool.chunkSize;
    const numChunks = Math.max(1, Math.ceil(numGaussians / chunkSize));

    const availableLayers = new Set<ChunkLayer>(['position', 'geometric', 'color']);
    const layouts: Partial<Record<ChunkLayer, LayerLayout>> = {
        position: { stride: POSITION_STRIDE, fields: positionFields() },
        geometric: { stride: GEOMETRIC_STRIDE, fields: geometricFields() },
        color: { stride: colorStride(0), fields: colorFields(0) }
    };

    const meta: ChunkSourceMetadata = {
        numGaussians,
        numLods: 1,
        lodCounts: [numGaussians],
        chunkSize,
        numChunks: [numChunks],
        shBands: 0,
        extraColumns: [],
        transform: Transform.PLY.clone(),
        availableLayers,
        layouts
    };

    // Reusable scratch for the contiguous-chunk and gather range reads (reads are
    // sequential per the ChunkSource contract, so single shared buffers suffice).
    let scratch: Uint8Array | null = null;
    let gatherScratch: Uint8Array | null = null;

    const read = async (request: ReadRequest): Promise<void> => {
        if ((request.lod ?? 0) !== 0) {
            throw new Error('readSplat: only lod 0 is supported');
        }
        const pos = request.position ? new Float32Array(request.position.data) : null;
        const geo = request.geometric ? new Float32Array(request.geometric.data) : null;
        const col = request.color ? new Float32Array(request.color.data) : null;
        if (!pos && !geo && !col) return;

        // Decode `n` records held in `buf` (record `t` at byte `srcRec(t)*32`)
        // into the output slots given by `dstRow`. Shared by both the
        // contiguous-chunk and the scatter-gather paths so they stay identical.
        const decode = (buf: Uint8Array, n: number, dstRow: (t: number) => number, srcRec: (t: number) => number = t => t): void => {
            const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
            for (let t = 0; t < n; t++) {
                const o = srcRec(t) * BYTES_PER_SPLAT;
                const d = dstRow(t);
                if (pos) decodePosition(dv, o, pos, d);
                if (geo) decodeGeometric(dv, buf, o, geo, d);
                if (col) decodeColor(buf, o, col, d);
            }
        };

        if ('indices' in request) {
            // Gather: order output slots by source row so reads run forward,
            // coalescing nearby records into one bounded range read.
            const { indices, indexOffset, count } = request;
            if (count <= 0) return;
            const slot = sortGatherSlots(count, s => indices[indexOffset + s]);
            const rowAt = (t: number) => indices[indexOffset + slot[t]];

            for (const run of gatherRuns(count, t => rowAt(t) * BYTES_PER_SPLAT, BYTES_PER_SPLAT)) {
                const firstRow = run.firstByte / BYTES_PER_SPLAT;
                const byteLen = run.recordCount * BYTES_PER_SPLAT;

                let buf: Uint8Array;
                if (resident) {
                    buf = resident.subarray(run.firstByte, run.firstByte + byteLen);
                } else {
                    if (!gatherScratch || gatherScratch.length < byteLen) {
                        gatherScratch = new Uint8Array(byteLen);
                    }
                    buf = gatherScratch.subarray(0, byteLen);
                    if (await readExact(source.read(run.firstByte, run.firstByte + byteLen), buf, 0, byteLen) !== byteLen) {
                        throw new Error(`readSplat: short gather read at row ${firstRow}`);
                    }
                }
                const { j0 } = run;
                decode(buf, run.j1 - j0, t => slot[j0 + t], t => rowAt(j0 + t) - firstRow);
            }
            return;
        }

        // Contiguous chunk: rows [start, start + count).
        const start = request.chunkIndex * chunkSize;
        const count = Math.min(chunkSize, numGaussians - start);
        if (count <= 0) {
            throw new Error(`readSplat: chunkIndex ${request.chunkIndex} out of range`);
        }
        const byteStart = start * BYTES_PER_SPLAT;
        const byteLen = count * BYTES_PER_SPLAT;
        let buf: Uint8Array;
        if (resident) {
            buf = resident.subarray(byteStart, byteStart + byteLen);
        } else {
            scratch ??= new Uint8Array(chunkSize * BYTES_PER_SPLAT);
            buf = scratch.subarray(0, byteLen);
            if (await readExact(source.read(byteStart, byteStart + byteLen), buf, 0, byteLen) !== byteLen) {
                throw new Error(`readSplat: short read for chunk ${request.chunkIndex}`);
            }
        }
        decode(buf, count, t => t);
    };

    return fileChunkSource(source, meta, read);
};

/**
 * Eager `.splat` -> `DataTable` decode. Retained as the independent oracle for
 * the lazy {@link readSplat}'s tests; not used by the pipeline.
 *
 * @param source - The read source for the `.splat` file.
 * @returns A DataTable of the splat data (in `Transform.PLY` space).
 * @ignore
 */
const decodeSplatToDataTable = async (source: ReadSource): Promise<DataTable> => {
    const fileBuffer = await source.read().readAll();
    const fileSize = fileBuffer.length;
    if (fileSize % BYTES_PER_SPLAT !== 0) {
        throw new Error('Invalid .splat file: file size is not a multiple of 32 bytes');
    }
    const numSplats = fileSize / BYTES_PER_SPLAT;
    if (numSplats === 0) {
        throw new Error('Invalid .splat file: file is empty');
    }

    const x = new Float32Array(numSplats), y = new Float32Array(numSplats), z = new Float32Array(numSplats);
    const s0 = new Float32Array(numSplats), s1 = new Float32Array(numSplats), s2 = new Float32Array(numSplats);
    const c0 = new Float32Array(numSplats), c1 = new Float32Array(numSplats), c2 = new Float32Array(numSplats);
    const op = new Float32Array(numSplats);
    const r0 = new Float32Array(numSplats), r1 = new Float32Array(numSplats), r2 = new Float32Array(numSplats), r3 = new Float32Array(numSplats);

    const dv = new DataView(fileBuffer.buffer, fileBuffer.byteOffset, fileBuffer.byteLength);
    const pos = new Float32Array(3);
    const geo = new Float32Array(8);
    const col = new Float32Array(3);
    for (let i = 0; i < numSplats; i++) {
        const o = i * BYTES_PER_SPLAT;
        decodePosition(dv, o, pos, 0);
        decodeGeometric(dv, fileBuffer, o, geo, 0);
        decodeColor(fileBuffer, o, col, 0);
        x[i] = pos[0]; y[i] = pos[1]; z[i] = pos[2];
        r0[i] = geo[0]; r1[i] = geo[1]; r2[i] = geo[2]; r3[i] = geo[3];
        s0[i] = geo[4]; s1[i] = geo[5]; s2[i] = geo[6]; op[i] = geo[7];
        c0[i] = col[0]; c1[i] = col[1]; c2[i] = col[2];
    }

    return new DataTable([
        new Column('x', x), new Column('y', y), new Column('z', z),
        new Column('scale_0', s0), new Column('scale_1', s1), new Column('scale_2', s2),
        new Column('f_dc_0', c0), new Column('f_dc_1', c1), new Column('f_dc_2', c2),
        new Column('opacity', op),
        new Column('rot_0', r0), new Column('rot_1', r1), new Column('rot_2', r2), new Column('rot_3', r3)
    ], Transform.PLY);
};

export { readSplat, decodeSplatToDataTable };
