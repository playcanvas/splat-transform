import {
    type ChunkData,
    type ChunkLayer,
    type ChunkSource,
    type ChunkDataPool,
    SH_REST_COUNTS
} from '../chunk';
import { type FileSystem } from '../io/write';
import { bakeTransform } from '../ops';
import { Transform } from '../utils';

const GEOMETRIC_COLS = ['rot_0', 'rot_1', 'rot_2', 'rot_3', 'scale_0', 'scale_1', 'scale_2', 'opacity'];

const plyType = (uint: boolean): string => (uint ? 'uint' : 'float');

type WritePlyStreamingOptions = {
    filename: string;
};

// One output column: which layer + byte offset to read it from, and whether
// it's a u32 (an `other` extra) rather than f32.
type OutColumn = { name: string; layer: ChunkLayer; layerByteOffset: number; uint: boolean };

// A "run": a maximal contiguous span of output columns fed by a single layer.
// Interleaving each run is a straight per-row block copy (see writePlyStreaming).
type LayerRun = { layer: ChunkLayer; dstStart: number; words: number; srcStrideWords: number };

/**
 * Stream a {@link ChunkSource} out to a binary little-endian PLY file.
 *
 * The source's pending `meta.transform` is baked into `Transform.PLY` space via
 * the shared {@link bakeTransform} op (the streaming analog of `convertToSpace`)
 * — this writer contains no transform logic of its own. It then reads the baked
 * source chunk-by-chunk (reusing pooled buffers) and writes interleaved records
 * in the canonical layer order (position, geometric, color, then `other`
 * extras). Peak working set is one set of layer buffers, independent of scene
 * size.
 *
 * @param source - The source to write.
 * @param pool - Chunk-data pool for the temporary read buffers; its `chunkSize` must match the source's.
 * @param options - Output options (filename).
 * @param fs - File system to write through.
 */
const writePlyStreaming = async (
    source: ChunkSource,
    pool: ChunkDataPool,
    options: WritePlyStreamingOptions,
    fs: FileSystem
): Promise<void> => {
    // Bake into PLY space up front; from here the writer is transform-agnostic.
    const baked = bakeTransform(source, Transform.PLY);
    const { meta } = baked;
    const N = meta.numGaussians;
    const layers = meta.availableLayers;

    // Build the output column list in canonical layer order.
    const columns: OutColumn[] = [];
    if (layers.has('position')) {
        columns.push(
            { name: 'x', layer: 'position', layerByteOffset: 0, uint: false },
            { name: 'y', layer: 'position', layerByteOffset: 4, uint: false },
            { name: 'z', layer: 'position', layerByteOffset: 8, uint: false }
        );
    }
    if (layers.has('geometric')) {
        GEOMETRIC_COLS.forEach((name, i) => {
            columns.push({ name, layer: 'geometric', layerByteOffset: i * 4, uint: false });
        });
    }
    if (layers.has('color')) {
        const names = ['f_dc_0', 'f_dc_1', 'f_dc_2', ...Array.from({ length: SH_REST_COUNTS[meta.shBands] }, (_, k) => `f_rest_${k}`)];
        names.forEach((name, i) => {
            columns.push({ name, layer: 'color', layerByteOffset: i * 4, uint: false });
        });
    }
    if (layers.has('other')) {
        meta.extraColumns.forEach((e, i) => {
            columns.push({ name: e.name, layer: 'other', layerByteOffset: i * 4, uint: e.type === 'uint32' });
        });
    }

    const recordStride = columns.length * 4;
    const recordF = recordStride >> 2; // 32-bit words per output record

    // Re-interleave plan. The interleave step is the writer hot path; instead of
    // scattering column-by-column (a per-element type branch + array-of-arrays
    // indirection) we exploit a structural invariant of the layer model:
    //
    //   Within a layer, the source field order (how the reader packed the layer
    //   buffer) is identical to the output column order (both canonical). So each
    //   layer contributes a *contiguous block* of the output record, copied per
    //   row as one straight block move. And because interleaving only relocates
    //   32-bit words — it never interprets values — we copy as Uint32 regardless
    //   of float/uint column type, so no per-element branch is needed.
    //
    // A "run" captures one such block: copy `words` words per row from the layer
    // buffer (row stride `srcStrideWords`) into the record at word offset `dstStart`.
    const runs: LayerRun[] = [];
    for (let c = 0; c < columns.length;) {
        const layer = columns[c].layer;
        let e = c;
        while (e < columns.length && columns[e].layer === layer) {
            // Invariant the block copy relies on: output word (e - c) reads source
            // word (e - c). Fail loud if column ordering ever stops matching.
            if ((columns[e].layerByteOffset >> 2) !== e - c) {
                throw new Error('writePlyStreaming: layer columns not in packed canonical order');
            }
            e++;
        }
        runs.push({ layer, dstStart: c, words: e - c, srcStrideWords: meta.layouts[layer]!.stride >> 2 });
        c = e;
    }

    // Header (matches writePly: no comments here, single vertex element).
    const header = [
        'ply',
        'format binary_little_endian 1.0',
        `element vertex ${N}`,
        ...columns.map(c => `property ${plyType(c.uint)} ${c.name}`),
        'end_header'
    ];

    const writer = await fs.createWriter(options.filename);
    await writer.write(new TextEncoder().encode(`${header.join('\n')}\n`));

    const chunkSize = meta.chunkSize;
    const numChunks = meta.numChunks[0] ?? 0;
    const outRecord = new Uint8Array(chunkSize * recordStride);
    const outU32 = new Uint32Array(outRecord.buffer);

    for (let k = 0; k < numChunks; k++) {
        const count = Math.min(chunkSize, N - k * chunkSize);

        // Acquire one buffer per available layer (pool reuses across chunks).
        const acquired: Partial<Record<ChunkLayer, ChunkData>> = {};
        for (const layer of layers) {
            acquired[layer] = pool.acquire(layer, meta.layouts[layer]!, count);
        }

        await baked.read({
            chunkIndex: k,
            position: acquired.position,
            geometric: acquired.geometric,
            color: acquired.color,
            other: acquired.other
        });

        // Re-interleave the (already-baked) chunk into the output record: one
        // contiguous block copy per layer, as raw 32-bit words. Little-endian
        // only, matching the binary PLY format and every supported runtime.
        for (const run of runs) {
            const src = new Uint32Array(acquired[run.layer]!.data);
            const ss = run.srcStrideWords;
            const ds = run.dstStart;
            const w = run.words;
            for (let i = 0; i < count; i++) {
                const s = i * ss;
                const d = i * recordF + ds;
                for (let j = 0; j < w; j++) outU32[d + j] = src[s + j];
            }
        }

        await writer.write(outRecord.subarray(0, count * recordStride));

        for (const layer of layers) acquired[layer]!.release();
    }

    await writer.close();
};

export { writePlyStreaming };
