import { type FileSystem } from '../io/write';
import { bakeTransform } from '../ops';
import {
    type ChunkData,
    type ChunkLayer,
    type ChunkSource,
    type ChunkDataPool,
    SH_REST_COUNTS
} from '../source';
import { Transform } from '../utils';

const GEOMETRIC_COLS = ['rot_0', 'rot_1', 'rot_2', 'rot_3', 'scale_0', 'scale_1', 'scale_2', 'opacity'];

const plyType = (uint: boolean): string => (uint ? 'uint' : 'float');

type WritePlyStreamingOptions = {
    filename: string;
};

// One output column: which layer + byte offset to read it from, and whether
// it's a u32 (an `other` extra) rather than f32.
type OutColumn = { name: string; layer: ChunkLayer; layerByteOffset: number; uint: boolean };

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
    const recordF = recordStride >> 2; // floats per output record

    // Per-column source plan as float/u32 indices, hoisted out of the hot loop.
    const colSrcFloat = columns.map(c => c.layerByteOffset >> 2);
    const colUint = columns.map(c => c.uint);

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
    const outF32 = new Float32Array(outRecord.buffer);
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

        // Interleave the (already-baked) chunk data into the output record via
        // typed-array views (not DataView). Little-endian only, which matches
        // the binary PLY format and every supported runtime.
        const nCols = columns.length;
        const colF32 = columns.map(c => new Float32Array(acquired[c.layer]!.data));
        const colU32 = columns.map(c => new Uint32Array(acquired[c.layer]!.data));
        const colStrideF = columns.map(c => acquired[c.layer]!.stride >> 2);

        for (let i = 0; i < count; i++) {
            const ob = i * recordF;
            for (let c = 0; c < nCols; c++) {
                const si = i * colStrideF[c] + colSrcFloat[c];
                if (colUint[c]) {
                    outU32[ob + c] = colU32[c][si];
                } else {
                    outF32[ob + c] = colF32[c][si];
                }
            }
        }

        await writer.write(outRecord.subarray(0, count * recordStride));

        for (const layer of layers) acquired[layer]!.release();
    }

    await writer.close();
};

export { writePlyStreaming };
