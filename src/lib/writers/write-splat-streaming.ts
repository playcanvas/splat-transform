import { type ChunkDataPool, type ChunkSource } from '../chunk';
import { type FileSystem } from '../io/write';
import { bakeTransform } from '../ops';
import { logger, Transform } from '../utils';

// SH DC -> linear color constant (SH band-0 basis coefficient).
const SH_C0 = 0.28209479177387814;

type WriteSplatStreamingOptions = {
    filename: string;
};

/**
 * Stream a {@link ChunkSource} out to the `.splat` format (antimatter15 /
 * PlayCanvas viewer layout): one fixed 32-byte record per gaussian —
 * `position.xyz` (3×f32), `exp(scale).xyz` (3×f32), colour RGBA (4×u8) and the
 * rotation quaternion (4×u8).
 *
 * The source's pending `meta.transform` is baked into `Transform.PLY` space via
 * {@link bakeTransform} up front, then the baked source is read chunk-by-chunk
 * (reusing pooled buffers). Peak working set is one chunk's worth of buffers,
 * independent of scene size.
 *
 * @param source - The source to write.
 * @param pool - Chunk-data pool for the temporary read buffers; its `chunkSize` must match the source's.
 * @param options - Output options (filename).
 * @param fs - File system to write through.
 */
const writeSplatStreaming = async (
    source: ChunkSource,
    pool: ChunkDataPool,
    options: WriteSplatStreamingOptions,
    fs: FileSystem
): Promise<void> => {
    const baked = bakeTransform(source, Transform.PLY);
    const { meta } = baked;
    const layers = meta.availableLayers;
    if (!layers.has('position') || !layers.has('geometric') || !layers.has('color')) {
        throw new Error('writeSplatStreaming: source must have position, geometric and color layers');
    }

    const N = meta.numGaussians;
    const clamp = (x: number): number => Math.max(0, Math.min(255, x));

    const writer = await fs.createWriter(options.filename);
    try {
        const chunkSize = meta.chunkSize;
        const numChunks = meta.numChunks[0] ?? 0;
        const record = new Uint8Array(chunkSize * 32);
        const dv = new DataView(record.buffer);

        const bar = logger.bar('Writing', numChunks);

        for (let k = 0; k < numChunks; k++) {
            const count = Math.min(chunkSize, N - k * chunkSize);

            const pos = pool.acquire('position', meta.layouts.position!, count);
            const geo = pool.acquire('geometric', meta.layouts.geometric!, count);
            const col = pool.acquire('color', meta.layouts.color!, count);

            await baked.read({ chunkIndex: k, position: pos, geometric: geo, color: col });

            const P = pos.field('position');   // count × 3
            const R = geo.field('rotation');   // count × 4
            const S = geo.field('scale');      // count × 3
            const O = geo.field('opacity');    // count × 1
            const C = col.field('dc');         // count × 3

            for (let i = 0; i < count; i++) {
                const off = i * 32;
                dv.setFloat32(off + 0, P[i * 3 + 0], true);
                dv.setFloat32(off + 4, P[i * 3 + 1], true);
                dv.setFloat32(off + 8, P[i * 3 + 2], true);

                dv.setFloat32(off + 12, Math.exp(S[i * 3 + 0]), true);
                dv.setFloat32(off + 16, Math.exp(S[i * 3 + 1]), true);
                dv.setFloat32(off + 20, Math.exp(S[i * 3 + 2]), true);

                dv.setUint8(off + 24, clamp((0.5 + C[i * 3 + 0] * SH_C0) * 255));
                dv.setUint8(off + 25, clamp((0.5 + C[i * 3 + 1] * SH_C0) * 255));
                dv.setUint8(off + 26, clamp((0.5 + C[i * 3 + 2] * SH_C0) * 255));
                dv.setUint8(off + 27, clamp((1 / (1 + Math.exp(-O[i]))) * 255));

                dv.setUint8(off + 28, clamp(R[i * 4 + 0] * 128 + 128));
                dv.setUint8(off + 29, clamp(R[i * 4 + 1] * 128 + 128));
                dv.setUint8(off + 30, clamp(R[i * 4 + 2] * 128 + 128));
                dv.setUint8(off + 31, clamp(R[i * 4 + 3] * 128 + 128));
            }

            await writer.write(record.subarray(0, count * 32));

            pos.release();
            geo.release();
            col.release();

            bar.tick();
        }

        bar.end();

        await writer.close();
    } catch (err) {
        try {
            await writer.abort();
        } catch {
            // already failing — swallow secondary abort errors
        }
        throw err;
    }
};

export { writeSplatStreaming };
