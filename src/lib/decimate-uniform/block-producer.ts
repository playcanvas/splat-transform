import { type ChunkData, type ChunkLayer, type ChunkSource, type ChunkSourceMetadata, type ReadRequest } from '../chunk';

/**
 * One output chunk of the merge stream. The views hold exactly `count`
 * records at the layer strides and alias the generator's rolling scratch —
 * valid only until the generator's next `next()`; the consumer's read copies
 * them out (yielding views instead of slices avoids a full extra copy of
 * every output byte).
 */
type ChunkPayload = {
    count: number;
    position: Float32Array;
    geometric: Float32Array;
    color: Float32Array;
    other?: Uint32Array;
};

/**
 * A single-sequential-pass {@link ChunkSource} over an async generator of
 * chunk payloads — how the decimation merge stream feeds the PLY writer
 * (or `compact` / `writePlyStreaming` for intermediate generations) without
 * ever materializing the output.
 *
 * Contract: chunk reads must arrive in order (0, 1, 2, …), each at most
 * once; gather reads are not supported. Anything else throws — decimate
 * output supports exactly one sequential pass.
 *
 * @param meta - Exact output metadata (counts are known before streaming).
 * @param produce - Factory for the payload generator (invoked lazily on first read).
 * @returns The stream-once source.
 */
const createBlockProducerSource = (
    meta: ChunkSourceMetadata,
    produce: () => AsyncGenerator<ChunkPayload>
): ChunkSource => {
    let generator: AsyncGenerator<ChunkPayload> | null = null;
    let nextChunk = 0;
    let done = false;

    const read = async (request: ReadRequest): Promise<void> => {
        if ('indices' in request) {
            throw new Error('decimate output supports a single sequential pass (gather reads are not available)');
        }
        if ((request.lod ?? 0) !== 0) {
            throw new Error(`decimate output has a single LOD (requested lod ${request.lod})`);
        }
        if (request.chunkIndex !== nextChunk) {
            throw new Error(
                `decimate output supports a single sequential pass (expected chunk ${nextChunk}, got ${request.chunkIndex})`
            );
        }
        if (done) {
            throw new Error('decimate output exhausted');
        }
        generator ??= produce();
        const { value, done: exhausted } = await generator.next();
        if (exhausted || !value) {
            done = true;
            throw new Error(`decimate output ended early at chunk ${request.chunkIndex}`);
        }
        const payload = value;
        const expected = Math.min(meta.chunkSize, meta.numGaussians - request.chunkIndex * meta.chunkSize);
        if (payload.count !== expected) {
            throw new Error(`decimate output chunk ${request.chunkIndex}: expected ${expected} rows, produced ${payload.count}`);
        }

        const fill = (cd: ChunkData | undefined, layer: ChunkLayer): void => {
            if (!cd) return;
            const src = payload[layer as 'position' | 'geometric' | 'color' | 'other'];
            if (!src) {
                throw new Error(`decimate output has no '${layer}' layer`);
            }
            const bytes = payload.count * cd.stride;
            new Uint8Array(cd.data, 0, bytes).set(new Uint8Array(src.buffer, src.byteOffset, bytes));
        };
        fill(request.position, 'position');
        fill(request.geometric, 'geometric');
        fill(request.color, 'color');
        fill(request.other, 'other');

        nextChunk++;
        if (nextChunk >= (meta.numChunks[0] ?? 0)) done = true;
    };

    const close = async (): Promise<void> => {
        done = true;
        await generator?.return?.(undefined as never);
        generator = null;
    };

    return { meta, read, close };
};

export { createBlockProducerSource, type ChunkPayload };
