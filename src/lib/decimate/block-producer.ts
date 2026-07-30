import { type ChunkData, type ChunkSource, type ChunkSourceMetadata, type ReadRequest } from '../chunk';

/**
 * One output chunk's destination views for the merge stream — typed windows
 * over the consumer's {@link ChunkData} buffers, sized to the chunk's exact
 * row count. Layers the consumer didn't request are absent and skipped.
 */
type DestBuffers = {
    position?: Float32Array;
    geometric?: Float32Array;
    color?: Float32Array;
    other?: Uint32Array;
};

/**
 * A single-sequential-pass {@link ChunkSource} over the merge-stream
 * generator — how decimation feeds the PLY writer (or `compact` /
 * `writePlyStreaming` for intermediate generations) without ever
 * materializing the output. The generator fills the consumer's chunk
 * buffers directly (no intermediate copy of the output bytes).
 *
 * Contract: chunk reads must arrive in order (0, 1, 2, …), each at most
 * once; gather reads are not supported. Anything else throws — decimate
 * output supports exactly one sequential pass.
 *
 * @param meta - Exact output metadata (counts are known before streaming).
 * @param produce - Factory for the merge-stream generator (invoked lazily on first read).
 * @returns The stream-once source.
 */
const createBlockProducerSource = (
    meta: ChunkSourceMetadata,
    produce: () => AsyncGenerator<number, void, DestBuffers>
): ChunkSource => {
    let generator: AsyncGenerator<number, void, DestBuffers> | null = null;
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
        if (!generator) {
            generator = produce();
            await generator.next();   // prime: run to the first dest handshake
        }

        const expected = Math.min(meta.chunkSize, meta.numGaussians - request.chunkIndex * meta.chunkSize);
        const view = <T extends Float32Array | Uint32Array>(
            cd: ChunkData | undefined,
            ctor: new (b: ArrayBuffer, o: number, n: number) => T,
            perRow: number
        ): T | undefined => (cd ? new ctor(cd.data, 0, expected * perRow) : undefined);
        const dest: DestBuffers = {
            position: view(request.position, Float32Array, 3),
            geometric: view(request.geometric, Float32Array, 8),
            color: view(request.color, Float32Array, (meta.layouts.color?.stride ?? 0) >> 2),
            other: view(request.other, Uint32Array, (meta.layouts.other?.stride ?? 0) >> 2)
        };

        const { value, done: exhausted } = await generator.next(dest);
        if (exhausted || value === undefined) {
            done = true;
            throw new Error(`decimate output ended early at chunk ${request.chunkIndex}`);
        }
        if (value !== expected) {
            throw new Error(`decimate output chunk ${request.chunkIndex}: expected ${expected} rows, produced ${value}`);
        }

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

export { createBlockProducerSource, type DestBuffers };
