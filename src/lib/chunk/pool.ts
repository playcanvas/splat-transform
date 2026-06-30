import { type ChunkData, ChunkDataImpl } from './data';
import { type ChunkLayer, type LayerLayout, DEFAULT_CHUNK_SIZE } from './layout';

/**
 * A pool-backed allocator for {@link ChunkData} buffers.
 *
 * The pool owns a single `chunkSize` (the gaussians-per-chunk granularity);
 * every buffer it hands out is backed by an `ArrayBuffer` of full capacity
 * (`chunkSize * stride`). A short final chunk's buffer therefore shares a pool
 * slot with full-chunk buffers of the same layer stride — the pool keys on the
 * buffer's byte size, so reuse is by capacity, not by the (possibly smaller)
 * `count`.
 *
 * No `GraphicsDevice` is required: these are CPU buffers. A consumer that needs
 * GPU-resident data uploads `chunkData.data` itself.
 *
 * Pool growth is bounded by `maxPooledBytes` (default 2 GB). On release, if
 * pooling the buffer would exceed the cap, it is dropped (left to the garbage
 * collector) instead. Call {@link ChunkDataPool.trim} to free pooled buffers
 * down to a target.
 */
interface ChunkDataPool {
    /** Gaussians-per-chunk granularity this pool allocates for. */
    readonly chunkSize: number;

    /**
     * Acquire a {@link ChunkData} buffer for the given layer/layout holding
     * `count` gaussians of valid data (`0 < count <= chunkSize`). Reuses a
     * pooled buffer of matching capacity if available; otherwise allocates a
     * new one.
     */
    acquire(layer: ChunkLayer, layout: LayerLayout, count: number): ChunkData;

    /** Total bytes currently held by callers (not in the pool). */
    readonly bytesInUse: number;

    /** Total bytes free-listed and ready to be reused. */
    readonly bytesPooled: number;

    /** Free pooled buffers until `bytesPooled <= targetBytes`. */
    trim(targetBytes: number): void;

    /** Drop all pooled buffers. Buffers in use are unaffected. */
    destroy(): void;
}

/**
 * Create a CPU `ChunkData` pool.
 *
 * @param options - Pool options.
 * @param options.chunkSize - Gaussians per chunk (default 1M). Should match
 * the `chunkSize` of any source it services.
 * @param options.maxPooledBytes - Cap on bytes held in the free list (default 2 GB).
 * @returns A new {@link ChunkDataPool}.
 */
const createChunkDataPool = (
    options?: { chunkSize?: number; maxPooledBytes?: number }
): ChunkDataPool => {
    const chunkSize = options?.chunkSize ?? DEFAULT_CHUNK_SIZE;
    const maxPooledBytes = options?.maxPooledBytes ?? 2 * 1024 * 1024 * 1024;

    if (chunkSize <= 0) {
        throw new Error(`createChunkDataPool: chunkSize must be > 0 (got ${chunkSize})`);
    }

    // Pool: capacity (bytes) -> stack of free ArrayBuffers (LIFO for hot reuse).
    const pool = new Map<number, ArrayBuffer[]>();

    let bytesInUse = 0;
    let bytesPooled = 0;

    const release = (chunkData: ChunkDataImpl): void => {
        const cap = chunkData.data.byteLength;
        bytesInUse -= cap;
        if (bytesPooled + cap > maxPooledBytes) {
            return; // pool full; drop the buffer
        }
        let stack = pool.get(cap);
        if (!stack) {
            stack = [];
            pool.set(cap, stack);
        }
        stack.push(chunkData.data);
        bytesPooled += cap;
    };

    const acquire = (layer: ChunkLayer, layout: LayerLayout, count: number): ChunkData => {
        if (count <= 0) {
            throw new Error(`ChunkDataPool.acquire: count must be > 0 (got ${count})`);
        }
        if (count > chunkSize) {
            throw new Error(`ChunkDataPool.acquire: count ${count} exceeds chunkSize ${chunkSize}`);
        }

        // Allocate at full chunk capacity so short (final) chunks reuse the same
        // pool slot as full chunks of the same layer stride.
        const capacity = chunkSize * layout.stride;

        let data: ArrayBuffer | undefined;
        const stack = pool.get(capacity);
        if (stack && stack.length > 0) {
            data = stack.pop();
            bytesPooled -= capacity;
        } else {
            data = new ArrayBuffer(capacity);
        }
        bytesInUse += capacity;

        return new ChunkDataImpl({
            layer,
            count,
            stride: layout.stride,
            fields: layout.fields,
            data: data!,
            onRelease: release
        });
    };

    const trim = (targetBytes: number): void => {
        const target = targetBytes < 0 ? 0 : targetBytes;
        if (bytesPooled <= target) return;
        for (const stack of pool.values()) {
            while (stack.length > 0 && bytesPooled > target) {
                const buf = stack.pop()!;
                bytesPooled -= buf.byteLength;
            }
            if (bytesPooled <= target) break;
        }
    };

    const destroy = (): void => {
        pool.clear();
        bytesPooled = 0;
    };

    return {
        chunkSize,
        acquire,
        trim,
        destroy,
        get bytesInUse() {
            return bytesInUse;
        },
        get bytesPooled() {
            return bytesPooled;
        }
    };
};

export { type ChunkDataPool, createChunkDataPool };
