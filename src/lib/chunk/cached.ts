import { type ChunkData } from './data';
import { InMemoryChunkSource } from './in-memory';
import { type ChunkLayer } from './layout';
import { createChunkDataPool, type ChunkDataPool } from './pool';
import { type ReadRequest, type ChunkSource } from './source';

const LAYERS: readonly ChunkLayer[] = ['position', 'geometric', 'color', 'other'];

/**
 * Wrap a lazy source in an LRU decode cache with a user-set byte budget.
 *
 * Decoded chunk bytes are cached per `(lod, chunkIndex, layer)`. A `read` for a
 * cached chunk copies straight from the cache; a miss faults all the chunk's
 * still-missing layers through the parent in a single `read`, caches them, then
 * evicts least-recently-used entries until the cache is back within `maxBytes`.
 *
 * Purpose: amortize re-decode/re-read of an expensive lazy source across passes
 * (a writer that scans a layer twice, LOD's repeated per-unit gathers, a
 * whole-texture SOG decode hit many times) and give the caller one knob —
 * `maxBytes` — to trade peak memory against re-reads. With `maxBytes` ≈ one
 * layer, sequential layer-by-layer consumption keeps the current layer hot and
 * evicts the previous (layer-at-a-time), with no thrash because the access is
 * sequential.
 *
 * An already-resident {@link InMemoryChunkSource} is returned unchanged — caching
 * it would just double its memory.
 *
 * @param parent - Source to cache (typically file-backed / lazy).
 * @param options - Cache options.
 * @param options.maxBytes - Soft cap on cached bytes; may be briefly exceeded to serve one read.
 * @returns A caching view over `parent`, or `parent` itself if already resident.
 */
const cached = (parent: ChunkSource, options: { maxBytes: number }): ChunkSource => {
    if (parent instanceof InMemoryChunkSource) {
        return parent;
    }

    const { maxBytes } = options;
    const { meta } = parent;
    const pool: ChunkDataPool = createChunkDataPool({ chunkSize: meta.chunkSize });

    // LRU: Map iteration order is insertion order; re-insert on touch so the
    // oldest key is always first.
    const cache = new Map<string, ArrayBuffer>();
    let bytes = 0;

    const keyOf = (lod: number, chunkIndex: number, layer: ChunkLayer): string => {
        return `${lod}:${chunkIndex}:${layer}`;
    };

    const touch = (k: string): ArrayBuffer => {
        const v = cache.get(k)!;
        cache.delete(k);
        cache.set(k, v);
        return v;
    };

    const evict = (): void => {
        while (bytes > maxBytes && cache.size > 0) {
            const oldest = cache.keys().next().value as string;
            bytes -= cache.get(oldest)!.byteLength;
            cache.delete(oldest);
        }
    };

    const read = async (request: ReadRequest): Promise<void> => {
        // Gather requests bypass the chunk cache — it keys by chunkIndex and can't
        // serve arbitrary rows — and pass straight through to the parent (whose
        // own residency, e.g. resident SOG textures, amortizes the gather).
        if ('indices' in request) {
            return parent.read(request);
        }
        const lod = request.lod ?? 0;
        const { chunkIndex } = request;
        const count = Math.min(meta.chunkSize, meta.lodCounts[lod] - chunkIndex * meta.chunkSize);

        const wanted = LAYERS.filter(l => request[l]);
        const missing = wanted.filter(l => !cache.has(keyOf(lod, chunkIndex, l)));

        if (missing.length > 0) {
            // Fault every missing layer of this chunk in ONE parent read (a lazy
            // reader decodes the record range once, not once per layer).
            const temp: Partial<Record<ChunkLayer, ChunkData>> = {};
            for (const l of missing) temp[l] = pool.acquire(l, meta.layouts[l]!, count);
            await parent.read({
                chunkIndex,
                lod,
                position: temp.position,
                geometric: temp.geometric,
                color: temp.color,
                other: temp.other
            });
            for (const l of missing) {
                const cd = temp[l]!;
                const slice = cd.data.slice(0, cd.count * cd.stride);
                cache.set(keyOf(lod, chunkIndex, l), slice);
                bytes += slice.byteLength;
                cd.release();
            }
        }

        // Copy cached bytes into the caller's buffers; touch = mark recently used.
        for (const l of wanted) {
            const buf = touch(keyOf(lod, chunkIndex, l));
            const dst = request[l]!;
            new Uint8Array(dst.data, 0, buf.byteLength).set(new Uint8Array(buf));
        }

        // Evict AFTER copying, so the chunk we just served isn't dropped mid-read.
        evict();
        return Promise.resolve();
    };

    const close = (): Promise<void> => {
        cache.clear();
        bytes = 0;
        pool.destroy();
        return parent.close();
    };

    return { meta, read, close };
};

export { cached };
