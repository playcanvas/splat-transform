import { concatSource } from '../ops';
import {
    type ChunkDataPool,
    type ChunkReadRequest,
    type ChunkSource,
    type ChunkSourceMetadata
} from '../source';

/**
 * One contiguous run of gaussians within a LOD, backed by a sub-file that is
 * decoded lazily. `decode` must return a single-LOD `ChunkSource` of exactly
 * `count` gaussians, chunked at the same `chunkSize` as the pool passed to
 * {@link containerSource} (e.g. `readSpz(subFile, pool)`).
 */
type ContainerSegment = {
    count: number;
    decode: () => Promise<ChunkSource>;
};

/**
 * Present a container of lazily-decoded sub-files (LCC / LCC2: many SPZ/SOG
 * chunk files, LOD-grouped) as one **multi-LOD `ChunkSource`**, without ever
 * holding the whole scene resident.
 *
 * Each LOD's segments are stitched with {@link concatSource} (the proven
 * block-copy); a static-meta proxy stands in for each segment so concat can
 * compute layout up front, and the sub-file is decoded — and LRU-cached — only
 * when a `read` actually touches it. Reading chunks in order decodes each
 * sub-file once; the cache bounds resident decoded sub-files to `cacheSize`.
 *
 * All sub-files must share a layout (SH bands / layers / extras / transform);
 * the layout is taken from the first segment and a band mismatch throws (callers
 * needing heterogeneous bands must fall back to the eager `DataTable` path).
 *
 * @param segmentsByLod - Segments per output LOD, in output (file) order.
 * @param pool - Pool for the concat block-copy temporaries; `chunkSize` must match the segments'.
 * @param opts - Tuning options.
 * @param opts.cacheSize - Max resident decoded sub-files (default 3, min 2 so a chunk straddling two segments can be served).
 * @returns A lazy multi-LOD source over the container.
 */
const containerSource = async (
    segmentsByLod: ContainerSegment[][],
    pool: ChunkDataPool,
    opts: { cacheSize?: number } = {}
): Promise<ChunkSource> => {
    const cacheSize = Math.max(2, opts.cacheSize ?? 3);
    const cache = new Map<ContainerSegment, Promise<ChunkSource>>();
    const order: ContainerSegment[] = [];

    const decode = (seg: ContainerSegment): Promise<ChunkSource> => {
        const hit = cache.get(seg);
        if (hit) return hit;
        const p = seg.decode();
        cache.set(seg, p);
        order.push(seg);
        while (order.length > cacheSize) {
            const evicted = order.shift()!;
            const ep = cache.get(evicted)!;
            cache.delete(evicted);
            ep.then(s => s.close()).catch(() => {});
        }
        return p;
    };

    const firstSeg = segmentsByLod.find(segs => segs.length > 0)?.[0];
    if (!firstSeg) {
        throw new Error('containerSource: no segments');
    }
    // The (uniform) layout is taken from the first segment; it stays cached for
    // its imminent first read.
    const layout = (await decode(firstSeg)).meta;
    const { chunkSize } = layout;

    const lodCounts = segmentsByLod.map(segs => segs.reduce((acc, s) => acc + s.count, 0));

    // Static-meta proxy: concatSource reads its `meta` to lay out the LOD up
    // front; the actual sub-source is decoded (and cached) only on `read`.
    const proxy = (seg: ContainerSegment): ChunkSource => ({
        meta: {
            ...layout,
            numGaussians: seg.count,
            numLods: 1,
            lodCounts: [seg.count],
            numChunks: [Math.max(1, Math.ceil(seg.count / chunkSize))]
        },
        read: async (req: ChunkReadRequest): Promise<void> => {
            const src = await decode(seg);
            if (src.meta.shBands !== layout.shBands) {
                throw new Error(
                    `containerSource: sub-file SH band mismatch (${src.meta.shBands} vs ${layout.shBands})`
                );
            }
            await src.read(req);
        },
        close: () => Promise.resolve()
    });

    const perLod = segmentsByLod.map(segs => (segs.length > 0 ? concatSource(segs.map(proxy), pool) : null));

    const meta: ChunkSourceMetadata = {
        ...layout,
        numGaussians: lodCounts[0] ?? 0,
        numLods: segmentsByLod.length,
        lodCounts,
        numChunks: lodCounts.map(c => Math.ceil(c / chunkSize))
    };

    const read = (req: ChunkReadRequest): Promise<void> => {
        const src = perLod[req.lod ?? 0];
        if (!src) {
            throw new Error(`containerSource: empty LOD ${req.lod ?? 0}`);
        }
        return src.read({ ...req, lod: 0 });
    };

    const close = async (): Promise<void> => {
        await Promise.all([...cache.values()].map(p => p.then(s => s.close()).catch(() => {})));
        cache.clear();
        order.length = 0;
    };

    return { meta, read, close };
};

export { containerSource, type ContainerSegment };
