import { type Transform } from '../utils';
import { type ChunkData } from './chunk-data';
import { type ChunkDataPool, type LayerLayout } from './chunk-data-pool';
import { type ChunkSource, type ReadRequest, type ChunkSourceMetadata } from './chunk-source';
import {
    colorFields,
    colorStride,
    GEOMETRIC_STRIDE,
    geometricFields,
    type ExtraColumn,
    type ChunkLayer,
    otherLayout,
    positionFields,
    POSITION_STRIDE,
    type SHBands
} from './layout';

/**
 * Per-layer, per-LOD, per-chunk CPU-resident byte storage.
 *
 * Indexed as `buffers[layer][lod][chunkIndex]`. `undefined` for layers the
 * source doesn't carry. Each stored buffer is exactly `count * stride` bytes.
 */
type LayerChunkBuffers = {
    [L in ChunkLayer]?: ReadonlyArray<ReadonlyArray<ArrayBuffer>>;
};

/**
 * Lightweight CPU-resident source. Constructed directly from a set of
 * per-layer per-LOD per-chunk `ArrayBuffer`s. Used as:
 *
 * - The result of {@link compact} (densified output of a lazy combinator).
 * - A synthetic source for tests.
 * - The materialization for whole-blob formats (SPZ, MJS) once decoded.
 *
 * The source holds the provided buffers for its entire lifetime. `close()`
 * drops the references so they can be garbage collected.
 */
class InMemoryChunkSource implements ChunkSource {
    meta: ChunkSourceMetadata;
    private buffers: LayerChunkBuffers | null;
    // One u32 view per chunk buffer, per (layer, lod), built on first use and
    // reused — a gather pass touches every chunk, so views are made once.
    private viewCache: Map<string, Uint32Array[]> = new Map();

    constructor(meta: ChunkSourceMetadata, buffers: LayerChunkBuffers) {
        this.meta = meta;
        this.buffers = buffers;
    }

    read(request: ReadRequest): Promise<void> {
        if (this.buffers === null) {
            throw new Error('InMemoryChunkSource.read: source has been closed');
        }
        const lod = request.lod ?? 0;

        if ('indices' in request) {
            // Gather: output row j receives resident row indices[indexOffset + j],
            // a per-layer dispatch onto gatherRows.
            const { indices, indexOffset, count } = request;
            const gather = (cd: ChunkData | undefined, layer: ChunkLayer): void => {
                if (cd) this.gatherRows(layer, lod, indices, indexOffset, count, cd.data);
            };
            gather(request.position, 'position');
            gather(request.geometric, 'geometric');
            gather(request.color, 'color');
            gather(request.other, 'other');
            return Promise.resolve();
        }

        const { chunkIndex } = request;
        const buffers = this.buffers;

        const fill = (chunkData: ChunkData | undefined, layer: ChunkLayer): void => {
            if (!chunkData) return;
            const layerBuffers = buffers[layer];
            if (!layerBuffers) {
                throw new Error(`InMemoryChunkSource: layer '${layer}' not available`);
            }
            const buf = layerBuffers[lod]?.[chunkIndex];
            if (buf === undefined) {
                throw new Error(
                    `InMemoryChunkSource: missing buffer for layer='${layer}' lod=${lod} chunkIndex=${chunkIndex}`
                );
            }
            const expected = chunkData.count * chunkData.stride;
            if (buf.byteLength !== expected) {
                throw new Error(
                    `InMemoryChunkSource: buffer size mismatch for layer='${layer}' lod=${lod} chunk=${chunkIndex}: expected ${expected}, got ${buf.byteLength}`
                );
            }
            new Uint8Array(chunkData.data, 0, expected).set(new Uint8Array(buf));
        };

        fill(request.position, 'position');
        fill(request.geometric, 'geometric');
        fill(request.color, 'color');
        fill(request.other, 'other');

        return Promise.resolve();
    }

    // Lazily build (and cache) one u32 view per chunk buffer of a (layer, lod),
    // so a multi-chunk gather doesn't re-create views on every call.
    private layerViews(layer: ChunkLayer, lod: number): Uint32Array[] {
        const key = `${layer}:${lod}`;
        let views = this.viewCache.get(key);
        if (!views) {
            const layerBuffers = this.buffers![layer];
            if (!layerBuffers) {
                throw new Error(`InMemoryChunkSource.gatherRows: layer '${layer}' not available`);
            }
            views = layerBuffers[lod].map(b => new Uint32Array(b));
            this.viewCache.set(key, views);
        }
        return views;
    }

    /**
     * Gather `count` gaussians' records for one layer into `dst`, packed: output
     * row `j` receives parent row `indices[indexOffset + j]`. A tight 32-bit-word
     * copy with per-layer constants and chunk views hoisted out of the loop — the
     * batch form of a per-row copy, for permute/gather combinators random-
     * accessing a resident scene.
     *
     * @param layer - Which layer to gather.
     * @param lod - LOD index.
     * @param indices - Source gaussian indices.
     * @param indexOffset - Offset into `indices` of the first row to gather.
     * @param count - Number of rows to gather.
     * @param dst - Destination buffer; receives `count` packed records.
     */
    gatherRows(
        layer: ChunkLayer,
        lod: number,
        indices: Uint32Array,
        indexOffset: number,
        count: number,
        dst: ArrayBuffer
    ): void {
        if (this.buffers === null) {
            throw new Error('InMemoryChunkSource.gatherRows: source has been closed');
        }
        const views = this.layerViews(layer, lod);
        const sw = this.meta.layouts[layer]!.stride >>> 2; // u32 words per gaussian
        const chunkSize = this.meta.chunkSize;
        const out = new Uint32Array(dst);
        for (let j = 0; j < count; j++) {
            const g = indices[indexOffset + j];
            const chunk = Math.floor(g / chunkSize);
            const src = views[chunk];
            const so = (g - chunk * chunkSize) * sw;
            const dof = j * sw;
            for (let w = 0; w < sw; w++) {
                out[dof + w] = src[so + w];
            }
        }
    }

    close(): Promise<void> {
        this.buffers = null;
        this.viewCache.clear();
        return Promise.resolve();
    }
}

type LayerBuffers = ReadonlyArray<ReadonlyArray<ArrayBuffer>>;

/**
 * Convenience constructor for an `InMemoryChunkSource` built from raw layer buffers.
 *
 * `lodCounts` must be supplied explicitly (`lodCounts[0] === numGaussians`):
 * the per-LOD gaussian count can't be inferred from chunk counts alone because
 * the final chunk of each LOD may be short.
 * @param params - Source construction parameters.
 * @param params.numGaussians - Total gaussian count (LOD 0).
 * @param params.chunkSize - Gaussians per chunk.
 * @param params.shBands - SH band count of the color layer.
 * @param params.extraColumns - Descriptors for the `other` layer columns.
 * @param params.transform - Pending coordinate-space transform.
 * @param params.lodCounts - Gaussians per LOD; `lodCounts[0]` must equal `numGaussians`.
 * @param params.position - Per-LOD per-chunk position buffers, or undefined.
 * @param params.geometric - Per-LOD per-chunk geometric buffers, or undefined.
 * @param params.color - Per-LOD per-chunk color buffers, or undefined.
 * @param params.other - Per-LOD per-chunk other buffers, or undefined.
 * @returns The constructed `InMemoryChunkSource`.
 */
const createInMemoryChunkSource = (params: {
    numGaussians: number;
    chunkSize: number;
    shBands: SHBands;
    extraColumns?: ReadonlyArray<ExtraColumn>;
    transform: Transform;
    /** Gaussians per LOD. `lodCounts[0]` must equal `numGaussians`. */
    lodCounts: ReadonlyArray<number>;
    /** Per-LOD per-chunk layer buffers, or `undefined` if the source lacks the layer. */
    position?: LayerBuffers;
    geometric?: LayerBuffers;
    color?: LayerBuffers;
    other?: LayerBuffers;
}): InMemoryChunkSource => {
    const {
        numGaussians, chunkSize, shBands, transform, lodCounts,
        position, geometric, color, other
    } = params;
    const extras = params.extraColumns ?? [];

    if (lodCounts.length === 0) {
        throw new Error('createInMemoryChunkSource: lodCounts must be non-empty');
    }
    if (lodCounts[0] !== numGaussians) {
        throw new Error(
            `createInMemoryChunkSource: lodCounts[0] (${lodCounts[0]}) must equal numGaussians (${numGaussians})`
        );
    }

    const numLods = lodCounts.length;
    const numChunks = lodCounts.map(c => Math.ceil(c / chunkSize));

    const availableLayers = new Set<ChunkLayer>();
    const layouts: Partial<Record<ChunkLayer, LayerLayout>> = {};
    const buffers: LayerChunkBuffers = {};

    const register = (layer: ChunkLayer, data: LayerBuffers, layout: LayerLayout): void => {
        if (data.length !== numLods) {
            throw new Error(
                `createInMemoryChunkSource: layer '${layer}' has ${data.length} LODs, expected ${numLods}`
            );
        }
        for (let l = 0; l < numLods; l++) {
            if (data[l].length !== numChunks[l]) {
                throw new Error(
                    `createInMemoryChunkSource: layer '${layer}' lod ${l} has ${data[l].length} chunks, expected ${numChunks[l]}`
                );
            }
        }
        availableLayers.add(layer);
        layouts[layer] = layout;
        buffers[layer] = data;
    };

    if (position) register('position', position, { stride: POSITION_STRIDE, fields: positionFields() });
    if (geometric) register('geometric', geometric, { stride: GEOMETRIC_STRIDE, fields: geometricFields() });
    if (color) register('color', color, { stride: colorStride(shBands), fields: colorFields(shBands) });
    if (other) {
        const ol = otherLayout(extras);
        register('other', other, { stride: ol.stride, fields: ol.fields });
    }

    if (availableLayers.size === 0) {
        throw new Error('createInMemoryChunkSource: at least one layer must be provided');
    }

    const meta: ChunkSourceMetadata = {
        numGaussians,
        numLods,
        lodCounts: lodCounts.slice(),
        chunkSize,
        numChunks,
        shBands,
        extraColumns: extras,
        transform,
        availableLayers,
        layouts
    };

    return new InMemoryChunkSource(meta, buffers);
};

/**
 * Densify a source into a fresh {@link InMemoryChunkSource} by reading every chunk
 * of every available layer in order. Useful for:
 *
 * - Tests that need to compare two sources byte-for-byte.
 * - Materializing the output of lazy combinators so subsequent random-access
 *   reads don't re-decode the parent every time.
 *
 * Acquires one `ChunkData` buffer per available layer at a time (via `pool`),
 * reads the parent into it, copies the valid bytes into a fresh `ArrayBuffer`,
 * and releases. The pool's `chunkSize` must be >= the source's.
 * @param src - The source to densify.
 * @param pool - The `ChunkData` pool used for the temporary read buffers.
 * @returns A fresh `InMemoryChunkSource` holding the materialized data.
 */
const compact = async (
    src: ChunkSource,
    pool: ChunkDataPool
): Promise<InMemoryChunkSource> => {
    const { meta } = src;
    const layers: ChunkLayer[] = (['position', 'geometric', 'color', 'other'] as ChunkLayer[])
    .filter(l => meta.availableLayers.has(l));

    const out: { [L in ChunkLayer]?: ArrayBuffer[][] } = {};
    for (const l of layers) {
        out[l] = meta.lodCounts.map((): ArrayBuffer[] => []);
    }

    for (let lod = 0; lod < meta.numLods; lod++) {
        const total = meta.lodCounts[lod];
        for (let k = 0; k < meta.numChunks[lod]; k++) {
            const count = Math.min(meta.chunkSize, total - k * meta.chunkSize);
            const acquired: Partial<Record<ChunkLayer, ChunkData>> = {};
            for (const l of layers) {
                acquired[l] = pool.acquire(l, meta.layouts[l]!, count);
            }
            await src.read({
                chunkIndex: k,
                lod,
                position: acquired.position,
                geometric: acquired.geometric,
                color: acquired.color,
                other: acquired.other
            });
            for (const l of layers) {
                const chunkData = acquired[l]!;
                out[l]![lod].push(chunkData.data.slice(0, chunkData.count * chunkData.stride));
                chunkData.release();
            }
        }
    }

    return new InMemoryChunkSource(meta, out);
};

export { InMemoryChunkSource, createInMemoryChunkSource, compact };
