import { type ChunkFieldMap, type ChunkLayer } from './layout';

/**
 * A CPU-resident buffer holding one layer's data for one chunk of gaussians.
 *
 * A "chunk" is a row-range — a contiguous subset of gaussians; a `ChunkData`
 * holds one layer's data for one such chunk (it carries a `.layer`, and you
 * acquire and bind `position`/`geometric`/`color`/`other` separately). Buffers
 * are acquired from a {@link ChunkDataPool}, filled by
 * {@link ChunkSource.read}, used by consumers (writers, kernels that upload
 * `data` to the GPU themselves), and then released back to the pool. The
 * underlying `ArrayBuffer` is reused across subsequent acquisitions of the same
 * byte size.
 *
 * `count` is the number of gaussians of valid data this buffer holds; it
 * matches the source's `chunkSize` except for the final (short) chunk. The
 * backing `data` buffer is allocated at full chunk capacity (`chunkSize *
 * stride`), so the valid region is always the leading `count * stride` bytes.
 * `stride` is the bytes per gaussian, dictated by the layer (and, for `color`
 * and `other`, by the SH band count or extras schema).
 */
interface ChunkData {
    /** Which layer this buffer holds. */
    readonly layer: ChunkLayer;
    /** Number of gaussians of valid data this buffer holds. */
    readonly count: number;
    /** Bytes per gaussian for this buffer's layer. */
    readonly stride: number;
    /** Field name -> byte offset / component descriptor within the stride. */
    readonly fields: ChunkFieldMap;
    /**
     * CPU buffer holding this layer's interleaved per-gaussian records. Its
     * capacity may exceed `count * stride` (it is sized for a full chunk);
     * only the leading `count * stride` bytes are meaningful.
     */
    readonly data: ArrayBuffer;

    /**
     * Extract one named field as a tight (de-interleaved) typed-array over the
     * valid rows. The result is a copy — fields are generally a sub-span of the
     * stride, so a zero-copy view isn't possible.
     */
    field(name: string): Float32Array | Uint32Array;

    /**
     * Return this buffer to its {@link ChunkDataPool} for reuse.
     * After this call the buffer must not be referenced again.
     */
    release(): void;
}

/**
 * Internal release hook called by {@link ChunkDataImpl.release}. Implemented by
 * the {@link ChunkDataPool} to return the underlying buffer to the pool.
 */
type ReleaseFn = (chunkData: ChunkDataImpl) => void;

class ChunkDataImpl implements ChunkData {
    readonly layer: ChunkLayer;
    readonly count: number;
    readonly stride: number;
    readonly fields: ChunkFieldMap;
    readonly data: ArrayBuffer;

    private released = false;
    private readonly onRelease: ReleaseFn;

    constructor(options: {
        layer: ChunkLayer;
        count: number;
        stride: number;
        fields: ChunkFieldMap;
        data: ArrayBuffer;
        onRelease: ReleaseFn;
    }) {
        this.layer = options.layer;
        this.count = options.count;
        this.stride = options.stride;
        this.fields = options.fields;
        this.data = options.data;
        this.onRelease = options.onRelease;
    }

    field(name: string): Float32Array | Uint32Array {
        if (this.released) {
            throw new Error('ChunkData.field: cannot read a released buffer');
        }
        const f = this.fields[name];
        if (!f) {
            throw new Error(`ChunkData.field: unknown field '${name}' for layer '${this.layer}'`);
        }

        // Every layout field is a 4-byte-aligned f32/u32, so decode with typed
        // views and word indices — a per-element DataView loop here costs ~10×
        // and sits under whole-scene passes (e.g. the LOD writer's slim
        // extract). Little-endian runtimes only, like the rest of the chunk
        // model (see write-ply-streaming).
        const n = this.count * f.components;
        const strideW = this.stride >> 2;
        const offW = f.byteOffset >> 2;
        if (f.type === 'float32') {
            const out = new Float32Array(n);
            const src = new Float32Array(this.data);
            if (offW === 0 && f.components === strideW) {
                out.set(src.subarray(0, n)); // full-stride field: one block copy
            } else {
                const c = f.components;
                for (let i = 0; i < this.count; i++) {
                    const s = i * strideW + offW;
                    const d = i * c;
                    for (let j = 0; j < c; j++) out[d + j] = src[s + j];
                }
            }
            return out;
        }
        const out = new Uint32Array(n);
        const src = new Uint32Array(this.data);
        if (offW === 0 && f.components === strideW) {
            out.set(src.subarray(0, n)); // full-stride field: one block copy
        } else {
            const c = f.components;
            for (let i = 0; i < this.count; i++) {
                const s = i * strideW + offW;
                const d = i * c;
                for (let j = 0; j < c; j++) out[d + j] = src[s + j];
            }
        }
        return out;
    }

    release(): void {
        if (this.released) {
            return;
        }
        this.released = true;
        this.onRelease(this);
    }
}

export { type ChunkData, type ReleaseFn, ChunkDataImpl };
