/**
 * PLY index-gather equivalence: reading arbitrary row subsets via
 * `read({indices, ...})` must decode byte-identical values to a whole-chunk
 * read of the same rows, across index patterns chosen to exercise the
 * run/gap/window coalescing logic.
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

import { createTestDataTable, encodePlyBinary } from './helpers/test-utils.mjs';
import { createChunkDataPool } from '../src/lib/chunk/index.js';
import { readPly } from '../src/lib/readers/read-ply.js';

// Minimal seekable ReadSource over a buffer (same pattern as write-lod.test.mjs).
class BufferReadSource {
    constructor(data) {
        this.data = data;
        this.size = data.length;
        this.seekable = true;
    }

    read(start = 0, end = this.size) {
        let offset = Math.max(0, Math.min(start, this.size));
        const limit = Math.max(offset, Math.min(end, this.size));
        const data = this.data;
        return {
            async pull(target) {
                const remaining = limit - offset;
                if (remaining <= 0) return 0;
                const n = Math.min(target.length, remaining);
                target.set(data.subarray(offset, offset + n));
                offset += n;
                return n;
            }
        };
    }

    close() {}
}

const N = 5000;

const openSource = async () => {
    const table = createTestDataTable(N, { includeSH: true, shBands: 3 });
    const ply = encodePlyBinary(table);
    const pool = createChunkDataPool();
    const source = await readPly(new BufferReadSource(ply), pool);
    return { source, pool };
};

// Whole-scene reference: read chunk 0 (N < chunkSize so it's one chunk).
const readReference = async (source, pool) => {
    const { layouts } = source.meta;
    const pos = pool.acquire('position', layouts.position, N);
    const geo = pool.acquire('geometric', layouts.geometric, N);
    const col = pool.acquire('color', layouts.color, N);
    await source.read({ chunkIndex: 0, position: pos, geometric: geo, color: col });
    const snap = (cd) => new Float32Array(cd.data.slice(0, N * cd.stride));
    const ref = { position: snap(pos), geometric: snap(geo), color: snap(col) };
    pos.release(); geo.release(); col.release();
    return ref;
};

const gatherAndCompare = async (source, pool, ref, indices, indexOffset = 0) => {
    const count = indices.length - indexOffset;
    const { layouts } = source.meta;
    const pos = pool.acquire('position', layouts.position, count);
    const geo = pool.acquire('geometric', layouts.geometric, count);
    const col = pool.acquire('color', layouts.color, count);
    await source.read({ indices, indexOffset, count, position: pos, geometric: geo, color: col });
    for (const [layer, cd] of [['position', pos], ['geometric', geo], ['color', col]]) {
        const got = new Float32Array(cd.data, 0, count * (cd.stride >> 2));
        const sw = cd.stride >> 2;
        for (let j = 0; j < count; j++) {
            const src = indices[indexOffset + j];
            for (let w = 0; w < sw; w++) {
                assert.strictEqual(
                    got[j * sw + w], ref[layer][src * sw + w],
                    `${layer} row ${j} (src ${src}) word ${w}`
                );
            }
        }
    }
    pos.release(); geo.release(); col.release();
};

describe('readPly index gather', () => {
    it('matches chunk reads across coalescing patterns', async () => {
        const { source, pool } = await openSource();
        const ref = await readReference(source, pool);

        // consecutive run (single merged read)
        await gatherAndCompare(source, pool, ref, Uint32Array.from({ length: 64 }, (_, i) => 100 + i));
        // small gaps (merge candidates: gaps of 1..8 rows)
        await gatherAndCompare(source, pool, ref, Uint32Array.from({ length: 200 }, (_, i) => i * 7));
        // large gaps (beyond any merge threshold: separate reads)
        await gatherAndCompare(source, pool, ref, Uint32Array.from([0, 1200, 2400, 3600, 4800, N - 1]));
        // duplicates and unsorted request order
        await gatherAndCompare(source, pool, ref, Uint32Array.from([42, 7, 42, 4999, 7, 0, 42]));
        // deterministic pseudo-random scatter
        const scatter = new Uint32Array(500);
        let s = 12345;
        for (let i = 0; i < scatter.length; i++) {
            s = (s * 1103515245 + 12345) >>> 0;
            scatter[i] = s % N;
        }
        await gatherAndCompare(source, pool, ref, scatter);
        // non-zero indexOffset
        const offs = new Uint32Array(20);
        offs.set([9, 9, 9], 0); // ignored prefix
        for (let i = 3; i < 20; i++) offs[i] = (i * 271) % N;
        await gatherAndCompare(source, pool, ref, offs, 3);

        await source.close();
    });

    it('handles runs beyond the window cap and gaps straddling the merge threshold', async () => {
        // createTestDataTable(shBands: 3) → 14 base + 45 SH float properties
        // = 59 × 4 B = 236 B/record. 40,000 rows ≈ 9.4 MB of records, so a
        // full-range consecutive gather must split across the 8 MB window cap.
        const BIG = 40000;
        const table = createTestDataTable(BIG, { includeSH: true, shBands: 3 });
        const ply = encodePlyBinary(table);
        const pool = createChunkDataPool();
        const source = await readPly(new BufferReadSource(ply), pool);
        const { layouts } = source.meta;

        // Reference: whole-range chunk read (BIG < chunkSize, so chunk 0 holds all rows).
        const refPos = pool.acquire('position', layouts.position, BIG);
        await source.read({ chunkIndex: 0, position: refPos });
        const ref = new Float32Array(refPos.data.slice(0, BIG * refPos.stride));
        refPos.release();

        const compare = async (indices) => {
            const count = indices.length;
            const pos = pool.acquire('position', layouts.position, count);
            await source.read({ indices, indexOffset: 0, count, position: pos });
            const got = new Float32Array(pos.data, 0, count * 3);
            for (let j = 0; j < count; j++) {
                const src = indices[j];
                for (let w = 0; w < 3; w++) {
                    assert.strictEqual(got[j * 3 + w], ref[src * 3 + w], `row ${j} (src ${src}) word ${w}`);
                }
            }
            pos.release();
        };

        // Consecutive run spanning the whole file: ≈9.4 MB > 8 MB, must split into windows.
        await compare(Uint32Array.from({ length: BIG }, (_, i) => i));

        // Row distances straddling the 64 KB merge threshold at this stride:
        // 277 × 236 = 65,372 B (≤ 65,536: merges) vs 278 × 236 = 65,608 B
        // (> 65,536: splits). Every row must land in its output slot either way.
        for (const step of [277, 278]) {
            const n = Math.floor((BIG - 1) / step) + 1;
            await compare(Uint32Array.from({ length: n }, (_, i) => i * step));
        }

        await source.close();
    });

    it('handles count <= 0 as a no-op', async () => {
        const { source, pool } = await openSource();
        const pos = pool.acquire('position', source.meta.layouts.position, 1);
        await source.read({ indices: new Uint32Array([0]), indexOffset: 0, count: 0, position: pos });
        pos.release();
        await source.close();
    });
});
