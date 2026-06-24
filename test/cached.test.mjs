/**
 * cached(): LRU decode cache with a user byte budget.
 *  - passthrough over an already-resident source
 *  - byte-identical reads; all layers of a chunk fault in one parent read
 *  - tight budget evicts cold chunks (re-read); generous budget retains them
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

import { createTestDataTable } from './helpers/test-utils.mjs';
import { dataTableToChunkSource } from '../src/lib/compat/data-table.js';
import { cached, createChunkDataPool } from '../src/lib/source/index.js';

// Wrap a source to count read() calls (and defeat the resident passthrough,
// since the wrapper is not an InMemoryChunkSource).
function counting(inner) {
    const state = { reads: 0 };
    const src = {
        meta: inner.meta,
        read: (req) => {
            state.reads++;
            return inner.read(req);
        },
        close: () => inner.close()
    };
    return { src, state };
}

async function readChunk(src, pool, chunkIndex, layers) {
    const meta = src.meta;
    const count = Math.min(meta.chunkSize, meta.numGaussians - chunkIndex * meta.chunkSize);
    const acq = {};
    for (const l of layers) acq[l] = pool.acquire(l, meta.layouts[l], count);
    await src.read({ chunkIndex, position: acq.position, geometric: acq.geometric, color: acq.color, other: acq.other });
    const out = {};
    for (const l of layers) out[l] = new Uint8Array(acq[l].data.slice(0, acq[l].count * acq[l].stride));
    for (const l of layers) acq[l].release();
    return out;
}

describe('cached: LRU decode cache', () => {
    it('passes through an already-resident source unchanged', () => {
        const base = dataTableToChunkSource(createTestDataTable(20), 8);
        assert.strictEqual(cached(base, { maxBytes: 1 << 20 }), base);
    });

    it('returns byte-identical data and faults all layers of a chunk in one parent read', async () => {
        const dt = createTestDataTable(100, { includeSH: true, shBands: 1 });
        const base = dataTableToChunkSource(dt, 32);
        const { src, state } = counting(base);
        const c = cached(src, { maxBytes: 1 << 30 });
        const pool = createChunkDataPool({ chunkSize: 32 });
        const layers = ['position', 'geometric', 'color'];

        const a = await readChunk(c, pool, 0, layers);
        assert.strictEqual(state.reads, 1, 'all layers of a chunk fault in one parent read');

        const b = await readChunk(c, pool, 0, layers);
        assert.strictEqual(state.reads, 1, 'cached chunk is not re-read');
        for (const l of layers) assert.deepStrictEqual(b[l], a[l]);

        const ref = await readChunk(base, pool, 0, layers);
        for (const l of layers) assert.deepStrictEqual(a[l], ref[l], `layer '${l}' must match uncached`);
    });

    it('evicts cold chunks under a tight budget but retains under a generous one', async () => {
        const dt = createTestDataTable(200); // SH0: position chunk = 32 * 12 = 384 B
        const mk = () => counting(dataTableToChunkSource(dt, 32));
        const pool = createChunkDataPool({ chunkSize: 32 });

        const tight = mk();
        const ct = cached(tight.src, { maxBytes: 32 * 12 }); // ~one position chunk
        await readChunk(ct, pool, 0, ['position']);
        await readChunk(ct, pool, 1, ['position']); // evicts chunk 0
        assert.strictEqual(tight.state.reads, 2);
        await readChunk(ct, pool, 0, ['position']); // chunk 0 cold -> re-fault
        assert.strictEqual(tight.state.reads, 3, 'cold chunk re-read under tight budget');

        const big = mk();
        const cb = cached(big.src, { maxBytes: 1 << 30 });
        await readChunk(cb, pool, 0, ['position']);
        await readChunk(cb, pool, 1, ['position']);
        await readChunk(cb, pool, 0, ['position']); // still cached
        assert.strictEqual(big.state.reads, 2, 'no re-read under generous budget');
    });
});
