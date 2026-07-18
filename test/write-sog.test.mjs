/**
 * The SOG writer (`writeSogSource`, ChunkSource input) and its DataTable adapter
 * (`writeSog`, which wraps the table via the migration shim).
 *
 *  - SH0: the two read paths agree byte-for-byte — decode -> writeSog (DataTable
 *    shim) vs readPly -> writeSogSource (native) — proving the deterministic
 *    machinery (per-layer gather, Morton order, encoding, texel layout, meta).
 *  - SH3: round-trips within tolerance — k-means clustering is non-deterministic
 *    across runs (random init), so the SH path is validated by decode + epsilon,
 *    the same way the existing SOG goldens are.
 */

import assert from 'node:assert';
import { dirname, join } from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';

import { compareSummaries, computeStatsView } from './helpers/summary-compare.mjs';
import { createTestDataTable, encodePlyBinary } from './helpers/test-utils.mjs';
import { dataTableToChunkSource } from '../src/lib/compat/data-table.js';
import { readFile as readRawFile } from '../src/lib/io/read/index.js';
import {
    Transform, readSog,
    MemoryFileSystem, MemoryReadFileSystem, ZipReadFileSystem, WebPCodec
} from '../src/lib/index.js';
import { materializeToDataTable } from '../src/lib/compat/data-table.js';
import { permuteSource } from '../src/lib/ops/index.js';
import { decodePlyToDataTable, readPly } from '../src/lib/readers/read-ply.js';
import { readSogSource } from '../src/lib/readers/read-sog.js';
import { createChunkDataPool } from '../src/lib/chunk/index.js';
import { writeSog, writeSogSource } from '../src/lib/writers/write-sog.js';

const sourceFromBytes = (bytes) => {
    const rfs = new MemoryReadFileSystem();
    rfs.set('in.ply', bytes);
    return rfs.createSource('in.ply');
};

const assertFilesEqual = (legacyFs, nativeFs) => {
    const lk = [...legacyFs.results.keys()].sort();
    assert.deepStrictEqual([...nativeFs.results.keys()].sort(), lk, 'same set of output files');
    for (const k of lk) {
        const lv = legacyFs.results.get(k);
        const nv = nativeFs.results.get(k);
        assert.ok(
            Buffer.compare(
                Buffer.from(lv.buffer, lv.byteOffset, lv.byteLength),
                Buffer.from(nv.buffer, nv.byteOffset, nv.byteLength)
            ) === 0,
            `bytes differ for ${k}`
        );
    }
};

const __dirname = dirname(fileURLToPath(import.meta.url));
WebPCodec.wasmUrl = join(__dirname, '..', 'lib', 'webp.wasm');

describe('writeSogSource: native SOG from a ChunkSource', () => {

    it('end-to-end: readPly -> writeSogSource == decode -> writeSog (SH0)', async () => {
        const dt = createTestDataTable(300);
        const plyBytes = encodePlyBinary(dt);

        const legacyDt = await decodePlyToDataTable(await sourceFromBytes(plyBytes));
        const legacyFs = new MemoryFileSystem();
        await writeSog({ filename: 'out.sog', dataTable: legacyDt, bundle: false, iterations: 5, logging: 'silent' }, legacyFs);

        const pool = createChunkDataPool();
        const src = await readPly(await sourceFromBytes(plyBytes), pool);
        const nativeFs = new MemoryFileSystem();
        await writeSogSource(src, pool, { filename: 'out.sog', bundle: false, iterations: 5, logging: 'silent' }, nativeFs);

        assertFilesEqual(legacyFs, nativeFs);
    });

    it('SH3 round-trips within tolerance (k-means is non-deterministic)', async () => {
        const dt = createTestDataTable(5000, { includeSH: true, shBands: 3 });
        dt.transform = Transform.PLY.clone();
        const expected = await computeStatsView(dt);

        const pool = createChunkDataPool();
        const fs = new MemoryFileSystem();
        await writeSogSource(dataTableToChunkSource(dt, pool.chunkSize), pool,
            { filename: 'out.sog', bundle: true, iterations: 5, logging: 'silent' }, fs);

        const rfs = new MemoryReadFileSystem();
        rfs.set('out.sog', fs.results.get('out.sog'));
        const zip = new ZipReadFileSystem(await rfs.createSource('out.sog'));
        const decoded = await readSog(zip, 'meta.json');

        assert.strictEqual(decoded.numRows, dt.numRows);
        compareSummaries(await computeStatsView(decoded), expected, { tolerance: 0.5, allowExtraColumns: true });
    });

    it('flat splats (scale_2 = -Infinity, a valid filterNaN survivor) produce a finite codebook', async () => {
        // Regression: one -Infinity pooled into the scales quantizer used to
        // NaN-poison all 256 codebook entries (JSON nulls; readers decode
        // log-scale 0 — unit-sized splats).
        const dt = createTestDataTable(300);
        dt.transform = Transform.PLY.clone();
        const s2 = dt.getColumnByName('scale_2').data;
        let numFlat = 0;
        let finiteMin = Infinity;
        for (let i = 0; i < s2.length; i++) {
            if (i % 10 === 0) {
                s2[i] = -Infinity;
                numFlat++;
            }
        }
        for (const name of ['scale_0', 'scale_1', 'scale_2']) {
            for (const v of dt.getColumnByName(name).data) {
                if (isFinite(v) && v < finiteMin) finiteMin = v;
            }
        }

        const pool = createChunkDataPool();
        const fs = new MemoryFileSystem();
        await writeSogSource(dataTableToChunkSource(dt, pool.chunkSize), pool,
            { filename: 'out.sog', bundle: true, iterations: 5, logging: 'silent' }, fs);

        const rfs = new MemoryReadFileSystem();
        rfs.set('out.sog', fs.results.get('out.sog'));
        const zip = new ZipReadFileSystem(await rfs.createSource('out.sog'));
        const meta = JSON.parse(new TextDecoder().decode(await readRawFile(zip, 'meta.json')));
        assert.ok(meta.scales.codebook.every(v => typeof v === 'number' && isFinite(v)),
            'scales codebook must contain only finite numbers');

        // decode: exactly the flat splats come back below the finite range
        const decoded = await readSog(zip, 'meta.json');
        const out = decoded.getColumnByName('scale_2').data;
        let decodedFlat = 0;
        for (const v of out) {
            assert.ok(isFinite(v), `decoded scale ${v} not finite`);
            if (v <= finiteMin - 10) decodedFlat++;
        }
        assert.strictEqual(decodedFlat, numFlat);
    });
});

describe('readSogSource: native chunked SOG read', () => {
    // Write a SOG, then read it two ways from the SAME bytes: a full sequential
    // read() (via materialize) and a random-access readRows() (via permuteSource).
    // Both share the per-gaussian decode, so gathered row j must EXACTLY equal
    // full row order[j] — k-means non-determinism is at write time, not read.
    it('readRows gathers rows identically to a full read (SH3)', async () => {
        const dt = createTestDataTable(300, { includeSH: true, shBands: 3 });
        dt.transform = Transform.PLY.clone();

        const pool = createChunkDataPool();
        const fs = new MemoryFileSystem();
        await writeSogSource(dataTableToChunkSource(dt, pool.chunkSize), pool,
            { filename: 'out.sog', bundle: true, iterations: 5, logging: 'silent' }, fs);

        const rfs = new MemoryReadFileSystem();
        rfs.set('out.sog', fs.results.get('out.sog'));
        const zip = new ZipReadFileSystem(await rfs.createSource('out.sog'));

        const src = await readSogSource(zip, 'meta.json', pool);
        assert.strictEqual(src.meta.shBands, 3);

        const full = await materializeToDataTable(src, pool);           // via read()
        assert.strictEqual(full.numRows, dt.numRows);

        const order = Uint32Array.from([299, 0, 150, 5, 1, 42]);        // shuffled subset
        const gathered = await materializeToDataTable(permuteSource(src, order), pool); // via readRows()
        assert.strictEqual(gathered.numRows, order.length);

        for (const name of full.columnNames) {
            const e = full.getColumnByName(name).data;
            const g = gathered.getColumnByName(name).data;
            for (let j = 0; j < order.length; j++) {
                assert.strictEqual(g[j], e[order[j]], `column '${name}' out-row ${j}`);
            }
        }
        await src.close();
    });
});

describe('writeSogSource: indices ordering contract', () => {
    it('rejects a sub-length indices (must be a full-length ordering, not a subset filter)', async () => {
        const dt = createTestDataTable(200);
        const pool = createChunkDataPool();
        const src = dataTableToChunkSource(dt, pool.chunkSize);
        await assert.rejects(
            writeSogSource(src, pool,
                { filename: 'out.sog', bundle: false, iterations: 5, logging: 'silent', indices: new Uint32Array(100) },
                new MemoryFileSystem()),
            /full-length ordering|must equal the source's gaussian count/
        );
    });

    it('accepts a full-length ordering (a permutation of all rows)', async () => {
        const dt = createTestDataTable(200);
        const pool = createChunkDataPool();
        const src = dataTableToChunkSource(dt, pool.chunkSize);
        const order = new Uint32Array(dt.numRows);
        for (let i = 0; i < order.length; i++) order[i] = order.length - 1 - i; // reverse permutation
        const fs = new MemoryFileSystem();
        await writeSogSource(src, pool,
            { filename: 'out.sog', bundle: true, iterations: 5, logging: 'silent', indices: order }, fs);
        assert.ok(fs.results.get('out.sog'), 'wrote the bundled sog');
    });
});
