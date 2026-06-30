/**
 * Tests for the LCC reader.
 *
 * LCC has no writer, so fixtures are hand-packed: a meta.lcc + index.bin (the
 * quadtree unit table) + data.bin (32-byte splat records) + shcoef.bin (64-byte
 * SH records). The A/B verifies the chunked `readLccSource` decodes identically
 * to the eager `readLcc` (both share parseIndexBin/processUnit, so this pins the
 * chunked refactor regardless of fixture realism).
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

import { MemoryReadFileSystem } from '../src/lib/index.js';
import { readLcc, readLccSource, readLccEnvironmentSource } from '../src/lib/readers/read-lcc.js';
import { materializeToDataTable } from '../src/lib/compat/data-table.js';
import { permuteSource } from '../src/lib/ops/index.js';
import { createChunkDataPool } from '../src/lib/source/index.js';

// One quadtree unit, two LODs: LOD 0 = 3 splats, LOD 1 = 2 splats. data.bin
// lays the 5 splats out contiguously (LOD 0 at offset 0, LOD 1 at offset 96);
// shcoef.bin mirrors at 2× offsets (64 bytes/splat). Values are arbitrary but
// distinct so a row-mapping bug would surface.
const makeLccFixture = ({ sh = true } = {}) => {
    const lod0 = 3, lod1 = 2;
    const n = lod0 + lod1;

    const dataBin = new Uint8Array(n * 32);
    const dv = new DataView(dataBin.buffer);
    for (let i = 0; i < n; i++) {
        const o = i * 32;
        dv.setFloat32(o + 0, i + 0.1, true);
        dv.setFloat32(o + 4, i + 0.2, true);
        dv.setFloat32(o + 8, i + 0.3, true);
        dv.setUint8(o + 12, (i * 10) & 255);
        dv.setUint8(o + 13, (i * 20) & 255);
        dv.setUint8(o + 14, (i * 30) & 255);
        dv.setUint8(o + 15, 100 + i);
        dv.setUint16(o + 16, i * 1000, true);
        dv.setUint16(o + 18, i * 1100, true);
        dv.setUint16(o + 20, i * 1200, true);
        dv.setUint16(o + 22, 12345, true); // rot lo
        dv.setUint16(o + 24, 6789, true);  // rot hi
        dv.setUint16(o + 26, i + 1, true); // nx
        dv.setUint16(o + 28, i + 2, true); // ny
        dv.setUint16(o + 30, i + 3, true); // nz
    }

    let shcoefBin = new Uint8Array(0);
    if (sh) {
        shcoefBin = new Uint8Array(n * 64);
        const sv = new DataView(shcoefBin.buffer);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < 15; j++) sv.setUint32(i * 64 + j * 4, (i * 100 + j) >>> 0, true);
        }
    }

    // index.bin: int16 x, int16 y, then per LOD (int32 points, int64 offset, int32 size).
    const indexBin = new Uint8Array(2 + 2 + 2 * 16);
    const iv = new DataView(indexBin.buffer);
    iv.setInt16(0, 0, true);
    iv.setInt16(2, 0, true);
    iv.setInt32(4, lod0, true); iv.setBigInt64(8, 0n, true); iv.setInt32(16, lod0 * 32, true);
    iv.setInt32(20, lod1, true); iv.setBigInt64(24, BigInt(lod0 * 32), true); iv.setInt32(32, lod1 * 32, true);

    const meta = {
        fileType: sh ? 'Quality' : 'Portable',
        attributes: [
            { name: 'scale', min: [-5, -5, -5], max: [5, 5, 5] },
            { name: 'shcoef', min: [-1, -1, -1], max: [1, 1, 1] }
        ],
        splats: [lod0, lod1],
        totalLevel: 2
    };

    const fs = new MemoryReadFileSystem();
    fs.set('meta.lcc', new TextEncoder().encode(JSON.stringify(meta)));
    fs.set('index.bin', indexBin);
    fs.set('data.bin', dataBin);
    if (sh) fs.set('shcoef.bin', shcoefBin);
    return fs;
};

const opts = (lodSelect = []) => ({ lodSelect });

describe('readLccSource (chunked) vs readLcc (eager)', () => {
    for (const sh of [true, false]) {
        it(`flatten(chunked) matches merged(eager) row-for-row${sh ? ' with SH' : ' (no SH)'}`, async () => {
            const eager = (await readLcc(makeLccFixture({ sh }), 'meta.lcc', opts()))[0];

            const pool = createChunkDataPool();
            const src = await readLccSource(makeLccFixture({ sh }), 'meta.lcc', opts(), pool);
            assert.strictEqual(src.meta.numLods, 2);
            assert.deepStrictEqual([...src.meta.lodCounts], [3, 2]);

            const flat = await materializeToDataTable(src, pool);
            await src.close();

            assert.strictEqual(flat.numRows, eager.numRows);
            assert.ok(!flat.hasColumn('lod'), 'chunked source carries no lod column (LOD is structural)');
            for (const name of eager.columnNames) {
                if (name === 'lod') continue;
                const e = eager.getColumnByName(name).data;
                const f = flat.getColumnByName(name).data;
                for (let i = 0; i < eager.numRows; i++) {
                    assert.strictEqual(f[i], e[i], `column '${name}' row ${i}`);
                }
            }
        });
    }

    it('reads a single selected LOD as a one-LOD source', async () => {
        const pool = createChunkDataPool();
        const src = await readLccSource(makeLccFixture(), 'meta.lcc', opts([1]), pool);
        assert.strictEqual(src.meta.numLods, 1);
        assert.deepStrictEqual([...src.meta.lodCounts], [2]); // LOD 1 = 2 splats
        await src.close();
    });

    it('readRows gathers from a non-zero structural LOD (request.lod)', async () => {
        // Eager LOD 1 (2 splats) as the oracle; chunked source carries both LODs.
        const eager1 = (await readLcc(makeLccFixture(), 'meta.lcc', opts([1])))[0];

        const pool = createChunkDataPool();
        const src = await readLccSource(makeLccFixture(), 'meta.lcc', opts(), pool);
        assert.strictEqual(src.meta.numLods, 2);

        const count = 2;
        const idx = new Uint32Array([1, 0]); // shuffled within LOD 1
        const acq = {};
        for (const layer of ['position', 'geometric', 'color', 'other']) {
            acq[layer] = pool.acquire(layer, src.meta.layouts[layer], count);
        }
        await src.read({ indices: idx, indexOffset: 0, count, lod: 1, ...acq });

        const pos = acq.position.field('position'); // count × 3
        const op = acq.geometric.field('opacity');  // count
        const dc = acq.color.field('dc');            // count × 3
        const nx = acq.other.field('nx');            // count
        for (let j = 0; j < count; j++) {
            const e = idx[j];
            assert.strictEqual(pos[j * 3], eager1.getColumnByName('x').data[e], `x out-row ${j}`);
            assert.strictEqual(op[j], eager1.getColumnByName('opacity').data[e], `opacity out-row ${j}`);
            assert.strictEqual(dc[j * 3], eager1.getColumnByName('f_dc_0').data[e], `f_dc_0 out-row ${j}`);
            assert.strictEqual(nx[j], eager1.getColumnByName('nx').data[e], `nx out-row ${j}`);
        }
        for (const layer of ['position', 'geometric', 'color', 'other']) acq[layer].release();
        await src.close();
    });
});

// Write one 32-byte data record (+ 64-byte SH record) with values derived from
// `seed`, so each global row is distinct and a mapping bug surfaces. Mirrors the
// value scheme of makeLccFixture's records.
const putRecord = (dv, db, sv, sb, seed, sh) => {
    dv.setFloat32(db + 0, seed + 0.1, true);
    dv.setFloat32(db + 4, seed + 0.2, true);
    dv.setFloat32(db + 8, seed + 0.3, true);
    dv.setUint8(db + 12, (seed * 10) & 255);
    dv.setUint8(db + 13, (seed * 20) & 255);
    dv.setUint8(db + 14, (seed * 30) & 255);
    dv.setUint8(db + 15, 100 + seed);
    dv.setUint16(db + 16, seed * 1000, true);
    dv.setUint16(db + 18, seed * 1100, true);
    dv.setUint16(db + 20, seed * 1200, true);
    dv.setUint16(db + 22, 12345, true);
    dv.setUint16(db + 24, 6789, true);
    dv.setUint16(db + 26, seed + 1, true);
    dv.setUint16(db + 28, seed + 2, true);
    dv.setUint16(db + 30, seed + 3, true);
    if (sh) for (let j = 0; j < 15; j++) sv.setUint32(sb + j * 4, (seed * 100 + j) >>> 0, true);
};

// Two quadtree units, one LOD. Unit A (3 splats, global 0..2) is placed at data
// byte 192; unit B (4 splats, global 3..6) at byte 32 — non-contiguous and out of
// index order, so the global gaussian order (index.bin order: A then B) differs
// from the byte order. This exercises the per-unit offset map, a chunk straddling
// a unit boundary, and readRows' byte-offset sort/coalesce. shcoef.bin mirrors at
// 2× offsets. Row values key off the GLOBAL index, so eager and chunked agree.
const makeMultiUnitFixture = ({ sh = true } = {}) => {
    const aPts = 3, bPts = 4, n = aPts + bPts;
    const aOff = 192, bOff = 32;
    const dataBin = new Uint8Array(aOff + aPts * 32); // 288 bytes (gaps at 0..32, 160..192)
    const dv = new DataView(dataBin.buffer);
    let shcoefBin = new Uint8Array(0), sv = null;
    if (sh) {
        shcoefBin = new Uint8Array((aOff + aPts * 32) * 2);
        sv = new DataView(shcoefBin.buffer);
    }
    for (let r = 0; r < aPts; r++) putRecord(dv, aOff + r * 32, sv, (aOff + r * 32) * 2, r, sh);
    for (let r = 0; r < bPts; r++) putRecord(dv, bOff + r * 32, sv, (bOff + r * 32) * 2, aPts + r, sh);

    // index.bin: 2 units, totalLevel=1 (record = int16 x,y + int32 points, int64 offset, int32 size).
    const indexBin = new Uint8Array(2 * (4 + 16));
    const iv = new DataView(indexBin.buffer);
    iv.setInt16(0, 0, true); iv.setInt16(2, 0, true);
    iv.setInt32(4, aPts, true); iv.setBigInt64(8, BigInt(aOff), true); iv.setInt32(16, aPts * 32, true);
    iv.setInt16(20, 1, true); iv.setInt16(22, 0, true);
    iv.setInt32(24, bPts, true); iv.setBigInt64(28, BigInt(bOff), true); iv.setInt32(36, bPts * 32, true);

    const meta = {
        fileType: sh ? 'Quality' : 'Portable',
        attributes: [
            { name: 'scale', min: [-5, -5, -5], max: [5, 5, 5] },
            { name: 'shcoef', min: [-1, -1, -1], max: [1, 1, 1] }
        ],
        splats: [n],
        totalLevel: 1
    };

    const fs = new MemoryReadFileSystem();
    fs.set('meta.lcc', new TextEncoder().encode(JSON.stringify(meta)));
    fs.set('index.bin', indexBin);
    fs.set('data.bin', dataBin);
    if (sh) fs.set('shcoef.bin', shcoefBin);
    return fs;
};

describe('readLccSource: multi-unit, non-contiguous offsets', () => {
    for (const sh of [true, false]) {
        it(`A/B with a chunk straddling a unit boundary${sh ? ' with SH' : ' (no SH)'}`, async () => {
            const eager = (await readLcc(makeMultiUnitFixture({ sh }), 'meta.lcc', opts()))[0];

            const pool = createChunkDataPool({ chunkSize: 4 }); // chunk 0 = A0,A1,A2,B0 (straddle)
            const src = await readLccSource(makeMultiUnitFixture({ sh }), 'meta.lcc', opts(), pool);
            assert.strictEqual(src.meta.numGaussians, 7);
            assert.deepStrictEqual([...src.meta.numChunks], [2]);

            const flat = await materializeToDataTable(src, pool);
            await src.close();

            assert.strictEqual(flat.numRows, eager.numRows);
            for (const name of eager.columnNames) {
                if (name === 'lod') continue;
                const e = eager.getColumnByName(name).data;
                const f = flat.getColumnByName(name).data;
                for (let i = 0; i < eager.numRows; i++) {
                    assert.strictEqual(f[i], e[i], `column '${name}' row ${i}`);
                }
            }
        });
    }

    it('readRows gathers shuffled cross-unit rows (via permuteSource)', async () => {
        const eager = (await readLcc(makeMultiUnitFixture(), 'meta.lcc', opts()))[0];

        const pool = createChunkDataPool({ chunkSize: 4 });
        const src = await readLccSource(makeMultiUnitFixture(), 'meta.lcc', opts(), pool);

        const order = new Uint32Array([5, 0, 3, 6, 1, 2, 4]); // shuffled, spans both units
        const gathered = await materializeToDataTable(permuteSource(src, order), pool);
        await src.close();

        assert.strictEqual(gathered.numRows, order.length);
        for (const name of eager.columnNames) {
            if (name === 'lod') continue;
            const e = eager.getColumnByName(name).data;
            const g = gathered.getColumnByName(name).data;
            for (let j = 0; j < order.length; j++) {
                assert.strictEqual(g[j], e[order[j]], `column '${name}' out-row ${j}`);
            }
        }
    });

    it('pool peak is one set of layer buffers (sub-block bounded, scene-size independent)', async () => {
        const chunkSize = 4;
        const pool = createChunkDataPool({ chunkSize });
        const src = await readLccSource(makeMultiUnitFixture({ sh: true }), 'meta.lcc', opts(), pool);
        const flat = await materializeToDataTable(src, pool);
        await src.close();

        assert.strictEqual(flat.numRows, 7);
        assert.strictEqual(pool.bytesInUse, 0, 'all buffers released');
        // position(12) + geometric(32) + color(3+45 SH coeffs -> 192) + other(nx,ny,nz -> 12).
        const expected = (12 + 32 + (3 + 45) * 4 + 12) * chunkSize;
        assert.strictEqual(pool.bytesPooled, expected, 'pool footprint is one set of layer buffers');
    });

    it('exposes the other layer (normals) for both SH and no-SH', async () => {
        for (const sh of [true, false]) {
            const pool = createChunkDataPool();
            const src = await readLccSource(makeMultiUnitFixture({ sh }), 'meta.lcc', opts(), pool);
            assert.ok(src.meta.availableLayers.has('other'));
            assert.deepStrictEqual(src.meta.extraColumns.map(e => e.name), ['nx', 'ny', 'nz']);
            assert.strictEqual(src.meta.shBands, sh ? 3 : 0);
            await src.close();
        }
    });
});

describe('readLccEnvironmentSource', () => {
    it('returns null when there is no environment.bin', async () => {
        const pool = createChunkDataPool();
        const env = await readLccEnvironmentSource(makeLccFixture({ sh: false }), 'meta.lcc', pool);
        assert.strictEqual(env, null);
    });

    it('decodes environment.bin into a resident source', async () => {
        const fs = makeLccFixture({ sh: false }); // Portable -> 32-byte env records
        const envBin = new Uint8Array(2 * 32);
        const ev = new DataView(envBin.buffer);
        for (let i = 0; i < 2; i++) {
            const o = i * 32;
            ev.setFloat32(o + 0, i + 0.5, true);
            ev.setFloat32(o + 4, i + 1.5, true);
            ev.setFloat32(o + 8, i + 2.5, true);
            ev.setUint8(o + 15, 128); // opacity
            ev.setUint16(o + 16, 30000, true);
            ev.setUint16(o + 18, 30000, true);
            ev.setUint16(o + 20, 30000, true);
        }
        fs.set('environment.bin', envBin);

        const pool = createChunkDataPool();
        const env = await readLccEnvironmentSource(fs, 'meta.lcc', pool);
        assert.ok(env, 'environment source should be present');
        assert.strictEqual(env.meta.numGaussians, 2);
        assert.ok(env.meta.availableLayers.has('position'));
        await env.close();
    });
});
