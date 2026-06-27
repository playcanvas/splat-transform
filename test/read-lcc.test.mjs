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
import { readLcc, readLccSource } from '../src/lib/readers/read-lcc.js';
import { materializeToDataTable } from '../src/lib/compat/data-table.js';
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
});
