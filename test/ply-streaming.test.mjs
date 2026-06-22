/**
 * Streaming PLY pipeline tests: lazy PLY ChunkSource -> ops -> streaming PLY
 * writer.
 *
 *  - byte-identical to the legacy readPly -> processDataTable([translate]) ->
 *    writePly path (canonical-order input);
 *  - value round-trip with non-canonical input (the layer model canonicalises
 *    column order but preserves every value);
 *  - file-backed memory proof: peak pool footprint is one set of layer buffers,
 *    independent of scene size, with no temp files written.
 */

import assert from 'node:assert';
import { mkdtemp, writeFile as fsWriteFile, rm, readdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, it } from 'node:test';

import { Vec3 } from 'playcanvas';

import { encodePlyBinary } from './helpers/test-utils.mjs';
import { NodeReadFileSystem } from '../src/cli/node-file-system.js';
import {
    Column, DataTable, Transform,
    processDataTable, readPly, writePly,
    MemoryReadFileSystem, MemoryFileSystem
} from '../src/lib/index.js';
import { mapSource } from '../src/lib/ops/index.js';
import { readPlyChunked } from '../src/lib/readers/read-ply-chunked.js';
import { createChunkDataPool } from '../src/lib/source/index.js';
import { writePlyStreaming } from '../src/lib/writers/write-ply-streaming.js';

const SH_COEFFS = [0, 3, 8, 15];

// Build an all-float DataTable with columns already in canonical layer order
// (x,y,z, rot, scale, opacity, f_dc, f_rest). Deterministic values.
function makeCanonicalDataTable(count, shBands = 0) {
    const names = [
        'x', 'y', 'z',
        'rot_0', 'rot_1', 'rot_2', 'rot_3',
        'scale_0', 'scale_1', 'scale_2',
        'opacity',
        'f_dc_0', 'f_dc_1', 'f_dc_2'
    ];
    for (let k = 0; k < SH_COEFFS[shBands] * 3; k++) names.push(`f_rest_${k}`);

    const columns = names.map((name, ci) => {
        const data = new Float32Array(count);
        for (let i = 0; i < count; i++) {
            // varied, deterministic, exactly representable-ish values
            data[i] = Math.fround((i % 1000) * 0.25 - ci * 1.5 + (ci % 3) * 0.125);
        }
        return new Column(name, data);
    });
    return new DataTable(columns);
}

function byteEqual(a, b) {
    if (a.byteLength !== b.byteLength) return false;
    return Buffer.compare(
        Buffer.from(a.buffer, a.byteOffset, a.byteLength),
        Buffer.from(b.buffer, b.byteOffset, b.byteLength)
    ) === 0;
}

function sourceFromBytes(bytes, name = 'in.ply') {
    const rfs = new MemoryReadFileSystem();
    rfs.set(name, bytes);
    return rfs.createSource(name);
}

describe('streaming PLY pipeline (chunked read -> transform -> streaming write)', () => {
    it('is byte-identical to the legacy read -> process([translate]) -> write path', async () => {
        const dt = makeCanonicalDataTable(50, 1); // multi-chunk + short last at chunkSize 16
        const plyBytes = encodePlyBinary(dt);
        const translate = new Vec3(1.5, -2.0, 3.25);

        // legacy path
        const dtOld = await readPly(await sourceFromBytes(plyBytes));
        const dtProc = await processDataTable(dtOld, [{ kind: 'translate', value: translate.clone() }]);
        const oldFs = new MemoryFileSystem();
        await writePly({
            filename: 'out.ply',
            plyData: { comments: [], elements: [{ name: 'vertex', dataTable: dtProc }] }
        }, oldFs);
        const oldBytes = oldFs.results.get('out.ply');

        // new streaming path
        const pool = createChunkDataPool({ chunkSize: 16 });
        const src = await readPlyChunked(await sourceFromBytes(plyBytes), pool);
        const mapped = mapSource(src, new Transform(translate.clone()));
        const newFs = new MemoryFileSystem();
        await writePlyStreaming(mapped, pool, { filename: 'out.ply' }, newFs);
        await mapped.close();
        const newBytes = newFs.results.get('out.ply');

        assert.ok(byteEqual(newBytes, oldBytes), 'streaming output must be byte-identical to legacy output');
    });

    it('is byte-identical for a rotate (positions + quaternions + SH rotation)', async () => {
        const dt = makeCanonicalDataTable(50, 3); // SH3 exercises all SH-rotation bands
        const plyBytes = encodePlyBinary(dt);
        const euler = new Vec3(90, 30, 0);

        const dtOld = await readPly(await sourceFromBytes(plyBytes));
        const dtProc = await processDataTable(dtOld, [{ kind: 'rotate', value: euler.clone() }]);
        const oldFs = new MemoryFileSystem();
        await writePly({ filename: 'out.ply', plyData: { comments: [], elements: [{ name: 'vertex', dataTable: dtProc }] } }, oldFs);
        const oldBytes = oldFs.results.get('out.ply');

        const pool = createChunkDataPool({ chunkSize: 16 });
        const src = await readPlyChunked(await sourceFromBytes(plyBytes), pool);
        const mapped = mapSource(src, new Transform().fromEulers(euler.x, euler.y, euler.z));
        const newFs = new MemoryFileSystem();
        await writePlyStreaming(mapped, pool, { filename: 'out.ply' }, newFs);
        await mapped.close();

        assert.ok(byteEqual(newFs.results.get('out.ply'), oldBytes), 'rotate output must be byte-identical to legacy');
    });

    it('is byte-identical for a scale (positions + log-scales)', async () => {
        const dt = makeCanonicalDataTable(50, 0);
        const plyBytes = encodePlyBinary(dt);

        const dtOld = await readPly(await sourceFromBytes(plyBytes));
        const dtProc = await processDataTable(dtOld, [{ kind: 'scale', value: 2.0 }]);
        const oldFs = new MemoryFileSystem();
        await writePly({ filename: 'out.ply', plyData: { comments: [], elements: [{ name: 'vertex', dataTable: dtProc }] } }, oldFs);
        const oldBytes = oldFs.results.get('out.ply');

        const pool = createChunkDataPool({ chunkSize: 16 });
        const src = await readPlyChunked(await sourceFromBytes(plyBytes), pool);
        const mapped = mapSource(src, new Transform(undefined, undefined, 2.0));
        const newFs = new MemoryFileSystem();
        await writePlyStreaming(mapped, pool, { filename: 'out.ply' }, newFs);
        await mapped.close();

        assert.ok(byteEqual(newFs.results.get('out.ply'), oldBytes), 'scale output must be byte-identical to legacy');
    });

    it('preserves every value through the layer model (non-canonical input, no transform)', async () => {
        // test-utils order is NOT canonical; the chunk model canonicalises column
        // order but must preserve the value of every column.
        const { createTestDataTable } = await import('./helpers/test-utils.mjs');
        const dt = createTestDataTable(40, { includeSH: true, shBands: 2 });
        const plyBytes = encodePlyBinary(dt);

        const pool = createChunkDataPool({ chunkSize: 7 });
        const src = await readPlyChunked(await sourceFromBytes(plyBytes), pool);
        const fs = new MemoryFileSystem();
        await writePlyStreaming(src, pool, { filename: 'out.ply' }, fs);
        await src.close();

        const out = await readPly(await sourceFromBytes(fs.results.get('out.ply')));
        assert.strictEqual(out.numRows, dt.numRows);
        assert.deepStrictEqual([...out.columnNames].sort(), [...dt.columnNames].sort());
        for (const name of dt.columnNames) {
            assert.deepStrictEqual(out.getColumnByName(name).data, dt.getColumnByName(name).data, `column '${name}' value mismatch`);
        }
    });

    it('streams from disk with a bounded, scene-size-independent memory footprint', async () => {
        const dir = await mkdtemp(join(tmpdir(), 'ply-streaming-'));
        const inPath = join(dir, 'in.ply');
        try {
            const N = 500_000;
            const chunkSize = 65536;
            const dt = makeCanonicalDataTable(N, 0); // SH0: position 12 + geometric 32 + color 12 = 56 B/g
            await fsWriteFile(inPath, encodePlyBinary(dt));

            const pool = createChunkDataPool({ chunkSize });
            const rfs = new NodeReadFileSystem();
            const src = await readPlyChunked(await rfs.createSource(inPath), pool);
            const mapped = mapSource(src, new Transform(new Vec3(1, 2, 3)));
            const outFs = new MemoryFileSystem();
            await writePlyStreaming(mapped, pool, { filename: 'out.ply' }, outFs);
            await mapped.close();

            // Output is correct size.
            const outBytes = outFs.results.get('out.ply');
            const decoded = await readPly(await sourceFromBytes(outBytes));
            assert.strictEqual(decoded.numRows, N);

            // Memory proof: the pool only ever allocated ONE set of layer buffers
            // (position 12 + geometric 32 + color 12 = 56 bytes/gaussian × chunkSize),
            // reused across all chunks — independent of N. A non-streaming impl
            // would scale with N (here the full scene is ~28 MB vs ~3.7 MB pooled).
            assert.strictEqual(pool.bytesInUse, 0, 'all buffers released');
            assert.strictEqual(pool.bytesPooled, 56 * chunkSize, 'pool footprint is one set of layer buffers');

            // No temp files written by the pipeline (output went to memory).
            assert.deepStrictEqual(await readdir(dir), ['in.ply']);
        } finally {
            await rm(dir, { recursive: true, force: true });
        }
    });
});
