/**
 * Tests for the lod-meta.json file contract emitted by writeLod.
 */

import assert from 'node:assert';
import { dirname, join } from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';

import { Column, DataTable, Transform, WebPCodec } from '../src/lib/index.js';
import { MemoryFileSystem } from '../src/lib/io/write/index.js';
import { writeLod } from '../src/lib/writers/write-lod.js';
import { version } from '../src/lib/version.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
WebPCodec.wasmUrl = join(__dirname, '..', 'lib', 'webp.wasm');

// Build a minimal splat table (no SH, so no GPU device is needed) with one
// row per entry of `lods`. Constructed directly in PLY space so writeLod's
// convertToSpace is a no-op.
const makeTable = (lods) => {
    const n = lods.length;
    const fill = (value) => new Float32Array(n).fill(value);
    const ramp = (scale) => new Float32Array(Array.from({ length: n }, (_, i) => i * scale));

    return new DataTable([
        new Column('x', ramp(1)),
        new Column('y', ramp(0.5)),
        new Column('z', ramp(0.25)),
        new Column('rot_0', fill(1)),
        new Column('rot_1', fill(0)),
        new Column('rot_2', fill(0)),
        new Column('rot_3', fill(0)),
        new Column('scale_0', fill(-3)),
        new Column('scale_1', fill(-3)),
        new Column('scale_2', fill(-3)),
        new Column('f_dc_0', fill(0)),
        new Column('f_dc_1', fill(0)),
        new Column('f_dc_2', fill(0)),
        new Column('opacity', fill(0)),
        new Column('lod', new Float32Array(lods))
    ], Transform.PLY);
};

const writeScene = async (lods, envRows) => {
    const fs = new MemoryFileSystem();
    await writeLod({
        filename: '/scene/lod-meta.json',
        dataTable: makeTable(lods),
        envDataTable: envRows > 0 ? makeTable(new Array(envRows).fill(0)) : null,
        iterations: 1,
        chunkCount: 1,
        chunkExtent: 16
    }, fs);
    const meta = JSON.parse(new TextDecoder().decode(fs.results.get('/scene/lod-meta.json')));
    return { fs, meta };
};

describe('writeLod', function () {
    it('writes lod-meta.json header fields and chunk references', async function () {
        const { fs, meta } = await writeScene([0, 0, 0, 1, 1], 0);

        assert.strictEqual(meta.version, 1);
        assert.strictEqual(meta.asset.generator, `splat-transform v${version}`);
        assert.strictEqual(meta.count, 5);
        assert.deepStrictEqual(meta.counts, [3, 2]);
        assert.strictEqual(meta.lodLevels, 2);
        assert.ok(!('environment' in meta), 'environment should be omitted when there are no environment splats');
        assert.deepStrictEqual([...meta.filenames].sort(), ['0_0/meta.json', '1_0/meta.json']);

        // single small chunk: the tree is one leaf referencing both lod levels
        assert.strictEqual(meta.filenames[meta.tree.lods['0'].file], '0_0/meta.json');
        assert.strictEqual(meta.filenames[meta.tree.lods['1'].file], '1_0/meta.json');
        assert.deepStrictEqual(
            { offset: meta.tree.lods['0'].offset, count: meta.tree.lods['0'].count },
            { offset: 0, count: 3 }
        );
        assert.deepStrictEqual(
            { offset: meta.tree.lods['1'].offset, count: meta.tree.lods['1'].count },
            { offset: 0, count: 2 }
        );

        // referenced chunk SOGs are written
        assert.ok(fs.results.has('/scene/0_0/meta.json'));
        assert.ok(fs.results.has('/scene/1_0/meta.json'));
    });

    it('references the environment SOG when environment splats are present', async function () {
        const { fs, meta } = await writeScene([0, 0, 0], 2);

        assert.strictEqual(meta.environment, 'env/meta.json');
        assert.ok(fs.results.has('/scene/env/meta.json'));
    });
});
