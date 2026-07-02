/**
 * Tests for the streaming stats implementation (computeStats / the `stats`
 * process action).
 *
 *  - the action passes the source through unchanged and emits the info block
 *    plus a stats table (text) or a columnar per-LOD `stats` array (JSON)
 *  - exact fields (min/max/mean/stdDev/nan/inf) match an exact oracle; the
 *    grouped median lands within its bin-width error bound
 *  - stats are per-LOD (no cross-LOD flattening) and per-column display
 *    transforms match the historical formatting
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

import { computeStatsView } from './helpers/summary-compare.mjs';
import { createTestDataTable } from './helpers/test-utils.mjs';
import { createChunkDataPool, createInMemoryChunkSource } from '../src/lib/chunk/index.js';
import { dataTableToChunkSource } from '../src/lib/compat/data-table.js';
import { Column, DataTable, Transform, computeStats, logger, processDataTable } from '../src/lib/index.js';
import { processSource } from '../src/lib/process-source.js';
import { forwardTransforms } from '../src/lib/value-transforms.js';

// Capture logger `output` events for the duration of `fn`.
const captureOutput = async (fn) => {
    const outputs = [];
    logger.setRenderer({ handle: (e) => e.kind === 'output' && outputs.push(e.text) });
    try {
        await fn();
    } finally {
        logger.setRenderer({ handle: () => {} });
    }
    return outputs;
};

// Exact reference statistics for a column's values (sorts a copy).
const exactStats = (values) => {
    const valid = [];
    let nanCount = 0;
    let infCount = 0;
    for (const v of values) {
        if (Number.isNaN(v)) nanCount++;
        else if (!Number.isFinite(v)) infCount++;
        else valid.push(v);
    }
    valid.sort((a, b) => a - b);
    const n = valid.length;
    const mean = valid.reduce((a, b) => a + b, 0) / n;
    const stdDev = Math.sqrt(valid.reduce((a, v) => a + (v - mean) ** 2, 0) / n);
    const median = n === 0 ? NaN : (n % 2 === 0 ? (valid[n / 2 - 1] + valid[n / 2]) / 2 : valid[n >> 1]);
    return { min: valid[0], max: valid[n - 1], median, mean, stdDev, nanCount, infCount };
};

// Position-only DataTable with the given x values (y/z zero).
const xTable = (xValues) => {
    const n = xValues.length;
    return new DataTable([
        new Column('x', Float32Array.from(xValues)),
        new Column('y', new Float32Array(n)),
        new Column('z', new Float32Array(n))
    ]);
};

describe('stats process action', () => {
    it('passes the source through unchanged and emits the text shape', async () => {
        const pool = createChunkDataPool();
        const src = dataTableToChunkSource(createTestDataTable(20, { includeSH: true, shBands: 1 }));
        let out;
        const outputs = await captureOutput(async () => {
            out = await processSource(src, [{ kind: 'stats' }], pool);
        });
        assert.strictEqual(out, src); // pass-through
        assert.strictEqual(outputs.length, 1);
        const text = outputs[0];
        assert.doesNotMatch(text, /# Summary|# File info|Row Count/);
        assert.match(text, /^gaussian: yes\n/);
        assert.match(text, /gaussians: 20/);
        assert.match(text, /\n\n\| Column +\| min +\|/); // blank line, then the table
        const tableRows = text.split('\n').filter(l => l.startsWith('|')).length;
        assert.strictEqual(tableRows, 2 + 14 + 9); // header + separator + standard cols + f_rest_0..8
    });

    it('emits JSON as a superset of info JSON with aligned columnar arrays', async () => {
        const pool = createChunkDataPool();
        const src = dataTableToChunkSource(createTestDataTable(20, { includeSH: true, shBands: 1 }));
        const outputs = await captureOutput(async () => {
            await processSource(src, [{ kind: 'info', format: 'json' }, { kind: 'stats', format: 'json' }], pool);
        });
        const info = JSON.parse(outputs[0]);
        const stats = JSON.parse(outputs[1]);
        const { stats: lods, ...head } = stats;
        assert.deepStrictEqual(head, info); // stats ⊃ info
        assert.strictEqual(lods.length, 1);
        assert.strictEqual(lods[0].lod, 0);
        assert.strictEqual(lods[0].numGaussians, 20);
        assert.deepStrictEqual(lods[0].columns, info.columns);
        for (const field of ['min', 'max', 'median', 'mean', 'stdDev', 'nanCount', 'infCount', 'histogram']) {
            assert.strictEqual(lods[0][field].length, info.columns.length, `${field} length`);
        }
        for (const counts of lods[0].histogram) {
            assert.strictEqual(counts.length, 16);
        }
    });

    it('bridges through processDataTable and leaves the table unchanged', async () => {
        const dt = createTestDataTable(20);
        const outputs = await captureOutput(() => processDataTable(dt, [{ kind: 'stats', format: 'json' }]));
        assert.strictEqual(outputs.length, 1);
        const json = JSON.parse(outputs[0]);
        assert.strictEqual(json.gaussian, true);
        assert.strictEqual(json.stats.length, 1);
        assert.strictEqual(dt.numRows, 20);
    });
});

describe('computeStats accuracy', () => {
    it('matches an exact oracle per column (median within bin error)', async () => {
        const dt = createTestDataTable(300, { includeSH: true, shBands: 1 });
        const { lods } = await computeStats(dt);
        const lod = lods[0];
        assert.strictEqual(lods.length, 1);
        assert.strictEqual(lod.numGaussians, 300);
        for (let i = 0; i < lod.columns.length; i++) {
            const name = lod.columns[i];
            const expected = exactStats(dt.getColumnByName(name).data);
            assert.ok(Math.abs(lod.min[i] - expected.min) <= 1e-5, `${name}.min`);
            assert.ok(Math.abs(lod.max[i] - expected.max) <= 1e-5, `${name}.max`);
            assert.ok(Math.abs(lod.mean[i] - expected.mean) <= 1e-5 * (1 + Math.abs(expected.mean)), `${name}.mean`);
            assert.ok(Math.abs(lod.stdDev[i] - expected.stdDev) <= 1e-5 * (1 + expected.stdDev), `${name}.stdDev`);
            assert.strictEqual(lod.nanCount[i], expected.nanCount, `${name}.nanCount`);
            assert.strictEqual(lod.infCount[i], expected.infCount, `${name}.infCount`);
            const medianTolerance = Math.max((expected.max - expected.min) / 64, 1e-6);
            assert.ok(Math.abs(lod.median[i] - expected.median) <= medianTolerance, `${name}.median (${lod.median[i]} vs ${expected.median})`);
            assert.strictEqual(lod.histogram[i].reduce((a, b) => a + b, 0), 300 - expected.nanCount - expected.infCount, `${name}.histogram total`);
        }
    });

    it('survives histogram range expansion (clustered seed, distant outliers)', async () => {
        const values = [];
        for (let i = 0; i < 1000; i++) values.push((i % 97) * 0.00001);
        values.push(1000, -1000, 500, -500);
        const dt = xTable(values);
        const { lods } = await computeStats(dt);
        const i = lods[0].columns.indexOf('x');
        const expected = exactStats(dt.getColumnByName('x').data);
        assert.strictEqual(lods[0].min[i], -1000);
        assert.strictEqual(lods[0].max[i], 1000);
        assert.ok(Math.abs(lods[0].mean[i] - expected.mean) <= 1e-5 * (1 + Math.abs(expected.mean)), 'mean');
        assert.ok(Math.abs(lods[0].median[i] - expected.median) <= (expected.max - expected.min) / 64, 'median');
        assert.strictEqual(lods[0].histogram[i].reduce((a, b) => a + b, 0), values.length);
    });

    it('reports stats per LOD without cross-LOD flattening', async () => {
        const positions = (values) => {
            const arr = new Float32Array(values.length * 3);
            values.forEach((v, i) => {
                arr[i * 3] = v;
            });
            return arr.buffer;
        };
        const src = createInMemoryChunkSource({
            numGaussians: 4,
            chunkSize: 1024,
            shBands: 0,
            transform: new Transform(),
            lodCounts: [4, 2],
            position: [[positions([0, 1, 2, 3])], [positions([100, 101])]]
        });
        const { lods } = await computeStats(src);
        assert.strictEqual(lods.length, 2);
        const x0 = lods[0].columns.indexOf('x');
        assert.strictEqual(lods[0].numGaussians, 4);
        assert.strictEqual(lods[0].min[x0], 0);
        assert.strictEqual(lods[0].max[x0], 3);
        assert.strictEqual(lods[1].numGaussians, 2);
        assert.strictEqual(lods[1].min[x0], 100);
        assert.strictEqual(lods[1].max[x0], 101);

        // Text output renders one labelled table per LOD.
        const pool = createChunkDataPool();
        const outputs = await captureOutput(() => processSource(src, [{ kind: 'stats' }], pool));
        assert.match(outputs[0], /lod 0: 4 gaussians/);
        assert.match(outputs[0], /lod 1: 2 gaussians/);
    });

    it('applies display transforms in JSON output (opacity/scale/f_dc)', async () => {
        const dt = createTestDataTable(50);
        const { lods } = await computeStats(dt);
        const outputs = await captureOutput(() => processDataTable(dt, [{ kind: 'stats', format: 'json' }]));
        const json = JSON.parse(outputs[0]);
        for (const name of ['opacity', 'scale_0', 'f_dc_0']) {
            const i = lods[0].columns.indexOf(name);
            const expected = +forwardTransforms[name](lods[0].mean[i]).toPrecision(6);
            assert.strictEqual(json.stats[0].mean[i], expected, `${name} display mean`);
        }
        // Untransformed columns pass through raw.
        const xi = lods[0].columns.indexOf('x');
        assert.strictEqual(json.stats[0].mean[xi], lods[0].mean[xi]);
    });

    it('accepts a DataTable and a ChunkSource with identical results', async () => {
        const dt = createTestDataTable(100, { includeSH: true, shBands: 1 });
        const fromTable = await computeStats(dt);
        const src = dataTableToChunkSource(dt);
        const fromSource = await computeStats(src);
        await src.close();
        assert.deepStrictEqual(fromTable, fromSource);
    });
});

describe('computeStats edge cases', () => {
    it('handles a constant column (never seeds the histogram)', async () => {
        const dt = xTable(new Array(10).fill(5));
        const view = await computeStatsView(dt);
        const x = view.columns.x;
        assert.strictEqual(x.min, 5);
        assert.strictEqual(x.max, 5);
        assert.strictEqual(x.median, 5);
        assert.strictEqual(x.mean, 5);
        assert.strictEqual(x.stdDev, 0);
        const expected = new Array(16).fill(0);
        expected[8] = 10;
        assert.deepStrictEqual(x.histogram, expected);
    });

    it('handles an all-NaN column (NaN stats, empty histogram, JSON nulls)', async () => {
        const dt = xTable(new Array(8).fill(NaN));
        const view = await computeStatsView(dt);
        const x = view.columns.x;
        assert.ok(Number.isNaN(x.min) && Number.isNaN(x.max) && Number.isNaN(x.median) && Number.isNaN(x.mean) && Number.isNaN(x.stdDev));
        assert.strictEqual(x.nanCount, 8);
        assert.deepStrictEqual(x.histogram, new Array(16).fill(0));

        const outputs = await captureOutput(() => processDataTable(dt, [{ kind: 'stats', format: 'json' }]));
        const json = JSON.parse(outputs[0]);
        const i = json.stats[0].columns.indexOf('x');
        assert.strictEqual(json.stats[0].min[i], null);
        assert.strictEqual(json.stats[0].median[i], null);
    });

    it('counts Infinity separately and excludes it from min/max', async () => {
        const dt = xTable([Infinity, -Infinity, 1, 2, 3]);
        const view = await computeStatsView(dt);
        const x = view.columns.x;
        assert.strictEqual(x.infCount, 2);
        assert.strictEqual(x.nanCount, 0);
        assert.strictEqual(x.min, 1);
        assert.strictEqual(x.max, 3);
    });

    it('reads uint32 extra columns through a u32 view', async () => {
        const n = 6;
        const dt = new DataTable([
            new Column('x', new Float32Array(n)),
            new Column('y', new Float32Array(n)),
            new Column('z', new Float32Array(n)),
            new Column('ids', Uint32Array.from([10, 20, 30, 40, 50, 60]))
        ]);
        const view = await computeStatsView(dt);
        assert.strictEqual(view.columns.ids.min, 10);
        assert.strictEqual(view.columns.ids.max, 60);
        assert.strictEqual(view.columns.ids.mean, 35);
    });

    it('reports a non-gaussian source honestly', async () => {
        const dt = xTable([1, 2, 3]);
        const outputs = await captureOutput(() => processDataTable(dt, [{ kind: 'stats', format: 'json' }]));
        const json = JSON.parse(outputs[0]);
        assert.strictEqual(json.gaussian, false);
        assert.deepStrictEqual(json.stats[0].columns, ['x', 'y', 'z']);
    });
});
