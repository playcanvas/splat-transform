/**
 * SOG v3 (default) vs v2: rotation encodings, opacity codebook with excess
 * opacity above 1, blind data, and what v2 drops.
 */

import assert from 'node:assert';
import { dirname, join } from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';

import { createTestDataTable, encodePlyBinary } from './helpers/test-utils.mjs';
import { createChunkDataPool } from '../src/lib/chunk/index.js';
import { materializeToDataTable } from '../src/lib/compat/data-table.js';
import { readFile as readRawFile } from '../src/lib/io/read/index.js';
import {
    Column, logger,
    MemoryFileSystem, MemoryReadFileSystem, ZipReadFileSystem, WebPCodec
} from '../src/lib/index.js';
import { readPly } from '../src/lib/readers/read-ply.js';
import { readSogSource } from '../src/lib/readers/read-sog.js';
import { writeSogSource } from '../src/lib/writers/write-sog.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
WebPCodec.wasmUrl = join(__dirname, '..', 'lib', 'webp.wasm');

// mulberry32: deterministic, full-period
const rng = (seed) => () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

// Uniform random unit quaternions (normalised 4D gaussians).
const setRandomRotations = (dt, seed) => {
    const r = rng(seed);
    const g = () => Math.sqrt(-2 * Math.log(Math.max(r(), 1e-300))) * Math.cos(2 * Math.PI * r());
    const cols = [0, 1, 2, 3].map((k) => dt.getColumnByName(`rot_${k}`).data);
    for (let i = 0; i < dt.numRows; i++) {
        const q = [g(), g(), g(), g()];
        const l = Math.hypot(...q);
        for (let k = 0; k < 4; k++) cols[k][i] = q[k] / l;
    }
};

const plySource = async (dt, pool) => {
    const rfs = new MemoryReadFileSystem();
    rfs.set('in.ply', encodePlyBinary(dt));
    return readPly(await rfs.createSource('in.ply'), pool);
};

// Write a SOG bundle and return { meta, table } read back through our reader.
const roundTrip = async (dt, options = {}) => {
    const pool = createChunkDataPool();
    const src = await plySource(dt, pool);
    const fs = new MemoryFileSystem();
    await writeSogSource(src, pool, { filename: 'out.sog', bundle: true, iterations: 3, logging: 'silent', ...options }, fs);
    const rfs = new MemoryReadFileSystem();
    rfs.set('out.sog', fs.results.get('out.sog'));
    const zip = new ZipReadFileSystem(await rfs.createSource('out.sog'));
    const meta = JSON.parse(new TextDecoder().decode(await readRawFile(zip, 'meta.json')));
    const sog = await readSogSource(zip, 'meta.json', pool);
    const table = await materializeToDataTable(sog, pool);
    await sog.close();
    return { meta, table };
};

// Rotation error per gaussian in degrees. Rows are matched by position, since
// the writer reorders by Morton code.
const rotationErrors = (src, out) => {
    const key = (t, i) => ['x', 'y', 'z'].map((c) => Math.round(2 * t.getColumnByName(c).data[i])).join(',');
    const byKey = new Map();
    for (let i = 0; i < src.numRows; i++) byKey.set(key(src, i), i);
    const errs = [];
    for (let j = 0; j < out.numRows; j++) {
        const i = byKey.get(key(out, j));
        let d = 0;
        for (let k = 0; k < 4; k++) d += src.getColumnByName(`rot_${k}`).data[i] * out.getColumnByName(`rot_${k}`).data[j];
        const l = Math.sqrt([0, 1, 2, 3].reduce((a, k) => a + out.getColumnByName(`rot_${k}`).data[j] ** 2, 0));
        errs.push((2 * Math.acos(Math.min(1, Math.abs(d) / l)) * 180) / Math.PI);
    }
    return errs;
};

const stats = (errs) => ({
    max: Math.max(...errs),
    rms: Math.sqrt(errs.reduce((a, e) => a + e * e, 0) / errs.length)
});

const captureWarnings = async (fn) => {
    const warnings = [];
    logger.setRenderer({ handle: (e) => e.kind === 'message' && e.level === 'warn' && warnings.push(e.text) });
    try {
        await fn();
    } finally {
        logger.setRenderer({ handle: () => {} });
    }
    return warnings;
};

describe('SOG rotation encodings', () => {
    it('v2 rounds its smallest-three components (max < 1.0°, rms < 0.45°)', async () => {
        const dt = createTestDataTable(4096);
        setRandomRotations(dt, 1);
        const { meta, table } = await roundTrip(dt, { version: 2 });
        assert.strictEqual(meta.version, 2);
        assert.ok(meta.quats, 'v2 keeps quats.webp');
        const { max, rms } = stats(rotationErrors(dt, table));
        // truncating writers measure ~1.8° / 0.71° here
        assert.ok(max < 1.0, `max ${max}`);
        assert.ok(rms < 0.45, `rms ${rms}`);
    });

    it('v3 exp map (max < 1.25°, rms < 0.65°), no quats texture', async () => {
        const dt = createTestDataTable(4096);
        setRandomRotations(dt, 2);
        const { meta, table } = await roundTrip(dt);
        assert.strictEqual(meta.version, 3);
        assert.strictEqual(meta.quats, undefined);
        const { max, rms } = stats(rotationErrors(dt, table));
        assert.ok(max < 1.25, `max ${max}`);
        assert.ok(rms < 0.65, `rms ${rms}`);
    });

    it('v3 represents the identity rotation exactly', async () => {
        const dt = createTestDataTable(64);
        for (let k = 0; k < 4; k++) dt.getColumnByName(`rot_${k}`).data.fill(k === 0 ? 1 : 0);
        const { table } = await roundTrip(dt);
        for (let j = 0; j < table.numRows; j++) {
            assert.strictEqual(table.getColumnByName('rot_0').data[j], 1);
            for (let k = 1; k < 4; k++) assert.strictEqual(table.getColumnByName(`rot_${k}`).data[j], 0);
        }
    });
});

describe('SOG v3 opacity', () => {
    it('without excess, decodes exactly as v2 (codebook i/255)', async () => {
        const dt = createTestDataTable(500);
        const v2 = await roundTrip(dt, { version: 2 });
        const v3 = await roundTrip(dt);
        assert.deepStrictEqual(v3.meta.opacity.codebook, Array.from({ length: 256 }, (_, i) => Math.fround(i / 255)));
        assert.deepStrictEqual(
            Array.from(v3.table.getColumnByName('opacity').data),
            Array.from(v2.table.getColumnByName('opacity').data)
        );
        assert.ok(!v3.table.getColumnByName('opacity_excess'), 'no excess column when nothing exceeds 1');
    });

    it('carries opacity above 1 and restores opacity_excess', async () => {
        const n = 2000;
        const dt = createTestDataTable(n);
        const r = rng(3);
        const excess = new Float32Array(n);
        const op = dt.getColumnByName('opacity').data;
        for (let i = 0; i < n; i++) {
            if (i % 3 === 0) {
                op[i] = 20; // saturated (sigmoid ~1)
                excess[i] = Math.exp(r() * Math.log(40)) - 1 + 1e-3; // A in (1, 40]
            }
        }
        dt.addColumn(new Column('opacity_excess', excess));
        const { meta, table } = await roundTrip(dt);

        const cb = meta.opacity.codebook;
        assert.strictEqual(cb.length, 256);
        assert.ok(cb.every((v, i) => i === 0 || v >= cb[i - 1]), 'codebook ascending');
        assert.strictEqual(cb[0], 0);
        assert.strictEqual(cb[191], 1, 'unit opacity is exact');

        // match rows by position, compare A = sigmoid(opacity) + excess
        const key = (t, i) => ['x', 'y', 'z'].map((c) => Math.round(2 * t.getColumnByName(c).data[i])).join(',');
        const byKey = new Map();
        for (let i = 0; i < n; i++) byKey.set(key(dt, i), i);
        const outEx = table.getColumnByName('opacity_excess').data;
        const outOp = table.getColumnByName('opacity').data;
        const sig = (v) => 1 / (1 + Math.exp(-v));
        let worstRel = 0, worstAbs = 0;
        for (let j = 0; j < table.numRows; j++) {
            const i = byKey.get(key(table, j));
            const aIn = sig(op[i]) + excess[i];
            const aOut = sig(outOp[j]) + outEx[j];
            if (aIn > 1) worstRel = Math.max(worstRel, Math.abs(aOut / aIn - 1));
            else worstAbs = Math.max(worstAbs, Math.abs(aOut - aIn));
            assert.ok(outEx[j] === 0 || sig(outOp[j]) > 0.999, 'excess only on saturated splats');
        }
        assert.ok(worstRel < 0.05, `over-unity relative error ${worstRel}`);
        assert.ok(worstAbs <= 0.5 / 191 + 1e-6, `sub-unity error ${worstAbs}`);
    });

    it('v2 drops excess opacity with a warning', async () => {
        const dt = createTestDataTable(100);
        const excess = new Float32Array(100);
        excess[0] = 2;
        dt.addColumn(new Column('opacity_excess', excess));
        let result;
        const warnings = await captureWarnings(async () => {
            result = await roundTrip(dt, { version: 2 });
        });
        assert.ok(warnings.some((w) => /opacity above 1/.test(w)), warnings.join('\n'));
        assert.ok(!result.table.getColumnByName('opacity_excess'));
    });
});

describe('SOG v3 blind data', () => {
    const withBlind = (values) => {
        const dt = createTestDataTable(values.length);
        dt.addColumn(new Column('segment', Uint32Array.from(values)));
        return dt;
    };

    for (const [bytes, max] of [[1, 0xff], [2, 0xffff], [3, 0xffffff], [4, 0xffffffff]]) {
        it(`round-trips ${bytes}-byte values exactly (incl. 0 and max)`, async () => {
            const r = rng(10 + bytes);
            const values = Array.from({ length: 300 }, (_, i) => (i === 0 ? 0 : i === 1 ? max : Math.floor(r() * max)));
            const dt = withBlind(values);
            const { meta, table } = await roundTrip(dt, { blind: { column: 'segment' } });
            assert.deepStrictEqual(meta.blind, { name: 'segment', bytes, files: ['blind.webp'] });
            const key = (t, i) => ['x', 'y', 'z'].map((c) => Math.round(2 * t.getColumnByName(c).data[i])).join(',');
            const byKey = new Map();
            for (let i = 0; i < dt.numRows; i++) byKey.set(key(dt, i), i);
            const out = table.getColumnByName('segment').data;
            for (let j = 0; j < table.numRows; j++) {
                assert.strictEqual(out[j], values[byKey.get(key(table, j))]);
            }
        });
    }

    it('honours an explicit byte count and rejects one that is too small', async () => {
        const dt = withBlind([1, 2, 300]);
        const { meta } = await roundTrip(dt, { blind: { column: 'segment', bytes: 4 } });
        assert.strictEqual(meta.blind.bytes, 4);
        await assert.rejects(roundTrip(dt, { blind: { column: 'segment', bytes: 1 } }), /needs 2 bytes/);
    });

    it('rejects a missing column and non-integer floats; accepts integral floats', async () => {
        const dt = withBlind([1, 2, 3]);
        await assert.rejects(roundTrip(dt, { blind: { column: 'nope' } }), /column 'nope' not found/);

        const f = createTestDataTable(3);
        f.addColumn(new Column('label', Float32Array.from([1, 2.5, 3])));
        await assert.rejects(roundTrip(f, { blind: { column: 'label' } }), /must be unsigned integers/);

        const ok = createTestDataTable(3);
        ok.addColumn(new Column('label', Float32Array.from([1, 2, 3])));
        const { meta } = await roundTrip(ok, { blind: { column: 'label' } });
        assert.strictEqual(meta.blind.bytes, 1);
    });

    it('v2 ignores blind data with a warning', async () => {
        const dt = withBlind([1, 2, 3]);
        let result;
        const warnings = await captureWarnings(async () => {
            result = await roundTrip(dt, { version: 2, blind: { column: 'segment' } });
        });
        assert.ok(warnings.some((w) => /cannot store blind data/.test(w)), warnings.join('\n'));
        assert.strictEqual(result.meta.blind, undefined);
    });
});

describe('SOG extras', () => {
    const extras = { capture: { device: 'xgrid-l2', date: '2026-10-27' }, tags: ['a', 'b'], n: 3 };

    const sourceWith = async (value) => {
        const { dataTableToChunkSource } = await import('../src/lib/compat/data-table.js');
        const { withExtras } = await import('../src/lib/extras.js');
        const pool = createChunkDataPool();
        return { pool, src: withExtras(dataTableToChunkSource(createTestDataTable(64), pool.chunkSize), value) };
    };

    const write = async (src, pool, options = {}) => {
        const fs = new MemoryFileSystem();
        await writeSogSource(src, pool, { filename: 'out.sog', bundle: true, iterations: 3, logging: 'silent', ...options }, fs);
        const rfs = new MemoryReadFileSystem();
        rfs.set('out.sog', fs.results.get('out.sog'));
        const zip = new ZipReadFileSystem(await rfs.createSource('out.sog'));
        return { zip, meta: JSON.parse(new TextDecoder().decode(await readRawFile(zip, 'meta.json'))) };
    };

    it('writes the source extras, an explicit value, or none (null), in v3 and v2', async () => {
        const { src, pool } = await sourceWith(extras);
        assert.deepStrictEqual((await write(src, pool)).meta.extras, extras);
        assert.deepStrictEqual((await write(src, pool, { version: 2 })).meta.extras, extras);
        assert.deepStrictEqual((await write(src, pool, { extras: { other: true } })).meta.extras, { other: true });
        assert.strictEqual((await write(src, pool, { extras: null })).meta.extras, undefined);
        const plain = await sourceWith(undefined);
        assert.strictEqual('extras' in (await write(plain.src, plain.pool)).meta, false);
    });

    it('round-trips through the SOG reader', async () => {
        const { src, pool } = await sourceWith(extras);
        const { zip } = await write(src, pool);
        const read = await readSogSource(zip, 'meta.json', pool);
        assert.deepStrictEqual(read.meta.extras, extras);
        assert.deepStrictEqual((await write(read, pool)).meta.extras, extras);
    });

    it('keeps extras when combined inputs agree and drops them when they differ', async () => {
        const { concatSource, stackLods } = await import('../src/lib/ops/index.js');
        const a = await sourceWith(extras);
        const b = await sourceWith(JSON.parse(JSON.stringify(extras)));
        const c = await sourceWith({ different: 1 });
        assert.deepStrictEqual(concatSource([a.src, b.src], a.pool).meta.extras, extras);
        assert.deepStrictEqual(stackLods([a.src, b.src]).meta.extras, extras);
        let warnings = await captureWarnings(async () => {
            assert.strictEqual(concatSource([a.src, c.src], a.pool).meta.extras, undefined);
        });
        assert.ok(warnings.some((w) => /different extras/.test(w)));
        warnings = await captureWarnings(async () => {
            assert.strictEqual(stackLods([a.src, c.src]).meta.extras, undefined);
        });
        assert.ok(warnings.some((w) => /different extras/.test(w)));
    });

    it('survives decimation', async () => {
        const { decimateSource } = await import('../src/lib/decimate-uniform/index.js');
        const { dataTableToChunkSource } = await import('../src/lib/compat/data-table.js');
        const { withExtras } = await import('../src/lib/extras.js');
        const pool = createChunkDataPool();
        const dt = createTestDataTable(2000);
        setRandomRotations(dt, 9);
        const out = await decimateSource(withExtras(dataTableToChunkSource(dt, pool.chunkSize), extras), pool, { targetCount: 500 });
        assert.deepStrictEqual(out.meta.extras, extras);
        await out.close();
    });
});

describe('SOG v3 SH palette', () => {
    // 70000 gaussians, each with its own random band-1 SH, and a palette larger than the scene, so
    // every gaussian is its own entry and labels run past 65535 into the third label byte
    const count = 70000;
    const makeData = () => {
        const dt = createTestDataTable(count, { includeSH: true, shBands: 1 });
        const r = rng(11);
        for (let k = 0; k < 9; k++) {
            const col = dt.getColumnByName(`f_rest_${k}`).data;
            for (let i = 0; i < count; i++) col[i] = 2 * r() - 1;
        }
        return dt;
    };

    it('stores more than 65536 entries with 24-bit labels', async () => {
        const src = makeData();
        const { meta, table } = await roundTrip(src, { paletteSize: 80000 });
        assert.strictEqual(meta.shN.count, count);

        const key = (t, i) => ['x', 'y', 'z'].map((c) => Math.round(2 * t.getColumnByName(c).data[i])).join(',');
        const byKey = new Map();
        for (let i = 0; i < src.numRows; i++) byKey.set(key(src, i), i);
        let maxErr = 0;
        for (let j = 0; j < table.numRows; j++) {
            const i = byKey.get(key(table, j));
            for (let k = 0; k < 9; k++) {
                const err = Math.abs(src.getColumnByName(`f_rest_${k}`).data[i] - table.getColumnByName(`f_rest_${k}`).data[j]);
                maxErr = Math.max(maxErr, err);
            }
        }
        // only the 8-bit codebook stage is lossy; a dropped third label byte would swap in random entries
        assert.ok(maxErr < 0.02, `max SH error ${maxErr}`);
    });

    it('v2 rejects a palette larger than 65536', async () => {
        await assert.rejects(roundTrip(makeData(), { version: 2, paletteSize: 80000 }), /palette size/);
    });
});
