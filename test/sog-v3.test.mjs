/**
 * SOG v3 (default) vs v2: rotation encodings, the fixed opacity byte with
 * true opacity above 1 (opacity_linear), blind data, and what v2 drops.
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
import { decodeOpacityByte, encodeOpacityByte, OPACITY_MAX } from '../src/lib/utils/opacity.js';
import { floatToHalfBits, halfBitsToFloat } from '../src/lib/utils/sog-blind.js';
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
    const sig = (v) => 1 / (1 + Math.exp(-v));
    // out row j -> source row, matched by position (the writer reorders by Morton code)
    const matchRows = (src, out) => {
        const key = (t, i) => ['x', 'y', 'z'].map((c) => Math.round(2 * t.getColumnByName(c).data[i])).join(',');
        const byKey = new Map();
        for (let i = 0; i < src.numRows; i++) byKey.set(key(src, i), i);
        return Array.from({ length: out.numRows }, (_, j) => byKey.get(key(out, j)));
    };

    it('the byte encoding is monotone, exact at 0, 1 and OPACITY_MAX, and round-trips', () => {
        assert.strictEqual(decodeOpacityByte(0), 0);
        assert.strictEqual(decodeOpacityByte(204), 1);
        assert.strictEqual(decodeOpacityByte(255), OPACITY_MAX);
        assert.strictEqual(encodeOpacityByte(1), 204);
        for (let b = 0; b < 256; b++) {
            assert.strictEqual(encodeOpacityByte(decodeOpacityByte(b)), b, `byte ${b}`);
            if (b > 0) assert.ok(decodeOpacityByte(b) > decodeOpacityByte(b - 1), `monotone at ${b}`);
        }
        assert.strictEqual(encodeOpacityByte(100), 255, 'clamped to OPACITY_MAX');
        assert.strictEqual(encodeOpacityByte(NaN), 0);
        // finer than v2's linear byte where translucent layers live
        for (const a of [0.005, 0.01, 0.02, 0.05]) {
            const err = Math.abs(decodeOpacityByte(encodeOpacityByte(a)) - a);
            assert.ok(err < 0.25 / 255, `error ${err} at ${a}`);
        }
    });

    it('without opacity above 1, stores opacity on the byte lattice and writes no opacity meta', async () => {
        const dt = createTestDataTable(500);
        const { meta, table } = await roundTrip(dt);
        assert.strictEqual(meta.opacity, undefined);
        assert.ok(!table.getColumnByName('opacity_linear'), 'no opacity_linear column when nothing exceeds 1');
        const src = matchRows(dt, table);
        const inOp = dt.getColumnByName('opacity').data, outOp = table.getColumnByName('opacity').data;
        for (let j = 0; j < table.numRows; j++) {
            const want = decodeOpacityByte(encodeOpacityByte(sig(inOp[src[j]])));
            assert.ok(Math.abs(sig(outOp[j]) - want) < 1e-5, `row ${j}: ${sig(outOp[j])} vs ${want}`);
        }
    });

    it('carries opacity above 1, clamped to OPACITY_MAX, and restores opacity_linear', async () => {
        const n = 2000;
        const dt = createTestDataTable(n);
        const r = rng(3);
        const linear = new Float32Array(n);
        const op = dt.getColumnByName('opacity').data;
        for (let i = 0; i < n; i++) {
            if (i % 3 === 0) {
                op[i] = 20; // saturated (sigmoid ~1)
                linear[i] = Math.exp(r() * Math.log(40)) + 1e-3; // A in (1, 40]
            } else {
                linear[i] = r(); // unrelated to the logit: the column wins
            }
        }
        dt.addColumn(new Column('opacity_linear', linear));
        const { meta, table } = await roundTrip(dt);
        assert.strictEqual(meta.opacity.max, OPACITY_MAX);

        const src = matchRows(dt, table);
        const outLin = table.getColumnByName('opacity_linear').data;
        const outOp = table.getColumnByName('opacity').data;
        for (let j = 0; j < table.numRows; j++) {
            const want = decodeOpacityByte(encodeOpacityByte(linear[src[j]]));
            assert.ok(Math.abs(outLin[j] - want) < 1e-5, `row ${j}: ${outLin[j]} vs ${want}`);
            assert.ok(outLin[j] <= OPACITY_MAX + 1e-6, 'clamped');
            assert.ok(Math.abs(sig(outOp[j]) - Math.min(outLin[j], 1)) < 1e-5, 'the logit holds the clamped opacity');
        }
    });

    it('v2 writes opacity_linear clamped to 1, with a warning', async () => {
        const dt = createTestDataTable(100);
        const linear = new Float32Array(100).fill(0.25);
        linear[0] = 2;
        dt.addColumn(new Column('opacity_linear', linear));
        let result;
        const warnings = await captureWarnings(async () => {
            result = await roundTrip(dt, { version: 2 });
        });
        assert.ok(warnings.some((w) => /opacity above 1/.test(w)), warnings.join('\n'));
        const { table } = result;
        assert.ok(!table.getColumnByName('opacity_linear'));
        const src = matchRows(dt, table);
        const outOp = table.getColumnByName('opacity').data;
        for (let j = 0; j < table.numRows; j++) {
            const want = src[j] === 0 ? 1 : Math.round(0.25 * 255) / 255;
            assert.ok(Math.abs(sig(outOp[j]) - want) < 1e-5, `row ${j}: ${sig(outOp[j])} vs ${want}`);
        }
    });
});

describe('SOG v3 blind data', () => {
    const withColumns = (n, columns) => {
        const dt = createTestDataTable(n);
        for (const [name, data] of Object.entries(columns)) dt.addColumn(new Column(name, data));
        return dt;
    };

    // out row j -> source row, matched by position (the writer reorders by Morton code)
    const sourceRows = (src, out) => {
        const key = (t, i) => ['x', 'y', 'z'].map((c) => Math.round(2 * t.getColumnByName(c).data[i])).join(',');
        const byKey = new Map();
        for (let i = 0; i < src.numRows; i++) byKey.set(key(src, i), i);
        return Array.from({ length: out.numRows }, (_, j) => byKey.get(key(out, j)));
    };

    for (const [type, max] of [['u8', 0xff], ['u16', 0xffff], ['u24', 0xffffff], ['u32', 0xffffffff]]) {
        it(`round-trips integers exactly, defaulting to the fewest bytes (${type})`, async () => {
            const r = rng(10 + max % 7);
            const values = Array.from({ length: 300 }, (_, i) => (i === 0 ? 0 : i === 1 ? max : Math.floor(r() * max)));
            const dt = withColumns(values.length, { segment: Uint32Array.from(values) });
            const { meta, table } = await roundTrip(dt, { blind: [[{ column: 'segment' }]] });
            assert.deepStrictEqual(meta.blind, [{ files: ['blind_0.webp'], fields: [{ name: 'segment', type }] }]);
            const rows = sourceRows(dt, table);
            const out = table.getColumnByName('segment').data;
            for (let j = 0; j < table.numRows; j++) assert.strictEqual(out[j], values[rows[j]]);
        });
    }

    it('packs four unorm8 floats into one texture', async () => {
        const n = 200;
        const r = rng(3);
        const cols = Object.fromEntries(['a', 'b', 'c', 'd'].map((k) => [k, Float32Array.from({ length: n }, (_, i) => (i === 0 ? 0 : i === 1 ? 1 : r()))]));
        const dt = withColumns(n, cols);
        const { meta, table } = await roundTrip(dt, { blind: [['a', 'b', 'c', 'd'].map((column) => ({ column, encoding: 'unorm8' }))] });
        assert.deepStrictEqual(meta.blind[0].fields, ['a', 'b', 'c', 'd'].map((name) => ({ name, type: 'unorm8' })), 'packed in order; the default [0, 1] range is not stored');
        const rows = sourceRows(dt, table);
        for (const k of ['a', 'b', 'c', 'd']) {
            const out = table.getColumnByName(k).data;
            for (let j = 0; j < n; j++) {
                const v = cols[k][rows[j]];
                assert.ok(Math.abs(out[j] - Math.round(v * 255) / 255) < 1e-6, `${k}: ${out[j]} vs ${v}`);
            }
        }
    });

    it('maps a unorm range and records it; rejects values outside it', async () => {
        const values = Float32Array.from([0.1, 5, 42]);
        const dt = withColumns(3, { depth: values });
        const { meta, table } = await roundTrip(dt, { blind: [[{ column: 'depth', encoding: 'unorm16', range: [0.1, 42] }]] });
        assert.deepStrictEqual(meta.blind[0].fields[0], { name: 'depth', type: 'unorm16', range: [0.1, 42] });
        const rows = sourceRows(dt, table);
        const out = table.getColumnByName('depth').data;
        for (let j = 0; j < 3; j++) assert.ok(Math.abs(out[j] - values[rows[j]]) <= (41.9 / 65535) * 0.5 + 1e-6);
        await assert.rejects(roundTrip(dt, { blind: [[{ column: 'depth', encoding: 'unorm8' }]] }), /outside its unorm8 range \[0, 1\]/);
    });

    it('accepts an endpoint value stored as f32 just outside its range', async () => {
        // 0.1 stores as 0.10000000149011612, just above the range's 0.1
        const values = Float32Array.from([0, 0.05, 0.1]);
        const dt = withColumns(3, { weight: values });
        const { table } = await roundTrip(dt, { blind: [[{ column: 'weight', encoding: 'unorm8', range: [0, 0.1] }]] });
        const rows = sourceRows(dt, table);
        const out = table.getColumnByName('weight').data;
        for (let j = 0; j < 3; j++) assert.ok(Math.abs(out[j] - values[rows[j]]) <= (0.1 / 255) * 0.5 + 1e-6, `row ${j}: ${out[j]}`);
    });

    it('stores f32 bits exactly and f16 rounded to nearest, mixed with an integer field', async () => {
        const f = Float32Array.from([0, -1.5, 3.14159, 1e-30, -65504, 123456.789]);
        const h = Float32Array.from([0, -1.5, 3.14159, 1e-5, -65504, 0.333]);
        const id = Uint32Array.from([0, 1, 2, 3, 4, 65535]);
        const dt = withColumns(6, { f, h, id });
        const { meta, table } = await roundTrip(dt, { blind: [[{ column: 'f', encoding: 'f32' }], [{ column: 'h', encoding: 'f16' }, { column: 'id', encoding: 'u16' }]] });
        assert.deepStrictEqual(meta.blind.map((t) => t.files[0]), ['blind_0.webp', 'blind_1.webp']);
        assert.deepStrictEqual(meta.blind[1].fields, [{ name: 'h', type: 'f16' }, { name: 'id', type: 'u16' }]);
        const rows = sourceRows(dt, table);
        for (let j = 0; j < 6; j++) {
            const i = rows[j];
            assert.strictEqual(table.getColumnByName('f').data[j], f[i]);
            assert.strictEqual(table.getColumnByName('h').data[j], halfBitsToFloat(floatToHalfBits(h[i])));
            assert.strictEqual(table.getColumnByName('id').data[j], id[i]);
        }
        await assert.rejects(roundTrip(withColumns(1, { h: Float32Array.from([70000]) }), { blind: [[{ column: 'h', encoding: 'f16' }]] }), /outside the f16 range/);
    });

    it('rejects malformed requests', async () => {
        const dt = withColumns(3, { a: Uint32Array.from([1, 2, 300]), b: Float32Array.from([1, 2.5, 3]) });
        const cases = [
            [[[{ column: 'nope' }]], /column 'nope' not found/],
            [[[{ column: 'a', encoding: 'u32' }, { column: 'b', encoding: 'u8' }]], /need 5 bytes; a blind texture holds 4/],
            [[[{ column: 'a' }], [{ column: 'a' }]], /stored more than once/],
            [[[{ column: 'a-b' }, { column: 'a_b' }]], /columns 'a-b' and 'a_b' share the shader name 'a_b'/],
            [[[{ column: '--' }]], /no letters, digits or '_'/],
            [[[{ column: 'a', encoding: 'u16', range: [0, 1] }]], /range applies only to unorm8/],
            [[[{ column: 'b', encoding: 'unorm8', range: [1, 1] }]], /invalid range/],
            [[[{ column: 'a', encoding: 'u8' }]], /u8 stores unsigned integers 0 to 255/],
            [[[{ column: 'b' }]], /stores unsigned integers/],
            [[[]], /at least one field/]
        ];
        for (const [blind, error] of cases) await assert.rejects(roundTrip(dt, { blind }), error);
    });

    it('accepts integral floats under the default integer encoding', async () => {
        const dt = withColumns(3, { label: Float32Array.from([1, 2, 3]) });
        const { meta, table } = await roundTrip(dt, { blind: [[{ column: 'label' }]] });
        assert.strictEqual(meta.blind[0].fields[0].type, 'u8');
        assert.deepStrictEqual(Array.from(table.getColumnByName('label').data).sort(), [1, 2, 3]);
    });

    it('v2 ignores blind data with a warning', async () => {
        const dt = withColumns(3, { segment: Uint32Array.from([1, 2, 3]) });
        let result;
        const warnings = await captureWarnings(async () => {
            result = await roundTrip(dt, { version: 2, blind: [[{ column: 'segment' }]] });
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

    it('writes the source extras, an explicit value, or none (null), in v3 only', async () => {
        const { src, pool } = await sourceWith(extras);
        assert.deepStrictEqual((await write(src, pool)).meta.extras, extras);
        assert.strictEqual('extras' in (await write(src, pool, { version: 2 })).meta, false, 'v2 output stays as released');
        assert.strictEqual('extras' in (await write(src, pool, { version: 2, extras: { other: true } })).meta, false, 'even when given');
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

describe('SOG v3 opacity above 1 in sources', () => {
    // A SOG of `n` gaussians with blind column `id`, and opacity 1.5 when `over`.
    const openSog = async (n, over, pool) => {
        const dt = createTestDataTable(n);
        dt.addColumn(new Column('id', Uint32Array.from({ length: n }, (_, i) => i)));
        if (over) dt.addColumn(new Column('opacity_linear', new Float32Array(n).fill(1.5)));
        const fs = new MemoryFileSystem();
        await writeSogSource(await plySource(dt, pool), pool, {
            filename: 'out.sog', bundle: true, iterations: 3, logging: 'silent', blind: [[{ column: 'id', encoding: 'u16' }]]
        }, fs);
        const rfs = new MemoryReadFileSystem();
        rfs.set('out.sog', fs.results.get('out.sog'));
        return readSogSource(new ZipReadFileSystem(await rfs.createSource('out.sog')), 'meta.json', pool);
    };

    // the opacity slot of a source's first `count` gaussians
    const readOpacity = async (src, pool, count, lod = 0) => {
        const g = pool.acquire('geometric', src.meta.layouts.geometric, count);
        await src.read({ chunkIndex: 0, lod, geometric: g });
        const values = Array.from({ length: count }, (_, i) => new Float32Array(g.data)[i * 8 + 7]);
        g.release();
        return values;
    };

    it('reads true opacity into the opacity slot, with no extra column, and flags it', async () => {
        const { concatSource, stackLods } = await import('../src/lib/ops/index.js');
        const pool = createChunkDataPool();
        const a = await openSog(64, true, pool);
        const b = await openSog(32, false, pool);
        const names = (src) => src.meta.extraColumns.map((c) => c.name);
        assert.deepStrictEqual(names(a), ['id']);
        assert.strictEqual(a.meta.opacityAboveOne, true);
        assert.strictEqual(b.meta.opacityAboveOne, undefined);
        assert.ok((await readOpacity(a, pool, 64)).every((v) => Math.abs(v - decodeOpacityByte(encodeOpacityByte(1.5))) < 1e-6));
        // sources with and without opacity above 1 combine, and the result carries the flag
        assert.strictEqual(concatSource([b, a], pool).meta.opacityAboveOne, true);
        assert.strictEqual(stackLods([b, a]).meta.opacityAboveOne, true);
    });

    it('keeps opacity above 1 in a streamed level whose units have it when the first level\'s do not', async () => {
        const { stackLods } = await import('../src/lib/ops/index.js');
        const { dataTableToChunkSource } = await import('../src/lib/compat/data-table.js');
        const { writeLodSource } = await import('../src/lib/writers/write-lod.js');
        const { readLodSource } = await import('../src/lib/readers/read-lod.js');
        const pool = createChunkDataPool();
        const fine = dataTableToChunkSource(createTestDataTable(64), pool.chunkSize);
        const coarseTable = createTestDataTable(16);
        coarseTable.addColumn(new Column('opacity_linear', new Float32Array(16).fill(1.5)));
        const coarse = dataTableToChunkSource(coarseTable, pool.chunkSize);

        const fs = new MemoryFileSystem();
        await writeLodSource({
            filename: '/scene/lod-meta.json', mainSource: stackLods([fine, coarse]), envSource: null,
            iterations: 1, chunkCount: 1, chunkExtent: 16
        }, fs);
        const rfs = new MemoryReadFileSystem();
        for (const [path, bytes] of fs.results) rfs.set(path, bytes);
        const lod = await readLodSource(rfs, '/scene/lod-meta.json', {}, pool);

        assert.strictEqual(lod.meta.opacityAboveOne, true, 'the scene is flagged though its first unit holds no opacity above 1');
        assert.ok((await readOpacity(lod, pool, 16, 1)).every((a) => Math.abs(a - decodeOpacityByte(encodeOpacityByte(1.5))) < 1e-6), 'level 1 keeps A = 1.5');
        assert.ok((await readOpacity(lod, pool, 64, 0)).every((a) => a > 0 && a <= 1), 'level 0 reads its own opacity');
        await lod.close();
    });
});

describe('opacity_linear in formats that store only the logit', () => {
    it('is baked into the opacity, since it wins over the logit', async () => {
        const { writeSource, readFile, getInputFormat } = await import('../src/lib/index.js');
        for (const [filename, outputFormat, tolerance] of [['out.compressed.ply', 'compressed-ply', 1 / 255], ['out.splat', 'splat', 1 / 255]]) {
            const pool = createChunkDataPool();
            const dt = createTestDataTable(8);
            dt.getColumnByName('opacity').data.fill(20); // the logit says opaque
            dt.addColumn(new Column('opacity_linear', new Float32Array(8).fill(0.25)));
            const fs = new MemoryFileSystem();
            await writeSource({ filename, outputFormat, source: await plySource(dt, pool), pool, options: {} }, fs);
            const rfs = new MemoryReadFileSystem();
            rfs.set(filename, fs.results.get(filename));
            const [back] = await readFile({ filename, inputFormat: getInputFormat(filename), fileSystem: rfs, pool });
            const op = (await materializeToDataTable(back, pool)).getColumnByName('opacity').data;
            for (const v of op) {
                const a = 1 / (1 + Math.exp(-v));
                assert.ok(Math.abs(a - 0.25) <= tolerance, `${outputFormat}: opacity ${a}`);
            }
        }
    });
});

describe('SH4 in the DataTable writers', () => {
    // channel-major f_rest_(i + nj): coefficient i of channel j; band-4 coefficients (i >= 15) are 9
    const table = (bands) => {
        const n = [0, 3, 8, 15, 24][bands];
        const dt = createTestDataTable(16);
        for (let k = 0; k < 3 * n; k++) {
            const i = k % n;
            dt.addColumn(new Column(`f_rest_${k}`, new Float32Array(16).fill(i < 15 ? (i + 1 + 15 * Math.floor(k / n)) / 64 : 9)));
        }
        return dt;
    };

    it('writeCompressedPly and writeGlb write an SH4 table as its first 3 bands', async () => {
        const { writeCompressedPly, writeGlb } = await import('../src/lib/index.js');
        const write = async (dt) => {
            const fs = new MemoryFileSystem();
            await writeCompressedPly({ filename: 'out.compressed.ply', dataTable: dt }, fs);
            await writeGlb({ filename: 'out.glb', dataTable: dt }, fs);
            return fs.results;
        };
        let sh4;
        const warnings = await captureWarnings(async () => {
            sh4 = await write(table(4));
        });
        assert.ok(warnings.some((w) => /4th SH band/.test(w)), warnings.join('\n'));
        const sh3 = await write(table(3));
        assert.deepStrictEqual(sh4.get('out.compressed.ply'), sh3.get('out.compressed.ply'), 'compressed PLY');
        assert.deepStrictEqual(sh4.get('out.glb'), sh3.get('out.glb'), 'GLB');
    });
});
