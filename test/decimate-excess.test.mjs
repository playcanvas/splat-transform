/**
 * Excess opacity under decimation: merges keep the mass the opacity <= 1 cap
 * cuts off as `opacity_excess`, later generations read it back as mass, and a
 * missing column behaves exactly like a zero one.
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

import { createChunkDataPool, createInMemoryChunkSource } from '../src/lib/chunk/index.js';
import { dataTableToChunkSource, materializeToDataTable } from '../src/lib/compat/data-table.js';
import { decimateSourceAdaptive } from '../src/lib/decimate/index.js';
import { decimateSource } from '../src/lib/decimate-uniform/index.js';
import { concatSource, stackLods } from '../src/lib/ops/index.js';
import { Transform } from '../src/lib/utils/index.js';

const mulberry = (seed) => {
    let t = seed >>> 0;
    return () => {
        t += 0x6d2b79f5;
        let r = Math.imul(t ^ (t >>> 15), t | 1);
        r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
        return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
};

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

// Knud Thomsen ellipsoid surface approximation, as the decimators use.
const P = 1.6075;
const area = (sx, sy, sz) => 4 * Math.PI * (((sx * sy) ** P + (sx * sz) ** P + (sy * sz) ** P) / 3) ** (1 / P);

/**
 * Dense, nearly opaque, heavily overlapping splats, so merges exceed opacity 1.
 *
 * @param {number} n - Gaussian count.
 * @param {number} seed - Seed.
 * @param {Float32Array|null} excess - Optional opacity_excess column.
 * @returns {{source, pool}} Source and pool.
 */
const makeOpaqueScene = (n, seed, excess = null) => {
    const r = mulberry(seed);
    const chunkSize = 512;
    const pos = new Float32Array(n * 3);
    const geo = new Float32Array(n * 8);
    const color = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
        pos.set([r(), r(), r()], i * 3);
        const q = [r() - 0.5, r() - 0.5, r() - 0.5, r() - 0.5];
        const qn = Math.hypot(...q);
        const s = () => Math.log(0.03 + r() * 0.05);
        geo.set([q[0] / qn, q[1] / qn, q[2] / qn, q[3] / qn, s(), s(), s(), 3 + r() * 2], i * 8);
        color.set([r() - 0.5, r() - 0.5, r() - 0.5], i * 3);
    }
    const chunks = (flat, comps) => {
        const out = [];
        for (let b = 0; b < n; b += chunkSize) out.push(flat.slice(b * comps, Math.min(n, b + chunkSize) * comps).buffer);
        return out;
    };
    const source = createInMemoryChunkSource({
        numGaussians: n,
        chunkSize,
        shBands: 0,
        extraColumns: excess ? [{ name: 'opacity_excess', type: 'float32' }] : [],
        transform: new Transform(),
        lodCounts: [n],
        position: [chunks(pos, 3)],
        geometric: [chunks(geo, 8)],
        color: [chunks(color, 3)],
        ...(excess ? { other: [chunks(new Uint32Array(excess.buffer), 1)] } : {})
    });
    return { source, pool: createChunkDataPool({ chunkSize }) };
};

// Read geometric + the excess column (0 when absent) of a whole source.
const readScene = async (src, pool) => {
    const { meta } = src;
    const n = meta.numGaussians;
    const geo = new Float32Array(n * 8);
    const pos = new Float32Array(n * 3);
    const field = meta.layouts.other?.fields.opacity_excess;
    const sw = field ? meta.layouts.other.stride >> 2 : 0;
    const excess = new Float32Array(n);
    let base = 0;
    for (let k = 0; k < meta.numChunks[0]; k++) {
        const count = Math.min(meta.chunkSize, n - base);
        const g = pool.acquire('geometric', meta.layouts.geometric, count);
        const p = pool.acquire('position', meta.layouts.position, count);
        const o = field ? pool.acquire('other', meta.layouts.other, count) : null;
        await src.read({ chunkIndex: k, geometric: g, position: p, ...(o ? { other: o } : {}) });
        geo.set(new Float32Array(g.data, 0, count * 8), base * 8);
        pos.set(new Float32Array(p.data, 0, count * 3), base * 3);
        if (o) {
            const w = new Float32Array(o.data, 0, count * sw);
            for (let i = 0; i < count; i++) excess[base + i] = w[i * sw + (field.byteOffset >> 2)];
        }
        g.release();
        p.release();
        o?.release();
        base += count;
    }
    return { geo, pos, excess, hasColumn: !!field };
};

const totalMass = ({ geo, excess }) => {
    let m = 0;
    for (let i = 0; i < excess.length; i++) {
        const o = i * 8;
        m += (sigmoid(geo[o + 7]) + excess[i]) * area(Math.exp(geo[o + 4]), Math.exp(geo[o + 5]), Math.exp(geo[o + 6]));
    }
    return m;
};

const decimators = [
    ['uniform', decimateSource],
    ['adaptive', decimateSourceAdaptive]
];

describe('decimation excess opacity', () => {
    for (const [name, decimate] of decimators) {
        it(`${name}: output carries opacity_excess and conserves mass`, async () => {
            const { source, pool } = makeOpaqueScene(3000, 1);
            const input = await readScene(source, pool);
            const out = await decimate(source, pool, { targetCount: 750 });
            const result = await readScene(out, pool);
            await out.close();

            assert.ok(result.hasColumn, 'output has opacity_excess');
            let over = 0;
            for (let i = 0; i < result.excess.length; i++) {
                const e = result.excess[i];
                assert.ok(e >= 0 && Number.isFinite(e), `row ${i} excess ${e}`);
                if (e > 0) {
                    over++;
                    assert.ok(sigmoid(result.geo[i * 8 + 7]) > 0.999, `row ${i}: excess only on saturated splats`);
                }
            }
            assert.ok(over > 0, 'the scene is dense enough for merges to exceed opacity 1');
            const rel = Math.abs(totalMass(result) / totalMass(input) - 1);
            assert.ok(rel < 0.02, `mass drift ${rel}`);
        });

        it(`${name}: a second generation reads the excess as mass`, async () => {
            const { source, pool } = makeOpaqueScene(3000, 2);
            const input = await readScene(source, pool);
            // decimate output is one streaming pass: materialize it, as the CLI
            // does by writing the level to PLY
            const level1 = await decimate(source, pool, { targetCount: 1500 });
            const table = await materializeToDataTable(level1, pool);
            await level1.close();
            assert.ok(table.getColumnByName('opacity_excess'), 'level 1 carries the column');
            const level2 = await decimate(dataTableToChunkSource(table, pool.chunkSize), pool, { targetCount: 500 });
            const result = await readScene(level2, pool);
            await level2.close();
            const rel = Math.abs(totalMass(result) / totalMass(input) - 1);
            assert.ok(rel < 0.03, `mass drift over two generations ${rel}`);
        });

        it(`${name}: a zero column decimates exactly like no column`, async () => {
            const a = makeOpaqueScene(2000, 3);
            const b = makeOpaqueScene(2000, 3, new Float32Array(2000));
            const outA = await readScene(await decimate(a.source, a.pool, { targetCount: 600 }), a.pool);
            const outB = await readScene(await decimate(b.source, b.pool, { targetCount: 600 }), b.pool);
            assert.deepStrictEqual(outA.geo, outB.geo);
            assert.deepStrictEqual(outA.pos, outB.pos);
            assert.deepStrictEqual(outA.excess, outB.excess);
        });
    }
});

describe('concatSource with opacity_excess', () => {
    it('zero-fills the column for inputs that lack it', async () => {
        const excess = Float32Array.from({ length: 100 }, (_, i) => i / 10);
        const a = makeOpaqueScene(100, 4, excess);
        const b = makeOpaqueScene(50, 5);
        const joined = concatSource([a.source, b.source], a.pool);
        const { excess: out, hasColumn } = await readScene(joined, a.pool);
        assert.ok(hasColumn);
        assert.deepStrictEqual(Array.from(out.subarray(0, 100)), Array.from(excess));
        assert.ok(out.subarray(100).every((v) => v === 0));
    });
});

describe('stackLods with opacity_excess', () => {
    it('zero-fills the column for levels that lack it (a raw LOD 0 under decimated levels)', async () => {
        const excess = Float32Array.from({ length: 40 }, (_, i) => i / 4);
        const fine = makeOpaqueScene(100, 6);
        const coarse = makeOpaqueScene(40, 7, excess);
        const stacked = stackLods([fine.source, coarse.source], fine.pool);
        assert.deepStrictEqual(stacked.meta.extraColumns.map((e) => e.name), ['opacity_excess']);
        const layout = stacked.meta.layouts.other;
        const read = async (lod, count) => {
            const o = fine.pool.acquire('other', layout, count);
            await stacked.read({ chunkIndex: 0, lod, other: o });
            const values = Array.from(new Float32Array(o.data, 0, count));
            o.release();
            return values;
        };
        assert.ok((await read(0, 100)).every((v) => v === 0));
        assert.deepStrictEqual(await read(1, 40), Array.from(excess));
        assert.throws(() => stackLods([fine.source, coarse.source]), /mismatch between LOD levels/);
    });
});
