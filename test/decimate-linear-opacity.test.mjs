/**
 * True opacity under decimation: merges keep their full mass, above 1 too, as
 * `opacity_linear`, later generations read it back as mass, and a missing
 * column behaves exactly like one holding `sigmoid(opacity)`.
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
 * @param {Float32Array|true|null} linear - Optional opacity_linear column (true: fill it with sigmoid(opacity)).
 * @returns {{source, pool}} Source and pool.
 */
const makeOpaqueScene = (n, seed, linear = null) => {
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
    if (linear === true) linear = Float32Array.from({ length: n }, (_, i) => sigmoid(geo[i * 8 + 7]));
    const chunks = (flat, comps) => {
        const out = [];
        for (let b = 0; b < n; b += chunkSize) out.push(flat.slice(b * comps, Math.min(n, b + chunkSize) * comps).buffer);
        return out;
    };
    const source = createInMemoryChunkSource({
        numGaussians: n,
        chunkSize,
        shBands: 0,
        extraColumns: linear ? [{ name: 'opacity_linear', type: 'float32' }] : [],
        transform: new Transform(),
        lodCounts: [n],
        position: [chunks(pos, 3)],
        geometric: [chunks(geo, 8)],
        color: [chunks(color, 3)],
        ...(linear ? { other: [chunks(new Uint32Array(linear.buffer), 1)] } : {})
    });
    return { source, pool: createChunkDataPool({ chunkSize }) };
};

// Read geometric + the true opacity (the column, else sigmoid(opacity)) of a whole source.
const readScene = async (src, pool) => {
    const { meta } = src;
    const n = meta.numGaussians;
    const geo = new Float32Array(n * 8);
    const pos = new Float32Array(n * 3);
    const field = meta.layouts.other?.fields.opacity_linear;
    const sw = field ? meta.layouts.other.stride >> 2 : 0;
    const linear = new Float32Array(n);
    let base = 0;
    for (let k = 0; k < meta.numChunks[0]; k++) {
        const count = Math.min(meta.chunkSize, n - base);
        const g = pool.acquire('geometric', meta.layouts.geometric, count);
        const p = pool.acquire('position', meta.layouts.position, count);
        const o = field ? pool.acquire('other', meta.layouts.other, count) : null;
        await src.read({ chunkIndex: k, geometric: g, position: p, ...(o ? { other: o } : {}) });
        const gv = new Float32Array(g.data, 0, count * 8);
        geo.set(gv, base * 8);
        pos.set(new Float32Array(p.data, 0, count * 3), base * 3);
        const w = o ? new Float32Array(o.data, 0, count * sw) : null;
        for (let i = 0; i < count; i++) {
            linear[base + i] = w ? w[i * sw + (field.byteOffset >> 2)] : sigmoid(gv[i * 8 + 7]);
        }
        g.release();
        p.release();
        o?.release();
        base += count;
    }
    return { geo, pos, linear, hasColumn: !!field };
};

const totalMass = ({ geo, linear }) => {
    let m = 0;
    for (let i = 0; i < linear.length; i++) {
        const o = i * 8;
        m += linear[i] * area(Math.exp(geo[o + 4]), Math.exp(geo[o + 5]), Math.exp(geo[o + 6]));
    }
    return m;
};

const decimators = [
    ['uniform', decimateSource],
    ['adaptive', decimateSourceAdaptive]
];

describe('decimation true opacity', () => {
    for (const [name, decimate] of decimators) {
        it(`${name}: output carries opacity_linear and conserves mass`, async () => {
            const { source, pool } = makeOpaqueScene(3000, 1);
            const input = await readScene(source, pool);
            const out = await decimate(source, pool, { targetCount: 750 });
            const result = await readScene(out, pool);
            await out.close();

            assert.ok(result.hasColumn, 'output has opacity_linear');
            let over = 0;
            for (let i = 0; i < result.linear.length; i++) {
                const a = result.linear[i];
                assert.ok(a > 0 && Number.isFinite(a), `row ${i} opacity_linear ${a}`);
                if (a > 1) over++;
                // the logit stays the clamped opacity, for readers that don't know the column
                const clamped = sigmoid(result.geo[i * 8 + 7]);
                assert.ok(Math.abs(clamped - Math.min(a, 1)) < 1e-5, `row ${i}: opacity ${clamped} vs min(${a}, 1)`);
            }
            assert.ok(over > 0, 'the scene is dense enough for merges to exceed opacity 1');
            const rel = Math.abs(totalMass(result) / totalMass(input) - 1);
            assert.ok(rel < 0.02, `mass drift ${rel}`);
        });

        it(`${name}: a second generation reads opacity_linear as mass`, async () => {
            const { source, pool } = makeOpaqueScene(3000, 2);
            const input = await readScene(source, pool);
            // decimate output is one streaming pass: materialize it, as the CLI
            // does by writing the level to PLY
            const level1 = await decimate(source, pool, { targetCount: 1500 });
            const table = await materializeToDataTable(level1, pool);
            await level1.close();
            assert.ok(table.getColumnByName('opacity_linear'), 'level 1 carries the column');
            const level2 = await decimate(dataTableToChunkSource(table, pool.chunkSize), pool, { targetCount: 500 });
            const result = await readScene(level2, pool);
            await level2.close();
            const rel = Math.abs(totalMass(result) / totalMass(input) - 1);
            assert.ok(rel < 0.03, `mass drift over two generations ${rel}`);
        });

        it(`${name}: a column holding sigmoid(opacity) decimates exactly like no column`, async () => {
            const a = makeOpaqueScene(2000, 3);
            const b = makeOpaqueScene(2000, 3, true);
            const outA = await readScene(await decimate(a.source, a.pool, { targetCount: 600 }), a.pool);
            const outB = await readScene(await decimate(b.source, b.pool, { targetCount: 600 }), b.pool);
            assert.deepStrictEqual(outA.geo, outB.geo);
            assert.deepStrictEqual(outA.pos, outB.pos);
            assert.deepStrictEqual(outA.linear, outB.linear);
        });

        it(`${name}: opacity_linear wins over the opacity logit`, async () => {
            // the same scene with every true opacity doubled carries twice the mass
            const a = makeOpaqueScene(2000, 8, true);
            const doubled = (await readScene(a.source, a.pool)).linear.map((v) => 2 * v);
            const b = makeOpaqueScene(2000, 8, doubled);
            const outA = await readScene(await decimate(a.source, a.pool, { targetCount: 600 }), a.pool);
            const outB = await readScene(await decimate(b.source, b.pool, { targetCount: 600 }), b.pool);
            const ratio = totalMass(outB) / totalMass(outA);
            assert.ok(Math.abs(ratio - 2) < 0.05, `mass ratio ${ratio}`);
        });
    }
});

describe('concatSource with opacity_linear', () => {
    it('fills the column with sigmoid(opacity) for inputs that lack it', async () => {
        const linear = Float32Array.from({ length: 100 }, (_, i) => i / 10);
        const a = makeOpaqueScene(100, 4, linear);
        const b = makeOpaqueScene(50, 5);
        const bTrue = (await readScene(b.source, b.pool)).linear;
        const joined = concatSource([a.source, b.source], a.pool);
        const { linear: out, hasColumn } = await readScene(joined, a.pool);
        assert.ok(hasColumn);
        assert.deepStrictEqual(Array.from(out.subarray(0, 100)), Array.from(linear));
        assert.deepStrictEqual(Array.from(out.subarray(100)), Array.from(bTrue));
    });
});

describe('stackLods with opacity_linear', () => {
    it('fills the column with sigmoid(opacity) for levels that lack it (a raw LOD 0 under decimated levels)', async () => {
        const linear = Float32Array.from({ length: 40 }, (_, i) => i / 4);
        const fine = makeOpaqueScene(100, 6);
        const fineTrue = (await readScene(fine.source, fine.pool)).linear;
        const coarse = makeOpaqueScene(40, 7, linear);
        const stacked = stackLods([fine.source, coarse.source], fine.pool);
        assert.deepStrictEqual(stacked.meta.extraColumns.map((e) => e.name), ['opacity_linear']);
        const layout = stacked.meta.layouts.other;
        const read = async (lod, count) => {
            const o = fine.pool.acquire('other', layout, count);
            await stacked.read({ chunkIndex: 0, lod, other: o });
            const values = Array.from(new Float32Array(o.data, 0, count));
            o.release();
            return values;
        };
        assert.deepStrictEqual(await read(0, 100), Array.from(fineTrue));
        assert.deepStrictEqual(await read(1, 40), Array.from(linear));
        assert.throws(() => stackLods([fine.source, coarse.source]), /mismatch between LOD levels/);
    });
});
