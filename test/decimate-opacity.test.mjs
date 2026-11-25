/**
 * True opacity under decimation: merges keep their full mass, above 1 too, in
 * the opacity slot, flag the output `opacityAboveOne`, and a later generation
 * reads that mass back, including through a PLY file (written with
 * `opacity_linear` in place of `opacity`).
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

import { createChunkDataPool, createInMemoryChunkSource } from '../src/lib/chunk/index.js';
import { decimateSourceAdaptive } from '../src/lib/decimate/index.js';
import { decimateSource } from '../src/lib/decimate-uniform/index.js';
import { MemoryFileSystem, MemoryReadFileSystem } from '../src/lib/index.js';
import { readPly } from '../src/lib/readers/read-ply.js';
import { Transform } from '../src/lib/utils/index.js';
import { writePlyStreaming } from '../src/lib/writers/write-ply-streaming.js';

const mulberry = (seed) => {
    let t = seed >>> 0;
    return () => {
        t += 0x6d2b79f5;
        let r = Math.imul(t ^ (t >>> 15), t | 1);
        r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
        return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
};

// Knud Thomsen ellipsoid surface approximation, as the decimators use.
const P = 1.6075;
const area = (sx, sy, sz) => 4 * Math.PI * (((sx * sy) ** P + (sx * sz) ** P + (sy * sz) ** P) / 3) ** (1 / P);

/**
 * Dense, nearly opaque, heavily overlapping splats, so merges exceed opacity 1.
 *
 * @param {number} n - Gaussian count.
 * @param {number} seed - Seed.
 * @returns {{source, pool}} Source and pool.
 */
const makeOpaqueScene = (n, seed) => {
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
        const opacity = 1 / (1 + Math.exp(-(3 + r() * 2))); // true opacity, 0.95 to 0.99
        geo.set([q[0] / qn, q[1] / qn, q[2] / qn, q[3] / qn, s(), s(), s(), opacity], i * 8);
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
        transform: new Transform(),
        lodCounts: [n],
        position: [chunks(pos, 3)],
        geometric: [chunks(geo, 8)],
        color: [chunks(color, 3)]
    });
    return { source, pool: createChunkDataPool({ chunkSize }) };
};

// Read the geometric layer of a whole source.
const readGeometric = async (src, pool) => {
    const { meta } = src;
    const n = meta.numGaussians;
    const geo = new Float32Array(n * 8);
    let base = 0;
    for (let k = 0; k < meta.numChunks[0]; k++) {
        const count = Math.min(meta.chunkSize, n - base);
        const g = pool.acquire('geometric', meta.layouts.geometric, count);
        await src.read({ chunkIndex: k, geometric: g });
        geo.set(new Float32Array(g.data, 0, count * 8), base * 8);
        g.release();
        base += count;
    }
    return geo;
};

const totalMass = (geo) => {
    let m = 0;
    for (let o = 0; o < geo.length; o += 8) {
        m += geo[o + 7] * area(Math.exp(geo[o + 4]), Math.exp(geo[o + 5]), Math.exp(geo[o + 6]));
    }
    return m;
};

// Write a source to PLY and read it back, with the PLY header text.
const throughPly = async (src, pool) => {
    const fs = new MemoryFileSystem();
    await writePlyStreaming(src, pool, { filename: 'level.ply' }, fs);
    const bytes = fs.results.get('level.ply');
    const rfs = new MemoryReadFileSystem();
    rfs.set('level.ply', bytes);
    const header = new TextDecoder().decode(bytes.subarray(0, 1024)).split('end_header')[0];
    return { source: await readPly(await rfs.createSource('level.ply'), pool), header };
};

const decimators = [
    ['uniform', decimateSource],
    ['adaptive', decimateSourceAdaptive]
];

describe('decimation true opacity', () => {
    for (const [name, decimate] of decimators) {
        it(`${name}: merges keep their full mass, above 1 too, and flag it`, async () => {
            const { source, pool } = makeOpaqueScene(3000, 1);
            const input = await readGeometric(source, pool);
            const out = await decimate(source, pool, { targetCount: 750 });
            assert.strictEqual(out.meta.opacityAboveOne, true);
            assert.deepStrictEqual(out.meta.extraColumns, [], 'no extra column: the opacity slot holds it');
            const result = await readGeometric(out, pool);
            await out.close();

            let over = 0;
            for (let o = 7; o < result.length; o += 8) {
                assert.ok(result[o] > 0 && Number.isFinite(result[o]), `opacity ${result[o]}`);
                if (result[o] > 1) over++;
            }
            assert.ok(over > 0, 'the scene is dense enough for merges to exceed opacity 1');
            const rel = Math.abs(totalMass(result) / totalMass(input) - 1);
            assert.ok(rel < 0.02, `mass drift ${rel}`);
        });

        it(`${name}: a second generation reads the mass back through a linear PLY`, async () => {
            const { source, pool } = makeOpaqueScene(3000, 2);
            const input = await readGeometric(source, pool);
            const level1 = await decimate(source, pool, { targetCount: 1500 });
            // decimate output is one streaming pass: spill it to PLY, as the CLI does
            const { source: reread, header } = await throughPly(level1, pool);
            await level1.close();
            assert.match(header, /property float opacity_linear\n/);
            assert.doesNotMatch(header, /property float opacity\n/);
            assert.strictEqual(reread.meta.opacityAboveOne, true);
            const level2 = await decimate(reread, pool, { targetCount: 500 });
            const result = await readGeometric(level2, pool);
            await level2.close();
            const rel = Math.abs(totalMass(result) / totalMass(input) - 1);
            assert.ok(rel < 0.03, `mass drift over two generations ${rel}`);
        });
    }
});
