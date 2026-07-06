/**
 * Synthetic gaussian-splat scenes for decimation tests: random splats
 * assembled into an InMemoryChunkSource at exact layer strides, plus the
 * flat view/position arrays the reference implementations consume.
 */

import { createInMemoryChunkSource, createChunkDataPool } from '../../src/lib/chunk/index.js';
import { Transform } from '../../src/lib/utils/index.js';

const mulberry = (seed) => {
    let t = seed >>> 0;
    return () => {
        t += 0x6d2b79f5;
        let r = Math.imul(t ^ (t >>> 15), t | 1);
        r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
        return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
};

const SH_REST = { 0: 0, 1: 9, 2: 24, 3: 45 };

/**
 * Build a random scene. Positions span [0,10)³; scales are small relative to
 * point spacing so merge costs stay well-conditioned.
 *
 * @param {number} n - Gaussian count.
 * @param {0|1|2|3} shBands - SH band count.
 * @param {number} seed - Deterministic seed.
 * @param {object} [opts] - Options: chunkSize (default 1024), extraColumns
 * (array of {name, type:'uint32'|'float32'}) to add an `other` layer.
 * @returns {{source, pool, view, pos, buffers}}
 */
export const makeSyntheticSource = async (n, shBands, seed, opts = {}) => {
    const chunkSize = opts.chunkSize ?? 1024;
    const colorDim = 3 + SH_REST[shBands];
    const r = mulberry(seed);

    const pos = new Float32Array(n * 3);
    const geo = new Float32Array(n * 8);
    const color = new Float32Array(n * colorDim);
    for (let i = 0; i < n; i++) {
        pos.set([r() * 10, r() * 10, r() * 10], i * 3);
        const q = [r() - 0.5, r() - 0.5, r() - 0.5, r() - 0.5];
        const qn = Math.hypot(...q) || 1;
        geo.set([
            q[0] / qn, q[1] / qn, q[2] / qn, q[3] / qn,
            Math.log(r() * 0.05 + 0.01), Math.log(r() * 0.05 + 0.01), Math.log(r() * 0.05 + 0.01),
            r() * 8 - 4
        ], i * 8);
        for (let c = 0; c < colorDim; c++) color[i * colorDim + c] = r() * 2 - 1;
    }
    const view = { pos, geo, color, colorDim };

    const extras = opts.extraColumns ?? [];
    const otherDim = extras.length;
    let other = null;
    if (otherDim > 0) {
        other = new Uint32Array(n * otherDim);
        for (let i = 0; i < n * otherDim; i++) other[i] = (r() * 0xFFFFFFFF) >>> 0;
    }

    const chunkify = (flat, comps, bytesPer) => {
        const bufs = [];
        for (let base = 0; base < n; base += chunkSize) {
            const count = Math.min(chunkSize, n - base);
            bufs.push(flat.slice(base * comps, (base + count) * comps).buffer);
        }
        void bytesPer;
        return bufs;
    };

    const source = createInMemoryChunkSource({
        numGaussians: n,
        chunkSize,
        shBands,
        extraColumns: extras,
        transform: new Transform(),
        lodCounts: [n],
        position: [chunkify(pos, 3)],
        geometric: [chunkify(geo, 8)],
        color: [chunkify(color, colorDim)],
        ...(other ? { other: [chunkify(other, otherDim)] } : {})
    });

    const pool = createChunkDataPool({ chunkSize });
    const posCols = {
        x: Float32Array.from({ length: n }, (_, i) => pos[i * 3]),
        y: Float32Array.from({ length: n }, (_, i) => pos[i * 3 + 1]),
        z: Float32Array.from({ length: n }, (_, i) => pos[i * 3 + 2])
    };

    return {
        source,
        pool,
        view,
        other,
        otherDim,
        pos: posCols,
        buffers: {
            position: [chunkify(pos, 3)],
            geometric: [chunkify(geo, 8)],
            color: [chunkify(color, colorDim)]
        }
    };
};
