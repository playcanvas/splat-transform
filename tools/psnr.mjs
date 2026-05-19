#!/usr/bin/env node
// PSNR comparison for two or more WebP images (must be same dimensions).
//
// Prerequisite: `npm run build` (this script imports `WebPCodec` from
// `../dist/index.mjs`, which doesn't exist until the package is built).
//
// Usage:
//   node tools/psnr.mjs <reference.webp> <candidate.webp> [candidate2.webp ...]
//
// For each candidate, reports PSNR (RGB), mean-luma delta vs reference,
// mean-saturation delta vs reference, and per-channel mean absolute
// difference. Higher PSNR / smaller MAD = closer to the reference.

import { readFile } from 'node:fs/promises';
import { WebPCodec } from '../dist/index.mjs';

const argv = process.argv.slice(2);
if (argv.length < 2) {
    console.error('usage: node tools/psnr.mjs <reference.webp> <candidate.webp> [candidate2 ...]');
    process.exit(2);
}

const codec = await WebPCodec.create();

const loadRgba = async (path) => {
    const buf = new Uint8Array(await readFile(path));
    return codec.decodeRGBA(buf);
};

const psnr = (refRgba, cand) => {
    if (refRgba.length !== cand.length) throw new Error('size mismatch');
    let sumSq = 0;
    let count = 0;
    for (let i = 0; i < refRgba.length; i += 4) {
        for (let c = 0; c < 3; c++) {
            const d = refRgba[i + c] - cand[i + c];
            sumSq += d * d;
            count++;
        }
    }
    const mse = sumSq / count;
    if (mse === 0) return Infinity;
    return 10 * Math.log10((255 * 255) / mse);
};

// Per-channel mean absolute difference, in display 0-255 units.
const channelMAD = (refRgba, cand) => {
    let r = 0, g = 0, b = 0, n = 0;
    for (let i = 0; i < refRgba.length; i += 4) {
        r += Math.abs(refRgba[i] - cand[i]);
        g += Math.abs(refRgba[i + 1] - cand[i + 1]);
        b += Math.abs(refRgba[i + 2] - cand[i + 2]);
        n++;
    }
    return { r: r / n, g: g / n, b: b / n };
};

// Mean luma (Rec.601) for tone-shift detection.
const meanLuma = (rgba) => {
    let sum = 0;
    const n = rgba.length / 4;
    for (let i = 0; i < rgba.length; i += 4) {
        sum += 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
    }
    return sum / n;
};

// Mean saturation: simple HSV-style saturation = (max-min)/max for each pixel.
const meanSaturation = (rgba) => {
    let sum = 0;
    const n = rgba.length / 4;
    for (let i = 0; i < rgba.length; i += 4) {
        const r = rgba[i], g = rgba[i + 1], b = rgba[i + 2];
        const mx = Math.max(r, g, b);
        const mn = Math.min(r, g, b);
        sum += mx > 0 ? (mx - mn) / mx : 0;
    }
    return sum / n;
};

const ref = await loadRgba(argv[0]);
console.log(`reference: ${argv[0]} ${ref.width}x${ref.height}`);
const refLuma = meanLuma(ref.rgba);
const refSat = meanSaturation(ref.rgba);
console.log(`  reference luma=${refLuma.toFixed(2)} sat=${refSat.toFixed(4)}`);
console.log('');

const pad = (s, n) => s + ' '.repeat(Math.max(0, n - s.length));
console.log(pad('candidate', 40) + 'PSNR(dB)  ΔLuma   ΔSat    MAD(R/G/B)');
console.log('-'.repeat(86));

for (const path of argv.slice(1)) {
    const cand = await loadRgba(path);
    if (cand.width !== ref.width || cand.height !== ref.height) {
        console.log(`${pad(path.split('/').pop(), 40)}  size mismatch (${cand.width}x${cand.height})`);
        continue;
    }
    const p = psnr(ref.rgba, cand.rgba);
    const mad = channelMAD(ref.rgba, cand.rgba);
    const candLuma = meanLuma(cand.rgba);
    const candSat = meanSaturation(cand.rgba);
    const dL = candLuma - refLuma;
    const dS = candSat - refSat;
    const pStr = p === Infinity ? 'inf' : p.toFixed(2);
    console.log(
        pad(path.split('/').pop(), 40) +
        pad(pStr, 9) + ' ' +
        pad((dL >= 0 ? '+' : '') + dL.toFixed(2), 8) +
        pad((dS >= 0 ? '+' : '') + dS.toFixed(4), 8) +
        `${mad.r.toFixed(2)} / ${mad.g.toFixed(2)} / ${mad.b.toFixed(2)}`
    );
}
