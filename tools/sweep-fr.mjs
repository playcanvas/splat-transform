#!/usr/bin/env node
// Full-scale confirmation on frustum-culled scenes: old vs new4 vs b4,
// 6 levels, poses −10°/0°/+10° (inside the +12°-widened culling frustum).
// Renders under scenes/sweep-fr/<scene>/; PSNR vs the culled source.
//
// Prerequisite: npm run build (imports WebPCodec from ../dist).
// Usage: node tools/sweep-fr.mjs [sky|snow]
import { execFileSync } from 'node:child_process';
import { readFileSync, mkdirSync, existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { WebPCodec } from '../dist/index.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const NODE = process.execPath;
const CLI = `${ROOT}/bin/cli.mjs`;

const L = [1, 2, 3, 4, 5, 6];
const SCENES = {
    sky: {
        source: `${ROOT}/scenes/fr-sky.ply`,
        P: [-5.632, 0.692, -2.550],
        T: [-1.578, 1.265, -1.143],
        methods: {
            old: L.map(l => `${ROOT}/scenes/fr-sky.old-${l}.ply`),
            new4: L.map(l => `${ROOT}/scenes/fr-sky.new4-${l}.ply`),
            b4: L.map(l => `${ROOT}/scenes/fr-sky.b4-${l}1.ply`),
            cw6: L.map(l => `${ROOT}/scenes/fr-sky.cw6-${l}1.ply`),
            new4r: L.map(l => `${ROOT}/scenes/fr-sky.new4r-${l}.ply`)
        }
    },
    snow: {
        source: `${ROOT}/scenes/fr-snow.ply`,
        P: [-143.774, 29.955, -27.240],
        T: [-108.901, 10.650, -46.631],
        methods: {
            old: L.map(l => `${ROOT}/scenes/fr-snow.old-${l}.ply`),
            new4: L.map(l => `${ROOT}/scenes/fr-snow.new4-${l}.ply`),
            b4: L.map(l => `${ROOT}/scenes/fr-snow.b4-${l}1.ply`),
            cw6: L.map(l => `${ROOT}/scenes/fr-snow.cw6-${l}1.ply`),
            new4r: L.map(l => `${ROOT}/scenes/fr-snow.new4r-${l}.ply`)
        }
    }
};
const ANGLES = [-10, 0, 10];

const which = process.argv[2] || 'sky';
const cfg = SCENES[which];
const OUT = `${ROOT}/scenes/sweep-fr/${which}`;
mkdirSync(OUT, { recursive: true });

const { P, T } = cfg;
const oy = P[1] - T[1];
const r = Math.hypot(P[0] - T[0], P[2] - T[2]);
const theta0 = Math.atan2(P[2] - T[2], P[0] - T[0]);

const render = (input, out, pos) => {
    if (existsSync(out)) return;
    execFileSync(NODE, [CLI, input, out, '--camera-pos', pos.join(','), '--camera-target', T.join(',')], { stdio: 'ignore' });
};

const codec = await WebPCodec.create();
const psnr = (aPath, bPath) => {
    const a = codec.decodeRGBA(readFileSync(aPath));
    const b = codec.decodeRGBA(readFileSync(bPath));
    const n = a.width * a.height;
    let se = 0;
    for (let i = 0; i < n; i++) {
        for (let c = 0; c < 3; c++) {
            const d = a.rgba[i * 4 + c] - b.rgba[i * 4 + c];
            se += d * d;
        }
    }
    const mse = se / (n * 3);
    return mse === 0 ? Infinity : 10 * Math.log10((255 * 255) / mse);
};

const methods = Object.keys(cfg.methods);
const perPose = {};
for (const ang of ANGLES) {
    const th = theta0 + (ang * Math.PI) / 180;
    const pos = [T[0] + r * Math.cos(th), T[1] + oy, T[2] + r * Math.sin(th)];
    const ref = `${OUT}/src_${ang}.webp`;
    render(cfg.source, ref, pos);
    perPose[ang] = {};
    for (const l of L) {
        perPose[ang][l] = {};
        for (const m of methods) {
            const ply = cfg.methods[m][l - 1];
            if (!existsSync(ply)) { perPose[ang][l][m] = null; continue; }
            const out = `${OUT}/${m}${l}_${ang}.webp`;
            render(ply, out, pos);
            perPose[ang][l][m] = psnr(ref, out);
        }
    }
}

console.log(`\n=== fr-${which} ===  PSNR dB vs culled source (mean over ${ANGLES.length} poses; pose 0 = user camera)`);
console.log(`  level  ${methods.map(m => m.padStart(8)).join('  ')}`);
for (const l of L) {
    const cells = methods.map((m) => {
        const vals = ANGLES.map(a => perPose[a][l][m]).filter(v => v != null);
        return (vals.length ? (vals.reduce((x, y) => x + y, 0) / vals.length).toFixed(2) : '   —').padStart(8);
    });
    console.log(`  L${l}     ${cells.join('  ')}`);
}
for (const ang of ANGLES) {
    console.log(`pose ${ang}°`);
    for (const l of L) {
        const cells = methods.map(m => (perPose[ang][l][m] == null ? '   —' : perPose[ang][l][m].toFixed(2)).padStart(8));
        console.log(`  L${l}  ${cells.join('  ')}`);
    }
}
