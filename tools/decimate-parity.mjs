#!/usr/bin/env node
/**
 * Output-parity check for `--decimate` against a reference build.
 *
 * The uniform decimator is bit-for-bit output-compatible with the 3.1.x
 * release, which is what makes it a usable reference baseline (see
 * src/lib/decimate-uniform/README.md and the `old` column in
 * scenes/DECIMATION-RESULTS.md). This script is how that claim is checked:
 * run N chained halvings through a reference binary's `--decimate` and this
 * working tree's `--decimate`, compare the outputs byte for byte, and report
 * PSNR for both against the undecimated source.
 *
 * Exits non-zero if any level differs, so it can gate a change.
 *
 * Prerequisites:
 *   - npm run build (imports WebPCodec from ../dist)
 *   - a reference binary on PATH, or --ref <path>. It is invoked with
 *     `--decimate`, so that flag must be the uniform algorithm there: any
 *     3.1.x build (3.2.x spelled it `--decimate-uniform`).
 *
 * Usage:
 *   node tools/decimate-parity.mjs [sky|snow] [options]
 *
 *   --ref <bin>        reference binary (default: splat-transform on PATH)
 *   --input <ply>      override the preset's scene (cameras still come from it)
 *   --halvings <n>     chained 50% levels (default: 6)
 *   --out <dir>        working directory (default: scenes/parity/<scene>)
 *   --skip-render      byte comparison only; no renders, no PSNR
 *
 * The preset scenes are the two the study uses. They live under scenes/, which
 * is gitignored — regenerate them with tools/frustum-cull.mjs if absent.
 */
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { WebPCodec } from '../dist/index.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const NODE = process.execPath;
const CLI = `${ROOT}/bin/cli.mjs`;

const SCENES = {
    sky: {
        source: `${ROOT}/scenes/fr-sky.ply`,
        P: [-5.632, 0.692, -2.550],
        T: [-1.578, 1.265, -1.143]
    },
    snow: {
        source: `${ROOT}/scenes/fr-snow.ply`,
        P: [-143.774, 29.955, -27.240],
        T: [-108.901, 10.650, -46.631]
    }
};
const ANGLES = [-10, 0, 10];

const argv = process.argv.slice(2);
const flag = (name, fallback) => {
    const i = argv.indexOf(`--${name}`);
    return i >= 0 && argv[i + 1] !== undefined ? argv[i + 1] : fallback;
};
const which = argv[0] && !argv[0].startsWith('--') ? argv[0] : 'sky';
if (!SCENES[which]) {
    console.error(`unknown scene "${which}" — expected one of: ${Object.keys(SCENES).join(', ')}`);
    process.exit(2);
}

const { P, T } = SCENES[which];
const source = resolve(flag('input', SCENES[which].source));
const ref = flag('ref', 'splat-transform');
const halvings = parseInt(flag('halvings', '6'), 10);
const outDir = resolve(flag('out', `${ROOT}/scenes/parity/${which}`));
const skipRender = argv.includes('--skip-render');

if (!existsSync(source)) {
    console.error(`scene not found: ${source}\nscenes/ is gitignored — regenerate with tools/frustum-cull.mjs, or pass --input.`);
    process.exit(2);
}
try {
    execFileSync(ref, ['--version'], { stdio: 'ignore' });
} catch {
    console.error(`reference binary not runnable: ${ref}\nInstall the reference release or pass --ref <path>.`);
    process.exit(2);
}
mkdirSync(outDir, { recursive: true });

const sha256 = path => createHash('sha256').update(readFileSync(path)).digest('hex');
const vertexCount = (path) => {
    const head = readFileSync(path).subarray(0, 2048).toString('ascii');
    const m = head.match(/element vertex (\d+)/);
    return m ? parseInt(m[1], 10) : -1;
};

// Chain `halvings` 50% steps, timing each. `ref` runs its own --decimate (the
// uniform algorithm at that revision); the working tree runs the explicit flag.
const chain = (label, argsFor) => {
    const paths = [];
    const secs = [];
    let input = source;
    for (let l = 1; l <= halvings; l++) {
        const out = `${outDir}/${which}.${label}-${l}.ply`;
        if (!existsSync(out)) {
            const started = Date.now();
            const [bin, ...pre] = argsFor.bin;
            execFileSync(bin, [...pre, input, ...argsFor.flags, '50%', out, '-w'], { stdio: 'ignore' });
            secs.push((Date.now() - started) / 1000);
        } else {
            secs.push(NaN);   // cached from an earlier run
        }
        paths.push(out);
        input = out;
    }
    return { paths, secs };
};

console.log(`\n=== ${which}: ${ref} --decimate  vs  this tree --decimate ===`);
console.log(`source ${source} (${vertexCount(source)} splats), ${halvings} halvings\n`);

const a = chain('ref', { bin: [ref], flags: ['--decimate'] });
const b = chain('uni', { bin: [NODE, CLI], flags: ['--decimate'] });

let mismatches = 0;
console.log('level        count    ref s    uni s   identical');
for (let l = 0; l < halvings; l++) {
    const identical = sha256(a.paths[l]) === sha256(b.paths[l]);
    if (!identical) mismatches++;
    const ca = vertexCount(a.paths[l]);
    const cb = vertexCount(b.paths[l]);
    console.log(
        `L${l + 1}   ${String(ca).padStart(10)}${ca === cb ? '' : ` (uni ${cb})`}  ` +
        `${a.secs[l].toFixed(1).padStart(7)}  ${b.secs[l].toFixed(1).padStart(7)}   ${identical ? 'YES' : 'NO'}`
    );
}
const total = s => s.reduce((x, y) => x + (Number.isNaN(y) ? 0 : y), 0).toFixed(1);
console.log(`\ntotal cascade: ref ${total(a.secs)}s   uniform ${total(b.secs)}s`);

if (!skipRender) {
    const codec = await WebPCodec.create();
    const render = (input, out, pos) => {
        if (existsSync(out)) return;
        execFileSync(NODE, [CLI, input, out, '--camera-pos', pos.join(','), '--camera-target', T.join(','), '-w'], { stdio: 'ignore' });
    };
    const rgbaOf = async (p) => {
        const { rgba } = await codec.decodeRGBA(new Uint8Array(readFileSync(p)));
        return rgba;
    };
    const psnr = (x, y) => {
        let sum = 0, n = 0;
        for (let i = 0; i < x.length; i += 4) {
            for (let c = 0; c < 3; c++) {
                const d = x[i + c] - y[i + c];
                sum += d * d;
                n++;
            }
        }
        const mse = sum / n;
        return mse === 0 ? Infinity : 10 * Math.log10(255 * 255 / mse);
    };

    const oy = P[1] - T[1];
    const r = Math.hypot(P[0] - T[0], P[2] - T[2]);
    const theta0 = Math.atan2(P[2] - T[2], P[0] - T[0]);
    const acc = { ref: new Array(halvings).fill(0), uni: new Array(halvings).fill(0) };

    for (const ang of ANGLES) {
        const th = theta0 + ang * Math.PI / 180;
        const pos = [T[0] + r * Math.cos(th), T[1] + oy, T[2] + r * Math.sin(th)];
        const srcRender = `${outDir}/${which}.src_${ang}.webp`;
        render(source, srcRender, pos);
        const srcRgba = await rgbaOf(srcRender);
        for (const [label, chained] of [['ref', a], ['uni', b]]) {
            for (let l = 0; l < halvings; l++) {
                const out = `${outDir}/${which}.${label}-${l + 1}_${ang}.webp`;
                render(chained.paths[l], out, pos);
                acc[label][l] += psnr(srcRgba, await rgbaOf(out)) / ANGLES.length;
            }
        }
    }

    console.log(`\nPSNR dB vs source (mean of ${ANGLES.length} poses)\n`);
    console.log('level      ref      uni     delta');
    for (let l = 0; l < halvings; l++) {
        const d = acc.uni[l] - acc.ref[l];
        console.log(
            `L${l + 1}    ${acc.ref[l].toFixed(2).padStart(7)}  ${acc.uni[l].toFixed(2).padStart(7)}  ` +
            `${(d >= 0 ? '+' : '') + d.toFixed(2)}`
        );
    }
}

console.log(mismatches === 0 ?
    `\nPASS — all ${halvings} levels byte-identical to ${ref}` :
    `\nFAIL — ${mismatches} of ${halvings} level(s) differ`);
console.log(`artifacts: ${outDir}`);
process.exit(mismatches === 0 ? 0 : 1);
