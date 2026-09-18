#!/usr/bin/env node
/**
 * Motion-blur slice-count comparison.
 *
 * Renders a camera orbit of `test/fixtures/motion-scene.mjs` whose speed
 * ramps up, down and up again, once per --motion-samples slice count, then
 * reports quality against a high-slice-count reference (PSNR per frame) and
 * the render time per frame, and assembles captioned 2×2 mosaics: stills at
 * the fastest and slowest frames, zoomed details, amplified difference
 * images, and an mp4 of the whole orbit. Mosaics are composed here and
 * handed to ffmpeg (on PATH) as raw RGBA, so only its encoders are needed.
 *
 * Build first (`npm run build`). Usage:
 *
 *   node tools/motion-blur-compare.mjs [--frames 48] [--res 1280x720]
 *        [--shutter 0.5] [--ref-samples 64] [--samples 1,4,16]
 *        [--out .bench/motion-blur] [--only 8,16,24] [--skip-render]
 *        [--detail cx,cy] [--zoom 6]
 *
 * Output layout: <out>/<mode>/NNNN.webp, <out>/summary.json,
 * <out>/compare.mp4, <out>/{still,detail,diff}-{fast,slow}.png.
 */
import { spawn, spawnSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { WebPCodec } from '../dist/index.mjs';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const cli = join(repoRoot, 'bin/cli.mjs');
const scene = join(repoRoot, 'test/fixtures/motion-scene.mjs');

// ---- args -------------------------------------------------------------
const argv = process.argv.slice(2);
const argValue = (name, dflt) => {
    const i = argv.indexOf(name);
    return i >= 0 && i + 1 < argv.length ? argv[i + 1] : dflt;
};
const frames = parseInt(argValue('--frames', '48'), 10);
const [width, height] = argValue('--res', '1280x720').split('x').map(Number);
const shutter = argValue('--shutter', '0.5');
const refSamples = argValue('--ref-samples', '64');
const sampleCounts = argValue('--samples', '1,4,16').split(',').map(Number);
if (sampleCounts.length !== 3) throw new Error('--samples takes exactly three slice counts (the 2×2 mosaic has three candidate tiles)');
const outDir = resolve(argValue('--out', join(repoRoot, '.bench/motion-blur')));
const only = argValue('--only', '').split(',').filter(Boolean).map(Number);
const skipRender = argv.includes('--skip-render');
const [detailCx, detailCy] = argValue('--detail', `${Math.round(width / 2)},${Math.round(height * 0.46)}`).split(',').map(Number);
const zoom = parseInt(argValue('--zoom', '6'), 10);

// Reference first, then the three candidate slice counts.
const modeNames = ['ref', ...sampleCounts.map(n => `slices-${n}`)];
const MODES = {
    ref: { label: `${refSamples} slices (reference)`, args: ['--motion-samples', refSamples] },
    ...Object.fromEntries(sampleCounts.map(n => [`slices-${n}`, { label: `${n} slice${n === 1 ? '' : 's'}`, args: ['--motion-samples', String(n)] }]))
};

// ---- camera path --------------------------------------------------------
// Orbit at radius R / height H around a fixed target. Angular speed
// dθ/du = Δθ·(1 − a·cos(3πu)): slow at u=0, fastest at 1/3, slow again at
// 2/3, fastest at 1 — speed up, slow down, speed up.
const R = 9, H = 3, TARGET = [0, 0.8, 0];
const THETA0 = -50 * Math.PI / 180, DTHETA = 100 * Math.PI / 180, A = 0.9;
const theta = u => THETA0 + DTHETA * (u - A * Math.sin(3 * Math.PI * u) / (3 * Math.PI));
const pose = (u) => {
    const t = theta(u);
    return [R * Math.sin(t), H, R * Math.cos(t)];
};
const degPerFrame = u => DTHETA * (1 - A * Math.cos(3 * Math.PI * u)) / frames * 180 / Math.PI;
const fmt = v => v.map(x => x.toFixed(6)).join(',');

const frameName = i => `${String(i).padStart(4, '0')}.webp`;
const frameList = only.length ? only : Array.from({ length: frames }, (_, i) => i);
const summaryPath = join(outDir, 'summary.json');

// ---- render ------------------------------------------------------------
// The CLI's verbose output nests timed groups by indentation: the Render
// group closes with a 4-space "done in", its Encoding child with 6.
const parseTimes = (text) => {
    const lines = text.split('\n');
    let render = NaN, encode = NaN;
    for (let i = 0; i < lines.length; i++) {
        const m = lines[i].match(/^ {4}done in ([\d.]+)s/);
        if (m) render = parseFloat(m[1]);
        if (/^ {4}▸ Encoding/.test(lines[i])) {
            for (let j = i + 1; j < lines.length; j++) {
                const e = lines[j].match(/^ {6}done in ([\d.]+)s/);
                if (e) { encode = parseFloat(e[1]); break; }
            }
        }
    }
    return { render, encode, raster: render - encode };
};

let results = {};
if (skipRender) {
    results = JSON.parse(readFileSync(summaryPath, 'utf8')).results;
} else {
    for (const mode of modeNames) {
        const spec = MODES[mode];
        const dir = join(outDir, mode);
        mkdirSync(dir, { recursive: true });
        results[mode] = {};
        process.stdout.write(`${mode.padEnd(11)}`);
        for (const i of frameList) {
            const out = join(dir, frameName(i));
            const args = [
                cli, scene,
                '--camera-pos', fmt(pose(i / frames)),
                '--camera-pos-end', fmt(pose((i + 1) / frames)),
                '--camera-target', fmt(TARGET),
                '--camera-fov', '60',
                '--resolution', `${width}x${height}`,
                '--shutter', shutter,
                ...spec.args,
                out, '-w', '--verbose'
            ];
            const t0 = performance.now();
            const r = spawnSync(process.execPath, args, { cwd: repoRoot, encoding: 'utf8', env: { ...process.env, NO_COLOR: '1' } });
            const wall = (performance.now() - t0) / 1000;
            if (r.status !== 0) {
                throw new Error(`render failed (${mode} frame ${i}):\n${r.stderr}\n${r.stdout}`);
            }
            results[mode][i] = { ...parseTimes(r.stdout + r.stderr), wall };
            process.stdout.write('.');
        }
        process.stdout.write('\n');
    }
}

// ---- quality -----------------------------------------------------------
const codec = await WebPCodec.create();
const load = path => codec.decodeRGBA(new Uint8Array(readFileSync(path))).rgba;
const psnr = (a, b) => {
    let sum = 0, n = 0;
    for (let i = 0; i < a.length; i += 4) {
        for (let c = 0; c < 3; c++) { const d = a[i + c] - b[i + c]; sum += d * d; n++; }
    }
    const mse = sum / n;
    return mse === 0 ? Infinity : 10 * Math.log10(255 * 255 / mse);
};
for (const i of frameList) {
    const ref = load(join(outDir, 'ref', frameName(i)));
    for (const mode of modeNames) {
        if (mode === 'ref') continue;
        results[mode][i].psnr = psnr(ref, load(join(outDir, mode, frameName(i))));
    }
}

// ---- summary -----------------------------------------------------------
const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
const rows = [];
for (const mode of modeNames) {
    const per = frameList.map(i => results[mode][i]);
    const row = {
        mode,
        label: MODES[mode].label,
        rasterMean: mean(per.map(p => p.raster)),
        renderMean: mean(per.map(p => p.render)),
        wallMean: mean(per.map(p => p.wall))
    };
    if (mode !== 'ref') {
        const ps = per.map(p => p.psnr).filter(Number.isFinite);
        row.psnrMean = mean(ps);
        row.psnrMin = Math.min(...ps);
    }
    rows.push(row);
}
const pad = (s, n) => String(s).padEnd(n);
console.log('');
console.log(`${pad('mode', 12)}${pad('raster s/frame', 16)}${pad('render s/frame', 16)}${pad('PSNR mean dB', 14)}${pad('PSNR min dB', 12)}`);
for (const r of rows) {
    console.log(`${pad(r.mode, 12)}${pad(r.rasterMean.toFixed(3), 16)}${pad(r.renderMean.toFixed(3), 16)}${pad(r.psnrMean?.toFixed(2) ?? '-', 14)}${pad(r.psnrMin?.toFixed(2) ?? '-', 12)}`);
}
console.log('');
console.log(`${pad('frame', 7)}${pad('deg/frame', 11)}${modeNames.filter(m => m !== 'ref').map(m => pad(`${m} dB`, 15)).join('')}`);
for (const i of frameList) {
    console.log(`${pad(i, 7)}${pad(degPerFrame((i + 0.5) / frames).toFixed(2), 11)}${modeNames.filter(m => m !== 'ref').map(m => pad(results[m][i].psnr.toFixed(2), 15)).join('')}`);
}
mkdirSync(outDir, { recursive: true });
writeFileSync(summaryPath, JSON.stringify({ frames, width, height, shutter, refSamples, sampleCounts, rows, results }, null, 2));

// ---- compositing --------------------------------------------------------
// 5×7 bitmap font for captions ('#' = lit), lowercase + digits + a few marks.
const GLYPHS = {
    a: ['.....', '.###.', '....#', '.####', '#...#', '#...#', '.####'],
    b: ['#....', '#....', '####.', '#...#', '#...#', '#...#', '####.'],
    c: ['.....', '.####', '#....', '#....', '#....', '#....', '.####'],
    d: ['....#', '....#', '.####', '#...#', '#...#', '#...#', '.####'],
    e: ['.....', '.###.', '#...#', '#####', '#....', '#...#', '.###.'],
    f: ['..###', '.#...', '.#...', '####.', '.#...', '.#...', '.#...'],
    g: ['.....', '.####', '#...#', '#...#', '.####', '....#', '.###.'],
    h: ['#....', '#....', '####.', '#...#', '#...#', '#...#', '#...#'],
    i: ['..#..', '.....', '.##..', '..#..', '..#..', '..#..', '.###.'],
    j: ['...#.', '.....', '..##.', '...#.', '...#.', '#..#.', '.##..'],
    k: ['#....', '#....', '#..#.', '#.#..', '##...', '#.#..', '#..#.'],
    l: ['.##..', '..#..', '..#..', '..#..', '..#..', '..#..', '.###.'],
    m: ['.....', '.....', '##.#.', '#.#.#', '#.#.#', '#...#', '#...#'],
    n: ['.....', '.....', '####.', '#...#', '#...#', '#...#', '#...#'],
    o: ['.....', '.....', '.###.', '#...#', '#...#', '#...#', '.###.'],
    p: ['.....', '####.', '#...#', '#...#', '####.', '#....', '#....'],
    q: ['.....', '.####', '#...#', '#...#', '.####', '....#', '....#'],
    r: ['.....', '.....', '#.##.', '##..#', '#....', '#....', '#....'],
    s: ['.....', '.....', '.####', '#....', '.###.', '....#', '####.'],
    t: ['.#...', '.#...', '####.', '.#...', '.#...', '.#..#', '..##.'],
    u: ['.....', '.....', '#...#', '#...#', '#...#', '#..##', '.##.#'],
    v: ['.....', '.....', '#...#', '#...#', '#...#', '.#.#.', '..#..'],
    w: ['.....', '.....', '#...#', '#...#', '#.#.#', '#.#.#', '.#.#.'],
    x: ['.....', '.....', '#...#', '.#.#.', '..#..', '.#.#.', '#...#'],
    y: ['.....', '.....', '#...#', '#...#', '.####', '....#', '.###.'],
    z: ['.....', '.....', '#####', '...#.', '..#..', '.#...', '#####'],
    0: ['.###.', '#...#', '#..##', '#.#.#', '##..#', '#...#', '.###.'],
    1: ['..#..', '.##..', '..#..', '..#..', '..#..', '..#..', '.###.'],
    2: ['.###.', '#...#', '....#', '...#.', '..#..', '.#...', '#####'],
    3: ['#####', '...#.', '..#..', '...#.', '....#', '#...#', '.###.'],
    4: ['...#.', '..##.', '.#.#.', '#..#.', '#####', '...#.', '...#.'],
    5: ['#####', '#....', '####.', '....#', '....#', '#...#', '.###.'],
    6: ['..##.', '.#...', '#....', '####.', '#...#', '#...#', '.###.'],
    7: ['#####', '....#', '...#.', '..#..', '.#...', '.#...', '.#...'],
    8: ['.###.', '#...#', '#...#', '.###.', '#...#', '#...#', '.###.'],
    9: ['.###.', '#...#', '#...#', '.####', '....#', '...#.', '.##..'],
    ' ': ['.....', '.....', '.....', '.....', '.....', '.....', '.....'],
    '(': ['..#..', '.#...', '#....', '#....', '#....', '.#...', '..#..'],
    ')': ['..#..', '...#.', '....#', '....#', '....#', '...#.', '..#..'],
    '|': ['..#..', '..#..', '..#..', '..#..', '..#..', '..#..', '..#..'],
    '-': ['.....', '.....', '.....', '#####', '.....', '.....', '.....'],
    '.': ['.....', '.....', '.....', '.....', '.....', '.##..', '.##..'],
    ',': ['.....', '.....', '.....', '.....', '.##..', '.##..', '.#...'],
    '/': ['....#', '....#', '...#.', '..#..', '.#...', '#....', '#....'],
    ':': ['.....', '.##..', '.##..', '.....', '.##..', '.##..', '.....']
};

const fillRect = (img, w, x0, y0, rw, rh, rgba) => {
    for (let y = y0; y < y0 + rh; y++) {
        for (let x = x0; x < x0 + rw; x++) {
            const o = (y * w + x) * 4;
            img[o] = rgba[0]; img[o + 1] = rgba[1]; img[o + 2] = rgba[2]; img[o + 3] = rgba[3];
        }
    }
};

// Caption in the tile's top-left corner over a translucent dark box.
const drawCaption = (img, w, x0, y0, text, scale) => {
    const glyphW = 6 * scale, glyphH = 7 * scale, padPx = 2 * scale;
    const boxW = text.length * glyphW + 2 * padPx, boxH = glyphH + 2 * padPx;
    for (let y = y0; y < y0 + boxH; y++) {
        for (let x = x0; x < x0 + boxW; x++) {
            const o = (y * w + x) * 4;
            img[o] = img[o] * 0.35; img[o + 1] = img[o + 1] * 0.35; img[o + 2] = img[o + 2] * 0.35;
        }
    }
    for (let k = 0; k < text.length; k++) {
        const g = GLYPHS[text[k].toLowerCase()] ?? GLYPHS[' '];
        for (let r = 0; r < 7; r++) {
            for (let c = 0; c < 5; c++) {
                if (g[r][c] === '#') {
                    fillRect(img, w, x0 + padPx + k * glyphW + c * scale, y0 + padPx + r * scale, scale, scale, [255, 255, 255, 255]);
                }
            }
        }
    }
};

// Nearest-neighbour zoom of a crop centred on (cx, cy) into a tile of the
// full frame size.
const zoomCrop = (src, cx, cy, z) => {
    const cw = Math.floor(width / z), ch = Math.floor(height / z);
    const x0 = Math.max(0, Math.min(width - cw, Math.round(cx - cw / 2)));
    const y0 = Math.max(0, Math.min(height - ch, Math.round(cy - ch / 2)));
    const out = new Uint8Array(width * height * 4);
    for (let y = 0; y < height; y++) {
        const sy = y0 + Math.min(ch - 1, Math.floor(y / z));
        for (let x = 0; x < width; x++) {
            const sx = x0 + Math.min(cw - 1, Math.floor(x / z));
            const s = (sy * width + sx) * 4, o = (y * width + x) * 4;
            out[o] = src[s]; out[o + 1] = src[s + 1]; out[o + 2] = src[s + 2]; out[o + 3] = 255;
        }
    }
    return out;
};

const absDiff = (a, b, gain) => {
    const out = new Uint8Array(a.length);
    for (let i = 0; i < a.length; i += 4) {
        for (let c = 0; c < 3; c++) out[i + c] = Math.min(255, Math.abs(a[i + c] - b[i + c]) * gain);
        out[i + 3] = 255;
    }
    return out;
};

// 2×2 mosaic of full-frame tiles with captions.
const mosaic = (tiles, captions) => {
    const W = width * 2, Hh = height * 2;
    const out = new Uint8Array(W * Hh * 4);
    const scale = Math.max(2, Math.round(height / 180));
    tiles.forEach((tile, k) => {
        const ox = (k % 2) * width, oy = Math.floor(k / 2) * height;
        for (let y = 0; y < height; y++) {
            out.set(tile.subarray(y * width * 4, (y + 1) * width * 4), ((oy + y) * W + ox) * 4);
        }
        drawCaption(out, W, ox + 3 * scale, oy + 3 * scale, captions[k], scale);
    });
    return { rgba: out, w: W, h: Hh };
};

const ffmpegOk = spawnSync('ffmpeg', ['-version'], { encoding: 'utf8' }).status === 0;
if (!ffmpegOk || only.length) {
    console.log('\nffmpeg unavailable or partial run: skipping stills/video');
    process.exit(0);
}

const writePng = ({ rgba, w, h }, path) => {
    const r = spawnSync('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', `${w}x${h}`, '-i', '-', '-frames:v', '1', path], { input: rgba });
    if (r.status !== 0) console.log(`ffmpeg failed for ${path}:\n${r.stderr}`);
    else console.log(`wrote ${path}`);
};

const fastFrame = Math.round(frames / 3), slowFrame = Math.round(2 * frames / 3);
for (const [tag, i] of [['fast', fastFrame], ['slow', slowFrame]]) {
    const imgs = modeNames.map(m => load(join(outDir, m, frameName(i))));
    const info = `frame ${i}, ${degPerFrame((i + 0.5) / frames).toFixed(2)} deg/frame`;
    const captions = modeNames.map((m, k) => (k === 0 ? `${MODES[m].label} - ${info}` : `${MODES[m].label}${results[m][i].psnr ? ` ${results[m][i].psnr.toFixed(1)} db` : ''}`));
    writePng(mosaic(imgs, captions), join(outDir, `still-${tag}.png`));
    writePng(mosaic(imgs.map(im => zoomCrop(im, detailCx, detailCy, zoom)), captions.map(c => `${c} (${zoom}x)`)), join(outDir, `detail-${tag}.png`));
    const gain = 8;
    writePng(mosaic(
        [imgs[0], ...imgs.slice(1).map(im => absDiff(imgs[0], im, gain))],
        [`${MODES.ref.label} - ${info}`, ...modeNames.slice(1).map(m => `|reference - ${m}| x${gain}`)]
    ), join(outDir, `diff-${tag}.png`));
}

// Video: compose every frame's mosaic and stream raw RGBA into ffmpeg.
await new Promise((done) => {
    const W = width * 2, Hh = height * 2;
    const ff = spawn('ffmpeg', [
        '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', `${W}x${Hh}`, '-framerate', '24', '-i', '-',
        '-c:v', 'libx264', '-crf', '14', '-pix_fmt', 'yuv420p', join(outDir, 'compare.mp4')
    ], { stdio: ['pipe', 'inherit', 'inherit'] });
    ff.on('close', (code) => {
        console.log(code === 0 ? `wrote ${join(outDir, 'compare.mp4')}` : `ffmpeg video failed (${code})`);
        done();
    });
    (async () => {
        for (const i of frameList) {
            const imgs = modeNames.map(m => load(join(outDir, m, frameName(i))));
            const info = `frame ${i}, ${degPerFrame((i + 0.5) / frames).toFixed(2)} deg/frame`;
            const captions = modeNames.map((m, k) => (k === 0 ? `${MODES[m].label} - ${info}` : `${MODES[m].label} ${results[m][i].psnr.toFixed(1)} db`));
            const { rgba } = mosaic(imgs, captions);
            if (!ff.stdin.write(rgba)) await new Promise(res => ff.stdin.once('drain', res));
        }
        ff.stdin.end();
    })();
});
