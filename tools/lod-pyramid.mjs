#!/usr/bin/env node
/**
 * Large-scene LOD-pyramid test: from one big PLY, decimate 50% repeatedly down
 * to <= --target gaussians, then stack all levels (finest = the original, LOD 0)
 * into a single lod-meta.json. Runs the whole recipe through one CLI version and
 * times every step under /usr/bin/time -l (wall + peak RSS). v2 also emits its
 * own peak cpu/gpu via --mem.
 *
 *   node tools/lod-pyramid.mjs --ver v2 --input <big.ply> [--target 1000000]
 *   node tools/lod-pyramid.mjs --ver v1 --input <big.ply>   # prod, 32GB heap
 */
import { spawnSync } from 'node:child_process';
import { mkdirSync, rmSync, existsSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const arg = (n, d) => { const i = process.argv.indexOf(n); return i >= 0 ? process.argv[i + 1] : d; };
const ver = arg('--ver', 'v2');
const input = arg('--input');
const target = parseInt(arg('--target', '1000000'), 10);
const stepTimeoutMs = parseInt(arg('--step-timeout', '2400'), 10) * 1000; // 40 min/step default

const V1CLI = '/Users/dhutchence/.nvm/versions/node/v22.16.0/lib/node_modules/@playcanvas/splat-transform/bin/cli.mjs';
const cfg = ver === 'v1'
    ? { bin: ['/Users/dhutchence/.nvm/versions/node/v22.16.0/bin/node', V1CLI], mem: false, env: { ...process.env, NODE_OPTIONS: '--max-old-space-size=32768' } }
    : { bin: ['/Users/dhutchence/.nvm/versions/node/v22.22.2/bin/node', 'bin/cli.mjs'], mem: true, env: process.env };

const V2NODE = '/Users/dhutchence/.nvm/versions/node/v22.22.2/bin/node';
const mbOf = (b) => (b == null ? null : Math.round(b / 1024 ** 2));
const parseBytes = (s) => { const m = /^([\d.]+)([KMGT]?B)$/.exec(s); return m ? Math.round(parseFloat(m[1]) * { B: 1, KB: 1024, MB: 1024 ** 2, GB: 1024 ** 3, TB: 1024 ** 4 }[m[2]]) : null; };

const run = (args) => {
    const t0 = performance.now();
    const res = spawnSync('/usr/bin/time', ['-l', ...cfg.bin, ...args], {
        cwd: repoRoot, env: cfg.env, encoding: 'utf8', maxBuffer: 256 * 1024 * 1024, timeout: stepTimeoutMs
    });
    const wallMs = Math.round(performance.now() - t0);
    const text = `${res.stdout ?? ''}\n${res.stderr ?? ''}`;
    const rss = /(\d+)\s+maximum resident set size/.exec(res.stderr ?? '');
    const gpu = cfg.mem ? /gpu=([\d.]+[KMGT]?B)/.exec(text) : null;
    return {
        ok: res.status === 0, status: res.status, timedOut: res.error?.code === 'ETIMEDOUT',
        wallMs, rssMB: rss ? mbOf(parseInt(rss[1], 10)) : null, gpuMB: gpu ? mbOf(parseBytes(gpu[1])) : null,
        tail: res.status === 0 ? '' : text.split('\n').filter(l => /error|Error|✗|heap|memory|ENOMEM/i.test(l)).slice(-4).join(' | ')
    };
};

// count via v2 --info (fast, reads every format), on a file or lod-meta.json.
const countOf = (p) => {
    if (!existsSync(p)) return null;
    const r = spawnSync(V2NODE, ['bin/cli.mjs', p, '-I', 'null'], { cwd: repoRoot, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024, timeout: 300_000 });
    const g = /gaussians:\s*(\d+)/.exec(r.stdout ?? '');
    if (g) return parseInt(g[1], 10);
    const lc = /lod counts:\s*([\d,\s]+)/.exec(r.stdout ?? '');
    return lc ? lc[1].split(',').reduce((a, b) => a + parseInt(b.trim(), 10), 0) : null;
};
const duMB = (p) => { try { const r = spawnSync('du', ['-sk', p], { encoding: 'utf8' }); return Math.round(parseInt(r.stdout.trim().split(/\s+/)[0], 10) / 1024); } catch { return null; } };

const wd = join(repoRoot, '.bench', 'pyramid', ver);
rmSync(wd, { recursive: true, force: true });
mkdirSync(wd, { recursive: true });

console.log(`LOD pyramid [${ver}] input=${input} target<=${target}`);
const steps = [];
let cur = input;
let curCount = countOf(input);
const levels = [{ path: input, count: curCount }];
console.log(`  L0 (input) = ${curCount} gaussians`);

let i = 0;
while (curCount != null && curCount > target) {
    i++;
    const out = join(wd, `L${i}.ply`);
    const memArgs = cfg.mem ? ['--mem'] : [];
    process.stdout.write(`  decimate L${i - 1}->L${i} (${curCount} -F 50%) ...`);
    const r = run(['-w', ...memArgs, cur, '-F', '50%', out]);
    const n = r.ok ? countOf(out) : null;
    steps.push({ step: `dec-L${i}`, ...r, count: n });
    console.log(` ${r.ok ? 'ok' : `FAIL(${r.status}${r.timedOut ? ' TIMEOUT' : ''})`} ${(r.wallMs / 1000).toFixed(1)}s rss=${r.rssMB}MB${r.gpuMB ? ` gpu=${r.gpuMB}MB` : ''} -> ${n}`);
    if (!r.ok) { console.error(`    ${r.tail}`); break; }
    levels.push({ path: out, count: n });
    cur = out; curCount = n;
}

let stack = null;
if (levels.length > 1 && steps.every(s => s.ok)) {
    const lodOut = join(wd, 'lod', 'lod-meta.json');
    mkdirSync(dirname(lodOut), { recursive: true });
    const memArgs = cfg.mem ? ['--mem'] : [];
    const stackArgs = ['-w', ...memArgs];
    levels.forEach((L, idx) => { stackArgs.push(L.path, '-l', String(idx)); });
    stackArgs.push(lodOut);
    process.stdout.write(`  stack ${levels.length} levels -> lod-meta.json ...`);
    stack = run(stackArgs);
    const lc = stack.ok ? countOf(lodOut) : null;
    console.log(` ${stack.ok ? 'ok' : `FAIL(${stack.status}${stack.timedOut ? ' TIMEOUT' : ''})`} ${(stack.wallMs / 1000).toFixed(1)}s rss=${stack.rssMB}MB${stack.gpuMB ? ` gpu=${stack.gpuMB}MB` : ''} total=${lc} size=${duMB(dirname(lodOut))}MB`);
    if (!stack.ok) console.error(`    ${stack.tail}`);
    stack.totalCount = lc;
}

const totalWall = steps.reduce((a, s) => a + s.wallMs, 0) + (stack?.wallMs ?? 0);
const peakRss = Math.max(...steps.map(s => s.rssMB ?? 0), stack?.rssMB ?? 0);
console.log(`\n[${ver}] levels: ${levels.map(l => l.count).join(' -> ')}`);
console.log(`[${ver}] total wall = ${(totalWall / 1000).toFixed(1)}s, peak RSS = ${peakRss}MB, stack ok = ${stack?.ok ?? false}`);
console.table([
    ...steps.map(s => ({ step: s.step, ok: s.ok, s: (s.wallMs / 1000).toFixed(1), rssMB: s.rssMB, gpuMB: s.gpuMB ?? '-', count: s.count })),
    stack ? { step: 'stack->lod', ok: stack.ok, s: (stack.wallMs / 1000).toFixed(1), rssMB: stack.rssMB, gpuMB: stack.gpuMB ?? '-', count: stack.totalCount } : null
].filter(Boolean));
