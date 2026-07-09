#!/usr/bin/env node
/**
 * Compatibility + performance suite: runs the same matrix of code paths through
 * BOTH the globally-installed prod CLI (v1.10.1) and the local built CLI
 * (v2.7.1, bin/cli.mjs -> dist), under `/usr/bin/time -l` so peak RSS and wall
 * time are measured uniformly (prod has no --memory). v2 also reports its own peak
 * cpu/gpu via --memory.
 *
 * Coverage: every reader (input format), every writer (output format), every
 * action, info/stats, plus a few chained/merge cases. Outputs are compared by
 * gaussian count (via v2 --info, which reads every format) where the op
 * preserves/derives a count, else by file existence. Intentional v1->v2
 * behaviour changes (lod semantics, the new decimator, stats formatting) are
 * flagged, not failed.
 *
 * Build first (npm run build), then: node tools/compat-suite.mjs [--only a,b]
 */
import { spawnSync } from 'node:child_process';
import { appendFileSync, mkdirSync, rmSync, existsSync, statSync, readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const argv = process.argv.slice(2);
const only = (argv[argv.indexOf('--only') + 1] ?? '').split(',').filter(Boolean);

const V1 = ['/Users/dhutchence/.nvm/versions/node/v22.16.0/bin/splat-transform'];
const V2NODE = '/Users/dhutchence/.nvm/versions/node/v22.22.2/bin/node';
const V2 = [V2NODE, 'bin/cli.mjs'];

const GS = '/Users/dhutchence/Snapchat/Dev/Assets/gs';
const S = {
    plySH0: `${GS}/PFOX.ply`,               // 131K, SH0
    ply2: `${GS}/biker.ply`,                // 152K, SH0 (merge)
    spzSH3: `${GS}/spz/hornedlizard.spz`,   // 786K, SH3
    cply: `${GS}/guitar.compressed.ply`,    // 90K, SH0
    splat: `${GS}/guitar.splat`,            // 90K, SH0
    sog: `${GS}/schindelar3d/elephant.sog`, // 362K, SH0 (bundle)
    sogMeta: `${GS}/neumarkt/meta.json`,    // 6.9M, SH0 (unbundled)
    lcc: `${GS}/LCC/Big Mirror/meta.lcc`,   // small
    lcc2: '/Users/dhutchence/Downloads/lcc2/sog_lcc2_wenbogong/18910214533144797.lcc2', // 23.9M, 6 LOD
    mjs: 'generators/gen-grid.mjs'
};

// group, name, input (string | string[]), args (before output), out (rel path),
// cmp: 'count'|'exists'|'none', mode: 'both'|'v2only', note?
const CASES = [
    // ---- readers: <format> -> ply, compare gaussian count ----
    { group: 'reader', name: 'ply-sh0', input: S.plySH0, args: [], out: 'o.ply', cmp: 'count', mode: 'both' },
    { group: 'reader', name: 'compressed-ply', input: S.cply, args: [], out: 'o.ply', cmp: 'count', mode: 'both' },
    { group: 'reader', name: 'splat', input: S.splat, args: [], out: 'o.ply', cmp: 'count', mode: 'both' },
    { group: 'reader', name: 'spz-sh3', input: S.spzSH3, args: [], out: 'o.ply', cmp: 'count', mode: 'both' },
    { group: 'reader', name: 'sog-bundle', input: S.sog, args: [], out: 'o.ply', cmp: 'count', mode: 'both' },
    { group: 'reader', name: 'sog-meta', input: S.sogMeta, args: [], out: 'o.ply', cmp: 'count', mode: 'both' },
    { group: 'reader', name: 'lcc', input: S.lcc, args: [], out: 'o.ply', cmp: 'count', mode: 'both' },
    { group: 'reader', name: 'mjs-gen', input: S.mjs, args: ['-p', 'width=200,height=200,scale=0.1'], out: 'o.ply', cmp: 'count', mode: 'both' },
    { group: 'reader', name: 'lcc2', input: S.lcc2, args: [], out: 'o.ply', cmp: 'count', mode: 'v2only', note: 'v2-only input format' },

    // ---- writers: SH3 spz -> <format>, compare count or existence ----
    { group: 'writer', name: 'ply', input: S.spzSH3, args: [], out: 'o.ply', cmp: 'count', mode: 'both' },
    { group: 'writer', name: 'compressed-ply', input: S.spzSH3, args: [], out: 'o.compressed.ply', cmp: 'count', mode: 'both' },
    { group: 'writer', name: 'sog-bundle', input: S.spzSH3, args: [], out: 'o.sog', cmp: 'count', mode: 'both' },
    { group: 'writer', name: 'sog-meta', input: S.spzSH3, args: [], out: 'sogdir/meta.json', cmp: 'count', mode: 'both' },
    { group: 'writer', name: 'csv', input: S.plySH0, args: [], out: 'o.csv', cmp: 'exists', mode: 'both' },
    { group: 'writer', name: 'glb', input: S.plySH0, args: [], out: 'o.glb', cmp: 'exists', mode: 'both' },
    { group: 'writer', name: 'html-bundle', input: S.plySH0, args: [], out: 'o.html', cmp: 'exists', mode: 'both' },
    { group: 'writer', name: 'voxel', input: S.plySH0, args: ['--voxel-size', '0.2'], v1Args: ['-R', '0.2'], out: 'o.voxel.json', cmp: 'exists', mode: 'both', note: 'GPU; v1 uses -R, v2 --voxel-size' },
    { group: 'writer', name: 'spz', input: S.spzSH3, args: [], out: 'o.spz', cmp: 'count', mode: 'v2only', note: 'v2-only output' },
    { group: 'writer', name: 'webp-image', input: S.plySH0, args: [], out: 'o.webp', cmp: 'exists', mode: 'v2only', note: 'v2-only output, GPU' },
    { group: 'writer', name: 'lod-from-lcc', input: S.lcc, args: [], out: 'lod/lod-meta.json', cmp: 'exists', mode: 'both', note: 'lod-selection semantics changed' },

    // ---- actions: ply -> ply, compare count ----
    { group: 'action', name: 'translate', input: S.plySH0, args: ['-t', '1,2,3'], out: 'o.ply', cmp: 'count', mode: 'both' },
    { group: 'action', name: 'rotate', input: S.plySH0, args: ['-r', '0,90,0'], out: 'o.ply', cmp: 'count', mode: 'both' },
    { group: 'action', name: 'scale', input: S.plySH0, args: ['-s', '0.5'], out: 'o.ply', cmp: 'count', mode: 'both' },
    { group: 'action', name: 'filter-nan', input: S.plySH0, args: ['-N'], out: 'o.ply', cmp: 'count', mode: 'both' },
    { group: 'action', name: 'filter-box', input: S.plySH0, args: ['-B', '-1,-1,-1,1,1,1'], out: 'o.ply', cmp: 'count', mode: 'both' },
    { group: 'action', name: 'filter-sphere', input: S.plySH0, args: ['-S', '0,0,0,2'], out: 'o.ply', cmp: 'count', mode: 'both' },
    { group: 'action', name: 'filter-value', input: S.plySH0, args: ['-V', 'opacity,gt,0.1'], out: 'o.ply', cmp: 'count', mode: 'both' },
    { group: 'action', name: 'filter-harmonics', input: S.spzSH3, args: ['-H', '1'], out: 'o.ply', cmp: 'count', mode: 'both', note: 'SH3->1' },
    { group: 'action', name: 'morton-order', input: S.plySH0, args: ['-m'], v1Args: ['-M'], out: 'o.ply', cmp: 'count', mode: 'both' },
    { group: 'action', name: 'decimate-50pct', input: S.plySH0, args: ['-d', '50%'], v1Args: ['-F', '50%'], out: 'o.ply', cmp: 'count', mode: 'both', note: 'new decimator; quality differs' },
    { group: 'action', name: 'filter-floaters', input: S.plySH0, args: ['-F'], out: 'o.ply', cmp: 'count', mode: 'v2only', note: 'v2-only, GPU' },

    // ---- info / stats ----
    { group: 'info', name: 'stats-summary', input: S.plySH0, args: ['--stats'], v1Args: ['-m'], out: 'null', cmp: 'none', mode: 'both', note: 'v1 --summary vs v2 --stats; output format differs' },
    { group: 'info', name: 'info', input: S.plySH0, args: ['--info'], out: 'null', cmp: 'none', mode: 'v2only', note: 'v2-only' },

    // ---- chained / merge ----
    { group: 'chain', name: 'transform-to-sog', input: S.plySH0, args: ['-t', '1,0,0', '-s', '2'], out: 'o.sog', cmp: 'count', mode: 'both' },
    { group: 'chain', name: 'merge-two', input: [S.plySH0, S.ply2], args: [], out: 'o.ply', cmp: 'count', mode: 'both', note: 'count = sum of inputs' },
    // Same file twice (matching schema): valid 2-LOD stack. NB: two DIFFERENT
    // scenes with mismatched layers (e.g. PFOX vs biker) make v2 correctly reject
    // the stack ('available-layer mismatch') where v1 silently allowed it — a v2
    // correctness guard, verified separately.
    { group: 'chain', name: 'lod-tag-two', input: [S.plySH0, S.plySH0], args: [], out: 'lod2/lod-meta.json', cmp: 'exists', mode: 'both', note: 'lod tagging (-l per input); semantics changed', perInputArgs: [['-l', '0'], ['-l', '1']] }
];

const parseBytes = (s) => {
    const m = /^([\d.]+)([KMGT]?B)$/.exec(s);
    if (!m) return null;
    return Math.round(parseFloat(m[1]) * { B: 1, KB: 1024, MB: 1024 ** 2, GB: 1024 ** 3, TB: 1024 ** 4 }[m[2]]);
};
const mb = (bytes) => (bytes == null ? null : Math.round(bytes / 1024 ** 2));

// Run a command under /usr/bin/time -l; return wall/maxRSS/exit + parsed gpu.
const timeRun = (bin, args, mem) => {
    const cmd = ['-l', ...bin, ...args];
    const t0 = performance.now();
    const res = spawnSync('/usr/bin/time', cmd, { cwd: repoRoot, encoding: 'utf8', maxBuffer: 256 * 1024 * 1024, timeout: 900_000 });
    const wallMs = Math.round(performance.now() - t0);
    const text = `${res.stdout ?? ''}\n${res.stderr ?? ''}`;
    const rss = /(\d+)\s+maximum resident set size/.exec(res.stderr ?? '');
    const gpu = mem ? /gpu=([\d.]+[KMGT]?B)/.exec(text) : null;
    return {
        ok: res.status === 0,
        status: res.status,
        wallMs,
        maxRssMB: rss ? mb(parseInt(rss[1], 10)) : null,
        gpuMB: gpu ? mb(parseBytes(gpu[1])) : null,
        tail: res.status === 0 ? '' : text.split('\n').slice(-8).join('\n')
    };
};

// Gaussian count via v2 --info (reads every format); accepts a file or the
// meta.json / lod-meta.json inside an output dir.
const countOf = (path) => {
    if (!existsSync(path)) return null;
    const res = spawnSync(V2NODE, ['bin/cli.mjs', path, '--info', 'null'], { cwd: repoRoot, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024, timeout: 300_000 });
    if (res.status !== 0) return null;
    const m = /gaussians:\s*(\d+)/.exec(res.stdout ?? '');
    // lod-meta reports per-lod; fall back to summed lod counts
    if (m) return parseInt(m[1], 10);
    const lc = /lod counts:\s*([\d,\s]+)/.exec(res.stdout ?? '');
    if (lc) return lc[1].split(',').reduce((a, b) => a + parseInt(b.trim(), 10), 0);
    return null;
};

const dirNonEmpty = (p) => { try { return statSync(p).isDirectory() ? readdirSync(p).length > 0 : statSync(p).size > 0; } catch { return false; } };

const buildArgs = (c, outPath, mem, label) => {
    const inputs = Array.isArray(c.input) ? c.input : [c.input];
    const parts = ['-w'];
    if (mem) parts.push('--memory');
    inputs.forEach((inp, i) => {
        parts.push(inp);
        if (c.perInputArgs?.[i]) parts.push(...c.perInputArgs[i]);
    });
    const args = label === 'v1' && c.v1Args ? c.v1Args : c.args;
    parts.push(...args, outPath);
    return parts;
};

const workdir = join(repoRoot, '.bench', 'compat');
mkdirSync(workdir, { recursive: true });
const resultsPath = join(workdir, 'results.jsonl');
const rows = [];

for (const c of CASES) {
    const id = `${c.group}/${c.name}`;
    if (only.length && !only.includes(c.name) && !only.includes(c.group)) continue;
    const inputs = Array.isArray(c.input) ? c.input : [c.input];
    if (inputs.some(p => !p.endsWith('.mjs') && !existsSync(p))) {
        console.error(`skip ${id}: missing input`);
        continue;
    }

    const caseDir = join(workdir, c.group, c.name);
    const run = (label, bin, mem) => {
        const od = join(caseDir, label);
        rmSync(od, { recursive: true, force: true });
        mkdirSync(od, { recursive: true });
        const outPath = c.out === 'null' ? 'null' : join(od, c.out);
        if (c.out !== 'null' && c.out.includes('/')) mkdirSync(dirname(outPath), { recursive: true });
        const r = timeRun(bin, buildArgs(c, outPath, mem, label), mem);
        r.outPath = outPath;
        return r;
    };

    process.stdout.write(`${id} ...`);
    const v2 = run('v2', V2, true);
    const v1 = c.mode === 'v2only' ? null : run('v1', V1, false);

    // comparison
    let cmp = 'n/a';
    if (c.cmp === 'count') {
        const c2 = v2.ok ? countOf(v2.outPath) : null;
        const c1 = v1 && v1.ok ? countOf(v1.outPath) : null;
        if (c.mode === 'v2only') cmp = c2 != null ? `v2=${c2}` : 'no-count';
        else if (c1 != null && c2 != null) cmp = c1 === c2 ? `match(${c2})` : `DIFF v1=${c1} v2=${c2}`;
        else cmp = `v1=${c1} v2=${c2}`;
    } else if (c.cmp === 'exists') {
        const e2 = v2.ok && c.out !== 'null' && dirNonEmpty(v2.outPath);
        const e1 = v1 && v1.ok && c.out !== 'null' && dirNonEmpty(v1.outPath);
        cmp = c.mode === 'v2only' ? (e2 ? 'v2-ok' : 'v2-missing') : `v1=${e1 ? 'ok' : 'X'} v2=${e2 ? 'ok' : 'X'}`;
    }

    const row = {
        id, note: c.note ?? '',
        v1ok: v1 ? v1.ok : 'n/a', v2ok: v2.ok,
        v1wall: v1 ? v1.wallMs : null, v2wall: v2.wallMs,
        wallRatio: v1 && v1.ok && v2.ok ? +(v1.wallMs / v2.wallMs).toFixed(2) : null,
        v1rssMB: v1 ? v1.maxRssMB : null, v2rssMB: v2.maxRssMB,
        rssRatio: v1 && v1.maxRssMB && v2.maxRssMB ? +(v1.maxRssMB / v2.maxRssMB).toFixed(2) : null,
        v2gpuMB: v2.gpuMB, cmp
    };
    rows.push(row);
    appendFileSync(resultsPath, `${JSON.stringify({ ts: new Date().toISOString(), ...row })}\n`);
    process.stdout.write(` v1=${row.v1ok} v2=${row.v2ok} wall x${row.wallRatio ?? '-'} rss x${row.rssRatio ?? '-'} ${cmp}\n`);
    if (v1 && !v1.ok && c.mode !== 'v2only') console.error(`   v1 FAIL(${v1.status}): ${v1.tail.replace(/\n/g, ' | ')}`);
    if (!v2.ok) console.error(`   v2 FAIL(${v2.status}): ${v2.tail.replace(/\n/g, ' | ')}`);
}

console.log(`\n=== compat-suite summary (v1=${'1.10.1'} vs v2 local) ===`);
console.table(rows.map(r => ({
    case: r.id, v1: r.v1ok, v2: r.v2ok,
    'v1 s': r.v1wall != null ? (r.v1wall / 1000).toFixed(1) : '-',
    'v2 s': (r.v2wall / 1000).toFixed(1),
    'wall x': r.wallRatio ?? '-',
    'v1 MB': r.v1rssMB ?? '-', 'v2 MB': r.v2rssMB ?? '-', 'rss x': r.rssRatio ?? '-',
    'v2 gpu': r.v2gpuMB ?? '-', compare: r.cmp
})));
console.log(`results -> ${resultsPath}`);
