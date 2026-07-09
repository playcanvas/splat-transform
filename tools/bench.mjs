#!/usr/bin/env node
/**
 * Benchmark harness: runs the BUILT CLI (bin/cli.mjs -> dist/cli.mjs) against
 * real scenes, one warmup + N timed repetitions per workload, and reports the
 * median wall time and peak memory (parsed from the CLI's --mem output).
 *
 * Build first (`npm run build`), then:
 *
 *   node tools/bench.mjs --label baseline
 *   node tools/bench.mjs --label after-x --only ply-copy,sog-ply --reps 5
 *
 * Results append to <workdir>/results.jsonl (one JSON object per rep) so
 * before/after labels can be diffed; a median summary table prints at the end.
 *
 * Scene locations default to this repo's `scenes/` directory and can be
 * overridden with BENCH_SCENES / BENCH_LCC2 (path to a .lcc2 meta file).
 */
import { spawnSync } from 'node:child_process';
import { appendFileSync, mkdirSync, rmSync, existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');

// ---- args -------------------------------------------------------------
const argv = process.argv.slice(2);
const argValue = (name, dflt) => {
    const i = argv.indexOf(name);
    return i >= 0 && i + 1 < argv.length ? argv[i + 1] : dflt;
};
const label = argValue('--label', 'unlabeled');
const only = argValue('--only', '').split(',').filter(Boolean);
const repsOverride = argValue('--reps', '');
const workdir = argValue('--workdir', join(repoRoot, '.bench'));

const scenes = process.env.BENCH_SCENES ?? join(repoRoot, 'scenes');
const lcc2 = process.env.BENCH_LCC2 ?? '/Users/dhutchence/Downloads/lcc2/sog_lcc2_wenbogong/18910214533144797.lcc2';
const lcc = process.env.BENCH_LCC ?? '/Users/dhutchence/Snapchat/Dev/Assets/gs/LCC/ConferenceHall/meta.lcc';

// ---- workloads ---------------------------------------------------------
// Each runs the full CLI pipeline on a real scene; `reps` reflects run length
// (long workloads get fewer). Output paths live under <workdir>/<name>/ and
// are wiped before every rep.
const WORKLOADS = [
    // streaming read -> streaming write; exercises BufferedReadStream, the
    // ply chunk decode, and the block-copy interleave hot path (6M rows, SH3)
    { name: 'ply-copy', reps: 3, input: join(scenes, 'vincent.ply'), out: 'out.ply', args: [] },

    // whole-scene SOG decode -> ply; exercises the per-gaussian texel decode
    // (unpackQuat / centroid lookups)
    { name: 'sog-ply', reps: 3, input: join(scenes, 'church.sog'), out: 'out.ply', args: [] },

    // single-scene ply -> streamed SOG; exercises extractSlim/calcBound
    // (ChunkData.field), per-unit k-means (12 units, SH3), unit gathers
    { name: 'ply-lod', reps: 2, input: join(scenes, 'vincent.ply'), out: 'lod-meta.json', args: [] },

    // lcc2 (sog sub-files) -> streamed SOG over 4 levels (~9.1M rows, 0 SH
    // bands so k-means is skipped); exercises containerSource/concatSource
    // gathers and the sog sub-file decode
    { name: 'lcc2-lod', reps: 2, input: lcc2, out: 'lod-meta.json', args: ['-O', '2,3,4,5'] },

    // merge-based decimation to 50% (6.1M rows); exercises the priority pass,
    // moment-match merges, and the merge stream terminal
    { name: 'decimate', reps: 2, input: join(scenes, 'decimate-pill.ply'), out: 'out.ply', args: ['-F', '50%'] },

    // v1 lcc (5 LODs, 6.18M finest, 0 SH bands so no k-means) -> streamed SOG;
    // GPU-free, dominated by the LOD writer's random-index gathers against
    // read-lcc's data.bin reads
    { name: 'lcc-lod', reps: 2, input: lcc, out: 'lod-meta.json', args: [] },

    // big-scene decimation (53.2M rows, 0 SH bands): the per-block scratch
    // churn in the priority pass only shows at this scale. One cold run per
    // side (no warmup) — a warmup would double a very long workload.
    { name: 'decimate-big', reps: 1, warmup: false, input: join(scenes, 'andrii-lod1.ply'), out: 'out.ply', args: ['-F', '50%'] }
];

// ---- helpers ------------------------------------------------------------
const parseBytes = (s) => {
    const m = /^([\d.]+)(B|KB|MB|GB|TB)$/.exec(s);
    if (!m) return null;
    const mult = { B: 1, KB: 1024, MB: 1024 ** 2, GB: 1024 ** 3, TB: 1024 ** 4 }[m[2]];
    return Math.round(parseFloat(m[1]) * mult);
};

const gitRev = () => {
    const r = spawnSync('git', ['rev-parse', '--short', 'HEAD'], { cwd: repoRoot, encoding: 'utf8' });
    const dirty = spawnSync('git', ['status', '--porcelain'], { cwd: repoRoot, encoding: 'utf8' });
    return `${r.stdout.trim()}${dirty.stdout.trim() ? '+dirty' : ''}`;
};

const median = (xs) => {
    const s = [...xs].sort((a, b) => a - b);
    return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
};

const runOnce = (w, outDir) => {
    rmSync(outDir, { recursive: true, force: true });
    mkdirSync(outDir, { recursive: true });
    const args = ['bin/cli.mjs', w.input, ...w.args, join(outDir, w.out), '-w', '--mem'];
    const t0 = performance.now();
    const res = spawnSync(process.execPath, args, { cwd: repoRoot, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
    const wallMs = performance.now() - t0;
    const text = `${res.stdout ?? ''}\n${res.stderr ?? ''}`;
    const done = /done in ([\d.]+)s/.exec(text);
    const peak = /peak cpu=([\d.]+[KMGT]?B)(?: gpu=([\d.]+[KMGT]?B))?/.exec(text);
    return {
        ok: res.status === 0,
        wallMs: Math.round(wallMs),
        cliS: done ? parseFloat(done[1]) : null,
        peakCpu: peak ? parseBytes(peak[1]) : null,
        peakGpu: peak?.[2] ? parseBytes(peak[2]) : null,
        tail: res.status === 0 ? undefined : text.split('\n').slice(-15).join('\n')
    };
};

// ---- main ---------------------------------------------------------------
mkdirSync(workdir, { recursive: true });
const resultsPath = join(workdir, 'results.jsonl');
const rev = gitRev();
const summary = [];

for (const w of WORKLOADS) {
    if (only.length > 0 && !only.includes(w.name)) continue;
    if (!existsSync(w.input)) {
        console.error(`skip ${w.name}: missing input ${w.input}`);
        continue;
    }
    const reps = repsOverride ? parseInt(repsOverride, 10) : w.reps;
    const outDir = join(workdir, w.name);

    if (w.warmup !== false) {
        process.stdout.write(`${w.name}: warmup`);
        const warm = runOnce(w, outDir);
        if (!warm.ok) {
            console.error(`\n${w.name} FAILED:\n${warm.tail}`);
            continue;
        }
    } else {
        process.stdout.write(`${w.name}:`);
    }
    const runs = [];
    for (let i = 0; i < reps; i++) {
        process.stdout.write(` rep${i + 1}`);
        const r = runOnce(w, outDir);
        if (!r.ok) {
            console.error(`\n${w.name} rep${i + 1} FAILED:\n${r.tail}`);
            break;
        }
        runs.push(r);
        appendFileSync(resultsPath, `${JSON.stringify({ ts: new Date().toISOString(), label, rev, workload: w.name, rep: i + 1, ...r })}\n`);
    }
    process.stdout.write('\n');
    if (runs.length > 0) {
        summary.push({
            workload: w.name,
            reps: runs.length,
            wallS: (median(runs.map(r => r.wallMs)) / 1000).toFixed(2),
            spreadS: ((Math.max(...runs.map(r => r.wallMs)) - Math.min(...runs.map(r => r.wallMs))) / 1000).toFixed(2),
            peakCpuMB: runs[0].peakCpu ? Math.round(median(runs.map(r => r.peakCpu)) / 1024 ** 2) : null,
            peakGpuMB: runs[0].peakGpu ? Math.round(median(runs.map(r => r.peakGpu)) / 1024 ** 2) : null
        });
    }
}

console.log(`\nlabel=${label} rev=${rev}`);
console.table(summary);
console.log(`results appended to ${resultsPath}`);
