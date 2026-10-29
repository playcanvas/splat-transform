/**
 * CLI parsing regression tests.
 */

import { spawn } from 'node:child_process';
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { describe, it } from 'node:test';
import assert from 'node:assert';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = dirname(__dirname);
const cliArgsEnvName = 'SPLAT_TRANSFORM_CLI_TEST_ARGS';

const cliBootstrap = `
const cliArgs = JSON.parse(process.env.${cliArgsEnvName});
// from source the codec resolves its wasm relative to src/lib; point it at the shipped copy
const { WebPCodec } = await import('./src/lib/utils/webp-codec.ts');
WebPCodec.wasmUrl = process.cwd() + '/lib/webp.wasm';
process.argv = ['node', 'src/cli/index.ts', ...cliArgs];
const { main } = await import('./src/cli/index.ts');
await main();
`;

const runCli = (args) => {
    return new Promise((resolve, reject) => {
        const child = spawn(process.execPath, [
            '--input-type=module',
            '--import',
            'tsx',
            '-e',
            cliBootstrap
        ], {
            cwd: rootDir,
            env: {
                ...process.env,
                NO_COLOR: '1',
                [cliArgsEnvName]: JSON.stringify(args)
            },
            stdio: ['ignore', 'pipe', 'pipe']
        });

        let stdout = '';
        let stderr = '';
        child.stdout.setEncoding('utf8');
        child.stderr.setEncoding('utf8');
        child.stdout.on('data', chunk => {
            stdout += chunk;
        });
        child.stderr.on('data', chunk => {
            stderr += chunk;
        });
        child.on('error', reject);
        child.on('close', code => {
            resolve({ code, stdout, stderr });
        });
    });
};

describe('CLI parsing', () => {
    it('allows filter-sphere coordinates below their array indexes', async () => {
        const result = await runCli([
            '--gpu',
            'cpu',
            'test/fixtures/splat/minimal.splat',
            '--filter-sphere',
            '0,1.6,0,15',
            'null'
        ]);

        assert.strictEqual(result.code, 0, `CLI failed:\n${result.stderr}\n${result.stdout}`);
    });

    it('allows negative filter-sphere center coordinates', async () => {
        const result = await runCli([
            '--gpu',
            'cpu',
            'test/fixtures/splat/minimal.splat',
            '--filter-sphere',
            '-1,0,-2,10',
            'null'
        ]);

        assert.strictEqual(result.code, 0, `CLI failed:\n${result.stderr}\n${result.stdout}`);
    });

    it('accepts --tag-lod -1 to tag an input as environment', async () => {
        const result = await runCli([
            '--gpu',
            'cpu',
            'test/fixtures/splat/minimal.splat',
            'test/fixtures/splat/minimal.splat',
            '--tag-lod',
            '-1',
            'null'
        ]);

        assert.strictEqual(result.code, 0, `CLI failed:\n${result.stderr}\n${result.stdout}`);
    });

    it('rejects --tag-lod values below -1', async () => {
        const result = await runCli([
            '--gpu',
            'cpu',
            'test/fixtures/splat/minimal.splat',
            '--tag-lod',
            '-2',
            'null'
        ]);

        assert.notStrictEqual(result.code, 0, 'CLI should reject --tag-lod -2');
        assert.match(result.stderr, /Must be >= 0, or -1/);
    });

    it('rejects --sog-version values other than 2 or 3', async () => {
        const result = await runCli(['--gpu', 'cpu', 'test/fixtures/splat/minimal.splat', '--sog-version', '4', 'null']);

        assert.notStrictEqual(result.code, 0, 'CLI should reject --sog-version 4');
        assert.match(result.stderr, /Invalid sog-version value: 4/);
    });

    it('packs repeated --sog-blind fields into blind textures', async () => {
        const { Column } = await import('../src/lib/index.js');
        const { createTestDataTable, encodePlyBinary } = await import('./helpers/test-utils.mjs');
        const dt = createTestDataTable(16);
        dt.addColumn(new Column('a', Float32Array.from({ length: 16 }, (_, i) => i / 15)));
        dt.addColumn(new Column('depth', Float32Array.from({ length: 16 }, (_, i) => 1 + i)));
        dt.addColumn(new Column('id', Uint32Array.from({ length: 16 }, (_, i) => i * 1000)));
        const dir = await mkdtemp(join(tmpdir(), 'st-blind-cli-'));
        try {
            await writeFile(join(dir, 'in.ply'), encodePlyBinary(dt));
            await mkdir(join(dir, 'out'));
            const result = await runCli([
                '--gpu', 'cpu', join(dir, 'in.ply'),
                '--sog-blind', 'a:unorm8,depth:unorm8[1,16]', '--sog-blind', 'id',
                join(dir, 'out', 'meta.json')
            ]);
            assert.strictEqual(result.code, 0, result.stderr);
            const meta = JSON.parse(await readFile(join(dir, 'out', 'meta.json'), 'utf8'));
            assert.deepStrictEqual(meta.blind, [
                { files: ['blind_0.webp'], fields: [{ name: 'a', type: 'unorm8', offset: 0 }, { name: 'depth', type: 'unorm8', offset: 1, min: 1, max: 16 }] },
                { files: ['blind_1.webp'], fields: [{ name: 'id', type: 'u16', offset: 0 }] }
            ]);
        } finally {
            await rm(dir, { recursive: true, force: true });
        }
    });

    it('rejects a malformed --sog-blind', async () => {
        for (const value of [':u8', 'id:5', 'id:u8:u8', 'id:unorm8[0]', 'a,,b', 'id:u9']) {
            const result = await runCli(['--gpu', 'cpu', 'test/fixtures/splat/minimal.splat', '--sog-blind', value, 'null']);

            assert.notStrictEqual(result.code, 0, `CLI should reject --sog-blind ${value}`);
            assert.match(result.stderr, /Invalid sog-blind value/);
        }
    });
});

describe('CLI decimate (terminal PLY restriction)', () => {
    it('rejects decimate with a non-PLY output', async () => {
        const { mkdtemp, rm } = await import('node:fs/promises');
        const { tmpdir } = await import('node:os');
        const { join } = await import('node:path');
        const dir = await mkdtemp(join(tmpdir(), 'st-decimate-cli-'));
        const result = await runCli([
            '--gpu', 'cpu',
            'test/fixtures/splat/minimal.splat',
            '--decimate', '50%',
            join(dir, 'out.csv')
        ]);
        await rm(dir, { recursive: true, force: true });
        assert.notStrictEqual(result.code, 0, 'CLI should reject non-PLY decimate output');
        assert.match(result.stderr, /must be the final action and the output must be \.ply/);
    });

    it('rejects decimate followed by another action', async () => {
        const { mkdtemp, rm } = await import('node:fs/promises');
        const { tmpdir } = await import('node:os');
        const { join } = await import('node:path');
        const dir = await mkdtemp(join(tmpdir(), 'st-decimate-cli-'));
        const result = await runCli([
            '--gpu', 'cpu',
            'test/fixtures/splat/minimal.splat',
            '--decimate', '50%',
            '--filter-nan',
            join(dir, 'out.ply')
        ]);
        await rm(dir, { recursive: true, force: true });
        assert.notStrictEqual(result.code, 0, 'CLI should reject actions after decimate');
        assert.match(result.stderr, /must be the final action/);
    });

    it('rejects decimate with null output', async () => {
        const result = await runCli([
            '--gpu', 'cpu',
            'test/fixtures/splat/minimal.splat',
            '--decimate', '50%',
            'null'
        ]);
        assert.notStrictEqual(result.code, 0, 'CLI should reject decimate without an output');
        assert.match(result.stderr, /must be the final action and the output must be \.ply/);
    });

    it('decimates to PLY with the exact target count in the header', async () => {
        const { mkdtemp, readFile: readFileFs, rm } = await import('node:fs/promises');
        const { tmpdir } = await import('node:os');
        const { join } = await import('node:path');
        const dir = await mkdtemp(join(tmpdir(), 'st-decimate-cli-'));
        const outPath = join(dir, 'out.ply');

        const result = await runCli([
            '--gpu', 'cpu',
            'test/fixtures/splat/minimal.splat',
            '--decimate', '50%',
            outPath
        ]);
        assert.strictEqual(result.code, 0, `CLI failed:\n${result.stderr}\n${result.stdout}`);

        const bytes = await readFileFs(outPath);
        const header = bytes.subarray(0, 512).toString('latin1');
        const match = header.match(/element vertex (\d+)/);
        assert.ok(match, 'output has a vertex element');
        const written = parseInt(match[1], 10);
        await rm(dir, { recursive: true, force: true });
        assert.ok(written > 0, 'non-empty output');
        // 50% of the fixture's count, rounded — read the input count from a
        // passthrough run's log line instead of hardcoding the fixture size.
        const info = await runCli(['--gpu', 'cpu', 'test/fixtures/splat/minimal.splat', '--info', 'json', 'null']);
        const parsed = JSON.parse(info.stdout.slice(info.stdout.indexOf('{')));
        const inputCount = parsed.count ?? parsed.numGaussians ?? parsed.lods?.[0]?.count;
        assert.strictEqual(written, Math.round(inputCount / 2), `50% of ${inputCount}`);
    });

    it('--decimate-adaptive runs the adaptive algorithm', async () => {
        const { mkdtemp, readFile: readFileFs, rm } = await import('node:fs/promises');
        const { tmpdir } = await import('node:os');
        const { join } = await import('node:path');
        const dir = await mkdtemp(join(tmpdir(), 'st-decimate-adaptive-cli-'));
        const balancedPath = join(dir, 'balanced.ply');

        const balanced = await runCli([
            '--gpu', 'cpu',
            'test/fixtures/splat/minimal.splat',
            '--decimate-adaptive', '50%',
            balancedPath
        ]);
        assert.strictEqual(balanced.code, 0, `balanced CLI failed:\n${balanced.stderr}\n${balanced.stdout}`);
        const header = (await readFileFs(balancedPath)).subarray(0, 1024).toString('ascii');
        const match = header.match(/element vertex (\d+)/);
        assert.ok(match, 'output has a vertex element');
        assert.strictEqual(parseInt(match[1], 10), 2);
        await rm(dir, { recursive: true, force: true });
    });
});

describe('CLI filter-nan (zero-norm rotation)', () => {
    it('drops all-zero rotation rows through the source path', async () => {
        const { mkdtemp, readFile: readFileFs, rm, writeFile } = await import('node:fs/promises');
        const { tmpdir } = await import('node:os');
        const { join } = await import('node:path');
        const { createTestDataTable, encodePlyBinary } = await import('./helpers/test-utils.mjs');

        const dataTable = createTestDataTable(4);
        // createTestDataTable writes identity quaternions; zeroing rot_0 makes
        // row 2 all-zero (rot_1..3 are already 0) -- the zero-padded-PLY shape
        dataTable.getColumnByName('rot_0').data[2] = 0;

        const dir = await mkdtemp(join(tmpdir(), 'st-filter-nan-cli-'));
        const inPath = join(dir, 'in.ply');
        const outPath = join(dir, 'out.ply');
        await writeFile(inPath, encodePlyBinary(dataTable));

        const result = await runCli([inPath, '--filter-nan', outPath]);
        assert.strictEqual(result.code, 0, `CLI failed:\n${result.stderr}\n${result.stdout}`);

        const header = (await readFileFs(outPath)).subarray(0, 512).toString('latin1');
        await rm(dir, { recursive: true, force: true });
        assert.match(header, /element vertex 3\n/, 'zero-norm rotation row should be dropped');
    });
});

describe('CLI image sequences', () => {
    it('refuses to overwrite existing frames of a camera track without -w', async () => {
        const { mkdtemp, rm, writeFile } = await import('node:fs/promises');
        const { tmpdir } = await import('node:os');
        const { join } = await import('node:path');
        const dir = await mkdtemp(join(tmpdir(), 'st-image-seq-cli-'));
        try {
            // Two identical frames; the track only has to load, since the
            // overwrite check runs before any rendering.
            const pose = { position: [0, 5, -8], target: [0, 0, 0] };
            await writeFile(join(dir, 'track.json'), JSON.stringify({ frameRate: 30, frames: [pose, pose] }));
            await writeFile(join(dir, 'seq.0001.webp'), 'stale');
            const result = await runCli([
                'test/fixtures/generator.mjs',
                '-p', 'width=2,height=2,spacing=1,scale=0.1',
                join(dir, 'seq.webp'),
                '--camera-track', join(dir, 'track.json')
            ]);
            assert.notStrictEqual(result.code, 0, 'the CLI should refuse to overwrite a frame');
            assert.match(result.stderr + result.stdout, /seq\.0001\.webp' already exists/);
        } finally {
            await rm(dir, { recursive: true, force: true });
        }
    });
});

describe('CLI --info (header-only report)', () => {
    const fixtures = ['minimal-raw.spz', 'minimal-v2.spz', 'minimal-v3.spz', 'minimal-v4.spz', 'minimal.splat'];

    const infoJson = async (file) => {
        const result = await runCli(['-q', file, '--info', 'json', 'null']);
        assert.strictEqual(result.code, 0, `CLI failed:\n${result.stderr}\n${result.stdout}`);
        return JSON.parse(result.stdout.slice(result.stdout.indexOf('{')));
    };

    it('reports the same info as a full read, model included', async () => {
        const { readFile, MemoryReadFileSystem, getInputFormat } = await import('../src/lib/index.js');
        const { readFile: readFileFs } = await import('node:fs/promises');
        for (const fixture of fixtures) {
            const file = `test/fixtures/splat/${fixture}`;
            const fileSystem = new MemoryReadFileSystem();
            fileSystem.set(fixture, new Uint8Array(await readFileFs(`${rootDir}/${file}`)));
            const inputFormat = getInputFormat(fixture);
            const [full] = await readFile({ filename: fixture, inputFormat, options: { lodSelect: [] }, params: [], fileSystem });

            const info = await infoJson(file);
            assert.strictEqual(info.format, inputFormat, fixture);
            assert.strictEqual(info.numGaussians, full.meta.numGaussians, fixture);
            assert.deepStrictEqual(info.lodCounts, [...full.meta.lodCounts], fixture);
            assert.strictEqual(info.shBands, full.meta.shBands, fixture);
            assert.strictEqual(info.model, full.meta.model, fixture);
            await full.close();
        }
    });

    it('reads a gzip-wrapped .spz header without decoding the scene', async () => {
        const { mkdtemp, rm, writeFile } = await import('node:fs/promises');
        const { tmpdir } = await import('node:os');
        const { join } = await import('node:path');
        const { gzipSync } = await import('node:zlib');
        const { randomBytes } = await import('node:crypto');

        // a v3 header (magic, version, numPoints) and an incompressible payload,
        // cut to its first 64KB: only a header-only read can report it
        const numPoints = 100000;
        const header = Buffer.alloc(16);
        header.writeUInt32LE(0x5053474e, 0);
        header.writeUInt32LE(3, 4);
        header.writeUInt32LE(numPoints, 8);
        header.writeUInt8(12, 13);
        const zipped = gzipSync(Buffer.concat([header, randomBytes(numPoints * 20)]));

        const dir = await mkdtemp(join(tmpdir(), 'st-info-cli-'));
        const file = join(dir, 'truncated.spz');
        await writeFile(file, zipped.subarray(0, 65536));
        try {
            const info = await infoJson(file);
            assert.strictEqual(info.numGaussians, numPoints);
            assert.strictEqual(info.shBands, 0);
        } finally {
            await rm(dir, { recursive: true, force: true });
        }
    });

    it('reports an antialiased .spz from its header flag', async () => {
        const { mkdtemp, readFile: readFileFs, rm, writeFile } = await import('node:fs/promises');
        const { tmpdir } = await import('node:os');
        const { join } = await import('node:path');
        const { gunzipSync, gzipSync } = await import('node:zlib');

        const raw = gunzipSync(await readFileFs(`${rootDir}/test/fixtures/splat/minimal-v3.spz`));
        raw[14] |= 0x1; // FLAG_ANTIALIASED
        const dir = await mkdtemp(join(tmpdir(), 'st-info-cli-'));
        const file = join(dir, 'antialiased.spz');
        await writeFile(file, gzipSync(raw));
        try {
            assert.strictEqual((await infoJson(file)).model, 'antialiased');
        } finally {
            await rm(dir, { recursive: true, force: true });
        }
    });

    it('prints the text form unchanged', async () => {
        const result = await runCli(['-q', 'test/fixtures/splat/minimal-v4.spz', '--info', 'null']);
        assert.strictEqual(result.code, 0, `CLI failed:\n${result.stderr}\n${result.stdout}`);
        assert.match(result.stdout, /^format: spz$/m);
        assert.match(result.stdout, /^gaussians: \d+$/m);
        assert.match(result.stdout, /^layers: position, geometric, color$/m);
    });
});


describe('CLI --sog-extras', () => {
    it('writes extras to lod-meta.json only, and the LOD reader carries them on', async () => {
        const dir = await mkdtemp(join(tmpdir(), 'sog-extras-'));
        try {
            const extras = { scene: 'minimal', ids: [1, 2, 3] };
            await writeFile(join(dir, 'extras.json'), JSON.stringify(extras));
            // LOD output takes PLY inputs
            const ply = join(dir, 'minimal.ply');
            let result = await runCli(['--gpu', 'cpu', '-w', 'test/fixtures/splat/minimal.splat', ply]);
            assert.strictEqual(result.code, 0, `CLI failed:\n${result.stderr}\n${result.stdout}`);
            const lodMeta = join(dir, 'lod', 'lod-meta.json');
            result = await runCli([
                '--gpu', 'cpu', '-w',
                ply, '-l', '0',
                ply, '-l', '1',
                '--sog-extras', join(dir, 'extras.json'),
                lodMeta
            ]);
            assert.strictEqual(result.code, 0, `CLI failed:\n${result.stderr}\n${result.stdout}`);
            assert.deepStrictEqual(JSON.parse(await readFile(lodMeta, 'utf8')).extras, extras);
            for (const unit of (await readdir(join(dir, 'lod'))).filter((d) => /^\d+_\d+$/.test(d))) {
                const unitMeta = JSON.parse(await readFile(join(dir, 'lod', unit, 'meta.json'), 'utf8'));
                assert.strictEqual(unitMeta.extras, undefined, `unit ${unit} carries no extras`);
            }

            // the LOD reader puts them on the source, so a SOG written from it keeps them
            const sogMeta = join(dir, 'out', 'meta.json');
            result = await runCli(['--gpu', 'cpu', '-w', lodMeta, sogMeta]);
            assert.strictEqual(result.code, 0, `CLI failed:\n${result.stderr}\n${result.stdout}`);
            assert.deepStrictEqual(JSON.parse(await readFile(sogMeta, 'utf8')).extras, extras);
        } finally {
            await rm(dir, { recursive: true, force: true });
        }
    });
});
