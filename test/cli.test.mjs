/**
 * CLI parsing regression tests.
 */

import { spawn } from 'node:child_process';
import { describe, it } from 'node:test';
import assert from 'node:assert';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = dirname(__dirname);
const cliArgsEnvName = 'SPLAT_TRANSFORM_CLI_TEST_ARGS';

const cliBootstrap = `
const cliArgs = JSON.parse(process.env.${cliArgsEnvName});
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
