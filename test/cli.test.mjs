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
});
