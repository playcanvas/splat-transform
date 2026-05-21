/**
 * Golden-image regression tests for the WebP renderer.
 *
 * Each case renders a small synthetic scene with a fixed camera, then
 * byte-compares the output to a checked-in golden .webp. Catches any
 * change to the rendering pipeline that would alter pixel output.
 *
 * The renderer is deterministic on a given GPU/driver — these tests
 * verify byte-exact output. If they fail after an intentional renderer
 * change, regenerate goldens via:
 *
 *     node test/render-golden.regenerate.mjs
 *
 * Fixtures use the generic `test/fixtures/generator.mjs` (a deterministic
 * grid of identical splats) with case-specific parameters. Run args and
 * golden paths come from `render-golden.cases.mjs` — single source of
 * truth shared with the regenerator.
 *
 * The test spawns `bin/cli.mjs`, which loads the built CLI in `dist/`.
 * `npm run build` must have run at least once before this test will
 * pass; the test skips with a clear error otherwise.
 */

import { spawn } from 'node:child_process';
import { statSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { describe, it } from 'node:test';
import assert from 'node:assert';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { CASES } from './render-golden.cases.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = dirname(__dirname);
const cliPath = join(rootDir, 'bin/cli.mjs');

// `bin/cli.mjs` imports the bundled `dist/cli.mjs` (gitignored, must be
// built). `npm run pretest` builds before `npm test`; direct invocation
// (`node --test test/render-golden.test.mjs`) without a prior build is
// also valid and the suite skips with a clear message.
const distExists = (() => {
    try {
        statSync(join(rootDir, 'dist/cli.mjs'));
        return true;
    } catch {
        return false;
    }
})();

const runCli = (args) => {
    return new Promise((resolve, reject) => {
        const child = spawn(process.execPath, [cliPath, ...args], {
            cwd: rootDir,
            env: { ...process.env, NO_COLOR: '1' },
            stdio: ['ignore', 'pipe', 'pipe']
        });

        let stdout = '';
        let stderr = '';
        child.stdout.setEncoding('utf8');
        child.stderr.setEncoding('utf8');
        child.stdout.on('data', chunk => { stdout += chunk; });
        child.stderr.on('data', chunk => { stderr += chunk; });
        child.on('error', reject);
        child.on('close', code => resolve({ code, stdout, stderr }));
    });
};

describe('Render goldens', { skip: distExists ? false : 'dist/cli.mjs missing — run `npm run build` first (or `npm test`, which builds via the pretest hook)' }, () => {
    for (const { name, args, goldenPath } of CASES) {
        it(`${name} matches golden`, async () => {
            const outPath = join(tmpdir(), `golden-${name}-${process.pid}.webp`);
            const result = await runCli([...args, outPath, '-w', '-q']);
            assert.strictEqual(
                result.code, 0,
                `CLI failed for ${name}:\n${result.stderr}\n${result.stdout}`
            );

            const [actual, expected] = await Promise.all([
                readFile(outPath),
                readFile(join(__dirname, goldenPath))
            ]);
            assert.strictEqual(
                actual.length, expected.length,
                `${name}: byte length ${actual.length} != golden ${expected.length}`
            );
            assert.ok(
                actual.equals(expected),
                `${name}: bytes differ from golden (${goldenPath})`
            );
        });
    }
});
