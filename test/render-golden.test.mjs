/**
 * Golden-image regression tests for the WebP renderer.
 *
 * Each case renders a small synthetic scene with a fixed camera, then
 * compares the decoded pixels to a checked-in golden .webp. Catches any
 * change to the rendering pipeline that would alter pixel output, while
 * staying independent of the (lossless) WebP encoder's effort setting.
 *
 * The renderer is deterministic on a given GPU/driver — these tests
 * verify byte-exact pixels. If they fail after an intentional renderer
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

// Decoded-pixel comparison: the codec ships in the built library.
const decodeRgba = async (bytes) => {
    const { WebPCodec } = await import(join(rootDir, 'dist/index.mjs'));
    const codec = await WebPCodec.create();
    return codec.decodeRGBA(new Uint8Array(bytes));
};

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
                readFile(outPath).then(decodeRgba),
                readFile(join(__dirname, goldenPath)).then(decodeRgba)
            ]);
            assert.strictEqual(
                `${actual.width}x${actual.height}`, `${expected.width}x${expected.height}`,
                `${name}: size differs from golden`
            );
            assert.ok(
                Buffer.from(actual.rgba.buffer, actual.rgba.byteOffset, actual.rgba.byteLength)
                .equals(Buffer.from(expected.rgba.buffer, expected.rgba.byteOffset, expected.rgba.byteLength)),
                `${name}: pixels differ from golden (${goldenPath})`
            );
        });
    }
});

describe('Render options', { skip: distExists ? false : 'dist/cli.mjs missing — run `npm run build` first' }, () => {
    it('defaults to 32 aperture samples and one sample matches a pinhole render', async () => {
        const args = CASES[0].args;
        const render = async (name, extra) => {
            const path = join(tmpdir(), `aperture-${name}-${process.pid}.webp`);
            const result = await runCli([...args, ...extra, path, '-w', '-q']);
            assert.strictEqual(result.code, 0, result.stderr);
            return (await decodeRgba(await readFile(path))).rgba;
        };
        const dof = ['--f-stop', '1', '--sensor-size', '0.5'];
        const defaults = await render('default', dof);
        assert.deepStrictEqual(defaults, await render('explicit', [...dof, '--dof-samples', '32']));
        const pinhole = await render('pinhole', []);
        assert.deepStrictEqual(pinhole, await render('single', [...dof, '--dof-samples', '1']));
        assert.notDeepStrictEqual(defaults, pinhole);
        const motion = ['--camera-pos-end', '1,5,-8', '--motion-samples', '2'];
        const motionImage = await render('motion', motion);
        assert.deepStrictEqual(motionImage, await render('motion-single', [...motion, ...dof, '--dof-samples', '1']));
        assert.notDeepStrictEqual(motionImage, await render('motion-dof', [...motion, ...dof, '--dof-samples', '4']));
    });

    it('defaults to 16 motion samples and a half-interval shutter', async () => {
        const args = [...CASES[0].args, '--camera-pos-end', '1,0,3'];
        const render = async (name, extra) => {
            const path = join(tmpdir(), `motion-default-${name}-${process.pid}.webp`);
            const result = await runCli([...args, ...extra, path, '-w', '-q']);
            assert.strictEqual(result.code, 0, result.stderr);
            return (await decodeRgba(await readFile(path))).rgba;
        };
        const defaults = await render('default', []);
        assert.deepStrictEqual(defaults, await render('explicit', ['--motion-samples', '16', '--shutter', '0.5']));
        assert.notDeepStrictEqual(defaults, await render('single', ['--motion-samples', '1']));
        assert.notDeepStrictEqual(defaults, await render('full', ['--shutter', '1']));
    });

    it('a still ignores the end target and end up without an end position', async () => {
        // Without --camera-pos-end there is no motion, so the end options
        // must not move the camera: the still is the start pose.
        const args = CASES[0].args;
        const plainPath = join(tmpdir(), `still-plain-${process.pid}.webp`);
        const endsPath = join(tmpdir(), `still-ends-${process.pid}.webp`);
        const plain = await runCli([...args, plainPath, '-w', '-q']);
        const ends = await runCli([...args, '--camera-target-end', '3,0,0', '--camera-up-end', '1,0,0', endsPath, '-w', '-q']);
        assert.strictEqual(plain.code, 0, plain.stderr);
        assert.strictEqual(ends.code, 0, ends.stderr);
        const [a, b] = await Promise.all([readFile(plainPath).then(decodeRgba), readFile(endsPath).then(decodeRgba)]);
        assert.ok(
            Buffer.from(a.rgba.buffer, a.rgba.byteOffset, a.rgba.byteLength).equals(Buffer.from(b.rgba.buffer, b.rgba.byteOffset, b.rgba.byteLength)),
            'end options changed a still'
        );
    });
});
