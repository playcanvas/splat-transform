#!/usr/bin/env node
/**
 * Regenerate the render-golden fixtures from the current CLI behaviour.
 *
 * Usage:
 *     node test/render-golden.regenerate.mjs
 *
 * Reads cases from `render-golden.cases.mjs` and rewrites each golden
 * .webp in place. Only run this when an intentional renderer change has
 * landed and you've verified the new output is correct; the goldens are
 * the *expected* output, so regenerating them blesses the new behaviour.
 */

import { spawnSync } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { CASES } from './render-golden.cases.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = dirname(__dirname);

for (const { name, args, goldenPath } of CASES) {
    const outPath = join(__dirname, goldenPath);
    const cli = join(rootDir, 'bin/cli.mjs');
    const result = spawnSync(process.execPath, [cli, ...args, outPath, '-w', '-q'], {
        cwd: rootDir,
        stdio: 'inherit'
    });
    if (result.status !== 0) {
        console.error(`Failed to regenerate ${name}`);
        process.exit(result.status ?? 1);
    }
    console.log(`Regenerated ${name} → ${goldenPath}`);
}
