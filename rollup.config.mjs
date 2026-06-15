import { execSync } from 'node:child_process';
import { resolve as resolvePath } from 'node:path';

import json from '@rollup/plugin-json';
import resolve from '@rollup/plugin-node-resolve';
import replace from '@rollup/plugin-replace';
import typescript from '@rollup/plugin-typescript';

import pkg from './package.json' with { type: 'json' };

const revision = (() => {
    try {
        return execSync('git rev-parse --short HEAD').toString().trim();
    } catch (e) {
        return 'unknown';
    }
})();

const versionReplace = () => replace({
    preventAssignment: true,
    values: {
        $_CURRENT_VERSION: pkg.version,
        $_CURRENT_REVISION: revision
    }
});

// Flips the workerBundled build flag to true in the library/CLI bundles. From
// source (tsx: dev, tests) the flag stays false and WorkerQueue runs tasks
// inline; in a build, dist/worker.mjs exists and is spawned from a URL.
const workerBundledPath = resolvePath('src/lib/workers/worker-bundled.ts');
const markWorkerBundled = () => ({
    name: 'mark-worker-bundled',
    load(id) {
        if (id === workerBundledPath) {
            return 'export const workerBundled = true;';
        }
        return null;
    }
});

// node builtins dynamically imported behind runtime guards in
// src/lib/workers; browser bundlers never execute those branches
const nodeExternals = ['node:worker_threads', 'node:os'];

// Worker build - self-contained worker entry shipped as dist/worker.mjs and
// spawned from a URL by WorkerQueue (built first so it exists before the
// other bundles, though they no longer depend on it at build time).
const worker = {
    input: 'src/lib/workers/worker-entry.ts',
    output: {
        dir: 'dist',
        format: 'esm',
        sourcemap: true,
        entryFileNames: 'worker.mjs'
    },
    external: nodeExternals,
    plugins: [
        versionReplace(),
        typescript({
            tsconfig: './tsconfig.json',
            declaration: false,
            declarationDir: undefined
        }),
        resolve(),
        json()
    ],
    cache: false
};

// Library build - ESM (platform agnostic)
const esm = {
    input: 'src/lib/index.ts',
    output: {
        dir: 'dist',
        format: 'esm',
        sourcemap: true,
        entryFileNames: 'index.mjs'
    },
    external: ['playcanvas', ...nodeExternals],
    plugins: [
        markWorkerBundled(),
        versionReplace(),
        typescript({
            tsconfig: './tsconfig.json',
            declaration: true,
            declarationDir: 'dist'
        }),
        resolve(),
        json()
    ],
    cache: false
};

// Library build - CommonJS (for non-module apps)
const cjs = {
    input: 'src/lib/index.ts',
    output: {
        dir: 'dist',
        format: 'cjs',
        sourcemap: true,
        entryFileNames: 'index.cjs',
        exports: 'named'
    },
    external: ['playcanvas', ...nodeExternals],
    plugins: [
        markWorkerBundled(),
        versionReplace(),
        typescript({
            tsconfig: './tsconfig.json',
            declaration: false,
            declarationDir: undefined
        }),
        resolve(),
        json()
    ],
    cache: false
};

// CLI build - Node.js specific
const cli = {
    input: 'src/cli/index.ts',
    output: {
        dir: 'dist',
        format: 'esm',
        sourcemap: true,
        entryFileNames: 'cli.mjs'
    },
    external: ['webgpu', ...nodeExternals],
    plugins: [
        markWorkerBundled(),
        versionReplace(),
        typescript({
            tsconfig: './tsconfig.json',
            declaration: false,
            declarationDir: undefined
        }),
        resolve(),
        json()
    ],
    cache: false
};

export default [worker, esm, cjs, cli];
