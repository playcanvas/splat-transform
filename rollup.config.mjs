import { execSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
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

// Replaces the worker-source placeholder module with the bundled worker code
// built by the `worker` config below (which must come first in the config
// array - rollup builds array configs in order). The worker code travels
// inside the library/CLI bundles as a string and is spawned from a Blob/data
// URL at runtime, so no separate worker file ships with the package.
const workerSourcePath = resolvePath('src/lib/workers/worker-source.ts');
const workerBundlePath = resolvePath('build/worker.mjs');
const inlineWorkerSource = () => ({
    name: 'inline-worker-source',
    load(id) {
        if (id === workerSourcePath) {
            this.addWatchFile(workerBundlePath);
            // the emscripten glue (lib/webp.mjs) calls
            // createRequire(import.meta.url), which throws inside a data: URL
            // worker; node builtins resolve from any base, so substitute one
            const source = readFileSync(workerBundlePath, 'utf8')
                .replace('createRequire(import.meta.url)', "createRequire(globalThis.process.cwd() + '/')");
            return `export const workerSource = ${JSON.stringify(source)};`;
        }
        return null;
    }
});

// node builtins dynamically imported behind runtime guards in
// src/lib/workers; browser bundlers never execute those branches
const nodeExternals = ['node:worker_threads', 'node:os'];

// Worker build - self-contained bundle of the worker entry point, inlined
// into the other bundles by inlineWorkerSource()
const worker = {
    input: 'src/lib/workers/worker-entry.ts',
    output: {
        dir: 'build',
        format: 'esm',
        sourcemap: false,
        entryFileNames: 'worker.mjs'
    },
    external: nodeExternals,
    plugins: [
        versionReplace(),
        typescript({
            tsconfig: './tsconfig.json',
            declaration: false,
            declarationDir: undefined,
            outDir: 'build',
            sourceMap: false,
            inlineSources: false
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
        inlineWorkerSource(),
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
        inlineWorkerSource(),
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
        inlineWorkerSource(),
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
