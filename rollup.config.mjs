import { execSync } from 'node:child_process';

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

// Library build - ESM (platform agnostic)
const esm = {
    input: 'src/lib/index.ts',
    output: {
        dir: 'dist',
        format: 'esm',
        sourcemap: true,
        entryFileNames: 'index.mjs'
    },
    external: ['playcanvas'],
    plugins: [
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
    external: ['playcanvas'],
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

// CLI build - Node.js specific
const cli = {
    input: 'src/cli/index.ts',
    output: {
        dir: 'dist',
        format: 'esm',
        sourcemap: true,
        entryFileNames: 'cli.mjs'
    },
    external: ['webgpu'],
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

export default [esm, cjs, cli];
