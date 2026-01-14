import json from '@rollup/plugin-json';
import resolve from '@rollup/plugin-node-resolve';
import typescript from '@rollup/plugin-typescript';

// Library build - platform agnostic
const library = {
    input: 'src/lib/index.ts',
    output: {
        dir: 'dist',
        format: 'esm',
        sourcemap: true,
        entryFileNames: 'index.mjs'
    },
    external: ['playcanvas'],
    plugins: [
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

export default [library, cli];
