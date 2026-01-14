import json from '@rollup/plugin-json';
import resolve from '@rollup/plugin-node-resolve';
import esbuild from 'rollup-plugin-esbuild';

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
        esbuild({
            target: 'es2022',
            tsconfig: './tsconfig.json'
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
        esbuild({
            target: 'es2022',
            platform: 'node',
            tsconfig: './tsconfig.json'
        }),
        resolve(),
        json()
    ],
    cache: false
};

export default [library, cli];
