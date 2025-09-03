import json from '@rollup/plugin-json';
import resolve from '@rollup/plugin-node-resolve';
import typescript from '@rollup/plugin-typescript';

const application = {
    input: 'src/index.ts',
    output: {
        dir: 'dist',
        format: 'esm',
        sourcemap: true,
        entryFileNames: '[name].mjs'
    },
    external: ['sharp', 'webgpu', 'jsdom'],
    plugins: [
        typescript(),
        resolve(),
        json()
    ],
    cache: false
};

export default [
    application
];
