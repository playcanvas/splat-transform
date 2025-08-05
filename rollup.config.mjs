import resolve from '@rollup/plugin-node-resolve';
import typescript from '@rollup/plugin-typescript';
import json from '@rollup/plugin-json';
import { string } from 'rollup-plugin-string';
import sass from 'sass';
import scss from 'rollup-plugin-scss';
import postcss from 'postcss';
import autoprefixer from 'autoprefixer';
import path from 'path';
const PCUI_DIR = path.resolve('node_modules/@playcanvas/pcui');


const application = {
    input: 'src/index.ts',
    output: {
        dir: 'dist',
        format: 'esm',
        sourcemap: true,
        entryFileNames: '[name].mjs',
    },
    external: ['sharp', 'webgpu', 'jsdom'],
    plugins: [
        typescript(),
        resolve(),
        json(),
        string({
            include: ['submodules/supersplat-viewer/dist/*']
        }),
        scss({
            sourceMap: true,
            runtime: sass,
            processor: (css) => {
                return postcss([autoprefixer])
                    .process(css, { from: undefined })
                    .then(result => result.css);
            },
            fileName: 'index.css',
            includePaths: [`${PCUI_DIR}/dist`],
            exclude: ['submodules/**']
        }),
    ],
    cache: false
};

export default [
    application
];
