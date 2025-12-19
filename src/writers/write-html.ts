import { dirname, basename, join } from 'node:path';

import { html, css, js } from '@playcanvas/supersplat-viewer';

import { writeSog } from './write-sog';
import { DataTable } from '../data-table/data-table';
import { FileSystem } from '../serialize/file-system';
import { MemoryFileSystem } from '../serialize/memory-file-system';
import { writeFile } from '../serialize/write-helpers';
import { toBase64 } from '../utils/base64';

type ViewerSettings = {
    camera?: {
        fov?: number;
        position?: [number, number, number];
        target?: [number, number, number];
        startAnim?: string;
        animTrack?: string;
    };
    background?: {
        color?: [number, number, number];
    };
    animTracks?: unknown[];
};

type WriteHtmlOptions = {
    filename: string;
    dataTable: DataTable;
    viewerSettingsJson?: any;
    bundle: boolean;
    iterations: number;
    deviceIdx: number;
};

const writeHtml = async (options: WriteHtmlOptions, fs: FileSystem) => {
    const { filename, dataTable, viewerSettingsJson, bundle, iterations, deviceIdx } = options;

    const pad = (text: string, spaces: number) => {
        const whitespace = ' '.repeat(spaces);
        return text.split('\n').map(line => whitespace + line).join('\n');
    };

    // Load viewer settings from file if provided
    const viewerSettings = (viewerSettingsJson ?? {})as ViewerSettings;

    // Merge provided settings with defaults
    const mergedSettings = {
        camera: {
            fov: 50,
            position: [2, 2, -2] as [number, number, number],
            target: [0, 0, 0] as [number, number, number],
            startAnim: 'none',
            animTrack: undefined as string | undefined,
            ...viewerSettings.camera
        },
        background: {
            color: [0.4, 0.4, 0.4] as [number, number, number],
            ...viewerSettings.background
        },
        animTracks: viewerSettings.animTracks ?? []
    };

    if (bundle) {
        // Bundled mode: embed everything in the HTML
        const memoryFs = new MemoryFileSystem();

        const sogFilename = 'temp.sog';
        await writeSog({
            filename: sogFilename,
            dataTable,
            bundle: true,
            iterations,
            deviceIdx
        }, memoryFs);

        // get the memory buffer
        const sogData = toBase64(memoryFs.results.get(sogFilename));

        const style = '<link rel="stylesheet" href="./index.css">';
        const script = 'import { main } from \'./index.js\';';
        const settings = 'settings: fetch(settingsUrl).then(response => response.json())';
        const content = 'fetch(contentUrl)';

        const resultHtml = html
        .replace(style, `<style>\n${pad(css, 12)}\n        </style>`)
        .replace(script, js)
        .replace(settings, `settings: ${JSON.stringify(mergedSettings)}`)
        .replace(content, `fetch("data:application/octet-stream;base64,${sogData}")`)
        .replace('.compressed.ply', '.sog');

        await writeFile(fs, filename, resultHtml);
    } else {
        // Unbundled mode: write separate files
        const outputDir = dirname(filename);
        const baseFilename = basename(filename, '.html');
        const sogFilename = `${baseFilename}.sog`;
        const sogPath = join(outputDir, sogFilename);

        // Write .sog file
        await writeSog({
            filename: sogPath,
            dataTable,
            bundle: true,
            iterations: 0,
            deviceIdx: -1
        }, fs);

        // Write CSS file
        const cssPath = join(outputDir, 'index.css');
        await writeFile(fs, cssPath, css);

        // Write JS file
        const jsPath = join(outputDir, 'index.js');
        await writeFile(fs, jsPath, js);

        // Generate HTML with external references
        const settings = 'settings: fetch(settingsUrl).then(response => response.json())';
        const content = 'fetch(contentUrl)';

        const resultHtml = html
        .replace(settings, `settings: ${JSON.stringify(mergedSettings)}`)
        .replace(content, `fetch("${sogFilename}")`)
        .replace('.compressed.ply', '.sog');

        await writeFile(fs, filename, resultHtml);
    }
};

export { writeHtml };
