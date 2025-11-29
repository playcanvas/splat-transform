import { open, readFile, unlink, writeFile, FileHandle } from 'node:fs/promises';
import { dirname, basename, join } from 'node:path';
import os from 'node:os';

import { html, css, js } from '@playcanvas/supersplat-viewer';

import { DataTable } from '../data-table';
import { writeSog } from './write-sog';
import { Options } from '../types';

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

const writeHtml = async (fileHandle: FileHandle, dataTable: DataTable, outputFilename: string, options: Options) => {
    const pad = (text: string, spaces: number) => {
        const whitespace = ' '.repeat(spaces);
        return text.split('\n').map(line => whitespace + line).join('\n');
    };

    // Load viewer settings from file if provided
    let viewerSettings: ViewerSettings = {};
    if (options.viewerSettingsPath) {
        const content = await readFile(options.viewerSettingsPath, 'utf-8');
        try {
            viewerSettings = JSON.parse(content);
        } catch (e) {
            throw new Error(`Failed to parse viewer settings JSON file: ${options.viewerSettingsPath}`);
        }
    }

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

    let generatedHtml: string;

    if (options.unbundled) {
        // Unbundled mode: write separate files
        const outputDir = dirname(outputFilename);
        const baseFilename = basename(outputFilename, '.html');
        const sogFilename = `${baseFilename}.sog`;
        const sogPath = join(outputDir, sogFilename);

        // Write .sog file
        const sogFile = await open(sogPath, 'w');
        await writeSog(sogFile, dataTable, sogPath, options);
        await sogFile.close();

        // Write CSS file
        const cssPath = join(outputDir, 'index.css');
        await writeFile(cssPath, css);

        // Write JS file
        const jsPath = join(outputDir, 'index.js');
        await writeFile(jsPath, js);

        // Generate HTML with external references
        const settings = 'settings: fetch(settingsUrl).then(response => response.json())';
        const content = 'fetch(contentUrl)';

        generatedHtml = html
            .replace(settings, `settings: ${JSON.stringify(mergedSettings)}`)
            .replace(content, `fetch("${sogFilename}")`)
            .replace('.compressed.ply', '.sog');
    } else {
        // Bundled mode: embed everything in the HTML
        const tempSogPath = `${os.tmpdir()}/temp.sog`;
        const tempSog = await open(tempSogPath, 'w+');
        await writeSog(tempSog, dataTable, tempSogPath, options);
        await tempSog.close();
        const openSog = await open(tempSogPath, 'r');
        const sogData = Buffer.from(await openSog.readFile()).toString('base64');
        await openSog.close();
        await unlink(tempSogPath);

        const style = '<link rel="stylesheet" href="./index.css">';
        const script = 'import { main } from \'./index.js\';';
        const settings = 'settings: fetch(settingsUrl).then(response => response.json())';
        const content = 'fetch(contentUrl)';

        generatedHtml = html
            .replace(style, `<style>\n${pad(css, 12)}\n        </style>`)
            .replace(script, js)
            .replace(settings, `settings: ${JSON.stringify(mergedSettings)}`)
            .replace(content, `fetch("data:application/octet-stream;base64,${sogData}")`)
            .replace('.compressed.ply', '.sog');
    }

    await fileHandle.write(new TextEncoder().encode(generatedHtml));
};

export { writeHtml };
