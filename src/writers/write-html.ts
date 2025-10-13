import { open, readFile, unlink, FileHandle } from 'node:fs/promises';
import os from 'node:os';

import { html, css, js } from '@playcanvas/supersplat-viewer';

import { DataTable } from '../data-table';
import { writeSog } from './write-sog';

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

const writeHtml = async (fileHandle: FileHandle, dataTable: DataTable, iterations: number, shMethod: 'cpu' | 'gpu', viewerSettingsPath?: string) => {
    const pad = (text: string, spaces: number) => {
        const whitespace = ' '.repeat(spaces);
        return text.split('\n').map(line => whitespace + line).join('\n');
    };

    // Load viewer settings from file if provided
    let viewerSettings: ViewerSettings = {};
    if (viewerSettingsPath) {
        const content = await readFile(viewerSettingsPath, 'utf-8');
        try {
            viewerSettings = JSON.parse(content);
        } catch (e) {
            throw new Error(`Failed to parse viewer settings JSON file: ${viewerSettingsPath}`);
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

    const tempSogPath = `${os.tmpdir()}/temp.sog`;
    const tempSog = await open(tempSogPath, 'w+');
    await writeSog(tempSog, dataTable, tempSogPath, iterations, shMethod);
    await tempSog.close();
    const openSog = await open(tempSogPath, 'r');
    const sogData = Buffer.from(await openSog.readFile()).toString('base64');
    await openSog.close();
    await unlink(tempSogPath);

    const style = '<link rel="stylesheet" href="./index.css">';
    const script = '<script type="module" src="./index.js"></script>';
    const settings = 'settings: fetch(settingsUrl).then(response => response.json())';
    const content = 'fetch(contentUrl)';

    const generatedHtml = html
    .replace(style, `<style>\n${pad(css, 12)}\n        </style>`)
    .replace(script, `<script type="module">\n${pad(js, 12)}\n        </script>`)
    .replace(settings, `settings: ${JSON.stringify(mergedSettings)}`)
    .replace(content, `fetch("data:application/octet-stream;base64,${sogData}")`)
    .replace('.compressed.ply', '.sog');

    await fileHandle.write(new TextEncoder().encode(generatedHtml));
};

export { writeHtml };
