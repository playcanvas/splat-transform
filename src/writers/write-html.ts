import { open, unlink, FileHandle } from 'node:fs/promises';
import os from 'node:os';

import { html, css, js } from '@playcanvas/supersplat-viewer';
import { Vec3 } from 'playcanvas';

import { DataTable } from '../data-table';
import { writeSog } from './write-sog';

const writeHtml = async (fileHandle: FileHandle, dataTable: DataTable, camera: Vec3, target: Vec3, iterations: number, shMethod: 'cpu' | 'gpu') => {
    const pad = (text: string, spaces: number) => {
        const whitespace = ' '.repeat(spaces);
        return text.split('\n').map(line => whitespace + line).join('\n');
    };
    const encodeBase64 = (bytes: Uint8Array) => {
        let binary = '';
        const len = bytes.byteLength;
        for (let i = 0; i < len; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return Buffer.from(binary, 'binary').toString('base64');
    };

    const experienceSettings = {
        camera: {
            fov: 50,
            position: [camera.x, camera.y, camera.z],
            target: [target.x, target.y, target.z],
            startAnim: 'none',
            animTrack: undefined as unknown as string | undefined
        },
        background: {
            color: [0.4, 0.4, 0.4]
        },
        animTracks: [] as unknown[]
    };

    const tempSogPath = `${os.tmpdir()}/temp.sog`;
    const tempSog = await open(tempSogPath, 'w+');
    await writeSog(tempSog, dataTable, tempSogPath, iterations, shMethod);
    await tempSog.close();
    const openSog = await open(tempSogPath, 'r');
    const sogData = encodeBase64(await openSog.readFile());
    await openSog.close();
    await unlink(tempSogPath);

    const style = '<link rel="stylesheet" href="./index.css">';
    const script = '<script type="module" src="./index.js"></script>';
    const settings = 'settings: fetch(settingsUrl).then(response => response.json())';
    const content = 'fetch(contentUrl)';

    const generatedHtml = html
    .replace(style, `<style>\n${pad(css, 12)}\n        </style>`)
    .replace(script, `<script type="module">\n${pad(js, 12)}\n        </script>`)
    .replace(settings, `settings: ${JSON.stringify(experienceSettings)}`)
    .replace(content, `fetch("data:application/octet-stream;base64,${sogData}")`)
    .replace('.compressed.ply', '.sog');

    await fileHandle.write(new TextEncoder().encode(generatedHtml));

};

export { writeHtml };
