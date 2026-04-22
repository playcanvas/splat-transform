import { html, css, js } from '@playcanvas/supersplat-viewer';
import { basename, dirname, join } from 'pathe';

import { writeSog } from './write-sog';
import { DataTable } from '../data-table';
import { type FileSystem, MemoryFileSystem, writeFile } from '../io/write';
import type { DeviceCreator } from '../types';
import { fmtBytes, logger, toBase64 } from '../utils';

const defaultSettings = {
    version: 2,
    tonemapping: 'none',
    highPrecisionRendering: false,
    background: { color: [0.4, 0.4, 0.4] },
    postEffectSettings: {
        sharpness: { enabled: false, amount: 0 },
        bloom: { enabled: false, intensity: 1, blurLevel: 2 },
        grading: { enabled: false, brightness: 0, contrast: 1, saturation: 1, tint: [1, 1, 1] },
        vignette: { enabled: false, intensity: 0.5, inner: 0.3, outer: 0.75, curvature: 1 },
        fringing: { enabled: false, intensity: 0.5 }
    },
    animTracks: [] as any[],
    cameras: [{
        initial: {
            position: [2, 2, -2],
            target: [0, 0, 0],
            fov: 75
        }
    }],
    annotations: [] as any[],
    startMode: 'default'
};

type WriteHtmlOptions = {
    filename: string;
    dataTable: DataTable;
    viewerSettingsJson?: any;
    bundle: boolean;
    iterations: number;
    createDevice?: DeviceCreator;
};

/**
 * Writes Gaussian splat data as a self-contained HTML viewer.
 *
 * Creates an interactive 3D viewer that can be opened directly in a browser.
 * Uses the PlayCanvas SuperSplat viewer for rendering.
 *
 * @param options - Options including filename, data, and viewer settings.
 * @param fs - File system for writing output files.
 * @ignore
 */
const writeHtml = async (options: WriteHtmlOptions, fs: FileSystem) => {
    const { filename, dataTable, viewerSettingsJson, bundle, iterations, createDevice } = options;

    const pad = (text: string, spaces: number) => {
        const whitespace = ' '.repeat(spaces);
        return text.split('\n').map(line => whitespace + line).join('\n');
    };

    const viewerSettings = viewerSettingsJson || defaultSettings;
    const encoder = new TextEncoder();

    if (bundle) {
        // Bundled mode: embed everything in the HTML
        const memoryFs = new MemoryFileSystem();

        const sogFilename = 'temp.sog';
        await writeSog({
            filename: sogFilename,
            dataTable,
            bundle: true,
            iterations,
            createDevice,
            silent: true
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
        .replace(settings, `settings: ${JSON.stringify(viewerSettings)}`)
        .replace(content, `fetch("data:application/octet-stream;base64,${sogData}")`)
        .replace('.compressed.ply', '.sog');

        const htmlBytes = encoder.encode(resultHtml);

        const writingGroup = logger.group('Writing');
        await writeFile(fs, filename, htmlBytes);
        logger.info(`${basename(filename)} (${fmtBytes(htmlBytes.byteLength)})`);
        writingGroup.end();
    } else {
        // Unbundled mode: write separate files
        const outputDir = dirname(filename);
        const baseFilename = basename(filename, '.html');
        const sogFilename = `${baseFilename}.sog`;
        const sogPath = join(outputDir, sogFilename);

        const writingGroup = logger.group('Writing');

        // Write .sog file (its files are emitted flat into our Writing group)
        await writeSog({
            filename: sogPath,
            dataTable,
            bundle: true,
            iterations,
            createDevice,
            omitWritingGroup: true
        }, fs);

        // Write CSS file
        const cssPath = join(outputDir, 'index.css');
        const cssBytes = encoder.encode(css);
        await writeFile(fs, cssPath, cssBytes);
        logger.info(`${basename(cssPath)} (${fmtBytes(cssBytes.byteLength)})`);

        // Write JS file
        const jsPath = join(outputDir, 'index.js');
        const jsBytes = encoder.encode(js);
        await writeFile(fs, jsPath, jsBytes);
        logger.info(`${basename(jsPath)} (${fmtBytes(jsBytes.byteLength)})`);

        // Write settings file
        const settingsPath = join(outputDir, 'settings.json');
        const settingsBytes = encoder.encode(JSON.stringify(viewerSettings, null, 4));
        await writeFile(fs, settingsPath, settingsBytes);
        logger.info(`${basename(settingsPath)} (${fmtBytes(settingsBytes.byteLength)})`);

        // Generate HTML with external references
        const content = 'fetch(contentUrl)';

        const resultHtml = html
        .replace(content, `fetch("${sogFilename}")`)
        .replace('.compressed.ply', '.sog');

        const htmlBytes = encoder.encode(resultHtml);
        await writeFile(fs, filename, htmlBytes);
        logger.info(`${basename(filename)} (${fmtBytes(htmlBytes.byteLength)})`);

        writingGroup.end();
    }
};

export { writeHtml };
