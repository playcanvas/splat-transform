import { css, js, renderViewerHtml } from '@playcanvas/supersplat-viewer';
import { defaultSettings } from '@playcanvas/supersplat-viewer/settings';
import { basename, dirname, join } from 'pathe';

import { logWrittenFile } from './utils';
import { writeSog } from './write-sog';
import { DataTable } from '../data-table';
import { type FileSystem, MemoryFileSystem, writeFile } from '../io/write';
import type { DeviceCreator } from '../types';
import { logger, toBase64 } from '../utils';

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

    const viewerSettings = viewerSettingsJson || defaultSettings('object');
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
            logging: 'silent'
        }, memoryFs);

        // get the memory buffer
        const sogData = toBase64(memoryFs.results.get(sogFilename));

        const resultHtml = renderViewerHtml({
            bootstrap: {
                settings: viewerSettings,
                contentUrl: `data:application/octet-stream;base64,${sogData}`,
                // a data: uri has no usable name, so this selects the sog parser
                contentFilename: 'scene.sog'
            },
            inlineCss: true,
            inlineJs: true
        });

        const htmlBytes = encoder.encode(resultHtml);

        const writingGroup = logger.group('Writing');
        await writeFile(fs, filename, htmlBytes);
        logWrittenFile(basename(filename), htmlBytes.byteLength);
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
            logging: 'flat'
        }, fs);

        // Write CSS file
        const cssPath = join(outputDir, 'index.css');
        const cssBytes = encoder.encode(css);
        await writeFile(fs, cssPath, cssBytes);
        logWrittenFile(basename(cssPath), cssBytes.byteLength);

        // Write JS file
        const jsPath = join(outputDir, 'index.js');
        const jsBytes = encoder.encode(js);
        await writeFile(fs, jsPath, jsBytes);
        logWrittenFile(basename(jsPath), jsBytes.byteLength);

        // Write settings file
        const settingsPath = join(outputDir, 'settings.json');
        const settingsBytes = encoder.encode(JSON.stringify(viewerSettings, null, 4));
        await writeFile(fs, settingsPath, settingsBytes);
        logWrittenFile(basename(settingsPath), settingsBytes.byteLength);

        // Generate HTML referencing the sibling files
        const resultHtml = renderViewerHtml({
            bootstrap: { contentUrl: sogFilename }
        });

        const htmlBytes = encoder.encode(resultHtml);
        await writeFile(fs, filename, htmlBytes);
        logWrittenFile(basename(filename), htmlBytes.byteLength);

        writingGroup.end();
    }
};

export { writeHtml };
