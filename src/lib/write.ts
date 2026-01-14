import { DataTable } from './data-table/data-table';
import { type FileSystem } from './io/write';
import { Options } from './types';
import { logger } from './utils/logger';
import { writeCompressedPly } from './writers/write-compressed-ply';
import { writeCsv } from './writers/write-csv';
import { writeHtml } from './writers/write-html';
import { writeLod } from './writers/write-lod';
import { writePly } from './writers/write-ply';
import { writeSog, type DeviceCreator } from './writers/write-sog';
import { writeSummary } from './writers/write-summary';

type OutputFormat = 'csv' | 'sog' | 'sog-bundle' | 'lod' | 'compressed-ply' | 'ply' | 'html' | 'html-bundle' | 'summary' | 'summary-md';

type WriteOptions = {
    filename: string;
    outputFormat: OutputFormat;
    dataTable: DataTable;
    envDataTable?: DataTable;
    options: Options;
    createDevice?: DeviceCreator;
};

const getOutputFormat = (filename: string, options: Options): OutputFormat => {
    const lowerFilename = filename.toLowerCase();

    if (lowerFilename.endsWith('.csv')) {
        return 'csv';
    } else if (lowerFilename.endsWith('lod-meta.json')) {
        return 'lod';
    } else if (lowerFilename.endsWith('.sog')) {
        return 'sog-bundle';
    } else if (lowerFilename.endsWith('meta.json')) {
        return 'sog';
    } else if (lowerFilename.endsWith('.compressed.ply')) {
        return 'compressed-ply';
    } else if (lowerFilename.endsWith('.ply')) {
        return 'ply';
    } else if (lowerFilename.endsWith('.html')) {
        return options.unbundled ? 'html' : 'html-bundle';
    } else if (lowerFilename.endsWith('.summary.json')) {
        return 'summary';
    } else if (lowerFilename.endsWith('.summary.md')) {
        return 'summary-md';
    }

    throw new Error(`Unsupported output file type: ${filename}`);
};

const writeFile = async (writeOptions: WriteOptions, fs: FileSystem) => {
    const { filename, outputFormat, dataTable, envDataTable, options, createDevice } = writeOptions;

    logger.log(`writing '${filename}'...`);

    // write the file data
    switch (outputFormat) {
        case 'csv':
            await writeCsv({ filename, dataTable }, fs);
            break;
        case 'sog':
        case 'sog-bundle':
            await writeSog({
                filename,
                dataTable,
                bundle: outputFormat === 'sog-bundle',
                iterations: options.iterations,
                createDevice
            }, fs);
            break;
        case 'lod':
            await writeLod({
                filename,
                dataTable,
                envDataTable,
                iterations: options.iterations,
                createDevice,
                chunkCount: options.lodChunkCount,
                chunkExtent: options.lodChunkExtent
            }, fs);
            break;
        case 'compressed-ply':
            await writeCompressedPly({ filename, dataTable }, fs);
            break;
        case 'ply':
            await writePly({
                filename,
                plyData: {
                    comments: [],
                    elements: [{
                        name: 'vertex',
                        dataTable: dataTable
                    }]
                }
            }, fs);
            break;
        case 'html':
        case 'html-bundle':
            await writeHtml({
                filename,
                dataTable,
                viewerSettingsJson: options.viewerSettingsJson,
                bundle: outputFormat === 'html-bundle',
                iterations: options.iterations,
                createDevice
            }, fs);
            break;
        case 'summary':
            await writeSummary({ filename, dataTable, format: 'json' }, fs);
            break;
        case 'summary-md':
            await writeSummary({ filename, dataTable, format: 'md' }, fs);
            break;
    }
};

export { getOutputFormat, writeFile, type OutputFormat };
