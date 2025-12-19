import { DataTable } from './data-table/data-table';
import { Options } from './types';
import { logger } from './utils/logger';
import { writeCompressedPly } from './writers/write-compressed-ply';
import { writeCsv } from './writers/write-csv';
import { writeHtml } from './writers/write-html';
import { writeLod } from './writers/write-lod';
import { writePly } from './writers/write-ply';
import { writeSog } from './writers/write-sog';
import { Platform } from './serialize/platform';

type OutputFormat = 'csv' | 'sog' | 'sog-bundle' | 'lod' | 'compressed-ply' | 'ply' | 'html' | 'html-bundle';

type WriteOptions = {
    filename: string;
    outputFormat: OutputFormat;
    dataTable: DataTable;
    envDataTable?: DataTable;
    options: Options;
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
    }

    throw new Error(`Unsupported output file type: ${filename}`);
};

const writeFile = async (writeOptions: WriteOptions, platform: Platform) => {
    const { filename, outputFormat, dataTable, envDataTable, options } = writeOptions;

    logger.info(`writing '${filename}'...`);

    // write the file data
    switch (outputFormat) {
        case 'csv':
            await writeCsv({ filename, dataTable }, platform);
            break;
        case 'sog':
        case 'sog-bundle':
            await writeSog({
                filename,
                dataTable,
                bundle: outputFormat === 'sog-bundle',
                iterations: options.iterations,
                deviceIdx: options.deviceIdx
            }, platform);
            break;
        case 'lod':
            await writeLod({
                filename,
                dataTable,
                envDataTable,
                iterations: options.iterations,
                deviceIdx: options.deviceIdx,
                chunkCount: options.lodChunkCount,
                chunkExtent: options.lodChunkExtent
            }, platform);
            break;
        case 'compressed-ply':
            await writeCompressedPly({ filename, dataTable }, platform);
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
            }, platform);
            break;
        case 'html':
        case 'html-bundle':
            await writeHtml({
                filename,
                dataTable,
                viewerSettingsJson: options.viewerSettingsJson,
                bundle: outputFormat === 'html-bundle',
                iterations: options.iterations,
                deviceIdx: options.deviceIdx
            }, platform);
            break;
    }
};

export { getOutputFormat, writeFile, type OutputFormat };
