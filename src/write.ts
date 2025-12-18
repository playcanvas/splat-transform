import { randomBytes } from 'crypto';
import { open, rename } from 'node:fs/promises';
import { basename, dirname, join } from 'node:path';

import { DataTable } from './data-table/data-table';
import { Options } from './types';
import { logger } from './utils/logger';
import { writeCompressedPly } from './writers/write-compressed-ply';
import { writeCsv } from './writers/write-csv';
import { writeHtml } from './writers/write-html';
import { writeLod } from './writers/write-lod';
import { writePly } from './writers/write-ply';
import { writeSog } from './writers/write-sog';

type OutputFormat = 'csv' | 'sog' | 'lod' | 'compressed-ply' | 'ply' | 'html';

const getOutputFormat = (filename: string): OutputFormat => {
    const lowerFilename = filename.toLowerCase();

    if (lowerFilename.endsWith('.csv')) {
        return 'csv';
    } else if (lowerFilename.endsWith('lod-meta.json')) {
        return 'lod';
    } else if (lowerFilename.endsWith('.sog') || lowerFilename.endsWith('meta.json')) {
        return 'sog';
    } else if (lowerFilename.endsWith('.compressed.ply')) {
        return 'compressed-ply';
    } else if (lowerFilename.endsWith('.ply')) {
        return 'ply';
    } else if (lowerFilename.endsWith('.html')) {
        return 'html';
    }

    throw new Error(`Unsupported output file type: ${filename}`);
};

const writeFile = async (filename: string, dataTable: DataTable, envDataTable: DataTable | null, options: Options) => {
    // get the output format, throws on failure
    const outputFormat = getOutputFormat(filename);

    logger.info(`writing '${filename}'...`);

    // write to a temporary file and rename on success
    const tmpFilename = `.${basename(filename)}.${process.pid}.${Date.now()}.${randomBytes(6).toString('hex')}.tmp`;
    const tmpPathname = join(dirname(filename), tmpFilename);

    // open the tmp output file
    const outputFile = await open(tmpPathname, 'wx');

    try {
        // write the file data
        switch (outputFormat) {
            case 'csv':
                await writeCsv(outputFile, dataTable);
                break;
            case 'sog':
                await writeSog(outputFile, dataTable, filename, options);
                break;
            case 'lod':
                await writeLod(outputFile, dataTable, envDataTable, filename, options);
                break;
            case 'compressed-ply':
                await writeCompressedPly(outputFile, dataTable);
                break;
            case 'ply':
                await writePly(outputFile, {
                    comments: [],
                    elements: [{
                        name: 'vertex',
                        dataTable: dataTable
                    }]
                });
                break;
            case 'html':
                await writeHtml(outputFile, dataTable, filename, options);
                break;
        }

        // flush to disk
        await outputFile.sync();
    } finally {
        await outputFile.close().catch(() => { /* ignore */ });
    }

    // atomically rename to target filename
    await rename(tmpPathname, filename);
};

export { getOutputFormat, writeFile, type OutputFormat };
