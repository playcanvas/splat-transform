import { basename } from 'pathe';

import { logWrittenFile } from './utils';
import { type FileSystem, writeFile } from '../io/write';
import { dataTableToGaussianCloud, getSpzModule, makeSpzPackOptions } from '../spz-module';
import { logger } from '../utils';

type WriteSpzOptions = {
    filename: string;
    dataTable: import('../data-table').DataTable;
    version?: 3 | 4;
};

/**
 * Writes Gaussian splat data to a bundled SPZ file using the official SPZ
 * WebAssembly backend.
 *
 * @param options - Options including filename and dataTable.
 * @param fs - File system for writing the output file.
 * @ignore
 */
const writeSpz = async (options: WriteSpzOptions, fs: FileSystem) => {
    const { filename, dataTable, version = 4 } = options;
    const writingGroup = logger.group('Writing');

    const spz = await getSpzModule();
    const packOptions = await makeSpzPackOptions({ version });
    const cloud = dataTableToGaussianCloud(dataTable);
    const bytes = spz.saveSpzToBuffer(cloud, packOptions);

    if (version === 4) {
        if (!(bytes[0] === 0x4e && bytes[1] === 0x47 && bytes[2] === 0x53 && bytes[3] === 0x50)) {
            throw new Error('SPZ writer requested version 4 but the backend did not emit an NGSP header');
        }
    } else if (!(bytes[0] === 0x1f && bytes[1] === 0x8b)) {
        throw new Error('SPZ writer requested a legacy version but the backend did not emit gzip-compressed output');
    }

    await writeFile(fs, filename, bytes);

    logWrittenFile(basename(filename), bytes.byteLength);
    writingGroup.end();
};

export { writeSpz };
