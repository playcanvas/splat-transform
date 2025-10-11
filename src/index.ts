import { randomBytes } from 'crypto';
import { lstat, open, rename } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { exit, hrtime } from 'node:process';
import { parseArgs } from 'node:util';

import { Vec3 } from 'playcanvas';

import { version } from '../package.json';
import { Column, DataTable, TypedArray } from './data-table';
import { ProcessAction, processDataTable } from './process';
import { isCompressedPly, decompressPly } from './readers/decompress-ply';
import { readKsplat } from './readers/read-ksplat';
import { readMjs, Param } from './readers/read-mjs';
import { readPly } from './readers/read-ply';
import { readSog } from './readers/read-sog';
import { readSplat } from './readers/read-splat';
import { readSpz } from './readers/read-spz';
import { writeCompressedPly } from './writers/write-compressed-ply';
import { writeCsv } from './writers/write-csv';
import { writeHtml } from './writers/write-html';
import { writePly } from './writers/write-ply';
import { writeSog } from './writers/write-sog';

type Options = {
    overwrite: boolean,
    help: boolean,
    version: boolean,
    cpu: boolean,
    iterations: number,
    viewerSettingsPath?: string
};

const fileExists = async (filename: string) => {
    try {
        await lstat(filename);
        return true;
    } catch (e: any) {
        if (e?.code === 'ENOENT') {
            return false;
        }
        throw e; // real error (permissions, etc)
    }
};

const readFile = async (filename: string, params: Param[]) => {
    const lowerFilename = filename.toLowerCase();
    let fileData;

    console.log(`reading '${filename}'...`);

    if (lowerFilename.endsWith('.mjs')) {
        fileData = await readMjs(filename, params);
    } else {
        const inputFile = await open(filename, 'r');

        if (lowerFilename.endsWith('.ksplat')) {
            fileData = await readKsplat(inputFile);
        } else if (lowerFilename.endsWith('.splat')) {
            fileData = await readSplat(inputFile);
        } else if (lowerFilename.endsWith('.sog') || lowerFilename.endsWith('meta.json')) {
            fileData = await readSog(inputFile, filename);
        } else if (lowerFilename.endsWith('.ply')) {
            const ply = await readPly(inputFile);
            if (isCompressedPly(ply)) {
                fileData = {
                    comments: ply.comments,
                    elements: [{ name: 'vertex', dataTable: decompressPly(ply) }]
                };
            } else {
                fileData = ply;
            }
        } else if (lowerFilename.endsWith('.spz')) {
            fileData = await readSpz(inputFile);
        } else {
            await inputFile.close();
            throw new Error(`Unsupported input file type: ${filename}`);
        }

        await inputFile.close();
    }
    return fileData;
};

const getOutputFormat = (filename: string) => {
    const lowerFilename = filename.toLowerCase();

    if (lowerFilename.endsWith('.csv')) {
        return 'csv';
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

const writeFile = async (filename: string, dataTable: DataTable, options: Options) => {
    // get the output format, throws on failure
    const outputFormat = getOutputFormat(filename);

    console.log(`writing '${filename}'...`);

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
                await writeSog(outputFile, dataTable, filename, options.iterations, options.cpu ? 'cpu' : 'gpu');
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
                await writeHtml(outputFile, dataTable, options.iterations, options.cpu ? 'cpu' : 'gpu', options.viewerSettingsPath);
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

// combine multiple tables into one
// columns with matching name and type are combined
const combine = (dataTables: DataTable[]) => {
    if (dataTables.length === 1) {
        // nothing to combine
        return dataTables[0];
    }

    const findMatchingColumn = (columns: Column[], column: Column) => {
        for (let i = 0; i < columns.length; ++i) {
            if (columns[i].name === column.name &&
                columns[i].dataType === column.dataType) {
                return columns[i];
            }
        }
        return null;
    };

    // make unique list of columns where name and type much match
    const columns = dataTables[0].columns.slice();
    for (let i = 1; i < dataTables.length; ++i) {
        const dataTable = dataTables[i];
        for (let j = 0; j < dataTable.columns.length; ++j) {
            if (!findMatchingColumn(columns, dataTable.columns[j])) {
                columns.push(dataTable.columns[j]);
            }
        }
    }

    // count total number of rows
    const totalRows = dataTables.reduce((sum, dataTable) => sum + dataTable.numRows, 0);

    // construct output dataTable
    const resultColumns = columns.map((column) => {
        const constructor = column.data.constructor as new (length: number) => TypedArray;
        return new Column(column.name, new constructor(totalRows));
    });
    const result = new DataTable(resultColumns);

    // copy data
    let rowOffset = 0;
    for (let i = 0; i < dataTables.length; ++i) {
        const dataTable = dataTables[i];

        for (let j = 0; j < dataTable.columns.length; ++j) {
            const column = dataTable.columns[j];
            const targetColumn = findMatchingColumn(result.columns, column);
            targetColumn.data.set(column.data, rowOffset);
        }

        rowOffset += dataTable.numRows;
    }

    return result;
};

const isGSDataTable = (dataTable: DataTable) => {
    if (![
        'x', 'y', 'z',
        'rot_0', 'rot_1', 'rot_2', 'rot_3',
        'scale_0', 'scale_1', 'scale_2',
        'f_dc_0', 'f_dc_1', 'f_dc_2',
        'opacity'
    ].every(c => dataTable.hasColumn(c))) {
        return false;
    }
    return true;
};

type File = {
    filename: string;
    processActions: ProcessAction[];
};

const parseArguments = () => {
    const { values: v, tokens } = parseArgs({
        tokens: true,
        strict: true,
        allowPositionals: true,
        allowNegative: true,
        options: {
            // global options
            overwrite: { type: 'boolean', short: 'w' },
            help: { type: 'boolean', short: 'h' },
            version: { type: 'boolean', short: 'v' },
            cpu: { type: 'boolean', short: 'c' },
            iterations: { type: 'string', short: 'i' },
            'viewer-settings': { type: 'string', short: 'E' },

            // file options
            translate: { type: 'string', short: 't', multiple: true },
            rotate: { type: 'string', short: 'r', multiple: true },
            scale: { type: 'string', short: 's', multiple: true },
            'filter-nan': { type: 'boolean', short: 'N', multiple: true },
            'filter-value': { type: 'string', short: 'V', multiple: true },
            'filter-harmonics': { type: 'string', short: 'H', multiple: true },
            'filter-box': { type: 'string', short: 'B', multiple: true },
            'filter-sphere': { type: 'string', short: 'S', multiple: true },
            params: { type: 'string', short: 'p', multiple: true }
        }
    });

    const parseNumber = (value: string): number => {
        const result = Number(value);
        if (isNaN(result)) {
            throw new Error(`Invalid number value: ${value}`);
        }
        return result;
    };

    const parseInteger = (value: string): number => {
        const result = parseInt(value, 10);
        if (isNaN(result)) {
            throw new Error(`Invalid integer value: ${value}`);
        }
        return result;
    };

    const parseVec3 = (value: string): Vec3 => {
        const parts = value.split(',').map(parseNumber);
        if (parts.length !== 3 || parts.some(isNaN)) {
            throw new Error(`Invalid Vec3 value: ${value}`);
        }
        return new Vec3(parts[0], parts[1], parts[2]);
    };

    const parseComparator = (value: string): 'lt' | 'lte' | 'gt' | 'gte' | 'eq' | 'neq' => {
        switch (value) {
            case 'lt': return 'lt';
            case 'lte': return 'lte';
            case 'gt': return 'gt';
            case 'gte': return 'gte';
            case 'eq': return 'eq';
            case 'neq': return 'neq';
            default:
                throw new Error(`Invalid comparator value: ${value}`);
        }
    };

    const files: File[] = [];

    const options: Options = {
        overwrite: v.overwrite ?? false,
        help: v.help ?? false,
        version: v.version ?? false,
        cpu: v.cpu ?? false,
        iterations: parseInteger(v.iterations ?? '10'),
        viewerSettingsPath: (v as any)['viewer-settings']
    };

    for (const t of tokens) {
        if (t.kind === 'positional') {
            files.push({
                filename: t.value,
                processActions: []
            });
        } else if (t.kind === 'option' && files.length > 0) {
            const current = files[files.length - 1];
            switch (t.name) {
                case 'translate':
                    current.processActions.push({
                        kind: 'translate',
                        value: parseVec3(t.value)
                    });
                    break;
                case 'rotate':
                    current.processActions.push({
                        kind: 'rotate',
                        value: parseVec3(t.value)
                    });
                    break;
                case 'scale':
                    current.processActions.push({
                        kind: 'scale',
                        value: parseNumber(t.value)
                    });
                    break;
                case 'filter-nan':
                    current.processActions.push({
                        kind: 'filterNaN'
                    });
                    break;
                case 'filter-value': {
                    const parts = t.value.split(',').map((p: string) => p.trim());
                    if (parts.length !== 3) {
                        throw new Error(`Invalid filter-value value: ${t.value}`);
                    }
                    current.processActions.push({
                        kind: 'filterByValue',
                        columnName: parts[0],
                        comparator: parseComparator(parts[1]),
                        value: parseNumber(parts[2])
                    });
                    break;
                }
                case 'filter-harmonics': {
                    const shBands = parseInteger(t.value);
                    if (![0, 1, 2, 3].includes(shBands)) {
                        throw new Error(`Invalid filter-harmonics value: ${t.value}. Must be 0, 1, 2, or 3.`);
                    }
                    current.processActions.push({
                        kind: 'filterBands',
                        value: shBands as 0 | 1 | 2 | 3
                    });

                    break;
                }
                case 'filter-box': {
                    const parts = t.value.split(',').map((p: string) => p.trim());
                    if (parts.length !== 6) {
                        throw new Error(`Invalid filter-box value: ${t.value}`);
                    }

                    const defaults = [-Infinity, -Infinity, -Infinity, Infinity, Infinity, Infinity];
                    const values: number[] = [];
                    for (let i = 0; i < 6; ++i) {
                        if (parts[i] === '' || parts[i] === '-') {
                            values[i] = defaults[i];
                        } else {
                            values[i] = parseNumber(parts[i]);
                        }
                    }

                    current.processActions.push({
                        kind: 'filterBox',
                        min: new Vec3(values[0], values[1], values[2]),
                        max: new Vec3(values[3], values[4], values[5])
                    });
                    break;
                }
                case 'filter-sphere': {
                    const parts = t.value.split(',').map((p: string) => p.trim());
                    if (parts.length !== 4) {
                        throw new Error(`Invalid filter-sphere value: ${t.value}`);
                    }
                    const values = parts.map(parseNumber);
                    current.processActions.push({
                        kind: 'filterSphere',
                        center: new Vec3(values[0], values[1], values[2]),
                        radius: values[3]
                    });
                    break;
                }
                case 'params': {
                    const params = t.value.split(',').map((p: string) => p.trim());
                    for (const param of params) {
                        const parts = param.split('=').map((p: string) => p.trim());
                        current.processActions.push({
                            kind: 'param',
                            name: parts[0],
                            value: parts[1] ?? ''
                        });
                    }
                    break;
                }
            }
        }
    }

    return { files, options };
};

const usage = `
Transform & filter Gaussian splats
===================================

USAGE
  splat-transform [GLOBAL] input [ACTIONS]  ...  output [ACTIONS]

  • Input files become the working set; ACTIONS are applied in order.
  • The last file is the output; actions after it modify the final result.

SUPPORTED INPUTS
    .ply   .compressed.ply   .sog   meta.json   .ksplat   .splat   .spz   .mjs

SUPPORTED OUTPUTS
    .ply   .compressed.ply   .sog   meta.json   .csv   .html

ACTIONS (can be repeated, in any order)
    -t, --translate        <x,y,z>             Translate splats by (x, y, z).
    -r, --rotate           <x,y,z>             Rotate splats by Euler angles (x, y, z), in degrees.
    -s, --scale            <factor>            Uniformly scale splats by factor.
    -N, --filter-nan                           Remove Gaussians with NaN or Inf values.
    -V, --filter-value     <name,cmp,value>    Keep splats where <name> <cmp> <value>
                                               cmp ∈ {lt,lte,gt,gte,eq,neq}
    -H, --filter-harmonics <0|1|2|3>           Remove spherical harmonic bands > n.
    -B, --filter-box       <mx,my,mz,Mx,My,Mz> Remove Gaussians outside box (min, max corners).
    -S, --filter-sphere    <x,y,z,radius>      Remove Gaussians outside sphere (center, radius).
    -p, --params           <key=val,...>       Pass parameters to .mjs generator script.

GLOBAL OPTIONS
    -h, --help                                 Show this help and exit.
    -v, --version                              Show version and exit.
    -w, --overwrite                            Overwrite output file if it exists.
    -c, --cpu                                  Use CPU for spherical harmonic compression.
    -i, --iterations       <n>                 Iterations for SOG SH compression (more = better). Default: 10.
    -E, --viewer-settings  <settings.json>     HTML viewer settings JSON file.

EXAMPLES
    # Scale then translate
    splat-transform bunny.ply -s 0.5 -t 0,0,10 bunny-scaled.ply

    # Merge two files with transforms
    splat-transform -w cloudA.ply -r 0,90,0 cloudB.ply -s 2 merged.compressed.ply

    # HTML viewer with default settings
    splat-transform bunny.ply bunny-viewer.html

    # HTML viewer with custom settings
    splat-transform -E settings.json bunny.ply bunny-viewer.html

GENERATORS (beta)
    # Generate synthetic splats
    splat-transform gen-grid.mjs -p width=500,height=500,scale=0.1 grid.ply
`;

const main = async () => {
    console.log(`splat-transform v${version}`);

    const startTime = hrtime();

    // read args
    const { files, options } = parseArguments();

    // show version and exit
    if (options.version) {
        exit(0);
    }

    // invalid args or show help
    if (files.length < 2 || options.help) {
        console.error(usage);
        exit(1);
    }

    const inputArgs = files.slice(0, -1);
    const outputArg = files[files.length - 1];

    // check overwrite before doing any work
    if (!options.overwrite && await fileExists(outputArg.filename)) {
        console.error(`File '${outputArg.filename}' already exists. Use -w option to overwrite.`);
        exit(1);
    }

    try {
        // read, filter, process input files
        const inputFiles = (await Promise.all(inputArgs.map(async (inputArg) => {
            // extract params
            const params = inputArg.processActions.filter(a => a.kind === 'param').map((p) => {
                return { name: p.name, value: p.value };
            });

            // read input
            const file = await readFile(resolve(inputArg.filename), params);

            // filter out non-gs data
            if (file.elements.length !== 1 || file.elements[0].name !== 'vertex') {
                throw new Error(`Unsupported data in file '${inputArg.filename}'`);
            }

            const element = file.elements[0];

            const { dataTable } = element;
            if (dataTable.numRows === 0 || !isGSDataTable(dataTable)) {
                throw new Error(`Unsupported data in file '${inputArg.filename}'`);
            }

            element.dataTable = processDataTable(dataTable, inputArg.processActions);

            return file;
        }))).filter(file => file !== null);

        // combine inputs into a single output dataTable
        const dataTable = processDataTable(
            combine(inputFiles.map(file => file.elements[0].dataTable)),
            outputArg.processActions
        );

        if (dataTable.numRows === 0) {
            throw new Error('No splats to write');
        }

        console.log(`Loaded ${dataTable.numRows} gaussians`);

        // write file
        await writeFile(resolve(outputArg.filename), dataTable, options);
    } catch (err) {
        // handle errors
        console.error(err);
        exit(1);
    }

    const endTime = hrtime(startTime);

    console.log(`done in ${endTime[0] + endTime[1] / 1e9}s`);

    // something in webgpu seems to keep the process alive after returning
    // from main so force exit
    exit(0);
};

export { main };
