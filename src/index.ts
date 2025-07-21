import { open } from 'node:fs/promises';
import { resolve } from 'node:path';
import { exit } from 'node:process';
import { parseArgs } from 'node:util';

import { Vec3 } from 'playcanvas';

import { version } from '../package.json';
import { Column, DataTable, TypedArray } from './data-table';
import { ProcessAction, process } from './process';
import { readKsplat } from './readers/read-ksplat';
import { readPly } from './readers/read-ply';
import { readSplat } from './readers/read-splat';
import { writeCompressedPly } from './writers/write-compressed-ply';
import { writeCsv } from './writers/write-csv';
import { writePly } from './writers/write-ply';
import { writeSogs } from './writers/write-sogs';

type Options = {
    overwrite: boolean,
    help: boolean,
    version: boolean
};

const readFile = async (filename: string) => {
    console.log(`reading '${filename}'...`);
    const inputFile = await open(filename, 'r');

    const lowerFilename = filename.toLowerCase();
    let fileData;

    if (lowerFilename.endsWith('.ksplat')) {
        fileData = await readKsplat(inputFile);
    } else if (lowerFilename.endsWith('.splat')) {
        fileData = await readSplat(inputFile);
    } else if (lowerFilename.endsWith('.ply')) {
        fileData = await readPly(inputFile);
    } else {
        await inputFile.close();
        throw new Error(`Unsupported input file type: ${filename}`);
    }

    await inputFile.close();
    return fileData;
};

const getOutputFormat = (filename: string) => {
    const lowerFilename = filename.toLowerCase();

    if (lowerFilename.endsWith('.csv')) {
        return 'csv';
    } else if (lowerFilename.endsWith('.json')) {
        return 'json';
    } else if (lowerFilename.endsWith('.compressed.ply')) {
        return 'compressed-ply';
    } else if (lowerFilename.endsWith('.ply')) {
        return 'ply';
    }

    throw new Error(`Unsupported output file type: ${filename}`);
};

const writeFile = async (filename: string, dataTable: DataTable, options: Options) => {

    const outputFormat = getOutputFormat(filename);

    // open the output file
    let outputFile;

    try {
        outputFile = await open(filename, options.overwrite ? 'w' : 'wx');
    } catch (err) {
        if (err.code === 'EEXIST') {
            console.error(`File '${filename}' already exists. Use -w option to overwrite.`);
            exit(1);
        } else {
            throw err;
        }
    }

    console.log(`writing '${filename}'...`);

    // write the data
    switch (outputFormat) {
        case 'csv':
            await writeCsv(outputFile, dataTable);
            break;
        case 'json':
            await writeSogs(outputFile, dataTable, filename);
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
    }

    await outputFile.close();
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
        options: {
            // global options
            overwrite: { type: 'boolean', short: 'w' },
            help: { type: 'boolean', short: 'h' },
            version: { type: 'boolean', short: 'v' },

            // file options
            translate: { type: 'string', short: 't', multiple: true },
            rotate: { type: 'string', short: 'r', multiple: true },
            scale: { type: 'string', short: 's', multiple: true },
            filterNaN: { type: 'boolean', short: 'n', multiple: true },
            filterByValue: { type: 'string', short: 'c', multiple: true },
            filterBands: { type: 'string', short: 'b', multiple: true }
        }
    });

    const parseNumber = (value: string): number => {
        const result = Number(value);
        if (isNaN(result)) {
            throw new Error(`Invalid number value: ${value}`);
        }
        return result;
    };

    const parseVec3 = (value: string): Vec3 => {
        const parts = value.split(',').map(Number);
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
        overwrite: v.overwrite || false,
        help: v.help || false,
        version: v.version || false
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
                case 'filterNaN':
                    current.processActions.push({
                        kind: 'filterNaN'
                    });
                    break;
                case 'filterByValue': {
                    const parts = t.value.split(',').map(p => p.trim());
                    if (parts.length !== 3) {
                        throw new Error(`Invalid filterByValue value: ${t.value}`);
                    }
                    current.processActions.push({
                        kind: 'filterByValue',
                        columnName: parts[0],
                        comparator: parseComparator(parts[1]),
                        value: parseNumber(parts[2])
                    });
                    break;
                }
                case 'filterBands': {
                    const shBands = parseNumber(t.value);
                    if (![0, 1, 2, 3].includes(shBands)) {
                        throw new Error(`Invalid filterBands value: ${t.value}. Must be 0, 1, 2, or 3.`);
                    }
                    current.processActions.push({
                        kind: 'filterBands',
                        value: shBands as 0 | 1 | 2 | 3
                    });

                    break;
                }
            }
        }
    }

    return { files, options };
};

const usage = `
Apply geometric transforms & filters to Gaussian-splat point clouds
===================================================================

USAGE
  splat-transform [GLOBAL]  <input.{ply|splat|ksplat}> [ACTIONS]  ...  <output.{ply|compressed.ply|meta.json|csv}> [ACTIONS]

  • Every time an input file appears, it becomes the current working set; the following
    ACTIONS are applied in the order listed.  
  • The last file on the command line is treated as the output; anything after it is
    interpreted as actions that modify the final result.

SUPPORTED INPUTS
    .ply   .splat   .ksplat

SUPPORTED OUTPUTS
    .ply   .compressed.ply   meta.json (SOGS)   .csv

ACTIONS (can be repeated, in any order)
    -t, --translate  x,y,z                  Translate splats by (x, y, z)
    -r, --rotate     x,y,z                  Rotate splats by Euler angles (deg)
    -s, --scale      x                      Uniformly scale splats by factor x
    -n, --filterNaN                         Remove any Gaussian containing NaN/Inf
    -c, --filterByValue name,cmp,value      Keep splats where  <name> <cmp> <value>
                                            cmp ∈ {lt,lte,gt,gte,eq,neq}
    -b, --filterBands  {0|1|2|3}            Strip spherical-harmonic bands > N

GLOBAL OPTIONS
    -w, --overwrite                         Overwrite output file if it already exists
    -h, --help                              Show this help and exit
    -v, --version                           Show version and exit

EXAMPLES
    # Simple scale-then-translate
    splat-transform bunny.ply -s 0.5 -t 0,0,10 bunny_scaled.ply

    # Chain two inputs and write compressed output, overwriting if necessary
    splat-transform -w cloudA.ply -r 0,90,0 cloudB.ply -s 2 merged.compressed.ply
`;

const main = async () => {
    console.log(`splat-transform v${version}`);

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

    try {
        // read, filter, process input files
        const inputFiles = await Promise.all(inputArgs.map(async (inputArg) => {
            const file = await readFile(resolve(inputArg.filename));

            // filter out non-gs files
            if (file.elements.length !== 1) {
                return null;
            }

            const element = file.elements[0];
            if (element.name !== 'vertex') {
                return null;
            }

            const { dataTable } = element;
            if (dataTable.numRows === 0 || !isGSDataTable(dataTable)) {
                return null;
            }

            file.elements[0].dataTable = process(dataTable, inputArg.processActions);

            return file;
        }));

        // combine inputs into a single output dataTable
        const dataTable = process(
            combine(inputFiles.map(file => file.elements[0].dataTable)),
            outputArg.processActions
        );

        // write file
        await writeFile(resolve(outputArg.filename), dataTable, options);
    } catch (err) {
        // handle errors
        console.error(`error: ${err.message}`);
        exit(1);
    }

    console.log('done');
};

export { main };
