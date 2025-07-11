import { open } from 'node:fs/promises';
import { resolve } from 'node:path';
import { exit, hrtime } from 'node:process';
import { parseArgs } from 'node:util';

import { Vec3 } from 'playcanvas';

import { version } from '../package.json';
import { Column, DataTable, TypedArray } from './data-table';
import { ProcessAction, process } from './process';
import { readPly } from './readers/read-ply';
import { writeCompressedPly } from './writers/write-compressed-ply';
import { writePly } from './writers/write-ply';
import { writeSogs } from './writers/write-sogs';

type Options = {
    overwrite: boolean
};

const readFile = async (filename: string) => {
    console.log(`reading '${filename}'...`);
    const inputFile = await open(filename, 'r');
    const plyData = await readPly(inputFile);
    await inputFile.close();
    return plyData;
};

const writeFile = async (filename: string, dataTable: DataTable, options: Options) => {

    // open the output file
    let outputFile;

    try {
        outputFile = await open(filename, options.overwrite ? 'w' : 'wx');
    } catch (err) {
        if (err.code === 'EEXIST') {
            console.error(`File '${filename}' already exists. Use -o option to overwrite.`);
            exit(1);
        } else {
            throw err;
        }
    }

    console.log(`writing '${filename}'...`);

    // write the data
    if (filename.endsWith('.json')) {
        await writeSogs(outputFile, dataTable, filename);
    } else if (filename.endsWith('.compressed.ply')) {
        await writeCompressedPly(outputFile, dataTable);
    } else {
        await writePly(outputFile, {
            comments: [],
            elements: [{
                name: 'vertex',
                dataTable: dataTable
            }]
        });
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
            translate: { type: 'string', short: 't', multiple: true },
            rotate: { type: 'string', short: 'r', multiple: true },
            scale: { type: 'string', short: 's', multiple: true },
            filterNaN: { type: 'boolean', short: 'n', multiple: true },
            filterByValue: { type: 'string', short: 'c', multiple: true },
            filterBands: { type: 'string', short: 'h', multiple: true },
            overwrite: { type: 'boolean', short: 'o' }
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
        overwrite: v.overwrite || false
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

const usage = `Usage: splat-transform input.ply [actions] input.ply [actions] ... output.ply [actions]
actions:
-translate     -t x,y,z                     Translate splats by (x, y, z)
-rotate        -r x,y,z                     Rotate splats by euler angles (x, y, z) (in degrees)
-scale         -s x                         Scale splats by x (uniform scaling)
-filterNaN     -n                           Remove gaussians containing any NaN or Inf value
-filterByValue -c name,comparator,value     Filter gaussians by a value. Specify the value name, comparator (lt, lte, gt, gte, eq, neq) and value
-filterBands   -h 1                         Filter spherical harmonic band data. Value must be 0, 1, 2 or 3.
-overwrite     -o                           Overwrite output file if it exists
`;

const main = async () => {
    console.log(`splat-transform v${version}`);

    const startTime = hrtime();

    // read args
    const { files, options } = parseArguments();
    if (files.length < 2) {
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

    const endTime = hrtime(startTime);

    console.log(`done in ${endTime[1] / 1e9}s`);
};

export { main };
