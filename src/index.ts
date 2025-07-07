import { exit } from 'node:process';
import { open } from 'node:fs/promises';
import { parseArgs } from 'node:util';
import { resolve } from 'node:path';

import { version } from '../package.json';
import { readPly } from './read-ply';

import { writeSogs } from './write-sogs';
import { writePly } from './write-ply';
import { writeCompressedPly } from './write-compressed-ply';

import { Quat, Vec3 } from 'playcanvas';
import { DataTable } from './data-table';

import { filter } from './filter';
import { transform } from './transform';

const readFile = async (filename: string) => {
    console.log(`reading '${filename}'...`);
    const inputFile = await open(filename, 'r');
    const plyData = await readPly(inputFile);
    await inputFile.close();
    return plyData;
};

const writeFile = async (filename: string, dataTable: DataTable) => {
    if (filename.endsWith('.json')) {
        await writeSogs(filename, dataTable);
    } else if (filename.endsWith('.compressed.ply')) {
        console.log(`writing '${filename}'...`);
        const outputFile = await open(filename, 'w');
        await writeCompressedPly(outputFile, dataTable);
        await outputFile.close();
    } else {
        console.log(`writing '${filename}'...`);
        const outputFile = await open(filename, 'w');
        await writePly(outputFile, {
            comments: [],
            elements: [{
                name: 'vertex',
                dataTable: dataTable
            }]
        });
        await outputFile.close();
    }
};

// combine the supplied tables into one
const combine = (dataTables: DataTable[]) => {
    if (dataTables.length === 1) {
        // nothing to combine
        return dataTables[0];
    }
};

type ProcessOptions = {
    transform?: {
        translate: Vec3;
        rotate: Quat;
        scale: number;
    },
    filter?: {
        invalid: boolean;
        invisible: boolean;
    }
};

// process a data table with standard options
const process = (dataTable: DataTable, options: ProcessOptions) => {
    let result = dataTable;

    // transform
    if (options.transform) {
        transform(
            result,
            options.transform.translate,
            options.transform.rotate,
            options.transform.scale
        );
    }

    // filter rows
    if (options.filter) {
        result = filter(
            result,
            options.filter.invalid,
            options.filter.invisible
        );
    }

    return result;
};

const isGSData = (dataTable: DataTable) => {
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

const parseArguments = () => {
    const { values: v, tokens } = parseArgs({
        tokens: true,
        strict: true,
        allowPositionals: true,
        options: {
            trans: { type: 'string', short: 't', multiple: true },
            rot: { type: 'string', short: 'r', multiple: true },
            scale: { type: 'string', short: 's', multiple: true },
            filter: { type: 'string', short: 'f', multiple: true, default: ['invalid,invisible'] },
            sh: { type: 'string', short: 'h', default: '3' },
        },
    });

    const args = [];
    let current: any = null;

    for (const t of tokens) {
        if (t.kind === 'positional') {
            current = { filename: t.value };
            args.push(current);
        } else if (t.kind === 'option' && current) {
            switch (t.name) {
                case 'trans':
                case 'rot':
                case 'scale':
                    current[t.name] = t.value.split(',').map(Number);
                    break;
                case 'filter':
                    current[t.name] = t.value.split(',').map(f => f.trim());
                    break;
            }
        }
    }

    const getProcessOptions = (args: any): ProcessOptions => {
        const processOptions: ProcessOptions = { };

        if (args.translate || args.rotate || args.scale) {
            const t = args.translate ?? [0, 0, 0];
            const r = args.rotate ?? [0, 0, 0];
            const s = args.scale ? args.scale[0] : 1;
            processOptions.transform = {
                translate: new Vec3(t[0], t[1], t[2]),
                rotate: new Quat().setFromEulerAngles(r[0], r[1], r[2]),
                scale: s
            };
        }

        if (args.filter) {
            processOptions.filter = {
                invalid: args.filter.includes('invalid'),
                invisible: args.filter.includes('invisible')
            };
        } else {
            processOptions.filter = {
                invalid: true,
                invisible: true
            };
        }

        return processOptions;
    };

    const files = args.map((arg) => {
        return {
            filename: arg.filename,
            processOptions: getProcessOptions(arg)
        }
    });

    return {
        files,
        sh: parseInt(v.sh, 10)
    };
}

const usage = `Usage: splat-transform input.ply [options] input.ply [options] ... output.ply [options]
options:
-translate -t x,y,z           Translate splats by x,y,z
-rotate -r    x,y,z           Rotate splats by euler angles x,y,z (in degrees)
-scale -s     x               Scale splats by x (uniform scaling)
-filter -f invalid,invisible  Filter splats by removing invalid splats (NaN, Inf)
`;

const main = async () => {
    console.log(`splat-transform v${version}`);

    // read args
    const args = parseArguments();
    if (args.files.length < 2) {
        console.error(usage);
        exit(1);
    }

    const inputArgs = args.files.slice(0, -1);
    const outputArg = args.files[args.files.length - 1];

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
            if (dataTable.numRows === 0 || !isGSData(dataTable)) {
                return null;
            }

            file.elements[0].dataTable = process(dataTable, inputArg.processOptions);

            return file;
        }));

        // combine inputs into single output dataTable
        const dataTable = process(
            combine(inputFiles.map(file => file.elements[0].dataTable)),
            outputArg.processOptions
        );

        // write file
        await writeFile(resolve(outputArg.filename), dataTable);
    } catch (err) {
        // handle errors
        console.error(`error: ${err.message}`);
        exit(1);
    }

    console.log('done');
};

export { main };
