import { lstat, mkdir, readFile as pathReadFile } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { exit, hrtime } from 'node:process';
import { parseArgs } from 'node:util';

import { GraphicsDevice, Vec3 } from 'playcanvas';

import { createDevice, enumerateAdapters } from './node-device';
import { NodeFileSystem, NodeReadFileSystem } from './node-file-system';
import { version } from '../../package.json';
import {
    combine,
    DataTable,
    getInputFormat,
    readFile,
    getOutputFormat,
    writeFile,
    processDataTable,
    type ProcessAction,
    type Options as LibOptions,
    logger
} from '../lib/index';

/**
 * CLI-specific options extending library options.
 */
interface CliOptions extends LibOptions {
    overwrite: boolean;
    help: boolean;
    version: boolean;
    quiet: boolean;
    listGpus: boolean;
    deviceIdx: number;  // -1 = auto, -2 = CPU, 0+ = GPU index
}

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

const parseArguments = async () => {
    const { values: v, tokens } = parseArgs({
        tokens: true,
        strict: true,
        allowPositionals: true,
        allowNegative: true,
        options: {
            // global options
            overwrite: { type: 'boolean', short: 'w', default: false },
            help: { type: 'boolean', short: 'h', default: false },
            version: { type: 'boolean', short: 'v', default: false },
            quiet: { type: 'boolean', short: 'q', default: false },
            iterations: { type: 'string', short: 'i', default: '10' },
            'list-gpus': { type: 'boolean', short: 'L', default: false },
            gpu: { type: 'string', short: 'g', default: '-1' },
            'lod-select': { type: 'string', short: 'O', default: '' },
            'viewer-settings': { type: 'string', short: 'E', default: '' },
            'lod-chunk-count': { type: 'string', short: 'C', default: '512' },
            'lod-chunk-extent': { type: 'string', short: 'X', default: '16' },
            unbundled: { type: 'boolean', short: 'U', default: false },

            // per-file options
            translate: { type: 'string', short: 't', multiple: true },
            rotate: { type: 'string', short: 'r', multiple: true },
            scale: { type: 'string', short: 's', multiple: true },
            'filter-nan': { type: 'boolean', short: 'N', multiple: true },
            'filter-value': { type: 'string', short: 'V', multiple: true },
            'filter-harmonics': { type: 'string', short: 'H', multiple: true },
            'filter-box': { type: 'string', short: 'B', multiple: true },
            'filter-sphere': { type: 'string', short: 'S', multiple: true },
            params: { type: 'string', short: 'p', multiple: true },
            lod: { type: 'string', short: 'l', multiple: true },
            summary: { type: 'boolean', short: 'm', multiple: true }
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
        const result = parseNumber(value);
        if (!Number.isInteger(result)) {
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

    // Parse gpu option - can be a number or "cpu"
    let deviceIdx: number;
    const gpuValue = v.gpu.toLowerCase();
    if (gpuValue === 'cpu') {
        deviceIdx = -2;  // -2 indicates CPU mode
    } else {
        deviceIdx = parseInteger(v.gpu);
        if (deviceIdx < -1) {
            throw new Error(`Invalid GPU index: ${deviceIdx}. Must be >= 0 or 'cpu'.`);
        }
    }

    const readJsonFile = async (path: string) => {
        const content = await pathReadFile(path, 'utf-8');
        try {
            return JSON.parse(content);
        } catch (e) {
            throw new Error(`Failed to parse viewer settings JSON file: ${path}`);
        }
    };

    const viewerSettingsPath = v['viewer-settings'];

    const options: CliOptions = {
        overwrite: v.overwrite,
        help: v.help,
        version: v.version,
        quiet: v.quiet,
        iterations: parseInteger(v.iterations),
        listGpus: v['list-gpus'],
        deviceIdx,
        lodSelect: v['lod-select'].split(',').filter(v => !!v).map(parseInteger),
        viewerSettingsJson: viewerSettingsPath && await readJsonFile(viewerSettingsPath),
        unbundled: v.unbundled,
        lodChunkCount: parseInteger(v['lod-chunk-count']),
        lodChunkExtent: parseInteger(v['lod-chunk-extent'])
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
                case 'lod': {
                    const lod = parseInteger(t.value);
                    if (lod < 0) {
                        throw new Error(`Invalid lod value: ${t.value}. Must be a non-negative integer.`);
                    }
                    current.processActions.push({
                        kind: 'lod',
                        value: lod
                    });
                    break;
                }
                case 'summary':
                    current.processActions.push({
                        kind: 'summary'
                    });
                    break;
            }
        }
    }

    return { files, options };
};

const usage = `
Transform and Filter Gaussian Splats
====================================

USAGE
  splat-transform [GLOBAL] input [ACTIONS]  ...  output [ACTIONS]

  • Input files become the working set; ACTIONS are applied in order.
  • The last file is the output; actions after it modify the final result.
  • Use /dev/null (Unix) or nul (Windows) as output to discard file output.

SUPPORTED INPUTS
    .ply   .compressed.ply   .sog   meta.json   .ksplat   .splat   .spz   .mjs   .lcc

SUPPORTED OUTPUTS
    .ply   .compressed.ply   .sog   meta.json   .csv   .html   /dev/null   nul

ACTIONS (can be repeated, in any order)
    -t, --translate        <x,y,z>          Translate Gaussians by (x, y, z)
    -r, --rotate           <x,y,z>          Rotate Gaussians by Euler angles (x, y, z), in degrees
    -s, --scale            <factor>         Uniformly scale Gaussians by factor
    -H, --filter-harmonics <0|1|2|3>        Remove spherical harmonic bands > n
    -N, --filter-nan                        Remove Gaussians with NaN or Inf values
    -B, --filter-box       <x,y,z,X,Y,Z>    Remove Gaussians outside box (min, max corners)
    -S, --filter-sphere    <x,y,z,radius>   Remove Gaussians outside sphere (center, radius)
    -V, --filter-value     <name,cmp,value> Keep Gaussians where <name> <cmp> <value>
                                              cmp ∈ {lt,lte,gt,gte,eq,neq}
    -p, --params           <key=val,...>    Pass parameters to .mjs generator script
    -l, --lod              <n>              Specify the level of detail, n >= 0
    -m, --summary                           Print per-column statistics to stdout

GLOBAL OPTIONS
    -h, --help                              Show this help and exit
    -v, --version                           Show version and exit
    -q, --quiet                             Suppress non-error output
    -w, --overwrite                         Overwrite output file if it exists
    -i, --iterations       <n>              Iterations for SOG SH compression (more=better). Default: 10
    -L, --list-gpus                         List available GPU adapters and exit
    -g, --gpu              <n|cpu>          Select device for SOG compression: GPU adapter index | 'cpu'
    -E, --viewer-settings  <settings.json>  HTML viewer settings JSON file
    -U, --unbundled                         Generate unbundled HTML viewer with separate files
    -O, --lod-select       <n,n,...>        Comma-separated LOD levels to read from LCC input
    -C, --lod-chunk-count  <n>              Approximate number of Gaussians per LOD chunk in K. Default: 512
    -X, --lod-chunk-extent <n>              Approximate size of an LOD chunk in world units (m). Default: 16

EXAMPLES
    # Scale then translate
    splat-transform bunny.ply -s 0.5 -t 0,0,10 bunny-scaled.ply

    # Merge two files with transforms and compress to SOG format
    splat-transform -w cloudA.ply -r 0,90,0 cloudB.ply -s 2 merged.sog

    # Generate unbundled HTML viewer with separate CSS, JS and SOG files
    splat-transform -U bunny.ply bunny-viewer.html

    # Generate synthetic splats using a generator script
    splat-transform gen-grid.mjs -p width=500,height=500,scale=0.1 grid.ply

    # Generate LOD with custom chunk size and node split size
    splat-transform -O 0,1,2 -C 1024 -X 32 input.lcc output/lod-meta.json

    # Print statistical summary, then write output
    splat-transform bunny.ply --summary output.ply

    # Print summary without writing a file (discard output)
    splat-transform bunny.ply --summary /dev/null
`;

const main = async () => {
    const startTime = hrtime();

    // read args
    const { files, options } = await parseArguments();

    // inject Node.js-specific logger with process.stdout for progress
    logger.setLogger({
        log: (...args) => console.log(...args),
        warn: (...args) => console.warn(...args),
        error: (...args) => console.error(...args),
        debug: (...args) => console.log(...args),
        progress: text => process.stdout.write(text)
    });

    // configure logger
    logger.setQuiet(options.quiet);

    logger.log(`splat-transform v${version}`);

    // show version and exit
    if (options.version) {
        exit(0);
    }

    // list GPUs and exit
    if (options.listGpus) {
        logger.log('Enumerating available GPU adapters...\n');
        try {
            const adapters = await enumerateAdapters();
            if (adapters.length === 0) {
                logger.log('No GPU adapters found.');
                logger.log('This could mean:');
                logger.log('  - WebGPU is not available on your system');
                logger.log('  - GPU drivers need to be updated');
                logger.log('  - Your GPU does not support WebGPU');
            } else {
                adapters.forEach((adapter) => {
                    logger.log(`[${adapter.index}] ${adapter.name}`);
                });
                logger.log('\nUse -g <index> to select a specific GPU adapter.');
            }
        } catch (err) {
            logger.error('Failed to enumerate GPU adapters:', err);
        }
        exit(0);
    }

    // invalid args or show help
    if (files.length < 2 || options.help) {
        logger.error(usage);
        exit(1);
    }

    const inputArgs = files.slice(0, -1);
    const outputArg = files[files.length - 1];

    const outputFilename = resolve(outputArg.filename);

    // Check for null output (/dev/null on Unix, nul on Windows)
    const isNullOutput = outputFilename === '/dev/null' ||
                         outputFilename.toLowerCase() === 'nul' ||
                         outputFilename.toLowerCase().endsWith('\\nul');

    let outputFormat: ReturnType<typeof getOutputFormat> | null = null;

    if (!isNullOutput) {
        outputFormat = getOutputFormat(outputFilename, options);

        if (options.overwrite) {
            // ensure target directory exists when using -w
            await mkdir(dirname(outputFilename), { recursive: true });
        } else {
            // check overwrite before doing any work
            if (await fileExists(outputFilename)) {
                logger.error(`File '${outputFilename}' already exists. Use -w option to overwrite.`);
                exit(1);
            }

            // for unbundled HTML, also check for additional files
            if (outputFormat === 'html' && options.unbundled) {
                const outputDir = dirname(outputFilename);
                const baseFilename = basename(outputFilename, '.html');
                const filesToCheck = [
                    join(outputDir, 'index.css'),
                    join(outputDir, 'index.js'),
                    join(outputDir, `${baseFilename}.sog`)
                ];

                for (const file of filesToCheck) {
                    if (await fileExists(file)) {
                        logger.error(`File '${file}' already exists. Use -w option to overwrite.`);
                        exit(1);
                    }
                }
            }
        }
    }

    try {
        // Create file system for reading (reused across all input files)
        const nodeFs = new NodeReadFileSystem();

        // read, filter, process input files
        const inputDataTables = (await Promise.all(inputArgs.map(async (inputArg) => {
            // extract params
            const params = inputArg.processActions.filter(a => a.kind === 'param').map((p) => {
                return { name: p.name, value: p.value };
            });

            // read input
            const filename = resolve(inputArg.filename);
            const inputFormat = getInputFormat(filename);

            // For mjs format, convert to file:// URL (Node.js-specific)
            const readFilename = inputFormat === 'mjs' ? `file://${filename}` : filename;

            const dataTables = await readFile({
                filename: readFilename,
                inputFormat,
                options,
                params,
                fileSystem: nodeFs
            });

            for (let i = 0; i < dataTables.length; ++i) {
                const dataTable = dataTables[i];

                if (dataTable.numRows === 0 || !isGSDataTable(dataTable)) {
                    throw new Error(`Unsupported data in file '${inputArg.filename}'`);
                }

                dataTables[i] = processDataTable(dataTable, inputArg.processActions);
            }

            return dataTables;
        }))).flat(1).filter(dataTable => dataTable !== null);

        // special-case the environment dataTable
        const envDataTables = inputDataTables.filter(dt => dt.hasColumn('lod') && dt.getColumnByName('lod').data.every(v => v === -1));
        const nonEnvDataTables = inputDataTables.filter(dt => !dt.hasColumn('lod') || dt.getColumnByName('lod').data.some(v => v !== -1));

        // combine inputs into a single output dataTable
        const dataTable = nonEnvDataTables.length > 0 && processDataTable(
            combine(nonEnvDataTables),
            outputArg.processActions
        );

        if (!dataTable || dataTable.numRows === 0) {
            throw new Error('No Gaussians to write');
        }

        const envDataTable = envDataTables.length > 0 && processDataTable(
            combine(envDataTables),
            outputArg.processActions
        );

        logger.log(`Loaded ${dataTable.numRows} gaussians`);

        // Skip file writing for null output
        if (!isNullOutput) {
            // Create device creator function with caching
            // deviceIdx: -1 = auto, -2 = CPU, 0+ = specific GPU index
            let cachedDevice: GraphicsDevice | undefined;
            const deviceCreator = options.deviceIdx === -2 ? undefined : async () => {
                if (cachedDevice) {
                    return cachedDevice;
                }

                let adapterName: string | undefined;
                if (options.deviceIdx >= 0) {
                    const adapters = await enumerateAdapters();
                    const adapter = adapters[options.deviceIdx];
                    if (adapter) {
                        adapterName = adapter.name;
                    } else {
                        logger.warn(`GPU adapter index ${options.deviceIdx} not found, using default`);
                    }
                }

                cachedDevice = await createDevice(adapterName);
                return cachedDevice;
            };

            // write file
            await writeFile({
                filename: outputFilename,
                outputFormat: outputFormat!,
                dataTable,
                envDataTable,
                options,
                createDevice: deviceCreator
            }, new NodeFileSystem());
        }
    } catch (err) {
        // handle errors
        logger.error(err);
        exit(1);
    }

    const endTime = hrtime(startTime);

    logger.log(`done in ${endTime[0] + endTime[1] / 1e9}s`);

    // something in webgpu seems to keep the process alive after returning
    // from main so force exit
    exit(0);
};

export { main };
