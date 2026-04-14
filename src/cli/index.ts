import { lstat, mkdir, readFile as pathReadFile } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import process, { exit, hrtime } from 'node:process';
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
    type FilterFloaters,
    type FilterCluster,
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
    mem: boolean;
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

const cliOptionsConfig = {
    // global options
    overwrite: { type: 'boolean', short: 'w', default: false },
    help: { type: 'boolean', short: 'h', default: false },
    version: { type: 'boolean', short: 'v', default: false },
    quiet: { type: 'boolean', short: 'q', default: false },
    mem: { type: 'boolean', default: false },
    iterations: { type: 'string', short: 'i', default: '10' },
    'list-gpus': { type: 'boolean', short: 'L', default: false },
    gpu: { type: 'string', short: 'g', default: '-1' },
    'lod-select': { type: 'string', short: 'O', default: '' },
    'viewer-settings': { type: 'string', short: 'E', default: '' },
    'lod-chunk-count': { type: 'string', short: 'C', default: '512' },
    'lod-chunk-extent': { type: 'string', short: 'X', default: '16' },
    unbundled: { type: 'boolean', short: 'U', default: false },
    'voxel-params': { type: 'string', default: '' },
    'voxel-external-fill': { type: 'string' },
    'voxel-carve-interior': { type: 'string' },
    'seed-pos': { type: 'string', default: '' },
    'collision-mesh': { type: 'boolean', short: 'K', default: false },
    'mesh-simplify-error': { type: 'string', default: '' },

    // per-file options
    translate: { type: 'string', short: 't', multiple: true },
    rotate: { type: 'string', short: 'r', multiple: true },
    scale: { type: 'string', short: 's', multiple: true },
    'filter-nan': { type: 'boolean', short: 'N', multiple: true },
    'filter-value': { type: 'string', short: 'V', multiple: true },
    'filter-harmonics': { type: 'string', short: 'H', multiple: true },
    'filter-box': { type: 'string', short: 'B', multiple: true },
    'filter-sphere': { type: 'string', short: 'S', multiple: true },
    'decimate': { type: 'string', short: 'F', multiple: true },
    'filter-cluster': { type: 'string', short: 'D', multiple: true },
    'filter-floaters': { type: 'string', short: 'G', multiple: true },
    params: { type: 'string', short: 'p', multiple: true },
    lod: { type: 'string', short: 'l', multiple: true },
    summary: { type: 'boolean', short: 'm', multiple: true },
    'morton-order': { type: 'boolean', short: 'M', multiple: true }
} as const;

const stringOptionNames = new Set(Object.entries(cliOptionsConfig)
.filter(([, v]) => v.type === 'string')
.flatMap(([name, v]) => [`--${name}`, ...('short' in v ? [`-${v.short}`] : [])])
);

const optionalValueOptions = new Set([
    '--filter-cluster', '-D', '--filter-floaters', '-G',
    '--voxel-external-fill', '--voxel-carve-interior',
    '--voxel-params'
]);

const isNumericValue = (s: string) => /^-?\d[\d.,e+-]*$/.test(s);

/**
 * Normalize argv so that all string options use the `=` form (`--option=value`).
 * This prevents parseArgs from misinterpreting negative numeric values (e.g.
 * `-0.5,0,0`) as flags. Optional-value options (e.g. `--filter-cluster`,
 * `--voxel-external-fill`) get an empty `=` when no value is provided.
 *
 * @param args - Raw command-line arguments (process.argv.slice(2)).
 * @returns Normalized argument array.
 */
const normalizeArgv = (args: string[]): string[] => {
    const result: string[] = [];
    for (let i = 0; i < args.length; i++) {
        const arg = args[i];
        const next = args[i + 1];
        if (optionalValueOptions.has(arg)) {
            if (next !== undefined && isNumericValue(next)) {
                result.push(`${arg}=${next}`);
                i++;
            } else {
                result.push(`${arg}=`);
            }
        } else if (stringOptionNames.has(arg) && next !== undefined) {
            result.push(`${arg}=${next}`);
            i++;
        } else {
            result.push(arg);
        }
    }
    return result;
};

const parseArguments = async () => {
    const { values: v, tokens } = parseArgs({
        args: normalizeArgv(process.argv.slice(2)),
        tokens: true,
        strict: true,
        allowPositionals: true,
        allowNegative: true,
        options: cliOptionsConfig
    });

    const parseNumber = (value: string, min?: number): number => {
        const result = Number(value);
        if (!Number.isFinite(result)) {
            throw new Error(`Invalid number value: ${value}`);
        }
        if (min !== undefined && result < min) {
            throw new Error(`Value must be >= ${min}, got ${value}`);
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

    const parseVec = (value: string, count: number): number[] => {
        const parts = value.split(',').map(p => parseNumber(p));
        if (parts.length !== count) {
            throw new Error(`Expected ${count} comma-separated values, got ${parts.length}: ${value}`);
        }
        return parts;
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

    // Parse voxel processing options
    const voxelParamsStr = v['voxel-params'];
    const externalFillStr = v['voxel-external-fill'];
    const carveInteriorStr = v['voxel-carve-interior'];
    const seedPosStr = v['seed-pos'];

    let voxelResolution = 0.05;
    let opacityCutoff = 0.1;
    if (voxelParamsStr) {
        const parts = voxelParamsStr.split(',').map((p: string) => p.trim());
        if (parts.length >= 1 && parts[0] !== '') {
            voxelResolution = parseNumber(parts[0], 0);
        }
        if (parts.length >= 2) {
            opacityCutoff = parseNumber(parts[1], 0);
        }
    }

    let navExteriorRadius: number | undefined;
    if (externalFillStr !== undefined) {
        navExteriorRadius = externalFillStr ? parseNumber(externalFillStr, 0) : 1.6;
    }

    let navCapsule: { height: number; radius: number } | undefined;
    if (carveInteriorStr !== undefined) {
        if (carveInteriorStr) {
            const [height, radius] = parseVec(carveInteriorStr, 2);
            if (height < 0 || radius < 0) {
                throw new Error(`Invalid voxel-carve-interior value: ${carveInteriorStr}. Height and radius must be >= 0`);
            }
            navCapsule = { height, radius };
        } else {
            navCapsule = { height: 1.6, radius: 0.2 };
        }
    }
    let navSeed: { x: number; y: number; z: number };
    if (seedPosStr) {
        const [x, y, z] = parseVec(seedPosStr, 3);
        navSeed = { x, y, z };
    } else {
        navSeed = { x: 0, y: 0, z: 0 };
    }

    const meshSimplifyError = v['mesh-simplify-error'] ? parseNumber(v['mesh-simplify-error'], 0) : undefined;

    const options: CliOptions = {
        overwrite: v.overwrite,
        help: v.help,
        version: v.version,
        quiet: v.quiet,
        mem: v.mem,
        iterations: parseInteger(v.iterations),
        listGpus: v['list-gpus'],
        deviceIdx,
        lodSelect: v['lod-select'].split(',').filter(v => !!v).map(parseInteger),
        viewerSettingsJson: viewerSettingsPath && await readJsonFile(viewerSettingsPath),
        unbundled: v.unbundled,
        lodChunkCount: parseInteger(v['lod-chunk-count']),
        lodChunkExtent: parseInteger(v['lod-chunk-extent']),
        voxelResolution,
        opacityCutoff,
        navExteriorRadius,
        navCapsule,
        navSeed,
        collisionMesh: v['collision-mesh'],
        meshSimplifyError
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
                case 'translate': {
                    const [x, y, z] = parseVec(t.value, 3);
                    current.processActions.push({
                        kind: 'translate',
                        value: new Vec3(x, y, z)
                    });
                    break;
                }
                case 'rotate': {
                    const [x, y, z] = parseVec(t.value, 3);
                    current.processActions.push({
                        kind: 'rotate',
                        value: new Vec3(x, y, z)
                    });
                    break;
                }
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
                case 'morton-order':
                    current.processActions.push({
                        kind: 'mortonOrder'
                    });
                    break;
                case 'decimate': {
                    const value = t.value.trim();
                    let count: number | null = null;
                    let percent: number | null = null;

                    if (value.endsWith('%')) {
                        // Percentage mode
                        percent = parseNumber(value.slice(0, -1));
                        if (percent < 0 || percent > 100) {
                            throw new Error(`Invalid decimate percentage: ${value}. Must be between 0% and 100%.`);
                        }
                    } else {
                        // Count mode
                        count = parseInteger(value);
                        if (count < 0) {
                            throw new Error(`Invalid decimate count: ${value}. Must be a non-negative integer.`);
                        }
                    }

                    current.processActions.push({
                        kind: 'decimate',
                        count,
                        percent
                    });
                    break;
                }
                case 'filter-cluster': {
                    const fcAction: FilterCluster = { kind: 'filterCluster' };
                    if (t.value) {
                        const parts = t.value.split(',').map((p: string) => p.trim());
                        if (parts.length >= 1 && parts[0] !== '') {
                            fcAction.voxelResolution = parseNumber(parts[0]);
                        }
                        if (parts.length >= 2) {
                            fcAction.opacityCutoff = parseNumber(parts[1]);
                        }
                    }
                    if (navSeed) {
                        fcAction.seed = new Vec3(navSeed.x, navSeed.y, navSeed.z);
                    }
                    current.processActions.push(fcAction);
                    break;
                }
                case 'filter-floaters': {
                    const ffAction: FilterFloaters = { kind: 'filterFloaters' };
                    if (t.value) {
                        const parts = t.value.split(',').map((p: string) => p.trim());
                        if (parts.length >= 1 && parts[0] !== '') {
                            ffAction.voxelResolution = parseNumber(parts[0]);
                        }
                        if (parts.length >= 2) {
                            ffAction.opacityCutoff = parseNumber(parts[1]);
                        }
                        if (parts.length >= 3) {
                            ffAction.minContribution = parseNumber(parts[2]);
                        }
                    }
                    current.processActions.push(ffAction);
                    break;
                }
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
  • Use 'null' as output to discard file output.

SUPPORTED INPUTS
    .ply   .compressed.ply   .sog   meta.json   .ksplat   .splat   .spz   .mjs   .lcc

SUPPORTED OUTPUTS
    .ply   .compressed.ply   .sog   meta.json   lod-meta.json   .glb   .csv   .html   .voxel.json   null

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
                                              opacity, scale_*, f_dc_* use transformed values
                                              (linear opacity 0-1, linear scale, linear color 0-1).
                                              Append _raw for raw PLY values (e.g. opacity_raw).
    -F, --decimate         <n|n%>           Simplify to n Gaussians via progressive pairwise merging
                                              Use n% to keep a percentage of Gaussians
    -G, --filter-floaters  [size,op,min]    Remove Gaussians not contributing to any solid voxel.
                                              Evaluates each Gaussian at occupied voxel centers.
                                              Default: size=0.05, opacity=0.1, min=0.004 (1/255)
    -D, --filter-cluster   [res,op]         Keep only the connected cluster at --seed-pos.
                                              GPU-voxelizes at coarse resolution (res world units/voxel).
                                              Default: res=1.0, opacity=0.99
    -p, --params           <key=val,...>    Pass parameters to .mjs generator script
    -l, --lod              <n>              Specify the level of detail, n >= 0
    -m, --summary                           Print per-column statistics to stdout
    -M, --morton-order                      Reorder Gaussians by Morton code (Z-order curve)

GLOBAL OPTIONS
    -h, --help                              Show this help and exit
    -v, --version                           Show version and exit
    -q, --quiet                             Suppress non-error output
        --mem                               Show memory usage in progress output
    -w, --overwrite                         Overwrite output file if it exists
    -i, --iterations       <n>              Iterations for SOG SH compression (more=better). Default: 10
    -L, --list-gpus                         List available GPU adapters and exit
    -g, --gpu              <n|cpu>          Select device for SOG compression: GPU adapter index | 'cpu'
    -E, --viewer-settings  <settings.json>  HTML viewer settings JSON file
    -U, --unbundled                         Generate unbundled HTML viewer with separate files
    -O, --lod-select       <n,n,...>        Comma-separated LOD levels to read from LCC input
    -C, --lod-chunk-count  <n>              Approximate number of Gaussians per LOD chunk in K. Default: 512
    -X, --lod-chunk-extent <n>              Approximate size of an LOD chunk in world units (m). Default: 16
        --voxel-params     [size,opacity]    Voxel size and opacity threshold for .voxel.json. Default: 0.05,0.1
        --voxel-external-fill [size]        Fill exterior voxels by dilation from seed. Default size: 1.6
        --voxel-carve-interior [h,r]        Carve navigable interior using capsule flood fill from seed.
                                              Default: height=1.6, radius=0.2
        --seed-pos         <x,y,z>          Seed position for voxel processing and --filter-cluster. Default: 0,0,0
    -K, --collision-mesh                    Generate collision mesh (.collision.glb) with voxel output
        --mesh-simplify-error <n>           Max geometric error for collision mesh simplification as a fraction of voxelResolution. Default: 0.08

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

    # Generate voxel data
    splat-transform input.ply output.voxel.json

    # Generate voxel data with collision mesh
    splat-transform -K input.ply output.voxel.json

    # Generate voxel data with custom resolution and opacity threshold
    splat-transform --voxel-params 0.1,0.3 input.ply output.voxel.json

    # Generate voxel data with exterior fill and interior carve
    splat-transform --voxel-external-fill --voxel-carve-interior input.ply output.voxel.json

    # Generate voxel data with custom seed position and carve parameters
    splat-transform --seed-pos 1,0,0 --voxel-carve-interior 2.0,0.3 input.ply output.voxel.json

    # Print statistical summary, then write output
    splat-transform bunny.ply --summary output.ply

    # Print summary without writing a file (discard output)
    splat-transform bunny.ply -m null
`;

const main = async () => {
    const startTime = hrtime();

    // read args
    const { files, options } = await parseArguments();

    type Timing = [number, number];

    // timing state for anonymous progress blocks
    const hrtimeDelta = (start: Timing, end: Timing) => (end[0] - start[0]) + (end[1] - start[1]) / 1e9;

    let start: Timing | null = null;

    const err = console.error.bind(console);
    const warn = console.warn.bind(console);

    const formatMem = options.mem ? () => {
        const m = process.memoryUsage();
        const mb = (n: number) => `${(n / (1024 * 1024)).toFixed(0)}MB`;
        return `  [rss: ${mb(m.rss)}, heap: ${mb(m.heapUsed)}, ab: ${mb(m.arrayBuffers)}]`;
    } : () => '';

    // inject Node.js-specific logger - logs go to stderr, data output goes to stdout
    logger.setLogger({
        log: err,
        warn: warn,
        error: err,
        debug: err,
        output: console.log.bind(console),
        onProgress: (node) => {
            if (node.stepName) {
                err(`[${node.step}/${node.totalSteps}] ${node.stepName}${formatMem()}`);
            } else if (node.step === 0) {
                start = hrtime();
            } else {
                const displaySteps = 10;
                const curr = Math.round(displaySteps * node.step / node.totalSteps);
                const prev = Math.round(displaySteps * (node.step - 1) / node.totalSteps);
                if (curr > prev) process.stderr.write('#'.repeat(curr - prev));
                if (node.step === node.totalSteps) {
                    process.stderr.write(` (${hrtimeDelta(start, hrtime()).toFixed(3)}s)${formatMem()}\n`);
                }
            }
        }
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

    // Check for null output (discard file writing)
    const isNullOutput = outputArg.filename.toLowerCase() === 'null';

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
                    join(outputDir, 'settings.json'),
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
        // Create device creator function with caching (needed for processDataTable + writeFile)
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

        const processOptions = deviceCreator ? { createDevice: deviceCreator } : undefined;

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

                const isEnv = dataTable.hasColumn('lod') && dataTable.getColumnByName('lod').data.every(v => v === -1);
                if (!isEnv) {
                    dataTables[i] = await processDataTable(dataTable, inputArg.processActions, processOptions);
                }
            }

            return dataTables;
        }))).flat(1).filter(dataTable => dataTable !== null);

        // special-case the environment dataTable
        const envDataTables = inputDataTables.filter(dt => dt.hasColumn('lod') && dt.getColumnByName('lod').data.every(v => v === -1));
        const nonEnvDataTables = inputDataTables.filter(dt => !dt.hasColumn('lod') || dt.getColumnByName('lod').data.some(v => v !== -1));

        // combine inputs into a single output dataTable
        const dataTable = nonEnvDataTables.length > 0 && await processDataTable(
            combine(nonEnvDataTables),
            outputArg.processActions,
            processOptions
        );

        if (!dataTable || dataTable.numRows === 0) {
            throw new Error('No Gaussians to write');
        }

        const envDataTable = envDataTables.length > 0 && await processDataTable(
            combine(envDataTables),
            outputArg.processActions,
            processOptions
        );

        logger.log(`Total gaussians loaded: ${dataTable.numRows}`);

        // Skip file writing for null output
        if (!isNullOutput) {
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
