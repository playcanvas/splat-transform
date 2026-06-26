import { basename, dirname, resolve } from 'pathe';
import { BoundingBox, Mat4, Quat, Vec3 } from 'playcanvas';

import { logWrittenFile } from './utils';
import { writeSogSource } from './write-sog.js';
import { dataTableToChunkSource } from '../compat/data-table';
import { Column, DataTable, sortMortonOrder, convertToSpace } from '../data-table';
import { type FileSystem } from '../io/write';
import { permuteSource } from '../ops';
import { type ChunkDataPool, type ChunkSource, createChunkDataPool } from '../source';
import { BTreeNode, BTree } from '../spatial';
import type { DeviceCreator } from '../types';
import { logger, Transform } from '../utils';
import { version } from '../version';

type Aabb = {
    min: number[],
    max: number[]
};

type MetaLod = {
    file: number;
    offset: number;
    count: number;
};

type MetaNode = {
    bound: Aabb;
    children?: MetaNode[];
    lods?: { [key: number]: MetaLod };
};

type LodMeta = {
    version: number;
    asset: {
        generator: string;
    };
    count: number;
    counts: number[];
    lodLevels: number;
    environment?: string;
    filenames: string[];
    tree: MetaNode;
};

const boundUnion = (result: Aabb, a: Aabb, b: Aabb) => {
    const am = a.min;
    const aM = a.max;
    const bm = b.min;
    const bM = b.max;
    const rm = result.min;
    const rM = result.max;

    rm[0] = Math.min(am[0], bm[0]);
    rm[1] = Math.min(am[1], bm[1]);
    rm[2] = Math.min(am[2], bm[2]);
    rM[0] = Math.max(aM[0], bM[0]);
    rM[1] = Math.max(aM[1], bM[1]);
    rM[2] = Math.max(aM[2], bM[2]);
};

/**
 * The only per-gaussian column held resident for the partition: positions.
 * Everything else (rotation/scale for bounds, color/SH for encoding) is gathered
 * from the source on demand, so resident scales as ~12 B/gaussian regardless of
 * SH degree — the point of the streaming LOD writer for very large scenes.
 */
type SlimColumns = {
    x: Float32Array; y: Float32Array; z: Float32Array;
};

// Expand a batch of gathered (position, rotation, scale) records into ellipsoid
// AABBs and fold them into `min`/`max`. Pulled out of `calcBound` so the bounds
// pass can run over gathered batches; the math mirrors the legacy per-gaussian
// path exactly (quaternion order (rot_1, rot_2, rot_3, rot_0); scale = exp).
const accumulateBound = (
    min: number[], max: number[],
    pos: Float32Array, rot: Float32Array, scale: Float32Array, count: number
): void => {
    const p = new Vec3();
    const r = new Quat();
    const s = new Vec3();
    const mat4 = new Mat4();
    const a = new BoundingBox();
    const b = new BoundingBox();

    a.center.set(0, 0, 0);

    for (let i = 0; i < count; i++) {
        p.set(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
        r.set(rot[i * 4 + 1], rot[i * 4 + 2], rot[i * 4 + 3], rot[i * 4 + 0]).normalize();
        s.set(Math.exp(scale[i * 3]), Math.exp(scale[i * 3 + 1]), Math.exp(scale[i * 3 + 2]));
        mat4.setTRS(p, r, Vec3.ONE);

        a.halfExtents.set(s.x, s.y, s.z);
        b.setFromTransformedAabb(a, mat4);

        const m = b.getMin();
        const M = b.getMax();

        if (!isFinite(m.x) || !isFinite(m.y) || !isFinite(m.z) || !isFinite(M.x) || !isFinite(M.y) || !isFinite(M.z)) {
            logger.warn(`skipping invalid bounding box: min=(${m.x}, ${m.y}, ${m.z}) max=(${M.x}, ${M.y}, ${M.z})`);
            continue;
        }

        min[0] = Math.min(min[0], m.x);
        min[1] = Math.min(min[1], m.y);
        min[2] = Math.min(min[2], m.z);
        max[0] = Math.max(max[0], M.x);
        max[1] = Math.max(max[1], M.y);
        max[2] = Math.max(max[2], M.z);
    }
};

// Per-leaf ellipsoid AABB. Positions are resident, but rotation/scale are
// gathered from the source by index (batched by chunk) so the geometric layer is
// never wholly resident — the bounds-pass analog of the per-unit heavy gather.
const calcBound = async (source: ChunkSource, pool: ChunkDataPool, idx: Uint32Array): Promise<Aabb> => {
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];

    const batch = pool.chunkSize;
    const { layouts } = source.meta;

    for (let off = 0; off < idx.length; off += batch) {
        const count = Math.min(batch, idx.length - off);
        const pos = pool.acquire('position', layouts.position!, count);
        const geo = pool.acquire('geometric', layouts.geometric!, count);
        await source.readRows!({ indices: idx, indexOffset: off, count, position: pos, geometric: geo });
        accumulateBound(
            min, max,
            pos.field('position') as Float32Array,
            geo.field('rotation') as Float32Array,
            geo.field('scale') as Float32Array,
            count
        );
        pos.release();
        geo.release();
    }

    return { min, max };
};

// Group the gaussian indices under `parent` by their LOD level. Two passes
// (count, then fill) so each LOD's indices land in a tight `Uint32Array` rather
// than a `number[]` — the indices are the dominant retained bookkeeping for a
// large scene, so keeping them off the V8 heap (4 B each, no GC pressure) is
// what lets LOD export scale to hundreds of millions of splats.
const binIndices = (parent: BTreeNode, lod: Float32Array): Map<number, Uint32Array> => {
    const counts = new Map<number, number>();
    const tally = (node: BTreeNode) => {
        if (node.indices) {
            for (let i = 0; i < node.indices.length; ++i) {
                const lodValue = lod[node.indices[i]];
                counts.set(lodValue, (counts.get(lodValue) ?? 0) + 1);
            }
        } else {
            if (node.left) tally(node.left);
            if (node.right) tally(node.right);
        }
    };
    tally(parent);

    const result = new Map<number, Uint32Array>();
    const offset = new Map<number, number>();
    for (const [lodValue, count] of counts) {
        result.set(lodValue, new Uint32Array(count));
        offset.set(lodValue, 0);
    }

    const fill = (node: BTreeNode) => {
        if (node.indices) {
            for (let i = 0; i < node.indices.length; ++i) {
                const v = node.indices[i];
                const lodValue = lod[v];
                const o = offset.get(lodValue)!;
                result.get(lodValue)![o] = v;
                offset.set(lodValue, o + 1);
            }
        } else {
            if (node.left) fill(node.left);
            if (node.right) fill(node.right);
        }
    };
    fill(parent);

    return result;
};

/**
 * Read positions out of a source into flat per-gaussian arrays — one sequential
 * pass. Nothing else is materialized here (for a fixed-stride file source the
 * rotation/scale/color/SH bytes are read-and-discarded); rotation/scale are
 * gathered per leaf for bounds, and the heavy layers per unit at encode time.
 *
 * @param source - The single-LOD, PLY-space scene source.
 * @param pool - Pool for the temporary per-chunk read buffers.
 * @returns The flat position columns, indexed by gaussian.
 */
const extractSlim = async (source: ChunkSource, pool: ChunkDataPool): Promise<SlimColumns> => {
    const { meta } = source;
    const N = meta.numGaussians;
    const cols: SlimColumns = {
        x: new Float32Array(N),
        y: new Float32Array(N),
        z: new Float32Array(N)
    };

    const { chunkSize } = meta;
    const numChunks = meta.numChunks[0];
    for (let k = 0; k < numChunks; k++) {
        const count = Math.min(chunkSize, N - k * chunkSize);
        const pos = pool.acquire('position', meta.layouts.position!, count);
        await source.read({ chunkIndex: k, lod: 0, position: pos });

        const p = pos.field('position') as Float32Array;  // count × 3
        const base = k * chunkSize;
        for (let i = 0; i < count; i++) {
            const di = base + i;
            cols.x[di] = p[i * 3]; cols.y[di] = p[i * 3 + 1]; cols.z[di] = p[i * 3 + 2];
        }
        pos.release();
    }
    return cols;
};

type WriteLodSourceOptions = {
    filename: string;
    mainSource: ChunkSource;
    envSource: ChunkSource | null;
    /** Per-gaussian LOD level (from `--lod` tags), indexed as `mainSource` rows. */
    lod: Float32Array;
    iterations: number;
    createDevice?: DeviceCreator;
    chunkCount: number;
    chunkExtent: number;
};

/**
 * Writes Gaussian splat data to multi-LOD format with spatial chunking, reading
 * from resident chunk sources already in PLY space.
 *
 * Creates a hierarchical structure with multiple LOD levels, each stored in
 * separate SOG files, plus a binary-tree spatial index for view-dependent
 * loading. The partition / per-leaf bounds / lod binning run over flat analysis
 * columns extracted from `mainSource`; each unit's gaussians are gathered lazily
 * from `mainSource` via {@link permuteSource} and encoded chunk-native.
 *
 * @param options - Options including filename, sources, and chunking parameters.
 * @param fs - File system for writing output files.
 * @ignore
 */
const writeLodSource = async (options: WriteLodSourceOptions, fs: FileSystem) => {
    const { filename, mainSource, envSource, lod, iterations, createDevice, chunkCount, chunkExtent } = options;

    // Pool for slim extraction read buffers and the chunk-native SOG encodes.
    const pool = createChunkDataPool();

    const slim = await extractSlim(mainSource, pool);
    const hasEnv = !!envSource && envSource.meta.numGaussians > 0;

    const outputDir = dirname(filename);

    // ensure top-level output folder exists
    await fs.mkdir(outputDir);

    // construct a kd-tree based on centroids from all lods
    const centroidsTable = new DataTable([
        new Column('x', slim.x),
        new Column('y', slim.y),
        new Column('z', slim.z)
    ]);

    const bTree = new BTree(centroidsTable);

    // approximate number of gaussians we'll place into file units
    const binSize = chunkCount * 1024;
    const binDim = chunkExtent;

    // map of lod -> file units -> subunits (each subunit a tight Uint32Array of
    // gaussian indices). This is the bulk retained bookkeeping; Uint32Array keeps
    // it off the V8 heap at 4 B/gaussian.
    const lodFiles: Map<number, Uint32Array[][]> = new Map();
    const lodColumn = lod;
    const filenames: string[] = [];
    let lodLevels = 0;

    const build = async (node: BTreeNode): Promise<MetaNode> => {
        if (!node.indices && (node.count > binSize || (node.aabb && node.aabb.largestDim() > binDim))) {
            const children = [
                await build(node.left),
                await build(node.right)
            ];

            const bound = {
                min: [0, 0, 0],
                max: [0, 0, 0]
            };
            boundUnion(bound, children[0].bound, children[1].bound);

            return { bound, children };
        }

        const lods: { [key: number]: MetaLod } = { };
        const bins = binIndices(node, lodColumn);

        for (const [lodValue, indices] of bins) {
            if (!lodFiles.has(lodValue)) {
                lodFiles.set(lodValue, [[]]);
            }
            const fileList = lodFiles.get(lodValue);
            const fileIndex = fileList.length - 1;
            const lastFile = fileList[fileIndex];
            const fileSize = lastFile.reduce((acc, curr) => acc + curr.length, 0);

            const filename = `${lodValue}_${fileIndex}/meta.json`;
            if (filenames.indexOf(filename) === -1) {
                filenames.push(filename);
            }

            lods[lodValue] = {
                file: filenames.indexOf(filename),
                offset: fileSize,
                count: indices.length
            };

            lastFile.push(indices);

            if (fileSize + indices.length > binSize) {
                fileList.push([]);
            }

            lodLevels = Math.max(lodLevels, lodValue + 1);
        }

        // combine indices from all lods (as one Uint32Array) to bound over them
        let total = 0;
        for (const arr of bins.values()) total += arr.length;
        const allIndices = new Uint32Array(total);
        let o = 0;
        for (const arr of bins.values()) {
            allIndices.set(arr, o);
            o += arr.length;
        }

        const bound = await calcBound(mainSource, pool, allIndices);

        return { bound, lods };
    };

    const tree = await build(bTree.root);

    // count splats per lod level
    const counts = new Array(lodLevels).fill(0);
    for (const [lodValue, fileUnits] of lodFiles) {
        for (const fileUnit of fileUnits) {
            counts[lodValue] += fileUnit.reduce((acc, curr) => acc + curr.length, 0);
        }
    }

    const meta: LodMeta = {
        version: 1,
        asset: {
            generator: `splat-transform v${version}`
        },
        count: counts.reduce((acc, curr) => acc + curr, 0),
        counts,
        lodLevels,
        ...(hasEnv ? { environment: 'env/meta.json' } : {}),
        filenames,
        tree
    };

    // write the meta file with float precision quantization (approx. 32-bit float => ~7 significant digits)
    const replacer = (_key: string, value: any) => {
        if (typeof value === 'number') {
            if (!Number.isFinite(value)) return value;
            return Number.isInteger(value) ? value : +value.toPrecision(7);
        }
        return value;
    };

    const writingGroup = logger.group('Writing');

    // count the total number of sog units we'll write so the per-sog groups
    // can render as a numbered series
    let sogTotal = 0;
    if (hasEnv) sogTotal += 1;
    for (const [, fileUnits] of lodFiles) {
        for (const fu of fileUnits) {
            if (fu.length > 0) sogTotal += 1;
        }
    }

    let sogIndex = 0;

    // write the environment sog
    if (hasEnv) {
        sogIndex++;
        const envGroup = logger.group('env', { index: sogIndex, total: sogTotal });
        try {
            const envPathname = resolve(outputDir, 'env/meta.json');

            // ensure output folder exists before any files are written
            await fs.mkdir(dirname(envPathname));

            await writeSogSource(
                envSource!,
                pool,
                { filename: envPathname, bundle: false, iterations, createDevice, logging: 'flat' },
                fs
            );
        } finally {
            envGroup.end();
        }
    }

    // write lod-meta.json
    const metaJson = (new TextEncoder()).encode(JSON.stringify(meta, replacer));
    const writer = await fs.createWriter(filename);
    await writer.write(metaJson);
    await writer.close();
    logWrittenFile(basename(filename), writer.bytesWritten);

    // write file units
    for (const [lodValue, fileUnits] of lodFiles) {
        for (let i = 0; i < fileUnits.length; ++i) {
            const fileUnit = fileUnits[i];

            if (fileUnit.length === 0) {
                continue;
            }

            const groupName = `${lodValue}_${i}`;
            sogIndex++;
            const unitGroup = logger.group(groupName, { index: sogIndex, total: sogTotal });

            try {
                // ensure output folder exists before any files are written
                const pathname = resolve(outputDir, `${lodValue}_${i}/meta.json`);
                await fs.mkdir(dirname(pathname));

                // Morton-order each subunit and concatenate into the unit's
                // global row order.
                const totalIndices = fileUnit.reduce((acc, curr) => acc + curr.length, 0);
                const orderedIndices = new Uint32Array(totalIndices);
                for (let j = 0, offset = 0; j < fileUnit.length; ++j) {
                    orderedIndices.set(fileUnit[j], offset);
                    sortMortonOrder(centroidsTable, orderedIndices.subarray(offset, offset + fileUnit[j].length));
                    offset += fileUnit[j].length;
                }

                // Gather the ordered subset lazily from the scene source (no
                // per-unit copy) and encode via the chunk-native SOG writer. The
                // rows are already in write order, so pass an identity ordering
                // to skip the writer's own Morton pass.
                const unitSource = permuteSource(mainSource, orderedIndices);
                const identity = new Uint32Array(totalIndices);
                for (let j = 0; j < totalIndices; ++j) identity[j] = j;

                await writeSogSource(unitSource, pool, {
                    filename: pathname,
                    bundle: false,
                    iterations,
                    createDevice,
                    indices: identity,
                    logging: 'flat'
                }, fs);
            } finally {
                unitGroup.end();
            }
        }
    }

    writingGroup.end();
};

type WriteLodOptions = {
    filename: string;
    dataTable: DataTable;
    envDataTable: DataTable | null;
    iterations: number;
    createDevice?: DeviceCreator;
    chunkCount: number;
    chunkExtent: number;
};

/**
 * Writes Gaussian splat data to multi-LOD format with spatial chunking.
 *
 * DataTable-input adapter over {@link writeLodSource}: converts the input tables
 * to PLY space and repacks each into a resident `ChunkSource`, so the partition
 * then runs over chunks rather than named columns.
 *
 * @param options - Options including filename, data, and chunking parameters.
 * @param fs - File system for writing output files.
 * @ignore
 */
const writeLod = async (options: WriteLodOptions, fs: FileSystem) => {
    const { filename, iterations, createDevice, chunkCount, chunkExtent } = options;

    // Operate in PLY space so per-leaf bounds in tree.bound are in the same
    // coordinate frame as the SOG chunk data emitted by writeSog (which also
    // converts to Transform.PLY). Without this, view-dependent streaming
    // built on tree.bound picks the wrong chunks because the bounds are
    // 180°-Z-rotated relative to the splat positions inside them.
    // This intentionally mutates the input tables to avoid doubling peak
    // memory during LOD export. Callers should treat writeLod as consuming its
    // DataTable inputs.
    const dataTable = convertToSpace(options.dataTable, Transform.PLY, true);
    const envDataTable = options.envDataTable ? convertToSpace(options.envDataTable, Transform.PLY, true) : null;

    const lodColumn = dataTable.getColumnByName('lod');
    if (!lodColumn) {
        throw new Error('Missing lod assignment');
    }

    const mainSource = dataTableToChunkSource(dataTable);
    const envSource = (envDataTable && envDataTable.numRows > 0) ? dataTableToChunkSource(envDataTable) : null;

    await writeLodSource({
        filename,
        mainSource,
        envSource,
        lod: lodColumn.data as Float32Array,
        iterations,
        createDevice,
        chunkCount,
        chunkExtent
    }, fs);
};

export { writeLod, writeLodSource };
