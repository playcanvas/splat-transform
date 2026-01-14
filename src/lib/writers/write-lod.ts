import { dirname, resolve } from 'node:path';

import { BoundingBox, Mat4, Quat, Vec3 } from 'playcanvas';

import { writeSog, type DeviceCreator } from './write-sog.js';
import { TypedArray, DataTable } from '../data-table/data-table';
import { sortMortonOrder } from '../data-table/morton-order';
import { type FileSystem } from '../io/write';
import { BTreeNode, BTree } from '../spatial/b-tree';
import { logger } from '../utils/logger';


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
    lodLevels: number,
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

const calcBound = (dataTable: DataTable, indices: number[]): Aabb => {
    const x = dataTable.getColumnByName('x').data;
    const y = dataTable.getColumnByName('y').data;
    const z = dataTable.getColumnByName('z').data;
    const rx = dataTable.getColumnByName('rot_1').data;
    const ry = dataTable.getColumnByName('rot_2').data;
    const rz = dataTable.getColumnByName('rot_3').data;
    const rw = dataTable.getColumnByName('rot_0').data;
    const sx = dataTable.getColumnByName('scale_0').data;
    const sy = dataTable.getColumnByName('scale_1').data;
    const sz = dataTable.getColumnByName('scale_2').data;

    const p = new Vec3();
    const r = new Quat();
    const s = new Vec3();
    const mat4 = new Mat4();

    const a = new BoundingBox();
    const b = new BoundingBox();

    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];

    a.center.set(0, 0, 0);

    for (const index of indices) {
        p.set(x[index], y[index], z[index]);
        r.set(rx[index], ry[index], rz[index], rw[index]).normalize();
        s.set(Math.exp(sx[index]), Math.exp(sy[index]), Math.exp(sz[index]));
        mat4.setTRS(p, r, Vec3.ONE);

        a.halfExtents.set(s.x, s.y, s.z);
        b.setFromTransformedAabb(a, mat4);

        const m = b.getMin();
        const M = b.getMax();

        if (!isFinite(m.x) || !isFinite(m.y) || !isFinite(m.z) || !isFinite(M.x) || !isFinite(M.y) || !isFinite(M.z)) {
            logger.warn('Skipping invalid bounding box:', { m, M, index });
            continue;
        }

        min[0] = Math.min(min[0], m.x);
        min[1] = Math.min(min[1], m.y);
        min[2] = Math.min(min[2], m.z);
        max[0] = Math.max(max[0], M.x);
        max[1] = Math.max(max[1], M.y);
        max[2] = Math.max(max[2], M.z);
    }

    return { min, max };
};

const binIndices = (parent: BTreeNode, lod: TypedArray) => {
    const result = new Map<number, number[]>();

    // we've reached a leaf node, gather indices
    const recurse = (node: BTreeNode) => {
        if (node.indices) {

            for (let i = 0; i < node.indices.length; ++i) {
                const v = node.indices[i];
                const lodValue = lod[v];

                if (!result.has(lodValue)) {
                    result.set(lodValue, [v]);
                } else {
                    result.get(lodValue).push(v);
                }
            }
        } else {
            if (node.left) {
                recurse(node.left);
            }
            if (node.right) {
                recurse(node.right);
            }
        }
    };

    recurse(parent);

    return result;
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

const writeLod = async (options: WriteLodOptions, fs: FileSystem) => {
    const { filename, dataTable, envDataTable, iterations, createDevice, chunkCount, chunkExtent } = options;

    const outputDir = dirname(filename);

    // ensure top-level output folder exists
    await fs.mkdir(outputDir);

    // write the environment sog
    if (envDataTable?.numRows > 0) {
        const pathname = resolve(outputDir, 'env/meta.json');

        // ensure output folder exists before any files are written
        await fs.mkdir(dirname(pathname));

        logger.log(`writing ${pathname}...`);

        await writeSog({
            filename: pathname,
            dataTable: envDataTable,
            bundle: false,
            iterations,
            createDevice
        }, fs);
    }

    // construct a kd-tree based on centroids from all lods
    const centroidsTable = new DataTable([
        dataTable.getColumnByName('x'),
        dataTable.getColumnByName('y'),
        dataTable.getColumnByName('z')
    ]);

    const bTree = new BTree(centroidsTable);

    // approximate number of gaussians we'll place into file units
    const binSize = chunkCount * 1024;
    const binDim = chunkExtent;

    // map of lod -> fileBin[]
    // fileBin: number[][]
    const lodFiles: Map<number, number[][][]> = new Map();
    const lodColumn = dataTable.getColumnByName('lod')?.data;
    const filenames: string[] = [];
    let lodLevels = 0;

    if (!lodColumn) {
        throw new Error('Missing lod assignment');
    }

    const build = (node: BTreeNode): MetaNode => {
        if (!node.indices && (node.count > binSize || (node.aabb && node.aabb.largestDim() > binDim))) {
            const children = [
                build(node.left),
                build(node.right)
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

        // combine indices from all lods so we can calcuate bound over them
        const allIndices: number[] = Array.from(bins.values()).flat();

        const bound = calcBound(dataTable, allIndices);

        return { bound, lods };
    };

    const tree = build(bTree.root);
    const meta: LodMeta = {
        lodLevels,
        environment: (envDataTable?.numRows > 0) ? 'env/meta.json' : null,
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

    // write lod-meta.json
    const writer = await fs.createWriter(filename);
    writer.write((new TextEncoder()).encode(JSON.stringify(meta, replacer)));
    await writer.close();

    // write file units
    for (const [lodValue, fileUnits] of lodFiles) {
        for (let i = 0; i < fileUnits.length; ++i) {
            const fileUnit = fileUnits[i];

            if (fileUnit.length === 0) {
                continue;
            }

            // ensure output folder exists before any files are written
            const pathname = resolve(outputDir, `${lodValue}_${i}/meta.json`);
            await fs.mkdir(dirname(pathname));

            // generate an ordering for each subunit and append it to the unit's indices
            const totalIndices = fileUnit.reduce((acc, curr) => acc + curr.length, 0);
            const indices = new Uint32Array(totalIndices);
            for (let j = 0, offset = 0; j < fileUnit.length; ++j) {
                indices.set(fileUnit[j], offset);
                sortMortonOrder(dataTable, indices.subarray(offset, offset + fileUnit[j].length));
                offset += fileUnit[j].length;
            }

            // construct a new table from the ordered data
            const unitDataTable = dataTable.permuteRows(indices);

            // reset indices since we've generated ordering on the individual subunits
            for (let j = 0; j < indices.length; ++j) {
                indices[j] = j;
            }

            // write file unit to sog
            logger.log(`writing ${pathname}...`);

            await writeSog({
                filename: pathname,
                dataTable: unitDataTable,
                indices,
                bundle: false,
                iterations,
                createDevice
            }, fs);
        }
    }
};

export { writeLod };
