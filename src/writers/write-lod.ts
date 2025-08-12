import { FileHandle, mkdir, open } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { BoundingBox, Mat4, Quat, Vec3 } from 'playcanvas';
import { TypedArray, DataTable } from '../data-table';

import { KdTreeNode, KdTree } from '../utils/kd-tree';
import { generateOrdering } from '../ordering';

import { writeSogs } from './write-sogs.js';

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

type Meta = {
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

        min[0] = Math.min(min[0], m.x);
        min[1] = Math.min(min[1], m.y);
        min[2] = Math.min(min[2], m.z);
        max[0] = Math.max(max[0], M.x);
        max[1] = Math.max(max[1], M.y);
        max[2] = Math.max(max[2], M.z);
    }

    return { min, max };
};

const binIndices = (parent: KdTreeNode, lod: TypedArray) => {
    const result = new Map<number, number[]>();

    // we've reached a leaf node, gather indices
    const recurse = (node: KdTreeNode) => {
        const lodValue = lod[node.index];

        if (!result.has(lodValue)) {
            result.set(lodValue, [node.index]);
        } else {
            result.get(lodValue).push(node.index);
        }

        if (node.left) {
            recurse(node.left);
        }
        if (node.right) {
            recurse(node.right);
        }
    };

    recurse(parent);

    return result;
};

const writeLod = async (fileHandle: FileHandle, dataTable: DataTable, outputFilename: string, shIterations = 10, shMethod: 'cpu' | 'gpu') => {
    // construct a kd-tree based on centroids from all lods
    const centroidsTable = new DataTable([
        dataTable.getColumnByName('x'),
        dataTable.getColumnByName('y'),
        dataTable.getColumnByName('z')
    ]);

    const kdTree = new KdTree(centroidsTable);

    // map of lod -> fileUnit[]
    const lodFiles: Map<number, number[][][]> = new Map();
    const lodColumn = dataTable.getColumnByName('lod').data;
    const groupSize = 128 * 1024;
    const filenames: string[] = [];

    const build = (node: KdTreeNode): MetaNode => {
        if (node.count > groupSize) {
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

            lods[lodValue] = {
                file: filenames.length,
                offset: fileSize,
                count: indices.length
            };

            filenames.push(`${lodValue}_${fileIndex}/meta.json`);

            lastFile.push(indices);

            if (fileSize + indices.length > groupSize) {
                fileList.push([]);
            }
        }

        // combine indices from all lods so we can calcuate bound over them
        let allIndices: number[] = []
        for (const [lodValue, indices] of bins) {
            allIndices = allIndices.concat(indices);
        }

        const bound = calcBound(dataTable, allIndices);

        return { bound, lods };
    };

    const tree = build(kdTree.root);
    const meta: Meta = {
        filenames,
        tree
    };

    // write the meta file
    await fileHandle.write((new TextEncoder()).encode(JSON.stringify(meta, null, 4)));

    // write file units
    for (const [lodValue, fileUnits] of lodFiles) {
        for (let i = 0; i < fileUnits.length; ++i) {
            const fileUnit = fileUnits[i];

            if (fileUnit.length === 0) {
                continue;
            }

            // generate an ordering for each subunit and append it to the unit's indices
            const totalIndices = fileUnit.reduce((acc, curr) => acc + curr.length, 0);
            const indices = new Uint32Array(totalIndices);
            for (let j = 0, offset = 0; j < fileUnit.length; ++j) {
                indices.set(fileUnit[j], offset);
                generateOrdering(dataTable, indices.subarray(offset, offset + fileUnit[j].length));
                offset += fileUnit[j].length;
            }

            // construct a new table from the ordered data
            const unitDataTable = dataTable.permutedRows(indices);

            // reset indices since we've generated ordering on the individual subunits
            for (let j = 0; j < indices.length; ++j) {
                indices[j] = j;
            }

            // write file unit to sog
            const pathname = resolve(dirname(outputFilename), `lod${lodValue}_${i}/meta.json`);

            // ensure output folder exists
            await mkdir(dirname(pathname), { recursive: true });

            const outputFile = await open(pathname, 'w');

            console.log(`writing ${pathname}...`);

            await writeSogs(outputFile, unitDataTable, pathname, shIterations, shMethod, indices);

            await outputFile.close();
        }
    }
};

export { writeLod };
