import { DataTable } from '../data-table';

class Aabb {
    min: number[];
    max: number[];

    constructor(min: number[]=[], max: number[]=[]) {
        this.min = min;
        this.max = max;
    }

    largestAxis(): number {
        const { min, max } = this;
        const { length } = min;
        let result = -1;
        let l = -Infinity;
        for (let i = 0; i < length; ++i) {
            const e = max[i] - min[i];
            if (e > l) {
                l = e;
                result = i;
            }
        }
        return result;
    }

    largestDim(): number {
        const a = this.largestAxis();
        return this.max[a] - this.min[a];
    }

    fromCentroids(centroids: DataTable, indices: Uint32Array) {
        const { columns, numColumns } = centroids;
        const { min, max } = this;
        for (let j = 0; j < numColumns; j++) {
            const data = columns[j].data;
            let m = Infinity;
            let M = -Infinity;
            for (let i = 0; i < indices.length; i++) {
                const v = data[indices[i]];
                m = v < m ? v : m;
                M = v > M ? v : M;
            }
            min[j] = m;
            max[j] = M;
        }
        return this;
    }
};

interface BTreeNode {
    count: number;              // number of nodes including self
    aabb?: Aabb;
    index?: number;
    left?: BTreeNode;
    right?: BTreeNode;
}

class BTree {
    centroids: DataTable;
    root: BTreeNode;

    constructor(centroids: DataTable) {
        const recurse = (indices: Uint32Array): BTreeNode => {
            if (indices.length === 1) {
                return {
                    count: 1,
                    index: indices[0]
                };
            }

            const aabb = new Aabb().fromCentroids(centroids, indices);

            const col = aabb.largestAxis();
            const values = centroids.columns[col].data;
            indices.sort((a, b) => values[a] - values[b]);

            const mid = indices.length >> 1;
            const left = recurse(indices.subarray(0, mid));
            const right = recurse(indices.subarray(mid));

            return {
                count: 1 + left.count + right.count,
                aabb,
                left,
                right
            };
        }

        const indices = new Uint32Array(centroids.numRows);
        indices.forEach((v, i) => {
            indices[i] = i;
        });
        this.centroids = centroids;
        this.root = recurse(indices);
    }
}

export { BTreeNode, BTree };
