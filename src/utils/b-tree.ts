import { DataTable, TypedArray } from '../data-table';

// partition idx indices around the k-th largest element
const quickselect = (data: TypedArray, idx: Uint32Array, k: number): number => {
    const valAt = (p: number) => data[idx[p]];
    const swap = (i: number, j: number) => {
        const t = idx[i];
        idx[i] = idx[j];
        idx[j] = t;
    };

    const n = idx.length;
    let l = 0;
    let r = n - 1;

    for (; ;) {
        if (r <= l + 1) {
            if (r === l + 1 && valAt(r) < valAt(l)) swap(l, r);
            return idx[k];
        }

        // Median-of-three pivot selection (using values via idx)
        const mid = (l + r) >>> 1;
        swap(mid, l + 1);
        if (valAt(l) > valAt(r)) swap(l, r);
        if (valAt(l + 1) > valAt(r)) swap(l + 1, r);
        if (valAt(l) > valAt(l + 1)) swap(l, l + 1);

        let i = l + 1;
        let j = r;
        const pivotIdxVal = valAt(l + 1);
        const pivotIdx = idx[l + 1];

        // Partition around pivot
        for (; ;) {
            do {
                i++;
            } while (i <= r && valAt(i) < pivotIdxVal);
            do {
                j--;
            } while (j >= l && valAt(j) > pivotIdxVal);
            if (j < i) break;
            swap(i, j);
        }

        // Place pivot in its final position
        idx[l + 1] = idx[j];
        idx[j] = pivotIdx;

        // Narrow to the side containing k
        if (j >= k) r = j - 1;
        if (j <= k) l = i;
    }
};

class Aabb {
    min: number[];
    max: number[];

    constructor(min: number[] = [], max: number[] = []) {
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
}

interface BTreeNode {
    count: number;
    aabb: Aabb;
    indices?: Uint32Array;       // only for leaf nodes
    left?: BTreeNode;
    right?: BTreeNode;
}

class BTree {
    centroids: DataTable;
    root: BTreeNode;

    constructor(centroids: DataTable) {
        const recurse = (indices: Uint32Array): BTreeNode => {
            const aabb = new Aabb().fromCentroids(centroids, indices);

            if (indices.length <= 256) {
                return {
                    count: indices.length,
                    aabb,
                    indices
                };
            }

            const col = aabb.largestAxis();
            const values = centroids.columns[col].data;
            const mid = indices.length >>> 1;

            quickselect(values, indices, mid);

            const left = recurse(indices.subarray(0, mid));
            const right = recurse(indices.subarray(mid));

            return {
                count: left.count + right.count,
                aabb,
                left,
                right
            };
        };

        const { numRows } = centroids;
        const indices = new Uint32Array(numRows);
        for (let i = 0; i < numRows; ++i) {
            indices[i] = i;
        }

        this.centroids = centroids;
        this.root = recurse(indices);
    }
}

export { BTreeNode, BTree };
