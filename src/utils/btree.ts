import { DataTable } from '../data-table';

class Aabb {
    min: number[];
    max: number[];
    constructor(min: number[], max: number[]) {
        this.min = min;
        this.max = max;
    }
}

interface BTreeNode {
    index: number;
    left?: BTreeNode;
    right?: BTreeNode;
}

class BTree {
    centroids: DataTable;
    root: BTreeNode;

    constructor(centroids: DataTable) {
        const indices = new Uint32Array(centroids.numRows);
        indices.forEach((v, i) => {
            indices[i] = i;
        });
        this.centroids = centroids;

        const calcAabb = (): Aabb => {
            const { numColumns } = centroids;
            const bmin = centroids.columns.map(c => Infinity);
            const bmax = centroids.columns.map(c => -Infinity);
            for (let i = 0; i < centroids.numRows; i++) {
                for (let j = 0; j < numColumns; j++) {
                    const v = centroids.columns[j].data[i];
                    bmin[j] = Math.min(bmin[j], v);
                    bmax[j] = Math.max(bmax[j], v);
                }
            }
            return new Aabb(bmin, bmax);
        };

        const largestAxis = (aabb: Aabb): number => {
            const extents = aabb.max.map((max, i) => max - aabb.min[i]);
            return extents.indexOf(Math.max(...extents));
        };

        const splitAabb = (aabb: Aabb, col: number, value: number): { left: Aabb; right: Aabb } => {
            const left = new Aabb(
                aabb.min.slice(),
                aabb.max.slice()
            );
            const right = new Aabb(
                aabb.min.slice(),
                aabb.max.slice()
            );
            left.max[col] = value;
            right.min[col] = value;
            return { left, right };
        };

        const recurse = (indices: Uint32Array, aabb: Aabb): BTreeNode => {
            const col = largestAxis(aabb);
            const values = centroids.columns[col].data;
            indices.sort((a, b) => values[a] - values[b]);

            if (indices.length === 1) {
                return {
                    index: indices[0]
                };
            } else if (indices.length === 2) {
                return {
                    index: indices[0],
                    right: {
                        index: indices[1]
                    }
                };
            }

            const mid = indices.length >> 1;
            const aabbs = splitAabb(aabb, col, values[indices[mid]]);
            const left = recurse(indices.subarray(0, mid), aabbs.left);
            const right = recurse(indices.subarray(mid + 1), aabbs.right);

            return {
                index: indices[mid],
                left,
                right
            };
        }

        this.root = recurse(indices, calcAabb());
    }
}

export { BTree };
