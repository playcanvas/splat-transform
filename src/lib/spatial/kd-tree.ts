import { DataTable } from '../data-table/data-table';

interface KdTreeNode {
    index: number;
    count: number;          // self + children indices
    left?: KdTreeNode;
    right?: KdTreeNode;
}

class KdTree {
    centroids: DataTable;
    root: KdTreeNode;

    constructor(centroids: DataTable) {
        const build = (indices: Uint32Array, depth: number): KdTreeNode => {
            const { centroids } = this;
            const values = centroids.columns[depth % centroids.numColumns].data;
            indices.sort((a, b) => values[a] - values[b]);

            if (indices.length === 1) {
                return {
                    index: indices[0],
                    count: 1
                };
            } else if (indices.length === 2) {
                return {
                    index: indices[0],
                    count: 2,
                    right: {
                        index: indices[1],
                        count: 1
                    }
                };
            }

            const mid = indices.length >> 1;
            const left = build(indices.subarray(0, mid), depth + 1);
            const right = build(indices.subarray(mid + 1), depth + 1);

            return {
                index: indices[mid],
                count: 1 + left.count + right.count,
                left,
                right
            };
        };

        const indices = new Uint32Array(centroids.numRows);
        for (let i = 0; i < indices.length; ++i) {
            indices[i] = i;
        }

        this.centroids = centroids;
        this.root = build(indices, 0);
    }

    findNearest(point: Float32Array, filterFunc?: (index: number) => boolean) {
        const { centroids } = this;
        const { numColumns } = centroids;

        const calcDistance = (index: number) => {
            let l = 0;
            for (let i = 0; i < numColumns; ++i) {
                const v = centroids.columns[i].data[index] - point[i];
                l += v * v;
            }
            return l;
        };

        let mind = Infinity;
        let mini = -1;
        let cnt = 0;

        const recurse = (node: KdTreeNode, depth: number) => {
            const axis = depth % numColumns;
            const distance = point[axis] - centroids.columns[axis].data[node.index];
            const next = (distance > 0) ? node.right : node.left;

            cnt++;

            if (next) {
                recurse(next, depth + 1);
            }

            // check index
            if (!filterFunc || filterFunc(node.index)) {
                const thisd = calcDistance(node.index);
                if (thisd < mind) {
                    mind = thisd;
                    mini = node.index;
                }
            }

            // check the other side
            if (distance * distance < mind) {
                const other = next === node.right ? node.left : node.right;
                if (other) {
                    recurse(other, depth + 1);
                }
            }
        };

        recurse(this.root, 0);

        return { index: mini, distanceSqr: mind, cnt };
    }
}

export { KdTreeNode, KdTree };
