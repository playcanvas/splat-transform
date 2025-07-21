import { DataTable } from '../data-table';

interface KdTreeNode {
    index: number;
    left?: KdTreeNode;
    right?: KdTreeNode;
}

class KdTree {
    centroids: DataTable;
    root: KdTreeNode;

    constructor(centroids: DataTable) {
        const indices = new Uint32Array(centroids.numRows);
        indices.forEach((v, i) => {
            indices[i] = i;
        });
        this.centroids = centroids;
        this.root = this.build(indices, 0);
    }

    // construct a flat buffer representation of the tree
    flatten() {
        const { totalNodes } = this;

        const result = new Uint32Array(totalNodes * 3);
        let index = 0;

        const recurse = (node: KdTreeNode) => {
            const i = index++;
            result[i * 3] = node.index;
            result[i * 3 + 1] = node.left ? recurse(node.left) : 0;
            result[i * 3 + 2] = node.right ? recurse(node.right) : 0;
            return i;
        };

        recurse(this.root);

        return result;
    }

    get totalNodes() {
        const recurse = (node: KdTreeNode): number => {
            return 1 +
                (node.left ? recurse(node.left) : 0) +
                (node.right ? recurse(node.right) : 0);
        };
        return recurse(this.root);
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

    // traverse the kd-tree to find the nearest centroid
    findNearest2(point: Float32Array) {
        const calcDistance = (centroid: number) => {
            let result = 0;
            for (let i = 0; i < point.length; ++i) {
                result += (point[i] - this.centroids.columns[i].data[centroid]) ** 2;
            }
            return result;
        };
        const select = (failValue: number, sucessValue: number, test: boolean) => {
            return test ? sucessValue : failValue;
        };
        const { numColumns } = this.centroids;
        const kdTree = this.flatten();

        let mind = 1000000.0;
        let mini = 0;

        let stack: { node: number, depth: number }[] = [];

        for (let i = 0; i < 64; ++i) {
            stack[i] = { node: 0, depth: 0 };
        }

        // initialize first stack element to reference root element
        stack[0].node = 0;
        stack[0].depth = 0;

        let stackIndex = 1;

        let cnt = 0;

        while (stackIndex > 0) {
            // pop the top of the stack
            stackIndex--;
            const s = stack[stackIndex];

            const node = s.node * 3;
            const depth = s.depth;
            const centroid = kdTree[node];
            const left = kdTree[node + 1];
            const right = kdTree[node + 2];

            // calculate distance to the kdtree node
            const d = calcDistance(centroid);
            if (d < mind) {
                mind = d;
                mini = centroid;
            }

            // calculate distance to kdtree split plane
            const axis = depth % numColumns;
            const distance = point[axis] - this.centroids.columns[axis].data[centroid];
            const onRight = distance > 0.0;

            // push the other side if necessary
            if (distance * distance < mind) {
                let other = select(right, left, onRight);
                if (other > 0) {
                    stack[stackIndex].node = other;
                    stack[stackIndex].depth = depth + 1;
                    stackIndex++;
                }
            }

            // push the kdtree node of the side we are on
            const next = select(left, right, onRight);
            if (next > 0) {
                stack[stackIndex].node = next;
                stack[stackIndex].depth = depth + 1;
                stackIndex++;
            }

            cnt++;

            if (stackIndex >= 64) {
                console.log('err');
            }
        }

        return { index: mini, distanceSqr: mind, cnt };
    }

    findNearest3(point: Float32Array) {
        const calcDistance = (centroid: number) => {
            let result = 0;
            for (let i = 0; i < point.length; ++i) {
                result += (point[i] - this.centroids.columns[i].data[centroid]) ** 2;
            }
            return result;
        };

        let mind = 100000.0;
        let mini = 0;

        for (let i = 0; i < this.centroids.numRows; ++i) {
            let d = calcDistance(i);
            if (d < mind) {
                mind = d;
                mini = i;
            }
        }

        return { index: mini, distanceSqr: mind, cnt: this.centroids.numRows };
    }

    private build(indices: Uint32Array, depth: number): KdTreeNode {
        const { centroids } = this;
        const values = centroids.columns[depth % centroids.numColumns].data;
        indices.sort((a, b) => values[a] - values[b]);

        if (indices.length === 1) {
            return {
                index: indices[0]
            };
        } else if (indices.length === 2) {
            return {
                index : indices[0],
                right : {
                    index : indices[1]
                }
            };
        }

        const mid = indices.length >> 1;
        const left = this.build(indices.subarray(0, mid), depth + 1);
        const right = this.build(indices.subarray(mid + 1), depth + 1);
        return {
            index: indices[mid],
            left,
            right
        };
    }
}

export { KdTree };
