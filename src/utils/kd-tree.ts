interface KdTreeNode {
    index: number;
    left?: KdTreeNode;
    right?: KdTreeNode;
}

class KdTree {
    data: Float32Array[];
    root: KdTreeNode;

    constructor(data: Float32Array[]) {
        const indices = new Uint32Array(data[0].length);
        indices.forEach((v, i) => {
            indices[i] = i;
        });
        this.data = data;
        this.root = this.build(indices, 0);
    }

    findNearest(point: Float32Array, filterFunc?: (index: number) => boolean) {
        const { data } = this;
        const { length } = data;

        const calcDistance = (index: number) => {
            let l = 0;
            for (let i = 0; i < length; ++i) {
                const v = data[i][index] - point[i];
                l += v * v;
            }
            return l;
        };

        let mind = Infinity;
        let mini = -1;
        let cnt = 0;

        const recurse = (node: KdTreeNode, depth: number) => {
            const axis = depth % length;
            const distance = point[axis] - (this.data[axis])[node.index];
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

    private build(indices: Uint32Array, depth: number): KdTreeNode {
        const { data } = this;
        const values = data[depth % data.length];
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
