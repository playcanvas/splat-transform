import { DataTable } from '../data-table';

function argsortHuge(values: ArrayLike<number>, indices: Uint32Array, chunkSize = 2_000_000): void {
    const n = indices.length;
    if (n <= 1) return;

    interface Run { data: Uint32Array; pos: number; }
    const runs: Run[] = [];

    for (let start = 0; start < n; start += chunkSize) {
        const end = Math.min(start + chunkSize, n);
        const chunk = new Uint32Array(end - start);
        chunk.set(indices.subarray(start, end));
        chunk.sort((a, b) => values[a] - values[b]);
        runs.push({ data: chunk, pos: 0 });
    }

    const out = new Uint32Array(n);

    const heap: number[] = [];
    const less = (i: number, j: number) => {
        const ai = runs[i].data[runs[i].pos];
        const aj = runs[j].data[runs[j].pos];
        return values[ai] < values[aj];
    };
    const heapSwap = (i: number, j: number) => {
        const tmp = heap[i];
        heap[i] = heap[j];
        heap[j] = tmp;
    };
    const siftUp = (i: number) => {
        while (i) {
            const p = (i - 1) >> 1;
            if (!less(i, p)) break;
            heapSwap(i, p);
            i = p;
        }
    };
    const siftDown = (i: number) => {
        for (;;) {
            const l = i * 2 + 1;
            const r = l + 1;
            let m = i;
            if (l < heap.length && less(l, m)) m = l;
            if (r < heap.length && less(r, m)) m = r;
            if (m === i) break;
            heapSwap(i, m);
            i = m;
        }
    };

    for (let k = 0; k < runs.length; k++) {
        if (runs[k].pos < runs[k].data.length) {
            heap.push(k);
            siftUp(heap.length - 1);
        }
    }

    for (let i = 0; i < n; i++) {
        const top = heap[0];
        out[i] = runs[top].data[runs[top].pos++];
        if (runs[top].pos === runs[top].data.length) {
            const last = heap.pop();
            if (heap.length) {
                heap[0] = last as number;
                siftDown(0);
            }
        } else {
            siftDown(0);
        }
    }

    indices.set(out);
}

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

            // sorting code that works for smaller arrays
            // indices.sort((a, b) => values[a] - values[b]);

            // sorting code that works for larger arrays instead
            argsortHuge(values, indices);

            const mid = indices.length >> 1;
            const left = recurse(indices.subarray(0, mid));
            const right = recurse(indices.subarray(mid));

            return {
                count: 1 + left.count + right.count,
                aabb,
                left,
                right
            };
        };

        const indices = new Uint32Array(centroids.numRows);
        indices.forEach((v, i) => {
            indices[i] = i;
        });
        this.centroids = centroids;
        this.root = recurse(indices);
    }
}

export { BTreeNode, BTree };
