import { DataTable } from '../data-table';

interface KdTreeNode {
    index: number;
    count: number;          // self + children indices
    left?: KdTreeNode;
    right?: KdTreeNode;
}

const nthElement = (arr: Uint32Array, lo: number, hi: number, k: number, values: any) => {
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        const va = values[arr[lo]], vb = values[arr[mid]], vc = values[arr[hi]];
        let pivotIdx: number;
        if ((vb - va) * (vc - vb) >= 0) pivotIdx = mid;
        else if ((va - vb) * (vc - va) >= 0) pivotIdx = lo;
        else pivotIdx = hi;

        const pivotVal = values[arr[pivotIdx]];
        let tmp = arr[pivotIdx]; arr[pivotIdx] = arr[hi]; arr[hi] = tmp;
        let store = lo;
        for (let i = lo; i < hi; i++) {
            if (values[arr[i]] < pivotVal) {
                tmp = arr[i]; arr[i] = arr[store]; arr[store] = tmp;
                store++;
            }
        }
        tmp = arr[store]; arr[store] = arr[hi]; arr[hi] = tmp;

        if (store === k) return;
        else if (store < k) lo = store + 1;
        else hi = store - 1;
    }
};

class KdTree {
    centroids: DataTable;
    root: KdTreeNode;
    private colData: any[];

    constructor(centroids: DataTable) {
        const numCols = centroids.numColumns;
        const colData = centroids.columns.map(c => c.data);

        const indices = new Uint32Array(centroids.numRows);
        for (let i = 0; i < indices.length; ++i) {
            indices[i] = i;
        }

        const build = (lo: number, hi: number, depth: number): KdTreeNode => {
            const count = hi - lo + 1;

            if (count === 1) {
                return { index: indices[lo], count: 1 };
            }

            const values = colData[depth % numCols];

            if (count === 2) {
                if (values[indices[lo]] > values[indices[hi]]) {
                    const tmp = indices[lo]; indices[lo] = indices[hi]; indices[hi] = tmp;
                }
                return {
                    index: indices[lo],
                    count: 2,
                    right: { index: indices[hi], count: 1 }
                };
            }

            const mid = lo + (count >> 1);
            nthElement(indices, lo, hi, mid, values);

            const left = build(lo, mid - 1, depth + 1);
            const right = build(mid + 1, hi, depth + 1);

            return {
                index: indices[mid],
                count: 1 + left.count + right.count,
                left,
                right
            };
        };

        this.centroids = centroids;
        this.colData = colData;
        this.root = build(0, indices.length - 1, 0);
    }

    findNearest(point: Float32Array, filterFunc?: (index: number) => boolean) {
        const colData = this.colData;
        const numCols = colData.length;

        let mind = Infinity;
        let mini = -1;
        let cnt = 0;

        const recurse = (node: KdTreeNode, axis: number) => {
            const distance = point[axis] - colData[axis][node.index];
            const next = (distance > 0) ? node.right : node.left;
            const nextAxis = axis + 1 < numCols ? axis + 1 : 0;

            cnt++;

            if (next) {
                recurse(next, nextAxis);
            }

            if (!filterFunc || filterFunc(node.index)) {
                let thisd = 0;
                for (let c = 0; c < numCols; c++) {
                    const v = colData[c][node.index] - point[c];
                    thisd += v * v;
                }
                if (thisd < mind) {
                    mind = thisd;
                    mini = node.index;
                }
            }

            if (distance * distance < mind) {
                const other = next === node.right ? node.left : node.right;
                if (other) {
                    recurse(other, nextAxis);
                }
            }
        };

        recurse(this.root, 0);

        return { index: mini, distanceSqr: mind, cnt };
    }

    findKNearest(point: Float32Array, k: number, filterFunc?: (index: number) => boolean) {
        if (k <= 0) {
            return { indices: new Int32Array(0), distances: new Float32Array(0) };
        }
        k = Math.min(k, this.centroids.numRows);

        const colData = this.colData;
        const numCols = colData.length;

        // Bounded max-heap: stores (distance, index) pairs sorted so the
        // farthest element is at position 0, enabling O(1) pruning bound.
        const heapDist = new Float32Array(k).fill(Infinity);
        const heapIdx = new Int32Array(k).fill(-1);
        let heapSize = 0;

        const heapPush = (dist: number, idx: number) => {
            if (heapSize < k) {
                let pos = heapSize++;
                heapDist[pos] = dist;
                heapIdx[pos] = idx;
                while (pos > 0) {
                    const parent = (pos - 1) >> 1;
                    if (heapDist[parent] < heapDist[pos]) {
                        const td = heapDist[parent]; heapDist[parent] = heapDist[pos]; heapDist[pos] = td;
                        const ti = heapIdx[parent]; heapIdx[parent] = heapIdx[pos]; heapIdx[pos] = ti;
                        pos = parent;
                    } else {
                        break;
                    }
                }
            } else if (dist < heapDist[0]) {
                heapDist[0] = dist;
                heapIdx[0] = idx;
                let pos = 0;
                for (;;) {
                    const left = 2 * pos + 1;
                    const right = 2 * pos + 2;
                    let largest = pos;
                    if (left < k && heapDist[left] > heapDist[largest]) largest = left;
                    if (right < k && heapDist[right] > heapDist[largest]) largest = right;
                    if (largest === pos) break;
                    const td = heapDist[pos]; heapDist[pos] = heapDist[largest]; heapDist[largest] = td;
                    const ti = heapIdx[pos]; heapIdx[pos] = heapIdx[largest]; heapIdx[largest] = ti;
                    pos = largest;
                }
            }
        };

        const recurse = (node: KdTreeNode, axis: number) => {
            const distance = point[axis] - colData[axis][node.index];
            const next = (distance > 0) ? node.right : node.left;
            const nextAxis = axis + 1 < numCols ? axis + 1 : 0;

            if (next) {
                recurse(next, nextAxis);
            }

            if (!filterFunc || filterFunc(node.index)) {
                let thisd = 0;
                for (let c = 0; c < numCols; c++) {
                    const v = colData[c][node.index] - point[c];
                    thisd += v * v;
                }
                heapPush(thisd, node.index);
            }

            const bound = heapSize < k ? Infinity : heapDist[0];
            if (distance * distance < bound) {
                const other = next === node.right ? node.left : node.right;
                if (other) {
                    recurse(other, nextAxis);
                }
            }
        };

        recurse(this.root, 0);

        // Extract results sorted by distance (ascending)
        const resultIndices = new Int32Array(heapSize);
        const resultDist = new Float32Array(heapSize);
        for (let i = 0; i < heapSize; i++) {
            resultIndices[i] = heapIdx[i];
            resultDist[i] = heapDist[i];
        }

        // Simple insertion sort by distance (k is small)
        for (let i = 1; i < heapSize; i++) {
            const d = resultDist[i];
            const idx = resultIndices[i];
            let j = i - 1;
            while (j >= 0 && resultDist[j] > d) {
                resultDist[j + 1] = resultDist[j];
                resultIndices[j + 1] = resultIndices[j];
                j--;
            }
            resultDist[j + 1] = d;
            resultIndices[j + 1] = idx;
        }

        return { indices: resultIndices, distances: resultDist };
    }

    /**
     * Flatten the tree into GPU-friendly typed arrays. Each tree node is
     * assigned a tree-index in pre-order DFS. The arrays are parallel:
     * for tree-index `t`, the node holds splat `nodeSplatIdx[t]` whose
     * position is `(nodeX[t], nodeY[t], nodeZ[t])`. Children live at
     * `nodeLeft[t]` and `nodeRight[t]` (tree indices), with the sentinel
     * `0xFFFFFFFF` for missing children.
     *
     * Positions are denormalised at each tree node (rather than indirected
     * through `nodeSplatIdx` + the source position arrays) so a tree-walk
     * does one read per visit instead of two. Costs 12 bytes/node extra.
     *
     * Layout assumes the underlying `centroids` DataTable has columns
     * `x`, `y`, `z` (the first three columns). The constructor accepts
     * any column set, so callers must ensure these are present and first.
     *
     * @returns Parallel arrays of length N where N = number of points.
     * The root is at index 0.
     */
    flattenForGpu(): {
        nodeSplatIdx: Uint32Array;
        nodeX: Float32Array;
        nodeY: Float32Array;
        nodeZ: Float32Array;
        nodeLeft: Uint32Array;
        nodeRight: Uint32Array;
        rootIdx: number;
        } {
        const n = this.centroids.numRows;
        const nodeSplatIdx = new Uint32Array(n);
        const nodeX = new Float32Array(n);
        const nodeY = new Float32Array(n);
        const nodeZ = new Float32Array(n);
        const nodeLeft = new Uint32Array(n);
        const nodeRight = new Uint32Array(n);
        nodeLeft.fill(0xFFFFFFFF);
        nodeRight.fill(0xFFFFFFFF);

        const x = this.colData[0], y = this.colData[1], z = this.colData[2];

        // Iterative pre-order DFS: assign tree indices, then patch the parent's
        // left/right slot when each child is visited. JS recursion blows the
        // stack on heavily unbalanced trees, so we maintain the work stack
        // ourselves. Encoded entries: nodeRef + (parentTreeIdx, side) where
        // side ∈ {0 = left of parent, 1 = right of parent, 2 = root}.
        //
        // Max DFS depth is the tree's height. `KdTree.build` is recursive and
        // splits at the nthElement median, so the tree is near-balanced and
        // its height is bounded by JS's recursion limit (~10K). A fixed 64
        // entries is enough for any tree this codebase can actually build
        // (2^64 ≫ 10K) and avoids an `n+1`-sized scratch (~85 MB at N=17.9M).
        const stackCap = 64;
        const stackNode: KdTreeNode[] = [this.root];
        const stackParent = new Int32Array(stackCap);
        const stackSide = new Uint8Array(stackCap);
        stackParent[0] = -1;
        stackSide[0] = 2;
        let sp = 1;

        let cursor = 0;
        const rootIdx = cursor;
        while (sp > 0) {
            sp--;
            const node = stackNode[sp];
            const parent = stackParent[sp];
            const side = stackSide[sp];
            const treeIdx = cursor++;
            const splat = node.index;
            nodeSplatIdx[treeIdx] = splat;
            nodeX[treeIdx] = x[splat];
            nodeY[treeIdx] = y[splat];
            nodeZ[treeIdx] = z[splat];
            if (side === 0) nodeLeft[parent] = treeIdx;
            else if (side === 1) nodeRight[parent] = treeIdx;
            // Push right then left so left is popped first (pre-order).
            if (node.right) {
                stackNode[sp] = node.right;
                stackParent[sp] = treeIdx;
                stackSide[sp] = 1;
                sp++;
            }
            if (node.left) {
                stackNode[sp] = node.left;
                stackParent[sp] = treeIdx;
                stackSide[sp] = 0;
                sp++;
            }
        }

        return { nodeSplatIdx, nodeX, nodeY, nodeZ, nodeLeft, nodeRight, rootIdx };
    }
}

export { KdTreeNode, KdTree };
