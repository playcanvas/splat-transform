/**
 * KD-tree over raw column arrays: a pointer-node tree for CPU queries
 * (`KdTree`), and a direct-to-flat builder (`buildFlatKdTree`) emitting the
 * GPU traversal layout without intermediate node objects.
 *
 * Engine-free by contract: worker tasks build trees off-thread, and the
 * worker bundle inlines its whole import graph — an engine import here would
 * embed playcanvas into dist/worker.mjs (see the note atop workers/tasks.ts).
 *
 * `KdTree` dimensionality is the number of columns: 3 for spatial consumers,
 * arbitrary for k-means centroid assignment. `buildFlatKdTree` is 3D only.
 */

interface KdTreeNode {
    index: number;
    count: number;          // self + children indices
    left?: KdTreeNode;
    right?: KdTreeNode;
}

/**
 * A kd-tree in GPU-friendly flat typed arrays (the exact layout GpuKnn
 * uploads). For tree index `t`, the node holds splat `nodeSplatIdx[t]` at
 * position `nodePositions[t*3 + 0/1/2]`; children are
 * `nodeChildren[t*2 + 0/1]` (left, right) with the sentinel `0xFFFFFFFF`
 * for missing children. The root is at index 0. Node positions are
 * denormalised (rather than indirected through `nodeSplatIdx` + the source
 * position arrays) so a tree-walk does one read per visit instead of two.
 */
type FlatKdTree = {
    nodeSplatIdx: Uint32Array;
    nodePositions: Float32Array;
    nodeChildren: Uint32Array;
    rootIdx: number;
};

const nthElement = (arr: Uint32Array, lo: number, hi: number, k: number, values: ArrayLike<number>) => {
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        const va = values[arr[lo]], vb = values[arr[mid]], vc = values[arr[hi]];
        let pivotIdx: number;
        if ((vb - va) * (vc - vb) >= 0) pivotIdx = mid;
        else if ((va - vb) * (vc - va) >= 0) pivotIdx = lo;
        else pivotIdx = hi;

        const pivotVal = values[arr[pivotIdx]];

        // 3-way (Dutch National Flag) partition around pivotVal:
        //   [lo..lt-1] < pivot, [lt..gt] == pivot, [gt+1..hi] > pivot.
        // The 2-way Lomuto partition this replaces moved only strictly-less
        // elements, so an all-equal range shrank by one per pass and degenerated
        // to O(N^2) — fatal for inputs where many points share a coordinate
        // (e.g. a splat with every gaussian at the origin).
        let lt = lo, gt = hi, i = lo;
        let tmp: number;
        while (i <= gt) {
            const v = values[arr[i]];
            if (v < pivotVal) {
                tmp = arr[i]; arr[i] = arr[lt]; arr[lt] = tmp;
                lt++; i++;
            } else if (v > pivotVal) {
                tmp = arr[i]; arr[i] = arr[gt]; arr[gt] = tmp;
                gt--;
            } else {
                i++;
            }
        }

        if (k < lt) hi = lt - 1;
        else if (k > gt) lo = gt + 1;
        else return; // k within the equal block; arr[k] is the order statistic
    }
};

/**
 * Build a 3D kd-tree directly into the flat GPU layout — no intermediate
 * pointer-node graph (which costs ~50-80 B/node of JS objects and GC
 * pressure, prohibitive at part-scale point counts).
 *
 * Same structure as `KdTree`'s build: median split via `nthElement`, axis
 * cycling x→y→z by depth, two-point ranges ordered so the smaller value is
 * the node and the larger its right child. Nodes are emitted in pre-order
 * DFS (left subtree before right), root at index 0. Recursion depth is
 * bounded by the median split (≈ log2 n).
 *
 * @param x - Point x column.
 * @param y - Point y column.
 * @param z - Point z column.
 * @returns The flat tree over all points.
 */
const buildFlatKdTree = (x: Float32Array, y: Float32Array, z: Float32Array): FlatKdTree => {
    const n = x.length;
    const cols = [x, y, z];
    const indices = new Uint32Array(n);
    for (let i = 0; i < n; i++) indices[i] = i;

    const nodeSplatIdx = new Uint32Array(n);
    const nodePositions = new Float32Array(n * 3);
    const nodeChildren = new Uint32Array(n * 2).fill(0xFFFFFFFF);

    let cursor = 0;
    const emit = (splat: number): number => {
        const t = cursor++;
        nodeSplatIdx[t] = splat;
        const t3 = t * 3;
        nodePositions[t3] = x[splat];
        nodePositions[t3 + 1] = y[splat];
        nodePositions[t3 + 2] = z[splat];
        return t;
    };

    const build = (lo: number, hi: number, depth: number): number => {
        const count = hi - lo + 1;

        if (count === 1) return emit(indices[lo]);

        const values = cols[depth % 3];

        if (count === 2) {
            if (values[indices[lo]] > values[indices[hi]]) {
                const tmp = indices[lo]; indices[lo] = indices[hi]; indices[hi] = tmp;
            }
            const t = emit(indices[lo]);
            nodeChildren[t * 2 + 1] = emit(indices[hi]);
            return t;
        }

        const mid = lo + (count >> 1);
        nthElement(indices, lo, hi, mid, values);

        const t = emit(indices[mid]);
        nodeChildren[t * 2] = build(lo, mid - 1, depth + 1);
        nodeChildren[t * 2 + 1] = build(mid + 1, hi, depth + 1);
        return t;
    };

    if (n > 0) build(0, n - 1, 0);

    return { nodeSplatIdx, nodePositions, nodeChildren, rootIdx: 0 };
};

class KdTree {
    root: KdTreeNode;
    readonly colData: ArrayLike<number>[];
    readonly numRows: number;

    constructor(colData: ArrayLike<number>[]) {
        const numCols = colData.length;
        const numRows = colData[0].length;

        const indices = new Uint32Array(numRows);
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

        this.colData = colData;
        this.numRows = numRows;
        this.root = build(0, indices.length - 1, 0);
    }

    findNearest(point: ArrayLike<number>, filterFunc?: (index: number) => boolean) {
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

    findKNearest(point: ArrayLike<number>, k: number, filterFunc?: (index: number) => boolean) {
        if (k <= 0) {
            return { indices: new Int32Array(0), distances: new Float32Array(0) };
        }
        k = Math.min(k, this.numRows);

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
}

export { KdTree, buildFlatKdTree, type KdTreeNode, type FlatKdTree };
