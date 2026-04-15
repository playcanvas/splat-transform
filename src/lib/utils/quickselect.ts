import { TypedArray } from '../data-table';

/**
 * Partition indices around the k-th smallest element using quickselect
 * (median-of-three pivot selection).
 *
 * After this call, `idx[k]` holds the index of the k-th smallest value
 * in `data`, and all indices before k map to smaller-or-equal values.
 *
 * @param data - The data array to use for comparison values.
 * @param idx - The index array to partition (mutated in place).
 * @param k - The target partition index.
 * @returns The index value at position k after partitioning.
 */
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

    while (true) {
        if (r <= l + 1) {
            if (r === l + 1 && valAt(r) < valAt(l)) swap(l, r);
            return idx[k];
        }

        const mid = (l + r) >>> 1;
        swap(mid, l + 1);
        if (valAt(l) > valAt(r)) swap(l, r);
        if (valAt(l + 1) > valAt(r)) swap(l + 1, r);
        if (valAt(l) > valAt(l + 1)) swap(l, l + 1);

        let i = l + 1;
        let j = r;
        const pivotIdxVal = valAt(l + 1);
        const pivotIdx = idx[l + 1];

        while (true) {
            do {
                i++;
            } while (i <= r && valAt(i) < pivotIdxVal);
            do {
                j--;
            } while (j >= l && valAt(j) > pivotIdxVal);
            if (j < i) break;
            swap(i, j);
        }

        idx[l + 1] = idx[j];
        idx[j] = pivotIdx;

        if (j >= k) r = j - 1;
        if (j <= k) l = i;
    }
};

export { quickselect };
