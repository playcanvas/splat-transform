/**
 * Sort Float64 keys in place while applying the same permutation to paired
 * interleaved Uint32 values. Uses an iterative quicksort so it does not route
 * through V8's regular-array compare-function path.
 *
 * @param keys - Sort keys.
 * @param values - Interleaved value pairs associated with each key.
 * @param length - Number of key/value entries to sort.
 */
function sortKeyMaskPairs(keys: Float64Array, values: Uint32Array, length: number): void {
    if (length < 2) return;

    const swap = (a: number, b: number): void => {
        const key = keys[a]; keys[a] = keys[b]; keys[b] = key;
        const a2 = a * 2, b2 = b * 2;
        const lo = values[a2]; values[a2] = values[b2]; values[b2] = lo;
        const hi = values[a2 + 1]; values[a2 + 1] = values[b2 + 1]; values[b2 + 1] = hi;
    };

    const stack = new Int32Array(64);
    let sp = 0;
    stack[sp++] = 0;
    stack[sp++] = length - 1;

    while (sp > 0) {
        const hi = stack[--sp];
        const lo = stack[--sp];

        if (hi - lo < 16) {
            for (let i = lo + 1; i <= hi; i++) {
                const key = keys[i];
                const valueLo = values[i * 2];
                const valueHi = values[i * 2 + 1];
                let j = i - 1;
                while (j >= lo && keys[j] > key) {
                    keys[j + 1] = keys[j];
                    values[(j + 1) * 2] = values[j * 2];
                    values[(j + 1) * 2 + 1] = values[j * 2 + 1];
                    j--;
                }
                keys[j + 1] = key;
                values[(j + 1) * 2] = valueLo;
                values[(j + 1) * 2 + 1] = valueHi;
            }
            continue;
        }

        const mid = (lo + hi) >>> 1;
        if (keys[lo] > keys[mid]) swap(lo, mid);
        if (keys[lo] > keys[hi]) swap(lo, hi);
        if (keys[mid] > keys[hi]) swap(mid, hi);
        swap(mid, hi - 1);
        const pivot = keys[hi - 1];

        let i = lo;
        let j = hi - 1;
        while (true) {
            while (keys[++i] < pivot) { /* sentinel at hi */ }
            while (keys[--j] > pivot) { /* sentinel at lo */ }
            if (i >= j) break;
            swap(i, j);
        }
        swap(i, hi - 1);

        const leftSize = i - 1 - lo;
        const rightSize = hi - (i + 1);
        if (leftSize > rightSize) {
            stack[sp++] = lo;
            stack[sp++] = i - 1;
            if (rightSize > 0) {
                stack[sp++] = i + 1;
                stack[sp++] = hi;
            }
        } else {
            if (rightSize > 0) {
                stack[sp++] = i + 1;
                stack[sp++] = hi;
            }
            if (leftSize > 0) {
                stack[sp++] = lo;
                stack[sp++] = i - 1;
            }
        }
    }
}

export { sortKeyMaskPairs };
