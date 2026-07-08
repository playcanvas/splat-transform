import { type ChunkDataPool, type ChunkSource } from '../chunk';
import { logger } from '../utils';

// Sort `indices` in place into morton (Z-order), reading each point's xyz from
// strided coordinate arrays: point `g`'s coordinates are `xa[g*stride + ox]`,
// `ya[g*stride + oy]`, `za[g*stride + oz]`. This one core serves both the
// interleaved packing (one array, stride 3) and columnar positions (three
// arrays, stride 1) — see sortMortonInterleaved / sortMortonColumns.
//
// Recursive-refined: a bucket of >256 indices sharing the same 10-bit code is
// re-sorted by a finer pass over its sub-range.
const sortMortonStrided = (
    indices: Uint32Array,
    xa: ArrayLike<number>, ya: ArrayLike<number>, za: ArrayLike<number>,
    stride: number, ox: number, oy: number, oz: number
): void => {
    const generate = (indices: Uint32Array) => {
        if (indices.length === 0) {
            return;
        }

        // https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
        const encodeMorton3 = (x: number, y: number, z: number): number => {
            const Part1By2 = (v: number) => {
                v &= 0x000003ff;
                v = (v ^ (v << 16)) & 0xff0000ff;
                v = (v ^ (v << 8)) & 0x0300f00f;
                v = (v ^ (v << 4)) & 0x030c30c3;
                v = (v ^ (v << 2)) & 0x09249249;
                return v;
            };
            return (Part1By2(z) << 2) + (Part1By2(y) << 1) + Part1By2(x);
        };

        let mx = Infinity, my = Infinity, mz = Infinity;
        let Mx = -Infinity, My = -Infinity, Mz = -Infinity;

        // scene extents across these splats (world-space positions)
        for (let i = 0; i < indices.length; ++i) {
            const b = indices[i] * stride;
            const x = xa[b + ox], y = ya[b + oy], z = za[b + oz];
            if (x < mx) mx = x;
            if (x > Mx) Mx = x;
            if (y < my) my = y;
            if (y > My) My = y;
            if (z < mz) mz = z;
            if (z > Mz) Mz = z;
        }

        const xlen = Mx - mx, ylen = My - my, zlen = Mz - mz;
        if (!isFinite(xlen) || !isFinite(ylen) || !isFinite(zlen)) {
            logger.debug('invalid extents', xlen, ylen, zlen);
            return;
        }
        if (xlen === 0 && ylen === 0 && zlen === 0) {
            return; // all points identical
        }

        const xmul = (xlen === 0) ? 0 : 1024 / xlen;
        const ymul = (ylen === 0) ? 0 : 1024 / ylen;
        const zmul = (zlen === 0) ? 0 : 1024 / zlen;

        const morton = new Uint32Array(indices.length);
        for (let i = 0; i < indices.length; ++i) {
            const b = indices[i] * stride;
            const ix = Math.min(1023, (xa[b + ox] - mx) * xmul) >>> 0;
            const iy = Math.min(1023, (ya[b + oy] - my) * ymul) >>> 0;
            const iz = Math.min(1023, (za[b + oz] - mz) * zmul) >>> 0;
            morton[i] = encodeMorton3(ix, iy, iz);
        }

        // sort indices by morton code
        const order = new Uint32Array(indices.length);
        for (let i = 0; i < order.length; i++) {
            order[i] = i;
        }
        order.sort((a, b) => morton[a] - morton[b]);

        const tmpIndices = indices.slice();
        for (let i = 0; i < indices.length; ++i) {
            indices[i] = tmpIndices[order[i]];
        }

        // recursively refine the largest equal-code buckets
        let start = 0, end = 1;
        while (start < indices.length) {
            while (end < indices.length && morton[order[end]] === morton[order[start]]) {
                ++end;
            }
            if (end - start > 256) {
                generate(indices.subarray(start, end));
            }
            start = end;
        }
    };

    generate(indices);
};

/**
 * Sort `indices` in place into morton (Z-order) using interleaved positions
 * `[x, y, z, x, y, z, ...]` — the natural packing of the `position` layer.
 *
 * @param positions - Interleaved xyz; gaussian `g` is at `positions[g*3 + {0,1,2}]`.
 * @param indices - Indices to sort in place.
 */
const sortMortonInterleaved = (positions: ArrayLike<number>, indices: Uint32Array): void => {
    sortMortonStrided(indices, positions, positions, positions, 3, 0, 1, 2);
};

/**
 * Sort `indices` in place into morton (Z-order) using columnar positions: point
 * `g`'s coordinates are `x[g]`, `y[g]`, `z[g]`. The columnar sibling of
 * {@link sortMortonInterleaved} for resident position columns (e.g. the LOD
 * writer's slim centroids) and `DataTable` x/y/z columns.
 *
 * @param x - X coordinates, indexed by gaussian.
 * @param y - Y coordinates, indexed by gaussian.
 * @param z - Z coordinates, indexed by gaussian.
 * @param indices - Indices to sort in place.
 */
const sortMortonColumns = (x: ArrayLike<number>, y: ArrayLike<number>, z: ArrayLike<number>, indices: Uint32Array): void => {
    sortMortonStrided(indices, x, y, z, 1, 0, 0, 0);
};

/**
 * Compute a Morton (Z-order) permutation over a source's gaussians (LOD 0).
 *
 * Gathers the `position` layer into one interleaved `[x,y,z,...]` array (a
 * contiguous copy per chunk — no de-interleave) and runs the recursive-refined
 * sort. Returns `order` where `order[i]` is the gaussian that sorts to output
 * position `i`.
 *
 * Positions are read in the source's current space; bake first (e.g. via
 * `bakeTransform`) if the order must match a target coordinate space.
 *
 * @param src - Source to order (must carry a `position` layer).
 * @param pool - Pool for the temporary position read buffers; `chunkSize` must be >= the source's.
 * @returns The Morton permutation, length `src.meta.numGaussians`.
 */
const mortonOrder = async (src: ChunkSource, pool: ChunkDataPool): Promise<Uint32Array> => {
    const { meta } = src;
    if (!meta.availableLayers.has('position')) {
        throw new Error('mortonOrder: source has no position layer');
    }

    const n = meta.numGaussians;
    const positions = new Float32Array(3 * n);
    const layout = meta.layouts.position!;
    const numChunks = meta.numChunks[0] ?? 0;
    let base = 0;
    for (let k = 0; k < numChunks; k++) {
        const count = Math.min(meta.chunkSize, n - base);
        const cd = pool.acquire('position', layout, count);
        await src.read({ chunkIndex: k, position: cd });
        // position layer is packed [x,y,z]; copy straight into the interleaved buffer.
        positions.set(new Float32Array(cd.data, 0, count * 3), base * 3);
        cd.release();
        base += count;
    }

    const order = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
        order[i] = i;
    }
    sortMortonInterleaved(positions, order);
    return order;
};

export { mortonOrder, sortMortonInterleaved, sortMortonColumns };
