import { DataTable } from './data-table';
import { logger } from '../utils/logger.js';

// sort the provided indices into morton order
const sortMortonOrder = (dataTable: DataTable, indices: Uint32Array): void => {
    const xCol = dataTable.getColumnByName('x');
    const yCol = dataTable.getColumnByName('y');
    const zCol = dataTable.getColumnByName('z');

    if (!xCol || !yCol || !zCol) {
        logger.debug('missing required position columns');
        return;
    }

    const cx = xCol.data;
    const cy = yCol.data;
    const cz = zCol.data;

    const generate = (indices: Uint32Array) => {
        if (indices.length === 0) {
            return;
        }

        // https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
        const encodeMorton3 = (x: number, y: number, z: number) : number => {
            const Part1By2 = (x: number) => {
                x &= 0x000003ff;
                x = (x ^ (x << 16)) & 0xff0000ff;
                x = (x ^ (x <<  8)) & 0x0300f00f;
                x = (x ^ (x <<  4)) & 0x030c30c3;
                x = (x ^ (x <<  2)) & 0x09249249;
                return x;
            };

            return (Part1By2(z) << 2) + (Part1By2(y) << 1) + Part1By2(x);
        };

        let mx = Infinity;
        let my = Infinity;
        let mz = Infinity;
        let Mx = -Infinity;
        let My = -Infinity;
        let Mz = -Infinity;

        // calculate scene extents across all splats (using sort centers, because they're in world space)
        for (let i = 0; i < indices.length; ++i) {
            const ri = indices[i];
            const x = cx[ri];
            const y = cy[ri];
            const z = cz[ri];

            if (x < mx) mx = x;
            if (x > Mx) Mx = x;
            if (y < my) my = y;
            if (y > My) My = y;
            if (z < mz) mz = z;
            if (z > Mz) Mz = z;
        }

        const xlen = Mx - mx;
        const ylen = My - my;
        const zlen = Mz - mz;

        if (!isFinite(xlen) || !isFinite(ylen) || !isFinite(zlen)) {
            logger.debug('invalid extents', xlen, ylen, zlen);
            return;
        }

        // all points are identical
        if (xlen === 0 && ylen === 0 && zlen === 0) {
            return;
        }

        const xmul = (xlen === 0) ? 0 : 1024 / xlen;
        const ymul = (ylen === 0) ? 0 : 1024 / ylen;
        const zmul = (zlen === 0) ? 0 : 1024 / zlen;

        const morton = new Uint32Array(indices.length);
        for (let i = 0; i < indices.length; ++i) {
            const ri = indices[i];
            const x = cx[ri];
            const y = cy[ri];
            const z = cz[ri];

            const ix = Math.min(1023, (x - mx) * xmul) >>> 0;
            const iy = Math.min(1023, (y - my) * ymul) >>> 0;
            const iz = Math.min(1023, (z - mz) * zmul) >>> 0;

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

        // sort the largest buckets recursively
        let start = 0;
        let end = 1;
        while (start < indices.length) {
            while (end < indices.length && morton[order[end]] === morton[order[start]]) {
                ++end;
            }

            if (end - start > 256) {
                // logger.debug('sorting', end - start);
                generate(indices.subarray(start, end));
            }

            start = end;
        }
    };

    generate(indices);
};

export { sortMortonOrder };
