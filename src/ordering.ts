import { Splat } from './splat';

// sort the compressed indices into morton order
const generateOrdering = (splat: Splat, indices: Uint32Array) => {
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

    let minx: number;
    let miny: number;
    let minz: number;
    let maxx: number;
    let maxy: number;
    let maxz: number;

    const vertex = [0, 0, 0];
    const it = splat.createIterator(['x', 'y', 'z'], vertex);

    // calculate scene extents across all splats (using sort centers, because they're in world space)
    for (let i = 0; i < indices.length; ++i) {
        it(indices[i]);

        const x = vertex[0];
        const y = vertex[1];
        const z = vertex[2];

        if (minx === undefined) {
            minx = maxx = x;
            miny = maxy = y;
            minz = maxz = z;
        } else {
            if (x < minx) minx = x; else if (x > maxx) maxx = x;
            if (y < miny) miny = y; else if (y > maxy) maxy = y;
            if (z < minz) minz = z; else if (z > maxz) maxz = z;
        }
    }

    const xlen = maxx - minx;
    const ylen = maxy - miny;
    const zlen = maxz - minz;

    const xmul = isFinite(xlen) ? 1024 / xlen : 0;
    const ymul = isFinite(ylen) ? 1024 / ylen : 0;
    const zmul = isFinite(zlen) ? 1024 / zlen : 0;

    if (xmul === 0 && ymul === 0 && zmul === 0) {
        return;
    }

    const morton = new Uint32Array(indices.length);
    for (let i = 0; i < indices.length; ++i) {
        it(indices[i]);

        const x = vertex[0];
        const y = vertex[1];
        const z = vertex[2];

        const ix = Math.min(1023, Math.floor((x - minx) * xmul));
        const iy = Math.min(1023, Math.floor((y - miny) * ymul));
        const iz = Math.min(1023, Math.floor((z - minz) * zmul));

        morton[i] = encodeMorton3(ix, iy, iz);
    }

    // sort indices by morton code
    const mapping: Record<number, number> = {};
    indices.forEach((v, i) => mapping[v] = i);
    indices.sort((a, b) => morton[mapping[a]] - morton[mapping[b]]);

    // sort the largest buckets recursively
    let start = 0;
    let end = 0;
    while (start < indices.length) {
        while (end < indices.length && morton[start] === morton[end]) {
            ++end;
        }

        if (end - start > 16) {
            console.log('sorting', end - start);
            generateOrdering(splat, indices.subarray(start, end));
        }

        start = end;
    }
};

export { generateOrdering };
