// import { GaussianData } from './gaussian-data';
// import { GaussianRef } from './write-compressed-ply';

// // sort the compressed indices into morton order
// const generateOrdering = (refs: GaussianRef[]) => {
//     const generate = (indices: Uint32Array) => {
//         // https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
//         const encodeMorton3 = (x: number, y: number, z: number) : number => {
//             const Part1By2 = (x: number) => {
//                 x &= 0x000003ff;
//                 x = (x ^ (x << 16)) & 0xff0000ff;
//                 x = (x ^ (x <<  8)) & 0x0300f00f;
//                 x = (x ^ (x <<  4)) & 0x030c30c3;
//                 x = (x ^ (x <<  2)) & 0x09249249;
//                 return x;
//             };

//             return (Part1By2(z) << 2) + (Part1By2(y) << 1) + Part1By2(x);
//         };

//         let mx: number;
//         let my: number;
//         let mz: number;
//         let Mx: number;
//         let My: number;
//         let Mz: number;

//         const gdata = new GaussianData(['x', 'y', 'z']);

//         // calculate scene extents across all splats (using sort centers, because they're in world space)
//         for (let i = 0; i < indices.length; ++i) {
//             const ri = indices[i];
//             gdata.read(refs[ri].splat, refs[ri].i);

//             const x = gdata.values.x;
//             const y = gdata.values.y;
//             const z = gdata.values.z;

//             if (mx === undefined) {
//                 mx = Mx = x;
//                 my = My = y;
//                 mz = Mz = z;
//             } else {
//                 if (x < mx) mx = x; else if (x > Mx) Mx = x;
//                 if (y < my) my = y; else if (y > My) My = y;
//                 if (z < mz) mz = z; else if (z > Mz) Mz = z;
//             }
//         }

//         const xlen = Mx - mx;
//         const ylen = My - my;
//         const zlen = Mz - mz;

//         if (!isFinite(xlen) || !isFinite(ylen) || !isFinite(zlen)) {
//             console.log('invalid extents', xlen, ylen, zlen);
//             return;
//         }

//         const xmul = 1024 / xlen;
//         const ymul = 1024 / ylen;
//         const zmul = 1024 / zlen;

//         const morton = new Uint32Array(indices.length);
//         for (let i = 0; i < indices.length; ++i) {
//             const ri = indices[i];
//             gdata.read(refs[ri].splat, refs[ri].i);

//             const ix = Math.min(1023, (gdata.values.x - mx) * xmul) >>> 0;
//             const iy = Math.min(1023, (gdata.values.y - my) * ymul) >>> 0;
//             const iz = Math.min(1023, (gdata.values.z - mz) * zmul) >>> 0;

//             morton[i] = encodeMorton3(ix, iy, iz);
//         }

//         // sort indices by morton code
//         const order = indices.map((_, i) => i);
//         order.sort((a, b) => morton[a] - morton[b]);

//         const tmpIndices = indices.slice();
//         for (let i = 0; i < indices.length; ++i) {
//             indices[i] = tmpIndices[order[i]];
//         }

//         // sort the largest buckets recursively
//         let start = 0;
//         let end = 1;
//         while (start < indices.length) {
//             while (end < indices.length && morton[order[end]] === morton[order[start]]) {
//                 ++end;
//             }

//             if (end - start > 256) {
//                 // console.log('sorting', end - start);
//                 generate(indices.subarray(start, end));
//             }

//             start = end;
//         }
//     };

//     const indices = new Uint32Array(refs.length);
//     for (let i = 0; i < refs.length; ++i) {
//         indices[i] = i;
//     }

//     generate(indices);

//     return indices;
// };

// export { generateOrdering };
