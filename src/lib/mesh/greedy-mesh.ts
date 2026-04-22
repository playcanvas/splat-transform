import type { Mesh } from './marching-cubes';
import type { Bounds } from '../data-table';
import { BlockMaskBuffer } from '../voxel/block-mask-buffer';
import { mortonToXYZ, xyzToMorton } from '../voxel/morton';

const SOLID_LO = 0xFFFFFFFF >>> 0;
const SOLID_HI = 0xFFFFFFFF >>> 0;

/**
 * Test whether a voxel is occupied within a block's 64-bit mask.
 * Bit layout matches BlockMaskBuffer: bitIdx = lx + ly*4 + lz*16
 * (lo holds bits 0-31, hi holds bits 32-63).
 *
 * @param lo - Lower 32 bits of the block mask
 * @param hi - Upper 32 bits of the block mask
 * @param lx - Local X coordinate within the block (0-3)
 * @param ly - Local Y coordinate within the block (0-3)
 * @param lz - Local Z coordinate within the block (0-3)
 * @returns True if the voxel bit is set
 */
const isVoxelSet = (lo: number, hi: number, lx: number, ly: number, lz: number): boolean => {
    const bitIdx = lx + ly * 4 + lz * 16;
    if (bitIdx < 32) {
        return (lo & (1 << bitIdx)) !== 0;
    }
    return (hi & (1 << (bitIdx - 32))) !== 0;
};

interface BlockEntry {
    bx: number;
    by: number;
    bz: number;
    lo: number;
    hi: number;
    isSolid: boolean;
}

/**
 * Extract a triangle mesh from a binary voxel grid using greedy quad meshing.
 *
 * Produces the optimal axis-aligned mesh for the underlying voxel surface:
 * one rectangle per maximal coplanar run of exposed voxel faces, fanned to
 * two triangles. Output has zero geometric error and is typically 10-100x
 * smaller than marching cubes + meshopt simplify on the same data.
 *
 * Output triangles use clockwise / counter-clockwise winding such that the
 * resulting normal points outward from the occupied region.
 *
 * @param buffer - Voxel block data after filtering.
 * @param gridBounds - Grid bounds aligned to block boundaries.
 * @param voxelResolution - Size of each voxel in world units.
 * @returns Mesh with positions and indices.
 */
const greedyVoxelMesh = (
    buffer: BlockMaskBuffer,
    gridBounds: Bounds,
    voxelResolution: number
): Mesh => {
    const mixed = buffer.getMixedBlocks();
    const solidArr = buffer.getSolidBlocks();
    const masks = mixed.masks;

    // Global block lookup. Value = mixed block index, or -1 for solid blocks.
    const blockMap = new Map<number, number>();
    for (let i = 0; i < mixed.morton.length; i++) {
        blockMap.set(mixed.morton[i], i);
    }
    for (let i = 0; i < solidArr.length; i++) {
        blockMap.set(solidArr[i], -1);
    }

    if (blockMap.size === 0) {
        return { positions: new Float32Array(0), indices: new Uint32Array(0) };
    }

    const Vx = Math.round((gridBounds.max.x - gridBounds.min.x) / voxelResolution);
    const Vy = Math.round((gridBounds.max.y - gridBounds.min.y) / voxelResolution);
    const Vz = Math.round((gridBounds.max.z - gridBounds.min.z) / voxelResolution);
    const originX = gridBounds.min.x;
    const originY = gridBounds.min.y;
    const originZ = gridBounds.min.z;

    // Materialize block entries (cheap snapshot to avoid repeated Map iteration / mortonToXYZ).
    const allBlocks: BlockEntry[] = [];
    blockMap.forEach((entry, m) => {
        const [bx, by, bz] = mortonToXYZ(m);
        if (entry === -1) {
            allBlocks.push({ bx, by, bz, lo: SOLID_LO, hi: SOLID_HI, isSolid: true });
        } else {
            allBlocks.push({ bx, by, bz, lo: masks[entry * 2], hi: masks[entry * 2 + 1], isSolid: false });
        }
    });

    // Slow-path lookup for outer voxels in neighbor blocks.
    const outerOccupied = (bx: number, by: number, bz: number, lx: number, ly: number, lz: number): boolean => {
        if (bx < 0 || by < 0 || bz < 0) return false;
        const e = blockMap.get(xyzToMorton(bx, by, bz));
        if (e === undefined) return false;
        if (e === -1) return true;
        return isVoxelSet(masks[e * 2], masks[e * 2 + 1], lx, ly, lz);
    };

    // -------------------------------------------------------------------
    // Output buffers (capacity-doubling typed arrays).
    // -------------------------------------------------------------------
    let posCap = 1024;
    let posLen = 0;
    let positions = new Float32Array(posCap);
    let idxCap = 1024;
    let idxLen = 0;
    let indices = new Uint32Array(idxCap);

    const ensurePosCap = (need: number) => {
        if (posLen + need <= posCap) return;
        while (posLen + need > posCap) {
            posCap *= 2;
        }
        const grown = new Float32Array(posCap);
        grown.set(positions);
        positions = grown;
    };
    const ensureIdxCap = (need: number) => {
        if (idxLen + need <= idxCap) return;
        while (idxLen + need > idxCap) {
            idxCap *= 2;
        }
        const grown = new Uint32Array(idxCap);
        grown.set(indices);
        indices = grown;
    };

    // Vertex welder. Quad corners always sit on the integer voxel-grid
    // lattice, so we key by (ix, iy, iz) computed exactly from the world
    // position. Sharing vertices between adjacent quads (both coplanar
    // and at corners) is required for downstream meshopt simplification
    // to see the mesh as topologically connected.
    const lyStride = Vx + 1;
    const lzStride = lyStride * (Vy + 1);
    const vertexMap = new Map<number, number>();
    const getOrAddVertex = (x: number, y: number, z: number): number => {
        const ix = Math.round((x - originX) / voxelResolution);
        const iy = Math.round((y - originY) / voxelResolution);
        const iz = Math.round((z - originZ) / voxelResolution);
        const key = ix + iy * lyStride + iz * lzStride;
        const existing = vertexMap.get(key);
        if (existing !== undefined) return existing;
        ensurePosCap(3);
        const idx = posLen / 3;
        positions[posLen++] = x;
        positions[posLen++] = y;
        positions[posLen++] = z;
        vertexMap.set(key, idx);
        return idx;
    };

    // Emit a quad as two triangles: (0,1,2) and (0,2,3). Caller is
    // responsible for supplying corners in the correct order to produce
    // the desired outward normal.
    const emitQuad = (
        x0: number, y0: number, z0: number,
        x1: number, y1: number, z1: number,
        x2: number, y2: number, z2: number,
        x3: number, y3: number, z3: number
    ) => {
        const i0 = getOrAddVertex(x0, y0, z0);
        const i1 = getOrAddVertex(x1, y1, z1);
        const i2 = getOrAddVertex(x2, y2, z2);
        const i3 = getOrAddVertex(x3, y3, z3);
        ensureIdxCap(6);
        indices[idxLen++] = i0;
        indices[idxLen++] = i1;
        indices[idxLen++] = i2;
        indices[idxLen++] = i0;
        indices[idxLen++] = i2;
        indices[idxLen++] = i3;
    };

    // -------------------------------------------------------------------
    // Greedy 2D rectangle mesher over a Uint8Array slice mask.
    //
    // mask[u + v*U] is one of:
    //   0 = empty / face not present
    //   1 = available (face present, not yet consumed)
    //   2 = consumed by an emitted rectangle
    // -------------------------------------------------------------------
    const meshSlice = (
        mask: Uint8Array, U: number, V: number,
        emitter: (u0: number, v0: number, u1: number, v1: number) => void
    ) => {
        for (let v = 0; v < V; v++) {
            for (let u = 0; u < U; u++) {
                if (mask[u + v * U] !== 1) continue;

                // Maximal width along u.
                let w = 1;
                while (u + w < U && mask[(u + w) + v * U] === 1) w++;

                // Maximal height along v while every cell in the row is set.
                let h = 1;
                while (v + h < V) {
                    let rowFull = true;
                    for (let du = 0; du < w; du++) {
                        if (mask[(u + du) + (v + h) * U] !== 1) {
                            rowFull = false;
                            break;
                        }
                    }
                    if (!rowFull) break;
                    h++;
                }

                for (let dv = 0; dv < h; dv++) {
                    for (let du = 0; du < w; du++) {
                        mask[(u + du) + (v + dv) * U] = 2;
                    }
                }

                emitter(u, v, u + w, v + h);
            }
        }
    };

    // -------------------------------------------------------------------
    // Per-axis processing.
    //
    // For each axis a primary index (bx/by/bz) groups blocks; we sweep
    // slices along that axis, building one mask per slice, meshing it, and
    // immediately freeing it. Mask peak memory is O(U * V) per axis.
    // -------------------------------------------------------------------

    // Group blocks by each primary axis once.
    const blocksByBx = new Map<number, BlockEntry[]>();
    const blocksByBy = new Map<number, BlockEntry[]>();
    const blocksByBz = new Map<number, BlockEntry[]>();
    for (const b of allBlocks) {
        let arr = blocksByBx.get(b.bx);
        if (!arr) {
            arr = [];
            blocksByBx.set(b.bx, arr);
        }
        arr.push(b);
        arr = blocksByBy.get(b.by);
        if (!arr) {
            arr = [];
            blocksByBy.set(b.by, arr);
        }
        arr.push(b);
        arr = blocksByBz.get(b.bz);
        if (!arr) {
            arr = [];
            blocksByBz.set(b.bz, arr);
        }
        arr.push(b);
    }

    // ============ X axis ============
    {
        const U = Vy, V = Vz;
        const mask = new Uint8Array(U * V);
        const sortedBx: number[] = [];
        blocksByBx.forEach((_v, k) => sortedBx.push(k));
        sortedBx.sort((a, b) => a - b);

        // +X faces: inner voxel at vx = s-1, outer at vx = s.
        for (const bxInner of sortedBx) {
            for (let lxInner = 0; lxInner < 4; lxInner++) {
                const s = bxInner * 4 + lxInner + 1;
                if (s > Vx) continue;
                let any = false;
                for (const B of blocksByBx.get(bxInner)!) {
                    for (let lz = 0; lz < 4; lz++) {
                        for (let ly = 0; ly < 4; ly++) {
                            const innerOcc = B.isSolid || isVoxelSet(B.lo, B.hi, lxInner, ly, lz);
                            if (!innerOcc) continue;
                            let outerOcc: boolean;
                            if (lxInner < 3) {
                                outerOcc = B.isSolid || isVoxelSet(B.lo, B.hi, lxInner + 1, ly, lz);
                            } else {
                                outerOcc = outerOccupied(B.bx + 1, B.by, B.bz, 0, ly, lz);
                            }
                            if (!outerOcc) {
                                mask[(B.by * 4 + ly) + (B.bz * 4 + lz) * U] = 1;
                                any = true;
                            }
                        }
                    }
                }
                if (!any) continue;
                meshSlice(mask, U, V, (u0, v0, u1, v1) => {
                    const x = originX + s * voxelResolution;
                    const y0 = originY + u0 * voxelResolution;
                    const y1 = originY + u1 * voxelResolution;
                    const z0 = originZ + v0 * voxelResolution;
                    const z1 = originZ + v1 * voxelResolution;
                    // +X normal: (y0,z0) -> (y1,z0) -> (y1,z1) -> (y0,z1)
                    emitQuad(
                        x, y0, z0,
                        x, y1, z0,
                        x, y1, z1,
                        x, y0, z1
                    );
                });
                mask.fill(0);
            }
        }

        // -X faces: inner voxel at vx = s, outer at vx = s-1.
        for (const bxInner of sortedBx) {
            for (let lxInner = 0; lxInner < 4; lxInner++) {
                const s = bxInner * 4 + lxInner;
                let any = false;
                for (const B of blocksByBx.get(bxInner)!) {
                    for (let lz = 0; lz < 4; lz++) {
                        for (let ly = 0; ly < 4; ly++) {
                            const innerOcc = B.isSolid || isVoxelSet(B.lo, B.hi, lxInner, ly, lz);
                            if (!innerOcc) continue;
                            let outerOcc: boolean;
                            if (lxInner > 0) {
                                outerOcc = B.isSolid || isVoxelSet(B.lo, B.hi, lxInner - 1, ly, lz);
                            } else {
                                outerOcc = outerOccupied(B.bx - 1, B.by, B.bz, 3, ly, lz);
                            }
                            if (!outerOcc) {
                                mask[(B.by * 4 + ly) + (B.bz * 4 + lz) * U] = 1;
                                any = true;
                            }
                        }
                    }
                }
                if (!any) continue;
                meshSlice(mask, U, V, (u0, v0, u1, v1) => {
                    const x = originX + s * voxelResolution;
                    const y0 = originY + u0 * voxelResolution;
                    const y1 = originY + u1 * voxelResolution;
                    const z0 = originZ + v0 * voxelResolution;
                    const z1 = originZ + v1 * voxelResolution;
                    // -X normal: reverse winding
                    emitQuad(
                        x, y0, z0,
                        x, y0, z1,
                        x, y1, z1,
                        x, y1, z0
                    );
                });
                mask.fill(0);
            }
        }
    }

    // ============ Y axis ============
    {
        const U = Vx, V = Vz;
        const mask = new Uint8Array(U * V);
        const sortedBy: number[] = [];
        blocksByBy.forEach((_v, k) => sortedBy.push(k));
        sortedBy.sort((a, b) => a - b);

        // +Y faces
        for (const byInner of sortedBy) {
            for (let lyInner = 0; lyInner < 4; lyInner++) {
                const s = byInner * 4 + lyInner + 1;
                if (s > Vy) continue;
                let any = false;
                for (const B of blocksByBy.get(byInner)!) {
                    for (let lz = 0; lz < 4; lz++) {
                        for (let lx = 0; lx < 4; lx++) {
                            const innerOcc = B.isSolid || isVoxelSet(B.lo, B.hi, lx, lyInner, lz);
                            if (!innerOcc) continue;
                            let outerOcc: boolean;
                            if (lyInner < 3) {
                                outerOcc = B.isSolid || isVoxelSet(B.lo, B.hi, lx, lyInner + 1, lz);
                            } else {
                                outerOcc = outerOccupied(B.bx, B.by + 1, B.bz, lx, 0, lz);
                            }
                            if (!outerOcc) {
                                mask[(B.bx * 4 + lx) + (B.bz * 4 + lz) * U] = 1;
                                any = true;
                            }
                        }
                    }
                }
                if (!any) continue;
                meshSlice(mask, U, V, (u0, v0, u1, v1) => {
                    const y = originY + s * voxelResolution;
                    const x0 = originX + u0 * voxelResolution;
                    const x1 = originX + u1 * voxelResolution;
                    const z0 = originZ + v0 * voxelResolution;
                    const z1 = originZ + v1 * voxelResolution;
                    // +Y normal: (x0,z0) -> (x0,z1) -> (x1,z1) -> (x1,z0)
                    emitQuad(
                        x0, y, z0,
                        x0, y, z1,
                        x1, y, z1,
                        x1, y, z0
                    );
                });
                mask.fill(0);
            }
        }

        // -Y faces
        for (const byInner of sortedBy) {
            for (let lyInner = 0; lyInner < 4; lyInner++) {
                const s = byInner * 4 + lyInner;
                let any = false;
                for (const B of blocksByBy.get(byInner)!) {
                    for (let lz = 0; lz < 4; lz++) {
                        for (let lx = 0; lx < 4; lx++) {
                            const innerOcc = B.isSolid || isVoxelSet(B.lo, B.hi, lx, lyInner, lz);
                            if (!innerOcc) continue;
                            let outerOcc: boolean;
                            if (lyInner > 0) {
                                outerOcc = B.isSolid || isVoxelSet(B.lo, B.hi, lx, lyInner - 1, lz);
                            } else {
                                outerOcc = outerOccupied(B.bx, B.by - 1, B.bz, lx, 3, lz);
                            }
                            if (!outerOcc) {
                                mask[(B.bx * 4 + lx) + (B.bz * 4 + lz) * U] = 1;
                                any = true;
                            }
                        }
                    }
                }
                if (!any) continue;
                meshSlice(mask, U, V, (u0, v0, u1, v1) => {
                    const y = originY + s * voxelResolution;
                    const x0 = originX + u0 * voxelResolution;
                    const x1 = originX + u1 * voxelResolution;
                    const z0 = originZ + v0 * voxelResolution;
                    const z1 = originZ + v1 * voxelResolution;
                    // -Y normal: reverse winding
                    emitQuad(
                        x0, y, z0,
                        x1, y, z0,
                        x1, y, z1,
                        x0, y, z1
                    );
                });
                mask.fill(0);
            }
        }
    }

    // ============ Z axis ============
    {
        const U = Vx, V = Vy;
        const mask = new Uint8Array(U * V);
        const sortedBz: number[] = [];
        blocksByBz.forEach((_v, k) => sortedBz.push(k));
        sortedBz.sort((a, b) => a - b);

        // +Z faces
        for (const bzInner of sortedBz) {
            for (let lzInner = 0; lzInner < 4; lzInner++) {
                const s = bzInner * 4 + lzInner + 1;
                if (s > Vz) continue;
                let any = false;
                for (const B of blocksByBz.get(bzInner)!) {
                    for (let ly = 0; ly < 4; ly++) {
                        for (let lx = 0; lx < 4; lx++) {
                            const innerOcc = B.isSolid || isVoxelSet(B.lo, B.hi, lx, ly, lzInner);
                            if (!innerOcc) continue;
                            let outerOcc: boolean;
                            if (lzInner < 3) {
                                outerOcc = B.isSolid || isVoxelSet(B.lo, B.hi, lx, ly, lzInner + 1);
                            } else {
                                outerOcc = outerOccupied(B.bx, B.by, B.bz + 1, lx, ly, 0);
                            }
                            if (!outerOcc) {
                                mask[(B.bx * 4 + lx) + (B.by * 4 + ly) * U] = 1;
                                any = true;
                            }
                        }
                    }
                }
                if (!any) continue;
                meshSlice(mask, U, V, (u0, v0, u1, v1) => {
                    const z = originZ + s * voxelResolution;
                    const x0 = originX + u0 * voxelResolution;
                    const x1 = originX + u1 * voxelResolution;
                    const y0 = originY + v0 * voxelResolution;
                    const y1 = originY + v1 * voxelResolution;
                    // +Z normal: (x0,y0) -> (x1,y0) -> (x1,y1) -> (x0,y1)
                    emitQuad(
                        x0, y0, z,
                        x1, y0, z,
                        x1, y1, z,
                        x0, y1, z
                    );
                });
                mask.fill(0);
            }
        }

        // -Z faces
        for (const bzInner of sortedBz) {
            for (let lzInner = 0; lzInner < 4; lzInner++) {
                const s = bzInner * 4 + lzInner;
                let any = false;
                for (const B of blocksByBz.get(bzInner)!) {
                    for (let ly = 0; ly < 4; ly++) {
                        for (let lx = 0; lx < 4; lx++) {
                            const innerOcc = B.isSolid || isVoxelSet(B.lo, B.hi, lx, ly, lzInner);
                            if (!innerOcc) continue;
                            let outerOcc: boolean;
                            if (lzInner > 0) {
                                outerOcc = B.isSolid || isVoxelSet(B.lo, B.hi, lx, ly, lzInner - 1);
                            } else {
                                outerOcc = outerOccupied(B.bx, B.by, B.bz - 1, lx, ly, 3);
                            }
                            if (!outerOcc) {
                                mask[(B.bx * 4 + lx) + (B.by * 4 + ly) * U] = 1;
                                any = true;
                            }
                        }
                    }
                }
                if (!any) continue;
                meshSlice(mask, U, V, (u0, v0, u1, v1) => {
                    const z = originZ + s * voxelResolution;
                    const x0 = originX + u0 * voxelResolution;
                    const x1 = originX + u1 * voxelResolution;
                    const y0 = originY + v0 * voxelResolution;
                    const y1 = originY + v1 * voxelResolution;
                    // -Z normal: reverse winding
                    emitQuad(
                        x0, y0, z,
                        x0, y1, z,
                        x1, y1, z,
                        x1, y0, z
                    );
                });
                mask.fill(0);
            }
        }
    }

    return {
        positions: positions.subarray(0, posLen),
        indices: indices.subarray(0, idxLen)
    };
};

export { greedyVoxelMesh };
