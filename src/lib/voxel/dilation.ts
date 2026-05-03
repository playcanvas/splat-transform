import {
    BLOCK_EMPTY,
    BLOCK_SOLID,
    SOLID_HI,
    SOLID_LO,
    SOLID_WORD,
    SparseVoxelGrid,
    readBlockType
} from './sparse-voxel-grid';
import { GpuDilation } from '../gpu';
import { logger } from '../utils';

// ============================================================================
// GPU Dilation
//
// Chunked GPU separable 3D dilation that returns a fresh `SparseVoxelGrid`.
// Extracts each chunk (with halo) from the source as a dense bit grid, runs
// X/Z/Y compute passes on the GPU without a CPU round-trip between them,
// then OR's the inner-only output back into the destination. Source is
// read-only across the whole call; destination is built up chunk by chunk.
// ============================================================================

/** Inner chunk size in voxels per axis (must be a multiple of 4). */
const CHUNK_INNER = 512;

/**
 * Fast empty-chunk check. Scans only `types` (no mask reads) for the source
 * blocks that overlap the chunk's outer region; returns true if every block
 * is `BLOCK_EMPTY`. Lets `gpuDilate3` skip extract / dispatch / insert for
 * chunks far from the scene's occupied region.
 * @param src
 * @param ox
 * @param oy
 * @param oz
 * @param cx
 * @param cy
 * @param cz
 */
function chunkIsEmpty(
    src: SparseVoxelGrid,
    ox: number, oy: number, oz: number,
    cx: number, cy: number, cz: number
): boolean {
    const minBx = Math.max(0, Math.floor(ox / 4));
    const minBy = Math.max(0, Math.floor(oy / 4));
    const minBz = Math.max(0, Math.floor(oz / 4));
    const maxBx = Math.min(src.nbx, Math.ceil((ox + cx) / 4));
    const maxBy = Math.min(src.nby, Math.ceil((oy + cy) / 4));
    const maxBz = Math.min(src.nbz, Math.ceil((oz + cz) / 4));
    if (maxBx <= minBx || maxBy <= minBy || maxBz <= minBz) return true;

    const types = src.types;
    for (let bz = minBz; bz < maxBz; bz++) {
        for (let by = minBy; by < maxBy; by++) {
            for (let bx = minBx; bx < maxBx; bx++) {
                const blockIdx = bx + by * src.nbx + bz * src.bStride;
                if (readBlockType(types, blockIdx) !== BLOCK_EMPTY) {
                    return false;
                }
            }
        }
    }
    return true;
}

/**
 * Fast saturated-chunk check. Returns true if the chunk's outer region is
 * entirely within `src` bounds AND every overlapping block is `BLOCK_SOLID`
 * (so extract would produce all 1s). Dilation of an all-solid input is
 * trivially all-solid, letting the caller skip GPU dispatch and write
 * `BLOCK_SOLID` directly into the destination's inner region.
 * @param src
 * @param ox
 * @param oy
 * @param oz
 * @param cx
 * @param cy
 * @param cz
 */
function chunkIsSaturated(
    src: SparseVoxelGrid,
    ox: number, oy: number, oz: number,
    cx: number, cy: number, cz: number
): boolean {
    // Halo extends past grid → those bits are 0, not saturated.
    if (ox < 0 || oy < 0 || oz < 0) return false;
    if (ox + cx > src.nx || oy + cy > src.ny || oz + cz > src.nz) return false;

    const minBx = ox >> 2;
    const minBy = oy >> 2;
    const minBz = oz >> 2;
    const maxBx = (ox + cx + 3) >> 2;
    const maxBy = (oy + cy + 3) >> 2;
    const maxBz = (oz + cz + 3) >> 2;

    const types = src.types;
    for (let bz = minBz; bz < maxBz; bz++) {
        for (let by = minBy; by < maxBy; by++) {
            for (let bx = minBx; bx < maxBx; bx++) {
                const blockIdx = bx + by * src.nbx + bz * src.bStride;
                if (readBlockType(types, blockIdx) !== BLOCK_SOLID) {
                    return false;
                }
            }
        }
    }
    return true;
}

/**
 * Insert a fully-solid inner region into `dst` without going through the
 * dense path. Used when the source chunk is saturated and dilation is
 * trivially saturated.
 *
 * For each X-row of inner blocks (varying `bx`, fixed `by, bz`), the
 * `dst.types` slots are contiguous; we set 2 bits per block to `BLOCK_SOLID`
 * via word-level writes (`SOLID_WORD = 0x55555555`) instead of millions of
 * `orBlock` calls. Chunks here are always disjoint and dst blocks empty
 * before this call, so no merge logic is needed.
 * @param dst
 * @param innerOx
 * @param innerOy
 * @param innerOz
 * @param innerCx
 * @param innerCy
 * @param innerCz
 */
function insertSaturatedInner(
    dst: SparseVoxelGrid,
    innerOx: number, innerOy: number, innerOz: number,
    innerCx: number, innerCy: number, innerCz: number
): void {
    const minBx = Math.max(0, innerOx >> 2);
    const minBy = Math.max(0, innerOy >> 2);
    const minBz = Math.max(0, innerOz >> 2);
    const maxBx = Math.min(dst.nbx, (innerOx + innerCx + 3) >> 2);
    const maxBy = Math.min(dst.nby, (innerOy + innerCy + 3) >> 2);
    const maxBz = Math.min(dst.nbz, (innerOz + innerCz + 3) >> 2);

    const types = dst.types;
    const nbx = dst.nbx;
    const bStride = dst.bStride;

    for (let bz = minBz; bz < maxBz; bz++) {
        for (let by = minBy; by < maxBy; by++) {
            const rowBase = by * nbx + bz * bStride;
            const startIdx = rowBase + minBx;
            const endIdx = rowBase + maxBx; // exclusive
            let blockIdx = startIdx;
            while (blockIdx < endIdx) {
                const w = blockIdx >>> 4;
                const shift = (blockIdx & 15) << 1;
                const remainingInWord = 16 - (blockIdx & 15);
                const remainingInRow = endIdx - blockIdx;
                const blocksToWrite = remainingInWord < remainingInRow ? remainingInWord : remainingInRow;
                if (blocksToWrite === 16) {
                    types[w] = SOLID_WORD;
                } else {
                    const bits = blocksToWrite << 1;
                    const mask = (((1 << bits) - 1) >>> 0) << shift;
                    types[w] = ((types[w] & ~mask) | (SOLID_WORD & mask)) >>> 0;
                }
                blockIdx += blocksToWrite;
            }
        }
    }
}

/**
 * Apply a chunk's GPU-produced output (`typesOut` + `masksOut`) to `dst`.
 * Iterates inner blocks; for each, reads the precomputed type and (if MIXED)
 * the precomputed `lo`/`hi`, then writes directly into `dst.types` and
 * `dst.masks`. Replaces the dense-bit-reading hot loop with O(blocks) work
 * dominated by hash inserts for mixed blocks.
 * @param dst
 * @param typesOut
 * @param masksOut
 * @param cx
 * @param cy
 * @param cz
 * @param innerNx
 * @param innerNy
 * @param innerNz
 */
function applyChunkToDst(
    dst: SparseVoxelGrid,
    typesOut: Uint32Array,
    masksOut: Uint32Array,
    cx: number, cy: number, cz: number,
    innerNx: number, innerNy: number, innerNz: number
): void {
    const innerBx = innerNx >> 2;
    const innerBy = innerNy >> 2;
    const innerBz = innerNz >> 2;
    const dstNbx = dst.nbx;
    const dstBStride = dst.bStride;
    const dstTypes = dst.types;
    const dstMasks = dst.masks;

    const baseBx = cx >> 2;
    const baseBy = cy >> 2;
    const baseBz = cz >> 2;

    let innerIdx = 0;
    for (let bz = 0; bz < innerBz; bz++) {
        const globalBz = baseBz + bz;
        for (let by = 0; by < innerBy; by++) {
            const globalBy = baseBy + by;
            const baseGlobalIdx = baseBx + globalBy * dstNbx + globalBz * dstBStride;
            for (let bx = 0; bx < innerBx; bx++, innerIdx++) {
                const wordIdx = innerIdx >>> 4;
                const bitShift = (innerIdx & 15) << 1;
                const bt = (typesOut[wordIdx] >>> bitShift) & 3;
                if (bt === 0) continue;  // EMPTY

                const globalBlockIdx = baseGlobalIdx + bx;
                const w = globalBlockIdx >>> 4;
                const shift = (globalBlockIdx & 15) << 1;
                dstTypes[w] = ((dstTypes[w] & ~(3 << shift)) | (bt << shift)) >>> 0;

                if (bt === 2) {  // MIXED
                    const m2 = innerIdx * 2;
                    dstMasks.set(globalBlockIdx, masksOut[m2], masksOut[m2 + 1]);
                }
            }
        }
    }
}

/**
 * GPU separable 3D dilation. Chunks the grid into ~1024³ inner regions plus
 * a halo on each side, runs three GPU passes per chunk, and OR's the
 * dilated inner region into a fresh destination `SparseVoxelGrid`.
 *
 * Both `halfExtentXZ` and `halfExtentY` must be 0 or a multiple of 4 — the
 * sparse-chunk math requires the halo to be block-aligned. Callers that
 * derive the half-extent from a real-world radius should round up to the
 * nearest multiple of 4 voxels (the runtime cost is negligible vs. the
 * extra dilation distance).
 *
 * @param gpu - Reusable GPU dilation context (compiled shader + buffers).
 * @param src - Input sparse grid (read-only across the call).
 * @param halfExtentXZ - Dilation half-extent in voxels along X and Z.
 * Must be 0 or a multiple of 4.
 * @param halfExtentY - Dilation half-extent in voxels along Y.
 * Must be 0 or a multiple of 4.
 * @returns Newly allocated dilated sparse grid.
 */
async function gpuDilate3(
    gpu: GpuDilation,
    src: SparseVoxelGrid,
    halfExtentXZ: number,
    halfExtentY: number
): Promise<SparseVoxelGrid> {
    const dst = new SparseVoxelGrid(src.nx, src.ny, src.nz);

    if (halfExtentXZ % 4 !== 0 && halfExtentXZ !== 0) {
        // Halo must be block-aligned for the sparse path's chunk math.
        // Current callers (fill-floor 32, carve XZ 4) satisfy this; assert.
        throw new Error(`gpuDilate3: halfExtentXZ=${halfExtentXZ} must be a multiple of 4`);
    }
    if (halfExtentY % 4 !== 0 && halfExtentY !== 0) {
        throw new Error(`gpuDilate3: halfExtentY=${halfExtentY} must be a multiple of 4`);
    }

    const haloX = halfExtentXZ;
    const haloY = halfExtentY;
    const haloZ = halfExtentXZ;
    const haloBx = haloX >> 2;
    const haloBy = haloY >> 2;
    const haloBz = haloZ >> 2;

    // Round inner chunk down to multiple of 4 (block alignment).
    const innerStep = CHUNK_INNER & ~3;

    const numChunksX = Math.ceil(src.nx / innerStep);
    const numChunksY = Math.ceil(src.ny / innerStep);
    const numChunksZ = Math.ceil(src.nz / innerStep);
    const totalChunks = numChunksX * numChunksY * numChunksZ;

    interface InFlight {
        typesPromise: Promise<Uint32Array>;
        masksPromise: Promise<Uint32Array>;
        cx: number; cy: number; cz: number;
        innerNx: number; innerNy: number; innerNz: number;
    }

    let currentSlot = 0;
    let inflight: InFlight | null = null;

    const drainInflight = async (): Promise<void> => {
        if (!inflight) return;
        const f = inflight;
        inflight = null;
        const [typesOut, masksOut] = await Promise.all([f.typesPromise, f.masksPromise]);
        applyChunkToDst(dst, typesOut, masksOut, f.cx, f.cy, f.cz, f.innerNx, f.innerNy, f.innerNz);
    };

    gpu.uploadSrc(src);

    const bar = logger.bar('Dilating', totalChunks);
    try {
        for (let cz = 0; cz < src.nz; cz += innerStep) {
            for (let cy = 0; cy < src.ny; cy += innerStep) {
                for (let cx = 0; cx < src.nx; cx += innerStep) {
                    const innerNx = Math.min(innerStep, src.nx - cx);
                    const innerNy = Math.min(innerStep, src.ny - cy);
                    const innerNz = Math.min(innerStep, src.nz - cz);

                    const ox = cx - haloX;
                    const oy = cy - haloY;
                    const oz = cz - haloZ;
                    const outerNx = innerNx + 2 * haloX;
                    const outerNy = innerNy + 2 * haloY;
                    const outerNz = innerNz + 2 * haloZ;

                    if (chunkIsEmpty(src, ox, oy, oz, outerNx, outerNy, outerNz)) {
                        bar.tick();
                        continue;
                    }
                    if (chunkIsSaturated(src, ox, oy, oz, outerNx, outerNy, outerNz)) {
                        insertSaturatedInner(dst, cx, cy, cz, innerNx, innerNy, innerNz);
                        bar.tick();
                        continue;
                    }

                    const innerBx = innerNx >> 2;
                    const innerBy = innerNy >> 2;
                    const innerBz = innerNz >> 2;
                    const outerBx = outerNx >> 2;
                    const outerBy = outerNy >> 2;
                    const outerBz = outerNz >> 2;
                    const minBx = ox >> 2;
                    const minBy = oy >> 2;
                    const minBz = oz >> 2;

                    const { types: typesPromise, masks: masksPromise } = gpu.submitChunkSparse(
                        currentSlot,
                        minBx, minBy, minBz,
                        outerBx, outerBy, outerBz,
                        haloBx, haloBy, haloBz,
                        innerBx, innerBy, innerBz,
                        halfExtentXZ, halfExtentY
                    );

                    if (inflight) {
                        await drainInflight();
                    }

                    inflight = {
                        typesPromise,
                        masksPromise,
                        cx,
                        cy,
                        cz,
                        innerNx,
                        innerNy,
                        innerNz
                    };
                    currentSlot = (currentSlot + 1) % GpuDilation.NUM_SLOTS;

                    bar.tick();
                }
            }
        }

        if (inflight) {
            await drainInflight();
        }
    } finally {
        bar.end();
        gpu.releaseSrc();
    }
    return dst;
}

export { gpuDilate3 };
