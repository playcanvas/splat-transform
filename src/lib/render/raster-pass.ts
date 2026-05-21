import { GraphicsDevice } from 'playcanvas';

import { type RenderCamera, buildCameraBasis } from './camera';
import { FAR_PLANE_NEAR_FACTOR, TILE_SIZE } from './config';
import {
    SortScratch,
    getSplatColumnRefs,
    packChunkInput,
    sceneSHBands,
    sortCandidatesByDepth,
    splatInputStride
} from './preprocess';
import { DataTable } from '../data-table';
import { GpuSplatRasterizer } from '../gpu';
import { logger } from '../utils';

/**
 * Max gaussians per GPU dispatch. Bounds per-render input/projection
 * buffer sizes and the granularity of the depth-ordered chunked path.
 *
 * Each chunk dispatches its own project + rasterize-accumulate.
 * Running state persists across chunks; saturated pixels (T < 1e-4)
 * short-circuit on subsequent chunks via the rasterize shader's T
 * early-out.
 */
const CHUNK_CAP = 200_000;

interface BackgroundRGBA {
    r: number;
    g: number;
    b: number;
    a: number;
}

/**
 * Render a splat scene to an RGBA byte buffer.
 *
 * Whole-image pipeline:
 *
 *   1. Linear frustum cull — one pass over all splats, testing each
 *      centre against the camera frustum in camera-space. No BVH or
 *      per-splat AABB precomputation.
 *   2. Sort visible splats globally by camera-space z (front-to-back).
 *   3. Drive the rasterizer with one tile group covering the entire
 *      image, chunking the depth-sorted splat list into `CHUNK_CAP`-
 *      sized batches. The rasterizer's running-state buffer accumulates
 *      across chunks; `T < MIN_TRANSMITTANCE` short-circuits saturated
 *      pixels.
 *
 * Per-chunk (GPU tile-bin pipeline, fully GPU-resident): project (writes
 * per-splat tile coverage) → prefix-sum (writes emitOffsets + totalPairs)
 * → emit-pairs → prepare-indirect → radix sortIndirect → init-tile-offsets
 * → find-boundaries (atomicMin) → rasterize each tile's slice in depth
 * order. No per-chunk CPU readbacks; no per-tile-group seams.
 *
 * @param device - PlayCanvas WebGPU graphics device.
 * @param dataTable - Gaussian splat data in PlayCanvas-identity space.
 * @param camera - Camera parameters.
 * @param background - RGBA background composited under residual transmittance.
 * @returns RGBA byte array of length `camera.width × camera.height × 4`.
 */
const renderRasterPass = async (
    device: GraphicsDevice,
    dataTable: DataTable,
    camera: RenderCamera,
    background: BackgroundRGBA
): Promise<Uint8Array> => {
    if (!Number.isInteger(camera.width) || !Number.isInteger(camera.height) ||
        camera.width <= 0 || camera.height <= 0) {
        throw new Error(`Invalid resolution: ${camera.width}x${camera.height}`);
    }

    const width = camera.width;
    const height = camera.height;
    const basis = buildCameraBasis(camera);

    // ---- Frustum cull ----
    // Linear, centre-only near-plane test in camera space. For each splat:
    //   - cz = forward · (p - eye)
    //   - cull if cz <= near (matches the GPU project shader's near-plane
    //     test exactly)
    //
    // We deliberately don't test the side cones on the CPU. A proper
    // AABB-vs-cone test needs `exp(max(s))` per splat — on an 18 M scene
    // that's ~275 ms of extra CPU work, and at typical reference-render
    // FOVs (60–90°) the cone contains nearly every front-of-camera splat
    // anyway (measured 95.8 % retention on windmill at 90°). The splats
    // that the cone test *would* drop are caught by the GPU project
    // shader's 2D image-rect test, which uses the actual screen-space
    // radius — strictly tighter than the L∞-bound CPU test.
    //
    // If a future use case renders at narrow FOV (≤ 30°) where cone-
    // outside splats dominate, switching to an AABB-vs-cone test here
    // could pay for itself.
    const cullGroup = logger.group('Cull');
    const xCol = dataTable.getColumnByName('x')!.data as Float32Array;
    const yCol = dataTable.getColumnByName('y')!.data as Float32Array;
    const zCol = dataTable.getColumnByName('z')!.data as Float32Array;
    const numRows = dataTable.numRows;

    const ex = basis.eye.x, ey = basis.eye.y, ez = basis.eye.z;
    const fx = basis.forward.x, fy = basis.forward.y, fz = basis.forward.z;
    const near = camera.near;

    // Worst-case visible-count allocation. Right-sized at the end via subarray.
    const candidates = new Uint32Array(numRows);
    let candidateCount = 0;
    for (let i = 0; i < numRows; i++) {
        const cz = fx * (xCol[i] - ex) + fy * (yCol[i] - ey) + fz * (zCol[i] - ez);
        if (cz > near) candidates[candidateCount++] = i;
    }
    cullGroup.end();
    void FAR_PLANE_NEAR_FACTOR;  // kept in config for future use

    // ---- Image tile grid ----
    const imageTilesX = Math.ceil(width / TILE_SIZE);
    const imageTilesY = Math.ceil(height / TILE_SIZE);

    // ---- Global depth sort ----
    const numSHBands = sceneSHBands(dataTable);
    const inputStride = splatInputStride(numSHBands);
    const cols = getSplatColumnRefs(dataTable, numSHBands);
    const sortScratch = new SortScratch();
    sortCandidatesByDepth(cols, candidates, candidateCount, basis, sortScratch);

    // ---- Rasterizer (single group covering whole image) ----
    // The existing rasterizer accumulates depth-sorted splats into a
    // persistent running-state buffer across chunks within a group.
    // We use ONE group covering the whole image, so the running state
    // is the full output image and there are no inter-group seams.
    //
    // The rasterizer dispatches a square `groupSizeTiles × groupSizeTiles`
    // grid of workgroups; tiles outside the actual image's tile grid
    // early-out via `wgId.x >= groupTilesX || wgId.y >= groupTilesY`. We
    // size the square to the larger image axis.
    const groupSizeTiles = Math.max(imageTilesX, imageTilesY);
    const effectiveChunkCap = Math.max(1, Math.min(CHUNK_CAP, candidateCount));

    const rasterizer = new GpuSplatRasterizer(device, {
        numSHBands,
        groupSizeTiles,
        chunkCap: effectiveChunkCap,
        numSlots: 1,
        imageWidth: width,
        imageHeight: height,
        near: camera.near,
        rightX: basis.right.x, rightY: basis.right.y, rightZ: basis.right.z,
        downX: basis.down.x, downY: basis.down.y, downZ: basis.down.z,
        forwardX: basis.forward.x, forwardY: basis.forward.y, forwardZ: basis.forward.z,
        eyeX: basis.eye.x, eyeY: basis.eye.y, eyeZ: basis.eye.z,
        focalX: basis.focalX, focalY: basis.focalY,
        bgR: background.r, bgG: background.g, bgB: background.b, bgA: background.a
    });

    // Per-chunk CPU scratch.
    const chunkInput = new Float32Array(effectiveChunkCap * inputStride);

    // Single group at origin (0, 0), spanning the whole image tile grid.
    rasterizer.beginGroup(0, 0, 0, imageTilesX, imageTilesY);

    const rasterBar = logger.bar('rasterizing', Math.max(1, Math.ceil(candidateCount / effectiveChunkCap)));
    let completed = 0;
    for (let chunkStart = 0; chunkStart < candidateCount; chunkStart += effectiveChunkCap) {
        const chunkSize = Math.min(effectiveChunkCap, candidateCount - chunkStart);
        packChunkInput(cols, candidates, chunkStart, chunkSize, numSHBands, chunkInput);
        rasterizer.dispatchChunk(0, chunkInput, chunkSize);
        rasterBar.update(++completed);
    }
    rasterBar.end();

    const rawBytes = await rasterizer.finishGroup(0, imageTilesX, imageTilesY);

    rasterizer.destroy();

    // The rasterizer writes its group output at a stride of
    // `groupTilesX × TILE_SIZE` (= image width here, since we span the
    // image in tile units). The buffer's row stride matches the image's,
    // so we can return it directly when it's exactly image-sized. If the
    // image width isn't a multiple of TILE_SIZE the buffer rows are wider
    // than the image rows, and we need to repack.
    const groupPixelW = imageTilesX * TILE_SIZE;
    if (groupPixelW === width) {
        // Common case for image dimensions that are multiples of TILE_SIZE.
        const out = rawBytes.subarray(0, width * height * 4);
        // Make a copy so the returned buffer's lifetime is independent
        // of the GPU readback's typed array.
        return new Uint8Array(out);
    }

    const finalImage = new Uint8Array(width * height * 4);
    for (let row = 0; row < height; row++) {
        const srcOffset = row * groupPixelW * 4;
        const dstOffset = row * width * 4;
        finalImage.set(rawBytes.subarray(srcOffset, srcOffset + width * 4), dstOffset);
    }
    return finalImage;
};

export { renderRasterPass, CHUNK_CAP };
