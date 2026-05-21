import { GraphicsDevice } from 'playcanvas';

import { type RenderCamera, buildCameraBasis } from './camera';
import {
    PAIR_BUFFER_BUDGET_BYTES,
    PAIR_BUFFER_TOTAL_BYTES_PER_ELEMENT,
    TILE_SIZE
} from './config';
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

/**
 * Maximum sub-frame dimensions, in tiles. Renders larger than this get
 * split into a grid of independent sub-frames stitched together at the
 * end. At 1080p (120×68 tiles) this is exactly one sub-frame; at 4K it
 * becomes 2×2; at 8K it becomes 4×4.
 *
 * Why split: per-splat tile coverage scales with image_height² (focal
 * length is proportional to image height for a fixed FOV). At 8K a
 * splat that projects to ~few tiles at 1080p projects to ~hundreds of
 * tiles, which would either need a huge pair buffer or produce hard
 * truncation edges when clamped. Sub-frame rendering keeps each
 * sub-frame's working set ≤ 1080p-sized, so the per-splat coverage
 * stays in the same regime as the (well-tested) 1080p path. The
 * project shader's group-AABB cull skips splats outside the current
 * sub-frame, and the global CPU depth sort is shared across sub-frames
 * (no seam artifacts at sub-frame boundaries).
 */
const MAX_SUB_FRAME_TILES_X = Math.ceil(1920 / TILE_SIZE);  // 120
const MAX_SUB_FRAME_TILES_Y = Math.ceil(1080 / TILE_SIZE);  // 68

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
 *   3. Split image into a grid of sub-frames (each ≤ MAX_SUB_FRAME_TILES
 *      in either dimension) and render each independently. The rasterizer
 *      retains its per-sub-frame running state; the project shader's
 *      group-AABB cull skips splats outside the current sub-frame. The
 *      CPU depth sort is global, so depth ordering is consistent across
 *      sub-frames and there are no boundary seams.
 *
 * Per-chunk (GPU tile-bin pipeline, fully GPU-resident): project (writes
 * per-splat tile coverage) → prefix-sum (writes emitOffsets + totalPairs)
 * → emit-pairs → prepare-indirect → radix sortIndirect (key + value:
 * tile keys sorted, splat indices reordered) → init-tile-offsets →
 * find-boundaries (atomicMin) → rasterize each tile's slice in depth
 * order. No per-chunk CPU readbacks.
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

    // ---- Image tile grid + sub-frame partition ----
    const imageTilesX = Math.ceil(width / TILE_SIZE);
    const imageTilesY = Math.ceil(height / TILE_SIZE);
    const subFrameTilesX = Math.min(imageTilesX, MAX_SUB_FRAME_TILES_X);
    const subFrameTilesY = Math.min(imageTilesY, MAX_SUB_FRAME_TILES_Y);
    const numSubFramesX = Math.ceil(imageTilesX / subFrameTilesX);
    const numSubFramesY = Math.ceil(imageTilesY / subFrameTilesY);
    const numSubFrames = numSubFramesX * numSubFramesY;

    // ---- Global depth sort ----
    const numSHBands = sceneSHBands(dataTable);
    const inputStride = splatInputStride(numSHBands);
    const cols = getSplatColumnRefs(dataTable, numSHBands);
    const sortScratch = new SortScratch();
    sortCandidatesByDepth(cols, candidates, candidateCount, basis, sortScratch);

    // ---- Rasterizer (one instance, reused across sub-frames) ----
    // The rasterizer's buffers are sized to the MAX sub-frame
    // dimensions, not the full image. Each sub-frame calls beginGroup
    // with its actual dimensions (≤ max) and gets a clean running state.
    //
    // maxCoveragePerSplat = sub-frame tile area. Then every splat's
    // bbox-in-sub-frame ≤ maxCoveragePerSplat (since coverage is
    // geometrically bounded by the sub-frame's area), so the project
    // shader's coverage clamp never bites and emit-pairs writes the
    // entire bbox-in-sub-frame. No truncation → no hard tile edges, no
    // boundary seams. The cost is pair-buffer memory and chunkCap.
    const subFrameTiles = subFrameTilesX * subFrameTilesY;
    const maxCoveragePerSplat = 1 << Math.ceil(Math.log2(Math.max(1, subFrameTiles)));

    // chunkCap is bounded by two GPU constraints:
    //  (a) each individual pair buffer (tileKeys, splatValues, and the
    //      four radix-sort internal ping-pong buffers) must fit inside
    //      a single storage-buffer binding — capped at
    //      `maxStorageBufferBindingSize` (typically 128 MiB baseline,
    //      ~1 GiB on Apple Silicon, larger on desktop discrete GPUs).
    //  (b) the TOTAL of all six pair-sized buffers combined must stay
    //      within `PAIR_BUFFER_BUDGET_BYTES` — bounds the rasterizer's
    //      peak GPU footprint when chunkCap could otherwise saturate
    //      the binding limit on adapters with very large bindings.
    // @ts-ignore - limits is exposed by WebgpuGraphicsDevice
    const wgpuLimits = (device as { limits?: { maxStorageBufferBindingSize?: number } }).limits;
    const maxBindingBytes = wgpuLimits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
    const bindingChunkCap = Math.floor(maxBindingBytes / (maxCoveragePerSplat * 4));
    const budgetChunkCap = Math.floor(PAIR_BUFFER_BUDGET_BYTES / (maxCoveragePerSplat * PAIR_BUFFER_TOTAL_BYTES_PER_ELEMENT));
    const effectiveChunkCap = Math.max(
        1,
        Math.min(CHUNK_CAP, budgetChunkCap, bindingChunkCap, candidateCount)
    );

    const rasterizer = new GpuSplatRasterizer(device, {
        numSHBands,
        groupTilesX: subFrameTilesX,
        groupTilesY: subFrameTilesY,
        chunkCap: effectiveChunkCap,
        maxCoveragePerSplat,
        imageWidth: width,
        imageHeight: height,
        near: camera.near,
        rightX: basis.right.x,
        rightY: basis.right.y,
        rightZ: basis.right.z,
        downX: basis.down.x,
        downY: basis.down.y,
        downZ: basis.down.z,
        forwardX: basis.forward.x,
        forwardY: basis.forward.y,
        forwardZ: basis.forward.z,
        eyeX: basis.eye.x,
        eyeY: basis.eye.y,
        eyeZ: basis.eye.z,
        focalX: basis.focalX,
        focalY: basis.focalY,
        bgR: background.r,
        bgG: background.g,
        bgB: background.b,
        bgA: background.a
    });

    // Per-chunk CPU scratch.
    const chunkInput = new Float32Array(effectiveChunkCap * inputStride);

    // Final image buffer (image-sized) that each sub-frame's output is
    // copied into.
    const finalImage = new Uint8Array(width * height * 4);

    const numChunks = Math.max(1, Math.ceil(candidateCount / effectiveChunkCap));
    const rasterBar = logger.bar('rasterizing', numSubFrames * numChunks);
    let completed = 0;

    for (let sy = 0; sy < numSubFramesY; sy++) {
        for (let sx = 0; sx < numSubFramesX; sx++) {
            const tilesX = Math.min(subFrameTilesX, imageTilesX - sx * subFrameTilesX);
            const tilesY = Math.min(subFrameTilesY, imageTilesY - sy * subFrameTilesY);

            rasterizer.beginGroup(sx, sy, tilesX, tilesY);

            for (let chunkStart = 0; chunkStart < candidateCount; chunkStart += effectiveChunkCap) {
                const chunkSize = Math.min(effectiveChunkCap, candidateCount - chunkStart);
                packChunkInput(cols, candidates, chunkStart, chunkSize, numSHBands, chunkInput);
                rasterizer.dispatchChunk(chunkInput, chunkSize);
                rasterBar.update(++completed);
            }

            const subFrameBytes = await rasterizer.finishGroup();

            // Copy the sub-frame's pixels into the final image. The
            // rasterizer's output buffer row stride is the sub-frame's
            // pixel width (`tilesX × TILE_SIZE`), not the image width;
            // copy row-by-row into the right offset.
            const subPixelOriginX = sx * subFrameTilesX * TILE_SIZE;
            const subPixelOriginY = sy * subFrameTilesY * TILE_SIZE;
            const subPixelW = tilesX * TILE_SIZE;
            const subPixelH = tilesY * TILE_SIZE;
            const copyW = Math.min(subPixelW, width - subPixelOriginX);
            const copyH = Math.min(subPixelH, height - subPixelOriginY);
            for (let row = 0; row < copyH; row++) {
                const srcOffset = row * subPixelW * 4;
                const dstOffset = ((subPixelOriginY + row) * width + subPixelOriginX) * 4;
                finalImage.set(
                    subFrameBytes.subarray(srcOffset, srcOffset + copyW * 4),
                    dstOffset
                );
            }
        }
    }
    rasterBar.end();

    rasterizer.destroy();

    return finalImage;
};

export { renderRasterPass, CHUNK_CAP };
