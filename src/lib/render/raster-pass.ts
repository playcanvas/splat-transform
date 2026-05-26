import { GraphicsDevice } from 'playcanvas';

import { type Projection, type RenderCamera, buildCameraBasis } from './camera';
import {
    AA_DILATION_COV,
    DISCRIMINANT_FLOOR,
    JACOBIAN_LIMIT_FACTOR,
    PAIR_BUFFER_BUDGET_BYTES,
    PAIR_BUFFER_TOTAL_BYTES_PER_ELEMENT,
    SIGMA_CUTOFF,
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
    // Default to pinhole when projection is omitted — matches the
    // back-compat treatment in buildCameraBasis. All downstream branches
    // (CPU cull/sort, sub-frame split, GPU rasterizer) read this local.
    const projection: Projection = camera.projection ?? 'pinhole';
    const basis = buildCameraBasis(camera);

    // ---- Frustum cull ----
    // Linear, centre-only near-plane test. Pinhole tests cz > near (the
    // GPU project shader's exact near-plane condition). Equirect has no
    // forward direction in the cull sense — instead we cull a sphere of
    // radius `near` around the camera, matching the GPU project shader's
    // `r > near` test. In both modes we deliberately don't test the side
    // cones on the CPU. A proper AABB-vs-cone test needs `exp(max(s))`
    // per splat — on an 18 M scene that's ~275 ms of extra CPU work,
    // and at typical reference-render FOVs (60–90°) the cone contains
    // nearly every front-of-camera splat anyway (measured 95.8 %
    // retention on windmill at 90°). The splats that the cone test
    // *would* drop are caught by the GPU project shader's 2D image-rect
    // test, which uses the actual screen-space radius — strictly
    // tighter than the L∞-bound CPU test. For equirect the cone test
    // doesn't apply at all: every direction is in-view.
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
    if (projection === 'pinhole') {
        for (let i = 0; i < numRows; i++) {
            const cz = fx * (xCol[i] - ex) + fy * (yCol[i] - ey) + fz * (zCol[i] - ez);
            if (cz > near) candidates[candidateCount++] = i;
        }
    } else {
        const nearSq = near * near;
        for (let i = 0; i < numRows; i++) {
            const dx = xCol[i] - ex;
            const dy = yCol[i] - ey;
            const dz = zCol[i] - ez;
            if (dx * dx + dy * dy + dz * dz > nearSq) candidates[candidateCount++] = i;
        }
    }
    cullGroup.end();

    // ---- Image tile grid + sub-frame partition ----
    // Pinhole renders larger than ~1080p are split into sub-frames so
    // per-splat tile coverage stays bounded (see MAX_SUB_FRAME_TILES_*).
    // Equirect runs as a single sub-frame regardless of resolution: the
    // sub-frame X-cull below relies on a perspective Jacobian bound
    // that doesn't apply to equirect, and the wrap-around tile coverage
    // assumes the X tile range covers the full longitude. TODO: derive
    // an equirect-specific sub-frame bound (Jacobian magnitude ≤
    // max(W/(2π), H/π · 1/POLE_EPS)) for >4K equirect renders.
    const imageTilesX = Math.ceil(width / TILE_SIZE);
    const imageTilesY = Math.ceil(height / TILE_SIZE);
    const subFrameTilesX = projection === 'equirect' ?
        imageTilesX :
        Math.min(imageTilesX, MAX_SUB_FRAME_TILES_X);
    const subFrameTilesY = projection === 'equirect' ?
        imageTilesY :
        Math.min(imageTilesY, MAX_SUB_FRAME_TILES_Y);
    const numSubFramesX = Math.ceil(imageTilesX / subFrameTilesX);
    const numSubFramesY = Math.ceil(imageTilesY / subFrameTilesY);
    const numSubFrames = numSubFramesX * numSubFramesY;

    // ---- Global depth sort ----
    const numSHBands = sceneSHBands(dataTable);
    const inputStride = splatInputStride(numSHBands);
    const cols = getSplatColumnRefs(dataTable, numSHBands);
    const sortScratch = new SortScratch();
    sortCandidatesByDepth(cols, candidates, candidateCount, basis, projection, sortScratch);

    // ---- Per-sub-frame CPU cull ----
    // For multi-sub-frame renders, partition the depth-sorted candidate
    // list into per-sub-frame sub-lists so each sub-frame only packs/
    // uploads/projects the splats whose conservative screen bbox
    // overlaps it. Without this, every sub-frame pays the full upload
    // + project cost for every candidate (the GPU AABB cull only
    // discards them at the end of projection) — at 8K that's 16×
    // duplicate work.
    //
    // Bound is constructed to be ≥ the GPU project shader's computed
    // radius, so the cull is byte-exact: every splat the GPU would
    // include sits inside its bound bbox, and a "definitely outside"
    // verdict means the GPU AABB cull would also drop it.
    //
    // Derivation. The GPU computes radius = SIGMA_CUTOFF · sqrt(lambdaMax)
    // where lambdaMax is the larger eigenvalue of the projected 2D
    // covariance (C2D = J·C3D·Jᵀ + AA_DILATION_COV·I, with the
    // DISCRIMINANT_FLOOR safety also bumping it). Matrix theory:
    //   maxEig(J·C3D·Jᵀ) ≤ maxEig(J·Jᵀ) · maxEig(C3D)
    // and for our perspective Jacobian J = focal/cz · [[1, 0, -tx],
    // [0, 1, -ty]]:
    //   maxEig(J·Jᵀ) = (focal/cz)² · (1 + tx² + ty²)
    // with tx, ty clamped per JACOBIAN_LIMIT_FACTOR (same as the GPU
    // project shader's clamp). maxEig(C3D) = max-axis-scale² for an
    // anisotropic Gaussian. Adding AA_DILATION_COV and the
    // DISCRIMINANT_FLOOR safety bump:
    //   lambdaMax ≤ (focal/cz)² · (1+tx²+ty²) · maxScale²
    //              + AA_DILATION_COV + sqrt(DISCRIMINANT_FLOOR)
    // Per-splat order is preserved → per-sub-frame lists stay
    // depth-sorted.
    let subFrameLists: Uint32Array[];
    if (numSubFrames === 1) {
        subFrameLists = [candidates.subarray(0, candidateCount)];
    } else {
        const subFramePixelsX = subFrameTilesX * TILE_SIZE;
        const subFramePixelsY = subFrameTilesY * TILE_SIZE;
        const rx2 = basis.right.x, ry2 = basis.right.y, rz2 = basis.right.z;
        const dx2 = basis.down.x, dy2 = basis.down.y, dz2 = basis.down.z;
        const focalX = basis.focalX, focalY = basis.focalY;
        const halfW = width * 0.5, halfH = height * 0.5;
        const sxColRef = cols.scaleX, syColRef = cols.scaleY, szColRef = cols.scaleZ;
        // Tan-of-half-FOV cap; matches the project shader's Jacobian clamp.
        const limX = JACOBIAN_LIMIT_FACTOR * halfW / focalX;
        const limY = JACOBIAN_LIMIT_FACTOR * halfH / focalY;
        const limX2 = limX * limX;
        const limY2 = limY * limY;
        // Additive squared-radius safety: AA dilation + disc-floor bump.
        const lambdaSafety = AA_DILATION_COV + Math.sqrt(DISCRIMINANT_FLOOR);
        // Per-buffer base focal scale (max of focals so the bound is
        // valid against either axis).
        const focalMax = Math.max(focalX, focalY);

        // Packed Uint16 buffer of (sx0, sx1, sy0, sy1) per candidate.
        // sx0 = 0xFFFF sentinel marks "off-screen, skip in pass 2".
        // Uint16 (not Uint8) so sub-frame counts per axis aren't capped
        // at 254 by the storage width / sentinel collision.
        const SF_OFFSCREEN = 0xFFFF;
        const ranges = new Uint16Array(candidateCount * 4);
        const subFrameCounts = new Uint32Array(numSubFrames);

        // Pass 1: compute ranges, count per sub-frame.
        for (let i = 0; i < candidateCount; i++) {
            const idx = candidates[i];
            const wx = xCol[idx] - ex;
            const wy = yCol[idx] - ey;
            const wz = zCol[idx] - ez;
            const cz = fx * wx + fy * wy + fz * wz;
            // Candidate has already passed the near-plane CPU cull; cz > near
            // is therefore an invariant here, no need to retest.
            const cx = rx2 * wx + ry2 * wy + rz2 * wz;
            const cy = dx2 * wx + dy2 * wy + dz2 * wz;
            const invZ = 1.0 / cz;
            const screenX = focalX * cx * invZ + halfW;
            const screenY = focalY * cy * invZ + halfH;
            const maxScale = Math.max(
                Math.exp(sxColRef[idx]),
                Math.exp(syColRef[idx]),
                Math.exp(szColRef[idx])
            );
            // Jacobian factor: (focal/cz)² · (1 + tx² + ty²) with
            // tx = cx/cz, ty = cy/cz both clamped to ±lim. Matches the
            // project shader's clamp so the bound is tight.
            const tx = cx * invZ;
            const ty = cy * invZ;
            const txClamped = tx > limX ? limX : (tx < -limX ? -limX : tx);
            const tyClamped = ty > limY ? limY : (ty < -limY ? -limY : ty);
            const tx2 = Math.min(txClamped * txClamped, limX2);
            const ty2 = Math.min(tyClamped * tyClamped, limY2);
            const jFactorSq = 1 + tx2 + ty2;
            const lambdaMaxBound = (focalMax * invZ) * (focalMax * invZ) * jFactorSq * maxScale * maxScale + lambdaSafety;
            // +1 px ceil safety to match the GPU's `ceil(radius)`.
            const screenR = Math.ceil(SIGMA_CUTOFF * Math.sqrt(lambdaMaxBound)) + 1;
            const minX = screenX - screenR;
            const maxX = screenX + screenR;
            const minY = screenY - screenR;
            const maxY = screenY + screenR;
            if (maxX < 0 || minX >= width || maxY < 0 || minY >= height) {
                ranges[i * 4] = SF_OFFSCREEN;
                continue;
            }
            const sx0 = Math.max(0, Math.floor(minX / subFramePixelsX));
            const sx1 = Math.min(numSubFramesX - 1, Math.floor(maxX / subFramePixelsX));
            const sy0 = Math.max(0, Math.floor(minY / subFramePixelsY));
            const sy1 = Math.min(numSubFramesY - 1, Math.floor(maxY / subFramePixelsY));
            ranges[i * 4 + 0] = sx0;
            ranges[i * 4 + 1] = sx1;
            ranges[i * 4 + 2] = sy0;
            ranges[i * 4 + 3] = sy1;
            for (let sy = sy0; sy <= sy1; sy++) {
                for (let sx = sx0; sx <= sx1; sx++) {
                    subFrameCounts[sy * numSubFramesX + sx]++;
                }
            }
        }

        // Allocate exact-size lists and fill (pass 2).
        subFrameLists = new Array(numSubFrames);
        const insertIdx = new Uint32Array(numSubFrames);
        for (let s = 0; s < numSubFrames; s++) {
            subFrameLists[s] = new Uint32Array(subFrameCounts[s]);
        }
        for (let i = 0; i < candidateCount; i++) {
            const sx0 = ranges[i * 4 + 0];
            if (sx0 === SF_OFFSCREEN) continue;
            const sx1 = ranges[i * 4 + 1];
            const sy0 = ranges[i * 4 + 2];
            const sy1 = ranges[i * 4 + 3];
            const idx = candidates[i];
            for (let sy = sy0; sy <= sy1; sy++) {
                for (let sx = sx0; sx <= sx1; sx++) {
                    const s = sy * numSubFramesX + sx;
                    subFrameLists[s][insertIdx[s]++] = idx;
                }
            }
        }
    }

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
        projection,
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
        focusDistance: camera.focusDistance ?? 0,
        apertureScale: camera.apertureScale ?? 0,
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

    // Total chunks across all sub-frames (each sub-frame's list is its
    // own filtered count). Used only for progress reporting; counts
    // actual dispatched chunks, so empty sub-frames contribute 0.
    let totalChunks = 0;
    for (let s = 0; s < numSubFrames; s++) {
        totalChunks += Math.ceil(subFrameLists[s].length / effectiveChunkCap);
    }
    const rasterBar = logger.bar('rasterizing', Math.max(1, totalChunks));
    let completed = 0;

    for (let sy = 0; sy < numSubFramesY; sy++) {
        for (let sx = 0; sx < numSubFramesX; sx++) {
            const tilesX = Math.min(subFrameTilesX, imageTilesX - sx * subFrameTilesX);
            const tilesY = Math.min(subFrameTilesY, imageTilesY - sy * subFrameTilesY);

            rasterizer.beginGroup(sx, sy, tilesX, tilesY);

            const sfCandidates = subFrameLists[sy * numSubFramesX + sx];
            const sfCount = sfCandidates.length;
            for (let chunkStart = 0; chunkStart < sfCount; chunkStart += effectiveChunkCap) {
                const chunkSize = Math.min(effectiveChunkCap, sfCount - chunkStart);
                packChunkInput(cols, sfCandidates, chunkStart, chunkSize, numSHBands, chunkInput);
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
