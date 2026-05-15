import { GraphicsDevice } from 'playcanvas';

import { computeGroupFrustumPlanes, queryBvhFrustum } from './bvh-query';
import { type RenderCamera, buildCameraBasis } from './camera';
import {
    packChunkInput,
    sceneSHBands,
    sortCandidatesByDepth,
    splatInputStride
} from './preprocess';
import { DataTable, computeGaussianExtents } from '../data-table';
import { GpuSplatRasterizer, TILE_SIZE } from '../gpu';
import { GaussianBVH } from '../spatial';
import { logger } from '../utils';

/** Tiles per group axis. 8 × 8 = 64 tiles → 128 × 128 pixels per group. */
const GROUP_TILES = 8;

/** Pixels per group axis. */
const GROUP_PX = GROUP_TILES * TILE_SIZE;

/**
 * Max gaussians per GPU dispatch. Bounds per-slot GPU buffers and the
 * granularity of the depth-ordered chunked path.
 */
const CHUNK_CAP = 200_000;

/** Number of pipelined slots (CPU prep on one overlaps GPU work on the others). */
const NUM_SLOTS = 3;

interface BackgroundRGBA {
    r: number;
    g: number;
    b: number;
    a: number;
}

interface SlotState {
    chunkInput: Float32Array;   // CPU host buffer, sized to chunkCap * inputStride
    candidates: Uint32Array;    // BVH query result for this slot's current group
    pendingReadback: Promise<Uint8Array> | null;
    pendingGroupX: number;
    pendingGroupY: number;
    pendingGroupTilesX: number;
    pendingGroupTilesY: number;
}

/**
 * Top-level entry point: build the BVH, walk tile groups in raster order
 * with three-slot pipelining, return the rendered RGBA buffer.
 *
 * @param device - PlayCanvas WebGPU graphics device.
 * @param dataTable - Gaussian splat data in PlayCanvas-identity space.
 * @param camera - Camera parameters.
 * @param background - RGBA background composited under residual transmittance.
 * @returns RGBA byte array of length `camera.width × camera.height × 4`.
 */
const renderTileStream = async (
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

    // ---- Build BVH ----
    const bvhGroup = logger.group('BVH');
    const extentsResult = computeGaussianExtents(dataTable);
    const bvh = new GaussianBVH(dataTable, extentsResult.extents);

    // Far plane: distance from eye to the most distant scene corner.
    const sb = extentsResult.sceneBounds;
    const corners = [
        sb.min.x, sb.min.y, sb.min.z, sb.max.x, sb.min.y, sb.min.z,
        sb.min.x, sb.max.y, sb.min.z, sb.max.x, sb.max.y, sb.min.z,
        sb.min.x, sb.min.y, sb.max.z, sb.max.x, sb.min.y, sb.max.z,
        sb.min.x, sb.max.y, sb.max.z, sb.max.x, sb.max.y, sb.max.z
    ];
    let far = camera.near * 100;
    for (let k = 0; k < 24; k += 3) {
        const dx = corners[k] - basis.eye.x;
        const dy = corners[k + 1] - basis.eye.y;
        const dz = corners[k + 2] - basis.eye.z;
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (d > far) far = d;
    }
    bvhGroup.end();

    // ---- Image tile grid ----
    const imageTilesX = Math.ceil(width / TILE_SIZE);
    const imageTilesY = Math.ceil(height / TILE_SIZE);
    const numGroupsX = Math.ceil(imageTilesX / GROUP_TILES);
    const numGroupsY = Math.ceil(imageTilesY / GROUP_TILES);
    const numGroups = numGroupsX * numGroupsY;

    // ---- Rasterizer ----
    const numSHBands = sceneSHBands(dataTable);
    const inputStride = splatInputStride(numSHBands);

    const rasterizer = new GpuSplatRasterizer(device, {
        numSHBands,
        groupSizeTiles: GROUP_TILES,
        chunkCap: CHUNK_CAP,
        numSlots: NUM_SLOTS,
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

    // ---- Per-slot CPU scratch ----
    const slotState: SlotState[] = [];
    for (let s = 0; s < NUM_SLOTS; s++) {
        slotState.push({
            chunkInput: new Float32Array(CHUNK_CAP * inputStride),
            candidates: new Uint32Array(64 * 1024),
            pendingReadback: null,
            pendingGroupX: 0,
            pendingGroupY: 0,
            pendingGroupTilesX: 0,
            pendingGroupTilesY: 0
        });
    }

    const finalImage = new Uint8Array(width * height * 4);

    // Blit one slot's group bytes into the final image.
    const blitSlot = (bytes: Uint8Array, groupX: number, groupY: number, groupTilesX: number, groupTilesY: number): void => {
        const originX = groupX * GROUP_PX;
        const originY = groupY * GROUP_PX;
        const groupPixelW = groupTilesX * TILE_SIZE;
        const groupPixelH = groupTilesY * TILE_SIZE;
        const drawW = Math.min(groupPixelW, width - originX);
        const drawH = Math.min(groupPixelH, height - originY);
        for (let row = 0; row < drawH; row++) {
            const srcOffset = row * groupPixelW * 4;
            const dstOffset = ((originY + row) * width + originX) * 4;
            finalImage.set(bytes.subarray(srcOffset, srcOffset + drawW * 4), dstOffset);
        }
    };

    const planes = new Float32Array(24);

    const processGroup = (slotIdx: number, groupX: number, groupY: number): void => {
        const slot = slotState[slotIdx];
        const tileMinX = groupX * GROUP_TILES;
        const tileMinY = groupY * GROUP_TILES;
        const groupTilesXVal = Math.min(GROUP_TILES, imageTilesX - tileMinX);
        const groupTilesYVal = Math.min(GROUP_TILES, imageTilesY - tileMinY);
        const gx0 = tileMinX * TILE_SIZE;
        const gy0 = tileMinY * TILE_SIZE;
        const gx1 = gx0 + groupTilesXVal * TILE_SIZE;
        const gy1 = gy0 + groupTilesYVal * TILE_SIZE;

        // Frustum-plane query.
        computeGroupFrustumPlanes(basis, gx0, gy0, gx1, gy1, width, height, camera.near, far, planes);
        const qr = queryBvhFrustum(bvh, planes, slot.candidates);
        slot.candidates = qr.buffer;
        const candidateCount = qr.count;

        rasterizer.beginGroup(slotIdx, groupX, groupY, groupTilesXVal, groupTilesYVal);

        if (candidateCount > 0) {
            // Depth-sort always — chunks must be processed front-to-back.
            sortCandidatesByDepth(dataTable, slot.candidates, candidateCount, basis);

            for (let chunkStart = 0; chunkStart < candidateCount; chunkStart += CHUNK_CAP) {
                const chunkSize = Math.min(CHUNK_CAP, candidateCount - chunkStart);
                packChunkInput(
                    dataTable, slot.candidates,
                    chunkStart, chunkSize,
                    numSHBands, slot.chunkInput
                );
                rasterizer.dispatchChunk(slotIdx, slot.chunkInput, chunkSize);
            }
        }

        const readback = rasterizer.finishGroup(slotIdx, groupTilesXVal, groupTilesYVal);
        slot.pendingReadback = readback;
        slot.pendingGroupX = groupX;
        slot.pendingGroupY = groupY;
        slot.pendingGroupTilesX = groupTilesXVal;
        slot.pendingGroupTilesY = groupTilesYVal;
    };

    const rasterBar = logger.bar('rasterizing', numGroups);
    let groupIdx = 0;
    let completed = 0;
    for (let gy = 0; gy < numGroupsY; gy++) {
        for (let gx = 0; gx < numGroupsX; gx++) {
            const slotIdx = groupIdx % NUM_SLOTS;
            const slot = slotState[slotIdx];

            // Drain this slot's previous readback before reusing it.
            if (slot.pendingReadback) {
                const bytes = await slot.pendingReadback;
                blitSlot(bytes, slot.pendingGroupX, slot.pendingGroupY, slot.pendingGroupTilesX, slot.pendingGroupTilesY);
                slot.pendingReadback = null;
                rasterBar.update(++completed);
            }

            processGroup(slotIdx, gx, gy);
            groupIdx++;
        }
    }
    // Drain remaining slots.
    for (const slot of slotState) {
        if (slot.pendingReadback) {
            const bytes = await slot.pendingReadback;
            blitSlot(bytes, slot.pendingGroupX, slot.pendingGroupY, slot.pendingGroupTilesX, slot.pendingGroupTilesY);
            slot.pendingReadback = null;
            rasterBar.update(++completed);
        }
    }
    rasterBar.end();

    rasterizer.destroy();

    return finalImage;
};

export { renderTileStream, GROUP_TILES, GROUP_PX, CHUNK_CAP };
