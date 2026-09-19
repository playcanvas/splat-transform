import {
    BUFFERUSAGE_COPY_DST,
    BUFFERUSAGE_COPY_SRC,
    SHADERLANGUAGE_WGSL,
    SHADERSTAGE_COMPUTE,
    BindGroupFormat,
    BindStorageBufferFormat,
    BindUniformBufferFormat,
    Compute,
    ComputeRadixSort,
    GraphicsDevice,
    Shader,
    StorageBuffer,
    UniformBufferFormat,
    Vec2
} from 'playcanvas';

import { constantsChunk } from './shaders/chunks/constants';
import { covariance3D, covariance3DFns } from './shaders/chunks/covariance-3d';
import { jacobianEquirect } from './shaders/chunks/jacobian-equirect';
import { jacobianPinhole, jacobianPinholeFns } from './shaders/chunks/jacobian-pinhole';
import { projectionEquirect } from './shaders/chunks/projection-equirect';
import { projectionPinhole } from './shaders/chunks/projection-pinhole';
import { quatRotation } from './shaders/chunks/quat-rotation';
import { shBand1 } from './shaders/chunks/sh-band-1';
import { shBand2 } from './shaders/chunks/sh-band-2';
import { shBand3 } from './shaders/chunks/sh-band-3';
import { tileAabbEquirect } from './shaders/chunks/tile-aabb-equirect';
import { tileAabbPinhole } from './shaders/chunks/tile-aabb-pinhole';
import { tileWalkEquirect } from './shaders/chunks/tile-walk-equirect';
import { tileWalkPinhole } from './shaders/chunks/tile-walk-pinhole';
import { clearStateWgsl } from './shaders/clear-state';
import { depthKeysWgsl } from './shaders/depth-keys';
import { finalizeWgsl } from './shaders/finalize';
import { findBoundariesWgsl } from './shaders/find-boundaries';
import { initTileOffsetsWgsl } from './shaders/init-tile-offsets';
import { projectWgsl } from './shaders/project';
import { rasterizeBinnedWgsl } from './shaders/rasterize-binned';
import { SCAN_BLOCK, scanBlocksWgsl, scanSumsWgsl } from './shaders/scan-blocks';
import { tileBinEmitPairsWgsl } from './shaders/tile-bin-emit-pairs';
import { uniformsStruct, uniformFormatEntries } from './shaders/uniforms';
import { type CameraBasis, type Projection } from '../render/camera';
import { PAIR_BUFFER_BUDGET_BYTES, PAIR_BUFFER_TOTAL_BYTES_PER_ELEMENT, TILE_SIZE } from '../render/config';
import { type SplatColumnRefs } from '../render/preprocess';

/** 12 floats per projected splat: vec4 × 3. */
const PROJECTION_STRIDE_F32 = 12;

/** Base attribute columns (position, rotation, log-scale, opacity, DC colour). */
const BASE_COLUMNS = 14;

/**
 * Pairs the range cutter aims for per sort: the shared pair-buffer budget
 * spread over the six pair-sized buffers (two here, four inside the radix
 * sort). A single scan block whose pairs exceed this still sorts in one
 * range; the budget is a target, the allocation is always exact.
 */
const PAIR_BUDGET = Math.floor(PAIR_BUFFER_BUDGET_BYTES / PAIR_BUFFER_TOTAL_BYTES_PER_ELEMENT);

/**
 * Thrown by {@link GpuSceneRasterizer.uploadScene} when the device reports
 * out-of-memory for the resident copy. Callers fall back to the chunked
 * path, which streams the scene through a bounded input buffer instead.
 */
class ResidentUploadError extends Error {
    constructor(message: string) {
        super(`resident scene upload failed: ${message}`);
        this.name = 'ResidentUploadError';
    }
}

/**
 * Fixed per-scene configuration of a {@link GpuSceneRasterizer}.
 */
interface SceneRasterizerOptions {
    /** Number of SH bands above DC (0–3). Selects the SH shader variant and sizes the SH buffer. */
    numSHBands: 0 | 1 | 2 | 3;
    /** Camera projection; fixed per instance (specialises the shaders). */
    projection: Projection;
    /** Output image size in pixels. */
    imageWidth: number;
    imageHeight: number;
    /**
     * Largest group (sub-frame) in tiles. Images beyond it render as a
     * grid of groups sharing one projection pass each; the running-state
     * and output buffers are sized to this.
     */
    groupTilesX: number;
    groupTilesY: number;
    /** Clamp frame-filling splats to the image (see `SIZE_CLAMP_FRAC`); default true. */
    sizeClamp?: boolean;
    /** Compile the motion-blur shader variant; every render must then supply `basisB`. */
    motionBlur: boolean;
    /** RGBA background, each channel in [0, 1]. */
    bgR: number; bgG: number; bgB: number; bgA: number;
}

/**
 * Camera for one render. The basis rows are (right, down, forward) of the
 * world→camera rotation; `basisB` is the shutter-close basis when the
 * instance was built with `motionBlur`.
 */
interface SceneView {
    basis: CameraBasis;
    basisB?: CameraBasis;
    near: number;
    focusDistance: number;
    apertureScale: number;
}

const numSHCoeffsPerChannel = (bands: number): number => {
    return bands === 0 ? 0 : bands === 1 ? 3 : bands === 2 ? 8 : 15;
};

/**
 * Resident-scene splat rasterizer.
 *
 * The scene's columns are uploaded once, column-major; a render uploads
 * nothing. Per render the GPU keys every gaussian by view depth and radix
 * sorts the keys (stable, identity payload, so ties break by row index as
 * the CPU sort does); the sorted index buffer is the order the project
 * shader gathers attributes in. Gaussians behind the near plane are keyed
 * like any other and invalidated by the project shader.
 *
 * Per render and group (sub-frame): project every gaussian in one
 * dispatch → two-level block scan of their tile coverage → read back the
 * block prefixes and the total → allocate the pair buffers to exactly that
 * total (or cut the depth-sorted list into ranges at scan-block boundaries
 * so each sort stays within the pair budget) → for each range: emit pairs,
 * radix sort by tile, find tile boundaries, rasterize → finalize and read
 * back the group's pixels.
 *
 * Every shader is shared with the chunked `GpuSplatRasterizer`; the
 * resident variants only change how attributes are addressed, so both
 * paths produce identical pixels.
 */
class GpuSceneRasterizer {
    private device: GraphicsDevice;
    private options: SceneRasterizerOptions;
    private shaders: Shader[] = [];
    private bgFormats: BindGroupFormat[] = [];
    private radixSort: ComputeRadixSort;
    private sortKeyBits: number;
    /** Depth sort over the scene; separate from the tile sort so neither rebuilds its passes. */
    private depthSort: ComputeRadixSort;
    private maxDispatchDim: number;
    private dispatchSize = new Vec2();

    private depthKeysCompute: Compute;
    private projectCompute: Compute;
    private scanBlocksCompute: Compute;
    private scanSumsCompute: Compute;
    private emitCompute: Compute;
    private initTileOffsetsCompute: Compute;
    private clearStateCompute: Compute;
    private findBoundariesCompute: Compute;
    private rasterizeCompute: Compute;
    private finalizeCompute: Compute;

    // Scene (set by uploadScene).
    private numSplats = 0;
    private baseBuffer: StorageBuffer | null = null;
    private shBuffer: StorageBuffer | null = null;
    private depthKeysBuffer: StorageBuffer | null = null;
    private projBuffer: StorageBuffer | null = null;
    private coverageBuffer: StorageBuffer | null = null;
    private emitOffsetBuffer: StorageBuffer | null = null;
    private blockSumsBuffer: StorageBuffer | null = null;
    private blockPrefix = new Uint32Array(0);
    private totalReadback = new Uint32Array(1);

    // Per-instance.
    private tileOffsetsBuffer: StorageBuffer;
    private totalPairsBuffer: StorageBuffer;
    private runningStateBuffer: StorageBuffer;
    private outputBuffer: StorageBuffer;
    private tileKeysBuffer: StorageBuffer | null = null;
    private splatValuesBuffer: StorageBuffer | null = null;
    private pairCapacity = 0;

    /** Pixels per group axis. */
    readonly groupPixelW: number;
    readonly groupPixelH: number;

    constructor(device: GraphicsDevice, options: SceneRasterizerOptions) {
        this.device = device;
        this.options = options;
        this.groupPixelW = options.groupTilesX * TILE_SIZE;
        this.groupPixelH = options.groupTilesY * TILE_SIZE;
        // @ts-ignore - limits is a WebGPU-device property not on the public type.
        this.maxDispatchDim = (device as { limits?: { maxComputeWorkgroupsPerDimension?: number } }).limits?.maxComputeWorkgroupsPerDimension ?? 65535;

        if (options.motionBlur && options.projection !== 'pinhole') {
            throw new Error('GpuSceneRasterizer: motion blur is pinhole-only');
        }

        // Direct-dispatch radix sort: the pair count is known on the CPU by
        // the time each range sorts.
        this.radixSort = new ComputeRadixSort(device);
        const numTiles = options.groupTilesX * options.groupTilesY;
        const { radixBits } = this.radixSort;
        const passes = Math.max(1, Math.ceil(Math.log2(Math.max(2, numTiles)) / radixBits));
        this.sortKeyBits = Math.min(passes, Math.floor(32 / radixBits)) * radixBits;
        this.depthSort = new ComputeRadixSort(device);

        const coeffs = numSHCoeffsPerChannel(options.numSHBands);
        const projection = options.projection;

        const cincludes = new Map<string, string>([
            ['uniformsStruct', uniformsStruct],
            ['constants', constantsChunk],
            ['projectionPinhole', projectionPinhole],
            ['projectionEquirect', projectionEquirect],
            ['jacobianPinhole', jacobianPinhole],
            ['jacobianPinholeFns', jacobianPinholeFns],
            ['jacobianEquirect', jacobianEquirect],
            // No coverage cap: the pair buffers are sized from the measured
            // total, so the bbox is never truncated. The cap only has to be
            // unreachable; a group's tile area is the largest bbox possible.
            ['tileAabbPinhole', tileAabbPinhole(numTiles)],
            ['tileAabbEquirect', tileAabbEquirect(numTiles)],
            ['tileWalkPinhole', tileWalkPinhole],
            ['tileWalkEquirect', tileWalkEquirect],
            ['shBand1', shBand1],
            ['shBand2', shBand2],
            ['shBand3', shBand3],
            ['quatRotation', quatRotation],
            ['covariance3D', covariance3D],
            ['covariance3DFns', covariance3DFns]
        ]);
        const cdefines = new Map<string, string>([['SOA', '']]);
        if (projection === 'equirect') cdefines.set('PROJECTION_EQUIRECT', '');
        if (options.numSHBands >= 1) cdefines.set('SH_BAND_1', '');
        if (options.numSHBands >= 2) cdefines.set('SH_BAND_2', '');
        if (options.numSHBands >= 3) cdefines.set('SH_BAND_3', '');
        if (options.sizeClamp === false) cdefines.set('NO_SIZE_CLAMP', '');
        if (options.motionBlur) cdefines.set('MOTION_BLUR', '');

        const U = SHADERSTAGE_COMPUTE;
        const ro = (name: string) => new BindStorageBufferFormat(name, U, true);
        const rw = (name: string) => new BindStorageBufferFormat(name, U);

        const mk = (name: string, source: string, entries: BindStorageBufferFormat[]): Compute => {
            const bgFormat = new BindGroupFormat(device, [new BindUniformBufferFormat('uniforms', U), ...entries]);
            const shader = new Shader(device, {
                name,
                shaderLanguage: SHADERLANGUAGE_WGSL,
                cshader: source,
                // @ts-ignore - compute-only shader definition fields are not in the public Shader types.
                computeUniformBufferFormats: { uniforms: new UniformBufferFormat(device, uniformFormatEntries()) },
                // @ts-ignore
                computeBindGroupFormat: bgFormat,
                // @ts-ignore
                cincludes,
                // @ts-ignore
                cdefines
            });
            this.shaders.push(shader);
            this.bgFormats.push(bgFormat);
            return new Compute(device, shader, name);
        };

        this.depthKeysCompute = mk('scene-depth-keys', depthKeysWgsl(), [ro('splatsBase'), rw('sortKeys')]);
        this.projectCompute = mk('scene-project', projectWgsl(coeffs), [ro('splatsBase'), rw('projected'), rw('coverage'), ro('splatsSH'), ro('order')]);
        this.scanBlocksCompute = mk('scene-scan-blocks', scanBlocksWgsl(), [ro('coverage'), rw('emitOffset'), rw('blockSums')]);
        this.scanSumsCompute = mk('scene-scan-sums', scanSumsWgsl(), [rw('blockSums'), rw('totalPairs')]);
        this.emitCompute = mk('scene-emit-pairs', tileBinEmitPairsWgsl(), [ro('projected'), ro('emitOffset'), ro('coverage'), rw('tileKeys'), rw('splatValues'), ro('blockPrefix')]);
        this.initTileOffsetsCompute = mk('scene-init-tile-offsets', initTileOffsetsWgsl(), [ro('totalPairs'), rw('tileOffsets')]);
        this.clearStateCompute = mk('scene-clear-state', clearStateWgsl(), [rw('runningState')]);
        this.findBoundariesCompute = mk('scene-find-boundaries', findBoundariesWgsl(), [ro('totalPairs'), ro('sortedTileKeys'), rw('tileOffsets')]);
        this.rasterizeCompute = mk('scene-rasterize', rasterizeBinnedWgsl(), [ro('projected'), rw('runningState'), ro('tileOffsets'), ro('sortedSplatIndices')]);
        this.finalizeCompute = mk('scene-finalize', finalizeWgsl(), [ro('runningState'), rw('output')]);

        const groupPixels = this.groupPixelW * this.groupPixelH;
        this.tileOffsetsBuffer = new StorageBuffer(device, (numTiles + 1) * 4, 0);
        this.totalPairsBuffer = new StorageBuffer(device, 4, BUFFERUSAGE_COPY_SRC | BUFFERUSAGE_COPY_DST);
        this.runningStateBuffer = new StorageBuffer(device, groupPixels * 16, BUFFERUSAGE_COPY_DST);
        this.outputBuffer = new StorageBuffer(device, groupPixels * 4, BUFFERUSAGE_COPY_SRC);

        this.scanSumsCompute.setParameter('totalPairs', this.totalPairsBuffer);
        this.initTileOffsetsCompute.setParameter('totalPairs', this.totalPairsBuffer);
        this.initTileOffsetsCompute.setParameter('tileOffsets', this.tileOffsetsBuffer);
        this.findBoundariesCompute.setParameter('totalPairs', this.totalPairsBuffer);
        this.findBoundariesCompute.setParameter('tileOffsets', this.tileOffsetsBuffer);
        this.clearStateCompute.setParameter('runningState', this.runningStateBuffer);
        this.rasterizeCompute.setParameter('runningState', this.runningStateBuffer);
        this.rasterizeCompute.setParameter('tileOffsets', this.tileOffsetsBuffer);
        this.finalizeCompute.setParameter('runningState', this.runningStateBuffer);
        this.finalizeCompute.setParameter('output', this.outputBuffer);
    }

    /**
     * Upload the scene's columns and allocate the per-splat working buffers.
     * Runs inside an out-of-memory error scope: if the device refuses any of
     * the allocations, everything is released and a
     * {@link ResidentUploadError} is thrown so the caller can fall back.
     *
     * @param cols - The scene's column references (all rows).
     * @param numSplats - Row count.
     */
    async uploadScene(cols: SplatColumnRefs, numSplats: number): Promise<void> {
        this.releaseScene();
        const device = this.device;
        const coeffs = numSHCoeffsPerChannel(this.options.numSHBands);
        // @ts-ignore - wgpu is the underlying GPUDevice on WebgpuGraphicsDevice.
        const wgpu = (device as { wgpu?: { pushErrorScope?: (f: string) => void; popErrorScope?: () => Promise<{ message: string } | null> } }).wgpu;

        wgpu?.pushErrorScope?.('out-of-memory');
        let oom: { message: string } | null | undefined;
        try {
            const numBlocks = Math.ceil(numSplats / SCAN_BLOCK);
            this.baseBuffer = new StorageBuffer(device, numSplats * BASE_COLUMNS * 4, BUFFERUSAGE_COPY_DST);
            this.shBuffer = new StorageBuffer(device, Math.max(4, numSplats * coeffs * 3 * 4), BUFFERUSAGE_COPY_DST);
            // The depth sort borrows this as one of its ping-pong buffers
            // (destructive keys), so it needs the sorter's copy usages.
            this.depthKeysBuffer = new StorageBuffer(device, numSplats * 4, BUFFERUSAGE_COPY_SRC | BUFFERUSAGE_COPY_DST);
            this.projBuffer = new StorageBuffer(device, numSplats * PROJECTION_STRIDE_F32 * 4, 0);
            this.coverageBuffer = new StorageBuffer(device, numSplats * 4, 0);
            this.emitOffsetBuffer = new StorageBuffer(device, numSplats * 4, 0);
            this.blockSumsBuffer = new StorageBuffer(device, Math.max(1, numBlocks) * 4, BUFFERUSAGE_COPY_SRC);
            this.blockPrefix = new Uint32Array(numBlocks + 1);

            const base = [
                cols.x, cols.y, cols.z,
                cols.rotW, cols.rotX, cols.rotY, cols.rotZ,
                cols.scaleX, cols.scaleY, cols.scaleZ,
                cols.opacity, cols.fdcR, cols.fdcG, cols.fdcB
            ];
            for (let c = 0; c < BASE_COLUMNS; c++) {
                this.baseBuffer.write(c * numSplats * 4, base[c], 0, numSplats);
            }
            for (let k = 0; k < coeffs * 3; k++) {
                this.shBuffer.write(k * numSplats * 4, cols.shRest[k], 0, numSplats);
            }
        } finally {
            oom = await wgpu?.popErrorScope?.();
        }
        if (oom) {
            this.releaseScene();
            throw new ResidentUploadError(oom.message);
        }
        this.numSplats = numSplats;

        this.depthKeysCompute.setParameter('splatsBase', this.baseBuffer);
        this.depthKeysCompute.setParameter('sortKeys', this.depthKeysBuffer);
        this.projectCompute.setParameter('splatsBase', this.baseBuffer);
        this.projectCompute.setParameter('splatsSH', this.shBuffer);
        this.projectCompute.setParameter('projected', this.projBuffer);
        this.projectCompute.setParameter('coverage', this.coverageBuffer);
        this.scanBlocksCompute.setParameter('coverage', this.coverageBuffer);
        this.scanBlocksCompute.setParameter('emitOffset', this.emitOffsetBuffer);
        this.scanBlocksCompute.setParameter('blockSums', this.blockSumsBuffer);
        this.scanSumsCompute.setParameter('blockSums', this.blockSumsBuffer);
        this.emitCompute.setParameter('projected', this.projBuffer);
        this.emitCompute.setParameter('emitOffset', this.emitOffsetBuffer);
        this.emitCompute.setParameter('coverage', this.coverageBuffer);
        this.emitCompute.setParameter('blockPrefix', this.blockSumsBuffer);
        this.rasterizeCompute.setParameter('projected', this.projBuffer);
    }

    /**
     * Render one view.
     *
     * @param view - Camera basis (and shutter-close basis under motion blur).
     * @returns RGBA bytes, `imageWidth × imageHeight × 4`.
     */
    async render(view: SceneView): Promise<Uint8Array> {
        if (!this.depthKeysBuffer) {
            throw new Error('GpuSceneRasterizer: uploadScene before render');
        }
        if (this.options.motionBlur && !view.basisB) {
            throw new Error('GpuSceneRasterizer: motion blur render needs the shutter-close basis');
        }
        const o = this.options;
        const { imageWidth: width, imageHeight: height } = o;
        const imageTilesX = Math.ceil(width / TILE_SIZE);
        const imageTilesY = Math.ceil(height / TILE_SIZE);
        const numGroupsX = Math.ceil(imageTilesX / o.groupTilesX);
        const numGroupsY = Math.ceil(imageTilesY / o.groupTilesY);
        const image = new Uint8Array(width * height * 4);

        if (this.numSplats > 0) {
            // Depth sort: key every gaussian, sort all 32 key bits (float
            // order is exact), and bind the sorter's index buffer as the
            // order the project shader gathers in. The sorter keeps that
            // buffer until the next depth sort, so every group of this
            // render reads the same order. Skips writing sorted keys and
            // borrows the keys buffer as scratch; both are rewritten per
            // render anyway.
            this.setUniforms(view, 0, 0, Math.min(o.groupTilesX, imageTilesX), Math.min(o.groupTilesY, imageTilesY));
            this.dispatch2D(this.depthKeysCompute, Math.ceil(this.numSplats / 64), 'scene-depth-keys');
            this.depthSort.sort(this.depthKeysBuffer, this.numSplats, 32, undefined, true, true);
            const order = this.depthSort.sortedIndices;
            if (!order) {
                throw new Error('ComputeRadixSort returned a null index buffer after sort()');
            }
            this.projectCompute.setParameter('order', order);
            this.submit();
        }

        for (let gy = 0; gy < numGroupsY; gy++) {
            for (let gx = 0; gx < numGroupsX; gx++) {
                const tilesX = Math.min(o.groupTilesX, imageTilesX - gx * o.groupTilesX);
                const tilesY = Math.min(o.groupTilesY, imageTilesY - gy * o.groupTilesY);
                const bytes = await this.renderGroup(view, gx, gy, tilesX, tilesY);

                const originX = gx * this.groupPixelW;
                const originY = gy * this.groupPixelH;
                const groupW = tilesX * TILE_SIZE;
                const copyW = Math.min(groupW, width - originX);
                const copyH = Math.min(tilesY * TILE_SIZE, height - originY);
                for (let row = 0; row < copyH; row++) {
                    const src = row * groupW * 4;
                    image.set(bytes.subarray(src, src + copyW * 4), ((originY + row) * width + originX) * 4);
                }
            }
        }
        return image;
    }

    private async renderGroup(view: SceneView, gx: number, gy: number, tilesX: number, tilesY: number): Promise<Uint8Array> {
        const device = this.device;
        const count = this.numSplats;
        this.setUniforms(view, gx, gy, tilesX, tilesY);

        // Clear running state: colour 0, transmittance 1. Four pixels per thread.
        const groupPixels = tilesX * tilesY * TILE_SIZE * TILE_SIZE;
        this.clearStateCompute.setupDispatch(Math.ceil(groupPixels / (4 * 256)), 1, 1);
        device.computeDispatch([this.clearStateCompute], 'scene-clear-state');

        if (count > 0) {
            // Project every gaussian against this group (those behind the
            // near plane come out invalid, with no coverage), then scan the
            // tile coverage: block-local offsets plus block totals, then
            // the block totals into exclusive prefixes and a grand total.
            this.dispatch2D(this.projectCompute, Math.ceil(count / 64), 'scene-project');
            const numBlocks = Math.ceil(count / SCAN_BLOCK);
            this.dispatch2D(this.scanBlocksCompute, numBlocks, 'scene-scan-blocks');
            this.scanSumsCompute.setupDispatch(1, 1, 1);
            device.computeDispatch([this.scanSumsCompute], 'scene-scan-sums');
            this.submit();

            const [prefix, total] = await Promise.all([
                this.blockSumsBuffer!.read(0, numBlocks * 4, this.blockPrefix, true) as Promise<Uint32Array>,
                this.totalPairsBuffer.read(0, 4, this.totalReadback, true) as Promise<Uint32Array>
            ]);
            const totalPairs = total[0];

            // Cut the depth-sorted list into ranges at scan-block boundaries so
            // each sort stays within the pair budget. A range that is a single
            // block exceeding the budget is sorted whole.
            let startBlock = 0;
            while (startBlock < numBlocks) {
                const rangeBase = prefix[startBlock];
                let endBlock = startBlock + 1;
                while (endBlock < numBlocks && prefix[endBlock] - rangeBase <= PAIR_BUDGET) {
                    endBlock++;
                }
                // `prefix[endBlock]` past the last block is the grand total.
                const rangeEnd = endBlock < numBlocks ? prefix[endBlock] : totalPairs;
                const rangePairs = rangeEnd - rangeBase;
                if (rangePairs > 0) {
                    const splatStart = startBlock * SCAN_BLOCK;
                    const splatCount = Math.min(count, endBlock * SCAN_BLOCK) - splatStart;
                    this.rasterizeRange(splatStart, splatCount, rangeBase, rangePairs, tilesX, tilesY);
                }
                startBlock = endBlock;
            }
        }

        this.finalizeCompute.setupDispatch(tilesX, tilesY, 1);
        device.computeDispatch([this.finalizeCompute], 'scene-finalize');
        this.submit();
        const bytes = tilesX * TILE_SIZE * tilesY * TILE_SIZE * 4;
        return this.outputBuffer.read(0, bytes, null, true) as Promise<Uint8Array>;
    }

    /**
     * Emit, sort, bin and rasterize the pairs of one range of the sorted list.
     *
     * @param splatStart - First splat of the range (index into the sorted order).
     * @param splatCount - Splats in the range.
     * @param rangeBase - Pair slot of the range's first pair in the scan's numbering.
     * @param rangePairs - Pairs in the range.
     * @param tilesX - Active group width in tiles.
     * @param tilesY - Active group height in tiles.
     */
    private rasterizeRange(splatStart: number, splatCount: number, rangeBase: number, rangePairs: number, tilesX: number, tilesY: number): void {
        const device = this.device;
        this.ensurePairCapacity(rangePairs);

        // The range's pair count drives the sentinel and the boundary pass.
        // queue writes land before the commands recorded after them.
        this.totalReadback[0] = rangePairs;
        this.totalPairsBuffer.write(0, this.totalReadback, 0, 1);

        const emit = this.emitCompute;
        emit.setParameter('tileKeys', this.tileKeysBuffer!);
        emit.setParameter('splatValues', this.splatValuesBuffer!);
        emit.setParameter('chunkSize', splatCount);
        emit.setParameter('rangeStart', splatStart);
        emit.setParameter('emitBase', rangeBase);
        this.dispatch2D(emit, Math.ceil(splatCount / 64), 'scene-emit-pairs');

        // Stable sort by tile keeps each tile's pairs in emission (depth) order.
        this.radixSort.sort(this.tileKeysBuffer!, rangePairs, this.sortKeyBits, this.splatValuesBuffer!);
        const sortedKeys = this.radixSort.sortedKeys;
        const sortedValues = this.radixSort.sortedIndices;
        if (!sortedKeys || !sortedValues) {
            throw new Error('ComputeRadixSort returned null buffers after sort()');
        }

        const numTiles = this.options.groupTilesX * this.options.groupTilesY;
        this.initTileOffsetsCompute.setupDispatch(Math.ceil((numTiles + 1) / 64), 1, 1);
        device.computeDispatch([this.initTileOffsetsCompute], 'scene-init-tile-offsets');

        this.findBoundariesCompute.setParameter('sortedTileKeys', sortedKeys);
        this.dispatch2D(this.findBoundariesCompute, Math.ceil(rangePairs / 64), 'scene-find-boundaries');

        this.rasterizeCompute.setParameter('sortedSplatIndices', sortedValues);
        this.rasterizeCompute.setupDispatch(tilesX, tilesY, 1);
        device.computeDispatch([this.rasterizeCompute], 'scene-rasterize');

        // Capture this range's uniforms before the next range overwrites them.
        this.submit();
    }

    private ensurePairCapacity(pairs: number): void {
        if (pairs <= this.pairCapacity) return;
        this.tileKeysBuffer?.destroy();
        this.splatValuesBuffer?.destroy();
        this.tileKeysBuffer = new StorageBuffer(this.device, pairs * 4, 0);
        this.splatValuesBuffer = new StorageBuffer(this.device, pairs * 4, 0);
        this.pairCapacity = pairs;
    }

    private dispatch2D(compute: Compute, workgroups: number, name: string): void {
        Compute.calcDispatchSize(Math.max(1, workgroups), this.dispatchSize, this.maxDispatchDim);
        compute.setupDispatch(this.dispatchSize.x, this.dispatchSize.y, 1);
        this.device.computeDispatch([compute], name);
    }

    private setUniforms(view: SceneView, gx: number, gy: number, tilesX: number, tilesY: number): void {
        const o = this.options;
        const b = view.basis;
        const bb = view.basisB;
        const count = this.numSplats;
        const originX = gx * this.groupPixelW;
        const originY = gy * this.groupPixelH;
        const maxX = originX + tilesX * TILE_SIZE;
        const maxY = originY + tilesY * TILE_SIZE;
        const computes = [
            this.depthKeysCompute,
            this.clearStateCompute, this.projectCompute, this.scanBlocksCompute, this.scanSumsCompute,
            this.emitCompute, this.initTileOffsetsCompute, this.findBoundariesCompute,
            this.rasterizeCompute, this.finalizeCompute
        ];
        for (const c of computes) {
            c.setParameter('rightX', b.right.x); c.setParameter('rightY', b.right.y); c.setParameter('rightZ', b.right.z);
            c.setParameter('_p0', 0);
            c.setParameter('downX', b.down.x); c.setParameter('downY', b.down.y); c.setParameter('downZ', b.down.z);
            c.setParameter('_p1', 0);
            c.setParameter('forwardX', b.forward.x); c.setParameter('forwardY', b.forward.y); c.setParameter('forwardZ', b.forward.z);
            c.setParameter('_p2', 0);
            c.setParameter('eyeX', b.eye.x); c.setParameter('eyeY', b.eye.y); c.setParameter('eyeZ', b.eye.z);
            c.setParameter('_p3', 0);
            c.setParameter('focalX', b.focalX); c.setParameter('focalY', b.focalY);
            c.setParameter('near', view.near); c.setParameter('_p4', 0);
            c.setParameter('focusDistance', view.focusDistance);
            c.setParameter('apertureScale', view.apertureScale);
            c.setParameter('_p5', 0); c.setParameter('_p6', 0);
            c.setParameter('imageWidth', o.imageWidth); c.setParameter('imageHeight', o.imageHeight);
            c.setParameter('splatStride', 0);
            c.setParameter('chunkSize', count);
            c.setParameter('groupPixelMinX', originX);
            c.setParameter('groupPixelMinY', originY);
            c.setParameter('groupPixelMaxX', maxX);
            c.setParameter('groupPixelMaxY', maxY);
            c.setParameter('groupTilesX', tilesX);
            c.setParameter('groupTilesY', tilesY);
            c.setParameter('groupPixelOriginX', originX);
            c.setParameter('groupPixelOriginY', originY);
            c.setParameter('bgR', o.bgR); c.setParameter('bgG', o.bgG);
            c.setParameter('bgB', o.bgB); c.setParameter('bgA', o.bgA);
            c.setParameter('rightBX', bb?.right.x ?? 0); c.setParameter('rightBY', bb?.right.y ?? 0); c.setParameter('rightBZ', bb?.right.z ?? 0);
            c.setParameter('_p7', 0);
            c.setParameter('downBX', bb?.down.x ?? 0); c.setParameter('downBY', bb?.down.y ?? 0); c.setParameter('downBZ', bb?.down.z ?? 0);
            c.setParameter('_p8', 0);
            c.setParameter('forwardBX', bb?.forward.x ?? 0); c.setParameter('forwardBY', bb?.forward.y ?? 0); c.setParameter('forwardBZ', bb?.forward.z ?? 0);
            c.setParameter('_p9', 0);
            c.setParameter('eyeBX', bb?.eye.x ?? 0); c.setParameter('eyeBY', bb?.eye.y ?? 0); c.setParameter('eyeBZ', bb?.eye.z ?? 0);
            c.setParameter('_p10', 0);
            c.setParameter('numSplats', this.numSplats);
            c.setParameter('rangeStart', 0);
            c.setParameter('emitBase', 0);
            c.setParameter('_p11', 0);
        }
    }

    private submit(): void {
        // @ts-ignore - submit() is exposed by WebgpuGraphicsDevice but not on the public GraphicsDevice type.
        const submit = (this.device as { submit?: () => void }).submit;
        if (!submit) {
            throw new Error('GpuSceneRasterizer requires a GraphicsDevice with a submit() method (WebGPU backend).');
        }
        submit.call(this.device);
    }

    private releaseScene(): void {
        this.baseBuffer?.destroy();
        this.shBuffer?.destroy();
        this.depthKeysBuffer?.destroy();
        this.projBuffer?.destroy();
        this.coverageBuffer?.destroy();
        this.emitOffsetBuffer?.destroy();
        this.blockSumsBuffer?.destroy();
        this.baseBuffer = this.shBuffer = this.depthKeysBuffer = this.projBuffer = null;
        this.coverageBuffer = this.emitOffsetBuffer = this.blockSumsBuffer = null;
        this.numSplats = 0;
    }

    /** Release all GPU resources. */
    destroy(): void {
        this.releaseScene();
        this.tileKeysBuffer?.destroy();
        this.splatValuesBuffer?.destroy();
        this.tileOffsetsBuffer.destroy();
        this.totalPairsBuffer.destroy();
        this.runningStateBuffer.destroy();
        this.outputBuffer.destroy();
        this.radixSort.destroy();
        this.depthSort.destroy();
        for (const s of this.shaders) s.destroy();
        for (const f of this.bgFormats) f.destroy();
    }
}

export { GpuSceneRasterizer, ResidentUploadError, type SceneRasterizerOptions, type SceneView };
