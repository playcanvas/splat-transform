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

import { type ChunkData, type ChunkDataPool, type ChunkLayer, type ChunkSource, type ReadRequest, colorStride } from '../chunk';
import { type CameraBasis, type Projection } from '../render/camera';
import { PAIR_BUFFER_BUDGET_BYTES, PAIR_BUFFER_TOTAL_BYTES_PER_ELEMENT, TILE_SIZE, storageBindingLimit } from '../render/config';
import { accumulateWgsl } from './shaders/accumulate';
import { constantsChunk } from './shaders/chunks/constants';
import { covariance3D } from './shaders/chunks/covariance-3d';
import { jacobianEquirect } from './shaders/chunks/jacobian-equirect';
import { jacobianPinhole } from './shaders/chunks/jacobian-pinhole';
import { packRGBA8 } from './shaders/chunks/pack-rgba8';
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

/** 12 floats per projected splat: vec4 × 3. */
const PROJECTION_STRIDE_F32 = 12;

/** Floats per gaussian in the source's position and geometric layers. */
const POSITION_F32 = 3;
const GEOMETRIC_F32 = 8;

/**
 * Pairs the range cutter aims for per sort: the shared pair-buffer budget
 * spread over the six pair-sized buffers (two here, four inside the radix
 * sort). Each instance also caps a range so every pair buffer fits one
 * storage binding; the budget is a target, the allocation is always exact.
 */
const PAIR_BUDGET = Math.floor(PAIR_BUFFER_BUDGET_BYTES / PAIR_BUFFER_TOTAL_BYTES_PER_ELEMENT);

/**
 * Thrown by the upload methods when the device reports out-of-memory for
 * the scene's buffers. Callers fall back to a tier that holds less on the
 * GPU.
 */
class ResidentUploadError extends Error {
    constructor(message: string) {
        super(`scene upload failed: ${message}`);
        this.name = 'ResidentUploadError';
    }
}

/**
 * Fixed per-scene configuration of a {@link GpuSceneRasterizer}.
 */
interface SceneRasterizerOptions {
    /** Number of SH bands above DC (0–3). Selects the SH shader variant and sizes the colour layer. */
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
    /** RGBA background, each channel in [0, 1]. */
    bgR: number; bgG: number; bgB: number; bgA: number;
}

/**
 * Camera for one render. The basis rows are (right, down, forward) of the
 * world→camera rotation.
 */
interface SceneView {
    basis: CameraBasis;
    near: number;
    focusDistance: number;
    apertureScale: number;
}

/**
 * One range of the streamed path: `count` gaussians' layer records in
 * compositing order, as read from the source. The rasterizer releases the
 * buffers once uploaded.
 */
interface StreamedRange {
    count: number;
    position: ChunkData;
    geometric: ChunkData;
    color: ChunkData;
}

/** The scene's order for one view: `order[0..visible)` are the visible gaussians, front to back. */
interface SortedOrder {
    order: Uint32Array;
    visible: number;
}

/** Layers the rasterizer reads. */
type SceneLayer = Extract<ChunkLayer, 'position' | 'geometric' | 'color'>;

const numSHCoeffsPerChannel = (bands: number): number => {
    return bands === 0 ? 0 : bands === 1 ? 3 : bands === 2 ? 8 : 15;
};

/**
 * Recover the true pair prefixes from the GPU's u32 scan, which wraps past
 * 2³² pairs in one pass. Each block adds far fewer than 2³², so a prefix
 * below its predecessor marks exactly one wrap; the grand total is treated
 * as the prefix past the last block.
 *
 * @param prefix - Exclusive block prefixes as read back, `numBlocks` entries.
 * @param numBlocks - Blocks in the range.
 * @param total - The grand total as read back.
 * @param out - Receives `numBlocks + 1` true prefixes; the last is the true total.
 */
const unwrapPrefixes = (prefix: Uint32Array, numBlocks: number, total: number, out: Float64Array): void => {
    let carry = 0;
    let previous = 0;
    for (let b = 0; b <= numBlocks; b++) {
        const wrapped = b < numBlocks ? prefix[b] : total;
        if (wrapped < previous) carry += 2 ** 32;
        previous = wrapped;
        out[b] = wrapped + carry;
    }
};

/**
 * Scene splat rasterizer over a {@link ChunkSource}'s LOD 0.
 *
 * Two ways to hold the scene, chosen by the caller:
 *
 * - **Resident** (`uploadScene`): the source's three layers are uploaded once
 *   as they are stored (position, geometric and colour records), and a render
 *   sends only the camera. The GPU keys every gaussian by view depth, radix
 *   sorts the keys (stable, identity payload, so ties break by row index as
 *   the CPU sort does) and gathers attributes in that order.
 * - **Streamed** (`uploadPositions` or `prepareStreamed`): only the position
 *   layer is resident, or nothing at all, and the caller streams the
 *   gaussians' records per pass in depth order in ranges the rasterizer
 *   uploads into range-sized buffers. With the positions resident the GPU
 *   sorts and hands the order back (`sortedOrder`); otherwise the caller
 *   sorts. Attribute memory is bounded by the range size, so any scene whose
 *   positions fit renders.
 *
 * Either way, per render and group (sub-frame): project the gaussians of a
 * range in one dispatch → two-level block scan of their tile coverage → read
 * back the block prefixes and the total → pair buffers sized exactly (or the
 * range cut at scan-block boundaries so each sort stays within the pair
 * budget) → per cut: emit pairs, radix sort by tile, find tile boundaries,
 * rasterize → after the last range: finalize and read back the group. A
 * motion-blurred frame renders its shutter slices one after another per
 * group, accumulating each slice's composited colour in float on the GPU and
 * packing the mean on the last, so the frame costs one readback and
 * quantizes once.
 *
 * Every shader is shared with the chunked `GpuSplatRasterizer`; the scene
 * variants only change how attributes are addressed, so both produce
 * identical pixels.
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
    private colorF32: number;

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
    /** Slice accumulation, in place of finalize for multi-view frames. */
    private accumulateCompute: Compute;

    // Scene (set by the upload methods).
    /** Gaussians in the scene: the depth sort's extent. */
    private numSplats = 0;
    /** Rows the per-pass working buffers hold: the scene when resident, one range when streamed. */
    private capacity = 0;
    private streamed = false;
    /** The layers the project shader reads: the scene when resident, the current range when streamed. */
    private positionBuffer: StorageBuffer | null = null;
    private geometricBuffer: StorageBuffer | null = null;
    private colorBuffer: StorageBuffer | null = null;
    /** The scene's positions for the depth keys when streaming with the GPU sort. */
    private scenePositionBuffer: StorageBuffer | null = null;
    private depthKeysBuffer: StorageBuffer | null = null;
    /** Identity order for the streamed path: ranges arrive already sorted. */
    private identityOrderBuffer: StorageBuffer | null = null;
    private projBuffer: StorageBuffer | null = null;
    private coverageBuffer: StorageBuffer | null = null;
    private emitOffsetBuffer: StorageBuffer | null = null;
    private blockSumsBuffer: StorageBuffer | null = null;
    private blockPrefix = new Uint32Array(0);
    /** True pairs before each block and the true total, unwrapped from the u32 scan. */
    private blockBefore = new Float64Array(0);
    private totalReadback = new Uint32Array(1);
    private orderReadback = new Uint32Array(0);
    private visibleReadback = new Uint32Array(1);
    private zero = new Uint32Array(1);

    // Per-instance.
    private visibleCountBuffer: StorageBuffer;
    private tileOffsetsBuffer: StorageBuffer;
    private totalPairsBuffer: StorageBuffer;
    private runningStateBuffer: StorageBuffer;
    private outputBuffer: StorageBuffer;
    private accumBuffer: StorageBuffer | null = null;
    private tileKeysBuffer: StorageBuffer | null = null;
    private splatValuesBuffer: StorageBuffer | null = null;
    private pairCapacity = 0;
    /** Most pairs per sort: the budget, or fewer where a pair buffer would exceed a binding. */
    private pairCap: number;
    /** Readback of one scan block's per-splat offsets, for cutting inside an oversized block. */
    private blockOffsets = new Uint32Array(SCAN_BLOCK);

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
        this.pairCap = Math.min(PAIR_BUDGET, Math.floor(storageBindingLimit(device) / 4));

        // Direct-dispatch radix sort: the pair count is known on the CPU by
        // the time each range sorts.
        this.radixSort = new ComputeRadixSort(device);
        const numTiles = options.groupTilesX * options.groupTilesY;
        const { radixBits } = this.radixSort;
        const passes = Math.max(1, Math.ceil(Math.log2(Math.max(2, numTiles)) / radixBits));
        this.sortKeyBits = Math.min(passes, Math.floor(32 / radixBits)) * radixBits;
        this.depthSort = new ComputeRadixSort(device);

        const coeffs = numSHCoeffsPerChannel(options.numSHBands);
        this.colorF32 = 3 + 3 * coeffs;
        const projection = options.projection;

        const cincludes = new Map<string, string>([
            ['uniformsStruct', uniformsStruct],
            ['constants', constantsChunk],
            ['packRGBA8', packRGBA8],
            ['projectionPinhole', projectionPinhole],
            ['projectionEquirect', projectionEquirect],
            ['jacobianPinhole', jacobianPinhole],
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
            ['covariance3D', covariance3D]
        ]);
        const cdefines = new Map<string, string>([['SCENE', '']]);
        if (projection === 'equirect') cdefines.set('PROJECTION_EQUIRECT', '');
        if (options.numSHBands >= 1) cdefines.set('SH_BAND_1', '');
        if (options.numSHBands >= 2) cdefines.set('SH_BAND_2', '');
        if (options.numSHBands >= 3) cdefines.set('SH_BAND_3', '');
        if (options.sizeClamp === false) cdefines.set('NO_SIZE_CLAMP', '');

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

        this.depthKeysCompute = mk('scene-depth-keys', depthKeysWgsl(), [ro('position'), rw('sortKeys'), rw('visibleCount')]);
        this.projectCompute = mk('scene-project', projectWgsl(coeffs), [ro('position'), rw('projected'), rw('coverage'), ro('geometric'), ro('color'), ro('order')]);
        this.scanBlocksCompute = mk('scene-scan-blocks', scanBlocksWgsl(), [ro('coverage'), rw('emitOffset'), rw('blockSums')]);
        this.scanSumsCompute = mk('scene-scan-sums', scanSumsWgsl(), [rw('blockSums'), rw('totalPairs')]);
        this.emitCompute = mk('scene-emit-pairs', tileBinEmitPairsWgsl(), [ro('projected'), ro('emitOffset'), ro('coverage'), rw('tileKeys'), rw('splatValues'), ro('blockPrefix')]);
        this.initTileOffsetsCompute = mk('scene-init-tile-offsets', initTileOffsetsWgsl(), [ro('totalPairs'), rw('tileOffsets')]);
        this.clearStateCompute = mk('scene-clear-state', clearStateWgsl(), [rw('runningState')]);
        this.findBoundariesCompute = mk('scene-find-boundaries', findBoundariesWgsl(), [ro('totalPairs'), ro('sortedTileKeys'), rw('tileOffsets')]);
        this.rasterizeCompute = mk('scene-rasterize', rasterizeBinnedWgsl(), [ro('projected'), rw('runningState'), ro('tileOffsets'), ro('sortedSplatIndices')]);
        this.finalizeCompute = mk('scene-finalize', finalizeWgsl(), [ro('runningState'), rw('output')]);
        this.accumulateCompute = mk('scene-accumulate', accumulateWgsl(), [ro('runningState'), rw('accum'), rw('output')]);

        const groupPixels = this.groupPixelW * this.groupPixelH;
        this.visibleCountBuffer = new StorageBuffer(device, 4, BUFFERUSAGE_COPY_SRC | BUFFERUSAGE_COPY_DST);
        this.tileOffsetsBuffer = new StorageBuffer(device, (numTiles + 1) * 4, 0);
        this.totalPairsBuffer = new StorageBuffer(device, 4, BUFFERUSAGE_COPY_SRC | BUFFERUSAGE_COPY_DST);
        this.runningStateBuffer = new StorageBuffer(device, groupPixels * 16, BUFFERUSAGE_COPY_DST);
        this.outputBuffer = new StorageBuffer(device, groupPixels * 4, BUFFERUSAGE_COPY_SRC);
        this.accumulateCompute.setParameter('runningState', this.runningStateBuffer);
        this.accumulateCompute.setParameter('output', this.outputBuffer);

        this.depthKeysCompute.setParameter('visibleCount', this.visibleCountBuffer);
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
     * Hold the whole scene resident: upload the source's LOD 0 layers as
     * stored and allocate the per-splat working buffers. The allocations run
     * inside an out-of-memory error scope; if the device refuses any of them,
     * everything is released and a {@link ResidentUploadError} is thrown so
     * the caller can fall back to a streamed tier.
     *
     * @param source - The scene.
     * @param pool - Pool for the chunk read buffers.
     */
    async uploadScene(source: ChunkSource, pool: ChunkDataPool): Promise<void> {
        this.releaseScene();
        const device = this.device;
        const n = source.meta.lodCounts[0];
        const cs = colorStride(source.meta.shBands);
        await this.guardAllocation(() => {
            this.positionBuffer = new StorageBuffer(device, Math.max(4, n * POSITION_F32 * 4), BUFFERUSAGE_COPY_DST);
            this.geometricBuffer = new StorageBuffer(device, Math.max(4, n * GEOMETRIC_F32 * 4), BUFFERUSAGE_COPY_DST);
            this.colorBuffer = new StorageBuffer(device, Math.max(4, n * cs), BUFFERUSAGE_COPY_DST);
            this.allocateSort(n);
            this.allocateWorking(n);
        });
        this.numSplats = n;
        this.capacity = n;
        this.streamed = false;

        await this.readChunks(source, pool, ['position', 'geometric', 'color'], (rowStart, count, data) => {
            this.positionBuffer!.write(rowStart * POSITION_F32 * 4, new Float32Array(data.position!.data, 0, count * POSITION_F32), 0, count * POSITION_F32);
            this.geometricBuffer!.write(rowStart * GEOMETRIC_F32 * 4, new Float32Array(data.geometric!.data, 0, count * GEOMETRIC_F32), 0, count * GEOMETRIC_F32);
            this.colorBuffer!.write(rowStart * cs, new Float32Array(data.color!.data, 0, count * this.colorF32), 0, count * this.colorF32);
        });

        this.depthKeysCompute.setParameter('position', this.positionBuffer);
        this.bindWorking();
    }

    /**
     * Stream the scene with the GPU sort: only the position layer is
     * resident, for the depth keys; the gaussians' records arrive per pass in
     * ranges of up to `rangeRows`, in the order `sortedOrder` hands back.
     * Throws {@link ResidentUploadError} on out-of-memory.
     *
     * @param source - The scene.
     * @param pool - Pool for the chunk read buffers.
     * @param rangeRows - Most gaussians per streamed range.
     */
    async uploadPositions(source: ChunkSource, pool: ChunkDataPool, rangeRows: number): Promise<void> {
        this.releaseScene();
        const device = this.device;
        const n = source.meta.lodCounts[0];
        await this.guardAllocation(() => {
            this.scenePositionBuffer = new StorageBuffer(device, Math.max(4, n * POSITION_F32 * 4), BUFFERUSAGE_COPY_DST);
            this.allocateSort(n);
            this.allocateRange(rangeRows, source.meta.shBands);
        });
        this.numSplats = n;
        this.capacity = rangeRows;
        this.streamed = true;
        this.orderReadback = new Uint32Array(n);

        await this.readChunks(source, pool, ['position'], (rowStart, count, data) => {
            this.scenePositionBuffer!.write(rowStart * POSITION_F32 * 4, new Float32Array(data.position!.data, 0, count * POSITION_F32), 0, count * POSITION_F32);
        });

        this.depthKeysCompute.setParameter('position', this.scenePositionBuffer);
        this.bindWorking();
    }

    /**
     * Stream the scene with the caller's sort: nothing resident but the
     * range buffers. Throws {@link ResidentUploadError} on out-of-memory.
     *
     * @param numSplats - Gaussians in the scene.
     * @param rangeRows - Most gaussians per streamed range.
     * @param numSHBands - The scene's SH bands, sizing the colour range buffer.
     */
    async prepareStreamed(numSplats: number, rangeRows: number, numSHBands: 0 | 1 | 2 | 3): Promise<void> {
        this.releaseScene();
        await this.guardAllocation(() => {
            this.allocateRange(rangeRows, numSHBands);
        });
        this.numSplats = numSplats;
        this.capacity = rangeRows;
        this.streamed = true;
        this.bindWorking();
    }

    /**
     * Render one view, or the mean of several, from the resident scene. With
     * more than one view each is one shutter sample of a motion-blurred
     * frame: the samples accumulate on the GPU in float and the frame
     * quantizes once.
     *
     * @param views - One view, or the shutter samples of a frame.
     * @returns RGBA bytes, `imageWidth × imageHeight × 4`.
     */
    render(views: SceneView[]): Promise<Uint8Array> {
        if (!this.positionBuffer || this.streamed) {
            throw new Error('GpuSceneRasterizer: uploadScene before render');
        }
        this.checkViews(views);
        return this.renderGroups(views, async (view, tilesX, tilesY) => {
            this.depthSortPass();
            await this.rasterRange(this.numSplats, tilesX, tilesY);
        });
    }

    /**
     * Render one view, or the mean of several, streaming the gaussians'
     * records per pass. `ranges(view)` yields the view's visible gaussians
     * front to back in ranges of at most the prepared row count; the next
     * range is requested while the current one rasterizes. For a
     * multi-group image the ranges are requested once per group.
     *
     * @param views - One view, or the shutter slices of a frame.
     * @param ranges - Produces a view's depth-sorted ranges.
     * @returns RGBA bytes, `imageWidth × imageHeight × 4`.
     */
    renderStreamed(views: SceneView[], ranges: (view: SceneView) => AsyncIterable<StreamedRange>): Promise<Uint8Array> {
        if (!this.positionBuffer || !this.streamed) {
            throw new Error('GpuSceneRasterizer: uploadPositions or prepareStreamed before renderStreamed');
        }
        this.checkViews(views);
        return this.renderGroups(views, async (view, tilesX, tilesY) => {
            const it = ranges(view)[Symbol.asyncIterator]();
            let next = it.next();
            for (;;) {
                const { value: range, done } = await next;
                if (done) break;
                next = it.next();
                this.uploadRange(range);
                await this.rasterRange(range.count, tilesX, tilesY);
            }
        });
    }

    /**
     * Depth sort the scene for the view in the current uniforms and read the
     * order back: `order[0..visible)` are the gaussians the project shader
     * will accept, front to back. Streamed path with resident positions only;
     * call from within a `renderStreamed` range producer.
     *
     * @returns The sorted order and the visible count.
     */
    async sortedOrder(): Promise<SortedOrder> {
        if (!this.scenePositionBuffer) {
            throw new Error('GpuSceneRasterizer: sortedOrder needs uploadPositions');
        }
        if (this.numSplats === 0) return { order: this.orderReadback, visible: 0 };
        this.depthSortPass();
        const sorted = this.depthSort.sortedIndices!;
        const [order, visible] = await Promise.all([
            sorted.read(0, this.numSplats * 4, this.orderReadback, true) as Promise<Uint32Array>,
            this.visibleCountBuffer.read(0, 4, this.visibleReadback, true) as Promise<Uint32Array>
        ]);
        return { order, visible: visible[0] };
    }

    private checkViews(views: SceneView[]): void {
        if (views.length === 0) {
            throw new Error('GpuSceneRasterizer: render needs at least one view');
        }
    }

    /**
     * The group loop shared by both paths: per group, per slice, set the
     * uniforms, clear the running state, let `slice` rasterize the scene into
     * it, then finalize (one slice) or accumulate (several); read the group
     * back after its last slice. Slices run inside groups so the accumulator
     * stays group-sized.
     *
     * @param views - The slices.
     * @param slice - Rasterizes one slice into the running state.
     * @returns RGBA bytes of the whole image.
     */
    private async renderGroups(views: SceneView[], slice: (view: SceneView, tilesX: number, tilesY: number) => Promise<void>): Promise<Uint8Array> {
        const o = this.options;
        const { imageWidth: width, imageHeight: height } = o;
        const imageTilesX = Math.ceil(width / TILE_SIZE);
        const imageTilesY = Math.ceil(height / TILE_SIZE);
        const numGroupsX = Math.ceil(imageTilesX / o.groupTilesX);
        const numGroupsY = Math.ceil(imageTilesY / o.groupTilesY);
        const image = new Uint8Array(width * height * 4);
        if (views.length > 1 && !this.accumBuffer) {
            // The accumulator is group-sized; allocate it on the first multi-view frame.
            this.accumBuffer = new StorageBuffer(this.device, this.groupPixelW * this.groupPixelH * 16, 0);
            this.accumulateCompute.setParameter('accum', this.accumBuffer);
        }
        const pack = views.length > 1 ? this.accumulateCompute : this.finalizeCompute;
        const packName = views.length > 1 ? 'scene-accumulate' : 'scene-finalize';

        for (let gy = 0; gy < numGroupsY; gy++) {
            for (let gx = 0; gx < numGroupsX; gx++) {
                const tilesX = Math.min(o.groupTilesX, imageTilesX - gx * o.groupTilesX);
                const tilesY = Math.min(o.groupTilesY, imageTilesY - gy * o.groupTilesY);

                for (let s = 0; s < views.length; s++) {
                    this.setUniforms(views[s], gx, gy, tilesX, tilesY, s, views.length);
                    // Clear running state: colour 0, transmittance 1. Four pixels per thread.
                    const groupPixels = tilesX * tilesY * TILE_SIZE * TILE_SIZE;
                    this.clearStateCompute.setupDispatch(Math.ceil(groupPixels / (4 * 256)), 1, 1);
                    this.device.computeDispatch([this.clearStateCompute], 'scene-clear-state');
                    await slice(views[s], tilesX, tilesY);
                    pack.setupDispatch(tilesX, tilesY, 1);
                    this.device.computeDispatch([pack], packName);
                    this.submit();
                }
                const bytes = await this.outputBuffer.read(0, tilesX * TILE_SIZE * tilesY * TILE_SIZE * 4, null, true) as Uint8Array;

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

    /**
     * Depth sort for the view in the current uniforms: key every gaussian,
     * sort all 32 key bits (float order is exact) and, when resident, bind
     * the sorter's index buffer as the order the project shader gathers in.
     * Skips writing sorted keys and borrows the keys buffer as scratch; both
     * are rewritten per pass.
     */
    private depthSortPass(): void {
        if (this.numSplats === 0) return;
        this.visibleCountBuffer.write(0, this.zero, 0, 1);
        this.dispatch2D(this.depthKeysCompute, Math.ceil(this.numSplats / 64), 'scene-depth-keys');
        this.depthSort.sort(this.depthKeysBuffer!, this.numSplats, 32, undefined, true, true);
        const order = this.depthSort.sortedIndices;
        if (!order) {
            throw new Error('ComputeRadixSort returned a null index buffer after sort()');
        }
        if (!this.streamed) this.projectCompute.setParameter('order', order);
        this.submit();
    }

    /**
     * Rasterize `count` gaussians (the scene when resident, one range when
     * streamed) from the bound layer buffers into the active group's running
     * state: project, scan the coverage, then emit, sort and rasterize every
     * cut of the range that fits the pair budget.
     *
     * @param count - Gaussians to project.
     * @param tilesX - Active group width in tiles.
     * @param tilesY - Active group height in tiles.
     */
    private async rasterRange(count: number, tilesX: number, tilesY: number): Promise<void> {
        if (count === 0) return;
        const device = this.device;
        // Rows to project and scan: the range, not the scene.
        for (const c of [this.projectCompute, this.scanBlocksCompute, this.scanSumsCompute]) {
            c.setParameter('chunkSize', count);
        }

        // Project every gaussian against this group (those behind the near
        // plane come out invalid, with no coverage), then scan the tile
        // coverage: block-local offsets plus block totals, then the block
        // totals into exclusive prefixes and a grand total.
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
        // True pairs before each block (`before[numBlocks]` is the total):
        // the GPU scan is u32 and wraps past 2³² pairs in a pass.
        const before = this.blockBefore;
        unwrapPrefixes(prefix, numBlocks, total[0], before);

        // Cut the depth-sorted list into ranges at scan-block boundaries so
        // each sort stays within the pair cap; a block that exceeds it on
        // its own is cut inside, from its per-splat offsets.
        let startBlock = 0;
        while (startBlock < numBlocks) {
            const rangeBase = before[startBlock];
            let endBlock = startBlock + 1;
            if (before[endBlock] - rangeBase > this.pairCap) {
                await this.rasterBlock(startBlock, count, rangeBase, before[endBlock], tilesX, tilesY);
            } else {
                while (endBlock < numBlocks && before[endBlock + 1] - rangeBase <= this.pairCap) {
                    endBlock++;
                }
                const rangePairs = before[endBlock] - rangeBase;
                if (rangePairs > 0) {
                    const splatStart = startBlock * SCAN_BLOCK;
                    const splatCount = Math.min(count, endBlock * SCAN_BLOCK) - splatStart;
                    this.rasterizeCut(splatStart, splatCount, rangeBase, rangePairs, tilesX, tilesY);
                }
            }
            startBlock = endBlock;
        }
    }

    /**
     * Rasterize one scan block whose pairs exceed the cap: read back its
     * block-local per-splat offsets and cut it into ranges within the cap.
     * One splat's pairs are at most the group's tiles, which always fit.
     *
     * @param block - The scan block.
     * @param count - Gaussians in the range being rasterized.
     * @param blockBase - Pairs before the block.
     * @param blockEnd - Pairs before the next block.
     * @param tilesX - Active group width in tiles.
     * @param tilesY - Active group height in tiles.
     */
    private async rasterBlock(block: number, count: number, blockBase: number, blockEnd: number, tilesX: number, tilesY: number): Promise<void> {
        const splatStart = block * SCAN_BLOCK;
        const splatCount = Math.min(count, splatStart + SCAN_BLOCK) - splatStart;
        const offsets = await this.emitOffsetBuffer!.read(splatStart * 4, splatCount * 4, this.blockOffsets, true) as Uint32Array;
        // Block-local pairs before a splat; past the last splat, the block's total.
        const localBefore = (j: number): number => (j < splatCount ? offsets[j] : blockEnd - blockBase);
        let start = 0;
        while (start < splatCount) {
            const base = localBefore(start);
            let end = start + 1;
            while (end < splatCount && localBefore(end + 1) - base <= this.pairCap) {
                end++;
            }
            const pairs = localBefore(end) - base;
            if (pairs > 0) {
                this.rasterizeCut(splatStart + start, end - start, blockBase + base, pairs, tilesX, tilesY);
            }
            start = end;
        }
    }

    /**
     * Emit, sort, bin and rasterize the pairs of one cut of the sorted list.
     *
     * @param splatStart - First splat of the cut (index into the sorted order).
     * @param splatCount - Splats in the cut.
     * @param rangeBase - Pair slot of the cut's first pair in the scan's numbering.
     * @param rangePairs - Pairs in the cut.
     * @param tilesX - Active group width in tiles.
     * @param tilesY - Active group height in tiles.
     */
    private rasterizeCut(splatStart: number, splatCount: number, rangeBase: number, rangePairs: number, tilesX: number, tilesY: number): void {
        const device = this.device;
        this.ensurePairCapacity(rangePairs);

        // The cut's pair count drives the sentinel and the boundary pass.
        // queue writes land before the commands recorded after them.
        this.totalReadback[0] = rangePairs;
        this.totalPairsBuffer.write(0, this.totalReadback, 0, 1);

        const emit = this.emitCompute;
        emit.setParameter('tileKeys', this.tileKeysBuffer!);
        emit.setParameter('splatValues', this.splatValuesBuffer!);
        emit.setParameter('chunkSize', splatCount);
        emit.setParameter('rangeStart', splatStart);
        // The shader subtracts the base from the u32 block prefixes, so the
        // true base modulo 2³² gives the right slots for any cut under 2³² pairs.
        emit.setParameter('emitBase', rangeBase % 2 ** 32);
        this.dispatch2D(emit, Math.ceil(splatCount / 64), 'scene-emit-pairs');

        // Stable sort by tile keeps each tile's pairs in emission (depth) order.
        // The sort grows buffers only when its workgroup count grows.
        // Reserve whole workgroups using the active sorter's granularity.
        const sortBlock = this.radixSort.prepareIndirect()[1];
        this.radixSort.capacity = Math.max(this.radixSort.capacity, Math.ceil(rangePairs / sortBlock) * sortBlock);
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

        // Capture this cut's uniforms before the next cut overwrites them.
        this.submit();
    }

    /**
     * Upload one streamed range into the range buffers and release its
     * chunk buffers (the queue copies on write).
     *
     * @param range - The range.
     */
    private uploadRange(range: StreamedRange): void {
        const n = range.count;
        if (n > this.capacity) {
            throw new Error(`GpuSceneRasterizer: streamed range of ${n} exceeds the prepared ${this.capacity} rows`);
        }
        this.positionBuffer!.write(0, new Float32Array(range.position.data, 0, n * POSITION_F32), 0, n * POSITION_F32);
        this.geometricBuffer!.write(0, new Float32Array(range.geometric.data, 0, n * GEOMETRIC_F32), 0, n * GEOMETRIC_F32);
        this.colorBuffer!.write(0, new Float32Array(range.color.data, 0, n * this.colorF32), 0, n * this.colorF32);
        range.position.release();
        range.geometric.release();
        range.color.release();
    }

    /**
     * Read the source's LOD 0 chunk by chunk and hand each to `sink` with its
     * row offset. Buffers are released after the sink returns.
     *
     * @param source - The scene.
     * @param pool - Pool for the read buffers.
     * @param layers - Layers to read.
     * @param sink - Consumes one chunk.
     */
    private async readChunks(
        source: ChunkSource,
        pool: ChunkDataPool,
        layers: SceneLayer[],
        sink: (rowStart: number, count: number, data: Partial<Record<SceneLayer, ChunkData>>) => void
    ): Promise<void> {
        const { meta } = source;
        const n = meta.lodCounts[0];
        const chunkSize = meta.chunkSize;
        const numChunks = meta.numChunks[0] ?? 0;
        for (let k = 0; k < numChunks; k++) {
            const count = Math.min(chunkSize, n - k * chunkSize);
            const data: Partial<Record<SceneLayer, ChunkData>> = {};
            const request: ReadRequest = { chunkIndex: k, lod: 0, ...data };
            for (const layer of layers) {
                const cd = pool.acquire(layer, meta.layouts[layer]!, count);
                data[layer] = cd;
                (request as Record<string, unknown>)[layer] = cd;
            }
            await source.read(request);
            sink(k * chunkSize, count, data);
            for (const layer of layers) data[layer]!.release();
        }
    }

    /**
     * Run `allocate` inside an out-of-memory error scope, releasing the scene
     * and throwing {@link ResidentUploadError} if the device reports one.
     *
     * @param allocate - Creates the buffers.
     */
    private async guardAllocation(allocate: () => void): Promise<void> {
        // @ts-ignore - wgpu is the underlying GPUDevice on WebgpuGraphicsDevice.
        const wgpu = (this.device as { wgpu?: { pushErrorScope?: (f: string) => void; popErrorScope?: () => Promise<{ message: string } | null> } }).wgpu;
        wgpu?.pushErrorScope?.('out-of-memory');
        let oom: { message: string } | null | undefined;
        try {
            allocate();
        } finally {
            oom = await wgpu?.popErrorScope?.();
        }
        if (oom) {
            this.releaseScene();
            throw new ResidentUploadError(oom.message);
        }
    }

    /**
     * Keys buffer for `n` gaussians, and the depth sorter's ping-pong buffers:
     * the sorter allocates on its first sort, so run one on the still-empty
     * keys here, inside the caller's error scope.
     *
     * @param n - Gaussians in the scene.
     */
    private allocateSort(n: number): void {
        // The depth sort borrows this as one of its ping-pong buffers
        // (destructive keys), so it needs the sorter's copy usages.
        this.depthKeysBuffer = new StorageBuffer(this.device, Math.max(4, n * 4), BUFFERUSAGE_COPY_SRC | BUFFERUSAGE_COPY_DST);
        if (n > 0) {
            this.depthSort.sort(this.depthKeysBuffer, n, 32, undefined, true, true);
            this.submit();
        }
    }

    /**
     * Range buffers for the streamed path: the three layers for `rangeRows`
     * gaussians, the identity order, and the working buffers.
     *
     * @param rangeRows - Most gaussians per range.
     * @param numSHBands - Sizes the colour buffer.
     */
    private allocateRange(rangeRows: number, numSHBands: 0 | 1 | 2 | 3): void {
        const device = this.device;
        this.positionBuffer = new StorageBuffer(device, rangeRows * POSITION_F32 * 4, BUFFERUSAGE_COPY_DST);
        this.geometricBuffer = new StorageBuffer(device, rangeRows * GEOMETRIC_F32 * 4, BUFFERUSAGE_COPY_DST);
        this.colorBuffer = new StorageBuffer(device, rangeRows * colorStride(numSHBands), BUFFERUSAGE_COPY_DST);
        this.identityOrderBuffer = new StorageBuffer(device, rangeRows * 4, BUFFERUSAGE_COPY_DST);
        const identity = new Uint32Array(rangeRows);
        for (let i = 0; i < rangeRows; i++) identity[i] = i;
        this.identityOrderBuffer.write(0, identity, 0, rangeRows);
        this.allocateWorking(rangeRows);
    }

    /**
     * Per-row working buffers for `rows` gaussians: projection records,
     * coverage, block-local scan offsets and block sums.
     *
     * @param rows - Rows the buffers hold.
     */
    private allocateWorking(rows: number): void {
        const device = this.device;
        const numBlocks = Math.ceil(rows / SCAN_BLOCK);
        this.projBuffer = new StorageBuffer(device, Math.max(4, rows * PROJECTION_STRIDE_F32 * 4), 0);
        this.coverageBuffer = new StorageBuffer(device, Math.max(4, rows * 4), 0);
        // Read back one block at a time when a block's pairs exceed the cap.
        this.emitOffsetBuffer = new StorageBuffer(device, Math.max(4, rows * 4), BUFFERUSAGE_COPY_SRC);
        this.blockSumsBuffer = new StorageBuffer(device, Math.max(1, numBlocks) * 4, BUFFERUSAGE_COPY_SRC);
        this.blockPrefix = new Uint32Array(numBlocks + 1);
        this.blockBefore = new Float64Array(numBlocks + 1);
    }

    private bindWorking(): void {
        this.depthKeysCompute.setParameter('sortKeys', this.depthKeysBuffer ?? this.coverageBuffer);
        this.projectCompute.setParameter('position', this.positionBuffer);
        this.projectCompute.setParameter('geometric', this.geometricBuffer);
        this.projectCompute.setParameter('color', this.colorBuffer);
        if (this.streamed) this.projectCompute.setParameter('order', this.identityOrderBuffer);
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

    private setUniforms(view: SceneView, gx: number, gy: number, tilesX: number, tilesY: number, sliceIndex: number, sliceCount: number): void {
        const o = this.options;
        const b = view.basis;
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
        if (this.accumulateCompute) computes.push(this.accumulateCompute);
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
            c.setParameter('offsetX', b.offsetX ?? 0); c.setParameter('offsetY', b.offsetY ?? 0);
            c.setParameter('imageWidth', o.imageWidth); c.setParameter('imageHeight', o.imageHeight);
            c.setParameter('splatStride', 0);
            c.setParameter('chunkSize', this.capacity);
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
            c.setParameter('numSplats', this.numSplats);
            c.setParameter('rangeStart', 0);
            c.setParameter('emitBase', 0);
            c.setParameter('sliceIndex', sliceIndex);
            c.setParameter('sliceCount', sliceCount);
            c.setParameter('_p7', 0); c.setParameter('_p8', 0); c.setParameter('_p9', 0);
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
        this.positionBuffer?.destroy();
        this.geometricBuffer?.destroy();
        this.colorBuffer?.destroy();
        this.scenePositionBuffer?.destroy();
        this.depthKeysBuffer?.destroy();
        this.identityOrderBuffer?.destroy();
        this.projBuffer?.destroy();
        this.coverageBuffer?.destroy();
        this.emitOffsetBuffer?.destroy();
        this.blockSumsBuffer?.destroy();
        this.positionBuffer = this.geometricBuffer = this.colorBuffer = this.scenePositionBuffer = null;
        this.depthKeysBuffer = this.identityOrderBuffer = this.projBuffer = null;
        this.coverageBuffer = this.emitOffsetBuffer = this.blockSumsBuffer = null;
        this.orderReadback = new Uint32Array(0);
        this.numSplats = 0;
        this.capacity = 0;
        this.streamed = false;
        // The sorter keeps its ping-pong buffers across sorts of one count:
        // a fresh one, so a fallback after a refused allocation never reuses
        // the failed buffers and an unused sorter holds nothing.
        this.depthSort.destroy();
        this.depthSort = new ComputeRadixSort(this.device);
    }

    /** Release all GPU resources. */
    destroy(): void {
        this.releaseScene();
        this.tileKeysBuffer?.destroy();
        this.splatValuesBuffer?.destroy();
        this.visibleCountBuffer.destroy();
        this.tileOffsetsBuffer.destroy();
        this.totalPairsBuffer.destroy();
        this.runningStateBuffer.destroy();
        this.outputBuffer.destroy();
        this.accumBuffer?.destroy();
        this.radixSort.destroy();
        this.depthSort.destroy();
        for (const s of this.shaders) s.destroy();
        for (const f of this.bgFormats) f.destroy();
    }
}

export { GpuSceneRasterizer, ResidentUploadError, unwrapPrefixes, type SceneRasterizerOptions, type SceneView, type StreamedRange, type SortedOrder };
