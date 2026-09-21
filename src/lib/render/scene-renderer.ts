import { GraphicsDevice } from 'playcanvas';

import { type Projection, type RenderCamera, buildCameraBasis } from './camera';
import { TILE_SIZE, storageBindingLimit } from './config';
import { SortScratch, sortCandidatesByDepth } from './preprocess';
import { type ChunkDataPool, type ChunkSource, type SHBands, colorStride } from '../chunk';
import { GpuSceneRasterizer, ResidentUploadError, type SceneView, type SortedOrder, type StreamedRange } from '../gpu/gpu-scene-rasterizer';

/**
 * Pixel budget of one group (sub-frame): 4096², a 256 MB running-state
 * buffer, or less where the device's storage binding cannot hold that.
 * Larger images render as a grid of groups; each group projects the whole
 * visible list again with the group's cull, which is cheap next to the
 * per-group binning and rasterization it saves.
 */
const MAX_GROUP_PIXELS = 4096 * 4096;

/** Bytes per pixel of the largest group-sized buffers: the running state and the slice accumulator. */
const STATE_BYTES_PER_PIXEL = 16;

/** Bytes of the projection record per gaussian: the largest per-row working buffer. */
const PROJECTION_BYTES = 12 * 4;

/** Per-gaussian working bytes besides the layers: projection records, coverage, scan offsets. */
const WORKING_BYTES = PROJECTION_BYTES + 4 + 4;

/** Per-gaussian bytes of the depth sort: keys plus the sorter's three ping-pong buffers. */
const SORT_BYTES = 4 + 12;

interface BackgroundRGBA {
    r: number;
    g: number;
    b: number;
    a: number;
}

/**
 * How the scene is held for rendering, from most to least GPU memory:
 * `resident` (every layer on the GPU, nothing per pass), `streamed-gpu`
 * (positions and the depth sort on the GPU; attributes streamed from the
 * source per pass in sorted order), `streamed-cpu` (positions in RAM and the
 * sort on the CPU; only range-sized buffers on the GPU).
 */
type SceneTier = 'resident' | 'streamed-gpu' | 'streamed-cpu';

/** Fixed per-scene render settings; the camera pose varies per {@link SceneRenderer.render}. */
interface SceneRendererOptions {
    projection: Projection;
    width: number;
    height: number;
    background: BackgroundRGBA;
    /** Clamp frame-filling splats to the image; default true. */
    sizeClamp?: boolean;
    /** Most GPU bytes a tier may hold for the scene; tiers past it are skipped. Default: unlimited. */
    residentBudget?: number;
    /** Use this tier regardless of fit (tests); default picks the first that fits. */
    tier?: SceneTier;
}

/**
 * Bytes of GPU memory the resident tier holds for a scene: the three layers
 * as stored plus the per-splat working buffers (projection records,
 * coverage, scan offsets, depth keys and the depth sort's ping-pong buffers).
 *
 * @param numSplats - Row count.
 * @param numSHBands - SH bands above DC.
 * @returns Resident bytes.
 */
const residentSceneBytes = (numSplats: number, numSHBands: SHBands): number => {
    return numSplats * (12 + 32 + colorStride(numSHBands) + WORKING_BYTES + SORT_BYTES);
};

/**
 * Whether a scene's resident buffers each fit one storage binding on this
 * device: the largest per-splat buffer is the projection record or the
 * colour layer.
 *
 * @param device - The graphics device.
 * @param numSplats - Row count.
 * @param numSHBands - SH bands above DC.
 * @returns True when every per-scene buffer fits a binding.
 */
const residentSceneFits = (device: GraphicsDevice, numSplats: number, numSHBands: SHBands): boolean => {
    return Math.max(PROJECTION_BYTES, colorStride(numSHBands)) * numSplats <= storageBindingLimit(device);
};

/**
 * Renders many views of one scene held on the GPU, or streamed through it.
 *
 * Construct, `upload()` once, then `render()` per view. `upload` picks the
 * first {@link SceneTier} that fits the device's binding limit and the
 * caller's budget, falling to the next when the device refuses the
 * allocation. A render sends the GPU only the camera when resident; the
 * streamed tiers gather the gaussians' records from the source per pass in
 * depth order. See {@link GpuSceneRasterizer}.
 */
class SceneRenderer {
    private device: GraphicsDevice;
    private source: ChunkSource;
    private pool: ChunkDataPool;
    private options: SceneRendererOptions;
    private numSHBands: SHBands;
    private raster: GpuSceneRasterizer;
    /** Most gaussians per streamed range: a pool chunk, or fewer if a range buffer would exceed a binding. */
    private rangeRows: number;
    /** `streamed-cpu` only: the positions and the sort's scratch. */
    private positions: { x: Float32Array; y: Float32Array; z: Float32Array } | null = null;
    private candidates: Uint32Array | null = null;
    private sortScratch: SortScratch | null = null;

    /** Gaussians in the scene (LOD 0). */
    readonly numSplats: number;
    /** The tier `upload` chose, or null before it ran. */
    tier: SceneTier | null = null;
    /** GPU bytes the chosen tier holds for the scene. */
    gpuBytes = 0;

    constructor(device: GraphicsDevice, source: ChunkSource, pool: ChunkDataPool, options: SceneRendererOptions) {
        this.device = device;
        this.source = source;
        this.pool = pool;
        this.options = options;
        this.numSplats = source.meta.lodCounts[0];
        this.numSHBands = source.meta.shBands;
        const perRow = Math.max(PROJECTION_BYTES, colorStride(this.numSHBands));
        this.rangeRows = Math.max(1, Math.min(pool.chunkSize, Math.floor(storageBindingLimit(device) / perRow)));

        const imageTilesX = Math.ceil(options.width / TILE_SIZE);
        const imageTilesY = Math.ceil(options.height / TILE_SIZE);
        // Group tiles within the pixel budget and one storage binding.
        // Equirect binning wraps the X axis, so its X extent is never split.
        const maxGroupPixels = Math.min(MAX_GROUP_PIXELS, storageBindingLimit(device) / STATE_BYTES_PER_PIXEL);
        const maxGroupTiles = Math.floor(maxGroupPixels / (TILE_SIZE * TILE_SIZE));
        const groupTilesX = options.projection === 'equirect' ? imageTilesX : Math.min(imageTilesX, maxGroupTiles);
        const groupTilesY = Math.min(imageTilesY, Math.floor(maxGroupTiles / groupTilesX));
        if (groupTilesY < 1) {
            throw new Error(`SceneRenderer: a ${options.width}-pixel tile row exceeds the device's storage binding limit`);
        }
        const bg = options.background;
        this.raster = new GpuSceneRasterizer(device, {
            numSHBands: this.numSHBands,
            projection: options.projection,
            imageWidth: options.width,
            imageHeight: options.height,
            groupTilesX,
            groupTilesY,
            sizeClamp: options.sizeClamp,
            bgR: bg.r,
            bgG: bg.g,
            bgB: bg.b,
            bgA: bg.a
        });
    }

    /**
     * Load the scene into the first tier that fits. Throws
     * `ResidentUploadError` only if even the range buffers are refused.
     *
     * @returns The tier chosen.
     */
    async upload(): Promise<SceneTier> {
        const { device, source, pool, options, numSplats: n, numSHBands: bands } = this;
        const budget = options.residentBudget ?? Infinity;
        const forced = options.tier;
        const rangeBytes = this.rangeRows * (12 + 32 + colorStride(bands) + WORKING_BYTES + 4);

        if (forced === 'resident' || (forced === undefined && residentSceneFits(device, n, bands) && residentSceneBytes(n, bands) <= budget)) {
            try {
                await this.raster.uploadScene(source, pool);
                return this.chose('resident', residentSceneBytes(n, bands));
            } catch (e) {
                if (forced || !(e instanceof ResidentUploadError)) throw e;
            }
        }
        const streamedBytes = n * (12 + SORT_BYTES) + rangeBytes;
        if (forced === 'streamed-gpu' || (forced === undefined && 12 * n <= storageBindingLimit(device) && streamedBytes <= budget)) {
            try {
                await this.raster.uploadPositions(source, pool, this.rangeRows);
                return this.chose('streamed-gpu', streamedBytes);
            } catch (e) {
                if (forced || !(e instanceof ResidentUploadError)) throw e;
            }
        }
        await this.raster.prepareStreamed(n, this.rangeRows, bands);
        await this.loadPositions();
        return this.chose('streamed-cpu', rangeBytes);
    }

    private chose(tier: SceneTier, gpuBytes: number): SceneTier {
        this.tier = tier;
        this.gpuBytes = gpuBytes;
        return tier;
    }

    /**
     * Render one view.
     *
     * @param camera - Camera in the scene's space; must match the projection and
     * size the renderer was built with.
     * @returns RGBA bytes, `width × height × 4`.
     */
    render(camera: RenderCamera): Promise<Uint8Array> {
        return this.renderSlices([camera]);
    }

    /**
     * Render the mean of several views: the shutter samples of one
     * motion-blurred frame, each composited exactly at its instant,
     * accumulated on the GPU in float and quantized once.
     *
     * @param cameras - The samples; same constraints as {@link render}.
     * @returns RGBA bytes, `width × height × 4`.
     */
    renderSlices(cameras: RenderCamera[]): Promise<Uint8Array> {
        if (!this.tier) {
            throw new Error('SceneRenderer: upload before render');
        }
        const { projection, width, height } = this.options;
        const views = cameras.map((camera) => {
            if ((camera.projection ?? 'pinhole') !== projection || camera.width !== width || camera.height !== height) {
                throw new Error('SceneRenderer: camera projection or size differs from the renderer\'s');
            }
            return {
                basis: buildCameraBasis(camera),
                near: camera.near,
                focusDistance: camera.focusDistance ?? 0,
                apertureScale: camera.apertureScale ?? 0
            };
        });
        if (this.tier === 'resident') {
            return this.raster.render(views);
        }
        return this.raster.renderStreamed(views, view => this.ranges(view));
    }

    /**
     * The streamed tiers' range producer: the view's visible gaussians in
     * depth order, gathered from the source in ranges. The source coalesces
     * each gather into file-order reads internally.
     *
     * @param view - The view being rendered.
     * @yields One range at a time; the rasterizer releases its buffers.
     */
    private async *ranges(view: SceneView): AsyncGenerator<StreamedRange> {
        const { order, visible } = this.tier === 'streamed-gpu' ? await this.raster.sortedOrder() : this.cpuOrder(view);
        const { layouts } = this.source.meta;
        for (let base = 0; base < visible; base += this.rangeRows) {
            const count = Math.min(this.rangeRows, visible - base);
            const position = this.pool.acquire('position', layouts.position!, count);
            const geometric = this.pool.acquire('geometric', layouts.geometric!, count);
            const color = this.pool.acquire('color', layouts.color!, count);
            await this.source.read({ indices: order, indexOffset: base, count, lod: 0, position, geometric, color });
            yield { count, position, geometric, color };
        }
    }

    /**
     * `streamed-cpu`: read the position layer into RAM for the cull and sort.
     */
    private async loadPositions(): Promise<void> {
        const { source, pool, numSplats: n } = this;
        const x = new Float32Array(n);
        const y = new Float32Array(n);
        const z = new Float32Array(n);
        const { meta } = source;
        const numChunks = meta.numChunks[0] ?? 0;
        for (let k = 0; k < numChunks; k++) {
            const count = Math.min(meta.chunkSize, n - k * meta.chunkSize);
            const position = pool.acquire('position', meta.layouts.position!, count);
            await source.read({ chunkIndex: k, lod: 0, position });
            const p = new Float32Array(position.data, 0, count * 3);
            const rowStart = k * meta.chunkSize;
            for (let i = 0; i < count; i++) {
                x[rowStart + i] = p[i * 3];
                y[rowStart + i] = p[i * 3 + 1];
                z[rowStart + i] = p[i * 3 + 2];
            }
            position.release();
        }
        this.positions = { x, y, z };
        this.candidates = new Uint32Array(n);
        this.sortScratch = new SortScratch();
    }

    /**
     * `streamed-cpu`: the near-plane cull and depth sort on the CPU, with the
     * GPU key pass's tests.
     *
     * @param view - The view.
     * @returns The visible gaussians front to back.
     */
    private cpuOrder(view: SceneView): SortedOrder {
        const { x, y, z } = this.positions!;
        const candidates = this.candidates!;
        const { basis, near } = view;
        const { eye, forward } = basis;
        const n = this.numSplats;
        let count = 0;
        if (this.options.projection === 'pinhole') {
            for (let i = 0; i < n; i++) {
                const cz = forward.x * (x[i] - eye.x) + forward.y * (y[i] - eye.y) + forward.z * (z[i] - eye.z);
                if (cz > near) candidates[count++] = i;
            }
        } else {
            const nearSq = near * near;
            for (let i = 0; i < n; i++) {
                const dx = x[i] - eye.x, dy = y[i] - eye.y, dz = z[i] - eye.z;
                if (dx * dx + dy * dy + dz * dz > nearSq) candidates[count++] = i;
            }
        }
        sortCandidatesByDepth(this.positions!, candidates, count, basis, this.options.projection, this.sortScratch!);
        return { order: candidates, visible: count };
    }

    destroy(): void {
        this.raster.destroy();
        this.positions = this.candidates = this.sortScratch = null;
    }
}

export { SceneRenderer, residentSceneBytes, residentSceneFits, type SceneRendererOptions, type SceneTier };
