import { GraphicsDevice } from 'playcanvas';

import { type CameraBasis, type Projection, type RenderCamera, buildCameraBasis } from './camera';
import { TILE_SIZE, storageBindingLimit } from './config';
import { getSplatColumnRefs, numSHCoeffsPerChannel, sceneSHBands, type SplatColumnRefs } from './preprocess';
import { DataTable } from '../data-table';
import { GpuSceneRasterizer } from '../gpu/gpu-scene-rasterizer';

/**
 * Largest group (sub-frame) edge in tiles: 4096 px, a 256 MB running-state
 * buffer. Larger images render as a grid of groups; each group projects the
 * whole visible list again with the group's cull, which is cheap next to
 * the per-group binning and rasterization it saves.
 */
const MAX_GROUP_TILES = 4096 / TILE_SIZE;

interface BackgroundRGBA {
    r: number;
    g: number;
    b: number;
    a: number;
}

/** Fixed per-scene render settings; the camera pose varies per {@link SceneRenderer.render}. */
interface SceneRendererOptions {
    projection: Projection;
    width: number;
    height: number;
    /** Whether renders carry a shutter-close pose (motion blur). Fixed per instance. */
    motionBlur: boolean;
    background: BackgroundRGBA;
    /** Clamp frame-filling splats to the image; default true. */
    sizeClamp?: boolean;
}

/**
 * Bytes of GPU memory the resident path holds for a scene: the column
 * copies plus the per-splat working buffers (projection records, coverage,
 * scan offsets, depth keys and the depth sort's three ping-pong buffers).
 *
 * @param numSplats - Row count.
 * @param numSHBands - SH bands above DC.
 * @returns Resident bytes.
 */
const residentSceneBytes = (numSplats: number, numSHBands: number): number => {
    const coeffs = numSHCoeffsPerChannel(numSHBands);
    return numSplats * (14 * 4 + coeffs * 3 * 4 + 12 * 4 + 4 + 4 + 4 + 12);
};

/**
 * Whether a scene's resident buffers each fit one storage binding on this
 * device. The chunked path has no such limit and remains the fallback.
 *
 * @param device - The graphics device.
 * @param numSplats - Row count.
 * @param numSHBands - SH bands above DC.
 * @returns True when every per-scene buffer fits a binding.
 */
const residentSceneFits = (device: GraphicsDevice, numSplats: number, numSHBands: number): boolean => {
    const limit = storageBindingLimit(device);
    const coeffs = numSHCoeffsPerChannel(numSHBands);
    const largest = Math.max(14 * 4, coeffs * 3 * 4, 12 * 4) * numSplats;
    return largest <= limit;
};

/**
 * Renders many views of one scene held resident on the GPU.
 *
 * Construct, `upload()` once, then `render()` per view. A render sends the
 * GPU only the camera: the near-plane cull, the depth sort and the
 * attribute gather all run there. See {@link GpuSceneRasterizer}.
 */
class SceneRenderer {
    private device: GraphicsDevice;
    private dataTable: DataTable;
    private options: SceneRendererOptions;
    private cols: SplatColumnRefs;
    private numSHBands: 0 | 1 | 2 | 3;
    private raster: GpuSceneRasterizer;

    constructor(device: GraphicsDevice, dataTable: DataTable, options: SceneRendererOptions) {
        this.device = device;
        this.dataTable = dataTable;
        this.options = options;
        this.numSHBands = sceneSHBands(dataTable);
        this.cols = getSplatColumnRefs(dataTable, this.numSHBands);

        const imageTilesX = Math.ceil(options.width / TILE_SIZE);
        const imageTilesY = Math.ceil(options.height / TILE_SIZE);
        // Equirect binning wraps the X axis, so its X extent is never split.
        const groupTilesX = options.projection === 'equirect' ? imageTilesX : Math.min(imageTilesX, MAX_GROUP_TILES);
        const groupTilesY = Math.min(imageTilesY, MAX_GROUP_TILES);
        const bg = options.background;
        this.raster = new GpuSceneRasterizer(device, {
            numSHBands: this.numSHBands,
            projection: options.projection,
            imageWidth: options.width,
            imageHeight: options.height,
            groupTilesX,
            groupTilesY,
            sizeClamp: options.sizeClamp,
            motionBlur: options.motionBlur,
            bgR: bg.r,
            bgG: bg.g,
            bgB: bg.b,
            bgA: bg.a
        });
    }

    /**
     * Resident GPU bytes this scene needs.
     *
     * @returns Bytes.
     */
    get residentBytes(): number {
        return residentSceneBytes(this.dataTable.numRows, this.numSHBands);
    }

    /**
     * Upload the scene. Throws `ResidentUploadError` if the device runs out
     * of memory; the instance is then unusable and should be destroyed.
     *
     * @returns Resolves once the upload is issued and the error scope has cleared.
     */
    upload(): Promise<void> {
        return this.raster.uploadScene(this.cols, this.dataTable.numRows);
    }

    /**
     * Render one view.
     *
     * @param camera - Camera in the scene's space; must match the projection, size
     * and motion-blur setting the renderer was built with.
     * @returns RGBA bytes, `width × height × 4`.
     */
    render(camera: RenderCamera): Promise<Uint8Array> {
        return this.renderSlices([camera]);
    }

    /**
     * Render the mean of several views: the shutter slices of one
     * motion-blurred frame, accumulated on the GPU in float and quantized
     * once. Needs a renderer built with `motionBlur`.
     *
     * @param cameras - The slices, each carrying `shutterClose`; same constraints as {@link render}.
     * @returns RGBA bytes, `width × height × 4`.
     */
    renderSlices(cameras: RenderCamera[]): Promise<Uint8Array> {
        const { projection, width, height, motionBlur } = this.options;
        const views = cameras.map((camera) => {
            if ((camera.projection ?? 'pinhole') !== projection || camera.width !== width || camera.height !== height) {
                throw new Error('SceneRenderer: camera projection or size differs from the renderer\'s');
            }
            if (motionBlur !== (camera.shutterClose !== undefined)) {
                throw new Error('SceneRenderer: camera motion blur differs from the renderer\'s');
            }
            const basis = buildCameraBasis(camera);
            let basisB: CameraBasis | undefined;
            if (camera.shutterClose) {
                const { position, target, up } = camera.shutterClose;
                basisB = buildCameraBasis({ ...camera, position, target, up });
            }
            return {
                basis,
                basisB,
                near: camera.near,
                focusDistance: camera.focusDistance ?? 0,
                apertureScale: camera.apertureScale ?? 0
            };
        });
        return this.raster.render(views);
    }

    destroy(): void {
        this.raster.destroy();
    }
}

export { SceneRenderer, residentSceneBytes, residentSceneFits, type SceneRendererOptions };
