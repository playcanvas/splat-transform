import { type GraphicsDevice, Quat, Vec3 } from 'playcanvas';

import { Column, DataTable } from '../data-table';
import { GpuSplatRasterizer } from '../gpu';
import { type RenderCamera, buildCameraBasis } from '../render/camera';
import { PAIR_BUFFER_BUDGET_BYTES, PAIR_BUFFER_TOTAL_BYTES_PER_ELEMENT, TILE_SIZE } from '../render/config';
import {
    SortScratch, getSplatColumnRefs, packChunkInput, sortCandidatesByDepth, splatInputStride
} from '../render/preprocess';
import { Transform } from '../utils';

/**
 * Rendering side of the per-leaf LOD error metric: the views a leaf is judged
 * from, the renderers that draw its levels from those views (alone, or many
 * small leaves per image), and the image comparison. The writer gathers the
 * levels and orchestrates.
 */

type Aabb = {
    min: number[],
    max: number[]
};

/** A leaf's levels, gathered as tables in one shared space, plus its ellipsoid AABB. */
type LeafLevels = {
    bound: Aabb;
    levels: Map<number, DataTable>;
};

/**
 * Most pixels across a leaf's render. This is the metric's calibration: it fixes
 * the on-screen size at which a level is judged, since detail finer than a pixel
 * of the render is invisible to it. 512 across a leaf corresponds to a leaf
 * spanning roughly a quarter of a 2K-wide viewport — the band in which the
 * engine's budget allocator is actually trading levels off against each other.
 * The nearest leaves are bought first whatever their error says, and the far
 * field sits at its coarsest level regardless, so accuracy elsewhere buys little.
 */
const ERROR_VIEW_MAX_SIZE = 512;

/** Fewest pixels across a leaf's render. */
const ERROR_VIEW_MIN_SIZE = 64;

/**
 * Camera distance from the leaf centre, in bounding-sphere radii, when a leaf is
 * rendered alone. Far enough to be orthographic to within a pixel: the view
 * direction varies by under 0.3 degrees across the frame, so a thin slab seen
 * edge-on stays a sliver at the frame's edges as at its centre, and a leaf
 * measures the same alone as in an atlas, whose camera is equally distant.
 * Positions stay well within f32 precision at this range.
 */
const ERROR_VIEW_DISTANCE = 200;

/** Margin around a leaf's bounding sphere in its frame, as a factor of the radius. */
const ERROR_VIEW_MARGIN = 1.05;

/**
 * The six directions a leaf is viewed from, each with an up vector: the coordinate
 * axes under one fixed, generic rotation. Colour is stored as spherical harmonics
 * and evaluated along the view direction, and on a bare axis the basis functions
 * xy, yz, xz, xyz and z(x² − y²) are exactly zero, so colour held in those
 * coefficients would be invisible to axis-aligned views. Under this rotation every
 * basis function of bands 1 to 3 has magnitude at least 0.09 in every view.
 * Opposite views still agree on even bands and negate odd ones, so six views sample
 * each band at three directions: band 1 is seen whole, bands 2 and 3 in part.
 */
const ERROR_VIEWS: { direction: Vec3, up: Vec3 }[] = (() => {
    const rotation = new Quat().setFromEulerAngles(45, 20, 35);
    const axes = [Vec3.RIGHT, Vec3.UP, Vec3.FORWARD].map(axis => rotation.transformVector(axis, new Vec3()));
    return axes.flatMap((axis, i) => [1, -1].map(sign => ({
        direction: axis.clone().mulScalar(sign),
        up: axes[(i + 1) % 3]
    })));
})();

/** Gaussians per GPU dispatch; bounds the input and pair buffers. */
const CHUNK_CAP = 200_000;

/**
 * Renders the solo path keeps in flight before waiting for their readbacks: a
 * leaf's six views. A render's wall time is mostly the GPU round trip, not the
 * work, so issuing the views together and waiting once turns six round trips
 * into one.
 */
const SOLO_SLOTS = ERROR_VIEWS.length;

/** Gaussians sampled when estimating a leaf's finest splat footprint. */
const FOOTPRINT_SAMPLES = 4096;

/**
 * Most pixels across the image that batches of leaves are rendered into. 256
 * tiles a side: with cells two frames wide this holds 16 leaves at the full
 * 512-pixel frame and over a thousand at a 64-pixel one. Image sizes are kept to
 * powers of two. The rasterizer keeps 16 bytes of running state per pixel in one
 * storage buffer, 256 MiB at this size, so a device whose storage binding limit
 * is smaller gets a smaller atlas ({@link ErrorRenderer}).
 */
const ATLAS_MAX_SIZE = 4096;

/** Bytes of rasterizer running state per atlas pixel, one vec4f. */
const ATLAS_STATE_BYTES_PER_PIXEL = 16;

/**
 * Atlas cells are this many leaf frames across, the leaf frame centred, so that a
 * neighbour's splats cannot reach into a leaf's frame. A leaf is normalised to
 * unit bounding radius and the rasterizer cuts a splat off at three sigma, so a
 * splat of sigma `s` reaches `1 + 3 s` units from its cell centre; the next
 * cell's frame edge is `2 * 2.1 - 1.05 = 3.15` units away, so `s <= 0.5` leaves
 * a margin of over half a unit for the rasterizer's anti-alias dilation and the
 * pixel rounding of the frame. Leaves with a larger splat are rendered alone
 * ({@link fitsAtlas}).
 */
const ATLAS_CELL_FRAMES = 2;

/**
 * Largest splat sigma, as a fraction of the leaf's bounding radius, for the leaf
 * to share an atlas without reaching a neighbour's frame — see
 * {@link ATLAS_CELL_FRAMES}. A leaf's bound is the union of its splats' one-sigma
 * boxes, so the ratio never exceeds 1; for an isotropic splat alone in its leaf it
 * is 1 / sqrt(3), so such leaves render alone.
 */
const ATLAS_MAX_SIGMA_RATIO = 0.5;

/**
 * Camera distance for an atlas, as a multiple of the atlas width. Cells sit off
 * the camera axis, so this sets how far their view direction departs from the
 * one a solo render has: at 200 widths it is under 0.15 degrees,
 * under a pixel of skew across a frame.
 */
const ATLAS_DISTANCE = 200;

/** Largest leaf frame that goes into an atlas. */
const ATLAS_MAX_FRAME = ERROR_VIEW_MAX_SIZE;

/** Most gaussians in a leaf's finest level for it to go into an atlas. */
const ATLAS_MAX_GAUSSIANS = 8192;

/** Most finest-level gaussians in one atlas batch; keeps a batch to a few GPU chunks. */
const ATLAS_MAX_BATCH_GAUSSIANS = 1 << 16;

/**
 * Pixels across the renders of a leaf: enough to resolve the reference level's
 * finest content, up to {@link ERROR_VIEW_MAX_SIZE}.
 *
 * Beyond the point where every splat spans a few pixels, more resolution changes
 * nothing about a relative image error, so it only costs time. A dense leaf, whose
 * footprints are far smaller than a pixel at any affordable size, gets the full
 * calibration; a sparse leaf of large soft splats is judged at a size that resolves
 * them and no more, which is also what lets it share an atlas with other leaves.
 *
 * The footprint is the tenth percentile of the per-gaussian largest scale over a
 * sample of the level, so a few degenerate splats cannot drive the size.
 *
 * @param bound - The leaf's ellipsoid AABB.
 * @param reference - The leaf's finest level.
 * @returns A multiple of the tile size in `[ERROR_VIEW_MIN_SIZE, ERROR_VIEW_MAX_SIZE]`.
 */
const leafViewSize = (bound: Aabb, reference: DataTable): number => {
    const sx = reference.getColumnByName('scale_0')!.data as Float32Array;
    const sy = reference.getColumnByName('scale_1')!.data as Float32Array;
    const sz = reference.getColumnByName('scale_2')!.data as Float32Array;
    const n = sx.length;
    const step = Math.max(1, Math.floor(n / FOOTPRINT_SAMPLES));
    const radii: number[] = [];
    for (let i = 0; i < n; i += step) {
        radii.push(Math.exp(Math.max(sx[i], sy[i], sz[i])));
    }
    radii.sort((a, b) => a - b);
    const footprint = Math.max(radii[Math.floor(radii.length * 0.1)], 1e-6);

    const { min, max } = bound;
    const extent = Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
    // four pixels across the finest footprint's diameter
    const wanted = 2 * extent / footprint;
    const size = Math.min(ERROR_VIEW_MAX_SIZE, Math.max(ERROR_VIEW_MIN_SIZE, wanted));
    return Math.ceil(size / TILE_SIZE) * TILE_SIZE;
};

const boundRadius = (bound: Aabb): number => {
    const { min, max } = bound;
    return Math.max(Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / 2, 1e-3);
};

/**
 * The six views of a leaf rendered alone: pinhole cameras along each of the
 * {@link ERROR_VIEWS}, far enough out to frame the leaf's bounding sphere with a
 * little margin.
 *
 * @param bound - The leaf's ellipsoid AABB (the same volume the engine derives its
 * screen coverage from).
 * @param size - Pixels across each render, from {@link leafViewSize}.
 * @returns One camera per view direction.
 */
const leafCameras = (bound: Aabb, size: number): RenderCamera[] => {
    const { min, max } = bound;
    const cx = (min[0] + max[0]) / 2;
    const cy = (min[1] + max[1]) / 2;
    const cz = (min[2] + max[2]) / 2;
    const radius = boundRadius(bound);
    const distance = ERROR_VIEW_DISTANCE * radius;
    const fovY = 2 * Math.atan(ERROR_VIEW_MARGIN * radius / distance);

    return ERROR_VIEWS.map(({ direction, up }) => ({
        projection: 'pinhole',
        position: new Vec3(cx + direction.x * distance, cy + direction.y * distance, cz + direction.z * distance),
        target: new Vec3(cx, cy, cz),
        up,
        fovY,
        width: size,
        height: size,
        // the leaf occupies [distance - radius, distance + radius] in depth
        near: radius
    }));
};

/**
 * Renders a table from a view into a square RGBA image over a transparent black
 * background, so coverage lands in alpha and colour arrives premultiplied by it:
 * thinning and colour drift both register as a difference.
 *
 * One instance serves every render of one image size. The GPU rasterizer compiles
 * its pipelines per instance and the device caches them by bind-group identity,
 * so creating a rasterizer per render (as the general image writer does) would
 * recompile eight pipelines per image and grow that cache without bound over the
 * tens of thousands of renders a scene needs. Here only the view changes between
 * renders.
 */
class ViewRenderer {
    private device: GraphicsDevice;

    private rasterizer: GpuSplatRasterizer;

    private numSHBands: 0 | 1 | 2 | 3;

    private chunkCap: number;

    private chunkInput: Float32Array;

    private indices = new Uint32Array(0);

    private sortScratch = new SortScratch();

    /**
     * @param device - Graphics device the renders run on.
     * @param numSHBands - The scene's SH band count; fixes the input stride.
     * @param maxSize - Most pixels across an image; sizes the group.
     * @param maxCoveragePerSplat - Most tiles any one splat's footprint can cover, a
     * power of two; sizes the pair buffers.
     * @param slots - Renders that may be in flight at once; see {@link issue}.
     */
    constructor(device: GraphicsDevice, numSHBands: 0 | 1 | 2 | 3, maxSize: number, maxCoveragePerSplat: number, slots = 1) {
        const maxTiles = maxSize / TILE_SIZE;

        // Same two GPU limits the image writer honours: each pair buffer must fit one
        // storage binding, and all of them together must fit the pair budget.
        // @ts-ignore - limits is exposed by WebgpuGraphicsDevice
        const wgpuLimits = (device as { limits?: { maxStorageBufferBindingSize?: number } }).limits;
        const maxBindingBytes = wgpuLimits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
        this.chunkCap = Math.max(1, Math.min(
            CHUNK_CAP,
            Math.floor(maxBindingBytes / (maxCoveragePerSplat * 4)),
            Math.floor(PAIR_BUFFER_BUDGET_BYTES / (maxCoveragePerSplat * PAIR_BUFFER_TOTAL_BYTES_PER_ELEMENT))
        ));

        this.device = device;
        this.numSHBands = numSHBands;
        this.chunkInput = new Float32Array(this.chunkCap * splatInputStride(numSHBands));

        // The view fields are placeholders; setView supplies them per render.
        this.rasterizer = new GpuSplatRasterizer(device, {
            numSHBands,
            projection: 'pinhole',
            groupTilesX: maxTiles,
            groupTilesY: maxTiles,
            chunkCap: this.chunkCap,
            slots,
            maxCoveragePerSplat,
            imageWidth: maxSize,
            imageHeight: maxSize,
            near: 0,
            rightX: 1,
            rightY: 0,
            rightZ: 0,
            downX: 0,
            downY: 1,
            downZ: 0,
            forwardX: 0,
            forwardY: 0,
            forwardZ: 1,
            eyeX: 0,
            eyeY: 0,
            eyeZ: 0,
            focalX: 1,
            focalY: 1,
            focusDistance: 0,
            apertureScale: 0,
            bgR: 0,
            bgG: 0,
            bgB: 0,
            bgA: 0
        });
    }

    /**
     * Encode a render and start its readback without waiting for it. Up to `slots`
     * renders may be issued into distinct slots before {@link settle}; the
     * per-chunk buffers they share are safe because the queue executes in order.
     *
     * @param table - Splats in the same space as `camera`.
     * @param camera - The view; its image must fit the constructed size.
     * @param slot - Which in-flight slot to render into.
     * @returns The RGBA bytes of the render, `camera.width` square.
     */
    issue(table: DataTable, camera: RenderCamera, slot = 0): Promise<Uint8Array> {
        const count = table.numRows;
        const basis = buildCameraBasis(camera);

        // No frustum cull: every centre lies inside the framed bound, in front of
        // the camera. The project shader still drops anything the view misses.
        if (this.indices.length < count) this.indices = new Uint32Array(count);
        const indices = this.indices;
        for (let i = 0; i < count; i++) indices[i] = i;

        const cols = getSplatColumnRefs(table, this.numSHBands);
        sortCandidatesByDepth(cols, indices, count, basis, 'pinhole', this.sortScratch);

        const tiles = camera.width / TILE_SIZE;
        this.rasterizer.setView(basis, camera.near, camera.width, camera.height);
        this.rasterizer.beginGroup(0, 0, tiles, tiles, slot);
        for (let start = 0; start < count; start += this.chunkCap) {
            const size = Math.min(this.chunkCap, count - start);
            packChunkInput(cols, indices, start, size, this.numSHBands, this.chunkInput);
            this.rasterizer.dispatchChunk(this.chunkInput, size);
        }
        return this.rasterizer.finishGroup();
    }

    /**
     * Wait for issued renders, then end the device frame. Each chunk reserves
     * slots in the device's indirect-dispatch buffer, which the device only
     * recycles at frame end; the readbacks have drained the queue, so ending the
     * frame here is safe and keeps a pass of any length within the buffer.
     *
     * @param pending - The renders issued since the last settle.
     * @returns Their images, in order.
     */
    async settle(pending: Promise<Uint8Array>[]): Promise<Uint8Array[]> {
        const images = await Promise.all(pending);
        this.device.frameEnd();
        return images;
    }

    /**
     * @param table - Splats in the same space as `camera`.
     * @param camera - The view; its image must fit the constructed size.
     * @returns The RGBA bytes of the render, `camera.width` square.
     */
    async render(table: DataTable, camera: RenderCamera): Promise<Uint8Array> {
        return (await this.settle([this.issue(table, camera)]))[0];
    }

    destroy(): void {
        this.rasterizer.destroy();
    }
}

/**
 * @param image - RGBA bytes.
 * @param width - Image width in pixels.
 * @param x0 - Left edge of the rectangle.
 * @param y0 - Top edge of the rectangle.
 * @param size - Pixels across the square rectangle.
 * @returns The sum of squared channel values over the rectangle, channels in [0, 1].
 */
const sumSquaredRect = (image: Uint8Array, width: number, x0: number, y0: number, size: number): number => {
    let total = 0;
    for (let y = y0; y < y0 + size; y++) {
        const row = (y * width + x0) * 4;
        for (let i = row; i < row + size * 4; i++) {
            const v = image[i] / 255;
            total += v * v;
        }
    }
    return total;
};

/**
 * @param a - RGBA bytes.
 * @param b - RGBA bytes of the same size.
 * @param width - Image width in pixels.
 * @param x0 - Left edge of the rectangle.
 * @param y0 - Top edge of the rectangle.
 * @param size - Pixels across the square rectangle.
 * @returns The sum of squared channel differences over the rectangle, channels in [0, 1].
 */
const sumSquaredDifferenceRect = (
    a: Uint8Array, b: Uint8Array, width: number, x0: number, y0: number, size: number
): number => {
    let total = 0;
    for (let y = y0; y < y0 + size; y++) {
        const row = (y * width + x0) * 4;
        for (let i = row; i < row + size * 4; i++) {
            const d = (a[i] - b[i]) / 255;
            total += d * d;
        }
    }
    return total;
};

/** A leaf placed in an atlas: where its frame lands and how to move its splats there. */
type AtlasSlot = {
    leaf: LeafLevels;
    /** Leaf centre in scene space. */
    centre: number[];
    /** Scene units per atlas unit: the leaf's bounding radius. */
    radius: number;
    /** Atlas-plane coordinates of the cell centre, in atlas units. */
    u: number;
    v: number;
};

/**
 * One level of a batch of leaves, each normalised to unit bounding radius and
 * moved to its cell on the atlas plane. Translation and uniform scaling leave a
 * render unchanged up to the pixel grid, so rotations, opacities and colours are
 * copied as they are; log scales shift by the scale factor.
 *
 * @param members - The slots whose leaves have this level.
 * @param lod - The level.
 * @param right - Camera right axis: atlas `u` runs along it.
 * @param down - Camera down axis: atlas `v` runs along it.
 * @returns The level's splats for the whole batch, in atlas space.
 */
const assembleAtlas = (members: AtlasSlot[], lod: number, right: Vec3, down: Vec3): DataTable => {
    const first = members[0].leaf.levels.get(lod)!;
    const names = first.columns.map(c => c.name);
    const total = members.reduce((sum, m) => sum + m.leaf.levels.get(lod)!.numRows, 0);
    const out = names.map(() => new Float32Array(total));

    const ix = first.getColumnIndex('x'), iy = first.getColumnIndex('y'), iz = first.getColumnIndex('z');
    const scaleIndices = ['scale_0', 'scale_1', 'scale_2'].map(name => first.getColumnIndex(name));

    let row = 0;
    for (const m of members) {
        const table = m.leaf.levels.get(lod)!;
        const n = table.numRows;
        for (let c = 0; c < names.length; c++) out[c].set(table.columns[c].data as Float32Array, row);

        const inv = 1 / m.radius;
        const logInv = Math.log(inv);
        const ox = right.x * m.u + down.x * m.v;
        const oy = right.y * m.u + down.y * m.v;
        const oz = right.z * m.u + down.z * m.v;
        const x = out[ix], y = out[iy], z = out[iz];
        for (let r = row; r < row + n; r++) {
            x[r] = (x[r] - m.centre[0]) * inv + ox;
            y[r] = (y[r] - m.centre[1]) * inv + oy;
            z[r] = (z[r] - m.centre[2]) * inv + oz;
        }
        for (const s of scaleIndices) {
            const scale = out[s];
            for (let r = row; r < row + n; r++) scale[r] += logInv;
        }
        row += n;
    }

    return new DataTable(names.map((name, c) => new Column(name, out[c])), Transform.PLY);
};

/**
 * Measures per-leaf, per-level approximation error on rendered images.
 *
 * Each level present in a leaf is rendered from the six {@link ERROR_VIEWS}
 * and compared against the finest level present, which is the reference and
 * carries no error. A level's error is the squared RGBA difference summed over
 * every pixel of the leaf's frame in every view, divided by the reference's summed
 * squared RGBA — a relative image error that is zero for an identical level, grows
 * without bound as a level departs from the reference, and is dimensionless so
 * leaves of any physical size compare on one scale.
 *
 * A render costs about a millisecond whatever its size, so the pass is priced by
 * renders, and a scene cut into tens of thousands of small leaves would take
 * hours one leaf at a time. Small leaves are therefore batched: each is
 * normalised to unit bounding radius, placed in its own cell of an atlas in front
 * of one distant camera, and rendered together with the rest of the batch, once
 * per level per view. A leaf's frame in the atlas is the same size, seen from the
 * same six directions, as it would be alone; only the round trips are shared.
 * Leaves too detailed or too populous for that are still rendered alone.
 */
class ErrorRenderer {
    private solo: ViewRenderer;

    private atlas: ViewRenderer;

    /** Pixels across an atlas: {@link ATLAS_MAX_SIZE} or what the device can bind. */
    private atlasSize: number;

    /**
     * @param device - Graphics device the renders run on.
     * @param numSHBands - The scene's SH band count.
     */
    constructor(device: GraphicsDevice, numSHBands: 0 | 1 | 2 | 3) {
        // The atlas running state must fit one storage binding: the largest power
        // of two side whose pixels do. The WebGPU baseline of 128 MiB gives 2048.
        // @ts-ignore - limits is exposed by WebgpuGraphicsDevice
        const wgpuLimits = (device as { limits?: { maxStorageBufferBindingSize?: number } }).limits;
        const maxBindingBytes = wgpuLimits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
        this.atlasSize = Math.min(ATLAS_MAX_SIZE, 1 << Math.floor(Math.log2(Math.sqrt(maxBindingBytes / ATLAS_STATE_BYTES_PER_PIXEL))));

        // Alone, a splat's footprint can span the whole image.
        const soloTiles = ERROR_VIEW_MAX_SIZE / TILE_SIZE;
        this.solo = new ViewRenderer(device, numSHBands, ERROR_VIEW_MAX_SIZE, 1 << Math.ceil(Math.log2(soloTiles * soloTiles)), SOLO_SLOTS);

        // In an atlas a splat's footprint radius is at most three sigma, sigma at
        // most ATLAS_MAX_SIGMA_RATIO of a unit-radius leaf; widest at the widest frame.
        const largestCell = ATLAS_CELL_FRAMES * ATLAS_MAX_FRAME;
        const footprintPixels = 2 * 3 * ATLAS_MAX_SIGMA_RATIO * largestCell / (ATLAS_CELL_FRAMES * 2 * ERROR_VIEW_MARGIN);
        const footprintTiles = Math.ceil(footprintPixels / TILE_SIZE) + 2;
        this.atlas = new ViewRenderer(device, numSHBands, this.atlasSize, 1 << Math.ceil(Math.log2(footprintTiles * footprintTiles)));
    }

    /**
     * @param frame - Pixels across a leaf's frame, from {@link leafViewSize}.
     * @returns Most leaves of that frame one atlas holds.
     */
    atlasCapacity(frame: number): number {
        const cells = Math.floor(this.atlasSize / (ATLAS_CELL_FRAMES * frame));
        return cells * cells;
    }

    /**
     * Whether a leaf can share an atlas rather than needing renders of its own:
     * its frame and finest level must be small enough, and no splat of any level
     * may be large enough, relative to the leaf's bound, to reach a neighbour's
     * frame.
     *
     * @param frame - Pixels across the leaf's frame, from {@link leafViewSize}.
     * @param leaf - The leaf's levels and bound.
     * @returns True when the leaf goes into an atlas.
     */
    fitsAtlas(frame: number, leaf: LeafLevels): boolean {
        if (frame > ATLAS_MAX_FRAME) return false;
        const reference = leaf.levels.get(Math.min(...leaf.levels.keys()))!;
        if (reference.numRows > ATLAS_MAX_GAUSSIANS) return false;

        const limit = Math.log(ATLAS_MAX_SIGMA_RATIO * boundRadius(leaf.bound));
        for (const table of leaf.levels.values()) {
            for (const name of ['scale_0', 'scale_1', 'scale_2']) {
                const scale = table.getColumnByName(name)!.data as Float32Array;
                for (let i = 0; i < scale.length; i++) {
                    if (scale[i] > limit) return false;
                }
            }
        }
        return true;
    }

    /**
     * Errors of one leaf rendered alone.
     *
     * @param leaf - The leaf's levels and bound.
     * @param numLods - Number of structural LODs in the source.
     * @returns One raw (not yet monotone) error per LOD; zero for the reference and
     * for levels absent from the leaf.
     */
    async leafErrors(leaf: LeafLevels, numLods: number): Promise<number[]> {
        const errors = new Array(numLods).fill(0);
        const levels = [...leaf.levels.keys()].sort((a, b) => a - b);
        if (levels.length < 2) return errors;

        const reference = leaf.levels.get(levels[0])!;
        const size = leafViewSize(leaf.bound, reference);
        const cameras = leafCameras(leaf.bound, size);

        // All six views of a level go out together and are waited for once.
        const renderViews = (table: DataTable) => this.solo.settle(cameras.map((camera, v) => this.solo.issue(table, camera, v)));

        const referenceImages = await renderViews(reference);
        let energy = 0;
        for (const image of referenceImages) energy += sumSquaredRect(image, size, 0, 0, size);

        for (let i = 1; i < levels.length; i++) {
            const images = await renderViews(leaf.levels.get(levels[i])!);
            let difference = 0;
            for (let v = 0; v < cameras.length; v++) {
                difference += sumSquaredDifferenceRect(referenceImages[v], images[v], size, 0, 0, size);
            }
            // A reference that renders nothing leaves no energy to compare against: a
            // level that then draws anything is wholly wrong, one that draws nothing exact.
            errors[levels[i]] = energy > 0 ? difference / energy : (difference > 0 ? 1 : 0);
        }
        return errors;
    }

    /**
     * Errors of a batch of small leaves rendered together.
     *
     * Every leaf gets a frame of exactly `frame` pixels, the density a solo render
     * at that frame would have, in a cell three frames wide. The image is the
     * smallest power of two that holds the batch's cells, so a small or trailing
     * batch does not pay for a full-size readback.
     *
     * @param leaves - At most {@link atlasCapacity} leaves, each accepted by {@link fitsAtlas}.
     * @param frame - Pixels across each leaf's frame, at least each leaf's {@link leafViewSize}.
     * @param numLods - Number of structural LODs in the source.
     * @returns Per leaf, one raw (not yet monotone) error per LOD.
     */
    async atlasErrors(leaves: LeafLevels[], frame: number, numLods: number): Promise<number[][]> {
        const cells = Math.ceil(Math.sqrt(leaves.length));
        const cellPx = ATLAS_CELL_FRAMES * frame;
        const framePx = frame;
        const size = Math.min(this.atlasSize, 1 << Math.ceil(Math.log2(Math.max(TILE_SIZE, cells * cellPx))));
        const cellUnits = ATLAS_CELL_FRAMES * 2 * ERROR_VIEW_MARGIN;
        const pixelsPerUnit = cellPx / cellUnits;
        const extent = size / pixelsPerUnit;
        const distance = ATLAS_DISTANCE * extent;
        const fovY = 2 * Math.atan((size / 2) / (pixelsPerUnit * distance));

        const slots: AtlasSlot[] = leaves.map((leaf, k) => {
            const { min, max } = leaf.bound;
            return {
                leaf,
                centre: [0, 1, 2].map(i => (min[i] + max[i]) / 2),
                radius: boundRadius(leaf.bound),
                u: ((k % cells) - (cells - 1) / 2) * cellUnits,
                v: (Math.floor(k / cells) - (cells - 1) / 2) * cellUnits
            };
        });

        const levels = [...new Set(leaves.flatMap(leaf => [...leaf.levels.keys()]))].sort((a, b) => a - b);
        const referenceOf = leaves.map(leaf => Math.min(...leaf.levels.keys()));
        const energy = new Array(leaves.length).fill(0);
        const difference = leaves.map(() => new Array(numLods).fill(0));

        for (const { direction, up } of ERROR_VIEWS) {
            const camera: RenderCamera = {
                projection: 'pinhole',
                position: direction.clone().mulScalar(distance),
                target: new Vec3(0, 0, 0),
                up,
                fovY,
                width: size,
                height: size,
                near: distance - extent
            };
            const { right, down } = buildCameraBasis(camera);

            // Pixel rectangle of each leaf's frame, centred in its cell.
            const frames = slots.map(s => ({
                x0: Math.round(size / 2 + s.u * pixelsPerUnit - framePx / 2),
                y0: Math.round(size / 2 + s.v * pixelsPerUnit - framePx / 2)
            }));

            const images = new Map<number, Uint8Array>();
            for (const lod of levels) {
                const members = slots.filter(s => s.leaf.levels.has(lod));
                const table = assembleAtlas(members, lod, right, down);
                images.set(lod, await this.atlas.render(table, camera));
            }

            for (let k = 0; k < leaves.length; k++) {
                const reference = images.get(referenceOf[k])!;
                const { x0, y0 } = frames[k];
                energy[k] += sumSquaredRect(reference, size, x0, y0, framePx);
                for (const lod of leaves[k].levels.keys()) {
                    if (lod === referenceOf[k]) continue;
                    difference[k][lod] += sumSquaredDifferenceRect(reference, images.get(lod)!, size, x0, y0, framePx);
                }
            }
        }

        return leaves.map((leaf, k) => {
            const errors = new Array(numLods).fill(0);
            for (const lod of leaf.levels.keys()) {
                if (lod !== referenceOf[k]) errors[lod] = energy[k] > 0 ? difference[k][lod] / energy[k] : (difference[k][lod] > 0 ? 1 : 0);
            }
            return errors;
        });
    }

    destroy(): void {
        this.solo.destroy();
        this.atlas.destroy();
    }
}

export { ATLAS_MAX_BATCH_GAUSSIANS, ErrorRenderer, leafViewSize, type LeafLevels };
