import {
    BUFFERUSAGE_COPY_DST,
    BUFFERUSAGE_COPY_SRC,
    SHADERLANGUAGE_WGSL,
    SHADERSTAGE_COMPUTE,
    UNIFORMTYPE_UINT,
    BindGroupFormat,
    BindStorageBufferFormat,
    BindUniformBufferFormat,
    Compute,
    ComputeRadixSort,
    GraphicsDevice,
    Shader,
    StorageBuffer,
    UniformBufferFormat,
    UniformFormat
} from 'playcanvas';

import { constantsChunk } from './shaders/chunks/constants';
import { covariance3D } from './shaders/chunks/covariance-3d';
import { jacobianEquirect } from './shaders/chunks/jacobian-equirect';
import { jacobianPinhole } from './shaders/chunks/jacobian-pinhole';
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
import { finalizeWgsl } from './shaders/finalize';
import { findBoundariesWgsl } from './shaders/find-boundaries';
import { initTileOffsetsWgsl } from './shaders/init-tile-offsets';
import { prefixSumWgsl } from './shaders/prefix-sum';
import { prepareIndirectWgsl } from './shaders/prepare-indirect';
import { projectWgsl } from './shaders/project';
import { rasterizeBinnedWgsl } from './shaders/rasterize-binned';
import { tileBinEmitPairsWgsl } from './shaders/tile-bin-emit-pairs';
import { uniformsStruct, uniformFormatEntries } from './shaders/uniforms';
import { type Projection } from '../render/camera';
import { TILE_SIZE } from '../render/config';

/** 12 floats per projected splat: vec4 × 3. */
const PROJECTION_STRIDE_F32 = 12;

/** 4 floats per group pixel: (R, G, B, T). */
const RUNNING_STATE_STRIDE_F32 = 4;

/** RGBA8 output is one u32 per group pixel. */
const OUTPUT_STRIDE_U32 = 1;

/**
 * Configuration for a `GpuSplatRasterizer`. Fixed across the lifetime of
 * a render — `numSHBands` and the group tile dimensions determine GPU
 * buffer sizes and shader uniform layouts.
 *
 * Sizes are expressed as a "group" tile rectangle (`groupTilesX ×
 * groupTilesY`). For a single-pass render the group covers the whole
 * image, so the buffers are exactly image-sized. The group abstraction
 * is retained as a hook for future subframe splitting (each subframe is
 * an independent group sharing the global depth sort) — the project
 * shader's group-AABB cull and group-pixel-origin uniforms still
 * exercise this code path.
 */
interface SplatRasterizerOptions {
    /** Number of SH bands above DC (0–3). Determines input stride. */
    numSHBands: 0 | 1 | 2 | 3;
    /**
     * Camera projection mode. Specializes the project, emit-pairs and
     * rasterize-binned shaders. `pinhole` (default) uses the classical
     * perspective + EWA Jacobian path; `equirect` uses spherical
     * (atan2/asin) screen mapping, a non-linear Jacobian, radial view
     * depth, and tile-bin / rasterize paths that wrap the X axis at the
     * ±π longitude seam.
     */
    projection: Projection;
    /** Tiles per group along X (≤ imageTilesX). Sizes runningState/output. */
    groupTilesX: number;
    /** Tiles per group along Y (≤ imageTilesY). Sizes runningState/output. */
    groupTilesY: number;
    /** Max gaussians per chunk; sizes the input + projection + pair buffers. */
    chunkCap: number;
    /**
     * Hard upper bound on per-splat tile coverage. The project shader
     * clamps `coverage[i] = min(rawBboxArea, maxCoveragePerSplat)`, so
     * the pair buffer is bounded by `chunkCap × maxCoveragePerSplat`
     * regardless of scene/screen size. If the cap ever bites, the
     * emit-pairs shader walks the bbox row-major and stops once it
     * has written `coverage[i]` pairs — i.e. it truncates the bbox at
     * its bottom-right corner.
     *
     * The orchestrator sets this to the group's full tile area so the
     * clamp is geometrically unreachable (any in-group bbox ≤ group
     * area ≤ cap), making truncation a non-issue in practice. The cap
     * is retained as a defensive ceiling on the pair buffer.
     */
    maxCoveragePerSplat: number;
    /** Output image width in pixels (constant per render). */
    imageWidth: number;
    /** Output image height in pixels (constant per render). */
    imageHeight: number;
    /** Near plane distance in world units. */
    near: number;
    /** Camera basis: rows are (right, down, forward) of the world→camera rotation. */
    rightX: number; rightY: number; rightZ: number;
    downX: number; downY: number; downZ: number;
    forwardX: number; forwardY: number; forwardZ: number;
    /** Camera eye position in world space. */
    eyeX: number; eyeY: number; eyeZ: number;
    /** Focal lengths in pixel units. */
    focalX: number; focalY: number;
    /**
     * Camera-space Z of the focus plane, world units. Pinhole-only;
     * unused when `projection === 'equirect'`.
     */
    focusDistance: number;
    /**
     * DoF strength as a pixel-space scalar: the CoC radius in pixels when
     * `|1 − focusDistance/cz| = 1`. `0` disables defocus. The writer
     * derives this from `--f-stop` + `--sensor-size` using the thin-lens
     * CoC formula. Pinhole-only.
     */
    apertureScale: number;
    /** RGBA background, each channel in [0, 1]. */
    bgR: number; bgG: number; bgB: number; bgA: number;
}

const numSHCoeffsPerChannel = (bands: number): number => {
    return bands === 0 ? 0 : bands === 1 ? 3 : bands === 2 ? 8 : 15;
};


interface PipelineBuffers {
    inputBuffer: StorageBuffer;
    projBuffer: StorageBuffer;
    runningStateBuffer: StorageBuffer;
    outputBuffer: StorageBuffer;
    /** Per-tile offset table for binned rasterize: `(numTiles + 1) × u32`. */
    tileOffsetsBuffer: StorageBuffer;
    /**
     * Per-splat tile-coverage count, written by the project shader.
     * Consumed by the GPU prefix-sum kernel; never read by the CPU.
     */
    coverageBuffer: StorageBuffer;
    /**
     * GPU-computed exclusive prefix-sum of `coverageBuffer`.
     * `emitOffset[i]` is the first slot in `tileKeysBuffer`/`splatValuesBuffer`
     * that splat i writes to in the emit-pairs pass.
     */
    emitOffsetBuffer: StorageBuffer;
    /**
     * Single u32 holding the total pair count produced by the prefix-sum.
     * Read by `prepareIndirect`, `initTileOffsets`, `findBoundaries`,
     * and the radix sort's indirect dispatch. Never touched by the CPU.
     */
    totalPairsBuffer: StorageBuffer;
    /**
     * `tileIdx` (sort key) for each (tile, splat) pair. Sorted in place
     * by `ComputeRadixSort.sortIndirect` — afterwards `sortedKeys` on the
     * radix sort instance holds the sorted tile keys.
     */
    tileKeysBuffer: StorageBuffer;
    /**
     * `splatIdx` (sort value) for each (tile, splat) pair, passed to the
     * radix sort as `initialValues`. The sort writes the reordered splat
     * indices to its internal `sortedIndices` buffer.
     */
    splatValuesBuffer: StorageBuffer;
    projectCompute: Compute;
    prefixSumCompute: Compute;
    emitPairsCompute: Compute;
    prepareIndirectCompute: Compute;
    initTileOffsetsCompute: Compute;
    findBoundariesCompute: Compute;
    rasterizeBinnedCompute: Compute;
    finalizeCompute: Compute;
}


/**
 * GPU-accelerated splat rasterizer.
 *
 * Owns eight compute shaders — project, prefix-sum, emit-pairs,
 * prepare-indirect, init-tile-offsets, find-boundaries, rasterize-binned,
 * finalize-pack — a shared `ComputeRadixSort` (used in indirect mode,
 * key + value), and GPU buffers. The per-chunk pipeline is fully
 * GPU-resident: the caller never reads back coverage, sorted keys, or
 * tile offsets.
 *
 * Per-render flow:
 *   1. `beginGroup(...)` — clears the running state and sets uniforms
 *      for this group (covers the whole image for a single-pass render).
 *   2. For each chunk of depth-sorted splats: `dispatchChunk(data,
 *      chunkSize)` runs the whole tile-bin + rasterize pipeline in one
 *      submission — project + coverage → prefix-sum (writes emitOffsets
 *      + totalPairs) → emit-pairs → prepare-indirect → radix sortIndirect
 *      → init-tile-offsets → find-boundaries → rasterize-binned. No
 *      readbacks; one `submit()` per chunk to capture each compute's
 *      uniform state before the next chunk overwrites it.
 *   3. `finishGroup()` — dispatches finalize-pack and starts an async
 *      readback. Returns a `Promise<Uint8Array>` resolved when the GPU has
 *      finished writing this group's RGBA bytes.
 */
class GpuSplatRasterizer {
    private device: GraphicsDevice;
    private options: SplatRasterizerOptions;
    private projectShader: Shader;
    private prefixSumShader: Shader;
    private emitPairsShader: Shader;
    private prepareIndirectShader: Shader;
    private initTileOffsetsShader: Shader;
    private findBoundariesShader: Shader;
    private rasterizeBinnedShader: Shader;
    private finalizeShader: Shader;
    private projectBgFormat: BindGroupFormat;
    private prefixSumBgFormat: BindGroupFormat;
    private emitPairsBgFormat: BindGroupFormat;
    private prepareIndirectBgFormat: BindGroupFormat;
    private initTileOffsetsBgFormat: BindGroupFormat;
    private findBoundariesBgFormat: BindGroupFormat;
    private rasterizeBinnedBgFormat: BindGroupFormat;
    private finalizeBgFormat: BindGroupFormat;
    private buffers: PipelineBuffers;
    /**
     * Single shared `ComputeRadixSort` for the GPU tile-bin pipeline.
     * Used in key+value mode: tile-index keys + splat-index values.
     */
    private radixSort: ComputeRadixSort;
    /** sortIndirect numBits, derived from numTiles (multiple of 4). */
    private sortKeyBits: number;
    private clearStatePattern: Float32Array;
    /** Active group's tile dimensions, set by `beginGroup`. */
    private activeTilesX: number = 0;
    private activeTilesY: number = 0;

    /** Floats per gaussian in the input buffer (depends on SH band count). */
    readonly inputStride: number;
    /** Group tile dimensions (X). */
    readonly groupTilesX: number;
    /** Group tile dimensions (Y). */
    readonly groupTilesY: number;
    /** Max gaussians per chunk. */
    readonly chunkCap: number;
    /** Pixels per group axis (X). */
    readonly groupPixelW: number;
    /** Pixels per group axis (Y). */
    readonly groupPixelH: number;

    constructor(device: GraphicsDevice, options: SplatRasterizerOptions) {
        this.device = device;
        this.options = options;
        this.groupTilesX = options.groupTilesX;
        this.groupTilesY = options.groupTilesY;
        this.chunkCap = options.chunkCap;
        this.groupPixelW = options.groupTilesX * TILE_SIZE;
        this.groupPixelH = options.groupTilesY * TILE_SIZE;

        const numTiles = options.groupTilesX * options.groupTilesY;
        // Round up to multiple of 4 (radix sort uses 4-bit passes). Min 4
        // because ComputeRadixSort requires numBits ≥ 4. Max 32 (u32 key).
        const tileBits = Math.max(4, Math.min(32, Math.ceil(Math.log2(Math.max(2, numTiles)) / 4) * 4));
        this.sortKeyBits = tileBits;

        const coeffs = numSHCoeffsPerChannel(options.numSHBands);
        this.inputStride = 14 + 3 * coeffs;

        this.projectBgFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('splats', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('projected', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('coverage', SHADERSTAGE_COMPUTE)
        ]);
        this.prefixSumBgFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('coverage', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('emitOffset', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('totalPairs', SHADERSTAGE_COMPUTE)
        ]);
        this.emitPairsBgFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('projected', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('emitOffset', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('coverage', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('tileKeys', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('splatValues', SHADERSTAGE_COMPUTE)
        ]);
        this.prepareIndirectBgFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('totalPairs', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('indirectBuffer', SHADERSTAGE_COMPUTE)
        ]);
        this.initTileOffsetsBgFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('totalPairs', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('tileOffsets', SHADERSTAGE_COMPUTE)
        ]);
        this.findBoundariesBgFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('totalPairs', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('sortedTileKeys', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('tileOffsets', SHADERSTAGE_COMPUTE)
        ]);
        this.rasterizeBinnedBgFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('projected', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('runningState', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('tileOffsets', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('sortedSplatIndices', SHADERSTAGE_COMPUTE, true)
        ]);
        this.finalizeBgFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('runningState', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('output', SHADERSTAGE_COMPUTE)
        ]);

        // WGSL chunks made available to every shader. The engine's
        // compute-shader path runs the source through
        // `Preprocessor.run(cshader, cincludes, ...)`, so any
        // `#include "name"` directive resolves to the corresponding
        // value in this Map. Lookup is by name only — chunk-order
        // does not matter.
        //
        // Per-render-cap chunks (tileAabb / tileWalk) are constructed
        // here with the resolved `maxCoveragePerSplat` so the chunk
        // bodies stay JS-template-free.
        const projection = options.projection;
        const sharedCincludes = new Map<string, string>([
            ['uniformsStruct', uniformsStruct],
            ['constants', constantsChunk],
            ['projectionPinhole', projectionPinhole],
            ['projectionEquirect', projectionEquirect],
            ['jacobianPinhole', jacobianPinhole],
            ['jacobianEquirect', jacobianEquirect],
            ['tileAabbPinhole', tileAabbPinhole(options.maxCoveragePerSplat)],
            ['tileAabbEquirect', tileAabbEquirect(options.maxCoveragePerSplat)],
            ['tileWalkPinhole', tileWalkPinhole],
            ['tileWalkEquirect', tileWalkEquirect],
            ['shBand1', shBand1],
            ['shBand2', shBand2],
            ['shBand3', shBand3],
            ['quatRotation', quatRotation],
            ['covariance3D', covariance3D]
        ]);

        // Per-render variant flags consumed by `#ifdef` directives in
        // the WGSL sources. Presence-only — the empty value is fine
        // because the preprocessor only checks `defines.has(name)` for
        // `#ifdef`. See `engine/src/core/preprocessor.js` for how
        // `cdefines` is consumed.
        const sharedCdefines = new Map<string, string>();
        if (projection === 'equirect') {
            sharedCdefines.set('PROJECTION_EQUIRECT', '');
        }
        if (options.numSHBands >= 1) sharedCdefines.set('SH_BAND_1', '');
        if (options.numSHBands >= 2) sharedCdefines.set('SH_BAND_2', '');
        if (options.numSHBands >= 3) sharedCdefines.set('SH_BAND_3', '');

        const mkShader = (
            name: string,
            source: string,
            bgFormat: BindGroupFormat,
            uniformEntries: UniformFormat[] = uniformFormatEntries(),
            cincludes: Map<string, string> = sharedCincludes,
            cdefines: Map<string, string> = sharedCdefines
        ) => new Shader(device, {
            name,
            shaderLanguage: SHADERLANGUAGE_WGSL,
            cshader: source,
            // @ts-ignore - computeUniformBufferFormats / computeBindGroupFormat / cincludes / cdefines are not in public Shader types.
            computeUniformBufferFormats: {
                uniforms: new UniformBufferFormat(device, uniformEntries)
            },
            // @ts-ignore
            computeBindGroupFormat: bgFormat,
            // @ts-ignore
            cincludes,
            // @ts-ignore
            cdefines
        });

        // The prefix-sum kernel processes the chunk in 256-thread blocks
        // of `scanPerThread` elements; the constant is compile-time
        // embedded so this product must cover the largest chunk we ever
        // dispatch (= chunkCap).
        const scanPerThread = Math.ceil(options.chunkCap / 256);

        // Uniform format for the small prepareIndirect kernel — just the
        // two slot-base indices into the device's indirect dispatch
        // buffer. Different layout from the shared `Uniforms` block.
        const prepareIndirectUniforms: UniformFormat[] = [
            new UniformFormat('sortSlotBase', UNIFORMTYPE_UINT),
            new UniformFormat('boundariesSlotBase', UNIFORMTYPE_UINT)
        ];

        this.projectShader = mkShader('splat-project', projectWgsl(coeffs), this.projectBgFormat);
        this.prefixSumShader = mkShader('splat-tilebin-prefix-sum', prefixSumWgsl(scanPerThread), this.prefixSumBgFormat);
        this.emitPairsShader = mkShader('splat-tilebin-emit-pairs', tileBinEmitPairsWgsl(), this.emitPairsBgFormat);
        this.prepareIndirectShader = mkShader('splat-tilebin-prepare-indirect', prepareIndirectWgsl(), this.prepareIndirectBgFormat, prepareIndirectUniforms);
        this.initTileOffsetsShader = mkShader('splat-tilebin-init-tile-offsets', initTileOffsetsWgsl(), this.initTileOffsetsBgFormat);
        this.findBoundariesShader = mkShader('splat-tilebin-find-boundaries', findBoundariesWgsl(), this.findBoundariesBgFormat);
        this.rasterizeBinnedShader = mkShader('splat-rasterize-binned', rasterizeBinnedWgsl(), this.rasterizeBinnedBgFormat);
        this.finalizeShader = mkShader('splat-finalize-pack', finalizeWgsl(), this.finalizeBgFormat);

        // Buffer sizing. runningState/output cover exactly the group's
        // tile rectangle (groupTilesX × groupTilesY × TILE_SIZE²), not the
        // bounding-square of the larger dimension as before.
        const groupPixels = this.groupPixelW * this.groupPixelH;
        const inputBytes = options.chunkCap * this.inputStride * 4;
        const projBytes = options.chunkCap * PROJECTION_STRIDE_F32 * 4;
        const stateBytes = groupPixels * RUNNING_STATE_STRIDE_F32 * 4;
        const outputBytes = groupPixels * OUTPUT_STRIDE_U32 * 4;
        // Tile-bin buffer sizes. tileOffsets has one slot per tile plus a
        // trailing sentinel so a tile's slice end is just `tileOffsets[T + 1]`.
        const tileOffsetsBytes = (numTiles + 1) * 4;

        const coverageBytes = options.chunkCap * 4;
        const emitOffsetBytes = options.chunkCap * 4;
        // Pair buffers. Two parallel u32 buffers (tileKeys + splatValues)
        // sized to chunkCap × maxCoveragePerSplat. The project shader
        // clamps each splat's coverage at maxCoveragePerSplat so the
        // sum never exceeds this cap regardless of scene density.
        const pairsBytes = options.chunkCap * options.maxCoveragePerSplat * 4;
        const totalPairsBytes = 4;

        const inputBuffer = new StorageBuffer(device, inputBytes, BUFFERUSAGE_COPY_DST);
        const projBuffer = new StorageBuffer(device, projBytes, 0);
        const runningStateBuffer = new StorageBuffer(device, stateBytes, BUFFERUSAGE_COPY_DST);
        const outputBuffer = new StorageBuffer(device, outputBytes, BUFFERUSAGE_COPY_SRC);
        const tileOffsetsBuffer = new StorageBuffer(device, tileOffsetsBytes, 0);
        const coverageBuffer = new StorageBuffer(device, coverageBytes, 0);
        const emitOffsetBuffer = new StorageBuffer(device, emitOffsetBytes, 0);
        const tileKeysBuffer = new StorageBuffer(device, pairsBytes, 0);
        const splatValuesBuffer = new StorageBuffer(device, pairsBytes, 0);
        const totalPairsBuffer = new StorageBuffer(device, totalPairsBytes, 0);

        const projectCompute = new Compute(device, this.projectShader, 'splat-project');
        projectCompute.setParameter('splats', inputBuffer);
        projectCompute.setParameter('projected', projBuffer);
        projectCompute.setParameter('coverage', coverageBuffer);

        const prefixSumCompute = new Compute(device, this.prefixSumShader, 'splat-tilebin-prefix-sum');
        prefixSumCompute.setParameter('coverage', coverageBuffer);
        prefixSumCompute.setParameter('emitOffset', emitOffsetBuffer);
        prefixSumCompute.setParameter('totalPairs', totalPairsBuffer);

        const emitPairsCompute = new Compute(device, this.emitPairsShader, 'splat-tilebin-emit-pairs');
        emitPairsCompute.setParameter('projected', projBuffer);
        emitPairsCompute.setParameter('emitOffset', emitOffsetBuffer);
        emitPairsCompute.setParameter('coverage', coverageBuffer);
        emitPairsCompute.setParameter('tileKeys', tileKeysBuffer);
        emitPairsCompute.setParameter('splatValues', splatValuesBuffer);

        const prepareIndirectCompute = new Compute(device, this.prepareIndirectShader, 'splat-tilebin-prepare-indirect');
        prepareIndirectCompute.setParameter('totalPairs', totalPairsBuffer);
        // `indirectBuffer` is bound per-chunk after the device has
        // allocated its indirect-dispatch buffer.

        const initTileOffsetsCompute = new Compute(device, this.initTileOffsetsShader, 'splat-tilebin-init-tile-offsets');
        initTileOffsetsCompute.setParameter('totalPairs', totalPairsBuffer);
        initTileOffsetsCompute.setParameter('tileOffsets', tileOffsetsBuffer);

        const findBoundariesCompute = new Compute(device, this.findBoundariesShader, 'splat-tilebin-find-boundaries');
        findBoundariesCompute.setParameter('totalPairs', totalPairsBuffer);
        findBoundariesCompute.setParameter('tileOffsets', tileOffsetsBuffer);
        // `sortedTileKeys` is bound per-chunk after the radix sort runs.

        const rasterizeBinnedCompute = new Compute(device, this.rasterizeBinnedShader, 'splat-rasterize-binned');
        rasterizeBinnedCompute.setParameter('projected', projBuffer);
        rasterizeBinnedCompute.setParameter('runningState', runningStateBuffer);
        rasterizeBinnedCompute.setParameter('tileOffsets', tileOffsetsBuffer);
        // `sortedSplatIndices` is bound per-chunk inside `dispatchChunk`,
        // pointing at the radix sort's `sortedIndices` output buffer.

        const finalizeCompute = new Compute(device, this.finalizeShader, 'splat-finalize');
        finalizeCompute.setParameter('runningState', runningStateBuffer);
        finalizeCompute.setParameter('output', outputBuffer);

        this.buffers = {
            inputBuffer,
            projBuffer,
            runningStateBuffer,
            outputBuffer,
            tileOffsetsBuffer,
            coverageBuffer,
            emitOffsetBuffer,
            totalPairsBuffer,
            tileKeysBuffer,
            splatValuesBuffer,
            projectCompute,
            prefixSumCompute,
            emitPairsCompute,
            prepareIndirectCompute,
            initTileOffsetsCompute,
            findBoundariesCompute,
            rasterizeBinnedCompute,
            finalizeCompute
        };

        // CPU-side cleared running state: T = 1 per pixel, color = 0.
        // Reused across groups; uploaded to runningStateBuffer at beginGroup.
        this.clearStatePattern = new Float32Array(groupPixels * RUNNING_STATE_STRIDE_F32);
        for (let i = 0; i < groupPixels; i++) {
            this.clearStatePattern[i * 4 + 3] = 1; // T = 1
        }

        // `indirect: true` is required since engine 2.19 — indirect mode
        // moved from a per-call `sortIndirect()` behaviour to a constructor
        // option. Without it, `sortIndirect()` silently dispatches directly
        // with `maxElementCount` (the full pair-buffer capacity), sorting
        // uninitialized zeros to the front and producing empty tile slices.
        this.radixSort = new ComputeRadixSort(device, { indirect: true });

        // The per-chunk pipeline reserves 2 slots in the device's
        // indirect-dispatch buffer (one for the radix sort, one for
        // find-boundaries). PC resets the slot counter at frame-end
        // only; for an offline render, the counter monotonically grows
        // across (chunk × sub-frame) iterations. At 8K with sub-frame
        // split and chunkCap squeezed down by binding limits we can
        // hit ~70 k slots. Pre-allocate generously — each slot is only
        // 12 bytes, so 256 k slots = 3 MB.
        const wantedSlots = 256 * 1024;
        // @ts-ignore - maxIndirectDispatchCount is a public property on
        // the WebGPU device but not in the public GraphicsDevice type.
        const cur = (device as { maxIndirectDispatchCount?: number }).maxIndirectDispatchCount ?? 0;
        if (cur < wantedSlots) {
            // @ts-ignore
            (device as { maxIndirectDispatchCount: number }).maxIndirectDispatchCount = wantedSlots;
        }
    }

    /**
     * Apply the global (camera + image + background) uniforms to every
     * pipeline compute instance, plus the per-group origin/extent fields.
     *
     * The group abstraction is retained as a hook for future subframe
     * rendering — when a render is split into multiple groups, each call
     * sets the current group's pixel rectangle so the project shader's
     * AABB cull skips splats outside the group.
     *
     * @param groupX - Group index along X.
     * @param groupY - Group index along Y.
     * @param groupTilesX - Number of tiles in this group along X.
     * @param groupTilesY - Number of tiles in this group along Y.
     */
    private setUniforms(
        groupX: number,
        groupY: number,
        groupTilesX: number,
        groupTilesY: number
    ): void {
        const o = this.options;
        const originX = groupX * this.groupPixelW;
        const originY = groupY * this.groupPixelH;
        const maxX = originX + groupTilesX * TILE_SIZE;
        const maxY = originY + groupTilesY * TILE_SIZE;

        const b = this.buffers;
        for (const c of [
            b.projectCompute,
            b.prefixSumCompute,
            b.emitPairsCompute,
            b.initTileOffsetsCompute,
            b.findBoundariesCompute,
            b.rasterizeBinnedCompute,
            b.finalizeCompute
        ]) {
            c.setParameter('rightX', o.rightX); c.setParameter('rightY', o.rightY); c.setParameter('rightZ', o.rightZ);
            c.setParameter('_p0', 0);
            c.setParameter('downX', o.downX); c.setParameter('downY', o.downY); c.setParameter('downZ', o.downZ);
            c.setParameter('_p1', 0);
            c.setParameter('forwardX', o.forwardX); c.setParameter('forwardY', o.forwardY); c.setParameter('forwardZ', o.forwardZ);
            c.setParameter('_p2', 0);
            c.setParameter('eyeX', o.eyeX); c.setParameter('eyeY', o.eyeY); c.setParameter('eyeZ', o.eyeZ);
            c.setParameter('_p3', 0);
            c.setParameter('focalX', o.focalX); c.setParameter('focalY', o.focalY);
            c.setParameter('near', o.near); c.setParameter('_p4', 0);
            c.setParameter('focusDistance', o.focusDistance);
            c.setParameter('apertureScale', o.apertureScale);
            c.setParameter('_p5', 0); c.setParameter('_p6', 0);
            c.setParameter('imageWidth', o.imageWidth); c.setParameter('imageHeight', o.imageHeight);
            c.setParameter('splatStride', this.inputStride);
            // chunkSize set per-dispatch
            c.setParameter('groupPixelMinX', originX);
            c.setParameter('groupPixelMinY', originY);
            c.setParameter('groupPixelMaxX', maxX);
            c.setParameter('groupPixelMaxY', maxY);
            c.setParameter('groupTilesX', groupTilesX);
            c.setParameter('groupTilesY', groupTilesY);
            c.setParameter('groupPixelOriginX', originX);
            c.setParameter('groupPixelOriginY', originY);
            c.setParameter('bgR', o.bgR); c.setParameter('bgG', o.bgG);
            c.setParameter('bgB', o.bgB); c.setParameter('bgA', o.bgA);
        }
    }

    /**
     * Begin processing a group. Clears running state and sets uniforms.
     *
     * @param groupX - Group index along X.
     * @param groupY - Group index along Y.
     * @param groupTilesX - Number of tiles in this group along X.
     * @param groupTilesY - Number of tiles in this group along Y.
     */
    beginGroup(
        groupX: number,
        groupY: number,
        groupTilesX: number,
        groupTilesY: number
    ): void {
        this.setUniforms(groupX, groupY, groupTilesX, groupTilesY);
        this.activeTilesX = groupTilesX;
        this.activeTilesY = groupTilesY;
        const groupPixels = groupTilesX * groupTilesY * TILE_SIZE * TILE_SIZE;
        this.buffers.runningStateBuffer.write(
            0, this.clearStatePattern, 0, groupPixels * RUNNING_STATE_STRIDE_F32
        );
    }

    /**
     * Commit pending GPU work. Called at chunk boundaries so each chunk's
     * uniform-buffer values are captured before the next chunk overwrites
     * them — a `Compute` instance's persistent uniform buffer is updated
     * by `setParameter`, and the dispatch only captures the value on
     * submit. Within a chunk, every dispatch uses a distinct `Compute`
     * instance, so no internal submits are needed.
     */
    submit(): void {
        // @ts-ignore - submit() is exposed by WebgpuGraphicsDevice but not on the public GraphicsDevice type.
        const submit = (this.device as { submit?: () => void }).submit;
        if (!submit) {
            throw new Error('GpuSplatRasterizer requires a GraphicsDevice with a submit() method (WebGPU backend).');
        }
        submit.call(this.device);
    }

    /**
     * Reserve a fresh sort + find-boundaries slot pair in the device's
     * indirect-dispatch buffer for this chunk. The returned indices are
     * consumed by `dispatchTileBinChunk` (internally) and exposed for
     * cross-cutting use (e.g. the radix sort needs the sort slot).
     *
     * @returns Two fresh slot indices in the device's indirect dispatch
     * buffer: one for the radix sort's indirect dispatch, one for the
     * find-boundaries indirect dispatch.
     */
    private acquireIndirectSlots(): { sortSlot: number; boundariesSlot: number } {
        // @ts-ignore - getIndirectDispatchSlot exists on WebgpuGraphicsDevice.
        const get = (this.device as { getIndirectDispatchSlot?: (count?: number) => number }).getIndirectDispatchSlot;
        if (!get) {
            throw new Error('GpuSplatRasterizer requires a GraphicsDevice with getIndirectDispatchSlot() (WebGPU backend).');
        }
        const sortSlot = get.call(this.device, 1);
        const boundariesSlot = get.call(this.device, 1);
        return { sortSlot, boundariesSlot };
    }

    /**
     * Dispatch the entire per-chunk tile-bin + rasterize pipeline on the
     * GPU with zero CPU readbacks:
     *
     *   pack-and-upload → project + coverage → prefix-sum (writes
     *   emitOffsets + totalPairs) → emit-pairs (writes tileKeys +
     *   splatValues) → prepare-indirect (writes workgroup counts into
     *   the device's indirect-dispatch buffer for the sort and
     *   find-boundaries) → radix sortIndirect (key+value: tileKeys
     *   sorted, splatValues reordered) → init tile-offsets to sentinel
     *   → find-boundaries (atomicMin) → rasterize.
     *
     * All eight dispatches use distinct `Compute` instances, so their
     * persistent uniform buffers don't alias each other within a chunk;
     * a single `submit()` after the rasterize captures everything before
     * the next chunk starts overwriting `setParameter` values.
     *
     * @param chunkData - Float32Array containing `chunkSize × inputStride` floats.
     * @param chunkSize - Number of gaussians in this chunk (≤ chunkCap).
     */
    dispatchChunk(chunkData: Float32Array, chunkSize: number): void {
        if (chunkSize === 0) return;
        if (chunkSize > this.chunkCap) {
            throw new Error(`chunkSize ${chunkSize} exceeds chunkCap ${this.chunkCap}`);
        }
        const b = this.buffers;

        // --- 1. Upload chunk + project (writes projected, coverage). ---
        b.inputBuffer.write(0, chunkData, 0, chunkSize * this.inputStride);
        b.projectCompute.setParameter('chunkSize', chunkSize);
        b.projectCompute.setupDispatch(Math.ceil(chunkSize / 64), 1, 1);
        this.device.computeDispatch([b.projectCompute], 'splat-project');

        // --- 2. GPU prefix-sum (coverage → emitOffsets + totalPairs). ---
        b.prefixSumCompute.setParameter('chunkSize', chunkSize);
        b.prefixSumCompute.setupDispatch(1, 1, 1);
        this.device.computeDispatch([b.prefixSumCompute], 'splat-tilebin-prefix-sum');

        // --- 3. Emit (tileKey, splatValue) pairs. ---
        b.emitPairsCompute.setParameter('chunkSize', chunkSize);
        b.emitPairsCompute.setupDispatch(Math.ceil(chunkSize / 64), 1, 1);
        this.device.computeDispatch([b.emitPairsCompute], 'splat-tilebin-emit-pairs');

        // --- 4. Prepare indirect dispatch params for sort + find-boundaries. ---
        // @ts-ignore - indirectDispatchBuffer getter is WebGPU-only.
        const indirectBuf = (this.device as { indirectDispatchBuffer?: StorageBuffer | null }).indirectDispatchBuffer;
        const { sortSlot, boundariesSlot } = this.acquireIndirectSlots();
        if (!indirectBuf) {
            throw new Error('Device indirectDispatchBuffer not allocated (WebGPU backend required).');
        }
        b.prepareIndirectCompute.setParameter('indirectBuffer', indirectBuf);
        // Each slot is 3 u32s in the buffer; the shader writes to
        // indirectBuffer[base + {0,1,2}], so base = slot * 3.
        b.prepareIndirectCompute.setParameter('sortSlotBase', sortSlot * 3);
        b.prepareIndirectCompute.setParameter('boundariesSlotBase', boundariesSlot * 3);
        b.prepareIndirectCompute.setupDispatch(1, 1, 1);
        this.device.computeDispatch([b.prepareIndirectCompute], 'splat-tilebin-prepare-indirect');

        // --- 5. Radix sort the pairs (indirect dispatch, key + value). ---
        // tileKeysBuffer holds u32 tileIdx; splatValuesBuffer holds u32
        // splatIdx as the initial-values payload. After the sort:
        //   - radixSort.sortedKeys[i]    = the i-th sorted tile index
        //   - radixSort.sortedIndices[i] = the splat index originally
        //                                  paired with that tile
        // The radix sort is stable, so within each tile the splatValues
        // remain in their input order = depth-monotonic (from the CPU
        // pre-sort), giving depth-ordered compositing per tile for free.
        // numBits = sortKeyBits (rounded up to multiple of 4 from
        // ceil(log2(numTiles))) — only the minimum required passes run.
        const pairsCap = this.chunkCap * this.options.maxCoveragePerSplat;
        this.radixSort.sortIndirect(
            b.tileKeysBuffer, pairsCap, this.sortKeyBits, sortSlot,
            b.totalPairsBuffer, b.splatValuesBuffer
        );
        const sortedTileKeysBuf = this.radixSort.sortedKeys;
        const sortedSplatIndicesBuf = this.radixSort.sortedIndices;
        if (!sortedTileKeysBuf || !sortedSplatIndicesBuf) {
            throw new Error('ComputeRadixSort returned null sortedKeys/sortedIndices after sortIndirect()');
        }

        // --- 6. Init tile-offsets to the sentinel (= totalPairs). ---
        const numTiles = this.groupTilesX * this.groupTilesY;
        b.initTileOffsetsCompute.setupDispatch(Math.ceil((numTiles + 1) / 64), 1, 1);
        this.device.computeDispatch([b.initTileOffsetsCompute], 'splat-tilebin-init-tile-offsets');

        // --- 7. Find tile boundaries via atomicMin. ---
        b.findBoundariesCompute.setParameter('sortedTileKeys', sortedTileKeysBuf);
        b.findBoundariesCompute.setupIndirectDispatch(boundariesSlot);
        this.device.computeDispatch([b.findBoundariesCompute], 'splat-tilebin-find-boundaries');

        // --- 8. Rasterize: walk each tile's slice in depth order. ---
        b.rasterizeBinnedCompute.setParameter('sortedSplatIndices', sortedSplatIndicesBuf);
        b.rasterizeBinnedCompute.setParameter('chunkSize', chunkSize);
        b.rasterizeBinnedCompute.setupDispatch(this.groupTilesX, this.groupTilesY, 1);
        this.device.computeDispatch([b.rasterizeBinnedCompute], 'splat-rasterize-binned');

        this.submit();
    }

    /**
     * Finish processing a group. Dispatches finalize-pack and starts an
     * async readback of the group's RGBA8 pixel bytes.
     *
     * Dispatch + readback are sized to the ACTIVE group dimensions (set
     * by the most recent `beginGroup`), not the constructor-provided
     * maximum, so edge sub-frames smaller than the max don't pay for
     * unused workgroups or readback bytes.
     *
     * @returns Promise resolving to the active group's RGBA byte buffer
     * (`activeTilesX·16 × activeTilesY·16 × 4` bytes).
     */
    finishGroup(): Promise<Uint8Array> {
        const b = this.buffers;
        b.finalizeCompute.setupDispatch(this.activeTilesX, this.activeTilesY, 1);
        this.device.computeDispatch([b.finalizeCompute], 'splat-finalize');

        const activePixelW = this.activeTilesX * TILE_SIZE;
        const activePixelH = this.activeTilesY * TILE_SIZE;
        const groupOutputBytes = activePixelW * activePixelH * 4;
        return b.outputBuffer.read(0, groupOutputBytes, null, true) as Promise<Uint8Array>;
    }

    /**
     * Release all GPU resources.
     */
    destroy(): void {
        this.radixSort.destroy();
        const b = this.buffers;
        b.inputBuffer.destroy();
        b.projBuffer.destroy();
        b.runningStateBuffer.destroy();
        b.outputBuffer.destroy();
        b.tileOffsetsBuffer.destroy();
        b.coverageBuffer.destroy();
        b.emitOffsetBuffer.destroy();
        b.totalPairsBuffer.destroy();
        b.tileKeysBuffer.destroy();
        b.splatValuesBuffer.destroy();
        this.projectShader.destroy();
        this.prefixSumShader.destroy();
        this.emitPairsShader.destroy();
        this.prepareIndirectShader.destroy();
        this.initTileOffsetsShader.destroy();
        this.findBoundariesShader.destroy();
        this.rasterizeBinnedShader.destroy();
        this.finalizeShader.destroy();
        this.projectBgFormat.destroy();
        this.prefixSumBgFormat.destroy();
        this.emitPairsBgFormat.destroy();
        this.prepareIndirectBgFormat.destroy();
        this.initTileOffsetsBgFormat.destroy();
        this.findBoundariesBgFormat.destroy();
        this.rasterizeBinnedBgFormat.destroy();
        this.finalizeBgFormat.destroy();
    }
}

export { GpuSplatRasterizer, type SplatRasterizerOptions };
