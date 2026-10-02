import { basename, dirname, join } from 'pathe';
import { Vec3 } from 'playcanvas';

import type { ChunkDataPool, ChunkSource } from '../chunk';
import { Column, DataTable, computeGaussianExtents } from '../data-table';
import type { Bounds } from '../data-table';
import { GpuVoxelization } from '../gpu';
import { writeFile } from '../io/write';
import type { FileSystem } from '../io/write';
import { bakeTransform } from '../ops/bake-transform';
import { GaussianBVH } from '../spatial';
import type { Options, DeviceCreator } from '../types';
import { Transform, logger } from '../utils';
import { alignGridBounds, filterAndFillBlocks, voxelizeToBuffer } from '../voxel';
import { BlockMaskBuffer } from '../voxel/block-mask-buffer';

import { buildSparseOctreeFromBuffer, MAX_V1_MIXED_LEAVES } from './sparse-octree';
import { logWrittenFile } from './utils';
import { writeOctreeFiles } from './write-voxel';

/** Axis-aligned bounds in world coordinates (Y up). */
type VoxelTileBounds = { min: [number, number, number]; max: [number, number, number] };

/** A standard voxel 1.1 file and its ownership / overlap regions. */
type VoxelTile = {
    id: string;
    ix: number;
    iz: number;
    coreBounds: VoxelTileBounds;
    dataBounds: VoxelTileBounds;
    /** URL relative to the manifest. */
    url: string;
};

/** Version 1 tiled surface collision manifest. Missing tiles have no collision coverage. */
type VoxelTileManifest = {
    version: 1;
    voxelResolution: number;
    tileSize: number;
    overlap: number;
    fullBounds: VoxelTileBounds;
    tiles: VoxelTile[];
};

/** Options for surface voxel tiles. Navigation processing is unsupported. */
type WriteVoxelTilesOptions = Pick<
    Options,
    'voxelResolution' | 'opacityCutoff' | 'navExteriorRadius' | 'floorFill' | 'navCapsule' | 'collisionMesh'
> & {
    filename: string;
    /** XZ tile width. Rounded up to a multiple of four voxels. Default: 64. */
    tileSize?: number;
    /** XZ overlap. Rounded up to whole four-voxel blocks. Default: 8. Zero is supported. */
    overlap?: number;
    /** Maximum selected Gaussians per tile, also capped by GPU binding limits. Default: 4,000,000. */
    maxGaussiansPerTile?: number;
    createDevice?: DeviceCreator;
};

const toJsonBounds = ({ min, max }: Bounds): VoxelTileBounds => ({
    min: [min.x, min.y, min.z],
    max: [max.x, max.y, max.z]
});
const intersects = (a: Bounds, b: Bounds, pad = 0): boolean =>
    a.min.x <= b.max.x + pad &&
    a.max.x >= b.min.x - pad &&
    a.min.y <= b.max.y + pad &&
    a.max.y >= b.min.y - pad &&
    a.min.z <= b.max.z + pad &&
    a.max.z >= b.min.z - pad;

// Only one chunk's geometry is resident during each selection scan. Reuse the
// canonical Gaussian extent implementation so tile membership matches the BVH.
const readGeometry = async (
    source: ChunkSource,
    pool: ChunkDataPool,
    chunkIndex: number,
    indices?: Uint32Array
): Promise<DataTable> => {
    const { meta } = source;
    const count = indices?.length ?? Math.min(meta.chunkSize, meta.numGaussians - chunkIndex * meta.chunkSize);
    const position = pool.acquire('position', meta.layouts.position!, count);
    const geometric = pool.acquire('geometric', meta.layouts.geometric!, count);
    try {
        await source.read(
            indices ? { indices, indexOffset: 0, count, position, geometric } : { chunkIndex, position, geometric }
        );
        const p = new Float32Array(position.data);
        const g = new Float32Array(geometric.data);
        const names = ['x', 'y', 'z', 'rot_0', 'rot_1', 'rot_2', 'rot_3', 'scale_0', 'scale_1', 'scale_2', 'opacity'];
        const columns = names.map((name) => new Column(name, new Float32Array(count)));
        for (let i = 0; i < count; i++) {
            for (let j = 0; j < 3; j++) columns[j].data[i] = p[i * 3 + j];
            for (let j = 0; j < 8; j++) columns[3 + j].data[i] = g[i * 8 + j];
            if (!Number.isFinite(p[i * 3]) || !Number.isFinite(p[i * 3 + 1]) || !Number.isFinite(p[i * 3 + 2])) {
                throw new Error('Voxel tiles require finite positions; use --filter-nan first.');
            }
        }
        return new DataTable(columns, Transform.IDENTITY);
    } finally {
        position.release();
        geometric.release();
    }
};

const getExtents = (table: DataTable): ReturnType<typeof computeGaussianExtents> => {
    const result = computeGaussianExtents(table);
    if (result.invalidCount) throw new Error('Voxel tiles require finite Gaussian extents; use --filter-nan first.');
    return result;
};

// Convert occupied blocks from the guarded grid to the public data region.
const stripGuard = (input: BlockMaskBuffer, nx: number, ny: number, nz: number): BlockMaskBuffer => {
    const out = new BlockMaskBuffer();
    const sx = nx + 2;
    const sy = ny + 2;
    const append = (index: number, lo: number, hi: number): void => {
        const x = (index % sx) - 1;
        const y = (Math.floor(index / sx) % sy) - 1;
        const z = Math.floor(index / (sx * sy)) - 1;
        if (x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz) {
            out.addBlock(x + y * nx + z * nx * ny, lo, hi);
        }
    };
    for (const index of input.getSolidBlocks()) append(index, 0xffffffff, 0xffffffff);
    const mixed = input.getMixedBlocks();
    for (let i = 0; i < mixed.blockIdx.length; i++)
        append(mixed.blockIdx[i], mixed.masks[i * 2], mixed.masks[i * 2 + 1]);
    return out;
};

/**
 * Write surface collision as independently loadable voxel 1.1 files and a v1 manifest.
 *
 * The input must contain a single LOD. Reads bake the pending transform into
 * world coordinates once and request only position / geometric layers. A bounds
 * pass records one AABB per source chunk; each tile then scans intersecting chunks
 * and materializes only its bounded subset. This trades repeated reads for bounded
 * geometry residency; inherently eager source decoders retain their own memory.
 * The caller owns the source, pool and GPU device lifetimes.
 *
 * Files are written sequentially. The manifest is published last, only after all
 * nonempty tiles succeed. Failed runs can leave unreferenced tile files; use a new
 * output directory for each generation (in-place replacement is not transactional).
 *
 * @param source - A single-LOD Gaussian source.
 * @param pool - Read buffers with the same chunk size as the source.
 * @param options - Output path, voxel parameters, tile geometry and budget.
 * @param fs - Platform-independent output filesystem.
 * @returns The manifest that was written.
 * @throws If options are invalid, a tile exceeds the budget, or any tile fails.
 * @example
 * ```ts
 * await writeVoxelTiles(source, pool, {
 *     filename: 'scene.voxel-tiles.json', tileSize: 64, overlap: 8, createDevice
 * }, fs);
 * ```
 */
const writeVoxelTiles = async (
    source: ChunkSource,
    pool: ChunkDataPool,
    options: WriteVoxelTilesOptions,
    fs: FileSystem
): Promise<VoxelTileManifest> => {
    const { filename, voxelResolution = 0.05, opacityCutoff = 0.1, createDevice } = options;
    const requestedSize = options.tileSize ?? 64;
    const requestedOverlap = options.overlap ?? 8;
    const requestedLimit = options.maxGaussiansPerTile ?? 4_000_000;
    if (
        ![voxelResolution, requestedSize].every((n) => Number.isFinite(n) && n > 0) ||
        !Number.isFinite(requestedOverlap) ||
        requestedOverlap < 0 ||
        !Number.isFinite(opacityCutoff) ||
        opacityCutoff <= 0 ||
        opacityCutoff > 1 ||
        !Number.isSafeInteger(requestedLimit) ||
        requestedLimit <= 0
    ) {
        throw new Error(
            'Invalid voxel tile options: sizes and budget must be positive, overlap nonnegative, opacity in (0, 1].'
        );
    }
    if (
        options.navExteriorRadius !== undefined ||
        options.floorFill ||
        options.navCapsule !== undefined ||
        options.collisionMesh
    ) {
        throw new Error(
            'Tiled voxel output supports surfaces only; external fill, floor fill, carve and collision mesh are unsupported.'
        );
    }
    if (!createDevice) throw new Error('Tiled voxel output requires a GPU device.');
    if (source.meta.numLods !== 1) throw new Error('Tiled voxel output requires one LOD; select a single level first.');
    if (source.meta.numGaussians === 0) throw new Error('No Gaussians to write.');
    if (source.meta.numGaussians > 0x100000000)
        throw new Error('Tiled voxel source exceeds the 32-bit row-index limit.');
    if (!source.meta.availableLayers.has('position') || !source.meta.availableLayers.has('geometric')) {
        throw new Error('Tiled voxel output requires position and geometric layers.');
    }
    if (pool.chunkSize !== source.meta.chunkSize) throw new Error('Voxel tile pool chunk size must match the source.');

    const block = 4 * voxelResolution;
    const tileSize = Math.ceil(requestedSize / block - 1e-10) * block;
    const overlap = Math.max(0, Math.ceil(requestedOverlap / block - 1e-10)) * block;
    if (!Number.isFinite(block) || !Number.isFinite(tileSize) || !Number.isFinite(overlap) || tileSize <= 0) {
        throw new Error('Voxel tile sizes overflow the supported world grid.');
    }
    const baked = bakeTransform(source, Transform.IDENTITY);
    const chunkBounds: Bounds[] = [];
    const min = new Vec3(Infinity, Infinity, Infinity);
    const max = new Vec3(-Infinity, -Infinity, -Infinity);
    for (let c = 0; c < baked.meta.numChunks[0]; c++) {
        const table = await readGeometry(baked, pool, c);
        const { sceneBounds } = getExtents(table);
        chunkBounds.push(sceneBounds);
        min.min(sceneBounds.min);
        max.max(sceneBounds.max);
    }
    const fullBounds = alignGridBounds(min.x, min.y, min.z, max.x, max.y, max.z, voxelResolution);
    if (
        ![...toJsonBounds(fullBounds).min, ...toJsonBounds(fullBounds).max].every(Number.isFinite) ||
        fullBounds.min.x >= fullBounds.max.x ||
        fullBounds.min.y >= fullBounds.max.y ||
        fullBounds.min.z >= fullBounds.max.z
    ) {
        throw new Error('Voxel tile scene bounds must be finite and have positive extent on every axis.');
    }
    const nx = Math.ceil((fullBounds.max.x - fullBounds.min.x) / tileSize);
    const nz = Math.ceil((fullBounds.max.z - fullBounds.min.z) / tileSize);
    if (!Number.isSafeInteger(nx * nz) || nx * nz > 1_000_000) {
        throw new Error('Voxel tile grid exceeds one million candidates; increase tile size or filter outliers.');
    }
    const device = await createDevice();
    const limits = (device as unknown as { limits?: { maxBufferSize?: number; maxStorageBufferBindingSize?: number } })
        .limits;
    const limit = Math.min(
        requestedLimit,
        Math.floor(
            Math.min(limits?.maxBufferSize ?? 256 * 2 ** 20, limits?.maxStorageBufferBindingSize ?? 128 * 2 ** 20) / 64
        )
    );
    if (limit < 1) throw new Error('GPU buffer limit is too small for voxelization.');
    const manifest: VoxelTileManifest = {
        version: 1,
        voxelResolution,
        tileSize,
        overlap,
        fullBounds: toJsonBounds(fullBounds),
        tiles: []
    };
    // Namespace tiles by manifest so multiple exports in one folder cannot collide.
    const tileDirectory = `${basename(filename).replace(/(?:\.)?voxel-tiles\.json$/i, '') || 'voxel'}-tiles`;
    for (let ix = 0; ix < nx; ix++) {
        for (let iz = 0; iz < nz; iz++) {
            const id = `x${ix}_z${iz}`;
            const core: Bounds = {
                min: new Vec3(fullBounds.min.x + ix * tileSize, fullBounds.min.y, fullBounds.min.z + iz * tileSize),
                max: new Vec3(
                    Math.min(fullBounds.max.x, fullBounds.min.x + (ix + 1) * tileSize),
                    fullBounds.max.y,
                    Math.min(fullBounds.max.z, fullBounds.min.z + (iz + 1) * tileSize)
                )
            };
            const data = alignGridBounds(
                core.min.x - overlap,
                core.min.y,
                core.min.z - overlap,
                core.max.x + overlap,
                core.max.y,
                core.max.z + overlap,
                voxelResolution
            );
            // A private one-block guard supplies all six neighbors to cleanup,
            // even when public overlap is zero. Strip it before serialization.
            const sample: Bounds = {
                min: data.min.clone().subScalar(block),
                max: data.max.clone().addScalar(block)
            };
            const dimensions = [data.max.x - data.min.x, data.max.y - data.min.y, data.max.z - data.min.z].map((size) =>
                Math.round(size / block)
            );
            if (
                dimensions.some((size) => !Number.isSafeInteger(size) || size < 1 || size > 2 ** 17) ||
                !Number.isSafeInteger(dimensions.reduce((total, size) => total * (size + 2), 1))
            ) {
                throw new Error(
                    `Voxel tile ${id} exceeds the supported grid dimensions; use coarser voxels or filter outliers.`
                );
            }
            let selected = new Uint32Array(Math.min(1024, limit));
            let count = 0;
            for (let c = 0; c < chunkBounds.length; c++) {
                if (!intersects(chunkBounds[c], sample, voxelResolution / 2)) continue;
                const table = await readGeometry(baked, pool, c);
                const { extents } = getExtents(table);
                const [x, y, z] = ['x', 'y', 'z'].map((name) => table.getColumnByName(name).data);
                const [ex, ey, ez] = ['extent_x', 'extent_y', 'extent_z'].map(
                    (name) => extents.getColumnByName(name).data
                );
                const h = voxelResolution / 2;
                for (let r = 0; r < table.numRows; r++) {
                    if (
                        x[r] + ex[r] < sample.min.x - h ||
                        x[r] - ex[r] > sample.max.x + h ||
                        y[r] + ey[r] < sample.min.y - h ||
                        y[r] - ey[r] > sample.max.y + h ||
                        z[r] + ez[r] < sample.min.z - h ||
                        z[r] - ez[r] > sample.max.z + h
                    )
                        continue;
                    if (count === limit)
                        throw new Error(
                            `Voxel tile ${id} exceeds its ${limit} Gaussian budget; reduce --voxel-tile-size or overlap.`
                        );
                    if (count === selected.length) {
                        const grown = new Uint32Array(Math.min(limit, selected.length * 2));
                        grown.set(selected);
                        selected = grown;
                    }
                    selected[count++] = c * baked.meta.chunkSize + r;
                }
            }
            if (!count) continue;
            // Gather only this tile, in pool-sized requests. Keep buffer ownership
            // local so a failed source read releases every acquired buffer.
            let table: DataTable | undefined;
            for (let start = 0; start < count; start += baked.meta.chunkSize) {
                const part = await readGeometry(
                    baked,
                    pool,
                    0,
                    selected.subarray(start, Math.min(count, start + baked.meta.chunkSize))
                );
                table ??= new DataTable(
                    part.columns.map((column) => new Column(column.name, new Float32Array(count))),
                    Transform.IDENTITY
                );
                for (let column = 0; column < part.columns.length; column++) {
                    table.columns[column].data.set(part.columns[column].data, start);
                }
            }
            const extents = getExtents(table!).extents;
            const bvh = new GaussianBVH(table!, extents);
            const gpu = new GpuVoxelization(device);
            try {
                gpu.uploadAllGaussians(table!, extents);
                const buffer = await voxelizeToBuffer(bvh, gpu, sample, voxelResolution, opacityCutoff);
                const nbx = Math.round((data.max.x - data.min.x) / block);
                const nby = Math.round((data.max.y - data.min.y) / block);
                const nbz = Math.round((data.max.z - data.min.z) / block);
                const cleaned = filterAndFillBlocks(buffer, nbx + 2, nby + 2, nbz + 2, MAX_V1_MIXED_LEAVES);
                const filtered = stripGuard(cleaned, nbx, nby, nbz);
                cleaned.clear();
                buffer.clear();
                if (!filtered.count) continue;
                const octree = buildSparseOctreeFromBuffer(filtered, nbx, nby, nbz, data, data, voxelResolution);
                filtered.clear();
                const url = `${tileDirectory}/${id}/surface.voxel.json`;
                const path = join(dirname(filename), url);
                await fs.mkdir(dirname(path));
                await writeOctreeFiles(fs, path, octree);
                manifest.tiles.push({
                    id,
                    ix,
                    iz,
                    coreBounds: toJsonBounds(core),
                    dataBounds: toJsonBounds(data),
                    url
                });
            } finally {
                gpu.destroy();
            }
            logger.info(`Voxel tile ${id}: ${count} Gaussians`);
        }
    }
    const bytes = new TextEncoder().encode(JSON.stringify(manifest, null, 2));
    await writeFile(fs, filename, bytes);
    logWrittenFile(basename(filename), bytes.byteLength);
    return manifest;
};

export { writeVoxelTiles, type WriteVoxelTilesOptions, type VoxelTileManifest, type VoxelTile, type VoxelTileBounds };
