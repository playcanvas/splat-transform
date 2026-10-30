import assert from 'node:assert/strict';
import { describe, it, before, after } from 'node:test';

import { Column, DataTable, createChunkDataPool, getOutputFormat, writeSource, writeVoxel, writeVoxelTiles } from '../src/lib/index.js';
import { dataTableToChunkSource } from '../src/lib/compat/data-table.js';
import { MemoryFileSystem } from '../src/lib/io/write/index.js';
import { Generator } from '../generators/gen-voxel-tiles.mjs';
import { Transform } from '../src/lib/utils/index.js';
import { Vec3, Quat } from 'playcanvas';

function makeTable() {
    const generator = Generator.create();
    const columns = generator.columnNames.map(name => new Column(name, new Float32Array(generator.count)));
    const row = {};
    for (let i = 0; i < generator.count; i++) {
        generator.getRow(i, row);
        for (const column of columns) column.data[i] = row[column.name];
    }
    return new DataTable(columns);
}

function makeSource(table = makeTable()) {
    const source = dataTableToChunkSource(table, 127);
    const pool = createChunkDataPool({ chunkSize: 127, maxPooledBytes: 127 * 44 });
    const read = source.read.bind(source);
    const requests = [];
    source.read = async request => {
        assert.equal(request.color, undefined, 'writer must never load color / SH');
        assert.equal(request.other, undefined);
        for (const layer of ['position', 'geometric']) {
            if (request[layer]) assert.ok(request[layer].count <= 127);
        }
        requests.push(request.chunkIndex ?? 'gather');
        return read(request);
    };
    return { source, pool, requests };
}

const options = { filename: 'scene.voxel-tiles.json', voxelResolution: 0.1, tileSize: 4, overlap: 0 };
const fakeDevice = async () => ({ limits: { maxBufferSize: 64, maxStorageBufferBindingSize: 64 } });

describe('tiled surface output validation and budget', () => {
    it('recognizes both manifest names without changing single-voxel detection', () => {
        assert.equal(getOutputFormat('path/voxel-tiles.json', {}), 'voxel-tiles');
        assert.equal(getOutputFormat('path/scene.voxel-tiles.json', {}), 'voxel-tiles');
        assert.equal(getOutputFormat('scene.voxel.json', {}), 'voxel');
        assert.throws(() => getOutputFormat('notvoxel-tiles.json', {}));
    });

    it('rejects invalid sizes, opacity and budget before reading', async () => {
        for (const bad of [{ tileSize: 0 }, { tileSize: Infinity }, { overlap: -1 },
            { voxelResolution: NaN }, { opacityCutoff: 0 }, { opacityCutoff: 1.1 }, { maxGaussiansPerTile: 0.5 }]) {
            const { source, pool, requests } = makeSource();
            await assert.rejects(writeVoxelTiles(source, pool, { ...options, ...bad, createDevice: fakeDevice }, new MemoryFileSystem()), /Invalid voxel tile/);
            assert.equal(requests.length, 0);
        }
    });

    it('rejects nonlocal fill / navigation / mesh combinations', async () => {
        for (const bad of [{ navExteriorRadius: 1 }, { floorFill: true }, { navCapsule: { height: 1.8, radius: 0.2 } }, { collisionMesh: 'faces' }]) {
            const { source, pool, requests } = makeSource();
            await assert.rejects(writeVoxelTiles(source, pool, { ...options, ...bad, createDevice: fakeDevice }, new MemoryFileSystem()), /surfaces only/);
            assert.equal(requests.length, 0);
        }
    });

    it('fails before GPU upload / publishing when a tile exceeds the device budget', async () => {
        const { source, pool } = makeSource();
        const fs = new MemoryFileSystem();
        await assert.rejects(writeSource({ filename: options.filename, outputFormat: 'voxel-tiles', source, pool,
            options: { voxelResolution: 0.1, voxelTileSize: 4, voxelTileOverlap: 0 }, createDevice: fakeDevice }, fs), /exceeds its 1 Gaussian budget/);
        assert.equal(fs.results.size, 0);
        assert.equal(pool.bytesInUse, 0);
    });

    it('honors a tighter caller budget and releases read buffers on error', async () => {
        const { source, pool } = makeSource();
        await assert.rejects(writeVoxelTiles(source, pool, { ...options, maxGaussiansPerTile: 1,
            createDevice: async () => ({ limits: { maxBufferSize: 2 ** 30, maxStorageBufferBindingSize: 2 ** 30 } }) }, new MemoryFileSystem()), /exceeds its 1 Gaussian budget/);
        assert.equal(pool.bytesInUse, 0);
        source.read = async () => { throw new Error('read failure'); };
        await assert.rejects(writeVoxelTiles(source, pool, { ...options, createDevice: fakeDevice }, new MemoryFileSystem()), /read failure/);
        assert.equal(pool.bytesInUse, 0);
    });

    it('selects an ellipsoid crossing a tile even when its center is outside', async () => {
        const table = makeTable().clone({ rows: new Uint32Array([0, 1]) });
        table.getColumnByName('x').data.set([2, 2]);
        table.getColumnByName('scale_0').data.fill(Math.log(2));
        const { source, pool } = makeSource(table);
        // In the first tile both centers lie outside, but both AABBs intersect.
        await assert.rejects(writeVoxelTiles(source, pool, { ...options, tileSize: 0.4, createDevice: fakeDevice }, new MemoryFileSystem()), /x0_z0 exceeds its 1 Gaussian budget/);
    });

    it('releases buffers if a tile gather fails after selection', async () => {
        const { source, pool } = makeSource();
        const read = source.read.bind(source);
        source.read = async request => {
            if ('indices' in request) throw new Error('gather failure');
            return read(request);
        };
        const fs = new MemoryFileSystem();
        await assert.rejects(writeVoxelTiles(source, pool, { ...options,
            createDevice: async () => ({ limits: { maxBufferSize: 2 ** 30, maxStorageBufferBindingSize: 2 ** 30 } }) }, fs), /gather failure/);
        assert.equal(pool.bytesInUse, 0);
        assert.equal(fs.results.size, 0);
    });

    it('rejects an unselected multi-LOD source and an empty source', async () => {
        const { source, pool } = makeSource();
        const multi = { meta: { ...source.meta, numLods: 2 }, read: source.read, close: source.close };
        await assert.rejects(writeVoxelTiles(multi, pool, { ...options, createDevice: fakeDevice }, new MemoryFileSystem()), /one LOD/);
        const empty = { ...multi, meta: { ...source.meta, numGaussians: 0 } };
        await assert.rejects(writeVoxelTiles(empty, pool, { ...options, createDevice: fakeDevice }, new MemoryFileSystem()), /No Gaussians/);
    });
});

const gpuEnabled = process.env.TEST_WEBGPU === '1';
describe('tiled surface GPU integration', { skip: !gpuEnabled }, () => {
    let device;
    before(async () => {
        const { createDevice } = await import('../src/cli/node-device.js');
        device = await createDevice();
    });
    after(() => device?.destroy());
    const gpu = () => Promise.resolve(device);

    function decode(fs, path) {
        const metadata = JSON.parse(new TextDecoder().decode(fs.results.get(path)));
        const bytes = fs.results.get(path.replace('.voxel.json', '.voxel.bin'));
        const all = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
        return { metadata, nodes: all.subarray(0, metadata.nodeCount), leaves: all.subarray(metadata.nodeCount) };
    }

    // Independent decoder for the unchanged voxel 1.1 tree, sampled in world space.
    function occupied(tree, point) {
        const { metadata: m, nodes, leaves } = tree;
        const v = point.map((p, a) => Math.floor((p - m.gridBounds.min[a]) / m.voxelResolution + 1e-6));
        if (v.some((n, a) => n < 0 || point[a] >= m.gridBounds.max[a])) return false;
        const block = v.map(n => Math.floor(n / 4));
        let index = 0;
        for (let depth = m.treeDepth; depth >= 0; depth--) {
            const node = nodes[index];
            if (node === undefined) return false;
            if (node === 0xff000000) return true;
            const mask = node >>> 24;
            const base = node & 0xffffff;
            if (mask === 0) {
                const bit = (v[0] & 3) + (v[1] & 3) * 4 + (v[2] & 3) * 16;
                return !!(leaves[base * 2 + (bit >>> 5)] & (1 << (bit & 31)));
            }
            const bitIndex = depth - 1;
            const oct = ((block[0] >>> bitIndex) & 1) | (((block[1] >>> bitIndex) & 1) << 1) | (((block[2] >>> bitIndex) & 1) << 2);
            if (!(mask & (1 << oct))) return false;
            let offset = 0;
            for (let i = 0; i < oct; i++) if (mask & (1 << i)) offset++;
            index = base + offset;
        }
        return false;
    }

    it('matches monolithic world occupancy throughout cores and across seams with zero overlap', async () => {
        const table = makeTable();
        for (const axis of ['x', 'z']) {
            const positions = table.getColumnByName(axis).data;
            assert.ok(Math.max(...positions) - Math.min(...positions) >= 5,
                `public fixture must satisfy the viewer walking threshold on ${axis}`);
        }
        // An anisotropic rotated splat straddles a seam; its center is not in every affected tile.
        table.getColumnByName('scale_0').data[0] = Math.log(0.8);
        table.getColumnByName('rot_0').data[0] = Math.cos(Math.PI / 8);
        table.getColumnByName('rot_2').data[0] = Math.sin(Math.PI / 8);
        const fs = new MemoryFileSystem();
        const { source, pool } = makeSource(table);
        const manifest = await writeVoxelTiles(source, pool, { ...options, createDevice: gpu }, fs);
        await writeVoxel({ filename: 'whole.voxel.json', dataTable: table, voxelResolution: 0.1, opacityCutoff: 0.1, createDevice: gpu }, fs);
        const whole = decode(fs, 'whole.voxel.json');
        assert.ok(manifest.tiles.length > 1);
        assert.equal(manifest.overlap, 0);
        let comparisons = 0, occupiedCount = 0, seamComparisons = 0;
        for (const tile of manifest.tiles) {
            const part = decode(fs, tile.url);
            assert.equal(part.metadata.version, '1.1');
            assert.deepEqual(part.metadata.gridBounds, tile.dataBounds);
            for (const edge of [...tile.dataBounds.min, ...tile.dataBounds.max]) assert.ok(Math.abs(edge / 0.4 - Math.round(edge / 0.4)) < 1e-6);
            for (let x = tile.coreBounds.min[0] + 0.05; x < tile.coreBounds.max[0]; x += 0.1)
                for (let y = tile.coreBounds.min[1] + 0.05; y < tile.coreBounds.max[1]; y += 0.1)
                    for (let z = tile.coreBounds.min[2] + 0.05; z < tile.coreBounds.max[2]; z += 0.1) {
                        const point = [x, y, z];
                        const expected = occupied(whole, point);
                        assert.equal(occupied(part, point), expected, `tile ${tile.id} at ${point}`);
                        comparisons++;
                        if (expected) occupiedCount++;
                        if (x - tile.coreBounds.min[0] < 0.15 || tile.coreBounds.max[0] - x < 0.15) seamComparisons++;
                    }
        }
        assert.ok(comparisons > 10000 && occupiedCount > 100 && seamComparisons > 100);
        assert.equal(pool.bytesInUse, 0);
    });

    it('bakes world transforms once and publishes only populated files', async () => {
        const table = makeTable();
        table.transform = new Transform(new Vec3(5, 2, -3), new Quat().setFromEulerAngles(0, 90, 0), 1);
        const { source, pool } = makeSource(table);
        const fs = new MemoryFileSystem();
        const manifest = await writeVoxelTiles(source, pool, { ...options, tileSize: 3.9, overlap: 0.3, createDevice: gpu }, fs);
        assert.ok(Math.abs(manifest.tileSize - 4) < 1e-8);
        assert.ok(Math.abs(manifest.overlap - 0.4) < 1e-8);
        assert.ok(manifest.fullBounds.min[0] > 1 && manifest.fullBounds.min[0] < 2 && manifest.fullBounds.min[1] > 1);
        assert.equal(new Set(manifest.tiles.map(t => t.id)).size, manifest.tiles.length);
        for (const tile of manifest.tiles) {
            assert.ok(fs.results.has(tile.url));
            assert.ok(fs.results.has(tile.url.replace('.json', '.bin')));
            assert.ok(!tile.url.startsWith('/') && !tile.url.includes('..'));
        }
    });

    it('omits empty surface tiles and leaves no manifest when a binary write fails', async () => {
        const empty = makeTable();
        empty.getColumnByName('opacity').data.fill(-Infinity);
        const a = makeSource(empty);
        const emptyFs = new MemoryFileSystem();
        const manifest = await writeVoxelTiles(a.source, a.pool, { ...options, createDevice: gpu }, emptyFs);
        assert.deepEqual(manifest.tiles, []);
        assert.equal(emptyFs.results.size, 1);
        const b = makeSource();
        const fs = new MemoryFileSystem();
        const createWriter = fs.createWriter.bind(fs);
        fs.createWriter = path => {
            if (path.endsWith('.bin')) throw new Error('disk failure');
            return createWriter(path);
        };
        await assert.rejects(writeVoxelTiles(b.source, b.pool, { ...options, createDevice: gpu }, fs), /disk failure/);
        assert.equal(fs.results.has(options.filename), false);
        assert.equal(b.pool.bytesInUse, 0);
    });

    it('does not publish a URL for a spatially empty candidate between populated tiles', async () => {
        const table = makeTable().clone({ rows: new Uint32Array([0, 1]) });
        table.getColumnByName('x').data.set([-8, 8]);
        const { source, pool } = makeSource(table);
        const fs = new MemoryFileSystem();
        const manifest = await writeVoxelTiles(source, pool, { ...options, createDevice: gpu }, fs);
        const nx = Math.ceil((manifest.fullBounds.max[0] - manifest.fullBounds.min[0]) / manifest.tileSize);
        assert.ok(manifest.tiles.length > 0 && manifest.tiles.length < nx);
        assert.equal(manifest.tiles.some(tile => tile.coreBounds.min[0] < 0 && tile.coreBounds.max[0] > 0), false);
        assert.equal(fs.results.size, manifest.tiles.length * 2 + 1);
    });
});
