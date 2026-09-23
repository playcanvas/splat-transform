/**
 * The optional SOG `camera` block:
 *
 *  - built from a 3DGS training cameras.json entry;
 *  - carried through .sog / meta.json / lod-meta.json read and write unchanged
 *    (unknown keys included), once at the top of lod-meta.json;
 *  - moved with the scene by transform actions.
 */

import assert from 'node:assert';
import { dirname, join } from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';

import { Quat, Vec3 } from 'playcanvas';

import { createChunkDataPool } from '../src/lib/chunk/index.js';
import { dataTableToChunkSource, materializeToDataTable } from '../src/lib/compat/data-table.js';
import {
    Column, DataTable, MemoryFileSystem, MemoryReadFileSystem, Transform, WebPCodec,
    processSourceBridged, readFile, sogCameraFromCamerasJson, transformSogCamera, withCamera, writeSource
} from '../src/lib/index.js';
import { writeLodSource } from '../src/lib/writers/write-lod.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
WebPCodec.wasmUrl = join(__dirname, '..', 'lib', 'webp.wasm');

const camera = {
    convention: 'opencv',
    rig: 'camera',
    rest: { position: [0.5, -0.25, 1], rotation: [0, 0, 0, 1] },
    intrinsics: { fx: 1194.667, fy: 1194.667, cx: 1024, cy: 576, width: 2048, height: 1152 },
    stereo: { baseline_m: 0.063 },
    focus: { point: [0, 0, 1.68], subject_m: 2.14, near_m: 0.73, far_m: 66.2 },
    vendor: { anything: [1, 'two', { three: 3 }] }
};

// A PLY-space table from explicit positions (identity rotation, small splats).
const makeTable = (points) => {
    const n = points.length;
    const col = i => new Float32Array(points.map(p => p[i]));
    const fill = v => new Float32Array(n).fill(v);
    return new DataTable([
        new Column('x', col(0)), new Column('y', col(1)), new Column('z', col(2)),
        new Column('rot_0', fill(1)), new Column('rot_1', fill(0)), new Column('rot_2', fill(0)), new Column('rot_3', fill(0)),
        new Column('scale_0', fill(-3)), new Column('scale_1', fill(-3)), new Column('scale_2', fill(-3)),
        new Column('f_dc_0', fill(0)), new Column('f_dc_1', fill(0)), new Column('f_dc_2', fill(0)),
        new Column('opacity', fill(0))
    ], Transform.PLY);
};

const points = [[0.5, -0.25, 1], [0.5, -0.25, 2], [-1, 1, 3], [2, 0, 4]];

const sourceWithCamera = (cam = camera) => withCamera(dataTableToChunkSource(makeTable(points), 1 << 20), cam);

// Write `source` and return the MemoryFileSystem results.
const write = async (source, filename, outputFormat) => {
    const fs = new MemoryFileSystem();
    await writeSource({ filename, outputFormat, source, pool: createChunkDataPool(), options: { iterations: 1 } }, fs);
    return fs.results;
};

const readBack = async (results, filename, inputFormat) => {
    const rfs = new MemoryReadFileSystem();
    for (const [name, data] of results) rfs.set(name.split('/').pop(), data);
    const [source] = await readFile({ filename, inputFormat, fileSystem: rfs });
    return source;
};

const metaOf = results => JSON.parse(Buffer.from(results.get('meta.json')).toString());

describe('sogCameraFromCamerasJson', () => {
    const entry = (rotation, extra = {}) => ({
        id: 0, img_name: '00000', width: 1600, height: 1200,
        position: [1, 2, 3], rotation, fx: 1100, fy: 1120, ...extra
    });

    it('maps an INRIA cameras.json entry', () => {
        const cam = sogCameraFromCamerasJson([entry([[1, 0, 0], [0, 1, 0], [0, 0, 1]])]);
        assert.deepStrictEqual(cam, {
            convention: 'opencv',
            rest: { position: [1, 2, 3], rotation: [0, 0, 0, 1] },
            intrinsics: { fx: 1100, fy: 1120, cx: 800, cy: 600, width: 1600, height: 1200 }
        });
    });

    it('converts the camera-to-world rotation rows to a quaternion', () => {
        // 90 degrees about +y: camera +z (forward) looks down world +x
        const cam = sogCameraFromCamerasJson([
            entry([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            entry([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        ], 1);
        const [x, y, z, w] = cam.rest.rotation;
        const forward = new Quat(x, y, z, w).transformVector(new Vec3(0, 0, 1), new Vec3());
        assert.ok(forward.distance(new Vec3(1, 0, 0)) < 1e-9, `forward ${forward}`);
    });

    it('rejects a missing or malformed entry', () => {
        assert.throws(() => sogCameraFromCamerasJson({}), /expected an array/);
        assert.throws(() => sogCameraFromCamerasJson([], 0), /no camera at index 0/);
        assert.throws(() => sogCameraFromCamerasJson([entry([[1, 0, 0], [0, 1, 0]])]), /3x3 rotation/);
    });
});

describe('transformSogCamera', () => {
    it('returns an equal copy under identity', () => {
        const out = transformSogCamera(camera, new Transform());
        assert.deepStrictEqual(out, camera);
        assert.notStrictEqual(out.rest, camera.rest);
    });

    it('moves the rest pose, scales distances, and leaves the rest alone', () => {
        const t = new Transform(new Vec3(1, 2, 3), new Quat().setFromEulerAngles(0, 90, 0), 2);
        const out = transformSogCamera(camera, t);

        const p = t.transformPoint(new Vec3(0.5, -0.25, 1), new Vec3());
        assert.ok(new Vec3(...out.rest.position).distance(p) < 1e-9);
        const q = new Quat(...out.rest.rotation);
        const forward = q.transformVector(new Vec3(0, 0, 1), new Vec3());
        assert.ok(forward.distance(new Vec3(1, 0, 0)) < 1e-9, `forward ${forward}`);

        assert.strictEqual(out.stereo.baseline_m, 0.126);
        assert.deepStrictEqual(out.focus, { point: [0, 0, 3.36], subject_m: 4.28, near_m: 1.46, far_m: 132.4 });
        assert.deepStrictEqual(out.intrinsics, camera.intrinsics);
        assert.deepStrictEqual(out.vendor, camera.vendor);
        assert.deepStrictEqual(camera.rest.position, [0.5, -0.25, 1], 'input untouched');
    });
});

describe('SOG camera block', () => {
    it('writes the camera to meta.json and reads it back', async () => {
        const out = await write(sourceWithCamera(), 'meta.json', 'sog');
        assert.deepStrictEqual(metaOf(out).camera, camera);

        const source = await readBack(out, 'meta.json', 'sog');
        assert.deepStrictEqual(source.meta.camera, camera);
    });

    it('round-trips .sog -> .sog unchanged', async () => {
        const first = await write(sourceWithCamera(), 'a.sog', 'sog-bundle');
        const source = await readBack(first, 'a.sog', 'sog');
        const second = await write(source, 'meta.json', 'sog');
        assert.strictEqual(JSON.stringify(metaOf(second).camera), JSON.stringify(camera));
    });

    it('leaves meta.json unchanged without a camera', async () => {
        const out = await write(dataTableToChunkSource(makeTable(points), 1 << 20), 'meta.json', 'sog');
        assert.ok(!('camera' in metaOf(out)));
    });

    it('moves the rest pose with a rotate action', async () => {
        // the camera sits on one splat and looks at another 1 unit ahead
        const pool = createChunkDataPool();
        const rotated = await processSourceBridged(sourceWithCamera(), [
            { kind: 'rotate', value: new Vec3(0, 90, 0) },
            { kind: 'translate', value: new Vec3(1, 0, 0) }
        ], pool);
        const out = await write(rotated, 'meta.json', 'sog');
        const cam = metaOf(out).camera;

        // (the writer reorders splats, so match by position)
        const table = await materializeToDataTable(await readBack(out, 'meta.json', 'sog'), pool);
        const splats = Array.from({ length: table.numRows }, (_, i) => new Vec3(...['x', 'y', 'z'].map(c => table.getColumnByName(c).data[i])));
        const nearest = p => Math.min(...splats.map(s => s.distance(p)));

        const position = new Vec3(...cam.rest.position);
        const forward = new Quat(...cam.rest.rotation).transformVector(new Vec3(0, 0, 1), new Vec3());
        assert.ok(nearest(position) < 1e-2, 'camera stays on its splat');
        assert.ok(nearest(position.clone().add(forward)) < 1e-2, 'camera still looks at the next splat');
    });
});

describe('Streamed SOG camera block', () => {
    const writeLod = async (source) => {
        const fs = new MemoryFileSystem();
        await writeLodSource({
            filename: '/scene/lod-meta.json',
            mainSource: source,
            envSource: null,
            iterations: 1,
            chunkCount: 1,
            chunkExtent: 16
        }, fs);
        return fs.results;
    };

    it('writes the camera once, at the top level of lod-meta.json', async () => {
        const out = await writeLod(sourceWithCamera());
        const meta = JSON.parse(Buffer.from(out.get('/scene/lod-meta.json')).toString());
        assert.deepStrictEqual(meta.camera, camera);
        assert.deepStrictEqual(Object.keys(meta).slice(0, 3), ['version', 'asset', 'camera']);

        const units = [...out.keys()].filter(k => k.endsWith('/meta.json'));
        assert.ok(units.length > 0);
        for (const unit of units) {
            assert.ok(!('camera' in JSON.parse(Buffer.from(out.get(unit)).toString())), `${unit} repeats the camera`);
        }
    });

    it('reads it back from lod-meta.json', async () => {
        const out = await writeLod(sourceWithCamera());
        const rfs = new MemoryReadFileSystem();
        for (const [name, data] of out) rfs.set(name.replace('/scene/', ''), data);
        const [source] = await readFile({ filename: 'lod-meta.json', inputFormat: 'lod', fileSystem: rfs });
        assert.deepStrictEqual(source.meta.camera, camera);
    });
});
