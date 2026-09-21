import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { Vec3 } from 'playcanvas';

import { buildApertureCameras, buildCameraBasis } from '../src/lib/render/camera.js';
import { writeImage } from '../src/lib/writers/write-image.js';

const project = (camera, point) => {
    const b = buildCameraBasis(camera);
    const p = point.clone().sub(b.eye);
    return [
        b.focalX * p.dot(b.right) / p.dot(b.forward) + camera.width / 2 + b.offsetX,
        b.focalY * p.dot(b.down) / p.dot(b.forward) + camera.height / 2 + b.offsetY
    ];
};

describe('aperture cameras', () => {
    it('rejects invalid sample counts and equirect aperture sampling before creating a device', async () => {
        const options = { createDevice: () => assert.fail('validation must run before device creation') };
        for (const dofSamples of [0, -1, 1.5, NaN, Infinity]) {
            await assert.rejects(writeImage({ ...options, fStop: 1, dofSamples }, null), /dof-samples must be a positive integer/);
        }
        await assert.rejects(writeImage({ ...options, projection: 'equirect', dofSamples: 32 }, null), /dof-samples is not valid/);
    });

    const camera = {
        position: new Vec3(2, 3, 5), target: new Vec3(-1, 0, 0), up: new Vec3(0, 1, 0),
        width: 800, height: 600, fovY: Math.PI / 3, near: 0.1, focusDistance: 4
    };

    it('keeps the entire focus plane aligned for a rotated camera', () => {
        const b = buildCameraBasis(camera);
        const views = buildApertureCameras(camera, 0.2, 32);
        for (const [x, y] of [[0, 0], [-1, 0.5], [1.5, -1]]) {
            const point = camera.position.clone().add(b.forward.clone().mulScalar(4))
            .add(b.right.clone().mulScalar(x)).add(b.down.clone().mulScalar(y));
            const expected = project(camera, point);
            for (const view of views) {
                const actual = project(view, point);
                assert.ok(Math.abs(actual[0] - expected[0]) < 1e-10);
                assert.ok(Math.abs(actual[1] - expected[1]) < 1e-10);
                assert.equal(view.apertureScale, 0);
            }
        }
    });

    it('samples a centered disk and produces the expected defocus displacement', () => {
        const b = buildCameraBasis(camera);
        const point = camera.position.clone().add(b.forward.clone().mulScalar(2));
        const base = project(camera, point);
        for (const count of [1, 16, 31, 32, 64]) {
            const mean = new Vec3();
            const views = buildApertureCameras(camera, 0.2, count);
            assert.equal(views.length, count);
            for (const view of views) {
                const offset = view.position.clone().sub(camera.position);
                mean.add(offset);
                assert.ok(offset.length() <= 0.2 + 1e-12);
                assert.ok(Math.abs(offset.dot(b.forward)) < 1e-12);
                const actual = project(view, point);
                assert.ok(Math.abs(actual[0] - base[0] - b.focalX * offset.dot(b.right) * (1 / 4 - 1 / 2)) < 1e-10);
                assert.ok(Math.abs(actual[1] - base[1] - b.focalY * offset.dot(b.down) * (1 / 4 - 1 / 2)) < 1e-10);
            }
            assert.ok(mean.length() < 1e-12);
        }
    });
});
