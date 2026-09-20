/**
 * Default motion-blur sample count: derived from the screen-space motion of
 * the look-at point and of a frame-filling subject around it between the
 * shutter-open and shutter-close cameras, about 2 px per sample.
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

describe('motion sample count', () => {
    it('is one for a still camera and follows the screen-space motion of a dolly', async () => {
        const { Vec3 } = await import('playcanvas');
        const { motionSampleCount } = await import('../src/lib/render/index.js');
        // Pure sideways translation: the target moves with the camera.
        const cam = x => ({ position: new Vec3(x, 0, 10), target: new Vec3(x, 0, 0), up: new Vec3(0, 1, 0), fovY: Math.PI / 3, width: 1920, height: 1080, near: 0.2 });
        assert.strictEqual(motionSampleCount(cam(0), cam(0)), 1);

        // The nearest reference point sits d·tan(fov/2) in front of the
        // look-at point and moves Δx·focal/depth pixels.
        const focal = 540 / Math.tan(Math.PI / 6);
        const depth = 10 - 10 * Math.tan(Math.PI / 6);
        const expected = Math.ceil(0.1 * focal / depth / 2);
        assert.strictEqual(motionSampleCount(cam(0), cam(0.1)), expected);
        assert.ok(motionSampleCount(cam(0), cam(1)) > expected, 'more motion, more samples');
        assert.strictEqual(motionSampleCount(cam(0), cam(100)), 64, 'capped');
    });

    it('sees a zoom and an orbit even though the look-at point stays put', async () => {
        const { Vec3 } = await import('playcanvas');
        const { motionSampleCount } = await import('../src/lib/render/index.js');
        const still = { position: new Vec3(0, 0, 10), target: new Vec3(0, 0, 0), up: new Vec3(0, 1, 0), fovY: Math.PI / 3, width: 1920, height: 1080, near: 0.2 };
        assert.ok(motionSampleCount(still, { ...still, fovY: Math.PI / 2.5 }) > 1, 'zoom');
        const orbit = { ...still, position: new Vec3(10 * Math.sin(0.05), 0, 10 * Math.cos(0.05)) };
        assert.ok(motionSampleCount(still, orbit) > 1, 'orbit');
    });
});
