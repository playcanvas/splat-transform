import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { loadCameraTrack } from '../src/lib/render/camera-track.js';

const close = (a, b, eps = 1e-9) => {
    assert.ok(Math.abs(a.x - b.x) < eps && Math.abs(a.y - b.y) < eps && Math.abs(a.z - b.z) < eps, `${JSON.stringify(a)} != ${JSON.stringify(b)}`);
};

describe('camera track up vectors', () => {
    it('frame list: per-frame up, normalized lerp between frames, default for frames without one', () => {
        const track = loadCameraTrack({
            frames: [
                { position: [0, 0, 5], target: [0, 0, 0], up: [0, 1, 0] },
                { position: [0, 0, 5], target: [0, 0, 0], up: [1, 0, 0] },
                { position: [0, 0, 5], target: [0, 0, 0] }
            ]
        }, 60, { x: 0, y: 0, z: 1 });
        close(track.poseAt(0).up, { x: 0, y: 1, z: 0 });
        close(track.poseAt(1).up, { x: 1, y: 0, z: 0 });
        const s = Math.SQRT1_2;
        close(track.poseAt(0.5).up, { x: s, y: s, z: 0 });
        close(track.poseAt(2).up, { x: 0, y: 0, z: 1 });
    });

    it('frame list: normalizes ups on load so magnitude does not bias the lerp', () => {
        const track = loadCameraTrack({
            frames: [
                { position: [0, 0, 5], target: [0, 0, 0], up: [0, 3, 0] },
                { position: [0, 0, 5], target: [0, 0, 0], up: [0.5, 0, 0] }
            ]
        }, 60);
        const s = Math.SQRT1_2;
        close(track.poseAt(0).up, { x: 0, y: 1, z: 0 });
        close(track.poseAt(0.5).up, { x: s, y: s, z: 0 });
    });

    it('frame list: rejects malformed and zero ups', () => {
        const pose = up => ({ position: [0, 0, 5], target: [0, 0, 0], up });
        assert.throws(() => loadCameraTrack({ frames: [pose([0, 1])] }, 60), /frames\[0\]\.up must be an array of three numbers/);
        assert.throws(() => loadCameraTrack({ frames: [pose([0, 0, 0])] }, 60), /frames\[0\]\.up must not be a zero vector/);
        assert.throws(() => loadCameraTrack({ frames: [pose(undefined)] }, 60, { x: 0, y: 0, z: 0 }), /default up must not be a zero vector/);
    });

    it('frame list: opposing ups roll through the right vector instead of collapsing', () => {
        // Camera at +Z looking at the origin: forward is -Z, right = forward × up = +X for up +Y.
        const pose = up => ({ position: [0, 0, 5], target: [0, 0, 0], up });
        const track = loadCameraTrack({ frames: [pose([0, 1, 0]), pose([0, -3, 0])] }, 60);
        close(track.poseAt(0.5).up, { x: 1, y: 0, z: 0 });
        const s = Math.SQRT1_2;
        close(track.poseAt(0.25).up, { x: s, y: s, z: 0 });
        close(track.poseAt(0.75).up, { x: s, y: -s, z: 0 });
        for (let t = 0; t <= 1; t += 1 / 16) {
            const u = track.poseAt(t).up;
            assert.ok(Math.abs(Math.hypot(u.x, u.y, u.z) - 1) < 1e-9, `up at ${t} is not unit length`);
        }
    });

    it('editor document and viewer settings use the default up', () => {
        const up = { x: 0, y: 0, z: -1 };
        const editor = loadCameraTrack({
            timeline: { frames: 10, frameRate: 30 },
            poseSets: [{ poses: [
                { frame: 0, position: [0, 0, 5], target: [0, 0, 0], fov: 60 },
                { frame: 5, position: [5, 0, 0], target: [0, 0, 0], fov: 60 }
            ] }]
        }, 60, up);
        close(editor.poseAt(2.5).up, up);
        const viewer = loadCameraTrack({
            animTracks: [{
                duration: 1,
                frameRate: 30,
                keyframes: { times: [0, 30], values: { position: [0, 0, 5, 5, 0, 0], target: [0, 0, 0, 0, 0, 0], fov: [60, 60] } }
            }]
        }, 60, up);
        close(viewer.poseAt(15).up, up);
        close(loadCameraTrack({ frames: [{ position: [0, 0, 5], target: [0, 0, 0] }] }, 60).poseAt(0).up, { x: 0, y: 1, z: 0 });
    });
});
