import { Mat3 } from 'playcanvas';

import {
    type ChunkData,
    type ChunkReadRequest,
    type ChunkSource,
    type ChunkSourceMetadata,
    SH_REST_COUNTS
} from '../source';
import { RotateSH, Transform } from '../utils';

const SH_PER_CHANNEL = [0, 3, 8, 15];

/**
 * Bake a source's pending coordinate-space transform into a target space,
 * lazily and per chunk — the streaming analog of `convertToSpace`.
 *
 * Computes `delta = targetSpace⁻¹ · meta.transform` once; each `read` delegates
 * to the parent (filling the caller's buffers with raw data) and then applies
 * `delta` in place to whichever layers were requested, exactly as
 * `transformColumns` does for a `DataTable`:
 *  - `position`  — full TRS via `transformPoint`
 *  - `geometric` — compose the rotation onto the quaternion; add `log(scale)` to the log-scales
 *  - `color`     — rotate the SH rest coefficients (DC is unaffected)
 *  - `other`     — untouched (user data, not coordinate-dependent)
 *
 * The returned source reports `meta.transform = targetSpace` (its data is now
 * baked). Consumers (writers, GPU feeds) wrap their input with this once and
 * never reimplement transform handling.
 *
 * @param src - The parent source.
 * @param targetSpace - The coordinate space to bake into (e.g. `Transform.PLY`).
 * @returns A derived source whose reads yield data in `targetSpace`.
 */
const bakeTransform = (src: ChunkSource, targetSpace: Transform): ChunkSource => {
    const meta: ChunkSourceMetadata = { ...src.meta, transform: targetSpace.clone() };
    const delta = targetSpace.clone().invert().mul(src.meta.transform);

    if (delta.isIdentity()) {
        return { meta, read: req => src.read(req), close: () => src.close() };
    }

    const r = delta.rotation;
    const s = delta.scale;
    const rx = r.x, ry = r.y, rz = r.z, rw = r.w;
    const tx = delta.translation.x, ty = delta.translation.y, tz = delta.translation.z;
    const shBands = src.meta.shBands;
    const shPerCh = SH_PER_CHANNEL[shBands];
    const rotIdentity = Math.abs(Math.abs(rw) - 1) < 1e-6;
    const logS = Math.log(s);

    const rotateSH = (!rotIdentity && shPerCh > 0) ? new RotateSH(new Mat3().setFromQuat(r)) : null;
    const shScratch = rotateSH ? new Float32Array(shPerCh) : null;

    // Inlined per-element kernels (no Vec3/Quat objects). The float-op order
    // mirrors PlayCanvas's Quat.transformVector / Quat.mul2 exactly — and JS
    // arithmetic is float64 with rounding only on the Float32Array store, just
    // like the legacy path — so output stays byte-identical to convertToSpace.
    const bakePosition = (cd: ChunkData): void => {
        const p = new Float32Array(cd.data);
        for (let i = 0; i < cd.count; i++) {
            const o = i * 3;
            const x = p[o] * s, y = p[o + 1] * s, z = p[o + 2] * s;     // point * scale
            const ix = rw * x + ry * z - rz * y;                        // rotation.transformVector
            const iy = rw * y + rz * x - rx * z;
            const iz = rw * z + rx * y - ry * x;
            const iw = -rx * x - ry * y - rz * z;
            p[o]     = (ix * rw + iw * -rx + iy * -rz - iz * -ry) + tx;  // + translation
            p[o + 1] = (iy * rw + iw * -ry + iz * -rx - ix * -rz) + ty;
            p[o + 2] = (iz * rw + iw * -rz + ix * -ry - iy * -rx) + tz;
        }
    };

    const bakeGeometric = (cd: ChunkData): void => {
        const g = new Float32Array(cd.data); // [rot0..3, scale0..2, opacity] per row
        for (let i = 0; i < cd.count; i++) {
            const o = i * 8;
            if (!rotIdentity) {
                // rot_0 = w, rot_1..3 = x,y,z; q' = r * q  (Quat.mul2, lhs = r)
                const qx = g[o + 1], qy = g[o + 2], qz = g[o + 3], qw = g[o];
                g[o]     = rw * qw - rx * qx - ry * qy - rz * qz;
                g[o + 1] = rw * qx + rx * qw + ry * qz - rz * qy;
                g[o + 2] = rw * qy + ry * qw + rz * qx - rx * qz;
                g[o + 3] = rw * qz + rz * qw + rx * qy - ry * qx;
            }
            if (s !== 1) {
                g[o + 4] += logS;
                g[o + 5] += logS;
                g[o + 6] += logS;
            }
        }
    };

    const bakeColor = (cd: ChunkData): void => {
        if (!rotateSH || !shScratch) return; // DC is unaffected by rotation
        const stride = 3 + SH_REST_COUNTS[shBands]; // floats per row
        const c = new Float32Array(cd.data);
        for (let i = 0; i < cd.count; i++) {
            const base = i * stride + 3; // skip DC
            for (let j = 0; j < 3; j++) {
                const ch = base + j * shPerCh;
                for (let k = 0; k < shPerCh; k++) shScratch[k] = c[ch + k];
                rotateSH.apply(shScratch);
                for (let k = 0; k < shPerCh; k++) c[ch + k] = shScratch[k];
            }
        }
    };

    const read = async (req: ChunkReadRequest): Promise<void> => {
        await src.read(req);
        if (req.position) bakePosition(req.position);
        if (req.geometric) bakeGeometric(req.geometric);
        if (req.color) bakeColor(req.color);
    };

    return { meta, read, close: () => src.close() };
};

export { bakeTransform };
