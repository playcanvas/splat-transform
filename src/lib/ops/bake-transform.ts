import { Mat3, Quat, Vec3 } from 'playcanvas';

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
    const shBands = src.meta.shBands;
    const shPerCh = SH_PER_CHANNEL[shBands];
    const rotIdentity = Math.abs(Math.abs(r.w) - 1) < 1e-6;
    const logS = Math.log(s);

    const _v = new Vec3();
    const _q = new Quat();
    const rotateSH = (!rotIdentity && shPerCh > 0) ? new RotateSH(new Mat3().setFromQuat(r)) : null;
    const shScratch = rotateSH ? new Float32Array(shPerCh) : null;

    const bakePosition = (cd: ChunkData): void => {
        const p = new Float32Array(cd.data);
        for (let i = 0; i < cd.count; i++) {
            const o = i * 3;
            _v.set(p[o], p[o + 1], p[o + 2]);
            delta.transformPoint(_v, _v);
            p[o] = _v.x;
            p[o + 1] = _v.y;
            p[o + 2] = _v.z;
        }
    };

    const bakeGeometric = (cd: ChunkData): void => {
        const g = new Float32Array(cd.data); // [rot0..3, scale0..2, opacity] per row
        for (let i = 0; i < cd.count; i++) {
            const o = i * 8;
            if (!rotIdentity) {
                // rot_0 = w, rot_1..3 = x,y,z; q' = r * q
                _q.set(g[o + 1], g[o + 2], g[o + 3], g[o]).mul2(r, _q);
                g[o] = _q.w;
                g[o + 1] = _q.x;
                g[o + 2] = _q.y;
                g[o + 3] = _q.z;
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
