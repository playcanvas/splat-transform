import { type CameraBasis } from './camera';
import { GaussianBVH } from '../spatial';

/**
 * Compute the 6 world-space frustum planes for the view sub-frustum that
 * intersects a tile-group's pixel rectangle.
 *
 * Each plane is stored as `(nx, ny, nz, d)` where the inside-the-frustum
 * test is `n · p ≥ d`. Plane normals point *into* the frustum. The plane
 * order in `out` is: near, far, left, right, top, bottom (not relied on
 * by callers — `GaussianBVH.queryFrustumRawInto` treats all six uniformly).
 *
 * Side-plane derivation: a pixel edge at column `gx` corresponds to camera-
 * space `cx = (gx − cx0) / focalX · cz`. The plane containing all rays
 * through that edge is `right − (gx − cx0)/focalX · forward`. The "into
 * the frustum" direction is whichever sign of the resulting normal
 * satisfies the interior half-space.
 *
 * @param camera - Camera basis (right, down, forward, eye, focals).
 * @param gx0 - Group min pixel x (inclusive).
 * @param gy0 - Group min pixel y (inclusive).
 * @param gx1 - Group max pixel x (exclusive).
 * @param gy1 - Group max pixel y (exclusive).
 * @param width - Image width in pixels.
 * @param height - Image height in pixels.
 * @param near - Near plane.
 * @param far - Far plane.
 * @param out - 24-float output buffer; must be at least length 24.
 */
const computeGroupFrustumPlanes = (
    camera: CameraBasis,
    gx0: number, gy0: number, gx1: number, gy1: number,
    width: number, height: number,
    near: number, far: number,
    out: Float32Array
): void => {
    const rx = camera.right.x, ry = camera.right.y, rz = camera.right.z;
    const dx = camera.down.x, dy = camera.down.y, dz = camera.down.z;
    const fx = camera.forward.x, fy = camera.forward.y, fz = camera.forward.z;
    const ex = camera.eye.x, ey = camera.eye.y, ez = camera.eye.z;

    const txMin = (gx0 - width * 0.5) / camera.focalX;
    const txMax = (gx1 - width * 0.5) / camera.focalX;
    const tyMin = (gy0 - height * 0.5) / camera.focalY;
    const tyMax = (gy1 - height * 0.5) / camera.focalY;

    // Helper: write plane `n · p = d` into out[idx..idx+3]. d is computed
    // from a reference point `(nx·ex + ny·ey + nz·ez) ± offset` where
    // `+ offset` shifts the plane forward along the normal so the inside-
    // the-frustum test `n · p ≥ d` holds for points inside.
    const fxe = fx * ex + fy * ey + fz * ez;

    // Near plane: forward · (p − eye) ≥ near
    //   n = forward, d = forward·eye + near
    out[0] = fx; out[1] = fy; out[2] = fz; out[3] = fxe + near;

    // Far plane: forward · (p − eye) ≤ far
    //   ⇒ -forward · (p − eye) ≥ -far
    //   ⇒ n = -forward, d = -forward·eye - far
    out[4] = -fx; out[5] = -fy; out[6] = -fz; out[7] = -fxe - far;

    // Left plane: cx ≥ txMin · cz
    //   (right − txMin · forward) · (p − eye) ≥ 0
    //   n = right − txMin·forward, d = n·eye
    {
        const nx = rx - txMin * fx;
        const ny = ry - txMin * fy;
        const nz = rz - txMin * fz;
        const d = nx * ex + ny * ey + nz * ez;
        out[8] = nx; out[9] = ny; out[10] = nz; out[11] = d;
    }

    // Right plane: cx ≤ txMax · cz
    //   n = txMax·forward − right
    {
        const nx = txMax * fx - rx;
        const ny = txMax * fy - ry;
        const nz = txMax * fz - rz;
        const d = nx * ex + ny * ey + nz * ez;
        out[12] = nx; out[13] = ny; out[14] = nz; out[15] = d;
    }

    // Top plane: cy ≥ tyMin · cz
    //   n = down − tyMin·forward
    {
        const nx = dx - tyMin * fx;
        const ny = dy - tyMin * fy;
        const nz = dz - tyMin * fz;
        const d = nx * ex + ny * ey + nz * ez;
        out[16] = nx; out[17] = ny; out[18] = nz; out[19] = d;
    }

    // Bottom plane: cy ≤ tyMax · cz
    //   n = tyMax·forward − down
    {
        const nx = tyMax * fx - dx;
        const ny = tyMax * fy - dy;
        const nz = tyMax * fz - dz;
        const d = nx * ex + ny * ey + nz * ez;
        out[20] = nx; out[21] = ny; out[22] = nz; out[23] = d;
    }
};

/**
 * Run a frustum-plane BVH query, resizing `buffer` if it's too small.
 *
 * @param bvh - Pre-built `GaussianBVH` over the scene.
 * @param planes - 24-float plane array from `computeGroupFrustumPlanes`.
 * @param buffer - Reusable result buffer (grown if too small).
 * @returns Possibly-reallocated buffer and the candidate count.
 */
const queryBvhFrustum = (
    bvh: GaussianBVH,
    planes: Float32Array,
    buffer: Uint32Array
): { buffer: Uint32Array; count: number } => {
    let count = bvh.queryFrustumRawInto(planes, buffer, 0);
    if (count > buffer.length) {
        const newSize = Math.max(buffer.length * 2, count);
        const grown = new Uint32Array(newSize);
        count = bvh.queryFrustumRawInto(planes, grown, 0);
        return { buffer: grown, count };
    }
    return { buffer, count };
};

export { computeGroupFrustumPlanes, queryBvhFrustum };
