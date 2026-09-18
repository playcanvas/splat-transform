/**
 * WGSL source for the project compute. Reads raw splat fields from the
 * per-slot input buffer, evaluates SH for view-dependent color, computes
 * 2D inverse covariance + screen-space 3σ radius, and writes a packed
 * projection record per gaussian. Invalid splats (behind near, degenerate
 * covariance, outside group AABB) are written with `radius = 0` so the
 * rasterizer can early-out on the first vec4 load.
 *
 * Projection-mode variation (pinhole vs equirect) is handled at WGSL
 * preprocessor time via `#ifdef PROJECTION_EQUIRECT` blocks that pull
 * in the projection-specific chunks (screen mapping + Jacobian + tile
 * AABB). The per-render `MAX_COVERAGE_PER_SPLAT` cap is similarly
 * embedded by the tile-AABB chunk via JS-template substitution at
 * construction time (see `sharedCincludes` in the rasterizer ctor).
 *
 * Motion blur (`MOTION_BLUR`, pinhole only) projects each gaussian a
 * second time with the shutter-close basis, averages the two 2D
 * covariances, moves the centre to the shutter midpoint and stores the
 * half displacement in the record's two spare floats (`v0.w`, `v2.w`) for
 * the rasterizer to integrate over.
 *
 * @param coeffsPerChannel - Per-channel SH coefficient count (0/3/8/15).
 * @returns WGSL source for the project compute shader.
 */
const projectWgsl = (coeffsPerChannel: number) => /* wgsl */`
#include "uniformsStruct"
#include "constants"

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> splats: array<f32>;
@group(0) @binding(2) var<storage, read_write> projected: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> coverage: array<u32>;

#include "covariance3DFns"
#include "jacobianPinholeFns"

const SH_C0: f32 = 0.28209479177387814;
const SH_C1: f32 = 0.4886025119029199;
const SH_C2_0: f32 = 1.0925484305920792;
const SH_C2_1: f32 = -1.0925484305920792;
const SH_C2_2: f32 = 0.31539156525252005;
const SH_C2_3: f32 = -1.0925484305920792;
const SH_C2_4: f32 = 0.5462742152960396;
const SH_C3_0: f32 = -0.5900435899266435;
const SH_C3_1: f32 = 2.890611442640554;
const SH_C3_2: f32 = -0.4570457994644658;
const SH_C3_3: f32 = 0.3731763325901154;
const SH_C3_4: f32 = -0.4570457994644658;
const SH_C3_5: f32 = 1.445305721320277;
const SH_C3_6: f32 = -0.5900435899266435;

const COEFFS_PER_CHANNEL: u32 = ${coeffsPerChannel}u;

fn writeInvalid(idx: u32) {
    projected[idx * 3u + 0u] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    projected[idx * 3u + 1u] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    projected[idx * 3u + 2u] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    coverage[idx] = 0u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= uniforms.chunkSize) { return; }

    let base = i * uniforms.splatStride;

    let posX = splats[base + 0u];
    let posY = splats[base + 1u];
    let posZ = splats[base + 2u];
    let rotW = splats[base + 3u];
    let rotX = splats[base + 4u];
    let rotY = splats[base + 5u];
    let rotZ = splats[base + 6u];
    let lsX = splats[base + 7u];
    let lsY = splats[base + 8u];
    let lsZ = splats[base + 9u];
    let opacity = splats[base + 10u];
    let fdcR = splats[base + 11u];
    let fdcG = splats[base + 12u];
    let fdcB = splats[base + 13u];

    // World → camera
    let wx = posX - uniforms.eyeX;
    let wy = posY - uniforms.eyeY;
    let wz = posZ - uniforms.eyeZ;
    let cx = uniforms.rightX * wx + uniforms.rightY * wy + uniforms.rightZ * wz;
    let cy = uniforms.downX * wx + uniforms.downY * wy + uniforms.downZ * wz;
    let cz = uniforms.forwardX * wx + uniforms.forwardY * wy + uniforms.forwardZ * wz;

#ifdef PROJECTION_EQUIRECT
    #include "projectionEquirect"
#else
    #include "projectionPinhole"
#endif

    #include "quatRotation"
    #include "covariance3D"

#ifdef PROJECTION_EQUIRECT
    #include "jacobianEquirect"
#else
    #include "jacobianPinhole"
#endif

    // Depth and eye used for defocus and the SH view direction: the single
    // pose for a static render, the shutter midpoint under motion blur.
    var czView = cz;
    var eyeX = uniforms.eyeX;
    var eyeY = uniforms.eyeY;
    var eyeZ = uniforms.eyeZ;
    // Half of the screen-space displacement over the shutter (zero when
    // static), handed to the rasterizer through the projected record.
    var hx = 0.0;
    var hy = 0.0;

#ifdef MOTION_BLUR
    // Shutter-close pose: project centre and footprint again, average the
    // two 2D covariances and move the centre to the midpoint. Both ends
    // must sit in front of the near plane.
    let wxB = posX - uniforms.eyeBX;
    let wyB = posY - uniforms.eyeBY;
    let wzB = posZ - uniforms.eyeBZ;
    let cxB = uniforms.rightBX * wxB + uniforms.rightBY * wyB + uniforms.rightBZ * wzB;
    let cyB = uniforms.downBX * wxB + uniforms.downBY * wyB + uniforms.downBZ * wzB;
    let czB = uniforms.forwardBX * wxB + uniforms.forwardBY * wyB + uniforms.forwardBZ * wzB;
    if (czB <= uniforms.near) { writeInvalid(i); return; }
    let invZB = 1.0 / czB;
    let screenXB = uniforms.focalX * cxB * invZB + f32(uniforms.imageWidth) * 0.5;
    let screenYB = uniforms.focalY * cyB * invZB + f32(uniforms.imageHeight) * 0.5;
    let ccB = camCov(
        uniforms.rightBX, uniforms.rightBY, uniforms.rightBZ,
        uniforms.downBX, uniforms.downBY, uniforms.downBZ,
        uniforms.forwardBX, uniforms.forwardBY, uniforms.forwardBZ,
        sig00, sig01, sig02, sig11, sig12, sig22
    );
    let covB = cov2dPinhole(cxB, cyB, czB, invZB, ccB.c00, ccB.c01, ccB.c02, ccB.c11, ccB.c12, ccB.c22);
    cov00 = 0.5 * (cov00 + covB.x);
    cov01 = 0.5 * (cov01 + covB.y);
    cov11 = 0.5 * (cov11 + covB.z);
    hx = 0.5 * (screenXB - screenX);
    hy = 0.5 * (screenYB - screenY);
    screenX = screenX + hx;
    screenY = screenY + hy;
    czView = 0.5 * (cz + czB);
    eyeX = 0.5 * (uniforms.eyeX + uniforms.eyeBX);
    eyeY = 0.5 * (uniforms.eyeY + uniforms.eyeBY);
    eyeZ = 0.5 * (uniforms.eyeZ + uniforms.eyeBZ);
#endif

    cov00 = cov00 + AA_DILATION_COV;
    cov11 = cov11 + AA_DILATION_COV;

#ifndef PROJECTION_EQUIRECT
    // Defocus (DoF), pinhole only. Capture detPreDoF before dilating so the
    // alpha rescale below conserves integrated energy — without it,
    // defocused foreground splats over-occlude what is behind them.
    let detPreDoF = cov00 * cov11 - cov01 * cov01;
    let coc = uniforms.apertureScale * abs(1.0 - uniforms.focusDistance / czView);
    let cocVar = coc * coc;
    cov00 = cov00 + cocVar;
    cov11 = cov11 + cocVar;
#endif

    let det = cov00 * cov11 - cov01 * cov01;
    if (det <= 0.0) { writeInvalid(i); return; }

    let invDet = 1.0 / det;
    let covInvA = cov11 * invDet;
    let covInvB = -cov01 * invDet;
    let covInvC = cov00 * invDet;

    let mid = 0.5 * (cov00 + cov11);
    let disc = sqrt(max(DISCRIMINANT_FLOOR, mid * mid - det));
    let lambdaMax = mid + disc;
    let radiusRaw = SIGMA_CUTOFF * sqrt(lambdaMax);

    // Outlier-splat fade: huge splats (close-by mega-splats or pathological
    // training output) would otherwise project to a screen-spanning footprint
    // and tint the whole frame. Linearly fade alpha from 1 to 0 as the
    // un-clamped radius grows from fadeStart to fadeEnd, and discard
    // beyond. The bbox we hand to the rasterizer is clamped at fadeEnd
    // so the binner doesn't reserve tile coverage for a splat that
    // contributes zero anyway. Softer than a hard clamp: prevents the
    // visible pop as the camera approaches a clipped splat.
    //
    // Thresholds are fractions of image height so the SAME world-space
    // splats fade at every render resolution — preserves cross-
    // resolution consistency (e.g. 8K-downsampled-to-1080p matches
    // 1080p direct).
    //
    // NO_RADIUS_FADE (measurement renders) keeps every splat at full
    // alpha whatever its footprint; the group AABB and coverage cap
    // still bound the work.
    // Half of the segment the rasterizer integrates over (zero when
    // static), added to the bbox around the midpoint.
    let hLen = sqrt(hx * hx + hy * hy);
#ifdef NO_RADIUS_FADE
    let radiusFade = 1.0;
    let radius = ceil(radiusRaw + hLen);
#else
    let fadeStart = RADIUS_FADE_START_FRAC * f32(uniforms.imageHeight);
    let fadeEnd = RADIUS_FADE_END_FRAC * f32(uniforms.imageHeight);
    let radiusFade = clamp((fadeEnd - radiusRaw) / (fadeEnd - fadeStart), 0.0, 1.0);
    if (radiusFade <= 0.0) { writeInvalid(i); return; }
    let radius = ceil(min(radiusRaw, fadeEnd) + hLen);
#endif

    // Group AABB cull. The BVH frustum query may include splats whose
    // 3D AABB grazes the frustum but whose 2D footprint misses the group.
    let gx0 = f32(uniforms.groupPixelMinX);
    let gx1 = f32(uniforms.groupPixelMaxX);
    let gy0 = f32(uniforms.groupPixelMinY);
    let gy1 = f32(uniforms.groupPixelMaxY);
    if (screenX + radius < gx0 || screenX - radius >= gx1 ||
        screenY + radius < gy0 || screenY - radius >= gy1) {
        writeInvalid(i);
        return;
    }

    // View-dependent color via SH evaluation.
    let dpx = posX - eyeX;
    let dpy = posY - eyeY;
    let dpz = posZ - eyeZ;
    let dirLen = max(1e-30, sqrt(dpx * dpx + dpy * dpy + dpz * dpz));
    let dirX = dpx / dirLen;
    let dirY = dpy / dirLen;
    let dirZ = dpz / dirLen;

    var cR = SH_C0 * fdcR;
    var cG = SH_C0 * fdcG;
    var cB = SH_C0 * fdcB;

#ifdef SH_BAND_1
    #include "shBand1"
#endif
#ifdef SH_BAND_2
    #include "shBand2"
#endif
#ifdef SH_BAND_3
    #include "shBand3"
#endif

    let colR = max(0.0, cR + 0.5);
    let colG = max(0.0, cG + 0.5);
    let colB = max(0.0, cB + 0.5);

#ifndef PROJECTION_EQUIRECT
    // Energy-preserving alpha rescale for DoF. When apertureScale == 0,
    // detPreDoF == det so dofAlphaScale == 1 (no-op).
    let dofAlphaScale = sqrt(max(0.0, detPreDoF) / det);
#else
    let dofAlphaScale = 1.0;
#endif

    let alpha = (1.0 / (1.0 + exp(-opacity))) * radiusFade * dofAlphaScale;

    projected[i * 3u + 0u] = vec4<f32>(screenX, screenY, radius, hx);
    projected[i * 3u + 1u] = vec4<f32>(covInvA, covInvB, covInvC, alpha);
    projected[i * 3u + 2u] = vec4<f32>(colR, colG, colB, hy);

    // Per-splat tile-coverage count, clamped at maxCoveragePerSplat.
    // Tile indices are GROUP-LOCAL (= image-tile-index minus the
    // group's origin in tiles), so values cover [0, groupTilesX-1] ×
    // [0, groupTilesY-1] for every group regardless of its position in
    // the image. Splats outside the group's pixel rectangle were
    // already culled by the AABB check above.
    let tsz: f32 = f32(TILE_SIZE);
    let gox = f32(uniforms.groupPixelOriginX);
    let goy = f32(uniforms.groupPixelOriginY);
#ifdef PROJECTION_EQUIRECT
    #include "tileAabbEquirect"
#else
    #include "tileAabbPinhole"
#endif
}
`;

export { projectWgsl };
