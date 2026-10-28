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
 * @param coeffsPerChannel - Per-channel SH coefficient count (0/3/8/15/24).
 * @returns WGSL source for the project compute shader.
 */
const projectWgsl = (coeffsPerChannel: number) => /* wgsl */ `
#include "uniformsStruct"
#include "constants"

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(2) var<storage, read_write> projected: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> coverage: array<u32>;

// Splat attribute access. The chunked path uploads an interleaved record
// per gaussian (\`splats\`, stride \`splatStride\`); the scene rasterizer holds
// the source's layers as they are stored (\`position\` 3, \`geometric\` 8 and
// \`color\` 3 + SH floats per gaussian) and processes gaussians in the
// depth-sorted \`order\`. Both resolve to the same f32 values, so the two
// paths render identically.
#ifdef SCENE
@group(0) @binding(1) var<storage, read> position: array<f32>;
@group(0) @binding(4) var<storage, read> geometric: array<f32>;
@group(0) @binding(5) var<storage, read> color: array<f32>;
@group(0) @binding(6) var<storage, read> order: array<u32>;

const COLOR_STRIDE_F32: u32 = ${3 + 3 * coeffsPerChannel}u;

fn attr(c: u32, s: u32) -> f32 {
    if (c < 3u) { return position[s * 3u + c]; }
    if (c < 11u) { return geometric[s * 8u + (c - 3u)]; }
    return color[s * COLOR_STRIDE_F32 + (c - 11u)];
}

fn shc(k: u32, s: u32) -> f32 {
    return color[s * COLOR_STRIDE_F32 + 3u + k];
}
#else
@group(0) @binding(1) var<storage, read> splats: array<f32>;

fn attr(c: u32, s: u32) -> f32 {
    return splats[s * uniforms.splatStride + c];
}

fn shc(k: u32, s: u32) -> f32 {
    return splats[s * uniforms.splatStride + 14u + k];
}
#endif

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
const SH_C4_0: f32 = 2.5033429417967046;
const SH_C4_1: f32 = -1.7701307697799304;
const SH_C4_2: f32 = 0.9461746957575601;
const SH_C4_3: f32 = -0.6690465435572892;
const SH_C4_4: f32 = 0.10578554691520431;
const SH_C4_5: f32 = -0.6690465435572892;
const SH_C4_6: f32 = 0.47308734787878004;
const SH_C4_7: f32 = -1.7701307697799304;
const SH_C4_8: f32 = 0.6258357354491761;

const COEFFS_PER_CHANNEL: u32 = ${coeffsPerChannel}u;

// Over-unity profile (excess opacity): exp(-D/2 * max(0, r - (D-1))^2) for
// r in sigma units, which carries mass
//   I(D) = (D-1)^2/2 + 1/D + (D-1)/2 * sqrt(2 pi / D),  I(1) = 1.
fn profileMass(D: f32) -> f32 {
    let p = D - 1.0;
    return 0.5 * p * p + 1.0 / D + 0.5 * p * sqrt(6.283185307179586 / D);
}

// Inverse of profileMass by bisection; I(D) >= (D-1)^2/2 bounds D above.
fn profileParamForMass(m: f32) -> f32 {
    var lo = 1.0;
    var hi = 1.0 + sqrt(2.0 * m);
    for (var k = 0; k < 24; k = k + 1) {
        let mid = 0.5 * (lo + hi);
        if (profileMass(mid) < m) { lo = mid; } else { hi = mid; }
    }
    return 0.5 * (lo + hi);
}

fn writeInvalid(idx: u32) {
    projected[idx * 3u + 0u] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    projected[idx * 3u + 1u] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    projected[idx * 3u + 2u] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    coverage[idx] = 0u;
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wgId: vec3<u32>,
    @builtin(num_workgroups) numWg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    // Dispatches past the per-dimension workgroup limit are tiled in 2-D;
    // linearise so the chunked (1-D) and resident paths index alike.
    let i = (wgId.y * numWg.x + wgId.x) * 64u + lid.x;
    if (i >= uniforms.chunkSize) { return; }

#ifdef SCENE
    let s = order[i];
#else
    let s = i;
#endif

    let posX = attr(0u, s);
    let posY = attr(1u, s);
    let posZ = attr(2u, s);
    let rotW = attr(3u, s);
    let rotX = attr(4u, s);
    let rotY = attr(5u, s);
    let rotZ = attr(6u, s);
    let lsX = attr(7u, s);
    let lsY = attr(8u, s);
    let lsZ = attr(9u, s);
    let opacity = attr(10u, s);
    // LINEAR_OPACITY: the slot holds the true opacity A (may exceed 1)
    // rather than a logit; see render/linear-opacity.ts.
#ifdef LINEAR_OPACITY
    let amp = opacity;
#else
    let amp = 1.0 / (1.0 + exp(-opacity));
#endif
    // A > 1: draw the plateau profile at shape plateauD (0 = plain gaussian).
    var plateauD = 0.0;
    if (amp > 1.0) {
        plateauD = 1.0 + PLATEAU_SHRINK * (profileParamForMass(amp) - 1.0);
    }
    let fdcR = attr(11u, s);
    let fdcG = attr(12u, s);
    let fdcB = attr(13u, s);

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

    cov00 = cov00 + AA_DILATION_COV;
    cov11 = cov11 + AA_DILATION_COV;

#ifndef PROJECTION_EQUIRECT
    // Defocus (DoF), pinhole only. Capture detPreDoF before dilating so the
    // alpha rescale below conserves integrated energy — without it,
    // defocused foreground splats over-occlude what is behind them.
    let detPreDoF = cov00 * cov11 - cov01 * cov01;
    let coc = uniforms.apertureScale * abs(1.0 - uniforms.focusDistance / cz);
    let cocVar = coc * coc;
    cov00 = cov00 + cocVar;
    cov11 = cov11 + cocVar;
#endif

    // Footprint radius from the larger eigenvalue of the dilated covariance.
    let detU = cov00 * cov11 - cov01 * cov01;
    if (detU <= 0.0) { writeInvalid(i); return; }
    let mid = 0.5 * (cov00 + cov11);
    let disc = sqrt(max(DISCRIMINANT_FLOOR, mid * mid - detU));
    // The plateau profile reaches GAUSSIAN_FLOOR at (D'-1) + SIGMA_CUTOFF/sqrt(D')
    // sigma, which is SIGMA_CUTOFF at D' = 1.
    var extentSigma = SIGMA_CUTOFF;
    if (plateauD > 1.0) { extentSigma = (plateauD - 1.0) + SIGMA_CUTOFF / sqrt(plateauD); }
    var radiusRaw = extentSigma * sqrt(mid + disc);

    // Size clamp: a splat whose radius exceeds SIZE_CLAMP_FRAC of the
    // shorter image edge is scaled down uniformly (covariance × s²) until
    // it fits. Aspect, orientation and alpha are untouched — the editor's
    // behaviour — and the alpha rescale below still uses the unclamped
    // determinant so the clamp only shrinks the footprint, never
    // brightens it. NO_SIZE_CLAMP (measurement renders) renders every
    // splat at its true size.
#ifndef NO_SIZE_CLAMP
    let radiusLimit = SIZE_CLAMP_FRAC * f32(min(uniforms.imageWidth, uniforms.imageHeight));
    if (radiusRaw > radiusLimit) {
        let s = radiusLimit / radiusRaw;
        let s2 = s * s;
        cov00 = cov00 * s2;
        cov01 = cov01 * s2;
        cov11 = cov11 * s2;
        radiusRaw = radiusLimit;
    }
#endif

    let det = cov00 * cov11 - cov01 * cov01;
    if (det <= 0.0) { writeInvalid(i); return; }

    let invDet = 1.0 / det;
    let covInvA = cov11 * invDet;
    let covInvB = -cov01 * invDet;
    let covInvC = cov00 * invDet;

    let radius = ceil(radiusRaw);

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
    let dpx = posX - uniforms.eyeX;
    let dpy = posY - uniforms.eyeY;
    let dpz = posZ - uniforms.eyeZ;
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
#ifdef SH_BAND_4
    #include "shBand4"
#endif

    let colR = max(0.0, cR + 0.5);
    let colG = max(0.0, cG + 0.5);
    let colB = max(0.0, cB + 0.5);

#ifndef PROJECTION_EQUIRECT
    // Energy-preserving alpha rescale for DoF. When apertureScale == 0,
    // detPreDoF == detU so dofAlphaScale == 1 (no-op).
    let dofAlphaScale = sqrt(max(0.0, detPreDoF) / detU);
#else
    let dofAlphaScale = 1.0;
#endif

    // the plateau profile peaks at 1 by construction
    let alpha = select(amp, 1.0, plateauD > 1.0) * dofAlphaScale;

    projected[i * 3u + 0u] = vec4<f32>(screenX, screenY, radius, plateauD);
    projected[i * 3u + 1u] = vec4<f32>(covInvA, covInvB, covInvC, alpha);
    projected[i * 3u + 2u] = vec4<f32>(colR, colG, colB, 0.0);

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
