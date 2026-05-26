import {
    BUFFERUSAGE_COPY_DST,
    BUFFERUSAGE_COPY_SRC,
    SHADERLANGUAGE_WGSL,
    SHADERSTAGE_COMPUTE,
    UNIFORMTYPE_FLOAT,
    UNIFORMTYPE_UINT,
    BindGroupFormat,
    BindStorageBufferFormat,
    BindUniformBufferFormat,
    Compute,
    ComputeRadixSort,
    GraphicsDevice,
    Shader,
    StorageBuffer,
    UniformBufferFormat,
    UniformFormat
} from 'playcanvas';

import { type Projection } from '../render/camera';
import {
    AA_DILATION_COV,
    DISCRIMINANT_FLOOR,
    JACOBIAN_LIMIT_FACTOR,
    MIN_ALPHA,
    MIN_TRANSMITTANCE,
    OPACITY_CAP,
    POLE_EPS,
    RADIUS_FADE_END_FRAC,
    RADIUS_FADE_START_FRAC,
    SIGMA_CUTOFF,
    TILE_SIZE
} from '../render/config';

/**
 * Format a JS number as a WGSL `f32` literal. Adds an explicit `.0` so
 * integer-valued constants like `3` aren't parsed as `AbstractInt` —
 * keeps shaders readable when the constant flips to a fractional value.
 *
 * @param n - Numeric value to format.
 * @returns WGSL literal string with explicit `.0` for integer values.
 */
const wgslF32 = (n: number): string => {
    const s = n.toString();
    return s.includes('.') || s.includes('e') || s.includes('E') ? s : `${s}.0`;
};

/** 12 floats per projected splat: vec4 × 3. */
const PROJECTION_STRIDE_F32 = 12;

/** 4 floats per group pixel: (R, G, B, T). */
const RUNNING_STATE_STRIDE_F32 = 4;

/** RGBA8 output is one u32 per group pixel. */
const OUTPUT_STRIDE_U32 = 1;

/**
 * Configuration for a `GpuSplatRasterizer`. Fixed across the lifetime of
 * a render — `numSHBands` and the group tile dimensions determine GPU
 * buffer sizes and shader uniform layouts.
 *
 * Sizes are expressed as a "group" tile rectangle (`groupTilesX ×
 * groupTilesY`). For a single-pass render the group covers the whole
 * image, so the buffers are exactly image-sized. The group abstraction
 * is retained as a hook for future subframe splitting (each subframe is
 * an independent group sharing the global depth sort) — the project
 * shader's group-AABB cull and group-pixel-origin uniforms still
 * exercise this code path.
 */
interface SplatRasterizerOptions {
    /** Number of SH bands above DC (0–3). Determines input stride. */
    numSHBands: 0 | 1 | 2 | 3;
    /**
     * Camera projection mode. Specializes the project, emit-pairs and
     * rasterize-binned shaders. `pinhole` (default) uses the classical
     * perspective + EWA Jacobian path; `equirect` uses spherical
     * (atan2/asin) screen mapping, a non-linear Jacobian, radial view
     * depth, and tile-bin / rasterize paths that wrap the X axis at the
     * ±π longitude seam.
     */
    projection: Projection;
    /** Tiles per group along X (≤ imageTilesX). Sizes runningState/output. */
    groupTilesX: number;
    /** Tiles per group along Y (≤ imageTilesY). Sizes runningState/output. */
    groupTilesY: number;
    /** Max gaussians per chunk; sizes the input + projection + pair buffers. */
    chunkCap: number;
    /**
     * Hard upper bound on per-splat tile coverage. The project shader
     * clamps `coverage[i] = min(rawBboxArea, maxCoveragePerSplat)`, so
     * the pair buffer is bounded by `chunkCap × maxCoveragePerSplat`
     * regardless of scene/screen size. If the cap ever bites, the
     * emit-pairs shader walks the bbox row-major and stops once it
     * has written `coverage[i]` pairs — i.e. it truncates the bbox at
     * its bottom-right corner.
     *
     * The orchestrator sets this to the group's full tile area so the
     * clamp is geometrically unreachable (any in-group bbox ≤ group
     * area ≤ cap), making truncation a non-issue in practice. The cap
     * is retained as a defensive ceiling on the pair buffer.
     */
    maxCoveragePerSplat: number;
    /** Output image width in pixels (constant per render). */
    imageWidth: number;
    /** Output image height in pixels (constant per render). */
    imageHeight: number;
    /** Near plane distance in world units. */
    near: number;
    /** Camera basis: rows are (right, down, forward) of the world→camera rotation. */
    rightX: number; rightY: number; rightZ: number;
    downX: number; downY: number; downZ: number;
    forwardX: number; forwardY: number; forwardZ: number;
    /** Camera eye position in world space. */
    eyeX: number; eyeY: number; eyeZ: number;
    /** Focal lengths in pixel units. */
    focalX: number; focalY: number;
    /** RGBA background, each channel in [0, 1]. */
    bgR: number; bgG: number; bgB: number; bgA: number;
}

const numSHCoeffsPerChannel = (bands: number): number => {
    return bands === 0 ? 0 : bands === 1 ? 3 : bands === 2 ? 8 : 15;
};

/**
 * WGSL source for the project compute. Reads raw splat fields from the
 * per-slot input buffer, evaluates SH for view-dependent color, computes
 * 2D inverse covariance + screen-space 3σ radius, and writes a packed
 * projection record per gaussian. Invalid splats (behind near, degenerate
 * covariance, outside group AABB) are written with `radius = 0` so the
 * rasterizer can early-out on the first vec4 load.
 *
 * @param coeffsPerChannel - Per-channel SH coefficient count (0/3/8/15).
 * @param maxCoveragePerSplat - Hard upper bound on per-splat tile count.
 * Embedded into the shader's `coverage[i] = min(raw, cap)` clamp. The
 * orchestrator sets this to the group's full tile area so the clamp is
 * geometrically unreachable in practice; emit-pairs walks the bbox
 * row-major and truncates at the bottom-right if the cap is ever hit.
 * @param projection - Camera projection. `pinhole` swaps in the
 * classical perspective + EWA path; `equirect` swaps in spherical
 * mapping, a non-linear Jacobian, and wrap-aware X tile coverage.
 * @returns WGSL source for the project compute shader.
 */
const projectWgsl = (coeffsPerChannel: number, maxCoveragePerSplat: number, projection: Projection) => /* wgsl */`
struct Uniforms {
    rightX: f32, rightY: f32, rightZ: f32, _p0: f32,
    downX: f32, downY: f32, downZ: f32, _p1: f32,
    forwardX: f32, forwardY: f32, forwardZ: f32, _p2: f32,
    eyeX: f32, eyeY: f32, eyeZ: f32, _p3: f32,
    focalX: f32, focalY: f32, near: f32, _p4: f32,
    imageWidth: u32, imageHeight: u32, splatStride: u32, chunkSize: u32,
    groupPixelMinX: u32, groupPixelMinY: u32, groupPixelMaxX: u32, groupPixelMaxY: u32,
    groupTilesX: u32, groupTilesY: u32, groupPixelOriginX: u32, groupPixelOriginY: u32,
    bgR: f32, bgG: f32, bgB: f32, bgA: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> splats: array<f32>;
@group(0) @binding(2) var<storage, read_write> projected: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> coverage: array<u32>;

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

${projection === 'pinhole' ? /* wgsl */`
    if (cz <= uniforms.near) { writeInvalid(i); return; }

    let invZ = 1.0 / cz;
    let screenX = uniforms.focalX * cx * invZ + f32(uniforms.imageWidth) * 0.5;
    let screenY = uniforms.focalY * cy * invZ + f32(uniforms.imageHeight) * 0.5;
` : /* wgsl */`
    // Equirect: cull by radial distance (splats inside the near sphere
    // would otherwise project to a wildly unstable longitude). The
    // longitude is undefined at the camera origin and degenerates near
    // the poles; the POLE_EPS clamp on rxz below keeps the Jacobian
    // bounded for splats arbitrarily close to the zenith/nadir.
    let r2 = cx * cx + cy * cy + cz * cz;
    if (r2 <= uniforms.near * uniforms.near) { writeInvalid(i); return; }
    let r = sqrt(r2);
    let rxz2 = cx * cx + cz * cz;
    let rxz = sqrt(rxz2);
    let rxzClamped = max(rxz, ${wgslF32(POLE_EPS)} * r);

    let invTwoPi: f32 = 0.15915494309189535;
    let invPi:    f32 = 0.3183098861837907;
    let imgWf = f32(uniforms.imageWidth);
    let imgHf = f32(uniforms.imageHeight);
    let lon = atan2(cx, cz);
    let sinLat = clamp(cy / r, -1.0, 1.0);
    let lat = asin(sinLat);
    let screenX = (lon * invTwoPi + 0.5) * imgWf;
    // cy > 0 = camera-down axis = below horizon → lat > 0 → screenY in
    // the bottom half. screenY = 0 maps to the zenith (above the camera).
    let screenY = (lat * invPi + 0.5) * imgHf;
`}

    // Quaternion normalize.
    let qlen2 = rotW * rotW + rotX * rotX + rotY * rotY + rotZ * rotZ;
    if (qlen2 == 0.0) { writeInvalid(i); return; }
    let invQ = inverseSqrt(qlen2);
    let qw = rotW * invQ;
    let qx = rotX * invQ;
    let qy = rotY * invQ;
    let qz = rotZ * invQ;

    // Rotation matrix from unit quaternion.
    let xx = qx * qx; let yy = qy * qy; let zz = qz * qz;
    let xy = qx * qy; let xzq = qx * qz; let yz = qy * qz;
    let wxq = qw * qx; let wy_ = qw * qy; let wzq = qw * qz;
    let r00 = 1.0 - 2.0 * (yy + zz);
    let r01 = 2.0 * (xy - wzq);
    let r02 = 2.0 * (xzq + wy_);
    let r10 = 2.0 * (xy + wzq);
    let r11 = 1.0 - 2.0 * (xx + zz);
    let r12 = 2.0 * (yz - wxq);
    let r20 = 2.0 * (xzq - wy_);
    let r21 = 2.0 * (yz + wxq);
    let r22 = 1.0 - 2.0 * (xx + yy);

    let sx = exp(lsX);
    let sy = exp(lsY);
    let sz = exp(lsZ);

    let m00 = r00 * sx; let m01 = r01 * sy; let m02 = r02 * sz;
    let m10 = r10 * sx; let m11 = r11 * sy; let m12 = r12 * sz;
    let m20 = r20 * sx; let m21 = r21 * sy; let m22 = r22 * sz;

    let sig00 = m00 * m00 + m01 * m01 + m02 * m02;
    let sig01 = m00 * m10 + m01 * m11 + m02 * m12;
    let sig02 = m00 * m20 + m01 * m21 + m02 * m22;
    let sig11 = m10 * m10 + m11 * m11 + m12 * m12;
    let sig12 = m10 * m20 + m11 * m21 + m12 * m22;
    let sig22 = m20 * m20 + m21 * m21 + m22 * m22;

    let v00 = uniforms.rightX; let v01 = uniforms.rightY; let v02 = uniforms.rightZ;
    let v10 = uniforms.downX; let v11 = uniforms.downY; let v12 = uniforms.downZ;
    let v20 = uniforms.forwardX; let v21 = uniforms.forwardY; let v22 = uniforms.forwardZ;

    let t00 = v00 * sig00 + v01 * sig01 + v02 * sig02;
    let t01 = v00 * sig01 + v01 * sig11 + v02 * sig12;
    let t02 = v00 * sig02 + v01 * sig12 + v02 * sig22;
    let t10 = v10 * sig00 + v11 * sig01 + v12 * sig02;
    let t11 = v10 * sig01 + v11 * sig11 + v12 * sig12;
    let t12 = v10 * sig02 + v11 * sig12 + v12 * sig22;
    let t20 = v20 * sig00 + v21 * sig01 + v22 * sig02;
    let t21 = v20 * sig01 + v21 * sig11 + v22 * sig12;
    let t22 = v20 * sig02 + v21 * sig12 + v22 * sig22;

    let c00 = t00 * v00 + t01 * v01 + t02 * v02;
    let c01 = t00 * v10 + t01 * v11 + t02 * v12;
    let c02 = t00 * v20 + t01 * v21 + t02 * v22;
    let c11 = t10 * v10 + t11 * v11 + t12 * v12;
    let c12 = t10 * v20 + t11 * v21 + t12 * v22;
    let c22 = t20 * v20 + t21 * v21 + t22 * v22;

${projection === 'pinhole' ? /* wgsl */`
    // Pinhole Jacobian with x/z, y/z clamp. J is the 2×3 matrix
    //   [[jx0,   0, jx2],
    //    [  0, jy1, jy2]]
    // — the zero entries let us drop the u01·jy0 / u10 / u11·jy0 terms
    // in cov = J·Σ·Jᵀ that the equirect path retains.
    let limX = ${wgslF32(JACOBIAN_LIMIT_FACTOR)} * (f32(uniforms.imageWidth) * 0.5) / uniforms.focalX;
    let limY = ${wgslF32(JACOBIAN_LIMIT_FACTOR)} * (f32(uniforms.imageHeight) * 0.5) / uniforms.focalY;
    let txtz = clamp(cx * invZ, -limX, limX);
    let tytz = clamp(cy * invZ, -limY, limY);
    let jcx = txtz * cz;
    let jcy = tytz * cz;
    let jx0 = uniforms.focalX * invZ;
    let jx2 = -uniforms.focalX * jcx * invZ * invZ;
    let jy1 = uniforms.focalY * invZ;
    let jy2 = -uniforms.focalY * jcy * invZ * invZ;

    let u00 = jx0 * c00 + jx2 * c02;
    let u01 = jx0 * c01 + jx2 * c12;
    let u02 = jx0 * c02 + jx2 * c22;
    let u11 = jy1 * c11 + jy2 * c12;
    let u12 = jy1 * c12 + jy2 * c22;

    var cov00 = u00 * jx0 + u02 * jx2;
    let cov01 = u01 * jy1 + u02 * jy2;
    var cov11 = u11 * jy1 + u12 * jy2;
` : /* wgsl */`
    // Equirect Jacobian. With longitude θ = atan2(cx, cz) and latitude
    // φ = asin(cy/r) (cy is the camera-down axis, so φ>0 = below the
    // horizon), the per-axis screen derivatives multiplied by the pixel
    // scales (kx, ky) = (W/(2π), H/π) give the 2×3 Jacobian:
    //   ∂screenX/∂(cx,cy,cz) = kx · ( cz/rxz²,  0,             -cx/rxz² )
    //   ∂screenY/∂(cx,cy,cz) = ky · (-cx·cy/(r²·rxz),  rxz/r²,  -cy·cz/(r²·rxz) )
    // rxzClamped (≥ POLE_EPS·r) keeps every denominator finite as a
    // splat approaches the pole. j[0][1] = 0, j[1][0] ≠ 0 — the cov
    // formula carries the extra u00·jy0 / u10·jy0 / u11·jy0 terms that
    // the pinhole simplification dropped.
    let kx = imgWf * invTwoPi;
    let ky = imgHf * invPi;
    let invRxzC2 = 1.0 / (rxzClamped * rxzClamped);
    let invR2 = 1.0 / r2;
    let invR2Rxz = invR2 / rxzClamped;
    let jx0 =  kx * cz * invRxzC2;
    let jx2 = -kx * cx * invRxzC2;
    let jy0 = -ky * cx * cy * invR2Rxz;
    let jy1 =  ky * rxzClamped * invR2;
    let jy2 = -ky * cy * cz * invR2Rxz;

    let u00 = jx0 * c00 + jx2 * c02;
    let u01 = jx0 * c01 + jx2 * c12;
    let u02 = jx0 * c02 + jx2 * c22;
    let u10 = jy0 * c00 + jy1 * c01 + jy2 * c02;
    let u11 = jy0 * c01 + jy1 * c11 + jy2 * c12;
    let u12 = jy0 * c02 + jy1 * c12 + jy2 * c22;

    var cov00 = u00 * jx0 + u02 * jx2;
    let cov01 = u00 * jy0 + u01 * jy1 + u02 * jy2;
    var cov11 = u10 * jy0 + u11 * jy1 + u12 * jy2;
`}

    cov00 = cov00 + ${wgslF32(AA_DILATION_COV)};
    cov11 = cov11 + ${wgslF32(AA_DILATION_COV)};

    let det = cov00 * cov11 - cov01 * cov01;
    if (det <= 0.0) { writeInvalid(i); return; }

    let invDet = 1.0 / det;
    let covInvA = cov11 * invDet;
    let covInvB = -cov01 * invDet;
    let covInvC = cov00 * invDet;

    let mid = 0.5 * (cov00 + cov11);
    let disc = sqrt(max(${wgslF32(DISCRIMINANT_FLOOR)}, mid * mid - det));
    let lambdaMax = mid + disc;
    let radiusRaw = ${wgslF32(SIGMA_CUTOFF)} * sqrt(lambdaMax);

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
    let fadeStart = ${wgslF32(RADIUS_FADE_START_FRAC)} * f32(uniforms.imageHeight);
    let fadeEnd = ${wgslF32(RADIUS_FADE_END_FRAC)} * f32(uniforms.imageHeight);
    let radiusFade = clamp((fadeEnd - radiusRaw) / (fadeEnd - fadeStart), 0.0, 1.0);
    if (radiusFade <= 0.0) { writeInvalid(i); return; }
    let radius = ceil(min(radiusRaw, fadeEnd));

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

    ${coeffsPerChannel >= 3 ? `
    {
        let n = COEFFS_PER_CHANNEL;
        let shBase = base + 14u;
        let b0 = -SH_C1 * dirY;
        let b1 = SH_C1 * dirZ;
        let b2 = -SH_C1 * dirX;
        cR = cR + b0 * splats[shBase + 0u] + b1 * splats[shBase + 1u] + b2 * splats[shBase + 2u];
        cG = cG + b0 * splats[shBase + n + 0u] + b1 * splats[shBase + n + 1u] + b2 * splats[shBase + n + 2u];
        cB = cB + b0 * splats[shBase + 2u * n + 0u] + b1 * splats[shBase + 2u * n + 1u] + b2 * splats[shBase + 2u * n + 2u];
    }` : ''}
    ${coeffsPerChannel >= 8 ? `
    {
        let n = COEFFS_PER_CHANNEL;
        let shBase = base + 14u;
        let xx2 = dirX * dirX;
        let yy2 = dirY * dirY;
        let zz2 = dirZ * dirZ;
        let xy2 = dirX * dirY;
        let yz2 = dirY * dirZ;
        let xz2 = dirX * dirZ;
        let b3 = SH_C2_0 * xy2;
        let b4 = SH_C2_1 * yz2;
        let b5 = SH_C2_2 * (2.0 * zz2 - xx2 - yy2);
        let b6 = SH_C2_3 * xz2;
        let b7 = SH_C2_4 * (xx2 - yy2);
        cR = cR + b3 * splats[shBase + 3u] + b4 * splats[shBase + 4u] + b5 * splats[shBase + 5u] + b6 * splats[shBase + 6u] + b7 * splats[shBase + 7u];
        cG = cG + b3 * splats[shBase + n + 3u] + b4 * splats[shBase + n + 4u] + b5 * splats[shBase + n + 5u] + b6 * splats[shBase + n + 6u] + b7 * splats[shBase + n + 7u];
        cB = cB + b3 * splats[shBase + 2u * n + 3u] + b4 * splats[shBase + 2u * n + 4u] + b5 * splats[shBase + 2u * n + 5u] + b6 * splats[shBase + 2u * n + 6u] + b7 * splats[shBase + 2u * n + 7u];
    }` : ''}
    ${coeffsPerChannel >= 15 ? `
    {
        let n = COEFFS_PER_CHANNEL;
        let shBase = base + 14u;
        let xx2 = dirX * dirX;
        let yy2 = dirY * dirY;
        let zz2 = dirZ * dirZ;
        let xy2 = dirX * dirY;
        let b8 = SH_C3_0 * dirY * (3.0 * xx2 - yy2);
        let b9 = SH_C3_1 * xy2 * dirZ;
        let b10 = SH_C3_2 * dirY * (4.0 * zz2 - xx2 - yy2);
        let b11 = SH_C3_3 * dirZ * (2.0 * zz2 - 3.0 * xx2 - 3.0 * yy2);
        let b12 = SH_C3_4 * dirX * (4.0 * zz2 - xx2 - yy2);
        let b13 = SH_C3_5 * dirZ * (xx2 - yy2);
        let b14 = SH_C3_6 * dirX * (xx2 - 3.0 * yy2);
        cR = cR + b8 * splats[shBase + 8u] + b9 * splats[shBase + 9u] + b10 * splats[shBase + 10u] + b11 * splats[shBase + 11u] + b12 * splats[shBase + 12u] + b13 * splats[shBase + 13u] + b14 * splats[shBase + 14u];
        cG = cG + b8 * splats[shBase + n + 8u] + b9 * splats[shBase + n + 9u] + b10 * splats[shBase + n + 10u] + b11 * splats[shBase + n + 11u] + b12 * splats[shBase + n + 12u] + b13 * splats[shBase + n + 13u] + b14 * splats[shBase + n + 14u];
        cB = cB + b8 * splats[shBase + 2u * n + 8u] + b9 * splats[shBase + 2u * n + 9u] + b10 * splats[shBase + 2u * n + 10u] + b11 * splats[shBase + 2u * n + 11u] + b12 * splats[shBase + 2u * n + 12u] + b13 * splats[shBase + 2u * n + 13u] + b14 * splats[shBase + 2u * n + 14u];
    }` : ''}

    let colR = max(0.0, cR + 0.5);
    let colG = max(0.0, cG + 0.5);
    let colB = max(0.0, cB + 0.5);

    let alpha = (1.0 / (1.0 + exp(-opacity))) * radiusFade;

    projected[i * 3u + 0u] = vec4<f32>(screenX, screenY, radius, 0.0);
    projected[i * 3u + 1u] = vec4<f32>(covInvA, covInvB, covInvC, alpha);
    projected[i * 3u + 2u] = vec4<f32>(colR, colG, colB, 0.0);

    // Per-splat tile-coverage count, clamped at maxCoveragePerSplat.
    // Tile indices are GROUP-LOCAL (= image-tile-index minus the
    // group's origin in tiles), so values cover [0, groupTilesX-1] ×
    // [0, groupTilesY-1] for every group regardless of its position in
    // the image. Splats outside the group's pixel rectangle were
    // already culled by the AABB check above.
    let tsz: f32 = ${wgslF32(TILE_SIZE)};
    let gox = f32(uniforms.groupPixelOriginX);
    let goy = f32(uniforms.groupPixelOriginY);
${projection === 'pinhole' ? /* wgsl */`
    let minTX = max(0, i32(floor((screenX - radius - gox) / tsz)));
    let maxTX = min(i32(uniforms.groupTilesX) - 1, i32(floor((screenX + radius - gox) / tsz)));
    let minTY = max(0, i32(floor((screenY - radius - goy) / tsz)));
    let maxTY = min(i32(uniforms.groupTilesY) - 1, i32(floor((screenY + radius - goy) / tsz)));
    if (maxTX < minTX || maxTY < minTY) {
        coverage[i] = 0u;
    } else {
        let raw = u32(maxTX - minTX + 1) * u32(maxTY - minTY + 1);
        coverage[i] = min(raw, ${maxCoveragePerSplat}u);
    }
` : /* wgsl */`
    // Equirect: the X tile range can extend past the image edges into
    // negative or > groupTilesX-1 indices — those represent the same
    // splat seen across the ±π longitude seam. Coverage is the raw
    // span (capped at groupTilesX so a splat with radius > image_width
    // doesn't emit duplicate tile keys); emit-pairs walks [minTXraw ..
    // maxTXraw] in lock-step and applies a modular wrap when writing
    // tile keys. Y is clamped normally — equirect doesn't wrap across
    // poles.
    let minTXraw = i32(floor((screenX - radius - gox) / tsz));
    let maxTXraw = i32(floor((screenX + radius - gox) / tsz));
    let txCountRaw = maxTXraw - minTXraw + 1;
    let txCount = min(txCountRaw, i32(uniforms.groupTilesX));
    let minTY = max(0, i32(floor((screenY - radius - goy) / tsz)));
    let maxTY = min(i32(uniforms.groupTilesY) - 1, i32(floor((screenY + radius - goy) / tsz)));
    if (txCount <= 0 || maxTY < minTY) {
        coverage[i] = 0u;
    } else {
        let raw = u32(txCount) * u32(maxTY - minTY + 1);
        coverage[i] = min(raw, ${maxCoveragePerSplat}u);
    }
`}
}
`;

/**
 * Tile-bin emit-pairs shader. For each projected splat, emits
 * `coverage[i]` (tile, splat) pairs into two parallel buffers,
 * starting at `emitOffset[i]`:
 *
 *   tileKeys [emitOffset[i] + j] = tileIdx
 *   splatValues[emitOffset[i] + j] = splatIdx (= i)
 *
 * The orchestrator sizes maxCoveragePerSplat to cover a sub-frame's
 * entire tile area, so a splat's `coverage[i]` always equals its full
 * bbox-in-group tile count — no truncation, no seams. The walk emits
 * row-major over the bbox-in-group. A subsequent key+value radix sort
 * groups pairs by tile; within each tile, the splatIdx-as-value sort
 * preserves the chunk's depth order (splatIdx is monotonic in depth
 * from the CPU pre-sort).
 *
 * @param projection - Camera projection. `pinhole` walks the clamped
 * tile bbox directly; `equirect` walks the un-clamped X range and
 * applies a modular wrap, so a splat near the ±π longitude seam emits
 * tile keys on both sides of the image.
 * @returns WGSL source for the emit-pairs compute shader.
 */
const tileBinEmitPairsWgsl = (projection: Projection) => /* wgsl */`
struct Uniforms {
    rightX: f32, rightY: f32, rightZ: f32, _p0: f32,
    downX: f32, downY: f32, downZ: f32, _p1: f32,
    forwardX: f32, forwardY: f32, forwardZ: f32, _p2: f32,
    eyeX: f32, eyeY: f32, eyeZ: f32, _p3: f32,
    focalX: f32, focalY: f32, near: f32, _p4: f32,
    imageWidth: u32, imageHeight: u32, splatStride: u32, chunkSize: u32,
    groupPixelMinX: u32, groupPixelMinY: u32, groupPixelMaxX: u32, groupPixelMaxY: u32,
    groupTilesX: u32, groupTilesY: u32, groupPixelOriginX: u32, groupPixelOriginY: u32,
    bgR: f32, bgG: f32, bgB: f32, bgA: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> projected: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> emitOffset: array<u32>;
@group(0) @binding(3) var<storage, read> coverage: array<u32>;
@group(0) @binding(4) var<storage, read_write> tileKeys: array<u32>;
@group(0) @binding(5) var<storage, read_write> splatValues: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= uniforms.chunkSize) { return; }
    let cap = coverage[i];
    if (cap == 0u) { return; }
    let v0 = projected[i * 3u + 0u];
    let radius = v0.z;
    if (radius <= 0.0) { return; }
    let sX = v0.x;
    let sY = v0.y;
    let tsz: f32 = ${wgslF32(TILE_SIZE)};
    // Group-local tile indices (see project shader for rationale).
    let gox = f32(uniforms.groupPixelOriginX);
    let goy = f32(uniforms.groupPixelOriginY);
${projection === 'pinhole' ? /* wgsl */`
    let minTX = max(0, i32(floor((sX - radius - gox) / tsz)));
    let maxTX = min(i32(uniforms.groupTilesX) - 1, i32(floor((sX + radius - gox) / tsz)));
    let minTY = max(0, i32(floor((sY - radius - goy) / tsz)));
    let maxTY = min(i32(uniforms.groupTilesY) - 1, i32(floor((sY + radius - goy) / tsz)));
    if (maxTX < minTX || maxTY < minTY) { return; }

    var slot = emitOffset[i];
    let end = slot + cap;
    for (var ty: i32 = minTY; ty <= maxTY; ty = ty + 1) {
        if (slot >= end) { break; }
        for (var tx: i32 = minTX; tx <= maxTX; tx = tx + 1) {
            if (slot >= end) { break; }
            let t = u32(ty) * uniforms.groupTilesX + u32(tx);
            tileKeys[slot] = t;
            splatValues[slot] = i;
            slot = slot + 1u;
        }
    }
` : /* wgsl */`
    // Equirect: raw X range (possibly wrapping past the seam) — must
    // match the project shader's coverage computation exactly. Each
    // emitted tx is wrapped into [0, groupTilesX-1] via modular
    // arithmetic. The rasterize-binned shader compensates by wrapping
    // its per-pixel dx into [-W/2, W/2], so a wrapped tile pulls the
    // splat's footprint from the correct copy across the seam.
    let minTXraw = i32(floor((sX - radius - gox) / tsz));
    let maxTXraw = i32(floor((sX + radius - gox) / tsz));
    let txCountRaw = maxTXraw - minTXraw + 1;
    let groupTilesX_i = i32(uniforms.groupTilesX);
    let txCount = min(txCountRaw, groupTilesX_i);
    let minTY = max(0, i32(floor((sY - radius - goy) / tsz)));
    let maxTY = min(i32(uniforms.groupTilesY) - 1, i32(floor((sY + radius - goy) / tsz)));
    if (txCount <= 0 || maxTY < minTY) { return; }

    var slot = emitOffset[i];
    let end = slot + cap;
    for (var ty: i32 = minTY; ty <= maxTY; ty = ty + 1) {
        if (slot >= end) { break; }
        for (var k: i32 = 0; k < txCount; k = k + 1) {
            if (slot >= end) { break; }
            var tx = (minTXraw + k) % groupTilesX_i;
            if (tx < 0) { tx = tx + groupTilesX_i; }
            let t = u32(ty) * uniforms.groupTilesX + u32(tx);
            tileKeys[slot] = t;
            splatValues[slot] = i;
            slot = slot + 1u;
        }
    }
`}
}
`;

/**
 * Single-workgroup exclusive prefix-sum of the project shader's
 * per-splat `coverage[]` into `emitOffset[]`. Also writes the total
 * pair count into `totalPairs[0]` so downstream kernels and the radix
 * sort can size their dispatches without a CPU round trip.
 *
 * Layout: 256 threads, each processes `${SCAN_PER_THREAD}` elements
 * serially (chosen so that 256 × SCAN_PER_THREAD ≥ chunkCap). Phase 1
 * computes a per-thread partial sum; phase 2 has thread 0 scan the 256
 * partials in shared memory (negligible vs the 800-element block work);
 * phase 3 each thread re-walks its block writing the exclusive prefix.
 *
 * @param scanPerThread - Per-thread element budget; must satisfy `256 * scanPerThread >= chunkCap`.
 * @returns WGSL source for the prefix-sum compute shader.
 */
const prefixSumWgsl = (scanPerThread: number) => /* wgsl */`
struct Uniforms {
    rightX: f32, rightY: f32, rightZ: f32, _p0: f32,
    downX: f32, downY: f32, downZ: f32, _p1: f32,
    forwardX: f32, forwardY: f32, forwardZ: f32, _p2: f32,
    eyeX: f32, eyeY: f32, eyeZ: f32, _p3: f32,
    focalX: f32, focalY: f32, near: f32, _p4: f32,
    imageWidth: u32, imageHeight: u32, splatStride: u32, chunkSize: u32,
    groupPixelMinX: u32, groupPixelMinY: u32, groupPixelMaxX: u32, groupPixelMaxY: u32,
    groupTilesX: u32, groupTilesY: u32, groupPixelOriginX: u32, groupPixelOriginY: u32,
    bgR: f32, bgG: f32, bgB: f32, bgA: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> coverage: array<u32>;
@group(0) @binding(2) var<storage, read_write> emitOffset: array<u32>;
@group(0) @binding(3) var<storage, read_write> totalPairs: array<u32>;

const SCAN_THREADS: u32 = 256u;
const SCAN_PER_THREAD: u32 = ${scanPerThread}u;

var<workgroup> scratch: array<u32, SCAN_THREADS>;

@compute @workgroup_size(SCAN_THREADS)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n = uniforms.chunkSize;
    let base = tid * SCAN_PER_THREAD;

    var partial: u32 = 0u;
    for (var i: u32 = 0u; i < SCAN_PER_THREAD; i = i + 1u) {
        let idx = base + i;
        if (idx < n) {
            partial = partial + coverage[idx];
        }
    }
    scratch[tid] = partial;
    workgroupBarrier();

    if (tid == 0u) {
        var acc: u32 = 0u;
        for (var i: u32 = 0u; i < SCAN_THREADS; i = i + 1u) {
            let v = scratch[i];
            scratch[i] = acc;
            acc = acc + v;
        }
        totalPairs[0] = acc;
    }
    workgroupBarrier();

    var prefix: u32 = scratch[tid];
    for (var i: u32 = 0u; i < SCAN_PER_THREAD; i = i + 1u) {
        let idx = base + i;
        if (idx < n) {
            emitOffset[idx] = prefix;
            prefix = prefix + coverage[idx];
        }
    }
}
`;

/**
 * Prepares indirect-dispatch arguments by reading `totalPairs[0]` and
 * writing workgroup counts into two slots of the device's
 * `indirectDispatchBuffer` (each slot is 3 × u32 = `(x, y, z)`):
 *
 *   - `sortSlot`: workgroup count for `ComputeRadixSort.sortIndirect`,
 *     computed as `ceil(totalPairs / 2048)` (matches the radix sort's
 *     16×16 thread × 8 elements / thread = 2048 elements/workgroup).
 *   - `boundariesSlot`: workgroup count for `findBoundaries`, computed
 *     as `ceil(totalPairs / 64)`.
 *
 * Slot byte offsets are passed in via two u32 uniforms in a small ad-hoc
 * uniform block (NOT the shared `Uniforms` struct, because the slot
 * indices vary per chunk while the shared uniforms are set per group).
 *
 * @returns WGSL source for the prepare-indirect compute shader.
 */
const prepareIndirectWgsl = () => /* wgsl */`
struct PrepareUniforms {
    sortSlotBase: u32,
    boundariesSlotBase: u32,
}

@group(0) @binding(0) var<uniform> uniforms: PrepareUniforms;
@group(0) @binding(1) var<storage, read> totalPairs: array<u32>;
@group(0) @binding(2) var<storage, read_write> indirectBuffer: array<u32>;

const SORT_ELEMENTS_PER_WG: u32 = 2048u;
const BOUNDARIES_THREADS_PER_WG: u32 = 64u;
// WebGPU spec minimum for maxComputeWorkgroupsPerDimension. Any larger
// 1-D dispatch must be tiled into 2-D so both axes stay <= this bound.
// The consumer shaders linearise via WORKGROUP_ID = w_id.x + w_id.y * w_dim.x.
const MAX_DIM: u32 = 65535u;

fn splitWg(count: u32) -> vec2<u32> {
    if (count <= MAX_DIM) {
        return vec2<u32>(count, 1u);
    }
    let y = (count + MAX_DIM - 1u) / MAX_DIM;
    let x = (count + y - 1u) / y;
    return vec2<u32>(x, y);
}

@compute @workgroup_size(1)
fn main() {
    let n = totalPairs[0];
    let sortWg = (n + SORT_ELEMENTS_PER_WG - 1u) / SORT_ELEMENTS_PER_WG;
    let bndWg = (n + BOUNDARIES_THREADS_PER_WG - 1u) / BOUNDARIES_THREADS_PER_WG;
    let sortDim = splitWg(sortWg);
    let bndDim = splitWg(bndWg);
    let s = uniforms.sortSlotBase;
    let b = uniforms.boundariesSlotBase;
    indirectBuffer[s + 0u] = sortDim.x;
    indirectBuffer[s + 1u] = sortDim.y;
    indirectBuffer[s + 2u] = 1u;
    indirectBuffer[b + 0u] = bndDim.x;
    indirectBuffer[b + 1u] = bndDim.y;
    indirectBuffer[b + 2u] = 1u;
}
`;

/**
 * Initialises `tileOffsets[0 .. numTiles]` to the sentinel value
 * `totalPairs[0]` (= past-the-end). `findBoundaries` then atomicMin's
 * the actual first-pair-index for every non-empty tile; tiles with no
 * pairs keep the sentinel, which collapses to a zero-length slice when
 * the rasterize-binned shader reads `tileOffsets[T] .. tileOffsets[T+1]`.
 *
 * @returns WGSL source for the init-tile-offsets compute shader.
 */
const initTileOffsetsWgsl = () => /* wgsl */`
struct Uniforms {
    rightX: f32, rightY: f32, rightZ: f32, _p0: f32,
    downX: f32, downY: f32, downZ: f32, _p1: f32,
    forwardX: f32, forwardY: f32, forwardZ: f32, _p2: f32,
    eyeX: f32, eyeY: f32, eyeZ: f32, _p3: f32,
    focalX: f32, focalY: f32, near: f32, _p4: f32,
    imageWidth: u32, imageHeight: u32, splatStride: u32, chunkSize: u32,
    groupPixelMinX: u32, groupPixelMinY: u32, groupPixelMaxX: u32, groupPixelMaxY: u32,
    groupTilesX: u32, groupTilesY: u32, groupPixelOriginX: u32, groupPixelOriginY: u32,
    bgR: f32, bgG: f32, bgB: f32, bgA: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> totalPairs: array<u32>;
@group(0) @binding(2) var<storage, read_write> tileOffsets: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let numTiles = uniforms.groupTilesX * uniforms.groupTilesY;
    if (i > numTiles) { return; }
    tileOffsets[i] = totalPairs[0];
}
`;

/**
 * For every adjacent pair of sorted keys where the high-bit tile index
 * differs, atomicMin's the current position into `tileOffsets[t]` for
 * every tile `t` in `(prevTile, curTile]`. Combined with the sentinel
 * init this gives `tileOffsets[T]` = first index in `sortedKeys` whose
 * tile bits equal T (or the sentinel if T is empty).
 *
 * Dispatched indirectly with workgroup count `ceil(totalPairs / 64)` so
 * that we don't waste invocations on the unused tail of the pairs
 * buffer.
 *
 * @returns WGSL source for the find-boundaries compute shader.
 */
const findBoundariesWgsl = () => /* wgsl */`
struct Uniforms {
    rightX: f32, rightY: f32, rightZ: f32, _p0: f32,
    downX: f32, downY: f32, downZ: f32, _p1: f32,
    forwardX: f32, forwardY: f32, forwardZ: f32, _p2: f32,
    eyeX: f32, eyeY: f32, eyeZ: f32, _p3: f32,
    focalX: f32, focalY: f32, near: f32, _p4: f32,
    imageWidth: u32, imageHeight: u32, splatStride: u32, chunkSize: u32,
    groupPixelMinX: u32, groupPixelMinY: u32, groupPixelMaxX: u32, groupPixelMaxY: u32,
    groupTilesX: u32, groupTilesY: u32, groupPixelOriginX: u32, groupPixelOriginY: u32,
    bgR: f32, bgG: f32, bgB: f32, bgA: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> totalPairs: array<u32>;
@group(0) @binding(2) var<storage, read> sortedTileKeys: array<u32>;
@group(0) @binding(3) var<storage, read_write> tileOffsets: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wgId: vec3<u32>,
    @builtin(num_workgroups) numWg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    // 2-D dispatch (prepare-indirect splits to stay under the 65535
    // per-axis workgroup-count limit); linearise here.
    let linearWg = wgId.x + wgId.y * numWg.x;
    let i = linearWg * 64u + lid.x;
    let n = totalPairs[0];
    if (i >= n) { return; }

    // Reference uniforms once so the binding isn't dead-code-stripped
    // (keeps the BG format in sync with the shader expectations).
    let _u = uniforms.groupTilesX;

    let curTile = sortedTileKeys[i];
    // Sentinel for "no previous tile" — overflow makes prevTile+1 = 0u
    // so the for loop below cleanly handles the i = 0 case.
    let prevTileBits = select(0xFFFFFFFFu, sortedTileKeys[i - 1u], i > 0u);
    if (curTile == prevTileBits) { return; }
    for (var t: u32 = prevTileBits + 1u; t <= curTile; t = t + 1u) {
        atomicMin(&tileOffsets[t], i);
    }
}
`;

/**
 * Binned rasterize shader. Each workgroup handles one tile and only walks
 * the splats that have been pre-binned into it (tile-bin pre-pass on CPU
 * or GPU). Replaces the "walk all splats per pixel" loop in
 * `rasterizeWgsl` with "walk this tile's slice", which is the asymptotic
 * fix for performance at high splat counts.
 *
 * The slice is stored in two buffers:
 *   - `tileOffsets[T + 1]` — exclusive prefix sum: tile T's slice is
 *     `tileData[tileOffsets[T] .. tileOffsets[T + 1])`.
 *   - `tileData[]` — splat indices, grouped by tile, depth-sorted within
 *     each tile (the orchestrator's CPU pre-sort + stable per-splat
 *     binning produces this layout for free).
 *
 * The shader's bindings mirror `rasterizeWgsl` and add bindings 3, 4 for
 * the tile lists.
 *
 * @param projection - Camera projection. `equirect` wraps the per-pixel
 * `dx = px - splat.x` into `[-W/2, W/2]` so a tile on the opposite side
 * of the ±π longitude seam evaluates against the splat's nearer copy;
 * `pinhole` uses the raw delta.
 * @returns WGSL source for the binned-rasterize compute shader.
 */
const rasterizeBinnedWgsl = (projection: Projection) => /* wgsl */`
struct Uniforms {
    rightX: f32, rightY: f32, rightZ: f32, _p0: f32,
    downX: f32, downY: f32, downZ: f32, _p1: f32,
    forwardX: f32, forwardY: f32, forwardZ: f32, _p2: f32,
    eyeX: f32, eyeY: f32, eyeZ: f32, _p3: f32,
    focalX: f32, focalY: f32, near: f32, _p4: f32,
    imageWidth: u32, imageHeight: u32, splatStride: u32, chunkSize: u32,
    groupPixelMinX: u32, groupPixelMinY: u32, groupPixelMaxX: u32, groupPixelMaxY: u32,
    groupTilesX: u32, groupTilesY: u32, groupPixelOriginX: u32, groupPixelOriginY: u32,
    bgR: f32, bgG: f32, bgB: f32, bgA: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> projected: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> runningState: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> tileOffsets: array<u32>;
@group(0) @binding(4) var<storage, read> sortedSplatIndices: array<u32>;

@compute @workgroup_size(${TILE_SIZE}, ${TILE_SIZE}, 1)
fn main(
    @builtin(workgroup_id) wgId: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    if (wgId.x >= uniforms.groupTilesX || wgId.y >= uniforms.groupTilesY) { return; }

    let tileIdx = wgId.y * uniforms.groupTilesX + wgId.x;
    let sliceStart = tileOffsets[tileIdx];
    let sliceEnd = tileOffsets[tileIdx + 1u];

    let localPixelX = wgId.x * ${TILE_SIZE}u + lid.x;
    let localPixelY = wgId.y * ${TILE_SIZE}u + lid.y;
    let groupPixelW = uniforms.groupTilesX * ${TILE_SIZE}u;

    let imagePixelX = uniforms.groupPixelOriginX + localPixelX;
    let imagePixelY = uniforms.groupPixelOriginY + localPixelY;
    if (imagePixelX >= uniforms.imageWidth || imagePixelY >= uniforms.imageHeight) { return; }

    let pixelIdx = localPixelY * groupPixelW + localPixelX;
    var state = runningState[pixelIdx];
    var color = state.rgb;
    var T = state.a;

    if (T < ${wgslF32(MIN_TRANSMITTANCE)}) { return; }

    let px = f32(imagePixelX) + 0.5;
    let py = f32(imagePixelY) + 0.5;

${projection === 'equirect' ? /* wgsl */`
    let imgWf2 = f32(uniforms.imageWidth);
    let halfImgW = imgWf2 * 0.5;
` : ''}
    for (var i: u32 = sliceStart; i < sliceEnd; i = i + 1u) {
        if (T < ${wgslF32(MIN_TRANSMITTANCE)}) { break; }
        let splatIdx = sortedSplatIndices[i];
        let v0 = projected[splatIdx * 3u + 0u];
${projection === 'pinhole' ? /* wgsl */`
        let dx = px - v0.x;
` : /* wgsl */`
        // Equirect: a splat near the ±π longitude seam is tile-binned on
        // both sides of the image. Wrap dx into [-W/2, W/2] so a tile on
        // the opposite side of the seam pulls the splat's footprint from
        // the correct (nearer) copy.
        var dx = px - v0.x;
        if (dx > halfImgW) { dx = dx - imgWf2; }
        else if (dx < -halfImgW) { dx = dx + imgWf2; }
`}
        let dy = py - v0.y;
        let r = v0.z;
        if (r <= 0.0 || abs(dx) > r || abs(dy) > r) { continue; }
        let v1 = projected[splatIdx * 3u + 1u];
        let power = -0.5 * (v1.x * dx * dx + 2.0 * v1.y * dx * dy + v1.z * dy * dy);
        if (power > 0.0) { continue; }
        let alpha = min(${wgslF32(OPACITY_CAP)}, v1.w * exp(power));
        if (alpha < ${wgslF32(MIN_ALPHA)}) { continue; }
        let weight = T * alpha;
        let v2 = projected[splatIdx * 3u + 2u];
        color = color + weight * v2.rgb;
        T = T * (1.0 - alpha);
    }

    runningState[pixelIdx] = vec4<f32>(color, T);
}
`;

const finalizeWgsl = () => /* wgsl */`
struct Uniforms {
    rightX: f32, rightY: f32, rightZ: f32, _p0: f32,
    downX: f32, downY: f32, downZ: f32, _p1: f32,
    forwardX: f32, forwardY: f32, forwardZ: f32, _p2: f32,
    eyeX: f32, eyeY: f32, eyeZ: f32, _p3: f32,
    focalX: f32, focalY: f32, near: f32, _p4: f32,
    imageWidth: u32, imageHeight: u32, splatStride: u32, chunkSize: u32,
    groupPixelMinX: u32, groupPixelMinY: u32, groupPixelMaxX: u32, groupPixelMaxY: u32,
    groupTilesX: u32, groupTilesY: u32, groupPixelOriginX: u32, groupPixelOriginY: u32,
    bgR: f32, bgG: f32, bgB: f32, bgA: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> runningState: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(${TILE_SIZE}, ${TILE_SIZE}, 1)
fn main(
    @builtin(workgroup_id) wgId: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    if (wgId.x >= uniforms.groupTilesX || wgId.y >= uniforms.groupTilesY) { return; }

    let localPixelX = wgId.x * ${TILE_SIZE}u + lid.x;
    let localPixelY = wgId.y * ${TILE_SIZE}u + lid.y;
    let groupPixelW = uniforms.groupTilesX * ${TILE_SIZE}u;

    let pixelIdx = localPixelY * groupPixelW + localPixelX;
    let state = runningState[pixelIdx];

    let color = state.rgb + state.a * vec3<f32>(uniforms.bgR, uniforms.bgG, uniforms.bgB);
    let alphaOut = (1.0 - state.a) + state.a * uniforms.bgA;

    let r = u32(clamp(color.r, 0.0, 1.0) * 255.0 + 0.5);
    let g = u32(clamp(color.g, 0.0, 1.0) * 255.0 + 0.5);
    let bch = u32(clamp(color.b, 0.0, 1.0) * 255.0 + 0.5);
    let aOut = u32(clamp(alphaOut, 0.0, 1.0) * 255.0 + 0.5);

    output[pixelIdx] = r | (g << 8u) | (bch << 16u) | (aOut << 24u);
}
`;

/**
 * Build the UniformFormat entries that match the WGSL uniform block above.
 * All three shaders share the same uniform layout so a single
 * UniformBufferFormat description suffices (we instantiate it per shader).
 *
 * @returns Array of UniformFormat entries in declaration order.
 */
const uniformFormatEntries = (): UniformFormat[] => [
    new UniformFormat('rightX', UNIFORMTYPE_FLOAT),
    new UniformFormat('rightY', UNIFORMTYPE_FLOAT),
    new UniformFormat('rightZ', UNIFORMTYPE_FLOAT),
    new UniformFormat('_p0', UNIFORMTYPE_FLOAT),
    new UniformFormat('downX', UNIFORMTYPE_FLOAT),
    new UniformFormat('downY', UNIFORMTYPE_FLOAT),
    new UniformFormat('downZ', UNIFORMTYPE_FLOAT),
    new UniformFormat('_p1', UNIFORMTYPE_FLOAT),
    new UniformFormat('forwardX', UNIFORMTYPE_FLOAT),
    new UniformFormat('forwardY', UNIFORMTYPE_FLOAT),
    new UniformFormat('forwardZ', UNIFORMTYPE_FLOAT),
    new UniformFormat('_p2', UNIFORMTYPE_FLOAT),
    new UniformFormat('eyeX', UNIFORMTYPE_FLOAT),
    new UniformFormat('eyeY', UNIFORMTYPE_FLOAT),
    new UniformFormat('eyeZ', UNIFORMTYPE_FLOAT),
    new UniformFormat('_p3', UNIFORMTYPE_FLOAT),
    new UniformFormat('focalX', UNIFORMTYPE_FLOAT),
    new UniformFormat('focalY', UNIFORMTYPE_FLOAT),
    new UniformFormat('near', UNIFORMTYPE_FLOAT),
    new UniformFormat('_p4', UNIFORMTYPE_FLOAT),
    new UniformFormat('imageWidth', UNIFORMTYPE_UINT),
    new UniformFormat('imageHeight', UNIFORMTYPE_UINT),
    new UniformFormat('splatStride', UNIFORMTYPE_UINT),
    new UniformFormat('chunkSize', UNIFORMTYPE_UINT),
    new UniformFormat('groupPixelMinX', UNIFORMTYPE_UINT),
    new UniformFormat('groupPixelMinY', UNIFORMTYPE_UINT),
    new UniformFormat('groupPixelMaxX', UNIFORMTYPE_UINT),
    new UniformFormat('groupPixelMaxY', UNIFORMTYPE_UINT),
    new UniformFormat('groupTilesX', UNIFORMTYPE_UINT),
    new UniformFormat('groupTilesY', UNIFORMTYPE_UINT),
    new UniformFormat('groupPixelOriginX', UNIFORMTYPE_UINT),
    new UniformFormat('groupPixelOriginY', UNIFORMTYPE_UINT),
    new UniformFormat('bgR', UNIFORMTYPE_FLOAT),
    new UniformFormat('bgG', UNIFORMTYPE_FLOAT),
    new UniformFormat('bgB', UNIFORMTYPE_FLOAT),
    new UniformFormat('bgA', UNIFORMTYPE_FLOAT)
];

interface PipelineBuffers {
    inputBuffer: StorageBuffer;
    projBuffer: StorageBuffer;
    runningStateBuffer: StorageBuffer;
    outputBuffer: StorageBuffer;
    /** Per-tile offset table for binned rasterize: `(numTiles + 1) × u32`. */
    tileOffsetsBuffer: StorageBuffer;
    /**
     * Per-splat tile-coverage count, written by the project shader.
     * Consumed by the GPU prefix-sum kernel; never read by the CPU.
     */
    coverageBuffer: StorageBuffer;
    /**
     * GPU-computed exclusive prefix-sum of `coverageBuffer`.
     * `emitOffset[i]` is the first slot in `tileKeysBuffer`/`splatValuesBuffer`
     * that splat i writes to in the emit-pairs pass.
     */
    emitOffsetBuffer: StorageBuffer;
    /**
     * Single u32 holding the total pair count produced by the prefix-sum.
     * Read by `prepareIndirect`, `initTileOffsets`, `findBoundaries`,
     * and the radix sort's indirect dispatch. Never touched by the CPU.
     */
    totalPairsBuffer: StorageBuffer;
    /**
     * `tileIdx` (sort key) for each (tile, splat) pair. Sorted in place
     * by `ComputeRadixSort.sortIndirect` — afterwards `sortedKeys` on the
     * radix sort instance holds the sorted tile keys.
     */
    tileKeysBuffer: StorageBuffer;
    /**
     * `splatIdx` (sort value) for each (tile, splat) pair, passed to the
     * radix sort as `initialValues`. The sort writes the reordered splat
     * indices to its internal `sortedIndices` buffer.
     */
    splatValuesBuffer: StorageBuffer;
    projectCompute: Compute;
    prefixSumCompute: Compute;
    emitPairsCompute: Compute;
    prepareIndirectCompute: Compute;
    initTileOffsetsCompute: Compute;
    findBoundariesCompute: Compute;
    rasterizeBinnedCompute: Compute;
    finalizeCompute: Compute;
}


/**
 * GPU-accelerated splat rasterizer.
 *
 * Owns eight compute shaders — project, prefix-sum, emit-pairs,
 * prepare-indirect, init-tile-offsets, find-boundaries, rasterize-binned,
 * finalize-pack — a shared `ComputeRadixSort` (used in indirect mode,
 * key + value), and GPU buffers. The per-chunk pipeline is fully
 * GPU-resident: the caller never reads back coverage, sorted keys, or
 * tile offsets.
 *
 * Per-render flow:
 *   1. `beginGroup(...)` — clears the running state and sets uniforms
 *      for this group (covers the whole image for a single-pass render).
 *   2. For each chunk of depth-sorted splats: `dispatchChunk(data,
 *      chunkSize)` runs the whole tile-bin + rasterize pipeline in one
 *      submission — project + coverage → prefix-sum (writes emitOffsets
 *      + totalPairs) → emit-pairs → prepare-indirect → radix sortIndirect
 *      → init-tile-offsets → find-boundaries → rasterize-binned. No
 *      readbacks; one `submit()` per chunk to capture each compute's
 *      uniform state before the next chunk overwrites it.
 *   3. `finishGroup()` — dispatches finalize-pack and starts an async
 *      readback. Returns a `Promise<Uint8Array>` resolved when the GPU has
 *      finished writing this group's RGBA bytes.
 */
class GpuSplatRasterizer {
    private device: GraphicsDevice;
    private options: SplatRasterizerOptions;
    private projectShader: Shader;
    private prefixSumShader: Shader;
    private emitPairsShader: Shader;
    private prepareIndirectShader: Shader;
    private initTileOffsetsShader: Shader;
    private findBoundariesShader: Shader;
    private rasterizeBinnedShader: Shader;
    private finalizeShader: Shader;
    private projectBgFormat: BindGroupFormat;
    private prefixSumBgFormat: BindGroupFormat;
    private emitPairsBgFormat: BindGroupFormat;
    private prepareIndirectBgFormat: BindGroupFormat;
    private initTileOffsetsBgFormat: BindGroupFormat;
    private findBoundariesBgFormat: BindGroupFormat;
    private rasterizeBinnedBgFormat: BindGroupFormat;
    private finalizeBgFormat: BindGroupFormat;
    private buffers: PipelineBuffers;
    /**
     * Single shared `ComputeRadixSort` for the GPU tile-bin pipeline.
     * Used in key+value mode: tile-index keys + splat-index values.
     */
    private radixSort: ComputeRadixSort;
    /** sortIndirect numBits, derived from numTiles (multiple of 4). */
    private sortKeyBits: number;
    private clearStatePattern: Float32Array;
    /** Active group's tile dimensions, set by `beginGroup`. */
    private activeTilesX: number = 0;
    private activeTilesY: number = 0;

    /** Floats per gaussian in the input buffer (depends on SH band count). */
    readonly inputStride: number;
    /** Group tile dimensions (X). */
    readonly groupTilesX: number;
    /** Group tile dimensions (Y). */
    readonly groupTilesY: number;
    /** Max gaussians per chunk. */
    readonly chunkCap: number;
    /** Pixels per group axis (X). */
    readonly groupPixelW: number;
    /** Pixels per group axis (Y). */
    readonly groupPixelH: number;

    constructor(device: GraphicsDevice, options: SplatRasterizerOptions) {
        this.device = device;
        this.options = options;
        this.groupTilesX = options.groupTilesX;
        this.groupTilesY = options.groupTilesY;
        this.chunkCap = options.chunkCap;
        this.groupPixelW = options.groupTilesX * TILE_SIZE;
        this.groupPixelH = options.groupTilesY * TILE_SIZE;

        const numTiles = options.groupTilesX * options.groupTilesY;
        // Round up to multiple of 4 (radix sort uses 4-bit passes). Min 4
        // because ComputeRadixSort requires numBits ≥ 4. Max 32 (u32 key).
        const tileBits = Math.max(4, Math.min(32, Math.ceil(Math.log2(Math.max(2, numTiles)) / 4) * 4));
        this.sortKeyBits = tileBits;

        const coeffs = numSHCoeffsPerChannel(options.numSHBands);
        this.inputStride = 14 + 3 * coeffs;

        this.projectBgFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('splats', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('projected', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('coverage', SHADERSTAGE_COMPUTE)
        ]);
        this.prefixSumBgFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('coverage', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('emitOffset', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('totalPairs', SHADERSTAGE_COMPUTE)
        ]);
        this.emitPairsBgFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('projected', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('emitOffset', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('coverage', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('tileKeys', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('splatValues', SHADERSTAGE_COMPUTE)
        ]);
        this.prepareIndirectBgFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('totalPairs', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('indirectBuffer', SHADERSTAGE_COMPUTE)
        ]);
        this.initTileOffsetsBgFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('totalPairs', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('tileOffsets', SHADERSTAGE_COMPUTE)
        ]);
        this.findBoundariesBgFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('totalPairs', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('sortedTileKeys', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('tileOffsets', SHADERSTAGE_COMPUTE)
        ]);
        this.rasterizeBinnedBgFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('projected', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('runningState', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('tileOffsets', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('sortedSplatIndices', SHADERSTAGE_COMPUTE, true)
        ]);
        this.finalizeBgFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('runningState', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('output', SHADERSTAGE_COMPUTE)
        ]);

        const mkShader = (
            name: string,
            source: string,
            bgFormat: BindGroupFormat,
            uniformEntries: UniformFormat[] = uniformFormatEntries()
        ) => new Shader(device, {
            name,
            shaderLanguage: SHADERLANGUAGE_WGSL,
            cshader: source,
            // @ts-ignore - computeUniformBufferFormats / computeBindGroupFormat are not in public Shader types.
            computeUniformBufferFormats: {
                uniforms: new UniformBufferFormat(device, uniformEntries)
            },
            // @ts-ignore
            computeBindGroupFormat: bgFormat
        });

        // The prefix-sum kernel processes the chunk in 256-thread blocks
        // of `scanPerThread` elements; the constant is compile-time
        // embedded so this product must cover the largest chunk we ever
        // dispatch (= chunkCap).
        const scanPerThread = Math.ceil(options.chunkCap / 256);

        // Uniform format for the small prepareIndirect kernel — just the
        // two slot-base indices into the device's indirect dispatch
        // buffer. Different layout from the shared `Uniforms` block.
        const prepareIndirectUniforms: UniformFormat[] = [
            new UniformFormat('sortSlotBase', UNIFORMTYPE_UINT),
            new UniformFormat('boundariesSlotBase', UNIFORMTYPE_UINT)
        ];

        const projection = options.projection;
        this.projectShader = mkShader('splat-project', projectWgsl(coeffs, options.maxCoveragePerSplat, projection), this.projectBgFormat);
        this.prefixSumShader = mkShader('splat-tilebin-prefix-sum', prefixSumWgsl(scanPerThread), this.prefixSumBgFormat);
        this.emitPairsShader = mkShader('splat-tilebin-emit-pairs', tileBinEmitPairsWgsl(projection), this.emitPairsBgFormat);
        this.prepareIndirectShader = mkShader('splat-tilebin-prepare-indirect', prepareIndirectWgsl(), this.prepareIndirectBgFormat, prepareIndirectUniforms);
        this.initTileOffsetsShader = mkShader('splat-tilebin-init-tile-offsets', initTileOffsetsWgsl(), this.initTileOffsetsBgFormat);
        this.findBoundariesShader = mkShader('splat-tilebin-find-boundaries', findBoundariesWgsl(), this.findBoundariesBgFormat);
        this.rasterizeBinnedShader = mkShader('splat-rasterize-binned', rasterizeBinnedWgsl(projection), this.rasterizeBinnedBgFormat);
        this.finalizeShader = mkShader('splat-finalize-pack', finalizeWgsl(), this.finalizeBgFormat);

        // Buffer sizing. runningState/output cover exactly the group's
        // tile rectangle (groupTilesX × groupTilesY × TILE_SIZE²), not the
        // bounding-square of the larger dimension as before.
        const groupPixels = this.groupPixelW * this.groupPixelH;
        const inputBytes = options.chunkCap * this.inputStride * 4;
        const projBytes = options.chunkCap * PROJECTION_STRIDE_F32 * 4;
        const stateBytes = groupPixels * RUNNING_STATE_STRIDE_F32 * 4;
        const outputBytes = groupPixels * OUTPUT_STRIDE_U32 * 4;
        // Tile-bin buffer sizes. tileOffsets has one slot per tile plus a
        // trailing sentinel so a tile's slice end is just `tileOffsets[T + 1]`.
        const tileOffsetsBytes = (numTiles + 1) * 4;

        const coverageBytes = options.chunkCap * 4;
        const emitOffsetBytes = options.chunkCap * 4;
        // Pair buffers. Two parallel u32 buffers (tileKeys + splatValues)
        // sized to chunkCap × maxCoveragePerSplat. The project shader
        // clamps each splat's coverage at maxCoveragePerSplat so the
        // sum never exceeds this cap regardless of scene density.
        const pairsBytes = options.chunkCap * options.maxCoveragePerSplat * 4;
        const totalPairsBytes = 4;

        const inputBuffer = new StorageBuffer(device, inputBytes, BUFFERUSAGE_COPY_DST);
        const projBuffer = new StorageBuffer(device, projBytes, 0);
        const runningStateBuffer = new StorageBuffer(device, stateBytes, BUFFERUSAGE_COPY_DST);
        const outputBuffer = new StorageBuffer(device, outputBytes, BUFFERUSAGE_COPY_SRC);
        const tileOffsetsBuffer = new StorageBuffer(device, tileOffsetsBytes, 0);
        const coverageBuffer = new StorageBuffer(device, coverageBytes, 0);
        const emitOffsetBuffer = new StorageBuffer(device, emitOffsetBytes, 0);
        const tileKeysBuffer = new StorageBuffer(device, pairsBytes, 0);
        const splatValuesBuffer = new StorageBuffer(device, pairsBytes, 0);
        const totalPairsBuffer = new StorageBuffer(device, totalPairsBytes, 0);

        const projectCompute = new Compute(device, this.projectShader, 'splat-project');
        projectCompute.setParameter('splats', inputBuffer);
        projectCompute.setParameter('projected', projBuffer);
        projectCompute.setParameter('coverage', coverageBuffer);

        const prefixSumCompute = new Compute(device, this.prefixSumShader, 'splat-tilebin-prefix-sum');
        prefixSumCompute.setParameter('coverage', coverageBuffer);
        prefixSumCompute.setParameter('emitOffset', emitOffsetBuffer);
        prefixSumCompute.setParameter('totalPairs', totalPairsBuffer);

        const emitPairsCompute = new Compute(device, this.emitPairsShader, 'splat-tilebin-emit-pairs');
        emitPairsCompute.setParameter('projected', projBuffer);
        emitPairsCompute.setParameter('emitOffset', emitOffsetBuffer);
        emitPairsCompute.setParameter('coverage', coverageBuffer);
        emitPairsCompute.setParameter('tileKeys', tileKeysBuffer);
        emitPairsCompute.setParameter('splatValues', splatValuesBuffer);

        const prepareIndirectCompute = new Compute(device, this.prepareIndirectShader, 'splat-tilebin-prepare-indirect');
        prepareIndirectCompute.setParameter('totalPairs', totalPairsBuffer);
        // `indirectBuffer` is bound per-chunk after the device has
        // allocated its indirect-dispatch buffer.

        const initTileOffsetsCompute = new Compute(device, this.initTileOffsetsShader, 'splat-tilebin-init-tile-offsets');
        initTileOffsetsCompute.setParameter('totalPairs', totalPairsBuffer);
        initTileOffsetsCompute.setParameter('tileOffsets', tileOffsetsBuffer);

        const findBoundariesCompute = new Compute(device, this.findBoundariesShader, 'splat-tilebin-find-boundaries');
        findBoundariesCompute.setParameter('totalPairs', totalPairsBuffer);
        findBoundariesCompute.setParameter('tileOffsets', tileOffsetsBuffer);
        // `sortedTileKeys` is bound per-chunk after the radix sort runs.

        const rasterizeBinnedCompute = new Compute(device, this.rasterizeBinnedShader, 'splat-rasterize-binned');
        rasterizeBinnedCompute.setParameter('projected', projBuffer);
        rasterizeBinnedCompute.setParameter('runningState', runningStateBuffer);
        rasterizeBinnedCompute.setParameter('tileOffsets', tileOffsetsBuffer);
        // `sortedSplatIndices` is bound per-chunk inside `dispatchChunk`,
        // pointing at the radix sort's `sortedIndices` output buffer.

        const finalizeCompute = new Compute(device, this.finalizeShader, 'splat-finalize');
        finalizeCompute.setParameter('runningState', runningStateBuffer);
        finalizeCompute.setParameter('output', outputBuffer);

        this.buffers = {
            inputBuffer,
            projBuffer,
            runningStateBuffer,
            outputBuffer,
            tileOffsetsBuffer,
            coverageBuffer,
            emitOffsetBuffer,
            totalPairsBuffer,
            tileKeysBuffer,
            splatValuesBuffer,
            projectCompute,
            prefixSumCompute,
            emitPairsCompute,
            prepareIndirectCompute,
            initTileOffsetsCompute,
            findBoundariesCompute,
            rasterizeBinnedCompute,
            finalizeCompute
        };

        // CPU-side cleared running state: T = 1 per pixel, color = 0.
        // Reused across groups; uploaded to runningStateBuffer at beginGroup.
        this.clearStatePattern = new Float32Array(groupPixels * RUNNING_STATE_STRIDE_F32);
        for (let i = 0; i < groupPixels; i++) {
            this.clearStatePattern[i * 4 + 3] = 1; // T = 1
        }

        this.radixSort = new ComputeRadixSort(device);

        // The per-chunk pipeline reserves 2 slots in the device's
        // indirect-dispatch buffer (one for the radix sort, one for
        // find-boundaries). PC resets the slot counter at frame-end
        // only; for an offline render, the counter monotonically grows
        // across (chunk × sub-frame) iterations. At 8K with sub-frame
        // split and chunkCap squeezed down by binding limits we can
        // hit ~70 k slots. Pre-allocate generously — each slot is only
        // 12 bytes, so 256 k slots = 3 MB.
        const wantedSlots = 256 * 1024;
        // @ts-ignore - maxIndirectDispatchCount is a public property on
        // the WebGPU device but not in the public GraphicsDevice type.
        const cur = (device as { maxIndirectDispatchCount?: number }).maxIndirectDispatchCount ?? 0;
        if (cur < wantedSlots) {
            // @ts-ignore
            (device as { maxIndirectDispatchCount: number }).maxIndirectDispatchCount = wantedSlots;
        }
    }

    /**
     * Apply the global (camera + image + background) uniforms to every
     * pipeline compute instance, plus the per-group origin/extent fields.
     *
     * The group abstraction is retained as a hook for future subframe
     * rendering — when a render is split into multiple groups, each call
     * sets the current group's pixel rectangle so the project shader's
     * AABB cull skips splats outside the group.
     *
     * @param groupX - Group index along X.
     * @param groupY - Group index along Y.
     * @param groupTilesX - Number of tiles in this group along X.
     * @param groupTilesY - Number of tiles in this group along Y.
     */
    private setUniforms(
        groupX: number,
        groupY: number,
        groupTilesX: number,
        groupTilesY: number
    ): void {
        const o = this.options;
        const originX = groupX * this.groupPixelW;
        const originY = groupY * this.groupPixelH;
        const maxX = originX + groupTilesX * TILE_SIZE;
        const maxY = originY + groupTilesY * TILE_SIZE;

        const b = this.buffers;
        for (const c of [
            b.projectCompute,
            b.prefixSumCompute,
            b.emitPairsCompute,
            b.initTileOffsetsCompute,
            b.findBoundariesCompute,
            b.rasterizeBinnedCompute,
            b.finalizeCompute
        ]) {
            c.setParameter('rightX', o.rightX); c.setParameter('rightY', o.rightY); c.setParameter('rightZ', o.rightZ);
            c.setParameter('_p0', 0);
            c.setParameter('downX', o.downX); c.setParameter('downY', o.downY); c.setParameter('downZ', o.downZ);
            c.setParameter('_p1', 0);
            c.setParameter('forwardX', o.forwardX); c.setParameter('forwardY', o.forwardY); c.setParameter('forwardZ', o.forwardZ);
            c.setParameter('_p2', 0);
            c.setParameter('eyeX', o.eyeX); c.setParameter('eyeY', o.eyeY); c.setParameter('eyeZ', o.eyeZ);
            c.setParameter('_p3', 0);
            c.setParameter('focalX', o.focalX); c.setParameter('focalY', o.focalY);
            c.setParameter('near', o.near); c.setParameter('_p4', 0);
            c.setParameter('imageWidth', o.imageWidth); c.setParameter('imageHeight', o.imageHeight);
            c.setParameter('splatStride', this.inputStride);
            // chunkSize set per-dispatch
            c.setParameter('groupPixelMinX', originX);
            c.setParameter('groupPixelMinY', originY);
            c.setParameter('groupPixelMaxX', maxX);
            c.setParameter('groupPixelMaxY', maxY);
            c.setParameter('groupTilesX', groupTilesX);
            c.setParameter('groupTilesY', groupTilesY);
            c.setParameter('groupPixelOriginX', originX);
            c.setParameter('groupPixelOriginY', originY);
            c.setParameter('bgR', o.bgR); c.setParameter('bgG', o.bgG);
            c.setParameter('bgB', o.bgB); c.setParameter('bgA', o.bgA);
        }
    }

    /**
     * Begin processing a group. Clears running state and sets uniforms.
     *
     * @param groupX - Group index along X.
     * @param groupY - Group index along Y.
     * @param groupTilesX - Number of tiles in this group along X.
     * @param groupTilesY - Number of tiles in this group along Y.
     */
    beginGroup(
        groupX: number,
        groupY: number,
        groupTilesX: number,
        groupTilesY: number
    ): void {
        this.setUniforms(groupX, groupY, groupTilesX, groupTilesY);
        this.activeTilesX = groupTilesX;
        this.activeTilesY = groupTilesY;
        const groupPixels = groupTilesX * groupTilesY * TILE_SIZE * TILE_SIZE;
        this.buffers.runningStateBuffer.write(
            0, this.clearStatePattern, 0, groupPixels * RUNNING_STATE_STRIDE_F32
        );
    }

    /**
     * Commit pending GPU work. Called at chunk boundaries so each chunk's
     * uniform-buffer values are captured before the next chunk overwrites
     * them — a `Compute` instance's persistent uniform buffer is updated
     * by `setParameter`, and the dispatch only captures the value on
     * submit. Within a chunk, every dispatch uses a distinct `Compute`
     * instance, so no internal submits are needed.
     */
    submit(): void {
        // @ts-ignore - submit() is exposed by WebgpuGraphicsDevice but not on the public GraphicsDevice type.
        const submit = (this.device as { submit?: () => void }).submit;
        if (!submit) {
            throw new Error('GpuSplatRasterizer requires a GraphicsDevice with a submit() method (WebGPU backend).');
        }
        submit.call(this.device);
    }

    /**
     * Reserve a fresh sort + find-boundaries slot pair in the device's
     * indirect-dispatch buffer for this chunk. The returned indices are
     * consumed by `dispatchTileBinChunk` (internally) and exposed for
     * cross-cutting use (e.g. the radix sort needs the sort slot).
     *
     * @returns Two fresh slot indices in the device's indirect dispatch
     * buffer: one for the radix sort's indirect dispatch, one for the
     * find-boundaries indirect dispatch.
     */
    private acquireIndirectSlots(): { sortSlot: number; boundariesSlot: number } {
        // @ts-ignore - getIndirectDispatchSlot exists on WebgpuGraphicsDevice.
        const get = (this.device as { getIndirectDispatchSlot?: (count?: number) => number }).getIndirectDispatchSlot;
        if (!get) {
            throw new Error('GpuSplatRasterizer requires a GraphicsDevice with getIndirectDispatchSlot() (WebGPU backend).');
        }
        const sortSlot = get.call(this.device, 1);
        const boundariesSlot = get.call(this.device, 1);
        return { sortSlot, boundariesSlot };
    }

    /**
     * Dispatch the entire per-chunk tile-bin + rasterize pipeline on the
     * GPU with zero CPU readbacks:
     *
     *   pack-and-upload → project + coverage → prefix-sum (writes
     *   emitOffsets + totalPairs) → emit-pairs (writes tileKeys +
     *   splatValues) → prepare-indirect (writes workgroup counts into
     *   the device's indirect-dispatch buffer for the sort and
     *   find-boundaries) → radix sortIndirect (key+value: tileKeys
     *   sorted, splatValues reordered) → init tile-offsets to sentinel
     *   → find-boundaries (atomicMin) → rasterize.
     *
     * All eight dispatches use distinct `Compute` instances, so their
     * persistent uniform buffers don't alias each other within a chunk;
     * a single `submit()` after the rasterize captures everything before
     * the next chunk starts overwriting `setParameter` values.
     *
     * @param chunkData - Float32Array containing `chunkSize × inputStride` floats.
     * @param chunkSize - Number of gaussians in this chunk (≤ chunkCap).
     */
    dispatchChunk(chunkData: Float32Array, chunkSize: number): void {
        if (chunkSize === 0) return;
        if (chunkSize > this.chunkCap) {
            throw new Error(`chunkSize ${chunkSize} exceeds chunkCap ${this.chunkCap}`);
        }
        const b = this.buffers;

        // --- 1. Upload chunk + project (writes projected, coverage). ---
        b.inputBuffer.write(0, chunkData, 0, chunkSize * this.inputStride);
        b.projectCompute.setParameter('chunkSize', chunkSize);
        b.projectCompute.setupDispatch(Math.ceil(chunkSize / 64), 1, 1);
        this.device.computeDispatch([b.projectCompute], 'splat-project');

        // --- 2. GPU prefix-sum (coverage → emitOffsets + totalPairs). ---
        b.prefixSumCompute.setParameter('chunkSize', chunkSize);
        b.prefixSumCompute.setupDispatch(1, 1, 1);
        this.device.computeDispatch([b.prefixSumCompute], 'splat-tilebin-prefix-sum');

        // --- 3. Emit (tileKey, splatValue) pairs. ---
        b.emitPairsCompute.setParameter('chunkSize', chunkSize);
        b.emitPairsCompute.setupDispatch(Math.ceil(chunkSize / 64), 1, 1);
        this.device.computeDispatch([b.emitPairsCompute], 'splat-tilebin-emit-pairs');

        // --- 4. Prepare indirect dispatch params for sort + find-boundaries. ---
        // @ts-ignore - indirectDispatchBuffer getter is WebGPU-only.
        const indirectBuf = (this.device as { indirectDispatchBuffer?: StorageBuffer | null }).indirectDispatchBuffer;
        const { sortSlot, boundariesSlot } = this.acquireIndirectSlots();
        if (!indirectBuf) {
            throw new Error('Device indirectDispatchBuffer not allocated (WebGPU backend required).');
        }
        b.prepareIndirectCompute.setParameter('indirectBuffer', indirectBuf);
        // Each slot is 3 u32s in the buffer; the shader writes to
        // indirectBuffer[base + {0,1,2}], so base = slot * 3.
        b.prepareIndirectCompute.setParameter('sortSlotBase', sortSlot * 3);
        b.prepareIndirectCompute.setParameter('boundariesSlotBase', boundariesSlot * 3);
        b.prepareIndirectCompute.setupDispatch(1, 1, 1);
        this.device.computeDispatch([b.prepareIndirectCompute], 'splat-tilebin-prepare-indirect');

        // --- 5. Radix sort the pairs (indirect dispatch, key + value). ---
        // tileKeysBuffer holds u32 tileIdx; splatValuesBuffer holds u32
        // splatIdx as the initial-values payload. After the sort:
        //   - radixSort.sortedKeys[i]    = the i-th sorted tile index
        //   - radixSort.sortedIndices[i] = the splat index originally
        //                                  paired with that tile
        // The radix sort is stable, so within each tile the splatValues
        // remain in their input order = depth-monotonic (from the CPU
        // pre-sort), giving depth-ordered compositing per tile for free.
        // numBits = sortKeyBits (rounded up to multiple of 4 from
        // ceil(log2(numTiles))) — only the minimum required passes run.
        const pairsCap = this.chunkCap * this.options.maxCoveragePerSplat;
        this.radixSort.sortIndirect(
            b.tileKeysBuffer, pairsCap, this.sortKeyBits, sortSlot,
            b.totalPairsBuffer, b.splatValuesBuffer
        );
        const sortedTileKeysBuf = this.radixSort.sortedKeys;
        const sortedSplatIndicesBuf = this.radixSort.sortedIndices;
        if (!sortedTileKeysBuf || !sortedSplatIndicesBuf) {
            throw new Error('ComputeRadixSort returned null sortedKeys/sortedIndices after sortIndirect()');
        }

        // --- 6. Init tile-offsets to the sentinel (= totalPairs). ---
        const numTiles = this.groupTilesX * this.groupTilesY;
        b.initTileOffsetsCompute.setupDispatch(Math.ceil((numTiles + 1) / 64), 1, 1);
        this.device.computeDispatch([b.initTileOffsetsCompute], 'splat-tilebin-init-tile-offsets');

        // --- 7. Find tile boundaries via atomicMin. ---
        b.findBoundariesCompute.setParameter('sortedTileKeys', sortedTileKeysBuf);
        b.findBoundariesCompute.setupIndirectDispatch(boundariesSlot);
        this.device.computeDispatch([b.findBoundariesCompute], 'splat-tilebin-find-boundaries');

        // --- 8. Rasterize: walk each tile's slice in depth order. ---
        b.rasterizeBinnedCompute.setParameter('sortedSplatIndices', sortedSplatIndicesBuf);
        b.rasterizeBinnedCompute.setParameter('chunkSize', chunkSize);
        b.rasterizeBinnedCompute.setupDispatch(this.groupTilesX, this.groupTilesY, 1);
        this.device.computeDispatch([b.rasterizeBinnedCompute], 'splat-rasterize-binned');

        this.submit();
    }

    /**
     * Finish processing a group. Dispatches finalize-pack and starts an
     * async readback of the group's RGBA8 pixel bytes.
     *
     * Dispatch + readback are sized to the ACTIVE group dimensions (set
     * by the most recent `beginGroup`), not the constructor-provided
     * maximum, so edge sub-frames smaller than the max don't pay for
     * unused workgroups or readback bytes.
     *
     * @returns Promise resolving to the active group's RGBA byte buffer
     * (`activeTilesX·16 × activeTilesY·16 × 4` bytes).
     */
    finishGroup(): Promise<Uint8Array> {
        const b = this.buffers;
        b.finalizeCompute.setupDispatch(this.activeTilesX, this.activeTilesY, 1);
        this.device.computeDispatch([b.finalizeCompute], 'splat-finalize');

        const activePixelW = this.activeTilesX * TILE_SIZE;
        const activePixelH = this.activeTilesY * TILE_SIZE;
        const groupOutputBytes = activePixelW * activePixelH * 4;
        return b.outputBuffer.read(0, groupOutputBytes, null, true) as Promise<Uint8Array>;
    }

    /**
     * Release all GPU resources.
     */
    destroy(): void {
        this.radixSort.destroy();
        const b = this.buffers;
        b.inputBuffer.destroy();
        b.projBuffer.destroy();
        b.runningStateBuffer.destroy();
        b.outputBuffer.destroy();
        b.tileOffsetsBuffer.destroy();
        b.coverageBuffer.destroy();
        b.emitOffsetBuffer.destroy();
        b.totalPairsBuffer.destroy();
        b.tileKeysBuffer.destroy();
        b.splatValuesBuffer.destroy();
        this.projectShader.destroy();
        this.prefixSumShader.destroy();
        this.emitPairsShader.destroy();
        this.prepareIndirectShader.destroy();
        this.initTileOffsetsShader.destroy();
        this.findBoundariesShader.destroy();
        this.rasterizeBinnedShader.destroy();
        this.finalizeShader.destroy();
        this.projectBgFormat.destroy();
        this.prefixSumBgFormat.destroy();
        this.emitPairsBgFormat.destroy();
        this.prepareIndirectBgFormat.destroy();
        this.initTileOffsetsBgFormat.destroy();
        this.findBoundariesBgFormat.destroy();
        this.rasterizeBinnedBgFormat.destroy();
        this.finalizeBgFormat.destroy();
    }
}

export { GpuSplatRasterizer, type SplatRasterizerOptions };
