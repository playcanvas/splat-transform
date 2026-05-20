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
    GraphicsDevice,
    Shader,
    StorageBuffer,
    UniformBufferFormat,
    UniformFormat
} from 'playcanvas';

import {
    AA_DILATION_COV,
    DISCRIMINANT_FLOOR,
    JACOBIAN_LIMIT_FACTOR,
    MAX_COVERAGE_PER_SPLAT,
    MIN_ALPHA,
    MIN_TRANSMITTANCE,
    OPACITY_CAP,
    RADIUS_FADE_END_PX,
    RADIUS_FADE_START_PX,
    SIGMA_CUTOFF,
    TILE_SIZE
} from '../render/config';

/**
 * Format a JS number as a WGSL `f32` literal. Adds an explicit `.0` so
 * integer-valued constants like `3` aren't parsed as `AbstractInt` —
 * keeps shaders readable when the constant flips to a fractional value.
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
 * a render — `numSHBands` and the per-group sizes determine GPU buffer
 * sizes and shader uniform layouts.
 */
interface SplatRasterizerOptions {
    /** Number of SH bands above DC (0–3). Determines input stride. */
    numSHBands: 0 | 1 | 2 | 3;
    /** Tiles per group axis (e.g. 8 → 128×128-pixel groups). */
    groupSizeTiles: number;
    /** Max gaussians per chunk; sizes the per-slot input + projection buffers. */
    chunkCap: number;
    /** Number of pipelined slots; the caller picks the slot per group. */
    numSlots: number;
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
 * @returns WGSL source.
 */
const projectWgsl = (coeffsPerChannel: number) => /* wgsl */`
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

    if (cz <= uniforms.near) { writeInvalid(i); return; }

    let invZ = 1.0 / cz;
    let screenX = uniforms.focalX * cx * invZ + f32(uniforms.imageWidth) * 0.5;
    let screenY = uniforms.focalY * cy * invZ + f32(uniforms.imageHeight) * 0.5;

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

    // Jacobian with x/z, y/z clamp.
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
    // un-clamped radius grows from RADIUS_FADE_START_PX to RADIUS_FADE_END_PX,
    // and discard beyond. The bbox we hand to the rasterizer is clamped at
    // RADIUS_FADE_END_PX so the binner doesn't reserve tile coverage for a
    // splat that contributes zero anyway. Softer than a hard clamp: prevents
    // the visible pop as the camera approaches a clipped splat.
    let fadeStart = ${wgslF32(RADIUS_FADE_START_PX)};
    let fadeEnd = ${wgslF32(RADIUS_FADE_END_PX)};
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
 */
const rasterizeBinnedWgsl = () => /* wgsl */`
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
@group(0) @binding(4) var<storage, read> tileData: array<u32>;

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

    for (var i: u32 = sliceStart; i < sliceEnd; i = i + 1u) {
        if (T < ${wgslF32(MIN_TRANSMITTANCE)}) { break; }
        let splatIdx = tileData[i];
        let v0 = projected[splatIdx * 3u + 0u];
        let dx = px - v0.x;
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

interface Slot {
    inputBuffer: StorageBuffer;
    projBuffer: StorageBuffer;
    runningStateBuffer: StorageBuffer;
    outputBuffer: StorageBuffer;
    /** Per-tile offset table for binned rasterize: `(numTiles + 1) × u32`. */
    tileOffsetsBuffer: StorageBuffer;
    /** Tile-binned splat indices, sized for `chunkCap × MAX_COVERAGE`. */
    tileDataBuffer: StorageBuffer;
    projectCompute: Compute;
    rasterizeBinnedCompute: Compute;
    finalizeCompute: Compute;
}


/**
 * GPU-accelerated splat rasterizer.
 *
 * Owns three compute shaders (project, rasterize-binned, finalize-pack) and
 * per-slot GPU buffers. Designed to be driven by an external orchestrator
 * (`raster-pass.ts`) that handles BVH queries, depth sort, and tile-binning.
 *
 * Per-render flow:
 *   1. `beginGroup(slot, ...)` — clears the slot's running state and sets
 *      uniforms for this render.
 *   2. For each chunk of depth-sorted splats:
 *      a. `dispatchProject(slot, chunkData, chunkSize)` — uploads the
 *         chunk, dispatches the project pass, and readbacks the projection
 *         records so the CPU binner can compute per-tile splat lists.
 *      b. `rasterizeChunkBinned(slot, chunkSize, tileOffsets, tileData)` —
 *         uploads the pre-built per-tile splat lists and dispatches the
 *         binned rasterize pass. Running state persists across chunks;
 *         saturated pixels short-circuit via the T < MIN_TRANSMITTANCE
 *         early-out.
 *   3. `finishGroup(slot)` — dispatches finalize-pack and starts an async
 *      readback. Returns a `Promise<Uint8Array>` resolved when the GPU has
 *      finished writing this group's RGBA bytes.
 */
class GpuSplatRasterizer {
    private device: GraphicsDevice;
    private options: SplatRasterizerOptions;
    private projectShader: Shader;
    private rasterizeBinnedShader: Shader;
    private finalizeShader: Shader;
    private projectBgFormat: BindGroupFormat;
    private rasterizeBinnedBgFormat: BindGroupFormat;
    private finalizeBgFormat: BindGroupFormat;
    private slots: Slot[];
    private clearStatePattern: Float32Array;

    /** Floats per gaussian in the input buffer (depends on SH band count). */
    readonly inputStride: number;
    /** Tiles per group axis. */
    readonly groupSizeTiles: number;
    /** Max gaussians per chunk. */
    readonly chunkCap: number;
    /** Number of pipelined slots. */
    readonly numSlots: number;
    /** Pixels per group axis. */
    readonly groupPixelSize: number;

    constructor(device: GraphicsDevice, options: SplatRasterizerOptions) {
        this.device = device;
        this.options = options;
        this.groupSizeTiles = options.groupSizeTiles;
        this.chunkCap = options.chunkCap;
        this.numSlots = options.numSlots;
        this.groupPixelSize = options.groupSizeTiles * TILE_SIZE;

        const coeffs = numSHCoeffsPerChannel(options.numSHBands);
        this.inputStride = 14 + 3 * coeffs;

        this.projectBgFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('splats', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('projected', SHADERSTAGE_COMPUTE)
        ]);
        this.rasterizeBinnedBgFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('projected', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('runningState', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('tileOffsets', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('tileData', SHADERSTAGE_COMPUTE, true)
        ]);
        this.finalizeBgFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('runningState', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('output', SHADERSTAGE_COMPUTE)
        ]);

        const mkShader = (name: string, source: string, bgFormat: BindGroupFormat) => new Shader(device, {
            name,
            shaderLanguage: SHADERLANGUAGE_WGSL,
            cshader: source,
            // @ts-ignore - computeUniformBufferFormats / computeBindGroupFormat are not in public Shader types.
            computeUniformBufferFormats: {
                uniforms: new UniformBufferFormat(device, uniformFormatEntries())
            },
            // @ts-ignore
            computeBindGroupFormat: bgFormat
        });

        this.projectShader = mkShader('splat-project', projectWgsl(coeffs), this.projectBgFormat);
        this.rasterizeBinnedShader = mkShader('splat-rasterize-binned', rasterizeBinnedWgsl(), this.rasterizeBinnedBgFormat);
        this.finalizeShader = mkShader('splat-finalize-pack', finalizeWgsl(), this.finalizeBgFormat);

        // Slot resources.
        const groupPixels = this.groupPixelSize * this.groupPixelSize;
        const inputBytes = options.chunkCap * this.inputStride * 4;
        const projBytes = options.chunkCap * PROJECTION_STRIDE_F32 * 4;
        const stateBytes = groupPixels * RUNNING_STATE_STRIDE_F32 * 4;
        const outputBytes = groupPixels * OUTPUT_STRIDE_U32 * 4;
        // Tile-bin buffer sizes. tileOffsets has one slot per tile plus a
        // trailing sentinel so a tile's slice end is just `tileOffsets[T + 1]`.
        // tileData is bounded by chunkCap × MAX_COVERAGE_PER_SPLAT — a worst
        // case that's never reached in practice but keeps the allocation
        // bounded without per-chunk reallocation.
        const numTiles = options.groupSizeTiles * options.groupSizeTiles;
        const tileOffsetsBytes = (numTiles + 1) * 4;
        const tileDataBytes = options.chunkCap * MAX_COVERAGE_PER_SPLAT * 4;

        this.slots = [];
        for (let s = 0; s < this.numSlots; s++) {
            const inputBuffer = new StorageBuffer(device, inputBytes, BUFFERUSAGE_COPY_DST);
            // projBuffer needs COPY_SRC so the CPU binner can read screen
            // bounds after the project pass.
            const projBuffer = new StorageBuffer(device, projBytes, BUFFERUSAGE_COPY_SRC);
            const runningStateBuffer = new StorageBuffer(device, stateBytes, BUFFERUSAGE_COPY_DST);
            const outputBuffer = new StorageBuffer(device, outputBytes, BUFFERUSAGE_COPY_SRC);
            const tileOffsetsBuffer = new StorageBuffer(device, tileOffsetsBytes, BUFFERUSAGE_COPY_DST);
            const tileDataBuffer = new StorageBuffer(device, tileDataBytes, BUFFERUSAGE_COPY_DST);

            const projectCompute = new Compute(device, this.projectShader, `splat-project-slot-${s}`);
            projectCompute.setParameter('splats', inputBuffer);
            projectCompute.setParameter('projected', projBuffer);

            const rasterizeBinnedCompute = new Compute(device, this.rasterizeBinnedShader, `splat-rasterize-binned-slot-${s}`);
            rasterizeBinnedCompute.setParameter('projected', projBuffer);
            rasterizeBinnedCompute.setParameter('runningState', runningStateBuffer);
            rasterizeBinnedCompute.setParameter('tileOffsets', tileOffsetsBuffer);
            rasterizeBinnedCompute.setParameter('tileData', tileDataBuffer);

            const finalizeCompute = new Compute(device, this.finalizeShader, `splat-finalize-slot-${s}`);
            finalizeCompute.setParameter('runningState', runningStateBuffer);
            finalizeCompute.setParameter('output', outputBuffer);

            this.slots.push({
                inputBuffer,
                projBuffer,
                runningStateBuffer,
                outputBuffer,
                tileOffsetsBuffer,
                tileDataBuffer,
                projectCompute,
                rasterizeBinnedCompute,
                finalizeCompute
            });
        }

        // CPU-side cleared running state: T = 1 per pixel, color = 0. Reused
        // across groups; uploaded to slot.runningStateBuffer at beginGroup.
        this.clearStatePattern = new Float32Array(groupPixels * RUNNING_STATE_STRIDE_F32);
        for (let i = 0; i < groupPixels; i++) {
            this.clearStatePattern[i * 4 + 3] = 1; // T = 1
        }
    }

    /**
     * Apply the global (camera + image + background) uniforms to a slot's
     * compute instances and the per-group fields.
     *
     * @param slot - Slot index.
     * @param groupX - Group X index.
     * @param groupY - Group Y index.
     * @param groupTilesX - Number of tiles in this group along X (≤ groupSizeTiles).
     * @param groupTilesY - Number of tiles in this group along Y.
     */
    private setUniforms(
        slot: number,
        groupX: number,
        groupY: number,
        groupTilesX: number,
        groupTilesY: number
    ): void {
        const o = this.options;
        const originX = groupX * this.groupPixelSize;
        const originY = groupY * this.groupPixelSize;
        const maxX = originX + groupTilesX * TILE_SIZE;
        const maxY = originY + groupTilesY * TILE_SIZE;

        const slotObj = this.slots[slot];
        for (const c of [
            slotObj.projectCompute,
            slotObj.rasterizeBinnedCompute,
            slotObj.finalizeCompute
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
     * Begin processing on a slot. Clears running state and sets uniforms.
     *
     * @param slot - Slot index.
     * @param groupX - Group X index.
     * @param groupY - Group Y index.
     * @param groupTilesX - Tiles along X for this group.
     * @param groupTilesY - Tiles along Y for this group.
     */
    beginGroup(
        slot: number,
        groupX: number,
        groupY: number,
        groupTilesX: number,
        groupTilesY: number
    ): void {
        this.setUniforms(slot, groupX, groupY, groupTilesX, groupTilesY);
        const groupPixels = groupTilesX * groupTilesY * TILE_SIZE * TILE_SIZE;
        this.slots[slot].runningStateBuffer.write(
            0, this.clearStatePattern, 0, groupPixels * RUNNING_STATE_STRIDE_F32
        );
    }

    /**
     * Upload a chunk and run only the project pass. Returns a Promise that
     * resolves to the chunk's projection records (`chunkSize × 3` vec4s).
     * Used by the tile-binned path: CPU reads back screen bounds, builds
     * per-tile lists, then calls `rasterizeChunkBinned`.
     *
     * The returned Uint8Array is the raw bytes of the projection buffer's
     * first `chunkSize × PROJECTION_STRIDE_F32 × 4` bytes. View it as
     * Float32Array to read fields:
     *   - `f32[i * 12 + 0..2]` — screenX, screenY, radius (vec4 0)
     *   - `f32[i * 12 + 4..7]` — covInv, alpha (vec4 1)
     *   - `f32[i * 12 + 8..10]` — RGB (vec4 2)
     *
     * @param slot - Slot index.
     * @param chunkData - Float32Array containing `chunkSize × inputStride` floats.
     * @param chunkSize - Number of gaussians in this chunk (≤ chunkCap).
     * @returns Promise resolving to the readback bytes (length `chunkSize × 48`).
     */
    dispatchProject(slot: number, chunkData: Float32Array, chunkSize: number): Promise<Uint8Array> {
        if (chunkSize === 0) return Promise.resolve(new Uint8Array(0));
        if (chunkSize > this.chunkCap) {
            throw new Error(`chunkSize ${chunkSize} exceeds chunkCap ${this.chunkCap}`);
        }
        const s = this.slots[slot];
        s.inputBuffer.write(0, chunkData, 0, chunkSize * this.inputStride);

        s.projectCompute.setParameter('chunkSize', chunkSize);
        s.projectCompute.setupDispatch(Math.ceil(chunkSize / 64), 1, 1);
        this.device.computeDispatch([s.projectCompute], 'splat-project');

        // Submit so the per-Compute uniform buffer captures this chunk's
        // `chunkSize` before the next dispatchProject overwrites it.
        // @ts-ignore - submit() is exposed by WebgpuGraphicsDevice but not on the public GraphicsDevice type.
        const submit = (this.device as { submit?: () => void }).submit;
        if (!submit) {
            throw new Error('GpuSplatRasterizer requires a GraphicsDevice with a submit() method (WebGPU backend).');
        }
        submit.call(this.device);

        const projBytes = chunkSize * PROJECTION_STRIDE_F32 * 4;
        return s.projBuffer.read(0, projBytes, null, true) as Promise<Uint8Array>;
    }

    /**
     * Upload pre-computed tile lists and run the binned rasterize pass on
     * the projection produced by the most recent `dispatchProject` call
     * for this slot.
     *
     * Caller responsibilities:
     *   - `tileOffsets` has length `numTiles + 1` (where `numTiles =
     *     groupSizeTiles²`); `tileOffsets[T]` is the start of tile T's
     *     slice in `tileData`, and `tileOffsets[numTiles]` is the total
     *     pair count.
     *   - `tileData` is a flat array of `splatIdx` values, grouped by
     *     tile, depth-sorted within each tile.
     *
     * @param slot - Slot index.
     * @param chunkSize - The chunkSize last used with `dispatchProject` (sets the uniform).
     * @param tileOffsets - `(numTiles + 1) × u32`.
     * @param tileData - Variable-length `u32` array of splat indices.
     */
    rasterizeChunkBinned(
        slot: number,
        chunkSize: number,
        tileOffsets: Uint32Array,
        tileData: Uint32Array
    ): void {
        const s = this.slots[slot];

        s.tileOffsetsBuffer.write(0, tileOffsets, 0, tileOffsets.length);
        if (tileData.length > 0) {
            s.tileDataBuffer.write(0, tileData, 0, tileData.length);
        }

        s.rasterizeBinnedCompute.setParameter('chunkSize', chunkSize);
        s.rasterizeBinnedCompute.setupDispatch(this.groupSizeTiles, this.groupSizeTiles, 1);
        this.device.computeDispatch([s.rasterizeBinnedCompute], 'splat-rasterize-binned');

        // Submit so the uniform buffer captures this chunk's chunkSize.
        // @ts-ignore - submit() is exposed by WebgpuGraphicsDevice but not on the public GraphicsDevice type.
        const submit = (this.device as { submit?: () => void }).submit;
        if (!submit) {
            throw new Error('GpuSplatRasterizer requires a GraphicsDevice with a submit() method (WebGPU backend).');
        }
        submit.call(this.device);
    }

    /**
     * Finish processing a group. Dispatches finalize-pack and starts an
     * async readback of the group's RGBA8 pixel bytes.
     *
     * @param slot - Slot index.
     * @param groupTilesX - Tiles along X for this group.
     * @param groupTilesY - Tiles along Y for this group.
     * @returns Promise resolving to the group's RGBA byte buffer
     * (`groupTilesX·16 × groupTilesY·16 × 4` bytes).
     */
    finishGroup(slot: number, groupTilesX: number, groupTilesY: number): Promise<Uint8Array> {
        const s = this.slots[slot];
        s.finalizeCompute.setupDispatch(this.groupSizeTiles, this.groupSizeTiles, 1);
        this.device.computeDispatch([s.finalizeCompute], 'splat-finalize-pack');

        const groupOutputBytes = groupTilesX * TILE_SIZE * groupTilesY * TILE_SIZE * 4;
        return s.outputBuffer.read(0, groupOutputBytes, null, true) as Promise<Uint8Array>;
    }

    /**
     * Release all GPU resources.
     */
    destroy(): void {
        for (const s of this.slots) {
            s.inputBuffer.destroy();
            s.projBuffer.destroy();
            s.runningStateBuffer.destroy();
            s.outputBuffer.destroy();
            s.tileOffsetsBuffer.destroy();
            s.tileDataBuffer.destroy();
        }
        this.projectShader.destroy();
        this.rasterizeBinnedShader.destroy();
        this.finalizeShader.destroy();
        this.projectBgFormat.destroy();
        this.rasterizeBinnedBgFormat.destroy();
        this.finalizeBgFormat.destroy();
    }
}

export { GpuSplatRasterizer, type SplatRasterizerOptions };
