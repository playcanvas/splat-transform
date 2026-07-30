/**
 * Shared WGSL for the field-L2 decimation cost: constants, the Knud Thomsen
 * ellipsoid area, and Smith's closed-form symmetric-3×3 eigenvalues —
 * interpolated into the kernels that evaluate merged-Gaussian costs
 * (GpuEdgeCost, GpuRecost). The Gaussian cross-product stays per-kernel.
 *
 * Mirrors the CPU implementations in decimate/edge-cost-cpu.ts and
 * decimate/recost-core.ts — keep them in lockstep.
 */
const gaussianL2Wgsl = /* wgsl */`
const EPS_COV: f32 = 1e-8;
const PI_1_5: f32 = 5.5683279968317084;       // π^{3/2}
const TWO_PI_1_5: f32 = 15.749609945722419;   // (2π)^{3/2}
const TWO_PI_3: f32 = 2.0943951023931953;      // 2π/3
const ELLIP_P: f32 = 1.6075;
// Scale-free DC colour dissimilarity weight (4π·1e-6 in base-colour space —
// see COLOR_WEIGHT in decimate/edge-cost-cpu.ts, mirrored here).
const COLOR_WEIGHT: f32 = 1.2566370614359172e-5;

// Knud Thomsen ellipsoid surface area (matches CPU ellipsoidArea).
fn ellipsoidArea(sx: f32, sy: f32, sz: f32) -> f32 {
    let a = pow(sx * sy, ELLIP_P);
    let b = pow(sx * sz, ELLIP_P);
    let c = pow(sy * sz, ELLIP_P);
    return 4.0 * 3.141592653589793 * pow((a + b + c) / 3.0, 1.0 / ELLIP_P);
}

// Smith closed-form eigenvalues of a symmetric 3×3 (6 comps: xx,xy,xz,yy,yz,zz).
// Consumers feed these through ellipsoidArea (symmetric), so no ordering is
// promised beyond the Smith branch's (largest, middle, smallest).
fn eig3(m: array<f32, 6>) -> vec3f {
    let q = (m[0] + m[3] + m[5]) / 3.0;
    let p1 = m[1] * m[1] + m[2] * m[2] + m[4] * m[4];
    if (p1 <= 1e-30) {
        return vec3f(m[0], m[3], m[5]);
    }
    let p2 = (m[0] - q) * (m[0] - q) + (m[3] - q) * (m[3] - q) + (m[5] - q) * (m[5] - q) + 2.0 * p1;
    let p = sqrt(p2 / 6.0);
    let ip = 1.0 / p;
    let b00 = (m[0] - q) * ip; let b11 = (m[3] - q) * ip; let b22 = (m[5] - q) * ip;
    let b01 = m[1] * ip; let b02 = m[2] * ip; let b12 = m[4] * ip;
    let detB = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02) + b02 * (b01 * b12 - b11 * b02);
    let r = clamp(detB * 0.5, -1.0, 1.0);
    let phi = acos(r) / 3.0;
    let e0 = q + 2.0 * p * cos(phi);
    let e2 = q + 2.0 * p * cos(phi + TWO_PI_3);
    return vec3f(e0, 3.0 * q - e0 - e2, e2);
}
`;

export { gaussianL2Wgsl };
