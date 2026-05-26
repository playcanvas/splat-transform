/**
 * Quaternion → 3×3 rotation matrix, with early-out on zero-length quat.
 *
 * Reads:   rotW, rotX, rotY, rotZ (raw input quat components from the
 *          splat decode)
 * Defines: r00..r22 (rotation matrix entries, row-major)
 * Returns: early via writeInvalid + return if the quaternion has zero
 *          length (degenerate splat)
 *
 * Normalises the quaternion first so the rotation matrix is orthonormal
 * even if the upstream data stored an unnormalised quat.
 */
const quatRotation = /* wgsl */`
    let qlen2 = rotW * rotW + rotX * rotX + rotY * rotY + rotZ * rotZ;
    if (qlen2 == 0.0) { writeInvalid(i); return; }
    let invQ = inverseSqrt(qlen2);
    let qw = rotW * invQ;
    let qx = rotX * invQ;
    let qy = rotY * invQ;
    let qz = rotZ * invQ;

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
`;

export { quatRotation };
