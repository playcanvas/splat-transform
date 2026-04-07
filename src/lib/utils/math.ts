import { Mat4, Quat, Vec3 } from 'playcanvas';

const sigmoid = (v: number) => 1 / (1 + Math.exp(-v));

const _tv = new Vec3();

/**
 * A source-to-engine coordinate transform comprising translation, rotation
 * and uniform scale. Lives alongside a DataTable to describe how raw
 * column data maps to PlayCanvas engine coordinates.
 *
 * @example
 * ```ts
 * const t = new Transform().fromEulers(0, 0, 180);
 * console.log(t.isIdentity()); // false
 *
 * const inv = new Transform().invert(t);
 * console.log(t.mul(inv).isIdentity()); // true
 * ```
 */
class Transform {
    translation: Vec3;
    rotation: Quat;
    scale: number;

    constructor(translation?: Vec3, rotation?: Quat, scale?: number) {
        this.translation = translation ? translation.clone() : new Vec3();
        this.rotation = rotation ? rotation.clone() : new Quat();
        this.scale = scale ?? 1;
    }

    /**
     * Fills the provided Mat4 with the TRS matrix for this transform.
     *
     * @param result - The Mat4 to fill.
     * @returns The filled Mat4.
     */
    getMatrix(result: Mat4): Mat4 {
        return result.setTRS(this.translation, this.rotation, new Vec3(this.scale, this.scale, this.scale));
    }

    /**
     * Tests whether this transform equals another within the given tolerance.
     * Quaternion comparison accounts for double-cover (q and -q represent
     * the same rotation).
     *
     * @param other - The transform to compare against.
     * @param epsilon - Floating-point tolerance. Defaults to 1e-6.
     * @returns True if the transforms are equal within the tolerance.
     */
    equals(other: Transform, epsilon = 1e-6): boolean {
        const ta = this.translation;
        const tb = other.translation;
        if (Math.abs(ta.x - tb.x) > epsilon || Math.abs(ta.y - tb.y) > epsilon || Math.abs(ta.z - tb.z) > epsilon) {
            return false;
        }
        const ra = this.rotation;
        const rb = other.rotation;
        const dot = ra.x * rb.x + ra.y * rb.y + ra.z * rb.z + ra.w * rb.w;
        if (Math.abs(dot) < 1 - epsilon) {
            return false;
        }
        if (Math.abs(this.scale - other.scale) > epsilon) {
            return false;
        }
        return true;
    }

    /**
     * Tests whether this transform is effectively identity within the given tolerance.
     *
     * @param epsilon - Floating-point tolerance. Defaults to 1e-6.
     * @returns True if identity within the tolerance.
     */
    isIdentity(epsilon = 1e-6): boolean {
        return this.equals(Transform.IDENTITY, epsilon);
    }

    /**
     * Creates a deep copy of this transform.
     *
     * @returns A new Transform with the same values.
     */
    clone(): Transform {
        return new Transform(this.translation, this.rotation, this.scale);
    }

    /**
     * Sets this transform to the inverse of the given source transform.
     *
     * @param src - The transform to invert. Defaults to this (in-place).
     * @returns This transform (for chaining).
     */
    invert(src: Transform = this): Transform {
        this.scale = 1 / src.scale;
        this.rotation.copy(src.rotation).invert();
        this.translation.copy(src.translation).mulScalar(-this.scale);
        this.rotation.transformVector(this.translation, this.translation);
        return this;
    }

    /**
     * Sets this transform to the composition of a * b. Handles aliasing
     * (either a or b may be this).
     *
     * @param a - The first (left) transform.
     * @param b - The second (right) transform.
     * @returns This transform (for chaining).
     */
    mul2(a: Transform, b: Transform): Transform {
        // Translation must be computed first using original a.rotation
        a.rotation.transformVector(b.translation, _tv);
        _tv.mulScalar(a.scale).add(a.translation);

        this.rotation.mul2(a.rotation, b.rotation);
        this.scale = a.scale * b.scale;
        this.translation.copy(_tv);
        return this;
    }

    /**
     * Sets this transform to this * other.
     *
     * @param other - The transform to multiply with.
     * @returns This transform (for chaining).
     */
    mul(other: Transform): Transform {
        return this.mul2(this, other);
    }

    /**
     * Sets this transform to a rotation-only transform from Euler angles in degrees.
     *
     * @param x - Rotation around X axis in degrees.
     * @param y - Rotation around Y axis in degrees.
     * @param z - Rotation around Z axis in degrees.
     * @returns This transform (for chaining).
     */
    fromEulers(x: number, y: number, z: number): Transform {
        this.translation.set(0, 0, 0);
        this.rotation.setFromEulerAngles(x, y, z);
        this.scale = 1;
        return this;
    }

    static freeze(t: Transform): Readonly<Transform> {
        Object.freeze(t.translation);
        Object.freeze(t.rotation);
        return Object.freeze(t);
    }

    static IDENTITY = Transform.freeze(new Transform());

    /**
     * PLY coordinate convention: 180-degree rotation around Z.
     * Used by PLY, splat, KSplat, SPZ, and legacy SOG formats.
     */
    static PLY = Transform.freeze(new Transform().fromEulers(0, 0, 180));
}

export { sigmoid, Transform };
