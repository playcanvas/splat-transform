import { describe, it } from 'node:test';
import assert from 'node:assert';

import {
    computeGaussianInverse,
    evaluateGaussianAt
} from '../src/lib/data-table/gaussian-eval.js';

const EPSILON = 1e-6;

const assertClose = (actual, expected, tolerance, msg) => {
    assert.ok(
        Math.abs(actual - expected) < tolerance,
        `${msg}: expected ~${expected}, got ${actual}`
    );
};

describe('computeGaussianInverse', () => {
    it('should handle identity quaternion', () => {
        const g = computeGaussianInverse(1, 0, 0, 0, 0, 0, 0, 0);
        assertClose(g.qw, 1, EPSILON, 'qw');
        assertClose(g.qx, 0, EPSILON, 'qx');
        assertClose(g.qy, 0, EPSILON, 'qy');
        assertClose(g.qz, 0, EPSILON, 'qz');
        assertClose(g.isx, 1, EPSILON, 'isx (exp(-0))');
        assertClose(g.isy, 1, EPSILON, 'isy');
        assertClose(g.isz, 1, EPSILON, 'isz');
        assertClose(g.alpha, 0.5, EPSILON, 'alpha (sigmoid(0))');
    });

    it('should normalize non-unit quaternion', () => {
        const g = computeGaussianInverse(2, 0, 0, 0, 0, 0, 0, 0);
        assertClose(g.qw, 1, EPSILON, 'qw should be normalized');
        assertClose(g.qx, 0, EPSILON, 'qx');
    });

    it('should negate xyz components for inverse quaternion', () => {
        const s = Math.sqrt(0.5);
        const g = computeGaussianInverse(s, s, 0, 0, 0, 0, 0, 0);
        assertClose(g.qw, s, EPSILON, 'qw preserved');
        assertClose(g.qx, -s, EPSILON, 'qx negated for inverse');
        assertClose(g.qy, 0, EPSILON, 'qy');
        assertClose(g.qz, 0, EPSILON, 'qz');
    });

    it('should handle zero quaternion (degenerate)', () => {
        const g = computeGaussianInverse(0, 0, 0, 0, 0, 0, 0, 0);
        assertClose(g.qw, 0, EPSILON, 'qw zero');
        assertClose(g.qx, 0, EPSILON, 'qx zero');
        assertClose(g.qy, 0, EPSILON, 'qy zero');
        assertClose(g.qz, 0, EPSILON, 'qz zero');
    });

    it('should compute correct inverse scales', () => {
        const logScaleX = Math.log(2);
        const logScaleY = Math.log(0.5);
        const logScaleZ = Math.log(3);
        const g = computeGaussianInverse(1, 0, 0, 0, logScaleX, logScaleY, logScaleZ, 0);
        assertClose(g.isx, 0.5, EPSILON, 'isx = exp(-log(2)) = 0.5');
        assertClose(g.isy, 2, EPSILON, 'isy = exp(-log(0.5)) = 2');
        assertClose(g.isz, 1 / 3, EPSILON, 'isz = exp(-log(3)) = 1/3');
    });

    it('should handle zero log-scale (unit scale)', () => {
        const g = computeGaussianInverse(1, 0, 0, 0, 0, 0, 0, 0);
        assertClose(g.isx, 1, EPSILON, 'exp(-0) = 1');
        assertClose(g.isy, 1, EPSILON, 'exp(-0) = 1');
        assertClose(g.isz, 1, EPSILON, 'exp(-0) = 1');
    });

    it('should handle large negative log-scale (very large inverse scale)', () => {
        const g = computeGaussianInverse(1, 0, 0, 0, -10, -10, -10, 0);
        assertClose(g.isx, Math.exp(10), 1, 'large inverse scale');
        assert.ok(Number.isFinite(g.isx), 'should still be finite');
    });

    it('should compute correct alpha from opacity logit', () => {
        const g1 = computeGaussianInverse(1, 0, 0, 0, 0, 0, 0, 10);
        assert.ok(g1.alpha > 0.999, `high logit should give alpha near 1, got ${g1.alpha}`);

        const g2 = computeGaussianInverse(1, 0, 0, 0, 0, 0, 0, -10);
        assert.ok(g2.alpha < 0.001, `low logit should give alpha near 0, got ${g2.alpha}`);

        const g3 = computeGaussianInverse(1, 0, 0, 0, 0, 0, 0, 0);
        assertClose(g3.alpha, 0.5, EPSILON, 'logit 0 -> alpha 0.5');
    });
});

describe('evaluateGaussianAt', () => {
    it('should return alpha at the Gaussian center', () => {
        const g = computeGaussianInverse(1, 0, 0, 0, 0, 0, 0, 0);
        const val = evaluateGaussianAt(g, 0, 0, 0, 0, 0, 0);
        assertClose(val, 0.5, EPSILON, 'contribution at center should equal alpha');
    });

    it('should return alpha at center for non-zero position', () => {
        const g = computeGaussianInverse(1, 0, 0, 0, 0, 0, 0, 5);
        const val = evaluateGaussianAt(g, 10, 20, 30, 10, 20, 30);
        const expectedAlpha = 1 / (1 + Math.exp(-5));
        assertClose(val, expectedAlpha, EPSILON, 'contribution at center');
    });

    it('should decay with distance (isotropic)', () => {
        const g = computeGaussianInverse(1, 0, 0, 0, 0, 0, 0, 0);
        const atCenter = evaluateGaussianAt(g, 0, 0, 0, 0, 0, 0);
        const at1 = evaluateGaussianAt(g, 0, 0, 0, 1, 0, 0);
        const at2 = evaluateGaussianAt(g, 0, 0, 0, 2, 0, 0);

        assert.ok(at1 < atCenter, 'should decay from center');
        assert.ok(at2 < at1, 'should decay further');
        assert.ok(at2 > 0, 'should still be positive');
    });

    it('should be symmetric in all directions for identity rotation', () => {
        const g = computeGaussianInverse(1, 0, 0, 0, 0, 0, 0, 0);
        const atPx = evaluateGaussianAt(g, 0, 0, 0, 1, 0, 0);
        const atNx = evaluateGaussianAt(g, 0, 0, 0, -1, 0, 0);
        const atPy = evaluateGaussianAt(g, 0, 0, 0, 0, 1, 0);
        const atNy = evaluateGaussianAt(g, 0, 0, 0, 0, -1, 0);
        const atPz = evaluateGaussianAt(g, 0, 0, 0, 0, 0, 1);
        const atNz = evaluateGaussianAt(g, 0, 0, 0, 0, 0, -1);

        assertClose(atPx, atNx, EPSILON, '+x = -x');
        assertClose(atPx, atPy, EPSILON, '+x = +y');
        assertClose(atPx, atPz, EPSILON, '+x = +z');
        assertClose(atNy, atNz, EPSILON, '-y = -z');
    });

    it('should reflect anisotropy from non-uniform scale', () => {
        // Large scale in X (log(3)), small in Y (log(0.5)), unit in Z
        const g = computeGaussianInverse(1, 0, 0, 0, Math.log(3), Math.log(0.5), 0, 0);

        const atX1 = evaluateGaussianAt(g, 0, 0, 0, 1, 0, 0);
        const atY1 = evaluateGaussianAt(g, 0, 0, 0, 0, 1, 0);
        const atZ1 = evaluateGaussianAt(g, 0, 0, 0, 0, 0, 1);

        // Larger scale = wider Gaussian = higher contribution at same distance
        assert.ok(atX1 > atZ1, 'wider axis should have higher contribution');
        assert.ok(atZ1 > atY1, 'narrower axis should have lower contribution');
    });

    it('should handle rotation correctly (90 degrees around Z)', () => {
        const s = Math.sqrt(0.5);
        // 90-degree rotation around Z: qw=cos(45)=s, qz=sin(45)=s
        const g = computeGaussianInverse(s, 0, 0, s, Math.log(3), Math.log(0.5), 0, 0);

        // With rotation, the wide axis (X in local) maps to Y in world
        const atWorldX = evaluateGaussianAt(g, 0, 0, 0, 1, 0, 0);
        const atWorldY = evaluateGaussianAt(g, 0, 0, 0, 0, 1, 0);

        // After 90deg Z rotation: world X -> local -Y (narrow), world Y -> local X (wide)
        assert.ok(atWorldY > atWorldX,
            'rotated wide axis should have higher contribution');
    });

    it('should return near-zero far from center', () => {
        const g = computeGaussianInverse(1, 0, 0, 0, 0, 0, 0, 10);
        const val = evaluateGaussianAt(g, 0, 0, 0, 100, 0, 0);
        assert.ok(val < 1e-10, `far-away contribution should be near zero, got ${val}`);
    });

    it('should return 0 for zero-alpha Gaussian', () => {
        const g = computeGaussianInverse(1, 0, 0, 0, 0, 0, 0, -1000);
        const val = evaluateGaussianAt(g, 0, 0, 0, 0, 0, 0);
        assertClose(val, 0, EPSILON, 'zero-alpha Gaussian');
    });

    it('should handle degenerate zero quaternion without NaN', () => {
        const g = computeGaussianInverse(0, 0, 0, 0, 0, 0, 0, 0);
        const val = evaluateGaussianAt(g, 0, 0, 0, 1, 0, 0);
        assert.ok(Number.isFinite(val), `result should be finite, got ${val}`);
    });

    it('should handle very large scale without NaN', () => {
        const g = computeGaussianInverse(1, 0, 0, 0, 20, 20, 20, 0);
        const val = evaluateGaussianAt(g, 0, 0, 0, 1, 0, 0);
        assert.ok(Number.isFinite(val), `result should be finite, got ${val}`);
        assertClose(val, 0.5, EPSILON, 'very large scale -> point is at center relative to scale');
    });
});
