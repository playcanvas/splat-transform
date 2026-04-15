/**
 * Tests for the Transform class, transformColumns, computeWriteTransform,
 * inverse helpers, combine with different transforms, and clone behavior.
 */

import { describe, it, before } from 'node:test';
import assert from 'node:assert';

import {
    Column,
    DataTable,
    Transform,
    combine,
    processDataTable
} from '../src/lib/index.js';

import {
    transformColumns,
    computeWriteTransform
} from '../src/lib/data-table/transform.js';

import { createMinimalTestData } from './helpers/test-utils.mjs';
import { assertClose } from './helpers/summary-compare.mjs';

import { Mat4, Quat, Vec3 } from 'playcanvas';

// -- Transform class --

describe('Transform class', () => {
    it('constructor defaults to identity', async () => {
        const t = new Transform();
        assertClose(t.translation.x, 0, 1e-10, 'tx');
        assertClose(t.translation.y, 0, 1e-10, 'ty');
        assertClose(t.translation.z, 0, 1e-10, 'tz');
        assertClose(t.rotation.x, 0, 1e-10, 'rx');
        assertClose(t.rotation.y, 0, 1e-10, 'ry');
        assertClose(t.rotation.z, 0, 1e-10, 'rz');
        assertClose(t.rotation.w, 1, 1e-10, 'rw');
        assertClose(t.scale, 1, 1e-10, 'scale');
    });

    it('constructor clones inputs', async () => {
        const pos = new Vec3(1, 2, 3);
        const rot = new Quat().setFromEulerAngles(0, 90, 0);
        const t = new Transform(pos, rot, 2);

        pos.set(999, 999, 999);
        rot.set(0, 0, 0, 1);

        assertClose(t.translation.x, 1, 1e-10, 'tx');
        assertClose(t.translation.y, 2, 1e-10, 'ty');
        assertClose(t.translation.z, 3, 1e-10, 'tz');
        assertClose(t.scale, 2, 1e-10, 'scale');
    });

    it('IDENTITY is identity', async () => {
        assert.ok(Transform.IDENTITY.isIdentity(), 'IDENTITY should be identity');
    });

    it('fromEulers creates rotation-only transform', async () => {
        const t = new Transform().fromEulers(0, 0, 180);
        assert.ok(!t.isIdentity(), 'euler(0,0,180) should not be identity');
        assertClose(t.translation.x, 0, 1e-10, 'tx');
        assertClose(t.translation.y, 0, 1e-10, 'ty');
        assertClose(t.translation.z, 0, 1e-10, 'tz');
        assertClose(t.scale, 1, 1e-10, 'scale');
    });

    it('isIdentity with epsilon', async () => {
        const t = new Transform(new Vec3(1e-7, 0, 0));
        assert.ok(t.isIdentity(1e-6), 'should be identity within epsilon');
        assert.ok(!t.isIdentity(1e-8), 'should not be identity with tight epsilon');
    });

    it('clone creates independent copy', async () => {
        const t = new Transform().fromEulers(0, 0, 180);
        const c = t.clone();

        t.translation.set(999, 999, 999);

        assertClose(c.translation.x, 0, 1e-10, 'cloned tx');
        assert.ok(!c.isIdentity(), 'cloned should still have rotation');
    });

    it('invert produces correct inverse', async () => {
        const t = new Transform(new Vec3(1, 2, 3), new Quat().setFromEulerAngles(30, 45, 60), 2);
        const inv = t.clone().invert();
        const composed = t.mul(inv);
        assert.ok(composed.isIdentity(1e-5), 'T * T^-1 should be identity');
    });

    it('invert of identity is identity', async () => {
        const inv = Transform.IDENTITY.clone().invert();
        assert.ok(inv.isIdentity(), 'inverse of identity should be identity');
    });

    it('mul composes correctly', async () => {
        const translate = new Transform(new Vec3(10, 0, 0));
        const scale = new Transform(undefined, undefined, 2);

        // scale then translate: point * scale * translate? No...
        // mul(A, B) = A * B. Applied to point: A * (B * point).
        // So scale.mul(translate) applies translate first then scale.
        // We want: engine = translate * scale * raw
        // = translate.mul(scale) applied to point

        const composed = translate.mul(scale);

        // Apply to a point (1, 0, 0):
        const mat = new Mat4();
        composed.getMatrix(mat);
        const p = new Vec3(1, 0, 0);
        mat.transformPoint(p, p);

        // Expected: scale first (1*2=2), then translate (2+10=12)
        assertClose(p.x, 12, 1e-5, 'composed point x');
    });

    it('getMatrix matches TRS', async () => {
        const t = new Transform(new Vec3(1, 2, 3), new Quat().setFromEulerAngles(0, 90, 0), 2);
        const mat = new Mat4();
        t.getMatrix(mat);

        const expected = new Mat4().setTRS(
            new Vec3(1, 2, 3),
            new Quat().setFromEulerAngles(0, 90, 0),
            new Vec3(2, 2, 2)
        );

        for (let i = 0; i < 16; i++) {
            assertClose(mat.data[i], expected.data[i], 1e-5, `mat[${i}]`);
        }
    });

    it('euler(0,0,180) is self-inverse', async () => {
        const t = new Transform().fromEulers(0, 0, 180);
        const inv = t.clone().invert();
        const composed = t.clone().mul(inv);
        assert.ok(composed.isIdentity(1e-5), 'euler(0,0,180) * inverse should be identity');

        // Also: applying euler(0,0,180) twice should be identity
        const doubled = t.clone().mul(t);
        assert.ok(doubled.isIdentity(1e-5), 'euler(0,0,180)^2 should be identity');
    });
});

// -- computeWriteTransform --

describe('computeWriteTransform', () => {
    it('same-to-same returns null (identity delta)', async () => {
        const result = computeWriteTransform(Transform.PLY, Transform.PLY);
        assert.strictEqual(result, null, 'same-to-same delta should be null');
    });

    it('PLY-to-engine returns non-null (needs transform)', async () => {
        const result = computeWriteTransform(Transform.PLY, Transform.IDENTITY);
        assert.notStrictEqual(result, null, 'PLY-to-engine delta should not be null');
    });

    it('identity-to-identity returns null', async () => {
        const result = computeWriteTransform(Transform.IDENTITY, Transform.IDENTITY);
        assert.strictEqual(result, null, 'identity-to-identity should be null');
    });
});

// -- transformColumns --

describe('transformColumns', () => {
    let testData;

    before(() => {
        testData = createMinimalTestData();
    });

    it('returns original arrays when delta is null', async () => {
        const cols = transformColumns(testData, ['x', 'y', 'z'], null);
        assert.strictEqual(cols.get('x'), testData.getColumnByName('x').data, 'should be same reference');
        assert.strictEqual(cols.get('y'), testData.getColumnByName('y').data, 'should be same reference');
    });

    it('returns original arrays when delta is identity', async () => {
        const cols = transformColumns(testData, ['x', 'y', 'z'], Transform.IDENTITY);
        assert.strictEqual(cols.get('x'), testData.getColumnByName('x').data, 'should be same reference');
    });

    it('transforms positions with PLY transform', async () => {
        const cols = transformColumns(testData, ['x', 'y', 'z'], Transform.PLY);
        const rawX = testData.getColumnByName('x').data;
        const rawY = testData.getColumnByName('y').data;
        const rawZ = testData.getColumnByName('z').data;

        // euler(0,0,180) negates x and y
        for (let i = 0; i < testData.numRows; i++) {
            assertClose(cols.get('x')[i], -rawX[i], 1e-5, `x[${i}]`);
            assertClose(cols.get('y')[i], -rawY[i], 1e-5, `y[${i}]`);
            assertClose(cols.get('z')[i], rawZ[i], 1e-5, `z[${i}]`);
        }
    });

    it('returns original arrays for unaffected columns', async () => {
        const cols = transformColumns(testData, ['opacity', 'f_dc_0'], Transform.PLY);
        assert.strictEqual(cols.get('opacity'), testData.getColumnByName('opacity').data, 'opacity should be same reference');
        assert.strictEqual(cols.get('f_dc_0'), testData.getColumnByName('f_dc_0').data, 'f_dc_0 should be same reference');
    });

    it('transforms rotation columns', async () => {
        const cols = transformColumns(testData, ['rot_0', 'rot_1', 'rot_2', 'rot_3'], Transform.PLY);

        // Should return new arrays (not same reference)
        assert.notStrictEqual(cols.get('rot_0'), testData.getColumnByName('rot_0').data, 'rot_0 should be new array');
    });

    it('transforms scale columns when scale != 1', async () => {
        const delta = new Transform(undefined, undefined, 2);
        const cols = transformColumns(testData, ['scale_0', 'scale_1', 'scale_2'], delta);
        const rawScale0 = testData.getColumnByName('scale_0').data;
        const logS = Math.log(2);

        for (let i = 0; i < testData.numRows; i++) {
            assertClose(cols.get('scale_0')[i], rawScale0[i] + logS, 1e-5, `scale_0[${i}]`);
        }
    });

    it('does not transform scale columns when scale == 1', async () => {
        const cols = transformColumns(testData, ['scale_0'], Transform.PLY);
        assert.strictEqual(cols.get('scale_0'), testData.getColumnByName('scale_0').data, 'scale_0 should be same reference when scale=1');
    });
});

// -- Transform.transformPoint --

describe('Transform.transformPoint', () => {
    it('with identity transform is no-op', async () => {
        const point = new Vec3(1, 2, 3);
        Transform.IDENTITY.transformPoint(point, point);
        assertClose(point.x, 1, 1e-10, 'x');
        assertClose(point.y, 2, 1e-10, 'y');
        assertClose(point.z, 3, 1e-10, 'z');
    });

    it('inverse of PLY transform negates x and y', async () => {
        const point = new Vec3(1, 2, 3);
        Transform.PLY.clone().invert().transformPoint(point, point);
        assertClose(point.x, -1, 1e-5, 'x');
        assertClose(point.y, -2, 1e-5, 'y');
        assertClose(point.z, 3, 1e-5, 'z');
    });
});

// -- DataTable transform --

describe('DataTable transform', () => {
    it('defaults to identity', async () => {
        const dt = new DataTable([new Column('x', new Float32Array([1]))]);
        assert.ok(dt.transform.isIdentity(), 'default should be identity');
    });

    it('clone preserves transform', async () => {
        const dt = createMinimalTestData();
        dt.transform = Transform.PLY.clone();

        const cloned = dt.clone();
        assert.ok(!cloned.transform.isIdentity(), 'cloned should have non-identity transform');

        // Verify it's a deep copy
        dt.transform.translation.set(999, 999, 999);
        assertClose(cloned.transform.translation.x, 0, 1e-10, 'cloned should be independent');
    });

    it('clone with row selection preserves transform', async () => {
        const dt = createMinimalTestData();
        dt.transform = new Transform().fromEulers(90, 0, 180);

        const cloned = dt.clone({ rows: [0, 1, 2] });
        assert.ok(!cloned.transform.isIdentity(), 'row-subset clone should have non-identity transform');
        assert.strictEqual(cloned.numRows, 3);
    });
});

// -- combine with different transforms --

describe('combine with transforms', () => {
    it('preserves transform when all tables match', async () => {
        const dt1 = createMinimalTestData();
        const dt2 = createMinimalTestData();
        dt1.transform = Transform.PLY;
        dt2.transform = Transform.PLY;

        const result = combine([dt1, dt2]);
        assert.ok(!result.transform.isIdentity(), 'combined should keep shared transform');
        assert.strictEqual(result.numRows, dt1.numRows + dt2.numRows);
    });

    it('converts to engine space when transforms differ', async () => {
        const dt1 = createMinimalTestData();
        const dt2 = createMinimalTestData();
        dt1.transform = Transform.PLY;
        dt2.transform = Transform.IDENTITY;

        const result = combine([dt1, dt2]);
        assert.ok(result.transform.isIdentity(), 'combined should have identity transform');
        assert.strictEqual(result.numRows, dt1.numRows + dt2.numRows);
    });
});

// -- processDataTable spatial filters with transform --

describe('processDataTable spatial filters with transform', () => {
    it('filterBox works correctly with non-identity transform', async () => {
        const dt = createMinimalTestData();
        dt.transform = Transform.PLY;

        // In engine space, euler(0,0,180) negates x and y.
        // Raw x values are centered around 0 (range approx -1.5 to 1.5).
        // Engine x = -raw_x. So filtering engine x >= 0 keeps raw x <= 0.
        const result = await processDataTable(dt, [{
            kind: 'filterBox',
            min: new Vec3(0, -1e6, -1e6),
            max: new Vec3(1e6, 1e6, 1e6)
        }]);

        assert.ok(result.numRows > 0, 'should have some rows');
        assert.ok(result.numRows < dt.numRows, 'should have fewer rows');

        // Raw x values should all be <= 0 (since engine x >= 0 means raw x <= 0)
        const xCol = result.getColumnByName('x').data;
        for (let i = 0; i < result.numRows; i++) {
            assert.ok(xCol[i] <= 1e-5, `raw x[${i}] should be <= 0, got ${xCol[i]}`);
        }
    });

    it('filterSphere works correctly with non-identity transform', async () => {
        const dt = createMinimalTestData();
        dt.transform = Transform.PLY;

        // Engine (0,0,0) maps to raw (0,0,0) for euler(0,0,180)
        const result = await processDataTable(dt, [{
            kind: 'filterSphere',
            center: new Vec3(0, 0, 0),
            radius: 1.0
        }]);

        assert.ok(result.numRows > 0, 'should have some rows');

        // Remaining splats should be within radius 1 of raw origin
        const xCol = result.getColumnByName('x').data;
        const yCol = result.getColumnByName('y').data;
        const zCol = result.getColumnByName('z').data;
        for (let i = 0; i < result.numRows; i++) {
            const distSq = xCol[i] ** 2 + yCol[i] ** 2 + zCol[i] ** 2;
            assert.ok(distSq < 1.0 + 1e-5, `splat ${i} should be within radius`);
        }
    });

    it('filterBands preserves transform', async () => {
        const dt = createMinimalTestData({ includeSH: true, shBands: 2 });
        dt.transform = Transform.PLY;

        const result = await processDataTable(dt, [{
            kind: 'filterBands',
            value: 1
        }]);

        assert.ok(!result.transform.isIdentity(), 'should preserve non-identity transform');
    });
});

// -- Round-trip transform verification --

describe('Round-trip transform scenarios', () => {
    it('PLY round-trip: computeWriteTransform returns null', async () => {
        const delta = computeWriteTransform(Transform.PLY, Transform.PLY);
        assert.strictEqual(delta, null, 'PLY->PLY write should need no transform');
    });

    it('PLY->engine: data is correctly transformed to engine space', async () => {
        const dt = createMinimalTestData();
        dt.transform = Transform.PLY;

        const delta = computeWriteTransform(dt.transform, Transform.IDENTITY);
        assert.notStrictEqual(delta, null, 'should need a transform');

        const cols = transformColumns(dt, ['x', 'y', 'z'], delta);
        const rawX = dt.getColumnByName('x').data;
        const rawY = dt.getColumnByName('y').data;

        // euler(0,0,180) negates x and y
        for (let i = 0; i < dt.numRows; i++) {
            assertClose(cols.get('x')[i], -rawX[i], 1e-5, `engine x[${i}]`);
            assertClose(cols.get('y')[i], -rawY[i], 1e-5, `engine y[${i}]`);
        }
    });

    it('engine -> PLY: data is correctly transformed to PLY space', async () => {
        const dt = createMinimalTestData();
        dt.transform = Transform.IDENTITY;

        const delta = computeWriteTransform(dt.transform, Transform.PLY);
        assert.notStrictEqual(delta, null, 'should need a transform');

        const cols = transformColumns(dt, ['x', 'y', 'z'], delta);
        const rawX = dt.getColumnByName('x').data;
        const rawY = dt.getColumnByName('y').data;

        // inverse(euler(0,0,180)) * identity = euler(0,0,180)^-1 = euler(0,0,180)
        // So it also negates x and y
        for (let i = 0; i < dt.numRows; i++) {
            assertClose(cols.get('x')[i], -rawX[i], 1e-5, `ply x[${i}]`);
            assertClose(cols.get('y')[i], -rawY[i], 1e-5, `ply y[${i}]`);
        }
    });
});
