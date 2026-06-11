/**
 * Unit tests for the LCC2 reader's pure helpers.
 * Covers meta parsing (new/legacy protocols), octree LOD collection,
 * and LOD selection resolution.
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

import {
    parseLcc2Meta,
    getChildren,
    collectFileIndicesForLod,
    resolveLodSelection
} from '../src/lib/readers/read-lcc2.js';

describe('parseLcc2Meta', () => {
    it('parses the new protocol verbatim', () => {
        const meta = parseLcc2Meta(
            JSON.stringify({
                totalSplats: 100,
                lodSplats: [60, 40],
                totalLevels: 2,
                splatType: '.sog',
                root: { splatFiles: ['a.sog', 'b.sog'], data: { env: { name: 1 } } }
            }),
            'meta.lcc2'
        );

        assert.strictEqual(meta.totalSplats, 100);
        assert.deepStrictEqual(meta.lodSplats, [60, 40]);
        assert.strictEqual(meta.totalLevels, 2);
        assert.strictEqual(meta.splatType, '.sog');
        assert.deepStrictEqual(meta.splatFiles, ['a.sog', 'b.sog']);
        assert.strictEqual(meta.envFileIndex, 1);
    });

    it('maps legacy protocol fields and normalizes file names', () => {
        const meta = parseLcc2Meta(
            JSON.stringify({
                total_splats: 50,
                lod_3dgs_info: [10, 40],
                lod_level: 2,
                root: {
                    files: ['/a', '/b.sog'],
                    child_num: 1,
                    child: [{ child_num: 0, datatype: 'x' }]
                }
            }),
            'meta.lcc2'
        );

        assert.strictEqual(meta.totalSplats, 50);
        // lod_3dgs_info is reversed
        assert.deepStrictEqual(meta.lodSplats, [40, 10]);
        assert.strictEqual(meta.totalLevels, 2);
        // leading '/' dropped, '.sog' appended when missing
        assert.deepStrictEqual(meta.splatFiles, ['a.sog', 'b.sog']);
        // child_num -> childNum recursively; datatype -> dataType
        assert.strictEqual(meta.tree.childNum, 1);
        assert.strictEqual(meta.tree.child[0].childNum, 0);
        assert.strictEqual(meta.tree.child[0].dataType, 'x');
    });

    it('defaults splatType to .sog when absent', () => {
        const meta = parseLcc2Meta(
            JSON.stringify({
                totalLevels: 1,
                root: { splatFiles: ['a.sog'] }
            }),
            'meta.lcc2'
        );
        assert.strictEqual(meta.splatType, '.sog');
        assert.strictEqual(meta.envFileIndex, undefined);
    });

    it('tolerates trailing commas before braces', () => {
        const meta = parseLcc2Meta(
            '{ "totalLevels": 1, "root": { "splatFiles": ["a.sog"], } , }',
            'meta.lcc2'
        );
        assert.strictEqual(meta.totalLevels, 1);
        assert.deepStrictEqual(meta.splatFiles, ['a.sog']);
    });

    it('throws a descriptive error on invalid JSON', () => {
        assert.throws(
            () => parseLcc2Meta('{ not json', 'bad.lcc2'),
            /Failed to parse meta\.lcc2 as JSON: bad\.lcc2/
        );
    });
});

describe('getChildren', () => {
    it('returns [] when child is absent', () => {
        assert.deepStrictEqual(getChildren({}), []);
    });

    it('returns array children as-is', () => {
        const a = { id: 0 };
        const b = { id: 1 };
        assert.deepStrictEqual(getChildren({ child: [a, b] }), [a, b]);
    });

    it('returns values of an object-map child', () => {
        const a = { id: 0 };
        const b = { id: 1 };
        assert.deepStrictEqual(getChildren({ child: { 0: a, 1: b } }), [a, b]);
    });
});

describe('collectFileIndicesForLod', () => {
    // depth counts from 1 at the root's children; level = totalLevels - depth.
    // totalLevels = 2: depth 1 -> level 1, depth 2 -> level 0 (finest).
    const tree = {
        child: [
            {
                data: { '3dgs': { name: 0 } },
                child: [
                    { data: { '3dgs': { name: 1 } } },
                    { data: { '3dgs': { name: 2 } } }
                ]
            }
        ]
    };

    it('collects the deepest nodes for LOD 0', () => {
        const set = collectFileIndicesForLod(tree, 0, 2, undefined);
        assert.deepStrictEqual(
            [...set].sort((a, b) => a - b),
            [1, 2]
        );
    });

    it('collects shallower nodes for higher LOD', () => {
        const set = collectFileIndicesForLod(tree, 1, 2, undefined);
        assert.deepStrictEqual([...set], [0]);
    });

    it('skips the environment file index', () => {
        const set = collectFileIndicesForLod(tree, 0, 2, 2);
        assert.deepStrictEqual([...set], [1]);
    });

    it('handles object-map children', () => {
        const mapTree = { child: { 0: { data: { '3dgs': { name: 7 } } } } };
        const set = collectFileIndicesForLod(mapTree, 0, 1, undefined);
        assert.deepStrictEqual([...set], [7]);
    });
});

describe('resolveLodSelection', () => {
    it('returns all levels for an empty selection', () => {
        assert.deepStrictEqual(resolveLodSelection([], 3), [0, 1, 2]);
    });

    it('maps negative indices from the end', () => {
        assert.deepStrictEqual(resolveLodSelection([-1], 3), [2]);
    });

    it('filters out-of-range indices', () => {
        assert.deepStrictEqual(resolveLodSelection([0, 3, -4], 3), [0]);
    });

    it('preserves order and duplicates of the selection', () => {
        assert.deepStrictEqual(resolveLodSelection([2, 0], 3), [2, 0]);
    });
});
