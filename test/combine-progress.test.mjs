/**
 * Unit tests for CombineProgress: aggregate per-source progress callbacks
 * with engine-style total extrapolation while sources start reporting.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';

import { CombineProgress } from '../src/lib/io/read/combine-progress.js';

describe('CombineProgress', () => {

    it('passes through loaded/total for a single source', () => {
        const calls = [];
        const combine = new CombineProgress(1, (loaded, total) => calls.push({ loaded, total }));

        const a = combine.track('a');
        a(50, 200);
        a(150, 200);
        a(200, 200);

        assert.deepStrictEqual(calls, [
            { loaded: 50, total: 200 },
            { loaded: 150, total: 200 },
            { loaded: 200, total: 200 }
        ]);
    });

    it('extrapolates total when fewer sources have started than expected', () => {
        const calls = [];
        const combine = new CombineProgress(2, (loaded, total) => calls.push({ loaded, total }));

        // Only source A has reported so far. Expected count = 2, reporting = 1,
        // so total should be extrapolated as ceil(200 * 2 / 1) = 400.
        const a = combine.track('a');
        a(100, 200);

        assert.deepStrictEqual(calls, [{ loaded: 100, total: 400 }]);
    });

    it('stops extrapolating once all expected sources have reported', () => {
        const calls = [];
        const combine = new CombineProgress(2, (loaded, total) => calls.push({ loaded, total }));

        const a = combine.track('a');
        const b = combine.track('b');

        a(100, 200); // 1/2 reporting -> extrapolate to 400
        b(50, 300);  // 2/2 reporting -> sum directly: loaded 150, total 500

        assert.deepStrictEqual(calls, [
            { loaded: 100, total: 400 },
            { loaded: 150, total: 500 }
        ]);
    });

    it('treats undefined total as 0 for that source', () => {
        const calls = [];
        const combine = new CombineProgress(1, (loaded, total) => calls.push({ loaded, total }));

        const a = combine.track('a');
        a(42); // total omitted

        assert.deepStrictEqual(calls, [{ loaded: 42, total: 0 }]);
    });

    it('overwrites a sources contribution on subsequent updates', () => {
        const calls = [];
        const combine = new CombineProgress(2, (loaded, total) => calls.push({ loaded, total }));

        const a = combine.track('a');
        const b = combine.track('b');

        a(100, 200);
        b(0, 400);   // both reporting now: loaded 100, total 600
        a(150, 200); // updated A overrides previous: loaded 150, total 600
        b(400, 400); // updated B: loaded 550, total 600

        assert.deepStrictEqual(calls, [
            { loaded: 100, total: 400 }, // 1/2 reporting -> ceil(200*2/1) = 400
            { loaded: 100, total: 600 },
            { loaded: 150, total: 600 },
            { loaded: 550, total: 600 }
        ]);
    });

    it('reports loaded === total when every source has completed', () => {
        let last;
        const combine = new CombineProgress(3, (loaded, total) => {
            last = { loaded, total };
        });

        combine.track('a')(100, 100);
        combine.track('b')(200, 200);
        combine.track('c')(50, 50);

        assert.deepStrictEqual(last, { loaded: 350, total: 350 });
    });

    it('extrapolates with ceiling when partial totals do not divide evenly', () => {
        const calls = [];
        const combine = new CombineProgress(3, (loaded, total) => calls.push({ loaded, total }));

        // Two of three sources reporting with totals 100 and 101 -> sum = 201,
        // extrapolated total = ceil(201 * 3 / 2) = 302.
        combine.track('a')(0, 100);
        combine.track('b')(0, 101);

        assert.strictEqual(calls.at(-1).total, 302);
    });
});
