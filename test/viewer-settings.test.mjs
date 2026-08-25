/**
 * Packaging tests for the viewer-settings subpath re-export.
 */

import assert from 'node:assert';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';

import * as viewer from '@playcanvas/supersplat-viewer/settings';

// self-referencing import: resolves through this package's exports map, as a consumer would
import * as reexport from '@playcanvas/splat-transform/viewer-settings';

const __dirname = dirname(fileURLToPath(import.meta.url));

describe('viewer-settings subpath', () => {
    it('re-exports the complete viewer settings api', () => {
        for (const key of Object.keys(viewer)) {
            assert.ok(key in reexport, `missing export: ${key}`);
        }
    });

    it('agrees with the viewer on shared defaults', () => {
        assert.deepStrictEqual(reexport.defaultSettings(), viewer.defaultSettings());
        assert.deepStrictEqual(reexport.defaultPostEffectSettings(), viewer.defaultPostEffectSettings());
    });

    it('ships a flattened declaration next to the bundle', () => {
        assert.ok(existsSync(join(__dirname, '..', 'dist', 'viewer-settings.d.ts')),
            'dist/viewer-settings.d.ts should be copied by the build');
    });
});
