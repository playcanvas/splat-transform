/**
 * Single source of truth for the render-golden cases.
 *
 * Each entry drives both `render-golden.test.mjs` (compares CLI output to
 * a golden file) and `render-golden.regenerate.mjs` (rewrites the golden
 * file from the current CLI behaviour). Adding a case = adding it here.
 *
 *   name        Short identifier, used in test description and filename.
 *   args        Argv slice passed to the CLI before the output path.
 *               Excludes the output path itself and `-w` (added by the
 *               test/regenerator).
 *   goldenPath  Path to the golden .webp, relative to the `test/` dir.
 */

const CASES = [
    {
        // ~400 splats, single tile group (320×240 = 20×15 tiles, 1 group).
        // Primary regression target: catches any pipeline change that
        // alters pixel output on a "happy path" scene.
        name: 'tiny',
        args: [
            'test/fixtures/generator.mjs',
            '-p', 'width=20,height=20,spacing=1.0,scale=0.1',
            '--camera', '0,5,-8',
            '--look-at', '0,0,0',
            '--fov', '60',
            '--resolution', '320x240'
        ],
        goldenPath: 'fixtures/golden-render/tiny.webp'
    },
    {
        // ~48 K splats, multiple tile groups (640×360 = 40×23 tiles → 5×3
        // groups under the current renderer). Exercises per-group BVH
        // cull, depth sort, and chunk packing on a non-trivial scene.
        name: 'mid',
        args: [
            'test/fixtures/generator.mjs',
            '-p', 'width=220,height=220,spacing=0.3,scale=0.05',
            '--camera', '0,10,-15',
            '--look-at', '0,0,0',
            '--fov', '60',
            '--resolution', '640x360'
        ],
        goldenPath: 'fixtures/golden-render/mid.webp'
    },
    {
        // 'tiny' scene + DoF. Exercises the DoF code paths end-to-end:
        // CoC uniform plumbing, per-splat covariance dilation, energy-
        // preserving alpha rescale, and default focus-distance derivation
        // (look-at distance, since --focus-distance is omitted).
        // sensor-size scales f-stop into the unitless world the synthetic
        // scene lives in so the blur is clearly visible (~3 px CoC across
        // the grid's depth range).
        name: 'tiny-dof',
        args: [
            'test/fixtures/generator.mjs',
            '-p', 'width=20,height=20,spacing=1.0,scale=0.1',
            '--camera', '0,5,-8',
            '--look-at', '0,0,0',
            '--fov', '60',
            '--resolution', '320x240',
            '--f-stop', '2.8',
            '--sensor-size', '0.5'
        ],
        goldenPath: 'fixtures/golden-render/tiny-dof.webp'
    }
];

export { CASES };
