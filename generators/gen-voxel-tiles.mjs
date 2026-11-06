/**
 * Small public surface-collision example: a colored floor and a low wall.
 *
 * node bin/cli.mjs generators/gen-voxel-tiles.mjs scene.ply
 * node bin/cli.mjs generators/gen-voxel-tiles.mjs scene.voxel-tiles.json \
 *   --voxel-size 0.1 --voxel-tile-size 4 --voxel-tile-overlap 0.4
 *
 * The generator produces world-space Y-up data. No scene assets are required.
 */
class Generator {
    constructor() {
        this.rows = [];
        const add = (x, y, z, wall) => this.rows.push({ x, y, z, wall });
        for (let x = -4; x <= 4.001; x += 0.16) {
            // Both horizontal extents exceed the viewer's 5-unit walking threshold.
            for (let iz = 0; iz <= 38; iz++) add(x, 0, -3 + iz * 6 / 38, false);
        }
        for (let y = 0; y <= 1.601; y += 0.16) {
            for (let z = -1.6; z <= 1.601; z += 0.16) add(0.8, y, z, true);
        }
        this.count = this.rows.length;
        this.columnNames = ['x', 'y', 'z', 'scale_0', 'scale_1', 'scale_2',
            'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'rot_0', 'rot_1', 'rot_2', 'rot_3'];
    }

    getRow(index, row) {
        const point = this.rows[index];
        Object.assign(row, {
            x: point.x, y: point.y, z: point.z,
            scale_0: Math.log(0.12), scale_1: Math.log(0.12), scale_2: Math.log(0.12),
            f_dc_0: ((point.wall ? 0.85 : 0.18) - 0.5) / 0.28209479177387814,
            f_dc_1: ((point.wall ? 0.3 : 0.6) - 0.5) / 0.28209479177387814,
            f_dc_2: ((point.wall ? 0.16 : 0.72) - 0.5) / 0.28209479177387814,
            opacity: 5, rot_0: 1, rot_1: 0, rot_2: 0, rot_3: 0
        });
    }

    static create() { return new Generator(); }
}

export { Generator };
