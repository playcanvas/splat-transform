/**
 * Deterministic splat data generator for testing.
 * Creates a grid of Gaussian splats with predictable values.
 *
 * This module exports a Generator class compatible with splat-transform's .mjs reader.
 */

/**
 * Generator class that creates a grid of Gaussian splats.
 * All values are deterministic based on grid position.
 */
class Generator {
    /**
     * @param {number} width - Grid width
     * @param {number} height - Grid height
     * @param {number} spacing - Distance between splats
     * @param {number} scale - Size of each splat
     */
    constructor(width, height, spacing, scale) {
        this.count = width * height;
        this.width = width;
        this.height = height;
        this.spacing = spacing;
        this.scale = scale;

        this.columnNames = [
            'x', 'y', 'z',
            'scale_0', 'scale_1', 'scale_2',
            'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
            'rot_0', 'rot_1', 'rot_2', 'rot_3'
        ];

        // Precompute constants
        const SH_C0 = 0.28209479177387814;
        this.packClr = (c) => (c - 0.5) / SH_C0;
        this.logScale = Math.log(scale);
        // Pack opacity 0.9 using inverse sigmoid
        this.packedOpacity = -Math.log(1 / 0.9 - 1);

        this.getRow = (index, row) => {
            const gx = index % this.width;
            const gz = Math.floor(index / this.width);

            // Position: centered grid
            row.x = (gx - this.width / 2) * this.spacing;
            row.y = 0;
            row.z = (gz - this.height / 2) * this.spacing;

            // Scale: uniform, log-encoded
            row.scale_0 = this.logScale;
            row.scale_1 = this.logScale;
            row.scale_2 = this.logScale;

            // Color: gradient based on position (deterministic)
            const r = (gx + 1) / (this.width + 1);
            const g = (gz + 1) / (this.height + 1);
            const b = 0.5;
            row.f_dc_0 = this.packClr(r);
            row.f_dc_1 = this.packClr(g);
            row.f_dc_2 = this.packClr(b);

            // Opacity: 0.9 (sigmoid encoded)
            row.opacity = this.packedOpacity;

            // Rotation: identity quaternion
            row.rot_0 = 0;
            row.rot_1 = 0;
            row.rot_2 = 0;
            row.rot_3 = 1;
        };
    }

    /**
     * Factory method to create a Generator from parameters.
     * @param {Array<{name: string, value: string}>} params - Parameters
     * @returns {Generator} A new Generator instance
     */
    static create(params) {
        const floatParam = (name, defaultValue) => {
            const param = params.find(p => p.name === name);
            return param ? parseFloat(param.value) : defaultValue;
        };

        const width = Math.floor(floatParam('width', 4));
        const height = Math.floor(floatParam('height', 4));
        const spacing = floatParam('spacing', 1.0);
        const scale = floatParam('scale', 0.1);

        return new Generator(width, height, spacing, scale);
    }
}

export { Generator };
