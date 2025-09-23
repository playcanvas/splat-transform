class Generator {
    constructor(width, height, scale, color, alpha) {
        this.count = width * height;

        this.columnNames = [
            'x', 'y', 'z',
            'scale_0', 'scale_1', 'scale_2',
            'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
            'rot_0', 'rot_1', 'rot_2', 'rot_3'
        ];

        const SH_C0 = 0.28209479177387814;
        const invSigmoid = (opacity) => (opacity <= 0) ? -20 : (opacity >= 1) ? 20 : -Math.log(1 / opacity - 1);

        const gs = Math.log(scale);
        const gc = (color - 0.5) / SH_C0;
        const ga = invSigmoid(alpha);

        this.getRow = (index, row) =>{
            row.x = (index % width) * scale;
            row.y = 0;
            row.z = Math.floor(index / width) * scale;

            // e^x
            row.scale_0 = gs;
            row.scale_1 = gs;
            row.scale_2 = gs;

            row.f_dc_0 = gc;
            row.f_dc_1 = gc;
            row.f_dc_2 = gc;
            row.opacity = ga;

            row.rot_0 = 0;
            row.rot_1 = 0;
            row.rot_2 = 0;
            row.rot_3 = 1;
        };
    }

    static create(params) {
        const floatParam = (name, defaultValue) => parseFloat(params.find(p => p.name === name)?.value ?? defaultValue);

        const w = Math.floor(floatParam('width', 1000));
        const h = Math.floor(floatParam('height', 1000));
        const s = floatParam('scale', 1.0);
        const c = floatParam('color', 1.0);
        const a = floatParam('alpha', 1.0);

        console.log(`Generating grid ${w} x ${h} x ${s} x ${c} x ${a}`);

        return new Generator(w, h, s, c, a);
    }
};

export { Generator };
