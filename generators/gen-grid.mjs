class Generator {
    constructor(width, height, spacing, s, r, g, b, a) {
        this.count = width * height;

        this.columnNames = [
            'x', 'y', 'z',
            'scale_0', 'scale_1', 'scale_2',
            'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
            'rot_0', 'rot_1', 'rot_2', 'rot_3'
        ];

        const SH_C0 = 0.28209479177387814;
        const packClr = (c) => (c - 0.5) / SH_C0;
        const packOpacity = (opacity) => (opacity <= 0) ? -20 : (opacity >= 1) ? 20 : -Math.log(1 / opacity - 1);

        const gs = Math.log(s);      // e^x
        const gr = packClr(r);
        const gg = packClr(g);
        const gb = packClr(b);
        const ga = packOpacity(a);

        this.getRow = (index, row) =>{
            row.x = ((index % width) - width * 0.5) * spacing;
            row.y = 0;
            row.z = (Math.floor(index / width) - height * 0.5) * spacing;

            row.scale_0 = gs;
            row.scale_1 = gs;
            row.scale_2 = gs;

            row.f_dc_0 = gr;
            row.f_dc_1 = gg;
            row.f_dc_2 = gb;
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
        const spacing = floatParam('spacing', 1.0);
        const s = floatParam('scale', 0.1);
        const r = floatParam('r', 1);
        const g = floatParam('g', 1);
        const b = floatParam('b', 1);
        const a = floatParam('a', 1.0);

        console.log(`Generating grid width=${w} height=${h} spacing=${spacing} scale=${s} r=${r}, g=${g}, b=${b} alpha=${a}`);

        return new Generator(w, h, spacing, s, r, g, b, a);
    }
};

export { Generator };
