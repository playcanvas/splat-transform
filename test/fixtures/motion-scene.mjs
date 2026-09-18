/**
 * Deterministic motion-blur test scene for the .mjs reader.
 *
 * Mixes footprint sizes so blur behaviour shows at every scale: a textured
 * ground of small splats, thin tall posts, a handful of large and huge
 * ellipsoids (some semi-transparent, some rotated), and a cloud of tiny dust
 * splats. Everything derives from a seeded PRNG, so a given parameter set
 * always yields the same scene.
 *
 * Parameters (all optional, `-p name=value`):
 *   ground  grid size per side (default 100 → 100×100 ground splats)
 *   dust    number of dust splats (default 3000)
 *   seed    PRNG seed (default 1)
 */

const SH_C0 = 0.28209479177387814;
const packClr = c => (c - 0.5) / SH_C0;
const packOpacity = a => -Math.log(1 / a - 1);

// mulberry32
const makeRng = (seed) => {
    let s = seed >>> 0;
    return () => {
        s = (s + 0x6D2B79F5) >>> 0;
        let t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
};

class Generator {
    constructor(ground, dust, seed) {
        this.columnNames = [
            'x', 'y', 'z',
            'scale_0', 'scale_1', 'scale_2',
            'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
            'rot_0', 'rot_1', 'rot_2', 'rot_3'
        ];

        const rng = makeRng(seed);
        const rows = [];
        const add = (x, y, z, sx, sy, sz, r, g, b, alpha, q = [1, 0, 0, 0]) => {
            rows.push({
                x, y, z,
                scale_0: Math.log(sx), scale_1: Math.log(sy), scale_2: Math.log(sz),
                f_dc_0: packClr(r), f_dc_1: packClr(g), f_dc_2: packClr(b),
                opacity: packOpacity(alpha),
                rot_0: q[0], rot_1: q[1], rot_2: q[2], rot_3: q[3]
            });
        };

        // Ground: a 20 m checkerboard of small splats with a little per-splat
        // colour noise so streaks read against texture rather than flat tone.
        const spacing = 20 / ground;
        for (let i = 0; i < ground; i++) {
            for (let j = 0; j < ground; j++) {
                const x = (i - ground / 2 + 0.5) * spacing;
                const z = (j - ground / 2 + 0.5) * spacing;
                const check = ((Math.floor(x) + Math.floor(z)) & 1) === 0;
                const n = (rng() - 0.5) * 0.1;
                const base = check ? 0.75 : 0.35;
                add(x, 0, z, spacing * 0.35, spacing * 0.35, spacing * 0.35,
                    base + n, base + n * 0.5, base * 0.9 + n, 0.95);
            }
        }

        // Posts: thin, tall, saturated. Two rings.
        const postColours = [[0.9, 0.2, 0.2], [0.2, 0.85, 0.3], [0.25, 0.4, 0.95], [0.95, 0.85, 0.2]];
        for (let ring = 0; ring < 2; ring++) {
            const radius = ring === 0 ? 2.5 : 5;
            const count = ring === 0 ? 8 : 16;
            for (let k = 0; k < count; k++) {
                const a = (k / count) * Math.PI * 2 + ring * 0.2;
                const [r, g, b] = postColours[(k + ring) % postColours.length];
                add(Math.cos(a) * radius, 0.8, Math.sin(a) * radius, 0.04, 0.8, 0.04, r, g, b, 0.95);
            }
        }

        // Large ellipsoids: random size, height, colour and orientation, some
        // semi-transparent so energy conservation under blur is visible.
        for (let k = 0; k < 10; k++) {
            const a = rng() * Math.PI * 2;
            const radius = 1.5 + rng() * 4.5;
            const s = 0.4 + rng() * 0.8;
            const stretch = 1 + rng() * 1.5;
            const q = [rng() - 0.5, rng() - 0.5, rng() - 0.5, rng() - 0.5];
            const ql = Math.hypot(...q);
            add(Math.cos(a) * radius, 0.5 + rng() * 2.5, Math.sin(a) * radius,
                s * stretch, s, s / stretch,
                0.3 + rng() * 0.7, 0.3 + rng() * 0.7, 0.3 + rng() * 0.7,
                0.6 + rng() * 0.35, q.map(v => v / ql));
        }

        // Huge, faint, distant: whole-frame-scale footprints.
        for (let k = 0; k < 3; k++) {
            const a = (k / 3) * Math.PI * 2 + 0.7;
            const radius = 10 + k * 2;
            add(Math.cos(a) * radius, 3 + k, Math.sin(a) * radius, 2.5, 2.5, 2.5,
                0.5 + 0.15 * k, 0.6, 0.9 - 0.15 * k, 0.4);
        }

        // Dust: tiny bright points in a box above the ground.
        for (let k = 0; k < dust; k++) {
            const s = 0.015 + rng() * 0.015;
            const t = 0.7 + rng() * 0.3;
            add((rng() - 0.5) * 12, 0.2 + rng() * 4, (rng() - 0.5) * 12, s, s, s, t, t, t * 0.9, 0.9);
        }

        this.count = rows.length;
        this.getRow = (index, row) => {
            Object.assign(row, rows[index]);
        };
    }

    static create(params) {
        const num = (name, dflt) => {
            const p = params.find(p => p.name === name);
            return p ? parseFloat(p.value) : dflt;
        };
        return new Generator(Math.floor(num('ground', 100)), Math.floor(num('dust', 3000)), Math.floor(num('seed', 1)));
    }
}

export { Generator };
