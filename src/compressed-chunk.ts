import { Quat } from 'playcanvas';

import { sigmoid } from './utils/math';

const q = new Quat();

// process and compress a chunk of 256 splats
class CompressedChunk {
    static members = [
        'x', 'y', 'z',
        'scale_0', 'scale_1', 'scale_2',
        'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
        'rot_0', 'rot_1', 'rot_2', 'rot_3'
    ];

    size: number;
    data: any = {};

    // compressed data
    chunkData: Float32Array;
    position: Uint32Array;
    rotation: Uint32Array;
    scale: Uint32Array;
    color: Uint32Array;

    constructor(size = 256) {
        this.size = size;
        CompressedChunk.members.forEach((m) => {
            this.data[m] = new Float32Array(size);
        });
        this.chunkData = new Float32Array(18);
        this.position = new Uint32Array(size);
        this.rotation = new Uint32Array(size);
        this.scale = new Uint32Array(size);
        this.color = new Uint32Array(size);
    }

    set(index: number, data: any) {
        CompressedChunk.members.forEach((m) => {
            this.data[m][index] = data[m];
        });
    }

    pack() {
        const calcMinMax = (data: Float32Array) => {
            let min;
            let max;
            min = max = data[0];
            for (let i = 1; i < data.length; ++i) {
                const v = data[i];
                min = Math.min(min, v);
                max = Math.max(max, v);
            }
            return { min, max };
        };

        const normalize = (x: number, min: number, max: number) => {
            if (x <= min) return 0;
            if (x >= max) return 1;
            return (max - min < 0.00001) ? 0 : (x - min) / (max - min);
        };

        const data = this.data;

        const x = data.x;
        const y = data.y;
        const z = data.z;
        const scale_0 = data.scale_0;
        const scale_1 = data.scale_1;
        const scale_2 = data.scale_2;
        const rot_0 = data.rot_0;
        const rot_1 = data.rot_1;
        const rot_2 = data.rot_2;
        const rot_3 = data.rot_3;
        const f_dc_0 = data.f_dc_0;
        const f_dc_1 = data.f_dc_1;
        const f_dc_2 = data.f_dc_2;
        const opacity = data.opacity;

        const px = calcMinMax(x);
        const py = calcMinMax(y);
        const pz = calcMinMax(z);

        const sx = calcMinMax(scale_0);
        const sy = calcMinMax(scale_1);
        const sz = calcMinMax(scale_2);

        // clamp scale because sometimes values are at infinity
        const clamp = (v: number, min: number, max: number) => Math.max(min, Math.min(max, v));
        sx.min = clamp(sx.min, -20, 20);
        sx.max = clamp(sx.max, -20, 20);
        sy.min = clamp(sy.min, -20, 20);
        sy.max = clamp(sy.max, -20, 20);
        sz.min = clamp(sz.min, -20, 20);
        sz.max = clamp(sz.max, -20, 20);

        // convert f_dc_ to colors before calculating min/max and packaging
        const SH_C0 = 0.28209479177387814;
        for (let i = 0; i < f_dc_0.length; ++i) {
            f_dc_0[i] = f_dc_0[i] * SH_C0 + 0.5;
            f_dc_1[i] = f_dc_1[i] * SH_C0 + 0.5;
            f_dc_2[i] = f_dc_2[i] * SH_C0 + 0.5;
        }

        const cr = calcMinMax(f_dc_0);
        const cg = calcMinMax(f_dc_1);
        const cb = calcMinMax(f_dc_2);

        const packUnorm = (value: number, bits: number) => {
            const t = (1 << bits) - 1;
            return Math.max(0, Math.min(t, Math.floor(value * t + 0.5)));
        };

        const pack111011 = (x: number, y: number, z: number) => {
            return packUnorm(x, 11) << 21 |
                   packUnorm(y, 10) << 11 |
                   packUnorm(z, 11);
        };

        const pack8888 = (x: number, y: number, z: number, w: number) => {
            return packUnorm(x, 8) << 24 |
                   packUnorm(y, 8) << 16 |
                   packUnorm(z, 8) << 8 |
                   packUnorm(w, 8);
        };

        // pack quaternion into 2,10,10,10
        const packRot = (x: number, y: number, z: number, w: number) => {
            q.set(x, y, z, w).normalize();
            const a = [q.x, q.y, q.z, q.w];
            const largest = a.reduce((curr, v, i) => (Math.abs(v) > Math.abs(a[curr]) ? i : curr), 0);

            if (a[largest] < 0) {
                a[0] = -a[0];
                a[1] = -a[1];
                a[2] = -a[2];
                a[3] = -a[3];
            }

            const norm = Math.sqrt(2) * 0.5;
            let result = largest;
            for (let i = 0; i < 4; ++i) {
                if (i !== largest) {
                    result = (result << 10) | packUnorm(a[i] * norm + 0.5, 10);
                }
            }

            return result;
        };

        // pack
        for (let i = 0; i < this.size; ++i) {
            this.position[i] = pack111011(
                normalize(x[i], px.min, px.max),
                normalize(y[i], py.min, py.max),
                normalize(z[i], pz.min, pz.max)
            );

            this.rotation[i] = packRot(rot_0[i], rot_1[i], rot_2[i], rot_3[i]);

            this.scale[i] = pack111011(
                normalize(scale_0[i], sx.min, sx.max),
                normalize(scale_1[i], sy.min, sy.max),
                normalize(scale_2[i], sz.min, sz.max)
            );

            this.color[i] = pack8888(
                normalize(f_dc_0[i], cr.min, cr.max),
                normalize(f_dc_1[i], cg.min, cg.max),
                normalize(f_dc_2[i], cb.min, cb.max),
                sigmoid(opacity[i])
            );
        }

        this.chunkData.set([
            px.min, py.min, pz.min, px.max, py.max, pz.max,
            sx.min, sy.min, sz.min, sx.max, sy.max, sz.max,
            cr.min, cg.min, cb.min, cr.max, cg.max, cb.max
        ], 0);
    }
}

export { CompressedChunk };
