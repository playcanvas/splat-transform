import { Column, DataTable } from '../data-table';
import type { PlyData } from './read-ply';

// Size of a chunk in the compressed PLY format (number of splats per chunk)
const CHUNK_SIZE = 256;

const isCompressedPly = (ply: PlyData): boolean => {
    const hasShape = (elementName: string, columns: string[], type: string) => {
        const elem = ply.elements.find(e => e.name === elementName);
        if (!elem) return false;
        const dt = elem.dataTable;
        return columns.every((name) => {
            const col = dt.getColumnByName(name);
            return col && col.dataType === type;
        });
    };

    const chunkProperties = [
        'min_x',
        'min_y',
        'min_z',
        'max_x',
        'max_y',
        'max_z',
        'min_scale_x',
        'min_scale_y',
        'min_scale_z',
        'max_scale_x',
        'max_scale_y',
        'max_scale_z',
        'min_r',
        'min_g',
        'min_b',
        'max_r',
        'max_g',
        'max_b'
    ];

    const vertexProperties = [
        'packed_position',
        'packed_rotation',
        'packed_scale',
        'packed_color'
    ];

    const numElements = ply.elements.length;

    if (numElements !== 2 && numElements !== 3) return false;

    if (!hasShape('chunk', chunkProperties, 'float32')) return false;
    if (!hasShape('vertex', vertexProperties, 'uint32')) return false;

    // check optional spherical harmonics
    if (numElements === 3) {
        const elem = ply.elements.find(e => e.name === 'sh');
        if (!elem) {
            return false;
        }
        if ([9, 24, 45].indexOf(elem.dataTable.numColumns) === -1) {
            return false;
        }
        for (let i = 0; i < elem.dataTable.numColumns; ++i) {
            const col = elem.dataTable.getColumnByName(`f_rest_${i}`);
            if (!col || col.dataType !== 'uint8') {
                return false;
            }
        }
    }

    return true;
};

// Detects the compressed PLY schema and returns a decompressed DataTable, or null if not compressed.
const decompressPly = (ply: PlyData): DataTable | null => {
    const chunkData = ply.elements.find(e => e.name === 'chunk').dataTable;
    const getChunk = (name: string) => chunkData.getColumnByName(name)!.data as Float32Array;

    const vertexData = ply.elements.find(e => e.name === 'vertex').dataTable;
    const packed_position = vertexData.getColumnByName('packed_position')!.data as Uint32Array;
    const packed_rotation = vertexData.getColumnByName('packed_rotation')!.data as Uint32Array;
    const packed_scale = vertexData.getColumnByName('packed_scale')!.data as Uint32Array;
    const packed_color = vertexData.getColumnByName('packed_color')!.data as Uint32Array;

    const min_x = getChunk('min_x');
    const min_y = getChunk('min_y');
    const min_z = getChunk('min_z');
    const max_x = getChunk('max_x');
    const max_y = getChunk('max_y');
    const max_z = getChunk('max_z');
    const min_scale_x = getChunk('min_scale_x');
    const min_scale_y = getChunk('min_scale_y');
    const min_scale_z = getChunk('min_scale_z');
    const max_scale_x = getChunk('max_scale_x');
    const max_scale_y = getChunk('max_scale_y');
    const max_scale_z = getChunk('max_scale_z');
    const min_r = getChunk('min_r');
    const min_g = getChunk('min_g');
    const min_b = getChunk('min_b');
    const max_r = getChunk('max_r');
    const max_g = getChunk('max_g');
    const max_b = getChunk('max_b');

    const numSplats = vertexData.numRows;
    const numChunks = min_x.length;
    if (numChunks * CHUNK_SIZE < numSplats) {
        return null;
    }

    const columns: Column[] = [
        new Column('x', new Float32Array(numSplats)),
        new Column('y', new Float32Array(numSplats)),
        new Column('z', new Float32Array(numSplats)),
        new Column('f_dc_0', new Float32Array(numSplats)),
        new Column('f_dc_1', new Float32Array(numSplats)),
        new Column('f_dc_2', new Float32Array(numSplats)),
        new Column('opacity', new Float32Array(numSplats)),
        new Column('rot_0', new Float32Array(numSplats)),
        new Column('rot_1', new Float32Array(numSplats)),
        new Column('rot_2', new Float32Array(numSplats)),
        new Column('rot_3', new Float32Array(numSplats)),
        new Column('scale_0', new Float32Array(numSplats)),
        new Column('scale_1', new Float32Array(numSplats)),
        new Column('scale_2', new Float32Array(numSplats))
    ];

    const out = new DataTable(columns);

    const lerp = (a: number, b: number, t: number) => a * (1 - t) + b * t;
    const unpackUnorm = (value: number, bits: number) => {
        const t = (1 << bits) - 1;
        return (value & t) / t;
    };
    const unpack111011 = (value: number) => ({
        x: unpackUnorm(value >>> 21, 11),
        y: unpackUnorm(value >>> 11, 10),
        z: unpackUnorm(value, 11)
    });
    const unpack8888 = (value: number) => ({
        x: unpackUnorm(value >>> 24, 8),
        y: unpackUnorm(value >>> 16, 8),
        z: unpackUnorm(value >>> 8, 8),
        w: unpackUnorm(value, 8)
    });
    const unpackRot = (value: number) => {
        const norm = 1.0 / (Math.sqrt(2) * 0.5);
        const a = (unpackUnorm(value >>> 20, 10) - 0.5) * norm;
        const b = (unpackUnorm(value >>> 10, 10) - 0.5) * norm;
        const c = (unpackUnorm(value, 10) - 0.5) * norm;
        const m = Math.sqrt(Math.max(0, 1.0 - (a * a + b * b + c * c)));
        const which = value >>> 30;
        switch (which) {
            case 0:
                return { x: m, y: a, z: b, w: c };
            case 1:
                return { x: a, y: m, z: b, w: c };
            case 2:
                return { x: a, y: b, z: m, w: c };
            default:
                return { x: a, y: b, z: c, w: m };
        }
    };

    const SH_C0 = 0.28209479177387814;

    const ox = out.getColumnByName('x')!.data as Float32Array;
    const oy = out.getColumnByName('y')!.data as Float32Array;
    const oz = out.getColumnByName('z')!.data as Float32Array;
    const or0 = out.getColumnByName('rot_0')!.data as Float32Array;
    const or1 = out.getColumnByName('rot_1')!.data as Float32Array;
    const or2 = out.getColumnByName('rot_2')!.data as Float32Array;
    const or3 = out.getColumnByName('rot_3')!.data as Float32Array;
    const os0 = out.getColumnByName('scale_0')!.data as Float32Array;
    const os1 = out.getColumnByName('scale_1')!.data as Float32Array;
    const os2 = out.getColumnByName('scale_2')!.data as Float32Array;
    const of0 = out.getColumnByName('f_dc_0')!.data as Float32Array;
    const of1 = out.getColumnByName('f_dc_1')!.data as Float32Array;
    const of2 = out.getColumnByName('f_dc_2')!.data as Float32Array;
    const oo = out.getColumnByName('opacity')!.data as Float32Array;

    for (let i = 0; i < numSplats; ++i) {
        const ci = Math.floor(i / CHUNK_SIZE);

        const p = unpack111011(packed_position[i]);
        const r = unpackRot(packed_rotation[i]);
        const s = unpack111011(packed_scale[i]);
        const c = unpack8888(packed_color[i]);

        ox[i] = lerp(min_x[ci], max_x[ci], p.x);
        oy[i] = lerp(min_y[ci], max_y[ci], p.y);
        oz[i] = lerp(min_z[ci], max_z[ci], p.z);

        or0[i] = r.x;
        or1[i] = r.y;
        or2[i] = r.z;
        or3[i] = r.w;

        os0[i] = lerp(min_scale_x[ci], max_scale_x[ci], s.x);
        os1[i] = lerp(min_scale_y[ci], max_scale_y[ci], s.y);
        os2[i] = lerp(min_scale_z[ci], max_scale_z[ci], s.z);

        const cr = lerp(min_r[ci], max_r[ci], c.x);
        const cg = lerp(min_g[ci], max_g[ci], c.y);
        const cb = lerp(min_b[ci], max_b[ci], c.z);
        of0[i] = (cr - 0.5) / SH_C0;
        of1[i] = (cg - 0.5) / SH_C0;
        of2[i] = (cb - 0.5) / SH_C0;

        oo[i] = -Math.log(1 / c.w - 1);
    }

    // extract spherical harmonics
    const shElem = ply.elements.find(e => e.name === 'sh');
    if (shElem) {
        const shData = shElem.dataTable;
        for (let k = 0; k < shData.numColumns; ++k) {
            const col = shData.getColumn(k);
            const src = col.data as Uint8Array;
            const dst = new Float32Array(numSplats);
            for (let i = 0; i < numSplats; ++i) {
                const n = (src[i] === 0) ? 0 : (src[i] === 255) ? 1 : (src[i] + 0.5) / 256;
                dst[i] = (n - 0.5) * 8;
            }
            out.addColumn(new Column(col.name, dst));
        }
    }

    return out;
};

export { isCompressedPly, decompressPly };
