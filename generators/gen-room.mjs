/**
 * Gaussian splat generator: enclosed room (6 solid walls).
 *
 * Usage:
 *   splat-transform generators/gen-room.mjs output.voxel.json [options]
 *
 * Parameters (-p key=value):
 *   size     Room dimension in world units (default: 4)
 *   spacing  Distance between splats on each wall (default: 0.04)
 *   scale    Splat radius in world units (default: 0.03)
 *   a        Opacity 0-1 (default: 1)
 *   hole     Floor hole size in world units (default: 0). When set, a tunnel
 *            of length hole*2 extends downward from the opening.
 *
 * Example with exterior fill:
 *   splat-transform generators/gen-room.mjs -p size=4,spacing=0.04 \
 *     output.voxel.json --nav-exterior-radius 0.5
 */

// Per-face layout: [uAxis, vAxis, fixedAxis, fixedSide (0=min 1=max)]
const FACES = [
    [0, 2, 1, 0], // floor:   u=x, v=z, y=0
    [0, 2, 1, 1], // ceiling: u=x, v=z, y=size
    [1, 2, 0, 0], // left:    u=y, v=z, x=0
    [1, 2, 0, 1], // right:   u=y, v=z, x=size
    [0, 1, 2, 0], // back:    u=x, v=y, z=0
    [0, 1, 2, 1]  // front:   u=x, v=y, z=size
];

// Inward-facing normals (seen from inside the room)
const NORMALS = [
    [0, 1, 0],   // floor  → up
    [0, -1, 0],  // ceiling → down
    [1, 0, 0],   // left   → right
    [-1, 0, 0],  // right  → left
    [0, 0, 1],   // back   → forward
    [0, 0, -1]   // front  → backward
];

// Base material colors per face (linear RGB, 0-1)
const COLORS = [
    [0.55, 0.38, 0.22], // floor:   warm wood
    [0.92, 0.90, 0.86], // ceiling: cream white
    [0.50, 0.58, 0.70], // left:    blue-grey
    [0.72, 0.64, 0.52], // right:   warm beige
    [0.50, 0.62, 0.48], // back:    muted green
    [0.68, 0.50, 0.52]  // front:   dusty rose
];

// 8 corners of the box, offset outward by `cornerOffset`
const CORNER_SIGNS = [
    [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
    [1, -1, -1],  [1, -1, 1],  [1, 1, -1],  [1, 1, 1]
];

// Tunnel wall definitions: [fixedAxis, fixedSign, uAxis, vAxis, normalSign]
// fixedAxis: which axis is fixed (0=x, 2=z)
// fixedSign: -1 or +1 for which side of the tunnel
// uAxis: axis spanning the tunnel width
// vAxis: always 1 (y) for the tunnel height
// inward normal component on fixedAxis
const TUNNEL_WALLS = [
    { fixedAxis: 0, fixedSign: -1, uAxis: 2, normalX:  1, normalZ:  0 }, // left:  x = -holeHalf
    { fixedAxis: 0, fixedSign:  1, uAxis: 2, normalX: -1, normalZ:  0 }, // right: x = +holeHalf
    { fixedAxis: 2, fixedSign: -1, uAxis: 0, normalX:  0, normalZ:  1 }, // back:  z = -holeHalf
    { fixedAxis: 2, fixedSign:  1, uAxis: 0, normalX:  0, normalZ: -1 }  // front: z = +holeHalf
];

const TUNNEL_COLOR = [0.6, 0.6, 0.6];

class Generator {
    constructor(size, spacing, scale, opacity, hole) {
        const stepsPerEdge = Math.ceil(size / spacing) + 1;
        const splatPerFace = stepsPerEdge * stepsPerEdge;
        const wallCount = FACES.length * splatPerFace;
        const cornerCount = CORNER_SIGNS.length;

        const half = size * 0.5;
        const holeHalf = Math.min(hole, size) * 0.5;
        const tunnelLength = hole * 2;
        const stepsPerTunnelWidth = hole > 0 ? Math.ceil(hole / spacing) + 1 : 0;
        const stepsPerTunnelHeight = hole > 0 ? Math.ceil(tunnelLength / spacing) + 1 : 0;
        const splatPerTunnelWall = stepsPerTunnelWidth * stepsPerTunnelHeight;
        const tunnelCount = 4 * splatPerTunnelWall;

        this.count = wallCount + cornerCount + tunnelCount;

        this.columnNames = [
            'x', 'y', 'z',
            'scale_0', 'scale_1', 'scale_2',
            'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
            'rot_0', 'rot_1', 'rot_2', 'rot_3'
        ];

        const SH_C0 = 0.28209479177387814;
        const packClr = (c) => (c - 0.5) / SH_C0;
        const packOpacity = (o) => (o <= 0) ? -20 : (o >= 1) ? 20 : -Math.log(1 / o - 1);

        const gs = Math.log(scale);
        const gOpacity = packOpacity(opacity);

        // Virtual point light at upper-center of room
        const lightPos = [0, size * 0.35, 0];
        const lightIntensity = 1.2;
        const ambient = 0.15;

        const cornerOffset = size * 0.5;
        const cornerScale = Math.log(scale * 3);

        const emitLitSplat = (row, pos, normal, baseColor) => {
            row.x = pos[0];
            row.y = pos[1];
            row.z = pos[2];

            const lx = lightPos[0] - pos[0];
            const ly = lightPos[1] - pos[1];
            const lz = lightPos[2] - pos[2];
            const dist = Math.sqrt(lx * lx + ly * ly + lz * lz);
            const invDist = dist > 0 ? 1 / dist : 0;
            const ndotl = Math.max(0, (normal[0] * lx + normal[1] * ly + normal[2] * lz) * invDist);
            const attenuation = 1 / (1 + 0.15 * dist * dist);
            const diffuse = ndotl * attenuation * lightIntensity;

            const lit = ambient + diffuse;
            row.f_dc_0 = packClr(Math.min(1, baseColor[0] * lit));
            row.f_dc_1 = packClr(Math.min(1, baseColor[1] * lit));
            row.f_dc_2 = packClr(Math.min(1, baseColor[2] * lit));

            row.scale_0 = gs;
            row.scale_1 = gs;
            row.scale_2 = gs;
            row.opacity = gOpacity;
            row.rot_0 = 1;
            row.rot_1 = 0;
            row.rot_2 = 0;
            row.rot_3 = 0;
        };

        this.getRow = (index, row) => {
            // Corner markers
            if (index >= wallCount && index < wallCount + cornerCount) {
                const ci = index - wallCount;
                const signs = CORNER_SIGNS[ci];
                row.x = signs[0] * (half + cornerOffset);
                row.y = signs[1] * (half + cornerOffset);
                row.z = signs[2] * (half + cornerOffset);

                row.scale_0 = cornerScale;
                row.scale_1 = cornerScale;
                row.scale_2 = cornerScale;

                row.f_dc_0 = packClr(0.9);
                row.f_dc_1 = packClr(0.2);
                row.f_dc_2 = packClr(0.2);

                row.opacity = gOpacity;
                row.rot_0 = 1;
                row.rot_1 = 0;
                row.rot_2 = 0;
                row.rot_3 = 0;
                return;
            }

            // Tunnel walls
            if (index >= wallCount + cornerCount) {
                const ti = index - wallCount - cornerCount;
                const wallIdx = Math.floor(ti / splatPerTunnelWall);
                const localIdx = ti - wallIdx * splatPerTunnelWall;
                const iu = localIdx % stepsPerTunnelWidth;
                const iv = Math.floor(localIdx / stepsPerTunnelWidth);

                const tw = TUNNEL_WALLS[wallIdx];
                const pos = [0, 0, 0];
                pos[tw.fixedAxis] = tw.fixedSign * holeHalf;
                pos[tw.uAxis] = iu * spacing - holeHalf;
                pos[1] = -half - iv * (tunnelLength / Math.max(1, stepsPerTunnelHeight - 1));

                const normal = [tw.normalX, 0, tw.normalZ];
                emitLitSplat(row, pos, normal, TUNNEL_COLOR);
                return;
            }

            // Room wall faces
            const faceIdx = Math.floor(index / splatPerFace);
            const localIdx = index - faceIdx * splatPerFace;
            const iu = localIdx % stepsPerEdge;
            const iv = Math.floor(localIdx / stepsPerEdge);

            const [uAxis, vAxis, fixedAxis, fixedSide] = FACES[faceIdx];
            const pos = [0, 0, 0];
            pos[uAxis] = iu * spacing - half;
            pos[vAxis] = iv * spacing - half;
            pos[fixedAxis] = fixedSide * size - half;

            // Floor hole: skip splats inside the hole region
            const inHole = faceIdx === 0 && holeHalf > 0 &&
                Math.abs(pos[0]) < holeHalf && Math.abs(pos[2]) < holeHalf;
            if (inHole) {
                row.scale_0 = gs;
                row.scale_1 = gs;
                row.scale_2 = gs;
                row.f_dc_0 = 0;
                row.f_dc_1 = 0;
                row.f_dc_2 = 0;
                row.opacity = packOpacity(0);
                row.rot_0 = 1;
                row.rot_1 = 0;
                row.rot_2 = 0;
                row.rot_3 = 0;
                return;
            }

            emitLitSplat(row, pos, NORMALS[faceIdx], COLORS[faceIdx]);
        };
    }

    static create(params) {
        const floatParam = (name, defaultValue) =>
            parseFloat(params.find(p => p.name === name)?.value ?? defaultValue);

        const size = floatParam('size', 4);
        const spacing = floatParam('spacing', 0.04);
        const scale = floatParam('scale', 0.03);
        const a = floatParam('a', 1.0);
        const hole = Math.max(0, floatParam('hole', 0));

        const stepsPerEdge = Math.ceil(size / spacing) + 1;
        const tunnelLength = hole * 2;
        const stepsPerTunnelWidth = hole > 0 ? Math.ceil(hole / spacing) + 1 : 0;
        const stepsPerTunnelHeight = hole > 0 ? Math.ceil(tunnelLength / spacing) + 1 : 0;
        const tunnelSplats = 4 * stepsPerTunnelWidth * stepsPerTunnelHeight;
        const total = 6 * stepsPerEdge * stepsPerEdge + 8 + tunnelSplats;
        const holeStr = hole > 0 ? `, hole=${hole}, tunnel length=${tunnelLength}` : '';
        console.log(`Generating enclosed room: size=${size} spacing=${spacing} scale=${scale} opacity=${a}${holeStr} (${total} splats, 8 corner markers)`);

        return new Generator(size, spacing, scale, a, hole);
    }
}

export { Generator };
