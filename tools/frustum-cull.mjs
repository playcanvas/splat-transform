#!/usr/bin/env node
/**
 * Frustum-cull a binary splat PLY: keep only gaussians whose center (with a
 * per-splat 2σ margin) lies inside the view frustum of the given render
 * camera. Rows are passed through byte-for-byte (single streaming pass), so
 * the output is the same PLY with fewer vertices.
 *
 * The camera is specified in RENDER space (the coordinates used by the CLI's
 * --camera-pos/--camera-target). Raw PLY positions map to render space via
 * Transform.PLY = rot-z 180°: (x, y, z) → (−x, −y, z).
 *
 * Usage:
 *   node tools/frustum-cull.mjs --input in.ply --output out.ply \
 *     --camera-pos x,y,z --camera-target x,y,z \
 *     [--fov 60] [--aspect 1.7778] [--widen-deg 12] [--near 0.2]
 */
import { openSync, readSync, writeSync, closeSync } from 'node:fs';

const argv = process.argv.slice(2);
const argValue = (name, dflt) => {
    const i = argv.indexOf(name);
    return i >= 0 && i + 1 < argv.length ? argv[i + 1] : dflt;
};
const vec = (s) => s.split(',').map(Number);

const input = argValue('--input');
const output = argValue('--output');
const P = vec(argValue('--camera-pos'));
const T = vec(argValue('--camera-target'));
const fovY = (parseFloat(argValue('--fov', '60')) * Math.PI) / 180;
const aspect = parseFloat(argValue('--aspect', String(1280 / 720)));
const widen = (parseFloat(argValue('--widen-deg', '12')) * Math.PI) / 180;
const near = parseFloat(argValue('--near', '0.2'));
if (!input || !output || P.length !== 3 || T.length !== 3) {
    console.error('usage: frustum-cull.mjs --input in.ply --output out.ply --camera-pos x,y,z --camera-target x,y,z [--fov 60] [--aspect 1.7778] [--widen-deg 12] [--near 0.2]');
    process.exit(1);
}

// Camera basis in render space (up = +y, standard lookAt).
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const norm = (a) => { const l = Math.hypot(...a); return [a[0] / l, a[1] / l, a[2] / l]; };
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

const fwd = norm(sub(T, P));
let right = cross(fwd, [0, 1, 0]);
const rl = Math.hypot(...right);
right = rl > 1e-6 ? [right[0] / rl, right[1] / rl, right[2] / rl] : [1, 0, 0];
const up = cross(right, fwd);

const tanY = Math.tan(fovY / 2 + widen);
const tanX = Math.tan(Math.atan(Math.tan(fovY / 2) * aspect) + widen);
const slopeNormX = Math.sqrt(1 + tanX * tanX);
const slopeNormY = Math.sqrt(1 + tanY * tanY);

// ---- Parse header.
const fdIn = openSync(input, 'r');
const head = Buffer.alloc(65536);
readSync(fdIn, head, 0, head.length, 0);
const headText = head.toString('latin1');
const endIdx = headText.indexOf('end_header\n');
if (endIdx < 0) throw new Error('no end_header');
const headerLen = endIdx + 'end_header\n'.length;
const lines = headText.slice(0, endIdx).split('\n');
let count = 0;
const props = [];
for (const l of lines) {
    const mv = /^element vertex (\d+)/.exec(l);
    if (mv) count = parseInt(mv[1], 10);
    const mp = /^property (\w+) (\S+)/.exec(l);
    if (mp && count > 0) props.push({ type: mp[1], name: mp[2] });
}
const typeSize = { float: 4, float32: 4, double: 8, uchar: 1, uint8: 1, int: 4, uint: 4, uint32: 4, short: 2, ushort: 2 };
let stride = 0;
const off = {};
for (const p of props) {
    off[p.name] = stride;
    if (typeSize[p.type] !== 4 || (p.type !== 'float' && p.type !== 'float32')) {
        if (['x', 'y', 'z', 'scale_0', 'scale_1', 'scale_2'].includes(p.name)) {
            throw new Error(`property ${p.name} must be float32 (got ${p.type})`);
        }
    }
    stride += typeSize[p.type];
}
for (const n of ['x', 'y', 'z', 'scale_0', 'scale_1', 'scale_2']) {
    if (!(n in off)) throw new Error(`missing property ${n}`);
}
console.log(`${input}: ${count} vertices, stride ${stride}`);

// ---- Stream, test, write survivors.
const ROWS = 1 << 16;
const inBuf = Buffer.alloc(ROWS * stride);
const outBuf = Buffer.alloc(ROWS * stride);

// First pass counts survivors while writing rows to a temp offset — instead,
// write rows after a placeholder header, then rewrite the header space with
// exact padding. Simpler: two passes would re-read 3GB; instead write header
// with the count later using a fixed-width count field.
const countField = String(count); // survivors <= count, pad to same width
const headerOut = headText.slice(0, endIdx).replace(/^element vertex \d+$/m, `element vertex COUNT_PLACEHOLDER`) + 'end_header\n';

const fdOut = openSync(output, 'w');
// Reserve header space: replace placeholder with padded count at the end.
const headerTemplate = headerOut.replace('COUNT_PLACEHOLDER', countField); // max width
writeSync(fdOut, Buffer.from(headerTemplate, 'latin1'));
const headerOutLen = Buffer.byteLength(headerTemplate, 'latin1');

let kept = 0, read = 0, outRows = 0;
let inPos = headerLen;
while (read < count) {
    const rows = Math.min(ROWS, count - read);
    const bytes = rows * stride;
    let got = 0;
    while (got < bytes) {
        const n = readSync(fdIn, inBuf, got, bytes - got, inPos + got);
        if (n <= 0) throw new Error('short read');
        got += n;
    }
    inPos += bytes;
    outRows = 0;
    for (let r = 0; r < rows; r++) {
        const base = r * stride;
        // Raw → render space: rot-z 180°.
        const qx = -inBuf.readFloatLE(base + off.x);
        const qy = -inBuf.readFloatLE(base + off.y);
        const qz = inBuf.readFloatLE(base + off.z);
        const s0 = inBuf.readFloatLE(base + off.scale_0);
        const s1 = inBuf.readFloatLE(base + off.scale_1);
        const s2 = inBuf.readFloatLE(base + off.scale_2);
        const sigma = Math.exp(Math.max(s0, s1, s2));
        const margin = 2 * (Number.isFinite(sigma) ? sigma : 0);

        const dx = qx - P[0], dy = qy - P[1], dz = qz - P[2];
        const zc = dx * fwd[0] + dy * fwd[1] + dz * fwd[2];
        if (zc < near - margin) continue;
        const xc = dx * right[0] + dy * right[1] + dz * right[2];
        if (Math.abs(xc) > zc * tanX + margin * slopeNormX) continue;
        const yc = dx * up[0] + dy * up[1] + dz * up[2];
        if (Math.abs(yc) > zc * tanY + margin * slopeNormY) continue;

        inBuf.copy(outBuf, outRows * stride, base, base + stride);
        outRows++;
    }
    if (outRows > 0) writeSync(fdOut, outBuf, 0, outRows * stride);
    kept += outRows;
    read += rows;
    if (read % (1 << 22) < ROWS) console.log(`  ${read}/${count} scanned, ${kept} kept`);
}
closeSync(fdIn);

// Rewrite header with the real count, zero-padded to the reserved width
// (leading zeros parse cleanly everywhere; trailing spaces may not).
const headerFinal = headerTemplate.replace(
    `element vertex ${countField}`,
    `element vertex ${String(kept).padStart(countField.length, '0')}`
);
if (Buffer.byteLength(headerFinal, 'latin1') !== headerOutLen) throw new Error('header size drift');
writeSync(fdOut, Buffer.from(headerFinal, 'latin1'), 0, headerOutLen, 0);
closeSync(fdOut);
console.log(`${output}: kept ${kept} / ${count} (${((kept / count) * 100).toFixed(1)}%)`);
