/**
 * Merge stream + block-producer tests: exact output counts, merged rows equal
 * direct n-ary mergeGroup results, survivors pass through bit-exact, and the
 * producer enforces its single-sequential-pass contract.
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

import { makeSyntheticSource } from './helpers/synthetic-source.mjs';
import { createBlockProducerSource } from '../src/lib/decimate/block-producer.js';
import { mergeStream } from '../src/lib/decimate/merge-stream.js';
import { mergeGroup, createMergeScratch } from '../src/lib/decimate/moment-match.js';
import { kdPartition } from '../src/lib/decimate/partition.js';
import { runPriorityPass } from '../src/lib/decimate/priority.js';
import { selectMerges } from '../src/lib/decimate/select.js';

describe('mergeStream', () => {
    it('emits exact count; merged rows equal direct mergeGroup; survivors pass through', async () => {
        const n = 1200, k = 16, K = 4, target = 700;
        const { source, pool, view, pos } = await makeSyntheticSource(n, 1, 13, { chunkSize: 256 });
        const { order, blocks } = kdPartition(pos, 300);
        const cand = {
            idx: new Uint32Array(n * K).fill(0xFFFFFFFF),
            cost: new Float32Array(n * K).fill(Infinity)
        };
        const ctx = { source, pool, pos, order, blocks, K, k };
        await runPriorityPass(ctx, cand);
        const sel = selectMerges(cand, n, K, n - target);
        assert.strictEqual(sel.removed, n - target);

        const nextPositions = {
            x: new Float32Array(target),
            y: new Float32Array(target),
            z: new Float32Array(target)
        };
        const rows = { pos: [], geo: [], color: [] };
        for await (const payload of mergeStream({ ...ctx, selection: sel, nextPositions }, 256)) {
            rows.pos.push(new Float32Array(payload.position));
            rows.geo.push(new Float32Array(payload.geometric));
            rows.color.push(new Float32Array(payload.color));
        }
        const emitted = rows.pos.reduce((a, p) => a + p.length / 3, 0);
        assert.strictEqual(emitted, target);

        const flatPos = Float32Array.from(rows.pos.flatMap(a => [...a]));
        const flatGeo = Float32Array.from(rows.geo.flatMap(a => [...a]));
        const flatColor = Float32Array.from(rows.color.flatMap(a => [...a]));

        // Replicate emission order and verify each row against direct computation.
        const scratch = createMergeScratch();
        const out = { pos: new Float64Array(3), geo: new Float64Array(8), color: new Float64Array(view.colorDim) };
        let row = 0, checkedMerges = 0, checkedSurvivors = 0;
        for (const b of blocks) {
            for (let i = b.start; i < b.end; i++) {
                const g = order[i];
                const mg = sel.memberGroup[g];
                if (mg !== -1 && sel.groupMin[mg] !== g) continue;
                if (mg === -1) {
                    for (let c = 0; c < 3; c++) {
                        assert.ok(Math.abs(flatPos[row * 3 + c] - view.pos[g * 3 + c]) < 1e-6, `survivor ${g} pos[${c}]`);
                    }
                    for (let c = 0; c < 8; c++) {
                        assert.strictEqual(flatGeo[row * 8 + c], view.geo[g * 8 + c], `survivor ${g} geo[${c}] bit-exact`);
                    }
                    checkedSurvivors++;
                } else {
                    const members = [...sel.groupMembers.subarray(sel.groupOffsets[mg], sel.groupOffsets[mg + 1])];
                    mergeGroup(view, members, members.length, out, scratch);
                    for (let c = 0; c < 3; c++) {
                        assert.ok(Math.abs(flatPos[row * 3 + c] - out.pos[c]) < 1e-4, `group ${mg} pos[${c}]`);
                    }
                    for (let c = 0; c < 8; c++) {
                        assert.ok(Math.abs(flatGeo[row * 8 + c] - out.geo[c]) < 1e-3, `group ${mg} geo[${c}]: ${flatGeo[row * 8 + c]} vs ${out.geo[c]}`);
                    }
                    for (let c = 0; c < view.colorDim; c++) {
                        assert.ok(Math.abs(flatColor[row * view.colorDim + c] - out.color[c]) < 1e-4, `group ${mg} color[${c}]`);
                    }
                    checkedMerges++;
                }
                // nextPositions mirrors emitted rows
                assert.strictEqual(nextPositions.x[row], flatPos[row * 3]);
                row++;
            }
        }
        assert.strictEqual(row, target);
        assert.ok(checkedMerges > 50 && checkedSurvivors > 50, `coverage: ${checkedMerges} merges, ${checkedSurvivors} survivors`);
    });

    it('carries other columns (dominant member for merges, pass-through for survivors)', async () => {
        const n = 500, k = 16, K = 4, target = 350;
        const { source, pool, pos, other, otherDim } = await makeSyntheticSource(n, 0, 31, {
            chunkSize: 128,
            extraColumns: [{ name: 'tag', type: 'uint32' }]
        });
        const { order, blocks } = kdPartition(pos, 200);
        const cand = { idx: new Uint32Array(n * K).fill(0xFFFFFFFF), cost: new Float32Array(n * K).fill(Infinity) };
        const ctx = { source, pool, pos, order, blocks, K, k };
        await runPriorityPass(ctx, cand);
        const sel = selectMerges(cand, n, K, n - target);

        const rowsOther = [];
        for await (const payload of mergeStream({ ...ctx, selection: sel }, 128)) {
            assert.ok(payload.other, 'other layer present');
            rowsOther.push(new Uint32Array(payload.other));
        }
        const flatOther = Uint32Array.from(rowsOther.flatMap(a => [...a]));
        assert.strictEqual(flatOther.length, target * otherDim);
        // survivors keep their tag verbatim
        let row = 0;
        for (const b of blocks) {
            for (let i = b.start; i < b.end; i++) {
                const g = order[i];
                const mg = sel.memberGroup[g];
                if (mg !== -1 && sel.groupMin[mg] !== g) continue;
                if (mg === -1) {
                    assert.strictEqual(flatOther[row], other[g], `survivor ${g} tag`);
                } else {
                    const members = [...sel.groupMembers.subarray(sel.groupOffsets[mg], sel.groupOffsets[mg + 1])];
                    assert.ok(members.some(m => other[m] === flatOther[row]), `group ${mg} tag from a member`);
                }
                row++;
            }
        }
    });
});

describe('createBlockProducerSource', () => {
    const makeMeta = async () => {
        const { POSITION_STRIDE, GEOMETRIC_STRIDE, colorStride, positionFields, geometricFields, colorFields } =
            await import('../src/lib/chunk/layout.js');
        const { Transform } = await import('../src/lib/utils/index.js');
        return {
            numGaussians: 2,
            numLods: 1,
            lodCounts: [2],
            chunkSize: 1,
            numChunks: [2],
            shBands: 0,
            extraColumns: [],
            transform: new Transform(),
            availableLayers: new Set(['position', 'geometric', 'color']),
            layouts: {
                position: { stride: POSITION_STRIDE, fields: positionFields() },
                geometric: { stride: GEOMETRIC_STRIDE, fields: geometricFields() },
                color: { stride: colorStride(0), fields: colorFields(0) }
            }
        };
    };

    const onePayload = () => ({
        count: 1,
        position: new Float32Array(3),
        geometric: new Float32Array(8),
        color: new Float32Array(3)
    });

    it('rejects gather reads and out-of-order chunk reads', async () => {
        const meta = await makeMeta();
        async function* gen() {
            yield onePayload();
            yield onePayload();
        }
        const src = createBlockProducerSource(meta, gen);
        await assert.rejects(
            () => src.read({ indices: new Uint32Array([0]), indexOffset: 0, count: 1 }),
            /single sequential/
        );
        await assert.rejects(() => src.read({ chunkIndex: 1 }), /single sequential/);
        await src.close();
    });

    it('serves sequential chunk reads and copies payload bytes', async () => {
        const meta = await makeMeta();
        const { createChunkDataPool } = await import('../src/lib/chunk/index.js');
        const pool = createChunkDataPool({ chunkSize: 1 });
        async function* gen() {
            const p = onePayload();
            p.position[0] = 42;
            yield p;
            yield onePayload();
        }
        const src = createBlockProducerSource(meta, gen);
        const cd = pool.acquire('position', meta.layouts.position, 1);
        await src.read({ chunkIndex: 0, position: cd });
        assert.strictEqual(new Float32Array(cd.data)[0], 42);
        cd.release();
        const cd2 = pool.acquire('position', meta.layouts.position, 1);
        await src.read({ chunkIndex: 1, position: cd2 });
        cd2.release();
        await src.close();
    });
});
