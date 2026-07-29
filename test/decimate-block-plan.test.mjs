import assert from 'node:assert';
import { describe, it } from 'node:test';

import { makeSyntheticSource } from './helpers/synthetic-source.mjs';

import {
    allocatePlanPrefixes,
    readBlockPlanPrefix,
    storeBlockPlan
} from '../src/lib/decimate/block-allocation.js';
import { blockPlanMergeStream } from '../src/lib/decimate/block-merge-stream.js';
import { planBlockMerges, replayBlockPlan } from '../src/lib/decimate/block-plan.js';
import { createBlockProducerSource } from '../src/lib/decimate/block-producer.js';
import { buildSplatCache, CACHE_STRIDE } from '../src/lib/decimate/edge-cost-cpu.js';
import { kdPartition } from '../src/lib/decimate/partition.js';
import { MemoryReadSource } from '../src/lib/io/read/memory-file-system.js';
import { MemoryFileSystem } from '../src/lib/io/write/memory-file-system.js';

const NIL = 0xFFFFFFFF;

const makeScratch = () => {
    const writeFs = new MemoryFileSystem();
    const removed = [];
    return {
        writeFs,
        readFs: {
            async createSource(path) {
                const bytes = writeFs.results.get(path);
                if (!bytes) throw new Error(`missing ${path}`);
                return new MemoryReadSource(bytes);
            }
        },
        scratchDir: 'scratch',
        async remove(path) {
            removed.push(path);
            writeFs.results.delete(path);
        },
        removed
    };
};

const manualPlan = (costs, pairs) => ({
    costs: Float32Array.from(costs),
    pairs: Uint32Array.from(pairs),
    frozen: 0,
    unfrozen: 0
});

const randomBlock = (seed) => {
    let t = seed >>> 0;
    const rand = () => {
        t += 0x6d2b79f5;
        let r = Math.imul(t ^ (t >>> 15), t | 1);
        r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
        return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
    const coreCount = 6;
    const n = 7;
    const D = 6;
    const pos = new Float32Array(n * 3);
    const geo = new Float32Array(n * 8);
    const color = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
        pos[i * 3] = rand() * 2;
        pos[i * 3 + 1] = rand() * 2;
        pos[i * 3 + 2] = rand() * 2;
        geo[i * 8] = 1;
        geo[i * 8 + 4] = geo[i * 8 + 5] = geo[i * 8 + 6] = Math.log(0.03 + rand() * 0.2);
        geo[i * 8 + 7] = rand() * 4 - 2;
        color[i * 3] = rand() - 0.5;
        color[i * 3 + 1] = rand() - 0.5;
        color[i * 3 + 2] = rand() - 0.5;
    }
    const cache = new Float32Array(n * CACHE_STRIDE);
    buildSplatCache({ pos, geo, color, colorDim: 3 }, cache);
    const neighbors = new Uint32Array(n * D).fill(NIL);
    for (let i = 0; i < coreCount; i++) {
        let s = 0;
        for (let j = 0; j < n; j++) {
            if (j !== i) neighbors[i * D + s++] = j;
        }
    }
    return { cache, neighbors, D, coreCount, n };
};

describe('block-local merge planning', () => {
    it('freezes an all-halo coincident pool, then merges it when both rows become core', async () => {
        const pos = new Float32Array(6);
        const geo = new Float32Array(16);
        const color = new Float32Array(6).fill(0.25);
        for (let i = 0; i < 2; i++) {
            geo[i * 8] = 1;
            geo[i * 8 + 4] = geo[i * 8 + 5] = geo[i * 8 + 6] = Math.log(0.1);
        }
        const cache = new Float32Array(2 * CACHE_STRIDE);
        buildSplatCache({ pos, geo, color, colorDim: 3 }, cache);

        const boundary = await planBlockMerges({
            splatCache: cache,
            neighbors: Uint32Array.from([1, NIL]),
            D: 1,
            coreCount: 1,
            totalCount: 2
        });
        assert.strictEqual(boundary.costs.length, 0);
        assert.strictEqual(boundary.frozen, 1);

        const interior = await planBlockMerges({
            splatCache: cache,
            neighbors: Uint32Array.from([1, 0]),
            D: 1,
            coreCount: 2,
            totalCount: 2
        });
        assert.strictEqual(interior.costs.length, 1);
        assert.deepStrictEqual(Array.from(interior.pairs), [0, 1]);
        assert.ok(interior.diagnostics.waves > 0);
        assert.ok(interior.diagnostics.refreshes >= interior.costs.length);
        assert.ok(interior.diagnostics.heapPops >= interior.costs.length);
    });

    it('dynamically freezes and unfreezes through reverse-candidate invalidation', async () => {
        const input = randomBlock(1);
        const plan = await planBlockMerges({
            splatCache: input.cache,
            neighbors: input.neighbors,
            D: input.D,
            coreCount: input.coreCount,
            totalCount: input.n
        });
        assert.ok(plan.frozen > 0, 'halo minimum freezes at least one root');
        assert.ok(plan.unfrozen > 0, 'changed referenced core makes a frozen root eligible again');
        assert.ok(plan.diagnostics.reverseInvalidations > 0);
        assert.ok(plan.diagnostics.staleHeapPops <= plan.diagnostics.heapPops);
    });

    it('lets halo rows affect eligibility but never merges, removes, or duplicates them', async () => {
        const input = randomBlock(1);
        const plan = await planBlockMerges({
            splatCache: input.cache,
            neighbors: input.neighbors,
            D: input.D,
            coreCount: input.coreCount,
            totalCount: input.n
        });
        assert.ok(Array.from(plan.pairs).every(row => row < input.coreCount), 'every committed endpoint is core');
        const replay = replayBlockPlan(input.coreCount, plan);
        assert.strictEqual(replay.removed, plan.costs.length);
        assert.strictEqual(
            new Set(replay.groupMembers).size,
            replay.groupMembers.length,
            'core group members are unique'
        );
        assert.ok(!Array.from(replay.groupMembers).includes(input.coreCount), 'halo row is absent from output groups');
    });
});

describe('block-plan prefix allocation', () => {
    it('matches restricted global greedy for non-monotonic independent block sequences and groups', async () => {
        const scratch = makeScratch();
        const local = [
            manualPlan([1, 100, 2], [0, 1, 0, 2, 0, 3]),
            manualPlan([3, 4], [0, 1, 2, 3])
        ];
        const stored = [];
        for (let i = 0; i < local.length; i++) stored.push(await storeBlockPlan(scratch, 1, i, local[i]));

        // Single restricted-global reference: only each independent block's
        // next local commit is exposed, including its prefix dependency.
        const cursor = [0, 0];
        const expected = [];
        while (expected.length < 4) {
            let block = -1;
            for (let b = 0; b < local.length; b++) {
                if (cursor[b] === local[b].costs.length) continue;
                if (block < 0 || local[b].costs[cursor[b]] < local[block].costs[cursor[block]]) block = b;
            }
            expected.push([block, cursor[block]++]);
        }

        const actual = [];
        const result = await allocatePlanPrefixes(stored, scratch, 4, (block, index) => actual.push([block, index]));
        assert.deepStrictEqual(actual, expected, 'merge sequence matches restricted global greedy');
        assert.deepStrictEqual(Array.from(result.prefixes), cursor);

        for (let b = 0; b < local.length; b++) {
            const prefix = await readBlockPlanPrefix(stored[b], scratch, result.prefixes[b]);
            const replay = replayBlockPlan(4, prefix);
            assert.strictEqual(replay.removed, result.prefixes[b]);
            assert.ok(replay.groupMembers.every(member => member < 4));
        }
    });

    it('selects every productive prefix on capacity shortfall while retaining the exact available count', async () => {
        const scratch = makeScratch();
        const plans = [
            await storeBlockPlan(scratch, 1, 0, manualPlan([5, 1], [0, 1, 0, 2])),
            await storeBlockPlan(scratch, 1, 1, manualPlan([2], [0, 1]))
        ];
        const result = await allocatePlanPrefixes(plans, scratch, 99);
        assert.strictEqual(result.removed, 3);
        assert.deepStrictEqual(Array.from(result.prefixes), [2, 1]);
    });

    it('aborts a failed plan write without publishing a partial plan', async () => {
        let aborted = false;
        const scratch = {
            writeFs: {
                createWriter() {
                    return {
                        bytesWritten: 0,
                        write() {
                            throw new Error('write failed');
                        },
                        close() {},
                        abort() {
                            aborted = true;
                        }
                    };
                },
                async mkdir() {}
            },
            readFs: { async createSource() { throw new Error('not published'); } },
            scratchDir: 'scratch'
        };
        await assert.rejects(
            storeBlockPlan(scratch, 1, 0, manualPlan([1], [0, 1])),
            /write failed/
        );
        assert.ok(aborted);
    });
});

describe('block-plan output replay', () => {
    it('moment-matches selected prefixes one core at a time and cleans plan scratch', async () => {
        const n = 24;
        const { source, pool, pos } = await makeSyntheticSource(n, 1, 29, {
            chunkSize: 5,
            extraColumns: [{ name: 'tag', type: 'uint32' }]
        });
        const partition = kdPartition(pos, 8, 1);
        const scratch = makeScratch();
        const plans = [];
        const prefixes = new Uint32Array(partition.blocks.length).fill(1);
        for (let bi = 0; bi < partition.blocks.length; bi++) {
            plans.push(await storeBlockPlan(scratch, 1, bi, manualPlan([bi + 1], [0, 1])));
        }
        const outCount = n - partition.blocks.length;
        const meta = {
            ...source.meta,
            numGaussians: outCount,
            lodCounts: [outCount],
            numChunks: [Math.ceil(outCount / source.meta.chunkSize)]
        };
        const producer = createBlockProducerSource(meta, () => blockPlanMergeStream({
            source,
            pool,
            pos,
            order: partition.order,
            blocks: partition.blocks,
            plans,
            prefixes,
            scratch
        }, source.meta.chunkSize));

        let rows = 0;
        for (let c = 0; c < meta.numChunks[0]; c++) {
            const count = Math.min(meta.chunkSize, outCount - rows);
            const position = pool.acquire('position', meta.layouts.position, count);
            const geometric = pool.acquire('geometric', meta.layouts.geometric, count);
            const color = pool.acquire('color', meta.layouts.color, count);
            const other = pool.acquire('other', meta.layouts.other, count);
            await producer.read({ chunkIndex: c, position, geometric, color, other });
            for (const value of new Float32Array(position.data, 0, count * 3)) assert.ok(Number.isFinite(value));
            for (const value of new Float32Array(geometric.data, 0, count * 8)) assert.ok(Number.isFinite(value));
            position.release();
            geometric.release();
            color.release();
            other.release();
            rows += count;
        }
        assert.strictEqual(rows, outCount);
        await producer.close();
        await source.close();
        for (const plan of plans) await scratch.remove(plan.path);
        assert.strictEqual(scratch.writeFs.results.size, 0);
        assert.strictEqual(scratch.removed.length, plans.length);
    });
});
