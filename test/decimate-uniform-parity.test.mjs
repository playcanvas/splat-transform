/**
 * Output-parity guards for the `--decimate-uniform` decimator in
 * `src/lib/decimate-uniform/` — see that directory's README.
 *
 * That path is bit-for-bit output-compatible with the 3.1.6 binary, which is
 * what makes it a usable reference baseline, so "it still works" is not the
 * bar — "it still produces the same bytes" is. Equivalence was verified
 * end to end on the two study scenes (fr-sky 5.81M and fr-snow 26.1M, six
 * chained halvings, every level identical); these tests are the cheap
 * in-suite tripwire for accidental drift away from that state.
 *
 * 1. Candidates come from the exact k-NN — the property the halo collection
 *    plus verify/requery backstop exists to guarantee. Integer-only, so it is
 *    immune to cross-platform float differences.
 * 2. Candidate ids match a pinned digest. Also integer-only, and the digest
 *    was taken from the build verified byte-identical to 3.1.6.
 * 3. The GPU pass agrees with the CPU pass — the only coverage of that
 *    directory's own gpu-knn.ts and gpu-edge-cost.ts.
 */

import assert from 'node:assert';
import { createHash } from 'node:crypto';
import { after, before, describe, it } from 'node:test';

import { makeSyntheticSource } from './helpers/synthetic-source.mjs';
import { kdPartition } from '../src/lib/decimate-uniform/partition.js';
import { runPriorityPass } from '../src/lib/decimate-uniform/priority.js';

const N = 5000;
const K = 4;
const KNN_K = 16;
const SEED = 47;
// Small enough that the scene splits into several blocks, so halos, external
// rows and the verify/requery path all get exercised.
const BLOCK_SIZE = 1200;

/** Digest of `cand.idx` from the build verified output-identical to 3.1.6. */
const PINNED_IDX_DIGEST = 'aa4ffce1af6b16e0b02c46758891e078f352d98b23bab7da984adced8bca0e61';

let device = null;

before(async () => {
    try {
        const { createDevice } = await import('../src/cli/node-device.js');
        device = await createDevice();
    } catch {
        device = null;
    }
});

after(() => {
    device?.destroy?.();
});

const runLegacy = async (dev) => {
    const { source, pool, pos } = await makeSyntheticSource(N, 1, SEED, { chunkSize: 1024 });
    const { order, blocks } = kdPartition(pos, BLOCK_SIZE);
    const cand = {
        idx: new Uint32Array(N * K).fill(0xFFFFFFFF),
        cost: new Float32Array(N * K).fill(Infinity)
    };
    await runPriorityPass(
        { source, pool, pos, order, blocks, device: dev, K, k: KNN_K },
        cand
    );
    return { cand, pos };
};

// Brute-force exact k-NN sets (integer sets, no distance ties to resolve).
const exactNeighbourSets = (pos, k) => {
    const sets = [];
    const d2 = new Float64Array(N);
    const idx = new Uint32Array(N);
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            const dx = pos.x[j] - pos.x[i];
            const dy = pos.y[j] - pos.y[i];
            const dz = pos.z[j] - pos.z[i];
            d2[j] = j === i ? Infinity : dx * dx + dy * dy + dz * dz;
            idx[j] = j;
        }
        const ordered = Array.from(idx).sort((a, b) => d2[a] - d2[b]);
        sets.push(new Set(ordered.slice(0, k)));
    }
    return sets;
};

describe('uniform decimator parity', () => {
    it('draws candidates from the exact k-NN (halo + verify backstop)', async () => {
        const { cand, pos } = await runLegacy(undefined);
        const exact = exactNeighbourSets(pos, KNN_K);

        let checked = 0;
        for (let g = 0; g < N; g++) {
            for (let s = 0; s < K; s++) {
                const id = cand.idx[g * K + s];
                if (id === 0xFFFFFFFF) continue;
                assert.ok(
                    exact[g].has(id),
                    `gaussian ${g} candidate ${s} = ${id} is not among its exact ${KNN_K}-NN ` +
                    '(halo collection or verify/requery regressed)'
                );
                checked++;
            }
        }
        assert.ok(checked > N, `expected most gaussians to have candidates, checked ${checked}`);
    });

    it('candidate ids match the pinned 3.1.6-parity digest', async () => {
        const { cand } = await runLegacy(undefined);
        const digest = createHash('sha256').update(Buffer.from(cand.idx.buffer)).digest('hex');
        assert.strictEqual(
            digest, PINNED_IDX_DIGEST,
            'the uniform decimator changed its candidate selection — if that was ' +
            'deliberate, re-run the whole-scene comparison against the 3.1.6 binary and ' +
            're-baseline the `old` column before repinning (see the directory README)'
        );
    });

    it('GPU legacy pass agrees with the CPU legacy pass', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');

        const { cand: gpu } = await runLegacy(device);
        const { cand: cpu } = await runLegacy(undefined);

        let idSetAgree = 0, costAgree = 0;
        for (let g = 0; g < N; g++) {
            const gpuIds = new Set(), cpuIds = new Set();
            let rowCostsAgree = true;
            for (let s = 0; s < K; s++) {
                const cg = gpu.cost[g * K + s], cc = cpu.cost[g * K + s];
                if (Math.abs(cg - cc) > Math.max(1e-3, Math.abs(cc) * 1e-3)) rowCostsAgree = false;
                gpuIds.add(gpu.idx[g * K + s]);
                cpuIds.add(cpu.idx[g * K + s]);
            }
            if (rowCostsAgree) costAgree++;
            const inter = [...gpuIds].filter(x => cpuIds.has(x)).length;
            if (inter >= K - 1) idSetAgree++;   // allow one float-order swap at the K boundary
        }
        assert.ok(costAgree / N >= 0.99, `cost agreement ${(costAgree / N * 100).toFixed(2)}% (want >= 99%)`);
        assert.ok(idSetAgree / N >= 0.95, `candidate-id agreement ${(idSetAgree / N * 100).toFixed(2)}% (want >= 95%)`);
    });
});
