/**
 * Tests for k-means clustering (kmeansInterleaved + GpuKmeans).
 *
 * Two tiers:
 * - CPU-path tests always run (no device passed — kd-tree assign fallback).
 * - GPU tests run against the real flash-kmeans pipeline via a headless Dawn
 *   device and skip when no WebGPU adapter is available.
 *
 * k-means converges to local minima that depend on the random seeding, so
 * cross-path assertions use fixed initial centroids and compare against an
 * in-test reference Lloyd loop, or assert seeding-independent invariants
 * (labels are argmin of the returned centroids; centroids are the means of
 * their assigned points).
 *
 * The device is created in a `before` hook (not top-level await) so test
 * registration completes immediately; `--test-force-exit` in the npm test
 * script reaps Dawn's async runner, which otherwise keeps the child process
 * alive after the suites finish.
 */

import { after, before, describe, it } from 'node:test';
import assert from 'node:assert';

import { kmeansInterleaved, kmeansToColorError, shColorError } from '../src/lib/spatial/k-means.js';
import { GpuKmeans } from '../src/lib/gpu/gpu-kmeans.js';

// deterministic rng for reproducible data
const mulberry32 = (seed) => {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6D2B79F5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
};

// interleaved blobs around the given centers; point i belongs to blob
// i % centers.length, so points[0..k) hold one point of every blob
const makeBlobs = (centers, pointsPerBlob, numColumns, sigma, rand) => {
    const numRows = centers.length * pointsPerBlob;
    const points = new Float32Array(numRows * numColumns);
    for (let i = 0; i < numRows; i++) {
        const c = centers[i % centers.length];
        for (let j = 0; j < numColumns; j++) {
            points[i * numColumns + j] = c[j] + (rand() * 2 - 1) * sigma;
        }
    }
    return { points, numRows };
};

// sum over points of squared distance to the nearest centroid
const inertia = (points, numRows, numColumns, centroids, k) => {
    let total = 0;
    for (let i = 0; i < numRows; i++) {
        let best = Infinity;
        for (let c = 0; c < k; c++) {
            let d = 0;
            for (let j = 0; j < numColumns; j++) {
                const v = points[i * numColumns + j] - centroids[c * numColumns + j];
                d += v * v;
            }
            best = Math.min(best, d);
        }
        total += best;
    }
    return total;
};

// brute-force argmin labels against fixed centroids
const bruteForceLabels = (points, numRows, numColumns, centroids, k) => {
    const labels = new Uint32Array(numRows);
    for (let i = 0; i < numRows; i++) {
        let best = Infinity;
        let bestC = 0;
        for (let c = 0; c < k; c++) {
            let d = 0;
            for (let j = 0; j < numColumns; j++) {
                const v = points[i * numColumns + j] - centroids[c * numColumns + j];
                d += v * v;
            }
            if (d < best) {
                best = d;
                bestC = c;
            }
        }
        labels[i] = bestC;
    }
    return labels;
};

// reference Lloyd loop matching the implementation's semantics: T × (assign →
// update), returning the last assign's labels and the post-update centroids.
// No empty-cluster handling — callers seed one centroid per blob.
const referenceLloyd = (points, numRows, numColumns, centroids, k, iterations) => {
    let labels;
    for (let t = 0; t < iterations; t++) {
        labels = bruteForceLabels(points, numRows, numColumns, centroids, k);
        const sums = new Float64Array(k * numColumns);
        const counts = new Uint32Array(k);
        for (let i = 0; i < numRows; i++) {
            counts[labels[i]]++;
            for (let j = 0; j < numColumns; j++) {
                sums[labels[i] * numColumns + j] += points[i * numColumns + j];
            }
        }
        for (let c = 0; c < k; c++) {
            if (counts[c] > 0) {
                for (let j = 0; j < numColumns; j++) {
                    centroids[c * numColumns + j] = sums[c * numColumns + j] / counts[c];
                }
            }
        }
    }
    return labels;
};

// Lloyd invariants that hold at any local minimum: every point is assigned to
// its nearest centroid (labels precede the final update, so tolerate a small
// drift) and every non-empty centroid is the mean of its assigned points.
const assertLloydInvariants = (points, numRows, numColumns, centroids, labels, k) => {
    const expected = bruteForceLabels(points, numRows, numColumns, centroids, k);
    let mismatches = 0;
    for (let i = 0; i < numRows; i++) {
        if (labels[i] !== expected[i]) mismatches++;
    }
    assert.ok(mismatches / numRows < 0.01, `${mismatches}/${numRows} labels not nearest their centroid`);

    const sums = new Float64Array(k * numColumns);
    const counts = new Uint32Array(k);
    for (let i = 0; i < numRows; i++) {
        counts[labels[i]]++;
        for (let j = 0; j < numColumns; j++) {
            sums[labels[i] * numColumns + j] += points[i * numColumns + j];
        }
    }
    for (let c = 0; c < k; c++) {
        if (counts[c] === 0) continue;
        for (let j = 0; j < numColumns; j++) {
            const mean = sums[c * numColumns + j] / counts[c];
            assert.ok(Math.abs(centroids[c * numColumns + j] - mean) < 0.05,
                `centroid ${c}[${j}] = ${centroids[c * numColumns + j]} differs from cluster mean ${mean}`);
        }
    }
};

describe('kmeansInterleaved (CPU path)', () => {
    it('returns each point as its own centroid when numRows < k', async () => {
        const points = new Float32Array([1, 2, 3, 4, 5, 6]);
        const { centroids, labels } = await kmeansInterleaved(points, 3, 2, 8, 4);
        assert.deepStrictEqual([...centroids], [1, 2, 3, 4, 5, 6]);
        assert.deepStrictEqual([...labels], [0, 1, 2]);
    });

    it('converges to a valid Lloyd fixed point on blob data', async () => {
        const rand = mulberry32(1234);
        const centers = [[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]];
        const { points, numRows } = makeBlobs(centers, 50, 3, 0.5, rand);
        const k = 4;
        const { centroids, labels } = await kmeansInterleaved(points, numRows, 3, k, 10);
        assertLloydInvariants(points, numRows, 3, centroids, labels, k);
    });
});

describe('shColorError', () => {
    it('is zero for an exact palette and scales with a uniform coefficient error', () => {
        const numColumns = 9;
        const numRows = 100;
        const points = new Float32Array(numRows * numColumns).map((_, i) => Math.sin(i));
        const labels = new Uint32Array(numRows).map((_, i) => i);
        const weights = new Float32Array(numRows).fill(1);
        assert.strictEqual(shColorError(points, numRows, numColumns, points, labels, weights), 0);

        // every coefficient off by d: a channel's view-averaged squared error is
        // (3 coefficients) × d² / 4π, so the RMS is 255 · d · sqrt(3 / 4π)
        const d = 0.01;
        const shifted = points.map((v) => v + d);
        const expected = 255 * d * Math.sqrt(3 / (4 * Math.PI));
        const err = shColorError(points, numRows, numColumns, shifted, labels, weights);
        assert.ok(Math.abs(err - expected) < 1e-4, `${err} vs ${expected}`);
    });
});

// 'f32' hides shader-f16 so the f32 kernels run (the path taken on devices
// without it, e.g. NVIDIA under Vulkan) even where f16 is available
for (const precision of ['default', 'f32']) describe(`GpuKmeans (GPU path, ${precision})`, () => {
    // created in before() so registration never blocks; tests skip when the
    // environment has no usable WebGPU adapter
    let device = null;
    before(async () => {
        try {
            const { createDevice } = await import('../src/cli/node-device.js');
            device = await createDevice();
            if (precision === 'f32') device.supportsShaderF16 = false;
        } catch {
            device = null;
        }
    });
    after(() => device?.destroy());

    it('single iteration reproduces brute-force argmin labels exactly', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');

        // f16-exact coordinates with margins far above f16 epsilon so the
        // GPU's half-precision distance test cannot flip any assignment
        const rand = mulberry32(99);
        const numColumns = 4;
        const k = 8;
        const centers = Array.from({ length: k }, (_, i) => (
            Array.from({ length: numColumns }, (_, j) => ((i >> j) & 1) * 8)
        ));
        const numRows = 1024;
        const points = new Float32Array(numRows * numColumns);
        for (let i = 0; i < numRows; i++) {
            const c = centers[i % k];
            for (let j = 0; j < numColumns; j++) {
                // jitter in f16-exact quarter steps, well inside the margin
                points[i * numColumns + j] = c[j] + Math.round((rand() * 2 - 1) * 4) * 0.25;
            }
        }

        // fixed seeds: the true centers
        const centroids = new Float32Array(k * numColumns);
        centers.forEach((c, i) => centroids.set(c, i * numColumns));
        const reference = centroids.slice();
        const expected = referenceLloyd(points, numRows, numColumns, reference, k, 1);

        const gpu = new GpuKmeans(device, numColumns, k);
        try {
            const labels = new Uint32Array(numRows);
            await gpu.run(points, numRows, centroids, labels, 1);
            assert.deepStrictEqual([...labels], [...expected]);
            for (let i = 0; i < k * numColumns; i++) {
                assert.ok(Math.abs(centroids[i] - reference[i]) < 1e-2,
                    `centroid elem ${i}: gpu ${centroids[i]}, reference ${reference[i]}`);
            }
        } finally {
            gpu.destroy();
        }
    });

    it('separates nearby centroids far from the origin', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');
        if (precision !== 'f32') return t.skip('f16 cannot represent these coordinates exactly');

        // |x|² ≈ 3e8, where f32 resolves steps of 32: an uncentered |c|² - 2x·c
        // cannot tell a centroid at distance 0 from its neighbors at distance 1.
        // Centroids sit 1 apart along x; every point sits exactly on one.
        const numColumns = 3;
        const k = 16;
        const centroids = new Float32Array(k * numColumns);
        for (let c = 0; c < k; c++) centroids.set([8979 + c, 11021, 10003], c * numColumns);
        const numRows = 1024;
        const points = new Float32Array(numRows * numColumns);
        for (let i = 0; i < numRows; i++) {
            points.set([8979 + (i % k), 11021, 10003], i * numColumns);
        }
        const expected = bruteForceLabels(points, numRows, numColumns, centroids, k);

        const gpu = new GpuKmeans(device, numColumns, k);
        try {
            const labels = new Uint32Array(numRows);
            await gpu.run(points, numRows, centroids.slice(), labels, 1);
            assert.deepStrictEqual([...labels], [...expected]);
        } finally {
            gpu.destroy();
        }
    });

    it('matches a reference Lloyd loop within 1% inertia from identical seeds', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');

        const rand = mulberry32(4321);
        const numColumns = 8;
        const k = 32;
        const centers = Array.from({ length: k }, () => (
            Array.from({ length: numColumns }, () => (rand() * 2 - 1) * 20)
        ));
        const { points, numRows } = makeBlobs(centers, 640, numColumns, 0.25, rand);
        const iterations = 8;

        // identical seeds for both paths: one point from every blob
        const init = points.slice(0, k * numColumns);

        const refCentroids = init.slice();
        referenceLloyd(points, numRows, numColumns, refCentroids, k, iterations);

        const gpu = new GpuKmeans(device, numColumns, k);
        try {
            const gpuCentroids = init.slice();
            const gpuLabels = new Uint32Array(numRows);
            await gpu.run(points, numRows, gpuCentroids, gpuLabels, iterations);

            const refInertia = inertia(points, numRows, numColumns, refCentroids, k);
            const gpuInertia = inertia(points, numRows, numColumns, gpuCentroids, k);
            const diff = Math.abs(refInertia - gpuInertia) / refInertia;
            assert.ok(diff < 0.01, `inertia mismatch: reference ${refInertia}, gpu ${gpuInertia} (${(diff * 100).toFixed(2)}%)`);

            assertLloydInvariants(points, numRows, numColumns, gpuCentroids, gpuLabels, k);
        } finally {
            gpu.destroy();
        }
    });

    it('splits a large codebook into centroid ranges and keeps exact labels', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');

        // 64K points × 16000 centroids × 45 columns exceeds one assign
        // command buffer's work budget, so the argmin carries across centroid
        // ranges. Centroid i sits at 8 × (bits of i) in the first 14 columns:
        // any two differ by 8 in at least one column, so jitter of at most 1
        // per column (f16-exact quarter steps) leaves each point's generating
        // centroid as its unique nearest. 16000 is not a whole number of
        // tiles, so the last range is partial.
        const rand = mulberry32(7);
        const numColumns = 45;
        const k = 16000;
        const numRows = 65536;
        const centroids = new Float32Array(k * numColumns);
        for (let i = 0; i < k; i++) {
            for (let j = 0; j < 14; j++) centroids[i * numColumns + j] = ((i >> j) & 1) * 8;
        }
        const points = new Float32Array(numRows * numColumns);
        const expected = new Uint32Array(numRows);
        for (let i = 0; i < numRows; i++) {
            // most points belong to centroids past the first range
            const c = k - 1 - (i % k);
            expected[i] = c;
            for (let j = 0; j < numColumns; j++) {
                points[i * numColumns + j] = centroids[c * numColumns + j] + Math.round((rand() * 2 - 1) * 4) * 0.25;
            }
        }

        const computes = device._computes.size;
        const gpu = new GpuKmeans(device, numColumns, k);
        try {
            const labels = new Uint32Array(numRows);
            await gpu.run(points, numRows, centroids, labels, 1);
            const wrong = labels.findIndex((l, i) => l !== expected[i]);
            assert.strictEqual(wrong, -1, `point ${wrong}: label ${labels[wrong]}, expected ${expected[wrong]}`);
        } finally {
            gpu.destroy();
        }
        assert.strictEqual(device._computes.size, computes, 'GpuKmeans.destroy must release its computes');
    });

    it('e-means matches a reference Lloyd loop within 1% inertia from identical seeds', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');

        const rand = mulberry32(2468);
        const numColumns = 12;
        const k = 64;
        const centers = Array.from({ length: k }, () => (
            Array.from({ length: numColumns }, () => (rand() * 2 - 1) * 20)
        ));
        // 64 × 500 points: several batches per epoch, and the batch does not
        // divide the row count, so batches wrap the shuffled order
        const { points, numRows } = makeBlobs(centers, 500, numColumns, 0.25, rand);
        const init = points.slice(0, k * numColumns);

        const refCentroids = init.slice();
        referenceLloyd(points, numRows, numColumns, refCentroids, k, 10);

        const gpu = new GpuKmeans(device, numColumns, k);
        try {
            const centroids = init.slice();
            const labels = new Uint32Array(numRows);
            let epochs = 0;
            await gpu.runEmeans(points, numRows, centroids, labels, 2.5, () => epochs++);
            assert.strictEqual(epochs, 3, 'onEpoch fires once per epoch (2 whole + 1 partial)');

            const refInertia = inertia(points, numRows, numColumns, refCentroids, k);
            const gpuInertia = inertia(points, numRows, numColumns, centroids, k);
            const diff = Math.abs(refInertia - gpuInertia) / refInertia;
            assert.ok(diff < 0.01, `inertia mismatch: reference ${refInertia}, e-means ${gpuInertia} (${(diff * 100).toFixed(2)}%)`);

            // labels map back through the upload shuffle to the caller's rows,
            // and the closing centroids are their exact means
            assertLloydInvariants(points, numRows, numColumns, centroids, labels, k);
        } finally {
            gpu.destroy();
        }
    });

    it('e-means closes on weighted means', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');

        // two clusters, each made of two locations with weights 1 and 3: the
        // closing centroid is the 1:3 blend, not the midpoint
        const numColumns = 4;
        const k = 2;
        const locations = [[0, 0, 0, 0], [4, 4, 4, 4], [100, 0, 0, 0], [100, 8, 0, 0]];
        const pointWeights = [1, 3, 1, 3];
        const numRows = 4096;
        const points = new Float32Array(numRows * numColumns);
        const weights = new Float32Array(numRows);
        for (let i = 0; i < numRows; i++) {
            points.set(locations[i % 4], i * numColumns);
            weights[i] = pointWeights[i % 4];
        }
        const centroids = new Float32Array([2, 2, 2, 2, 100, 4, 0, 0]);

        const gpu = new GpuKmeans(device, numColumns, k);
        try {
            const labels = new Uint32Array(numRows);
            await gpu.runEmeans(points, numRows, centroids, labels, 2, undefined, weights);
            const expected = [3, 3, 3, 3, 100, 6, 0, 0];
            for (let i = 0; i < expected.length; i++) {
                assert.ok(Math.abs(centroids[i] - expected[i]) < 1e-3, `centroid elem ${i}: ${centroids[i]}, expected ${expected[i]}`);
            }
        } finally {
            gpu.destroy();
        }
    });

    it('e-means revives dead clusters as splits of real ones', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');

        // 4 distinct locations but k=32, seeded all on one location: the other
        // centroids only reach the rest by revival
        const distinct = [[1, 2.5, -3, 4], [5, 5, 5, 5], [-4, 0.5, 2, -1], [3, -6, 1, 2]];
        const numColumns = 4;
        const numRows = 8192;
        const k = 32;
        const points = new Float32Array(numRows * numColumns);
        for (let i = 0; i < numRows; i++) points.set(distinct[i % 4], i * numColumns);
        const centroids = new Float32Array(k * numColumns);
        for (let c = 0; c < k; c++) centroids.set(distinct[0], c * numColumns);

        const gpu = new GpuKmeans(device, numColumns, k);
        try {
            const labels = new Uint32Array(numRows);
            await gpu.runEmeans(points, numRows, centroids, labels, 3);

            for (let c = 0; c < k; c++) {
                const cv = [...centroids.subarray(c * numColumns, (c + 1) * numColumns)];
                assert.ok(cv.every(Number.isFinite), `centroid ${c} not finite: ${cv}`);
            }
            // every location ends up with a centroid of its own
            for (let i = 0; i < 4; i++) {
                const cv = [...centroids.subarray(labels[i] * numColumns, (labels[i] + 1) * numColumns)];
                const err = Math.hypot(...distinct[i].map((v, j) => v - cv[j]));
                assert.ok(err < 0.05, `point ${i} at ${distinct[i]} labelled to centroid ${cv}`);
            }
        } finally {
            gpu.destroy();
        }
    });

    describe('kmeansToColorError', () => {
        // 64 well separated prototypes in 9 dims, ±0.01 noise (about 0.7 levels)
        const numColumns = 9;
        const numRows = 64 * 200;
        const makeData = () => {
            const rand = mulberry32(31);
            const centers = Array.from({ length: 64 }, () => Array.from({ length: numColumns }, () => 2 * rand() - 1));
            return makeBlobs(centers, 200, numColumns, 0.01, rand).points;
        };
        const weights = new Float32Array(numRows).fill(1);

        it('returns a small palette meeting a loose target', async (t) => {
            if (!device) return t.skip('no WebGPU adapter available');
            const points = makeData();
            const { centroids, labels, probes } = await kmeansToColorError(points, numRows, numColumns, weights, 3, 4096, device);
            const k = centroids.length / numColumns;
            const err = shColorError(points, numRows, numColumns, centroids, labels, weights);
            assert.ok(err <= 3, `colour error ${err} above target`);
            assert.ok(k >= 64 && k <= 256, `palette of ${k} entries (probes ${JSON.stringify(probes)})`);
        });

        it('stops at the size limit when the target is out of reach', async (t) => {
            if (!device) return t.skip('no WebGPU adapter available');
            const points = makeData();
            const { centroids, probes } = await kmeansToColorError(points, numRows, numColumns, weights, 0.01, 1024, device);
            assert.strictEqual(centroids.length / numColumns, 1024, `probes ${JSON.stringify(probes)}`);
        });

        it('stores every point when a palette that large is needed and allowed', async (t) => {
            if (!device) return t.skip('no WebGPU adapter available');
            const points = makeData().subarray(0, 300 * numColumns);
            const w = weights.subarray(0, 300);
            const { centroids, labels } = await kmeansToColorError(points, 300, numColumns, w, 1e-6, 4096, device);
            assert.strictEqual(centroids.length / numColumns, 300);
            assert.strictEqual(shColorError(points, 300, numColumns, centroids, labels, w), 0);
        });
    });

    it('reseeds empty clusters to real points', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');

        // only 3 distinct locations but k=16 — most clusters must reseed
        const distinct = [[1, 2.5, -3], [5, 5, 5], [-4, 0.5, 2]];
        const numColumns = 3;
        const numRows = 1024;
        const k = 16;
        const points = new Float32Array(numRows * numColumns);
        for (let i = 0; i < numRows; i++) {
            points.set(distinct[i % 3], i * numColumns);
        }

        const { centroids } = await kmeansInterleaved(points, numRows, numColumns, k, 3, device);

        // every centroid must be (near) one of the three real locations —
        // never a zero vector or NaN from an empty cluster
        for (let c = 0; c < k; c++) {
            const cv = [...centroids.subarray(c * numColumns, (c + 1) * numColumns)];
            assert.ok(cv.every(Number.isFinite), `centroid ${c} not finite: ${cv}`);
            const nearest = Math.min(...distinct.map(p => Math.hypot(...p.map((v, j) => v - cv[j]))));
            assert.ok(nearest < 1e-2, `centroid ${c} = ${cv} is not one of the input points`);
        }
    });
});
