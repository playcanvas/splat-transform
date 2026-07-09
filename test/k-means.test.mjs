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

import { kmeansInterleaved } from '../src/lib/spatial/k-means.js';
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

describe('GpuKmeans (GPU path)', () => {
    // created in before() so registration never blocks; tests skip when the
    // environment has no usable WebGPU adapter
    let device = null;
    before(async () => {
        try {
            const { createDevice } = await import('../src/cli/node-device.js');
            device = await createDevice();
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
