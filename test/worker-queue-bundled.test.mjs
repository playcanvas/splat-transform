/**
 * Tests for the worker queue's worker transport, which only exists in the
 * built library (from source every task runs inline; see
 * worker-queue.test.mjs). Loads `dist/index.mjs`, which `npm test` builds via
 * the pretest hook; skipped with a message when it hasn't been built.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert';
import { mkdtempSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const rootDir = join(dirname(fileURLToPath(import.meta.url)), '..');
const distPath = join(rootDir, 'dist/index.mjs');

const skip = (() => {
    try {
        statSync(distPath);
        return false;
    } catch {
        return 'dist/index.mjs missing — run `npm run build` first (or `npm test`, which builds via the pretest hook)';
    }
})();

// the pool's state is module-global, so each test imports its own instance
const freshQueue = async (tag) => (await import(`${pathToFileURL(distPath).href}?${tag}`)).WorkerQueue;

// a worker script that records each start, then fails to load
const brokenWorker = () => {
    const dir = mkdtempSync(join(tmpdir(), 'splat-transform-worker-'));
    const starts = join(dir, 'starts.txt');
    const path = join(dir, 'broken-worker.mjs');
    writeFileSync(starts, '');
    writeFileSync(
        path,
        `import { appendFileSync } from 'node:fs';\nappendFileSync(${JSON.stringify(starts)}, 'x');\nthrow new Error('broken worker');\n`
    );
    return { path, starts: () => readFileSync(starts, 'utf8').length };
};

// more tasks than workers, so several workers are starting when the first fails
const runTasks = async (WorkerQueue) => {
    const results = await Promise.all(
        Array.from({ length: 5 }, () => WorkerQueue.run('quantize1d', {
            columns: [{ name: 'a', data: new Float32Array([1, 2, 3, 4]) }]
        }))
    );
    assert.strictEqual(results.length, 5);
    results.forEach((result) => assert.strictEqual(result.labels[0].data.length, 4));
};

describe('worker queue (bundled)', { skip }, () => {
    it('runs tasks inline, without respawning, when no worker can start', { timeout: 20_000 }, async () => {
        const WorkerQueue = await freshQueue('never-ready');
        const broken = brokenWorker();
        WorkerQueue.workerUrl = broken.path;
        WorkerQueue.maxWorkers = 4;

        await runTasks(WorkerQueue);

        assert.strictEqual(WorkerQueue.isInline, true);
        assert.strictEqual(broken.starts(), 4);
    });

    it('stops replacing workers that fail to start, even after earlier workers ran', { timeout: 20_000 }, async () => {
        const WorkerQueue = await freshQueue('ready-then-broken');
        WorkerQueue.maxWorkers = 4;
        await runTasks(WorkerQueue);
        assert.strictEqual(WorkerQueue.isInline, false);
        await WorkerQueue.destroy();

        // e.g. a deploy has since removed the worker script the pool loads
        const broken = brokenWorker();
        WorkerQueue.workerUrl = broken.path;
        await runTasks(WorkerQueue);

        assert.strictEqual(WorkerQueue.isInline, true);
        assert.strictEqual(broken.starts(), 4);
    });
});
