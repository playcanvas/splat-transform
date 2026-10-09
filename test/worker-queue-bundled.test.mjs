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
import { Worker as NodeWorker } from 'node:worker_threads';

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
const runTasks = async (WorkerQueue, count = 5, size = 4) => {
    const results = await Promise.all(
        Array.from({ length: count }, () => WorkerQueue.run('quantize1d', {
            columns: [{ name: 'a', data: Float32Array.from({ length: size }, (_, i) => Math.sin(i)) }]
        }))
    );
    assert.strictEqual(results.length, count);
    results.forEach((result) => assert.strictEqual(result.labels[0].data.length, size));
};

const settle = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// an instance on the pool's browser path (the Electron renderer check makes
// it skip the node path), where new Worker() runs synchronously in the launch
// loop rather than after an await
const browserQueue = async (tag) => {
    process.type = 'renderer';
    try {
        return await freshQueue(tag);
    } finally {
        delete process.type;
    }
};

// a browser-style Worker over worker_threads; once blocked, construction
// throws synchronously, as a CSP SecurityError does
class BrowserWorker {
    static blocked = false;
    static blockedLaunches = 0;
    static live = [];

    constructor(url) {
        if (BrowserWorker.blocked) {
            BrowserWorker.blockedLaunches++;
            throw new DOMException('worker blocked', 'SecurityError');
        }
        const worker = new NodeWorker(url);
        worker.on('message', (data) => this.onmessage?.({ data }));
        worker.on('error', (err) => this.onerror?.({ message: err.message }));
        this.postMessage = (message, transfer) => worker.postMessage(message, transfer);
        this.terminate = () => worker.terminate();
        this.crash = () => {
            worker.terminate();
            this.onerror?.({ message: 'worker crashed' });
        };
        BrowserWorker.live.push(this);
    }
}

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

    it('lets a surviving worker drain the queue without launching more after one fails to start', { timeout: 20_000 }, async () => {
        const WorkerQueue = await freshQueue('one-survivor');
        WorkerQueue.maxWorkers = 1;
        await runTasks(WorkerQueue, 1);

        // the running worker stays healthy, but new ones can no longer load
        const broken = brokenWorker();
        WorkerQueue.workerUrl = broken.path;
        WorkerQueue.maxWorkers = 4;
        // tasks long enough for the failed launches to happen while work is queued
        await runTasks(WorkerQueue, 8, 10_000);
        await settle(500);

        assert.strictEqual(broken.starts(), 3);
        assert.strictEqual(WorkerQueue.isInline, false);
    });

    it('tries workers again after destroy()', { timeout: 20_000 }, async () => {
        const WorkerQueue = await freshQueue('retry-after-destroy');
        WorkerQueue.workerUrl = brokenWorker().path;
        WorkerQueue.maxWorkers = 4;
        await runTasks(WorkerQueue);
        assert.strictEqual(WorkerQueue.isInline, true);

        await WorkerQueue.destroy();
        WorkerQueue.workerUrl = null;
        await runTasks(WorkerQueue);

        assert.strictEqual(WorkerQueue.isInline, false);
    });

    it('stops a launch batch at the first startup failure thrown by new Worker()', { timeout: 20_000 }, async () => {
        globalThis.Worker = BrowserWorker;
        try {
            const WorkerQueue = await browserQueue('browser-sync-throw');
            WorkerQueue.maxWorkers = 2;
            const tasks = Array.from({ length: 16 }, () => WorkerQueue.run('quantize1d', {
                columns: [{ name: 'a', data: Float32Array.from({ length: 10_000 }, (_, i) => Math.sin(i)) }]
            }));
            await tasks[0];

            // the worker that ran it crashes with work queued, and its
            // replacements now throw in new Worker()
            BrowserWorker.blocked = true;
            BrowserWorker.live[0].crash();
            const results = await Promise.allSettled(tasks);

            assert.strictEqual(results.filter((result) => result.status === 'rejected').length, 1);
            assert.strictEqual(BrowserWorker.blockedLaunches, 1);
            assert.strictEqual(WorkerQueue.isInline, false);
            await WorkerQueue.destroy();
        } finally {
            delete globalThis.Worker;
        }
    });
});
