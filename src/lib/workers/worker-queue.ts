import { taskHandlers, type TaskName, type TaskArgs, type TaskResult, type WorkerMessage } from './tasks';
import { workerBundled } from './worker-bundled';
import { WebPCodec } from '../utils/webp-codec';

type PendingTask = {
    task: TaskName;
    args: any;
    transfer: ArrayBuffer[];
    resolve: (result: any) => void;
    reject: (err: Error) => void;
};

// One spawned worker. `state` enforces the one-task-in-flight protocol
// invariant: tasks are only posted to idle (ready) workers.
type Slot = {
    state: 'starting' | 'idle' | 'busy';
    current: PendingTask | null;
    dead: boolean;
    post: (message: any, transfer: ArrayBuffer[]) => void;
    terminate: () => void;
    ref: () => void;
    unref: () => void;
};

// same runtime check as the emscripten glue in lib/webp.mjs
const isNode = typeof process !== 'undefined' && !!process.versions?.node && (process as any).type !== 'renderer';

let slots: Slot[] = [];
const queue: PendingTask[] = [];
const outstanding = new Set<Promise<any>>();

// user-configurable: max worker threads (null = auto), 0 forces inline
let maxWorkers: number | null = null;

// memoized once workers prove unavailable (spawn failed before any worker
// ever signalled ready, or the environment can't run them)
let unavailable = false;
let everReady = false;

let resolvedMaxWorkers: number | null = null;
let spawnLoopActive = false;

// user-configurable: URL of the shipped worker file (dist/worker.mjs). When
// null, resolved relative to this module - which works in Node and in
// bundlers that rewrite new URL('./worker.mjs', import.meta.url), but not in
// bundlers that don't (set this explicitly there, mirroring WebPCodec.wasmUrl)
let workerUrl: string | null = null;

const inlineMode = () => !workerBundled || unavailable || maxWorkers === 0;

const runTaskInline = async (task: PendingTask) => {
    try {
        const { result } = await taskHandlers[task.task](task.args);
        task.resolve(result);
    } catch (err) {
        task.reject(err instanceof Error ? err : new Error(String(err)));
    }
};

const drainQueueInline = () => {
    const tasks = queue.splice(0, queue.length);
    tasks.forEach(runTaskInline);
};

const removeSlot = (slot: Slot) => {
    slot.dead = true;
    const index = slots.indexOf(slot);
    if (index !== -1) {
        slots.splice(index, 1);
    }
};

// a worker died: before its ready signal this means the environment can't
// run workers (memoize and go inline); after, the in-flight task failed and
// the worker is replaced on demand
function onSlotDeath(slot: Slot, err: Error) {
    if (slot.dead) {
        return;
    }
    const wasStarting = slot.state === 'starting';
    const task = slot.current;
    removeSlot(slot);
    slot.terminate();

    if (task) {
        task.reject(err);
    }

    if (wasStarting && !everReady && slots.length === 0) {
        unavailable = true;
        drainQueueInline();
        return;
    }

    dispatch();
}

function onSlotMessage(slot: Slot, message: WorkerMessage) {
    if (slot.dead) {
        return;
    }

    if (message.type === 'ready') {
        everReady = true;
        slot.state = 'idle';
    } else {
        const task = slot.current;
        slot.current = null;
        slot.state = 'idle';

        if (message.type === 'result') {
            task.resolve(message.result);
        } else {
            const err = new Error(message.message);
            if (message.stack) {
                err.stack = message.stack;
            }
            task.reject(err);
        }
    }

    dispatch();
}

async function spawnSlot(slot: Slot) {
    try {
        // resolve relative to this module: dist/worker.mjs sits next to the
        // bundle (dist/index.mjs, dist/cli.mjs)
        const url = workerUrl ?? new URL('./worker.mjs', import.meta.url);

        if (isNode) {
            const { Worker } = await import('node:worker_threads');
            if (slot.dead) {
                return;
            }
            const worker = new Worker(url);
            slot.post = (message, transfer) => worker.postMessage(message, transfer);
            slot.terminate = () => {
                worker.terminate();
            };
            slot.ref = () => worker.ref();
            slot.unref = () => worker.unref();
            worker.on('message', (message: WorkerMessage) => onSlotMessage(slot, message));
            worker.on('error', (err: Error) => onSlotDeath(slot, err));
            // any exit we didn't initiate (onSlotDeath sets slot.dead before
            // terminating) means this worker is gone - drop the slot so it
            // can't stall the queue, whatever the exit code
            worker.on('exit', (code: number) => onSlotDeath(slot, new Error(`worker exited with code ${code}`)));
        } else {
            const worker = new Worker(url, { type: 'module' });
            slot.post = (message, transfer) => worker.postMessage(message, transfer);
            slot.terminate = () => worker.terminate();
            worker.onmessage = (event: MessageEvent) => onSlotMessage(slot, event.data);
            worker.onerror = event => onSlotDeath(slot, new Error(event.message || 'worker load failed'));
        }

        if (slot.dead) {
            slot.terminate();
            return;
        }

        // the worker can't resolve the wasm location itself (its
        // import.meta.url is a blob/data url), so hand it the host's
        slot.post({ type: 'init', wasmUrl: WebPCodec.resolveWasmUrl() }, []);
    } catch (err) {
        onSlotDeath(slot, err instanceof Error ? err : new Error(String(err)));
    }
}

// spawn workers to cover queued demand, up to the concurrency limit. Async
// because the default limit needs node:os; the guard flag collapses
// concurrent calls (slots are pushed synchronously inside the loop, so the
// limit holds). Never rejects - spawn failures route through onSlotDeath.
async function ensureSpawned() {
    if (spawnLoopActive || inlineMode()) {
        return;
    }
    spawnLoopActive = true;
    try {
        if (resolvedMaxWorkers === null) {
            const cores = isNode ?
                (await import('node:os')).availableParallelism() :
                (navigator.hardwareConcurrency || 2);
            resolvedMaxWorkers = Math.max(1, Math.min(4, cores - 1));
        }

        const max = maxWorkers ?? resolvedMaxWorkers;
        let available = slots.filter(s => s.state !== 'busy').length;

        while (slots.length < max && available < queue.length) {
            const slot: Slot = {
                state: 'starting',
                current: null,
                dead: false,
                post: () => {},
                terminate: () => {},
                ref: () => {},
                unref: () => {}
            };
            slots.push(slot);
            spawnSlot(slot);
            available++;
        }
    } catch {
        // import('node:os') is the only awaited call; treat any failure as
        // "no workers" and fall back to inline
        unavailable = true;
        drainQueueInline();
    } finally {
        spawnLoopActive = false;
    }
}

function dispatch() {
    while (queue.length > 0) {
        const slot = slots.find(s => s.state === 'idle');
        if (!slot) {
            ensureSpawned();
            break;
        }

        const task = queue.shift();
        slot.state = 'busy';
        slot.current = task;
        slot.ref();

        try {
            slot.post({ type: 'run', task: task.task, args: task.args }, task.transfer);
        } catch {
            // args didn't survive structured clone (e.g. Float16Array on
            // older node) - run this one task on the calling thread instead
            slot.state = 'idle';
            slot.current = null;
            runTaskInline(task);
        }
    }

    // let the node process exit while workers sit idle
    if (queue.length === 0) {
        slots.forEach((slot) => {
            if (slot.state === 'idle') {
                slot.unref();
            }
        });
    }
}

const enqueue = <T extends TaskName>(task: T, args: TaskArgs<T>, transfer: ArrayBuffer[]): Promise<TaskResult<T>> => {
    const promise = new Promise<TaskResult<T>>((resolve, reject) => {
        const pending: PendingTask = { task, args, transfer, resolve, reject };
        if (inlineMode()) {
            runTaskInline(pending);
        } else {
            queue.push(pending);
            dispatch();
        }
    });

    outstanding.add(promise);
    promise.catch(() => {}).then(() => outstanding.delete(promise));

    return promise;
};

const destroyPool = async () => {
    await Promise.allSettled([...outstanding]);
    const current = slots;
    slots = [];
    current.forEach((slot) => {
        slot.dead = true;
        slot.terminate();
    });
};

/**
 * A small cross-platform (Node + browser) worker pool running the CPU-heavy
 * tasks defined in tasks.ts off the main thread. The worker entry is built and
 * shipped as `dist/worker.mjs`; the pool spawns it from a URL resolved
 * relative to the library bundle. Node and bundlers that rewrite
 * `new Worker(new URL('./worker.mjs', import.meta.url))` (e.g. Vite, webpack)
 * pick it up automatically; with other bundlers, set `WorkerQueue.workerUrl`
 * to the deployed worker asset (mirroring `WebPCodec.wasmUrl`).
 *
 * Workers spawn lazily on demand and run one task at a time. When workers are
 * unavailable (running from source via tsx, `maxWorkers = 0`, or spawn fails)
 * every task runs inline on the calling thread instead - same code, same
 * results, just serial.
 */
class WorkerQueue {
    /**
     * Sets the URL of the worker script (the shipped dist/worker.mjs). Set
     * this before first use in browser environments whose bundler does not
     * rewrite `new Worker(new URL('./worker.mjs', import.meta.url))` - e.g.
     *
     * import workerUrl from '@playcanvas/splat-transform/worker?url';
     * WorkerQueue.workerUrl = workerUrl;
     *
     * @param value - Absolute URL of the worker script, or null to auto-resolve.
     */
    static set workerUrl(value: string | null) {
        workerUrl = value;
    }

    /**
     * URL of the worker script, or null when auto-resolved relative to the
     * bundle.
     *
     * @returns The configured worker URL, or null.
     */
    static get workerUrl() {
        return workerUrl;
    }

    /**
     * Sets the maximum number of worker threads. Set to 0 to force inline
     * execution on the calling thread, or null to auto-size. Configure once
     * before the first `run()` (as the CLI does); changing it while tasks are
     * in flight is not supported.
     *
     * @param value - Maximum worker threads, or null to auto-size.
     */
    static set maxWorkers(value: number | null) {
        maxWorkers = value;
    }

    /**
     * Maximum number of worker threads. Defaults to one less than the
     * available hardware concurrency, capped at 4. (Peak memory scales with
     * worker count, since each holds its own WebP WASM heap; 4 captures most
     * of the parallelism for SOG writes.)
     *
     * @returns The configured limit, or null when auto-sized.
     */
    static get maxWorkers() {
        return maxWorkers;
    }

    /**
     * True when tasks run inline on the calling thread (workers disabled or
     * unavailable).
     *
     * @ignore
     */
    static get isInline() {
        return inlineMode();
    }

    /**
     * Runs a named task, preferring a worker thread. Buffers listed in
     * `transfer` are transferred (not copied) to the worker and must not be
     * used by the caller afterwards.
     *
     * @param task - Name of the task to run.
     * @param args - Task arguments (structured-clone-friendly).
     * @param transfer - Buffers in args to transfer rather than copy.
     * @returns The task result.
     * @ignore
     */
    static run<T extends TaskName>(task: T, args: TaskArgs<T>, transfer: ArrayBuffer[] = []): Promise<TaskResult<T>> {
        return enqueue(task, args, transfer);
    }

    /**
     * Waits for in-flight tasks to settle, then terminates all workers.
     * Optional: idle workers don't keep the Node process alive, and workers
     * respawn lazily on the next run() call.
     *
     * @returns A promise that resolves once all workers are terminated.
     */
    static destroy() {
        return destroyPool();
    }
}

export { WorkerQueue };
