import { taskHandlers, type HostMessage, type WorkerMessage } from './tasks';
import { WebPCodec } from '../utils/webp-codec';

/**
 * Worker-side entry point, built and shipped as `dist/worker.mjs` (see
 * rollup.config.mjs) and spawned by WorkerQueue from a URL. Runs one task at a
 * time and posts the result back with its buffers transferred.
 */

const bind = (
    post: (message: WorkerMessage, transfer: ArrayBuffer[]) => void,
    listen: (handler: (message: HostMessage) => void) => void
) => {
    listen(async (message) => {
        if (message.type === 'init') {
            // resolved host-side and handed in, so the worker uses the same
            // wasm location as the host regardless of its own module URL
            WebPCodec.wasmUrl = message.wasmUrl;
            return;
        }

        try {
            const { result, transfer } = await taskHandlers[message.task](message.args);
            post({ type: 'result', result }, transfer);
        } catch (err) {
            post({
                type: 'error',
                message: err instanceof Error ? err.message : String(err),
                stack: err instanceof Error ? err.stack : undefined
            }, []);
        }
    });

    // unprompted readiness signal: lets the host distinguish "environment
    // cannot run workers" (no ready, fall back inline) from a task crashing
    // a live worker
    post({ type: 'ready' }, []);
};

// same guard as WorkerQueue's isNode: a real worker_threads worker, not an
// Electron renderer (where process.versions.node is present but messaging goes
// through the Web Worker scope)
if (typeof process !== 'undefined' && !!process.versions?.node && (process as any).type !== 'renderer') {
    // node MessagePorts buffer messages until a listener attaches, so the
    // host's init message survives this async import
    import('node:worker_threads').then(({ parentPort }) => {
        bind(
            (message, transfer) => parentPort.postMessage(message, transfer),
            handler => parentPort.on('message', handler)
        );
    });
} else {
    // tsconfig lib "dom" types postMessage/onmessage as Window's; cast to the
    // dedicated worker scope shape
    const scope = globalThis as any;
    bind(
        (message, transfer) => scope.postMessage(message, transfer),
        (handler) => {
            scope.onmessage = (event: MessageEvent) => handler(event.data);
        }
    );
}
