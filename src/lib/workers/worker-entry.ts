import { taskHandlers, type HostMessage, type WorkerMessage } from './tasks';
import { WebPCodec } from '../utils/webp-codec';

/**
 * Worker-side entry point, built as a self-contained bundle and inlined as a
 * string into the library/CLI bundles (see rollup.config.mjs), then spawned
 * by WorkerQueue from a Blob URL (browser) or data: URL (Node). Runs one
 * task at a time and posts the result back with its buffers transferred.
 */

const bind = (
    post: (message: WorkerMessage, transfer: ArrayBuffer[]) => void,
    listen: (handler: (message: HostMessage) => void) => void
) => {
    listen(async (message) => {
        if (message.type === 'init') {
            // resolved host-side: import.meta.url is useless in a blob/data
            // worker, so the wasm location must be handed in
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

if (typeof process !== 'undefined' && process.versions?.node) {
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
