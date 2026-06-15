import type { TypedArray } from '../data-table/data-table';
import { quantize1dColumns, type QuantizedColumns } from '../spatial/quantize-1d-core';
import { WebPCodec } from '../utils/webp-codec';

/**
 * Named task handlers runnable on a worker thread or inline on the calling
 * thread. Both transports execute these exact functions, so results are
 * identical regardless of where a task runs.
 *
 * Handlers take a single structured-clone-friendly argument object (typed
 * arrays, plain objects) and return the result plus the list of buffers to
 * transfer back to the host (ignored when running inline).
 *
 * NOTE: this module is bundled into the worker, so its runtime import graph
 * must stay lean - in particular free of DataTable, whose Transform member
 * drags in the playcanvas engine.
 */

type TaskOutput<T> = { result: T, transfer: ArrayBuffer[] };

const taskHandlers = {
    quantize1d: (args: { columns: { name: string, data: TypedArray }[], k?: number, alpha?: number }):
        TaskOutput<QuantizedColumns> => {
        const result = quantize1dColumns(args.columns, args.k, args.alpha);
        return {
            result,
            transfer: [result.centroids.buffer as ArrayBuffer, ...result.labels.map(c => c.data.buffer as ArrayBuffer)]
        };
    },

    encodeWebp: async (args: { rgba: Uint8Array, width: number, height: number }): Promise<TaskOutput<Uint8Array>> => {
        // create() memoizes the wasm module per realm (each worker compiles
        // its own copy on first use)
        const codec = await WebPCodec.create();
        const webp = codec.encodeLosslessRGBA(args.rgba, args.width, args.height);
        return { result: webp, transfer: [webp.buffer as ArrayBuffer] };
    }
};

type TaskName = keyof typeof taskHandlers;
type TaskArgs<T extends TaskName> = Parameters<typeof taskHandlers[T]>[0];
type TaskResult<T extends TaskName> = Awaited<ReturnType<typeof taskHandlers[T]>>['result'];

// Message protocol between host and worker. There are no per-task ids: each
// worker runs strictly one task at a time (enforced by the host's slot state
// machine), so every reply pairs with the single in-flight `run`.
type HostMessage =
    | { type: 'init', wasmUrl: string }
    | { type: 'run', task: TaskName, args: any };

type WorkerMessage =
    | { type: 'ready' }
    | { type: 'result', result: any }
    | { type: 'error', message: string, stack?: string };

export { taskHandlers, type TaskName, type TaskArgs, type TaskResult, type HostMessage, type WorkerMessage };
